---
name: pixelcanvas
description: >-
  Draw finished pictures with code (numpy + Pillow, no image model): plan, draw freehand with ready shapes,
  shading and exact text, check, look, then keep improving the picture as layers edited by small commands. Use
  whenever the user asks to draw, paint, illustrate or design anything visual: an illustration, character,
  animal, scene, poster, card, banner, wallpaper, logo or icon; a diagram, chart or dashboard; a UI mock-up; an
  engineering drawing or floor plan; a 3D product shot; one of 43 named art styles; text in any language. Also
  for every change to a picture already drawn: "make the fox bigger", "move the title".
  觸發詞：畫一張、幫我畫、插畫、海報、流程圖、圖表、工程圖、圖示。
  トリガー：絵を描いて、イラスト、ポスター、図、グラフ。
license: MIT
compatibility: Python 3.9+ with numpy and Pillow (installed by scripts/bootstrap.py); git for the first download. Works with or without vision.
metadata:
  author: treeleaves30760
  version: "0.5.0"
  repository: https://github.com/treeleaves30760/pixelcanvas
---

# pixelcanvas: draw with code, then keep improving the picture

You design the picture and draw it in Python; it stays a stack of layers, one per thing, improved by small
commands while the report names what to fix. `<root>` is the runtime folder named under "Local install location"
at the end of this page (only when that is empty: `python <folder of this SKILL.md>/scripts/bootstrap.py`, last
line = `<root>`). `studio` below means `python <root>/tools/studio.py`.

## Two kinds of picture

- **Checked**: several things, counts, colours, states, places, steps, a question or quoted text: it will be
  ticked off item by item, so every item must read at a glance. Follow "Checked pictures".
- **Enjoyed**: one subject, a mood or a style. Follow "Art direction". A request that is both is checked.
- **The medium is a requested item.** `new` reads oil, watercolour, ink, pixel art, 3D render and paints the
  drawing in it ([medium.md](references/medium.md)). Photo, realistic, any medium: lit solids, NO outlines
  (`volume=`, `spec=`, `shadow="ground"`). None named: shaded forms with a darker edge (`line=True`).

## The loop

1. **Inventory**: the first comment line of each section of the draft. Checked: one section per thing the request
   names, in the request's words: count, colour, state, then `|` and its place (`# apple: 3, green, one bitten |
   in bowl`); each implied fact worked out (the end state, the answer, what is gone), large enough to see at
   thumbnail size; quoted text in quotes. Enjoyed: the subject and its share of the frame, mood, light, palette.
2. **Write the whole draft in ONE file write** with your file tool (not a shell heredoc): `./art/NAME/draft.py`
   like the example below. It is one script cut into `# %% name` sections, back to front; a thing's parts stay
   in its section; names defined in one section are visible in the next.
   Under 100 lines, rendered as written: fix after you see it.
3. **Render and open `./art/NAME/look.png` in the same message**: `studio new NAME --prompt "<the request
   word for word>" --size WxH --out <delivery path>.png --from ./art/NAME/draft.py`
4. **Do what the `NEXT:` line says.** A failing section or a shape that does not read as the thing: edit that
   section of `draft.py`, then `studio add --from ./art/NAME/draft.py`. A wrong
   place, size, count, order or colour: one command from the table.
5. **Look and fix**: take the printed list line by line against `look.png`: does the picture show it at a
   glance? Fix a line that does not (its section, or one command), render and look again; NEXT counts three looks.
   Worse than before: `studio undo`. Several faults: `studio note add "3 shards hidden by the tape" "1 windows
   not shattered"` keeps the list for NEXT.
6. **Finished** when every line shows at a glance: the `--out` file is always current; tell the user its path.

After the first draft never rewrite the whole picture: one section or one command at a time. A later request
("make the fox bigger") continues the same state: `studio -p NAME scale fox 1.3` rewrites the delivered file.
`studio api sk.paint creature chart.bar` prints how each call is written, `studio api` what exists. Never read
the package source.

| Weak point | Command |
|---|---|
| things small in an empty frame | `crop X0 Y0 X1 Y1` (that region fills the frame) |
| wrong place or size | `move L --to X Y`, `scale L 1.4`, `fit L X0 Y0 X1 Y1`, `put A on B` (`in`, `left-of`, `above` ...) |
| hidden behind another | `order L above B` (`top`, `bottom`) |
| wrong count | `copy L --total N --row X0 Y0 X1 Y1` (apart, never touching); `L*` is L and its copies |
| wrong colour | `recolor L c4501c` |
| blends into its ground | `recolor L` darker or lighter; `set L shadow=ground`, `set L outline=on` (never in a photo) |
| missing or wrong text | `add title --text "Night Walk" --box 0.2 0.04 0.8 0.16 --color fff4d6` |
| a detail you cannot judge | `look --zoom X0 Y0 X1 Y1`; several at once: `batch "scale fox 1.3" "order moon below fox"` |

## The draft

Every picture that is not a chart, diagram, drawing or UI. A section sees `cv`, `sk` (a `Sketch`), `W`, `H`, `np`,
`math`, `random`, every kit, `box("fox")` and `anchor("fox", "mouth")` (where that layer is now).

```python
# %% wood
# wood: snowy ground receding to dark pines, night sky
cv.gradient([(0, "#0b1030"), (0.65, "#24305e"), (1, "#3d4a7a")])
for x in (0.08, 0.22, 0.78, 0.93):
    sk.paint(sk.polygon([(x, 0.18), (x - 0.07, 0.78), (x + 0.07, 0.78)]), "#132a2c")
sk.paint(sk.blob([(-0.05, 0.74), (0.4, 0.69), (1.05, 0.75), (1.05, 1.05), (-0.05, 1.05)]), [(0, "#8d9cc0"), (1, "#e6edf8")])

# %% fox
# fox: 1, orange, walking left to right | at centre
fox = creature(cv, "fox", at=(0.42, 0.88), size=0.4, pose="walk", facing="right")

# %% lantern
# lantern: glowing warm yellow, its handle in the fox's mouth
mx, my = anchor("fox", "mouth")                 # follows the fox when it moves
with sk.at(mx, my, 0.15):                       # local frame: 1 unit = 0.15 of the height
    sk.paint(sk.stroke([(0, 0), (0, 0.45)], 0.05), "#4a3526", shade=0.4)       # handle
    sk.paint(sk.box(-0.2, 0.45, 0.2, 1.1, 0.04), "#ffcf70", glow="#ffcf70", shade=0.3, highlight=0.6)
    sk.paint(sk.box(-0.25, 1.1, 0.25, 1.22, 0.03), "#5a4030", shade=0.5)

# %% title
# title: 'Night Walk' | at top
cv.text("Night Walk", box=(0.25, 0.04, 0.75, 0.17), color="#fff4d6", font="serif")

# %% finish [pass]
grade(paper(cv), "warm night", 0.7)     # enjoyed pictures only
```

- Units: `sk.*` and `cv.text` take fractions (x of the width; y and lengths of the height); other `cv.*` take pixels.
- Masks: `sk.circle(x, y, r)`, `ellipse(x, y, rx, ry, angle)`, `box(x0, y0, x1, y1, radius)`, `polygon(pts)`,
  `blob(pts)` (smooth, through the points), `stroke(pts, width or [w per point], taper="end")`, `arc(x, y, r,
  start°, end°, width)`; `a * b` clips, `np.maximum(a, b)` joins.
- `sk.paint(mask, color, ...)`: `volume=` (a round body), `shade=`, `highlight=`,
  `spec=` (glossy hot spot), `shadow="ground"` (contact shadow), `glow=color`, `line=True` (a darker edge;
  none in photos and 3D renders); `color=[(0, c0), (1, c1)]` with `angle=90` or `"radial"` is a gradient.
- `sk.spot(x, y, rx, ry, color, opacity)`: blush, light pool; `sk.label("text", at, to=point)`: a name card.
- Animals: `creature(cv, species, at=(x, feet_y), size=h, pose=, facing=)` draws a shaded animal that reads as
  its species (`studio api creature` lists them; one not listed: the nearest, then `recolor`).
- Solids: a surface in perspective, `t = tabletop(cv, top=0.55, front=0.95, horizon=0.4)` or `Ground(cv,
  horizon=0.45)`, gives spots `t.place(u, depth)`; `ball(cv, spot, r)` fruit; `cylinder(cv, spot, r, h,
  handle=True)` mugs, cans; `cuboid(cv, spot, (w, d, h), yaw=30)` boxes; `lathe(cv, spot, [(radius, height),
  ...])` bottles, vases.
- `cv.text(s, box=(x0, y0, x1, y1), color=c, font="serif", bold=True)` auto-fits. Text written ON a thing (a
  sign, a label) is drawn in that thing's section. Draw ONE of a counted thing; `copy` makes the rest.

## Checked pictures

- **Room for every item**: the items first, then every place, surface and material word of the request, each
  readable alone: at least a tenth of the height, whole, apart, unhidden; none over a third of the frame.
- **Plain light, plain finish**: one colour per object, shaded, a clear value step to its ground and neighbours
  (white and grey things on a mid or dark surface); no `grade`, `paper`, vignette or pattern on a requested colour.
- **Draw the part that names the thing**: a bicycle = two wheels, frame, handlebar, saddle; a teapot = body, spout,
  handle, lid. Each requested object gets its defining parts, a body shade, a highlight and a contact shadow
  (`volume=`, `spec=`, `shadow="ground"` or a solid); a bare silhouette reads as something else.
- **See-through and states on the object itself**: `sk.glass(mask, tint=c)` keeps the whole shape, `sk.level(mask,
  0.6, c)` a liquid; `sk.crack`, `sk.shards`, `sk.spill`, `sk.steam`. For steps and riddles draw the end state
  you worked out. What lies on or in front of a thing comes after it in the draft: nothing later may cover it.
- **Places**: left / right / above / below as stated, with a clear gap; "on" = visible contact and a contact
  shadow. The report measures places, counts and colours and gives the fixing command.
- **Labels**: a `sk.label` name card on every invented creature, every look-alike and the answer to a question.
  Quoted text spelled exactly, large and high in contrast.

## Art direction

- **One mood and one light**: `set scene light=top-right grade=golden`, or a last `[pass]` section as above
  (moods: warm night, golden, cosy, moody). The subject fills 40–70 % of the height.
- **One palette**: 4–6 colours plus one saturated accent; outlines a darker tone of the fill (ink styles:
  `line="ink"`), none in photos and 3D renders.
- **Build a named style from its parts**, never filter a finished picture: woodblock = flat fills, key lines,
  `limit_palette(cv, colors)`; watercolour = `sk.wash(m, color)`; comic = bold black outlines, flat colour,
  halftone dots; anime = soft cel, never a flat fill: `sk.cel(m, c, bands=2, line=True)` on skin, hair and cloth,
  `sk.sheen(hair)`, `sk.eye(...)`.

## Kits: where structure is the hard part

Charts, dashboards, diagrams, floor plans, drawings, UI: `studio new NAME --prompt "..."` without `--from`, follow
its NEXT (a `new_scene.py` starter, then `add base --scene`); put the request's exact labels and numbers in
the scene file's data. Other kits are ingredients inside sections.

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

Every command and the picture folder: [studio.md](references/studio.md). Small local models: **pixelcanvas-lite**.

## Local install location

(Filled in by `tools/install_skill.py`; when empty, run the bootstrap script named at the top.)
