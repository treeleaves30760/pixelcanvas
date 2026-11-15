# pixelcanvas

**Your coding agent draws finished pictures with code, then keeps making them better. Each picture is a living stack
of layers the agent edits with tools: one command moves, scales, recolours or redraws one thing, checks the picture
against your request and says what to do next. numpy + Pillow. No image model, no GPU, no API key.**

![The first draft and the finished picture](gallery/studio/campfire_fox_before_after.jpg)

*"A fox asleep beside a small campfire in a snowy pine forest at night, a full moon above the trees, cozy and
quiet"*: the first draft (left) and the finished picture (right).
[The whole session](gallery/studio/sessions/campfire_fox.txt).

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![Agent Skill](https://img.shields.io/badge/Agent%20Skill-SKILL.md-8A2BE2)
![MCP server](https://img.shields.io/badge/MCP-server-0a7)

[繁體中文](README-zh.md)

pixelcanvas is an **Agent Skill** and a **Python package**. It runs in Claude Code, Codex, Cursor, Gemini CLI,
opencode or any agent with a shell, and with local models too: it was tested with Qwen 3.8 Flash Next driving it in
Claude Code and in opencode. The agent reads [`SKILL.md`](skills/pixelcanvas/SKILL.md) and works in
**`pixelcanvas studio`**: the picture lives in `./art/NAME/` as one `# %% name` section of a draft file per object, with
the draft's inventory lines as the checklist, a look sheet and a full revision history.

```bash
npx skills add treeleaves30760/pixelcanvas
```

Then ask: *"Make a poster for a night market called Lantern Street"*, *"Draw the architecture of our commerce
platform"*, and later, about the same picture: *"make the fox bigger"*, *"put the moon on the left"*.

## How it works: a picture your agent keeps editing

The agent writes the whole first draft in one file, `./art/NAME/draft.py`: one script cut into `# %% name` sections,
back to front, one section per thing. The first comment line of a section is its inventory line (count, colours,
'quoted text', states, then `|` and its places):

```python
# %% apple
# apple: 3, green, one bitten | in bowl
ball(cv, t.place(0.3, 0.5), 0.07, base="#7cb342")
```

One command creates the picture from the draft. Every command after it changes one small thing, re-renders only what
changed (about a second), checks the picture against the inventory lines and prints one `NEXT:` line:

```bash
studio new fox --prompt "a fox carrying a lantern through a snowy wood at night, titled 'Night Walk'" --size 1920x1080 --out ./night_walk.png --from ./art/fox/draft.py
studio scale lantern 1.8                     # a change that is not code: one command
studio add --from ./art/fox/draft.py         # a change in code: edit one section, then only it is redrawn
studio note add "3 the lantern glow is faint"   # keeps a list of faults for NEXT to walk through
studio note done 1
```

`studio` is `pixelcanvas studio` (or `python tools/studio.py` from a clone). The `--out` file is rewritten after every
command. Every report ends in one line that says what to do next; when nothing measured is open it opens the look
sheet, and after three looks it says the picture is finished:

```text
request: 4 of 4 confirmed
look list ([x] = what can be measured holds; by eye: does the picture show each line at a glance?):
  [x] 1 wood: snowy ground receding to dark pines, night sky
  [x] 2 fox: 1, orange, walking left to right | at centre
  [x] 3 lantern: glowing warm yellow, its handle in the fox's mouth  <- worth checking: lantern covers 0.4% of the frame: scale lantern 1.8
  [x] 4 title: 'Night Walk' | at top
NEXT: look 1 of 3: open .../art/fox/look.png (the picture, then every item close up); a line above that does not show at a glance: edit its section ... then studio add --from ...
...
NEXT: finished: the picture is .../night_walk.png (open it before you tell the user)
```

The look sheet the agent opens: the picture with a grid and every layer's box, then one close-up per checklist item.

![The look sheet](gallery/studio/campfire_fox_look.jpg)

- **The inventory lines are the contract.** Each section's first comment line becomes a numbered item (its count,
  colours, quoted text and places) and is ticked from the layers when the measured parts hold and the layer is
  visible in the picture. An open item names its fix: `apple: 1 drawn, the request says 3: copy apple --total 3`,
  `apple is not in bowl (boxes: ...): put apple* in bowl`. Measured doubts appear as `worth checking` lines: a
  requested thing under 1.5 % of the frame, copies that touch, a thing that blends into what is behind it.
- **The smallest tool wins.** `move`, `scale`, `fit`, `put`, `order`, `crop`, `recolor`, `copy --total N --row`, `set L
  outline=on shadow=ground glow=ffcf70`, `set scene light=top-right grade=golden`; a shape that does not read gets
  its one section edited and redrawn with `add --from`. `scale` and `fit` pivot on the anchor point of a section placed
  with `anchor("fox", "mouth")`, `copy --row` leaves a gap, `put` keeps things in the frame, and a command that
  covers a thing or pushes it off the frame offers `undo`.
- **Nothing breaks the picture.** A section that fails keeps its last good render and the report names the line of the draft;
  `undo`, `redo` and `goto REV` restore code and parameters together; `compare 5 34` puts two revisions side by side.
- **The file you asked for is always current.** `--out ./night_walk.png` is rewritten after every command; finishing
  needs no extra step.
- **Ask the tool, not the source.** `studio api sk.paint creature` prints each call's exact arguments with an
  example; `studio api steam` lists every call with that word; `studio look --text` gives the numbers to a model
  without vision.

Every command and the picture folder: [studio.md](skills/pixelcanvas/references/studio.md).

## Drawn in the studio

Three pictures drawn through the CLI by following the loop above: first draft, then small fixes.
Each session log lists every command with the request score at that revision.

| | |
|---|---|
| ![Lantern Street poster](gallery/studio/night_market.jpg) | ![Still life](gallery/studio/still_life.jpg) |
| *"A poster for a night market titled 'Lantern Street': three red paper lanterns hanging from a string across the top, a steaming bowl of noodles in the middle, the lines 'Every Friday & Saturday' and '18:00 - 23:00' at the bottom, deep indigo night"*: finished at rev 13, [session](gallery/studio/sessions/night_market.txt) | *"A still life on a wooden table: three green apples in a white bowl, a blue mug right of the bowl, two yellow lemons in front of the bowl, a glass bottle on the left"*: finished at rev 21, [session](gallery/studio/sessions/still_life.txt) |
| ![Poster: first draft and delivered](gallery/studio/night_market_before_after.jpg) | ![Still life: first draft and delivered](gallery/studio/still_life_before_after.jpg) |
| The fixes copied the lantern to three and hung them evenly from the string, moved the bowl down so the steam clears the title, and filled the bowl with noodles, an egg, greens and chopsticks. | The fixes tinted the bottle pale green, copied the apples to three in a larger bowl and the lemons to two in front of it, and cropped to the table. |

![The delivered campfire fox](gallery/studio/campfire_fox.jpg)

The look sheets: [poster](gallery/studio/night_market_look.jpg),
[still life](gallery/studio/still_life_look.jpg).

## Follow-up requests are one command

A delivered picture keeps its state. "Make the fox bigger" is `studio -p campfire_fox scale fox 1.2`, and the file
the user has is redrawn. Four requests after the picture was finished (revs 29 to 32), one command each: `scale fox 1.2`, `recolor moon ffb36b`, `move moon --to 0.3 0.24` and `add owl --kit "creature('owl', at=(0.885, 0.5), size=0.11, pose='perch', facing='left')"`:

![Four follow-up commands on the delivered picture](gallery/studio/campfire_fox_follow_up.jpg)

A request that changes what is asked (another colour, count or text) edits that section's inventory line in
`draft.py`, then `studio -p NAME add --from ./art/NAME/draft.py`, and NEXT takes it from there.

## Every object is a real layer

`studio deliver PATH.png --layers` writes the picture as PSD (Photoshop, Photopea, GIMP, Affinity) and ORA (Krita, GIMP)
next to the PNG, one layer per object, named as in the studio. The still life above, layer by layer (table, bottle, bowl, apple, apple2, apple3, mug, lemon, lemon2):

![The layers of the delivered still life](gallery/studio/still_life_layers.jpg)

`studio export NAME.py` writes the same picture as one scene file for `tools/render.py`.

## Deliver it in a real art medium

A finished drawing can be delivered as a render, a flat graphic, an oil painting, a watercolour, an ink drawing or pixel
art. The layers become per-pixel buffers (local colour, a relief per shape, normals, depth, emission), one key light
lights them with form shadow, cast shadows on the floor and contact occlusion, and the medium paints the result as a
process of its own: oil strokes along the form, watercolour washes with pooled edges on paper, ink contours and hatching
that wraps the form, a pixel grid in a small palette. `studio new` reads the medium from the request ("a watercolour
of", "oil painting", "ink drawing", "pixel art", "3D render"); changing it later runs no section again.

```bash
studio set scene medium=watercolour          # render flat oil watercolour ink pixel off
studio set scene medium_mark=2 medium_strength=0.7 medium_palette=12 medium_paper=f4ecd8
studio stages                                # flat silhouettes -> relief clay -> lit colour -> the medium
```

The still life above, one `set scene medium=` per tile (render, flat, oil / watercolour, ink, pixel):

![The still life in six media](gallery/studio/still_life_media.jpg)

The medium checks keep the drawing's value pattern, the requested colours, the contrast of each thing and legible text;
a failing one is the `NEXT:` line, with its fix. Every dial and what each medium paints:
[medium.md](skills/pixelcanvas/references/medium.md).

## What you can ask for

![Pictures drawn by coding agents with pixelcanvas](gallery/wall.jpg)

Every picture below was drawn by pixelcanvas from the request under it. The scene files are in [`scenes/`](scenes/).

**Posters, print and type**

| | | |
|---|---|---|
| ![type poster](gallery/type_poster.jpg) | ![farmers market](gallery/farmers_market.jpg) | ![jazz poster](gallery/jazz_poster.jpg) |
| *"A bilingual English and Traditional Chinese concert poster for 'NOCTURNE' with a moon, a 'one night only' seal, the date, venue and tickets"* | *"A flyer for a weekend farmers' market in Taipei: title 週末小農市集, illustrated vegetables, warm paper background"* | *"A poster for a jazz night called 'Blue Hour', Friday 21:00 at The Velvet Room, a saxophone, deep blue and gold"* |
| ![cafe menu](gallery/cafe_menu.jpg) | ![readme banner](gallery/readme_banner.jpg) | ![app icons](gallery/app_icons.jpg) |
| *"A café menu for 'Fika & Co.': Coffee, Tea, Pastries, five items each with prices, a coffee cup and a croissant, serif type on cream paper"* | *"A README banner for 'driftwood', a dotfiles tool: bold name, the tagline 'your shell, everywhere', a terminal with three commands"* | *"Twelve app icons for a productivity suite in one style: rounded tiles, soft gradients, long shadows, each labelled"* |

**Screens, charts and diagrams**

| | | |
|---|---|---|
| ![meditation app](gallery/meditation_app.jpg) | ![landing page](gallery/landing_page.jpg) | ![dashboard](gallery/dashboard.jpg) |
| *"A meditation app home screen: 'Good evening, Sam', a breathing timer at 4:32, Sleep / Focus / Calm cards, a tab bar, dark purple theme"* | *"A landing page hero for Northwave: navigation, the headline 'Ship product updates your customers actually read', two buttons and a dashboard screenshot"* | *"An analytics dashboard for a subscription app: MRR, active users, churn and NPS tiles, revenue line, sign-ups by channel, plan mix donut"* |
| ![architecture](gallery/architecture.jpg) | ![flowchart](gallery/flowchart.jpg) | ![sequence](gallery/sequence.jpg) |
| *"An architecture diagram of a commerce platform: clients, CDN and gateway, services, an event bus with workers, data stores. Dark theme, grouped by tier"* | *"A flowchart of a release pipeline from a merged pull request to production, with test, smoke-test and approval decisions and retry loops"* | *"A sequence diagram of a passkey sign-in between the user, browser, API, auth service and user database, with an alt frame for a failed signature"* |

**Engineering drawings and figures**

| | | |
|---|---|---|
| ![bracket drawing](gallery/bracket_drawing.jpg) | ![floor plan](gallery/floor_plan.jpg) | ![stone lantern](gallery/stone_lantern.jpg) |
| *"An A3 engineering drawing of an aluminium angle bracket: front, top, right and section views, an isometric view, dimensions and a title block"* | *"A floor plan of a two-bedroom apartment: doors with swings, windows, furniture, room names with areas, dimensions, a north arrow and a scale bar"* | *"A textbook figure of a Japanese stone lantern with callouts naming its parts from the jewel on top to the base, and its height"* |

**3D and clay characters**

| | | |
|---|---|---|
| ![reading nook](gallery/reading_nook.jpg) | ![clay campsite](gallery/clay_campsite.jpg) | ![earbuds](gallery/earbuds.jpg) |
| *"An isometric cutaway of a cosy reading nook: armchair, floor lamp, a full bookshelf, a round rug, a cat on the armrest, rain outside"* | *"A clay hiker and her dog at a campsite at night: a glowing orange tent, a campfire with rising sparks, pines and the Milky Way"* | *"A product shot of white wireless earbuds in an open charging case on a pale pink seamless backdrop, soft studio light"* |
| ![clay cast](gallery/clay_cast.jpg) | ![clay zoo](gallery/clay_zoo.jpg) | ![toy diorama](gallery/toy_diorama.jpg) |
| *"Meet the cast: ten clay characters in a row, from an explorer to a superhero, each labelled"* | *"A field-guide sheet of thirty clay animals on little stands, each labelled"* | *"Toys on a checker floor: a car, a ball, a puppy, a waving kid, ABC blocks and a rocket"* |

**Illustration, nature and styles**

| | | |
|---|---|---|
| ![ukiyo-e wave](gallery/ukiyoe_wave.jpg) | ![animal stickers](gallery/animal_stickers.jpg) | ![campfire](gallery/campfire_lake.jpg) |
| *"A great wave with Mount Fuji in the distance and two fishing boats, in the style of a ukiyo-e woodblock print"* | *"A sticker sheet of nine cute animals, each with a white die-cut border and a tiny label, on a pastel background"* | *"A campfire on the shore of a mountain lake at night, sparks rising, the moon reflected in the water"* |
| ![aurora](gallery/aurora_lake.jpg) | ![sea sunset](gallery/sea_sunset.jpg) | ![studio still life](gallery/studio_still_life.jpg) |
| *"A wide banner: aurora over a mirror lake between pine forests, the title 'pixelcanvas'"* | *"The sun setting over a calm sea, its path of light on the waves"* | *"A studio still life on a marble board: a glass bottle, a terracotta mug, a glass sphere and a gold ball"* |
| ![art styles](gallery/art_styles.jpg) | ![pictograms](gallery/pictograms.jpg) | ![harbour light](gallery/harbour_light.jpg) |
| *"The same landscape in twelve media"*: 43 named styles, from oil and watercolour to stained glass and blueprint | 633 pictograms in six styles, ready to place by name | *"A red and white lighthouse at dusk, titled 'Harbour Light'"*: the scene file shown below |

## Install

Pick the route that fits your agent; each one is a single step.

| Route | Command |
|---|---|
| Any Agent Skills tool (Claude Code, Codex, Cursor, Gemini CLI, opencode ...) | `npx skills add treeleaves30760/pixelcanvas` |
| Claude Code plugin | `/plugin marketplace add treeleaves30760/pixelcanvas` then `/plugin install pixelcanvas@pixelcanvas` |
| pip, then give the skills to your agent | `pip install pixelcanvas && pixelcanvas install-skill claude` (or `agents`, `codex`, `cursor`, `opencode`) |
| From a clone | `git clone https://github.com/treeleaves30760/pixelcanvas && cd pixelcanvas && pip install -r requirements.txt && python tools/install_skill.py` |

The skill installs a `scripts/bootstrap.py` that sets up the runtime on first use, so the first request just works.
`pixelcanvas doctor` checks Python, numpy, Pillow, fonts and write access; `uvx pixelcanvas doctor` runs it without
installing. Every option, per tool and per project: [docs/INSTALL.md](docs/INSTALL.md).

## Scene files: one picture as one script

Everything the studio draws is Python on the kits, and a whole picture also fits in one scene file (`studio export
NAME.py` writes one). Shapes are placed in fractions of the canvas, and every object comes out lit and shaded, with
glow, soft-cel bands, outlines for flat and ink styles, and exact text built in:

```python
"""lighthouse: red and white, on a dark cliff at the right, lamp glowing, beam over the sea; title 'Harbour Light'"""
from pixelcanvas import Canvas
from pixelcanvas.illustrate import Sketch, paper, grade
PROMPT = "a red and white lighthouse at dusk, titled 'Harbour Light'"
DEFAULT_SIZE = (1600, 900)

def render(width=1600, height=900, seed=1, log=lambda step: None):
    cv = Canvas(width, height)
    sk = Sketch(cv, light=(-0.6, -0.4))
    log("sky and sea")
    cv.gradient([(0, "#1d2150"), (0.5, "#f29a72"), (0.6, "#6a78b0"), (1, "#1c2a55")])
    sk.spot(0.3, 0.6, 0.25, 0.012, "#ffd9a8", 0.6)                         # light on the water
    log("cliff")
    sk.paint(sk.blob([(0.5, 1.05), (0.6, 0.72), (0.78, 0.64), (1.05, 0.62), (1.05, 1.05)]), "#3c3048", shade=0.5)
    log("beam")
    cv.paint("#fff1c4", sk.polygon([(0.81, 0.2), (0.0, 0.3), (0.0, 0.52)]), 0.25)
    log("lighthouse")
    with sk.at(0.81, 0.42, 0.44):                                           # local frame: 1 unit = 0.44 of the height
        sk.paint(sk.polygon([(-0.12, 0.45), (-0.08, -0.35), (0.08, -0.35), (0.12, 0.45)]), "#f4efe6",
                 shade=0.4, line=True, name="lighthouse")
        for y in (-0.2, 0.1, 0.38):
            sk.paint(sk.box(-0.118, y - 0.06, 0.118, y + 0.06), "#d2423c", shade=0.4)
        sk.paint(sk.box(-0.07, -0.5, 0.07, -0.35, 0.02), "#ffe9a8", glow="#ffd27a")
        sk.paint(sk.polygon([(-0.11, -0.5), (0, -0.64), (0.11, -0.5)]), "#d2423c", line=True)
    log("title")
    cv.text("Harbour Light", box=(0.06, 0.08, 0.56, 0.22), color="#fff4e0", font="serif", bold=True)
    grade(paper(cv), "golden", 0.6)
    return cv
```

`python tools/render.py gallery/src/harbour_light.py --check` renders it and writes a report: image critique, a
text audit, every object and where it landed, and the request checked item by item, ending with a FIX block:

```text
PROMPT "a red and white lighthouse at dusk, titled 'Harbour Light'"
  every quoted text is drawn
  COVERAGE (subjects of the request and what answers to them):
    all 1 subjects of the request are accounted for
  AS REQUESTED (count, colour, place and text the request spells out, against the picture; ✗ names what to change):
    lighthouse: 1 drawn, 1 requested ✓; red and white requested, white painted ✓
    'Harbour Light': drawn ✓
FIX: nothing flagged - look at the picture once, then deliver
```

The picture that scene draws:

![Harbour Light](gallery/harbour_light.jpg)

Charts, dashboards, diagrams, floor plans, drawings and UI start from one of 30 starter templates and come into the
studio as one layer, with callouts and stamps on top as layers of their own:

```bash
pixelcanvas new sales --dir . --prompt "quarterly sales as a bar chart, titled 'Sales 2025'"
pixelcanvas studio new sales --prompt "quarterly sales as a bar chart, titled 'Sales 2025', stamped 'Draft'"
pixelcanvas studio add base --scene sales.py
pixelcanvas studio add stamp --text "Draft" --box 0.72 0.05 0.95 0.14 --color d63c2f --bold
```

## Match a reference image

Hand the agent a picture and say "in this style". `match.py` reads the reference as numbers, the agent draws with
them, and the same tool lists what still differs, largest gap first.

```bash
python tools/match.py REF.png               # style sheet: palette with roles, lines, tone bands, highlights
python tools/match.py REF.png DRAFT.png     # the draft's gaps to the reference and one match figure
```

On the anime illustration in [`scenes/illustration_anime.py`](scenes/illustration_anime.py):

```text
roles: background #508fdd, skin #ef8987, line #5d313a, accent #a62250
lines: #5d313a (coloured), width 0.0037 of the height (~4 px at 1080)
shading by region (largest first):
  #d1535e 11%: hard bands, 3 tones; shadow #b12854 = L -0.12, hue -13 deg; highlight band #f88f96 (L +0.17)
  #fde3d4 5% (skin): hard bands, 3 tones; shadow #dabbb0 = L -0.12, hue -10 deg
PASTE (scene docstring / constants):
  PALETTE = ["#508fdd", "#80b7ed", "#b8d4ec", "#f4efe9", "#d2555f", "#a62250", "#e89c2a", "#ef8987"]
  LINE, LINE_WIDTH = "#5d313a", 0.0037      # sk.cel(..., line=LINE, line_width=LINE_WIDTH)
  SHIFT, BANDS = (-0.119, -11.9), 2     # sk.cel(mask, base, bands=BANDS, shift=SHIFT)
```

A draft with purple hair against that reference:

```text
match 89/100 against the reference
  1. [palette gap 0.17] palette distance 0.021 (OKLab) - use the reference colours: #d2555f (draft nearest #896eb3), #a62250 (draft nearest #664697), #ef8987 (draft nearest #e59c35)
  2. [lines gap 0.26] line #372d3d width 0.0037 vs #5d313a width 0.0037 - line="#5d313a", line_width=0.0037
  3. [contrast gap 0.35] value range 0.35-1.00 vs 0.49-1.00 - less contrast
```

Inside the render loop, `render.py SCENE --check --ref REF.png` (or `REFERENCE = "REF.png"` in the scene) adds the
same gaps as a REFERENCE block. The protocol: [reference.md](skills/pixelcanvas/references/reference.md).

## What is in the box

| Kit | Draws | Reference |
|---|---|---|
| `illustrate` | freehand drawing: `Sketch` shapes in fractions, local frames, blobs, tapered strokes, lit shading, soft-cel bands, sheen, eyes and blush, stickers, washes; `cv.draw()` gives a supersampled Pillow layer | [illustration.md](skills/pixelcanvas/references/illustration.md) |
| `reference` | match a reference image: style sheet, paste-ready constants, gaps of a draft (`tools/match.py`) | [reference.md](skills/pixelcanvas/references/reference.md) |
| `text`, `layout` | type in any language with CJK fallback, auto-fit, effects; poster and card templates, themes | [typography.md](skills/pixelcanvas/references/typography.md) |
| `chart` | charts, KPI tiles and whole dashboards | [charts.md](skills/pixelcanvas/references/charts.md) |
| `diagram` | flowcharts, architecture, trees, sequences, timelines, tables, callouts, with automatic layout | [diagrams.md](skills/pixelcanvas/references/diagrams.md) |
| `ui` | app and web screens in device frames, with auto layout | [ui.md](skills/pixelcanvas/references/ui.md) |
| `drafting` | engineering sheets (views, sections, dimensions, title blocks) and floor plans, to PNG, SVG and DXF | [drafting.md](skills/pixelcanvas/references/drafting.md) |
| `vector` | shapes, paths, icons and groups to raster and SVG | [vector.md](skills/pixelcanvas/references/vector.md) |
| `pictograms` | 633 pictograms in six styles, also usable as masks | [pictograms.md](skills/pixelcanvas/references/pictograms.md) |
| `solid` | numpy ray-marched 3D: prefabs, materials, studio light, perspective and isometric cameras | [solid.md](skills/pixelcanvas/references/solid.md) |
| `clay` | 3D clay characters in 31 poses, 64 animal species, props | [clay.md](skills/pixelcanvas/references/clay.md) |
| `figure`, `creature` | posed, dressed people and animals on a skeleton | [figures.md](skills/pixelcanvas/references/figures.md), [creatures.md](skills/pixelcanvas/references/creatures.md) |
| `flora`, `elements`, `scenery` | trees, flowers, fire, smoke, water, weather, skies, landscapes | [flora.md](skills/pixelcanvas/references/flora.md), [elements.md](skills/pixelcanvas/references/elements.md) |
| `texture`, `form` | 80 lit materials; 2.5D objects and rooms with one light | [textures.md](skills/pixelcanvas/references/textures.md), [forms.md](skills/pixelcanvas/references/forms.md) |
| `backdrop`, `palette` | one-call backgrounds and wallpapers that keep a calm area for text; named palettes | [backdrops.md](skills/pixelcanvas/references/backdrops.md) |
| `stylize` | 43 art styles, chosen from the request with `styles pick` | [styles.md](skills/pixelcanvas/references/styles.md) |

`pixelcanvas kit find "REQUEST"` searches every kit at once, and 30 starter templates (`pixelcanvas new --list`)
cover posters, charts, diagrams, UI, floor plans, product shots, people, pets and more.

## Local models

The studio loop runs with a model on your own machine: one `NEXT:` line after every command keeps a small model on
track, and a model without image input checks the picture with `studio look --text` (each layer's box, size and mean colour).
It was tuned with Qwen 3.8 Flash Next driving it in Claude Code and in opencode. For a vision model in opencode,
declare that the model takes images ([docs/INSTALL.md](docs/INSTALL.md#local-models-in-opencode-the-studio-with-vision))
and it reads the `look` sheet itself.

For the smallest models there is the **pixelcanvas-lite** skill: the model writes one small JSON spec and `compose`
draws it through the kits, in 16 kinds from posters and wallpapers to dashboards, rooms and 3D scenes. Every run ends
with one `NEXT:` line. No Python to write.

```bash
pixelcanvas install-skill opencode --skill pixelcanvas-lite
pixelcanvas compose --example poster > poster.json
pixelcanvas compose poster.json --check
pixelcanvas compose poster.json --deliver final/
```

The spec format: [spec.md](skills/pixelcanvas/references/spec.md).

## Layers, SVG and DXF from scene files

In a scene file every `log("step")` becomes a layer: `render.py --layers` writes `SCENE.psd`, `SCENE.ora` and a
folder of PNGs, as `studio deliver PATH.png --layers` does for studio pictures. `--svg` writes the vector drawings (diagrams,
charts, UI, icons) as SVG, and `--dxf` writes drafting sheets for CAD.

```bash
python tools/render.py drafting_bracket --svg --dxf
python tools/render.py clay_campsite --layers
```

## MCP server

`pixelcanvas mcp` speaks the Model Context Protocol on stdin / stdout with 15 tools and every reference page as a
resource. Four of them are the studio: `studio` (any studio command, returns the report and the look sheet),
`studio_draft` (the whole draft in one call), `studio_layer` (redraw one object) and `studio_read`. The
others: `kinds`, `new_scene`, `write_scene`, `read_scene`, `render`, `look`, `find`, `reference`, `ask`, `animate`,
`layers`.

```bash
claude mcp add --scope project pixelcanvas -- python -m pixelcanvas.mcp_server
codex mcp add pixelcanvas -- python -m pixelcanvas.mcp_server
```

Claude Desktop, Cursor and opencode configs: [docs/INSTALL.md](docs/INSTALL.md#as-an-mcp-server).

## Reference pages

- [SKILL.md](skills/pixelcanvas/SKILL.md): the studio loop, the first draft, checked pictures, art direction, the kit table
- [studio.md](skills/pixelcanvas/references/studio.md): every studio command, the picture folder, the draft format, reports, looks
- [api.md](skills/pixelcanvas/references/api.md): the full API index
- [illustration.md](skills/pixelcanvas/references/illustration.md), [techniques.md](skills/pixelcanvas/references/techniques.md),
  [typography.md](skills/pixelcanvas/references/typography.md), [review.md](skills/pixelcanvas/references/review.md)
- one page per kit in [`skills/pixelcanvas/references/`](skills/pixelcanvas/references/)
- [CHANGELOG.md](CHANGELOG.md): what is new in 0.5

## Contributing

Pictures, kits and fixes are welcome. [`CLAUDE.md`](CLAUDE.md) has the project map and the steps for a new kit (a
registry, strict options, manifest hooks, a reference page whose code blocks run, a starter template and an example
scene). Run the tests before a pull request:

```bash
pip install -r requirements.txt pytest
python -m pytest -q
```

`python tools/build_gallery.py` rebuilds the gallery wall and its pictures from the scenes; `gallery/studio/` holds
the studio pictures and their session logs.

## Licence

MIT, see [LICENSE](LICENSE).
