# pixelcanvas

**Your coding agent draws finished pictures with code: posters, diagrams, dashboards, 3D scenes and
illustrations, with numpy + Pillow. No image model, no GPU, no API key.**

![Pictures drawn by coding agents with pixelcanvas](gallery/wall.jpg)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![Agent Skill](https://img.shields.io/badge/Agent%20Skill-SKILL.md-8A2BE2)
![MCP server](https://img.shields.io/badge/MCP-server-0a7)

[繁體中文](README-zh.md)

pixelcanvas is an **Agent Skill** and a **Python package**. Claude Code, Codex, opencode, Cursor, Gemini CLI or any
agent with a shell reads [`SKILL.md`](skills/pixelcanvas/SKILL.md), designs the picture, draws it in Python, runs a
check that names what to fix, and looks at the result before handing it over. Every picture is a short scene file:
exact text in any language, editable layers, SVG and DXF when you want them.

```bash
npx skills add treeleaves30760/pixelcanvas
```

Then ask: *"Make a bilingual poster for a piano night called NOCTURNE"*, *"Draw the architecture of our commerce
platform"*, *"An isometric reading nook with a cat on the armchair"*.

## What you can ask for

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
| *"The same landscape in twelve media"*: 43 named styles, from oil and watercolour to stained glass and blueprint | 633 pictograms in six styles, ready to place by name | *"A red and white lighthouse at dusk, titled 'Harbour Light'"*: the scene in the next section |

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

## How it works: plan, draw, check, look

**1. Plan.** The agent writes an inventory of the picture in the scene's docstring and copies the request into
`PROMPT`, word for word.

**2. Draw.** It writes the whole scene in one go. Shapes are placed in fractions of the canvas, and every
object comes out lit and shaded, with glow, soft-cel bands, outlines for flat and ink styles, and exact text built in:

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

**3. Check.** `python tools/render.py gallery/src/harbour_light.py --check` renders it and writes a report: image
critique, a text audit, every object and where it landed, and the request checked item by item. It ends with a
FIX block that names what to change, or clears the picture:

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

**4. Look.** The agent opens the PNG, and zooms in with `tools/look.py` (guides, zoom, value map, edges, probe)
where a detail needs a closer look. Here is the picture that scene draws:

![Harbour Light](gallery/harbour_light.jpg)

For structured pictures the agent starts from a template instead of a blank file:

```bash
pixelcanvas kit find "a corgi sitting under an oak tree at sunset"      # the call for each subject
pixelcanvas new corgi --prompt "a corgi sitting under an oak tree at sunset" --dir ./art
pixelcanvas render ./art/corgi.py --scale 0.5 --check
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

## Small local models

A model running on your own machine draws through the **pixelcanvas-lite** skill: it writes one small JSON spec and
`compose` draws it through the kits, in 16 kinds from posters and wallpapers to dashboards, rooms and 3D scenes.
Every run ends with one `NEXT:` line that says what to do. No Python to write.

```bash
pixelcanvas install-skill opencode --skill pixelcanvas-lite
pixelcanvas compose --example poster > poster.json
pixelcanvas compose poster.json --check
pixelcanvas compose poster.json --deliver final/
```

The spec format: [spec.md](skills/pixelcanvas/references/spec.md).

## Editable layers, SVG and DXF

Every `log("step")` in a scene becomes a layer. `--layers` writes the picture as `SCENE.psd` (Photoshop, Photopea,
GIMP, Affinity), `SCENE.ora` (Krita, GIMP) and a folder of PNGs, so a person can move, hide or repaint any object.
`--svg` writes the vector drawings (diagrams, charts, UI, icons) as SVG, and `--dxf` writes drafting sheets for CAD.

```bash
python tools/render.py drafting_bracket --svg --dxf
python tools/render.py clay_campsite --layers
```

## MCP server

`pixelcanvas mcp` speaks the Model Context Protocol on stdin / stdout with 11 tools (`kinds`, `new_scene`,
`write_scene`, `read_scene`, `render`, `look`, `find`, `reference`, `ask`, `animate`, `layers`) and every reference
page as a resource.

```bash
claude mcp add --scope project pixelcanvas -- python -m pixelcanvas.mcp_server
codex mcp add pixelcanvas -- python -m pixelcanvas.mcp_server
```

Claude Desktop, Cursor and opencode configs: [docs/INSTALL.md](docs/INSTALL.md#as-an-mcp-server).

## Reference pages

- [SKILL.md](skills/pixelcanvas/SKILL.md): the loop, freehand drawing, checked pictures, art direction, the kit table
- [api.md](skills/pixelcanvas/references/api.md): the full API index
- [illustration.md](skills/pixelcanvas/references/illustration.md), [techniques.md](skills/pixelcanvas/references/techniques.md),
  [typography.md](skills/pixelcanvas/references/typography.md), [review.md](skills/pixelcanvas/references/review.md)
- one page per kit in [`skills/pixelcanvas/references/`](skills/pixelcanvas/references/)
- [CHANGELOG.md](CHANGELOG.md): what is new in 0.4

## Contributing

Pictures, kits and fixes are welcome. [`CLAUDE.md`](CLAUDE.md) has the project map and the steps for a new kit (a
registry, strict options, manifest hooks, a reference page whose code blocks run, a starter template and an example
scene). Run the tests before a pull request:

```bash
pip install -r requirements.txt pytest
python -m pytest -q
```

`python tools/build_gallery.py` rebuilds this gallery from the scenes.

## Licence

MIT, see [LICENSE](LICENSE).
