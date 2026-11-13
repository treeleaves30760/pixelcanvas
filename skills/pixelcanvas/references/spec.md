# Spec mode (`tools/compose.py`, `pixelcanvas.spec`)

A picture as one small JSON file: name the things, say where they go, write the exact text. `compose.py` checks
the file, draws it through the kits, prints the same report as `render.py --check`, then a list of EDITS and one
closing `NEXT:` line. No Python to write, so it suits small local models and quick drafts alike; the
`pixelcanvas-lite` skill teaches only this.

```bash
python <root>/tools/compose.py --example poster > ./art/poster.json      # a complete spec for the kind
python <root>/tools/compose.py ./art/poster.json --check                  # draft + report + EDITS + NEXT
python <root>/tools/compose.py ./art/poster.json --deliver ./art/final/   # full size, copied
```

## Commands

| Command | What it does |
|---|---|
| `compose.py SPEC.json --check` | validate (all problems at once), draw a 0.5 draft, report, EDITS, one `NEXT:` line |
| `--deliver PATH` | full size (`--scale 1`) and copy PNG (+ SVG for diagrams, charts, UI) to PATH |
| `--scale 0.5` `--seed N` `--layers` `--svg` `--json` | as in `render.py` |
| `--set layers.1.size=0.5` `--set style=gouache` | change one value in the file, then draw |
| `--add '{"draw": "moon", "at": "top right"}'` `--remove moon` | append / remove a layer (index or name) |
| `--example KIND` `--kinds` `--schema` `--names WORD` | starting specs, the kinds, the format, drawable names |

Outputs go to `outputs/` next to the spec (`NAME_x0.5_preview.jpg`), exactly like a scene file.
The reader is forgiving: comments, trailing commas, single or curly quotes, a Markdown fence, text around the
JSON, a wrapper object, two objects in a row (the first is kept), `True` / `None`, bare words, `"objects"` for
`"layers"`, `"colour"`, `"position"`, sizes such as `"1920x1080"` or `["1080", "1350"]`, numbers in quotes,
`"two birds"` (count 2), `"ginger cat"` (coat ginger), a title in quote marks, `"pose": "sit, hold"`,
`"sky": "night stars"`, `"far": "dark trees"`, `"at": "in Sam's hands"`, `"right of the second person"`, cards
as a string, `"Yes -> Refill"` as an edge target (a labelled edge) are all read, and every repair is listed under
READ.
Placeholders copied from instructions (`"NAME"`, `"<the title>"`) and YAML are answered with one line that says
what to write instead.

## The format

| Field | Values |
|---|---|
| `pcscene` | `1` |
| `prompt` | the request, verbatim (the report checks quoted text, counts, style against it) |
| `kind` | `poster card wallpaper icons scene nature animals people landscape still_life room scene3d diagram chart dashboard ui` |
| `size` | `[w, h]`, `"1080x1350"`, or `portrait landscape square story thumbnail banner` (default per kind) |
| `palette` | a name (`sunset`), a description (`"deep blue and gold"`) or a list of colours |
| `background` | scenes: `sky ground far horizon`; designs: `backdrop color gradient texture landscape`; still life: `table wall` |
| `layers` | a list; each layer draws one thing (or `count` copies) or one text |
| `style` | `null` or a style name (`watercolor`, `gouache`, `pixel_art` ...); `null` + a medium in the prompt applies it |
| `post` | `{"vignette": 0.3, "grain": 0.01, "bloom": 0.2}` or `false` |
| `diagram` `chart` `dashboard` `ui` `scene3d` | the section of a whole-picture kind (below) |

**A thing layer**: `{"draw": NAME, "at": WHERE, "size": SIZE, "color": COLOUR, "count": N, "arrange": A}` plus
kit options in the same object: `pose`, `coat`, `facing` (animals), `describe`, `pose` (people), `style`
(pictograms), `material` (still-life objects).
NAME is anything `--names` prints: animals (`cat`, `sparrow`, `horse`), people (`person` with `describe`, or a
preset such as `chef`), plants and rocks (`oak`, `daisy`, `potted monstera`), effects (`sun`, `moon`, `cloud`,
`campfire`, `rain`, `stars`), still-life objects (`orange`, `mug`, `vase`, `bottle`, `book`), pictograms
(`saxophone`, `bicycle`) and icons (`heart icon`). An unknown name is an error that lists the closest names.

**WHERE**: `centre`, `left`, `right third`, `top left`, `bottom`, `far`, `foreground`, `in the sky`,
`on the ground`; relations to another layer: `left of cat`, `right of tree`, `next to`, `on table`, `under oak`,
`behind tree`, `in front of house`, `above tree`; or `[u, v]` fractions (the base of a thing that stands on the
ground, the centre of anything else) or a box `[u0, v0, u1, v1]`. In design kinds, words place things inside the
free picture area the text leaves. In scenes, a relation the prompt states ("a cat under the oak") places a
layer that has no relation of its own, and a thing that stands on the ground keeps its base on the ground.
**SIZE**: `tiny small medium large huge` or a fraction of the picture height. **A**: `row cluster scatter
column circle grid`.
Order does not matter in scenes: things farther back are drawn first, `behind` / `in front of` / `on` are
honoured, and things on the ground get contact shadows.

**A text layer**: `{"text": "exact words", "role": ROLE}` with ROLE in `title subtitle kicker details body caption
label badge footer byline quote date meta`. Posters use the poster template (title band, picture area, details in
columns); cards and wallpapers stack the text; scenes put titles on top and captions at the bottom. `"at"` or
`"box"` places one text explicitly; `"color"` sets its ink.

**Escape hatch**: `{"python": "my_shapes.py:draw_logo", "at": [0.3, 0.3, 0.7, 0.7]}` calls
`draw_logo(cv, box_in_pixels, layer)` from a file next to the spec.

## Sections

| Kind | Section |
|---|---|
| `diagram` | `{"title", "theme", "direction", "nodes": ["Web app", {"id", "label", "icon", "shape"}], "edges": [["Web app", "API", "HTTPS"]], "groups": {"Clients": ["Web app"]}}`; icons and database shapes are picked from the labels |
| `chart` | `{"type": "bar", "data": {"Q1": 120}, "title", "x", "xlabel", "ylabel", "yfmt"}`; types `bar barh line area pie donut scatter hist stacked heatmap radar spark` |
| `dashboard` | `{"title", "subtitle", "theme", "kpis": [{"label", "value", "delta", "spark"}], "charts": [chart, ...]}` |
| `ui` | `{"template": "home", "screen": "phone", "theme": "light", "accent", "content": {"title", "cards": [...]}}`; `home` (greeting + cards) or any `ui.template` name (`login`, `dashboard`, `chat` ...) |
| `scene3d` (kinds `room`, `scene3d`) | `{"room": {"walls", "floor"}, "objects": [{"draw": "bed", "at": "back left", "on": "desk", "color", "rotate"}], "lighting", "camera": "isometric"}` |

Rooms place furniture along the walls when `at` is left out; `on` stands an object on another (`lamp` on `desk`).
Without a room, the objects form one group and `at` (words or numbers) sets only their left-to-right order.

## The report, EDITS and NEXT

After the usual WARNINGS / TEXT / OBJECTS / PROMPT sections, compose prints at most five EDITS, phrased as
changes to the file, and always ends with exactly one `NEXT:` line: the one thing to do now.

```text
EDITS (2 of 2; make them all in art/poster.json):
  1. the request quotes 'Blue Hour' but no text layer contains it -> add {"text": "Blue Hour", "role": "title"}
  2. layers[2] 'bird' is 80% hidden by 'oak' -> set "at" to [0.31, 0.42] (or move 'oak')
NEXT: make the 2 EDITS in art/poster.json (write the whole file again), then run: python .../compose.py art/poster.json --check
```

`NEXT:` is one of: fix the PROBLEMS or EDITS and check again; `deliver: ...` (0 fail and no edit left, or four
checked drafts); `tell the user the picture is at /abs/path.png` after `--deliver`. Every other output (unknown
option, missing file, `--example`, `--names`, a Python file passed by mistake) ends the same way.

## Composed layouts

Each design kind lays itself out: a **poster** stands its largest thing as the focal subject on a lit disc,
the other things flank it on one baseline, and the template sets kicker, title, subtitle and details; a
**card** puts the illustration on an accent panel (the title's initial when there is none) beside a text column
of kicker, title, an accent rule and an author line with an initial badge; a **chart** gets a header band with a
one-line reading of the data, the chart on a card, and the top bar highlighted; a **wallpaper** sets its words
quietly in the calmest band (the sky above a `mountains`, `hills` or `dunes` backdrop) in ink that reads there.

Scenes compose themselves too: two or more **people** placed by plain words stand together as a group and turn
towards each other; a thing **under a tree** lies or stands beside the trunk, a step in front of it; a **still
life** gathers in the middle of the table; a **product shot** (`scene3d` without a room) stands its objects
side by side on a studio sweep, the tallest in the middle, and the camera frames the group; a **room** puts
floor lamps, armchairs and side tables in their own spots and draws small things on a table a little larger.
A phone `home` screen adds a progress ring and a week chart around the cards.
Friends by a **campfire** sit on either side of it; "playing the guitar" gives a person the `guitar` pose; `star`
in a scene is the starry sky; a nearly empty **room** gets a rug, a plant and a lamp on the desk. In posters,
`sea` or `ocean` is the `ocean` backdrop and kites, balloons and birds fly above the stage. In **diagrams** a
question (`"Tank empty?"`) is a diamond, `Start` / `End` are pills, and a step with no arrow is joined from the
step before it.

## Recipes

**Poster**: picture in the free area, text by the template.

```json
{"pcscene": 1, "kind": "poster", "prompt": "A jazz poster called 'Blue Hour', Friday 21:00, deep blue and gold",
 "palette": "deep blue and gold", "background": {"backdrop": "spotlight"},
 "layers": [{"draw": "saxophone", "at": "centre", "size": "large", "color": "gold"},
            {"text": "Blue Hour", "role": "title"}, {"text": "Friday 21:00", "role": "details"}]}
```

**Scene with relations and counts**:

```json
{"pcscene": 1, "kind": "animals", "prompt": "A ginger cat under an oak at sunset, two birds in the sky",
 "background": {"sky": "sunset", "ground": "grass", "far": "hills"},
 "layers": [{"draw": "oak", "at": "right third", "size": "huge"},
            {"draw": "ginger cat", "pose": "sit", "at": "under oak"},
            {"draw": "bird", "count": 2, "at": "top left"}]}
```

**Room in 3D**:

```json
{"pcscene": 1, "kind": "room", "prompt": "A cosy isometric bedroom with a bed, a desk with a lamp and a plant",
 "scene3d": {"room": {"walls": "#efe2cf", "floor": "wood"}, "camera": "isometric",
             "objects": [{"draw": "bed"}, {"draw": "desk"}, {"draw": "lamp", "on": "desk"}, {"draw": "plant"}]}}
```

In Python: `norm, problems = spec.validate(spec.load(path)[0])`, `cv = spec.render(norm, 960, 540)`;
`spec.LAST["placed"]` holds each layer's box and `spec.next_steps(norm, report)` builds the EDITS list.
