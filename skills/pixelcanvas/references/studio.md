# Studio: a picture you edit step by step

The studio keeps the picture as a stack of layers in `./art/NAME/`. You write the first draft as one Python
script cut into `# %% name` sections, one per thing; each section becomes a layer. Commands move, scale,
re-order, outline, shadow and recolour layers without drawing them again; every command re-renders only what
changed, checks the picture against the checklist your sections declare and prints one NEXT line. Follow the
NEXT lines: they fix failing sections, then open checklist items, then ask you to look at the picture until it
is finished.

```bash
# write ./art/fox/draft.py with your file tool, then one command creates the picture and draws it:
python <root>/tools/studio.py new fox --prompt "a fox carrying a lantern through a snowy wood at night, titled 'Night Walk'" --size 1920x1080 --out ./night_walk.png --from ./art/fox/draft.py
python <root>/tools/studio.py scale fox 1.3
python <root>/tools/studio.py note add "3 the lantern needs a warmer glow"
```

`new NAME --prompt "..."` without `--from` creates the picture and prints the path to write the draft to and
its format; `add --from` then draws it. `--out PATH.png` names the file the user gets: it is written after every
command that changes the picture, so it always holds the latest state and finishing needs no command;
`deliver PATH.png [--scale 2] [--layers]` writes another copy.
`pixelcanvas studio ...` and `python <root>/tools/pc.py studio ...` are the same command. The last picture used
is the current one; `-p NAME` picks another, `--dir DIR` another folder (default `./art`).

## The draft is one script

Sections run top to bottom and share names, exactly like one script: a helper `def house(x, y)`, a palette
`cols = {...}`, a generator `rnd = np.random.default_rng(3)`, a surface `t = tabletop(...)` or an animal
`fox = creature(...)` defined in one section is there in every section below it. Code before the first `# %%`
line (imports, helpers) is shared the same way. Each section is still its own layer: it is cached on its own,
and when you edit one section only it and the sections that take names from it are drawn again. A section that
fails leaves the others intact; a name no section above defines is reported in plain words (`house` is not
defined in this or an earlier section). `# %% sled dog` makes layer `sled_dog`; later commands accept both.

```python
def lamp(x, y, s):
    sk.paint(sk.circle(x, y, 0.02 * s), "#ffcf70", glow="#ffcf70")

# %% street
# street: night, wet cobbles
cv.gradient([(0, "#0b1030"), (0.6, "#24305e"), (1, "#3d4a7a")])
rnd = np.random.default_rng(5)
for i in range(30):
    sk.paint(sk.ellipse(rnd.uniform(0, 1), rnd.uniform(0.75, 1), 0.03, 0.01), "#4a5878")

# %% lamps
# lamps: 2, glowing | at centre
for x in (0.3, 0.7):
    sk.paint(sk.box(x - 0.005, 0.3, x + 0.005, 0.8), "#222831")
    lamp(x, 0.28, 1.5)
```

## Inventory lines: the checklist

The first comment line of each section is that thing's inventory line, and the checklist is built from them:

    # bus: 1, yellow, overturned on its side, wheels up to the sky, windows shattered | on road
    # caution tape: yellow, cordons off the scene | in front of bus
    # apple: 3, green | in bowl
    # sign: wooden, says 'OPEN' | left of door

`NAME: bit, bit, ... | place, place`. NAME is what the request calls the thing (section `tape`, item `caution
tape`). The bits: a leading count (`3`, `three`, `a pair of`), colours (`yellow`, `dark green`, `red and
white`, `#c84`), quoted text (`'OPEN'`, written in that section), and every other bit is a state you check by
eye. Places: `left of X`, `right of X`, `above X`, `below X`, `on X`, `in X`, `behind X`, `in front of X`, `next
to X`, `at left | right | top | bottom | centre | top left ...`, X being another section's name or item name.

| Measured | Ticked when | Fix the report names |
|---|---|---|
| drawn | the section's layer draws and is visible | the draft section to edit |
| count | that many copies (or separate parts, or kit records), none touching | `copy L --total N --row X0 Y0 X1 Y1` |
| place | the solid parts agree (glow, smoke, sparks and a thin shadow left out; above or beside a forest, a wall or a backdrop: beyond its mass); behind / in front of from the stack where they overlap | `put`, `move`, `order` |
| seen | at least half of it visible in the composite | `order L above X` (or `move` when it stands behind X) |
| colour | the colour covers a fifth of its paint or is its largest colour | `recolor L 4caf50` |
| text | the section writes it, inside the frame, readable | the draft section to edit, `fit`, `set L outline=...` |

One layer with several misses is fixed in that order: its place first, then what covers it. States (`overturned
on its side`) are for your eyes when you look. A section without an inventory line is the plain item `name`; a
backdrop or a pass gives an item only with an inventory line. From the request itself only quoted strings become
items (`text 'Closed'`); a quoted string an inventory line already writes is one item. Re-submitting the draft
rebuilds the items whose line changed and keeps the ticks of the others; `need add "TEXT" [--layer L]`, `need ok N
--why ".."`, `need set N "..."`, `need rm N` correct the list by hand, and items you add are never rebuilt.

## The picture folder

| Path | What it holds |
|---|---|
| `picture.json` | request, size, kind, seed, scene look, layers back to front with their place and effects, the program order of the sections, the checklist, notes, looks, revision |
| `layers/<name>.py` | one cell per section; `add --from` writes them, and `render` picks up a cell file you edit |
| `preview.png` | the current picture, long side 1280 px, always at this path |
| `look.png` | the sheet to open, written after every change: the picture with a grid and layer boxes, then every item close up in the picture |
| `cache/`, `history/` | each cell's render (an unchanged cell never runs again); every revision, cell code included |

## Cells

A cell sees:

| Name | Meaning |
|---|---|
| `cv`, `sk` | the layer's own canvas and `Sketch(cv, light=<scene light>)`: fractions, shading, glow, shadow |
| `W`, `H`, `S`, `seed`, `rng`, `np`, `math`, `random` | size in pixels, `S = H / 1080`, the seed, a seeded generator, numpy, math, `random` seeded per section |
| `creature`, `tabletop`, `Ground`, `ball`, `cylinder`, `cuboid`, `lathe` | animals and solids in one call |
| `paper`, `grade`, `limit_palette`, `manifest`, `Canvas` + all of `from pixelcanvas import *` | finishing, records, helpers |
| `flora`, `elements`, `solid`, `form`, `figure`, `clay`, `pictograms`, `backdrop`, `stylize` ... | the kits, loaded on first use |
| `box("fox")`, `anchor("fox", "mouth")` | where layer `fox` is now: its box, or a landmark (`center`, `top`, ...; a creature adds `mouth`, `nose`, `head`, `tail`, paws) |
| `scene("chart.py")` | the whole picture a scene file draws: `cv = scene("chart.py")` |

Names the reference pages teach are there too: `Frame`, `set_light`, `contact_shadow`, `cast_shadow`, `Room`,
`shelf`, `row`, `book`, the `sd_*` shapes, `Creature`, `person`, `Figure`; a quick-start block of a page runs as a
section once its `Canvas(...)` and `save` lines are left out. Colours take `"#3a4048"` or a bare `"3a4048"`, as the
commands do; `name=` on a kit call is ignored (the section name is the layer name). Before a section runs, every
keyword its calls do not take is reported at once, each with its did-you-mean and the call written out. A section
of comments only is skipped (`finish: no code, skipped`), no item and no error.

**A 3D scene** (`solid`) is one section: one camera, one light rig, and the shadows and occlusion between its
objects. Name every object (`sc.add(car(...), "car")`); the report lists the named objects with their boxes and
flags one cut by or outside the frame, so a whole 3D render in one section is still measured object by object.

A section that reads `box()` or `anchor()` follows that layer: move the fox and the lantern in its mouth is drawn
again at the new place. Roles: an object (default), a backdrop (fills the frame; found by itself) or a pass that
works on the picture below it (`# %% finish [pass]`, or found by itself). An error line names the line in the
draft and the likely cause; a wrong argument adds `-> studio api NAME`. A cell runs at most 60 seconds (`set scene
cell_timeout=N`). `studio api sk.paint creature` prints each call's arguments; `api` alone the short map.

## Commands

| Command | Does |
|---|---|
| `new NAME --prompt "REQUEST" [--size WxH] [--out PATH.png] [--kind K] --from DRAFT.py` | the picture and its first draft in one command; again on the same picture, the draft goes into it |
| `add --from DRAFT.py` | the draft again: changed sections are drawn again, unchanged ones come from the cache, moves and effects of sections that stay are kept, a new section goes where the draft puts it, a section that left the draft is removed |
| `add NAME --text "Night Walk" [--box X0 Y0 X1 Y1 --color fff4d6 --font serif --bold]` | a text layer |
| `add NAME --kit "creature('fox', pose='walk')"`, `add NAME --scene FILE.py` | a layer from one kit call; a whole chart, diagram or UI scene as a backdrop |
| `render` / `status [--all] [--fresh]` | picks up edited cell files, renders what changed, reports (`status` starts with the request) |
| `move L.. --to X Y` / `--by DX DY`, `scale L.. F`, `rotate L.. DEG`, `flip L..`, `fit L.. X0 Y0 X1 Y1` | place and size; scaled from the bottom centre (a section placed with `anchor()`: from that anchor point), drawn again larger, never blown up |
| `order L.. top\|bottom\|up\|down\|above X\|below X`, `hide`, `show`, `rm` | the stack |
| `copy L --total N [--row X0 Y0 X1 Y1 \| --grid X0 Y0 X1 Y1 --cols C]` | copies L2, L3 ... up to N, set apart with a gap (shrunk when the row is too narrow); with N copies already there it lays them out again |
| `set L key=value ..`, `recolor L.. COLOUR` | `outline`, `shadow=ground\|drop`, `glow`, `tint`, `opacity`, `blur` ...; a value runs to the next `key=` (`grade=warm night`); colour kept in its shading |
| `set scene key=value ..` | `light`, `medium=oil` (and `medium_strength`, `medium_mark`, `medium_palette`, `medium_paper`), `grade`, `paper`, `background`, `size`, `prompt`, `reviews=3`, `cell_timeout`, `out` |
| `stages [MEDIUM]` | one strip: flat silhouettes, relief clay, lit colour, the medium ([medium.md](medium.md)) |
| `put A.. on\|above\|below\|left-of\|right-of\|in\|next-to B` | snaps A to B by their solid parts, inside the frame: B moves along, or A shrinks to 0.8 at most; when that is not enough, or A already stands there, nothing changes and the line says what to do instead |
| `crop X0 Y0 X1 Y1 [--with L]`, `crop off` | that region enlarged to the frame, every layer drawn again sharp; a box of another shape is widened to the frame's and the line says so (and names a tighter crop when the result is nearly the whole frame) |
| `look` / `look L` / `look L1 L2` / `look --zoom X0 Y0 X1 Y1` / `look --text [L]` | the look sheet; one layer alone beside its place; a zoomed region; the numbers |
| `need ..`, `note add ".." [--layer L]`, `note done N` | the checklist; what a look found |
| `undo [N]`, `redo`, `history`, `goto REV`, `compare REV [REV]`, `batch "CMD" "CMD"` | every revision keeps the cell code too |
| `deliver [PATH.png] [--scale 2] [--layers]`, `export NAME.py` | the full-size picture (PSD and ORA with `--layers`); a scene file |

`NAME*` names a cell and its copies (`scale goldfish* 1.2`); colours are `ffcf70`, `#ffcf70` or `gold`.

## The report

```text
rev 2: add --from draft.py (new road, bus, glass, tape)    look: /work/art/bus/look.png (1280 px: picture + each item close up)
cells run: road 0.0s, bus 0.1s, glass 0.0s, tape 0.1s; from cache: 0; 0.5s
layers, back to front:
  bus   x 0.22-0.79  y 0.34-0.78   18.2% of frame  visible 100%
request: 2 of 3 confirmed
  [x] 1 bus: 1, yellow, overturned on its side, wheels up to the sky | on road
  [ ] 2 caution tape: yellow, cordons off the scene | in front of bus -> tape is not in front of bus (...): order tape above bus
  [x] 3 glass fragments: light blue, scattered around | on road
worth checking (suggestions, never blocking):
  - glass covers 0.2% of the frame: scale glass 3.0
NEXT: studio order tape above bus
```

A failing section names its line in the draft, and NEXT reads `edit section "# %% sign" in /work/draft.py (line
46: ...), then: studio add --from /work/draft.py`. When a command covers a requested thing (under 60 % visible)
or pushes it out of the frame, the report names the command and NEXT offers `undo` first. A request for a photo
or a realistic picture flags a section drawn with `line=True` (an outline reads as a drawing), and a thing that
blends into what is behind it gets `spoon blends into the table behind it (contrast 1.1:1): recolor spoon 777571`.
Every report whose picture changed says where the file is, just above the look list or NEXT: `picture file:
/work/bus.png (1920x1080, written now)` (without `--out`: the preview). The bottom of every report is what to read:
open items, the look list, `worth checking` lines, NEXT. Editing
`layers/L.py` directly works too; the next `add --from` says which version wins.

## Looks

When nothing measured is open (no failing section, no open item, no open note), NEXT asks you to look at the
sheet: the report prints every item with its full inventory line and any measured doubt beside it, and the first
line names `look.png`, the picture followed by each item close up in the picture, numbered like the list:

```text
look list ([x] = what can be measured holds; by eye: does the picture show each line at a glance?):
  [x] 1 bus: 1, yellow, overturned on its side, wheels up to the sky, windows shattered | on road
  [x] 3 glass fragments: 12, light blue, scattered around the bus | on road  <- worth checking: glass covers 0.2% of the frame: scale glass 3.0
NEXT: look 1 of 3: open /work/art/bus/look.png (the picture, then every item close up); a line above that does not show at a glance: edit its section in /work/draft.py, then studio add --from /work/draft.py (or one command: move, scale, order, crop ...); every line shows: finished - the picture is /work/bus.png
```

A line that does not show: fix it, and the next clean report asks for look 2. Every line shows: you are done.
Three looks by default (`set scene reviews=N`), one per change of the picture; after the last the picture is
finished for good: a clean report says `NEXT: finished: the picture is /work/bus.png (1920x1080, up to date; open it
before you tell the user)`, also after later changes, and only a failing section, an open item or an open note
brings work back. Without `--out` the finish is `studio deliver PATH.png`. To keep a weak point for later, `note add
"3 glass fragments too small to see"`: a note that starts with an item number is tied to that item's layer, the
item stays open, and NEXT walks through the notes until each is `note done K`.

## Recipe: exact counts and places

Draw one apple, then lay out the rest; `put` places by the solid parts.

```python
# %% table
# table: wooden
t = tabletop(cv, top=0.55, front=0.95, horizon=0.4, name="table")

# %% apple
# apple: 3, red | on table
ball(cv, t.place(0.3, 0.5), 0.07, base="#d63c2f")

# %% cup
# cup: blue | right of apple
cylinder(cv, t.place(0.75, 0.5), 0.07, 0.16, base="#3a6fd8", handle=True, name="cup")
```

```bash
studio copy apple --total 3 --row 0.1 0.5 0.55 0.75       # apple, apple2, apple3, apart
studio put cup right-of apple3
```

## Recipe: a finish over everything

A pass section works on the picture below it and keeps every layer editable:

```python
# %% finish [pass]
grade(paper(cv), "warm night", 0.6)
cv.vignette(0.3)
```

`set scene grade=golden paper=on` gives the same finish without a section; `hide finish` shows the raw colours.

## Recipe: a medium

`set scene medium=render|flat|oil|watercolour|ink|pixel` delivers the picture in that medium; `new` sets it from the
request ("a watercolour of", "oil painting", "pixel art"). The medium paints after every render; the picture file,
preview and look sheet show it, the checks keep measuring the layers, and changing it runs no section. Give round
shapes `volume=` and hollows `volume=-0.5` so it lights them right: [medium.md](medium.md).

## Recipe: a chart or diagram with marks on top

A chart, dashboard, diagram, floor plan, drawing or UI request is detected from the request (or `--kind
chart`): `new` lists the quoted texts plus one item for the data, and its NEXT runs `tools/new_scene.py` and then
`add base --scene`:

```bash
studio new sales --prompt "quarterly sales as a bar chart, titled 'Sales 2025', stamped 'Draft'"
python <root>/tools/new_scene.py sales_base --prompt "quarterly sales as a bar chart, titled 'Sales 2025'" --dir art/sales
studio add base --scene art/sales/sales_base.py
studio add stamp --text "Draft" --box 0.72 0.05 0.95 0.14 --color d63c2f --bold
```
