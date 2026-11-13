# Review checklist + symptom → fix

## The `--check` report: its last lines say what to fix

```bash
python tools/render.py scenes/still_life.py --check --out out/still_life.png    # any scene file, any --out path
```

One render at the scene's `DEFAULT_SIZE` (`--scale 0.5` for a draft), exactly that PNG written (folders are
created), the preview and the check sheet kept in `./art/outputs/`. The report ends with the absolute path of the
PNG and then the **FIX block**, at most 12 lines, so `| tail -15` always shows all of it.

A request that spells out things to check (three or more things, a count, a colour, a place, quoted text or steps)
is a **checked request**: the block lists every requested item the report cannot confirm, in the request's order,
one line each, and the picture critique stays above it as `[info]`. The line before the picture path reminds you to
look at each named item yourself:

```
the report confirms places, counts, sizes and colours, not that a shape reads as the thing: look at each named item in the picture and ask whether a stranger would name it
picture: /home/me/out/cafe.png
FIX: 6 requested items unconfirmed - fix these, render once more, look
[warn] bicycle: cut off at the left edge - move it inside the frame
[warn] cup: 2 drawn, the request says 3 - draw all 3
[warn] cups: 0.24 % of the canvas - draw it at least 3 % or it will not be seen
[warn] cat floats 2 % of the height above the table, the request says on - set the cat on the table: its bottom edge on the table's top, overlapping it sideways
[warn] cat: 19 % covered by sign - move one so both are whole
[warn] the request names a lamp: nothing recorded answers to it - draw it and pass name="lamp"
```

From the second render on the header reads `FIX: N requested items still unconfirmed - fix these and render once
more`; when all of them hold it reads `FIX: every requested item is confirmed - deliver`. Item lines: a thing cut by
the frame, covered by another named thing it has no relation to, under 3 % of the canvas, crowded out by one thing
over a third of the frame, a count, a colour (a colour on a fifth of the thing passes as "partly"), a place, quoted
text missing or misplaced, a named thing nothing records (draw it and pass `name=`). States stay with their noun:
"cups, one of them broken: drawn as "cup" - show the break".

Any other request gets `FIX (worst first): 0 fail, 2 warn - change these, render once more, look`: `[fail]` before
`[warn]`, the request's things first, then text, objects and the picture critique (`[warn] sparse_composition: ...`).
When nothing is wrong the block is one line: `FIX: nothing flagged - look at the picture once, then deliver`. Open
the PNG, make the changes in one edit, render once more. `--json` carries the same lines, in the same order, as
`fix`, and `checked` says which block it is.

What feeds the block, from `PROMPT` and what the scene recorded:

| Check | Fires when |
|---|---|
| size | a thing the request names covers under 0.5 % of the canvas and spans under a tenth of its short side ("lantern: 0.14 % of the canvas - draw it at least 3 % or it will not be seen"), or under half of it is visible |
| count | "three apples" and two recorded apples; "a green bench" and two benches |
| colour | the colour word against the colour painted inside the object's mask ("cup: painted green, the request says blue"); each counted instance; two-colour things pass when both colours are there |
| place | left of / right of / above / below / under; **on** = above with contact; **in** = inside the other's outline; **next to** / beside = closer than its own width; corners and sides of the frame |
| text | a quoted string not drawn (fail); letters under 3 % of the canvas height; "'ORCHARD' at the bottom" with its centre outside the bottom third |
| style | a named style (watercolour, sticker ...) never applied |

## The sections above the block

| Section | What it is | Trust it for |
|---|---|---|
| `WARNINGS` | image critique: exposure, tonal range, composition, banding, noise; on a styled picture, measured inside the style's paper border, with the style's own look as info | how the pixels behave |
| `TEXT` | `text.audit_text` on every `cv.text`, `Sketch.label` and the labels inside diagram, chart, UI and drafting drawings: clipped, too small, low contrast, overlapping, missing glyphs; each quoted string of the request with `drawn` / `missing`; the labels in reading order | every word you drew |
| `OBJECTS` | what was drawn, where (from `cv.manifest`): name, kind, centre, size, area %, visible %, clipped %, hidden_by, then relations ("cat is left of dog", "title is at the top centre of the canvas"). Repeated names merge; a picture of text only prints a reading order | positions, counts, what hides what |
| `CONNECTIONS` | diagrams only: every edge as `from → to (label)`, dashed as `⇢`; unconnected nodes, duplicate edges | "is every link there" |
| `PROMPT` | only with `PROMPT`: the issues (their change is in FIX), then `COVERAGE`: each thing the request names and what answers to it, with one info line per state or material ("teapot (cracked): drawn as "teapot" - show the crack; the report cannot see states"); then `AS REQUESTED`: count, colour, place and text, ✓ or ✗ | "did I draw what was asked" |

COVERAGE lists things only: adjectives, verbs, size words, quantities and positions ("small", "piece", "made",
"below") are never subjects; words about the look, the text content, light and shadow, the setting as a whole and
proper names are never "not recorded"; arrows and labels are never subjects. A worn thing answers to its wearer
("jacket ✓ person chef (worn)"), a part to its whole ("books ✓ bookshelf"). What hides what is read from the final
picture; a backdrop, footer band or device frame is never the cut subject; a tiled floor or a grid is never a horizon.
Counts skip sizes, times and ratios ("120 x 80 x 40 mm", "1:75", "21:00"). Every issue in `--json` is `{rule,
severity, message, where, value, threshold, fix}`; text you drew prints as written, in UTF-8.

**Text, counts, colours and positions come from the report: they are exact.** Kits and every call given `name=`
record what they draw, so "three red apples to the left of the cup" is checked in code. A vision model (yours or
`look.py ask`) is for what only eyes answer: "does this read as a fox?", "is the mood calm?".

### Styles

A style is never failed for its own look: what it does on purpose (white paper, flat areas, grain) is an info line
"expected for STYLE". The one style fail is `mostly_blank`: the style's options left a picture that is almost one
colour although the picture before the style was not.

### Tell the checker what you are making

```python
PROMPT = 'A poster titled "Night Tram" with three paper lanterns, watercolor style'  # the request, verbatim
KIND = "design"          # "painting" | "design" | "diagram" | "drawing"; leave it out to infer
```

- `PROMPT`: every quoted string (`"…"`, `'…'`, curly quotes, CJK corner brackets) must appear in the drawn text
  (case and spaces ignored); a style word (`stylize.pick`) asks for `stylize.apply(cv, name)`; "three lanterns"
  is compared with the objects named lantern. Try one without editing the file: `--prompt "..."`.
- `KIND`: diagrams and drawings skip the painterly and photo rules; designs (posters, UI screens, icon sheets,
  dashboards) skip the photo rules (scattered highlights, JPEG noise, narrow range, blown whites) and get the rest
  as `info`. Words such as `"dashboard"`, `"poster"`, `"ui"`, `"floor_plan"` map to their kind; a misspelt KIND
  stops the render with the closest kinds.

### Name what you draw by hand

Kit calls register themselves. For hand-made shapes, one line after painting keeps the report complete:

```python
from pixelcanvas import Canvas, manifest
cv = Canvas(640, 360, "#dfe7ef")
X, Y = cv.xy()
cup = ((X - 220) ** 2 / 60 ** 2 + (Y - 220) ** 2 / 80 ** 2 < 1).astype("float32")
cv.paint("#c0563c", cup)
manifest.note(cv, "object", "mug", mask=cup)                 # or bbox=(x0, y0, x1, y1) in px / fractions
manifest.note(cv, "object", "bottle", bbox=(0.45, 0.2, 0.55, 0.85))
rows = manifest.summary(cv)                                  # name, centre, size, visible %, clipped %, hidden_by
print(manifest.relations(cv))                                # ['mug is left of bottle']
issues = manifest.check(cv, prompt="two mugs and a bottle")  # -> prompt_count warning: 1 mug drawn
```

Three keys tell the report what an entry is, and a note of something a kit already recorded (same place, same or
alias name) renames that entry instead of adding a second one:

```python
from pixelcanvas import Canvas, manifest
cv = Canvas(640, 360)
manifest.note(cv, "shape", "backdrop", bbox=(0, 0, 1, 1), backdrop=True)      # never the subject, never cropped
manifest.note(cv, "chart", "month", bbox=(0.1, 0.2, 0.9, 0.8), count=12)      # "12 months" is satisfied
manifest.note(cv, "object", "glass panel", bbox=(0.2, 0.1, 0.5, 0.9), see_through=True)   # hides nothing
manifest.note(cv, "style", "sticker")                         # a look drawn by hand counts as the named style
print(manifest.check(cv, prompt="a sticker of a revenue line for the last 12 months"))     # []
```

## Animation

Scenes are parametric, so motion is one flag away. Give the scene an UPPER_CASE constant (`T = 0.0`) and sweep it:

```bash
python tools/render.py anim_campfire --animate T 0 1 --loop            # outputs/anim_campfire.gif + _frames.jpg
python tools/render.py SCENE --animate "SUN_X:0.2:0.8,SUN_Y:0.7:0.3" --frames 36 --fps 15 --bounce --ease inout
python tools/look.py rounds SCENE --gif                                  # the saved rounds as a "how it improved" GIF
```

One render per frame (`--frames 24`, `--fps 12`, default `--scale 0.5`); `--format gif|webp|apng`, `--out path`,
`--hold N` repeats the last frame, `--bounce` appends the frames reversed, `--dither` for gradients in a GIF, `--json`
lists the outputs. For a loop, drive phases with `sin / cos(2*pi*T)` and pass `--loop` (the frame at END equals START
and is left out). Open `<scene>_frames.jpg`: eight labelled frames, enough to see whether neighbouring frames change
smoothly and whether the last one leads back into the first. The file size is printed (a warning above ~8 MB: lower
`--scale`, fewer frames, or `--format webp`).

## How to "look"

| What you can do | How |
|---|---|
| Can read images | `--check --out PATH.png` → open that PNG and work through the FIX block; `outputs/NAME_check.jpg` and `look.py zoom` for details |
| Can't read images | the report is enough for text, objects and positions; `look.py report <png>` for tones and colours; `look.py ask` for "does it read as …" |
| Compare changes | `look.py sheet v1.png v2.png` / `compare A B` (side by side); `diff A B` (numbers + heat map) |

### Borrowed eyes: `look.py ask`

`python tools/look.py ask IMG "does this read as a fox sitting in snow?"` sends the image (≤ 1280 px) and the
question to a vision model; `look.py describe IMG` asks for objects with positions and colours, any text, the
style and visible flaws. `--json` for one JSON line. Configure once:

| `PIXELCANVAS_VLM` | Model |
|---|---|
| `ollama:qwen3.5:9b` | local Ollama at `http://localhost:11434` (`ollama:<model>@http://host:port` for another host) |
| `openai:<model>@<base_url>` | any OpenAI-compatible endpoint; key in `PIXELCANVAS_VLM_KEY` |

Ask yes / no or "which of A, B" questions about the whole picture. Take text, counts and positions from the report.

### Protocol for vision models

1. **First impression, then the numbers.** Look at the preview cold and write one sentence: what is the focal point,
   what feels wrong. Then read the `--check` report — it catches what eyes forgive (clipping, hidden objects, a
   misspelled title, a missing style).
2. **Positions → the OBJECTS table** (or `look.py guides IMG` for anything not recorded). Fix by changing numbers.
3. **Details → `look.py zoom IMG u0 v0 u1 v1`.** Zoom into the focal point, silhouette edges and text: previews
   hide aliasing, banding, seams, tofu boxes and stray pixels.
4. **Values → `look.py values IMG`.** The 4-level map should still read: a few big masses, the focal point at the
   extreme value. If two layers merge into one grey, change their value, not hue.
5. **Edges → `look.py edges IMG`.** The strongest edges should surround the focal point.
6. **Where did it break → `render.py --stages`**; **choosing a value → `--sweep NAME v1 v2 v3`** (or `--seeds`);
   **choosing a style → `--styles watercolor,ink_wash,gouache`**, then `--style NAME` or `stylize.apply`.
7. **Colours → `look.py probe IMG U,V ...`** for exact hex / luma / hue.

Be honest in the write-up: list what is actually wrong, with coordinates ("seam at (0.41, 0.66)").

### Reading `look.py report`

- **Blown (>0.98)**: > 3% is too bright (sun glow too big, sky too bright); > 0.3% around the light source is normal.
- **Crushed (<0.02)**: > 5% → foreground too dark; night scenes up to 15% if it is silhouettes, not a black mass.
- **Contrast (std)**: daytime / dusk landscapes 0.18~0.28; night scenes and still lifes 0.06~0.14 — then judge the
  focal cell's contrast and p99. Daytime landscape < 0.12 → too grey; anything > 0.32 → too harsh.
- **Brightest 0.5% centroid**: should sit near your light source; widely spread → too many bright things.
- **3x3 grid**: the distance (middle row) brighter and lower-contrast than the foreground; rows within 0.05 →
  layers do not separate. **Texture** column: sky `flat` or `fine`; rock / forest `fine` or `coarse`.
- **Dominant colours**: top 2 > 70% → monotonous; check the accents you expect are there.
- **Hue**: 0° red, 30° orange, 60° yellow, 120° green, 180° cyan, 240° blue, 280° violet, 320° magenta; s < 0.1 is grey.
- **Vertical profile / boundaries**: the strongest transitions should sit at your horizon and ridge heights;
  Δ < 0.05 → that boundary is too weak.
- **Local checks**: `look.py crop IMG x0 y0 x1 y1 -o tmp.jpg` then `report tmp.jpg`.

## Checklist

1. **Exposure**: no large blown whites or crushed blacks; the sun is a bright point + glow (paintings).
2. **Atmospheric perspective**: farther = paler, closer to the haze colour, lower contrast; layers separate.
3. **Consistent lighting**: lit sides face one way; shadows point away; low light → rim light on silhouettes.
4. **Texture scale**: too small → fur / noise; too large → plastic.
5. **Edges**: anti-aliased silhouettes; fluffy cloud edges; contact shadows under objects.
6. **Water**: reflections mirrored about the eye level; glints only below the light.
7. **Repetition**: neighbours never share noise / seed; scattered objects vary in size and spacing.
8. **Composition**: focal point near a third; nothing important touching the frame (`object_at_edge`).
9. **Resolution**: draft and full size look the same (pixel constants × `S`).
10. **Text & design**: TEXT and PROMPT sections clean (exact spelling, no tofu, contrast, margins); one clear
    focal element; still readable in the `_check.jpg` thumbnail.
11. **The request**: the FIX block is clean, and every state COVERAGE lists (cracked, wet, rusty) shows at a glance.

## Symptom → fix

| Symptom | Likely cause | Fix |
|---|---|---|
| `prompt_text_missing` | text typed differently or never drawn | copy the quoted string exactly into `cv.text(...)` |
| `prompt_style_missing` | the request names a medium | preview `--style NAME`, then `stylize.apply(cv, name)` last; a look drawn by hand: `manifest.note(cv, "style", name)` |
| `prompt_count` | too many / too few of an object | draw exactly that many; give each its own `at=`; a series of points: `count=` on its entry |
| `object_hidden` | drawn before something opaque on top of it | draw it later or move one of them; glass or water on top: `see_through=True` |
| `object_off_canvas` / `object_at_edge` | `at=` too close to the frame | move inward, keep a 4% margin |
| `object_tiny` / `prompt_small` | size guessed too small | at least 3 % of the canvas for a named thing: larger `height=` / `size=` / radius |
| `prompt_colour` | painted another colour, or the highlight / shade took over | paint the solid areas the requested colour; keep highlights and shadow thin |
| `prompt_relation` | positions swapped; "on" without contact | move one of them; for "on", put its bottom edge on the other's top |
| `prompt_text_small` / `prompt_text_place` | the quoted text set too small or elsewhere | letters at least 4 % of the height; move it into the named third |
| Blown out, sun is a white blob | glow too strong / wide | lower glow strength, smaller sigma; `finish(knee=0.75)` |
| `mostly_blank` (fail) | the style's options removed the content | go back to the style's defaults; `stylize.options(NAME)` lists each option's default, range and meaning |
| `picture_area` / "... (expected for STYLE)" (info) | the style's paper border or its own signature | nothing to fix |
| `style_option_clamped` (info) | an option slightly outside its range was clamped | use a value inside the range from `stylize.options(NAME)` |
| Diagram sits small in the middle | a fixed `direction=` against the canvas shape | drop `direction=` (`"auto"` fills the canvas); tiers: `lanes="rows"` |
| Diagram text too small | many nodes on a small canvas | keep `font_size` ≥ 18 (px at 1080p, default 22); split the figure or use a larger canvas |
| label "sticks out of its own box" | label longer than its node | shorten it, add a line break, or widen the node (`width=`) |
| `node_unconnected` | a node with no edge | add the missing `d.edge(...)`, or drop the node |
| `edge_duplicate` | the same edge drawn twice | keep one `d.edge(...)` per connection (a list fans out: `d.edge(a, [b, c])`) |
| `edge_label_on_node` | an edge label covers a node | shorter label, `label_pos=0.3` / `0.7`, or `spacing=1.3` |
| Requested object missing from OBJECTS; COVERAGE says "not recorded" | painted by hand, never noted | `manifest.note(cv, "object", "jetty", mask=m)` after painting it |
| A hand-built object looks crude (a lumpy horn, a box-like windmill) | built from primitives when a pictogram exists | `pictograms.pictogram(cv, name, style="silhouette" \| "shaded")`, or fill `pictograms.mask(name, h, w, box)` with a texture |
| Grey and hazy | too much fog, foreground not dark | near fog → 0~0.1; darker foreground; `contrast(0.12)` |
| Layers don't separate | similar colours / fog amounts | spread fog values (0.8 / 0.55 / 0.3 / 0.1); add `mist` between |
| Clouds look like marble / cotton candy | warp too strong / softness too high | warp 0.35 → 0.2; lower softness; add a fine octave |
| Foreground flat, plastic | no mid / small-scale texture | relief fBm (±10%), `texture.fill`, small rocks / grass |
| Jagged silhouettes / text | hard thresholds | `clamp01(Y - ridge + 0.5)`, `sdf.fill(d)`, `cv.text` |
| Objects float | shadow detached from the contact point | narrow dark contact shadow under the object |
| Contradictory light | each object has its own light vector | one `light_dir` at the top of the file |
| Repetitive, tiled look | same noise / seed reused | different seed per layer; vary density |
| Draft and full size differ | pixel constants not × S | `x * S` / `sc.px(x)` everywhere |
| Tofu boxes instead of letters | font lacks the glyphs | omit `font=` (auto CJK) or `python tools/fonts.py --sheet` |
| Text hard to read | low value contrast behind it | `plate="scrim"`, outline or shadow; check `look.py values` |
| Design looks random | no grid, too many sizes / fonts | one margin; ≤ 2 fonts, ≤ 3 sizes; typography.md §7 |
