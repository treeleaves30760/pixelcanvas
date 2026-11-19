# Changelog

## 0.7.0 (2026-11-20)

People in a scene are posed on a 3D skeleton first and drawn in 2D over it, and the Claude Code plugin builds a
new picture from the coarsest decision down with four subagents.

### Posed figures in 3D
- `camera` (horizon, eye height, lens, tilt), `block` (box, stairs, cylinder, plane, railing, sizes in metres)
  and `prop` (stick, sword, umbrella, bag, ball, book, cup, held in a hand).
- `pose <who> --preset <name>`: 36 presets (run, sprint, jump, sit-ledge, sit-stairs, kneel, look-back,
  lean-railing, reach-viewer, ...), joint edits in degrees with anatomical limits, `--on <block>`, hands and feet
  that reach a block or a point, `--hand-shape-r fist|open|relaxed|point|grip|peace|cup`, heads or height, yaw.
- `dress <who> <pieces>` (18 garments with colour, length, fit, sleeve, style, trim), `hair3d <who> --style`
  (long, medium, bob, short, spiky, ponytail, twintails, braid, bun; bangs, bows, wind), `face <who>
  --expression` (10 expressions) on the 3D head.

### Drawn in 2D over a guide
- `render` turns the 3D scene into a faded guide that is never exported; a re-pose, re-dress or camera change
  updates it by itself.
- `draw all|<who>|scene [--part hair|face|arm-r|...]` lays anime line art over the guide: a body template by head
  count that keeps the pose's foreshortening, clothes with hems, cuffs and pleats, hair in clumps with pointed
  tips, hands per hand shape, a silhouette line heavier than inner lines, an id on every stroke
  (`line redo`, `line weight`, `line delete`).
- `flat auto` fills each drawn part; `shade all` lays hard-edged cel shadows from the light and a compact contact
  shadow under each figure.
- On a drawn picture `surface auto` leaves cel areas clean, `polish` keeps line colours and edges, and `fx blur`
  blurs each area only with itself.
- `export` of a blockout writes the 3D view; `pose --show joints` is read-only.

### Claude Code plugin
- `/pixelcanvas:draw <request>` and the subagents `pixelcanvas-director` (the brief), `pixelcanvas-blocker` (pose
  and guide), `pixelcanvas-painter` (drawing and finish) and `pixelcanvas-critic` (checks with a command per fix).

## 0.6.0 (2026-11-18)

pixelcanvas is now a command-line paint program an agent drives stage by stage. The old tool, where the
model wrote Python that drew vector art with kits of subject generators, is gone: the studio, the kits (flora,
clay, pictograms, figure, creature, solid, diagram, chart, ui, drafting and the rest), the art media, the JSON spec
mode, the scenes and templates, the `tools/` scripts, the MCP server, the `pixelcanvas` and `pixelcanvas-lite`
skills and the gallery. One name everywhere: the package, the command, the skill and the project folder
(`<name>.pixelcanvas/`) are all `pixelcanvas`, and every command is `pixelcanvas <command>`.

### The paint program

- **A picture is a project folder** (`pic.pixelcanvas/`): the reference, one PNG per layer, regions, an undo history and a
  journal of every command. `pixelcanvas undo` / `redo` restore it byte for byte.
- **Stages like a painter**: plan, sketch, line, flat, shade, detail, finish for cel, ink and poster pictures; plan,
  sketch, block-in, forms, detail, finish for paintings and photographs. Each stage paints on its own layer and has
  a gate pixelcanvas measures per region; `pixelcanvas stage done` moves on when it passes or when the agent gives a reason.
- **A report after every command**: scores `before -> after`, warnings, the worst spots, a `VIEW:` picture of the
  canvas, a `VIEW-REF:` picture of the reference and one `NEXT:` line. `pixelcanvas todo --all` prints the work list.
- **Three modes**: `trace` (stage commands follow the reference; the agent plans regions, orders the work, checks and
  refines), `hand` (the agent places every mark, the reference beside it) and `free` (a text request, no reference).
- **Redraw it, but ...**: `pixelcanvas new pic --ref reference.jpg --request "..."` asks, after the plan, for each change the
  request names (`pixelcanvas change add ...`, or `pixelcanvas change none` for a plain copy) and adds a change stage that makes
  them. A change edits the target, the picture the stages trace and score against; the original stays as
  `orig.png`. `pixelcanvas look --show changes` shows original and target side by side, `pixelcanvas change list` and `drop`
  manage the list, and a change made after tracing has begun marks only the affected regions and stages stale;
  `pixelcanvas undo` restores target and layers exactly. A project without a request, or with `change none`, runs the plain
  redraw unchanged.
  - Recolour: `pixelcanvas recolor hair --to "#8fdcb4"` moves a region to a new colour and keeps its shading, lines and
    highlights; the report gives hue before -> after, shading kept and stray patches of the old hue, and
    `pixelcanvas region refine hair --by hue|lines` snaps the region to a hue band or to the areas its lines enclose.
  - Poster: `pixelcanvas canvas --size 900x1200 --place x0,y0,x1,y1` places the reference in a new canvas (the regions move
    with it), `pixelcanvas palette map --to "dark,...,light" --accent "jacket=#e8644a"` maps the target to a few colours by
    value, `pixelcanvas graphic disc|rect|ring|frame <box> --color` lays a shape on the ground behind the figure, and
    `pixelcanvas change add text ...` letters the title in the finish stage. When the target no longer reads as the project's
    style, the change gate asks for `pixelcanvas style poster`.
  - New setting: `pixelcanvas cutout figure --regions a,b,c` cuts the figure out (it reports ground left inside, figure
    lost and edge width, and boxes the three weakest edge spots), `pixelcanvas background --from lake.pixelcanvas | picture.png |
    --gradient "#a,#b" | --color "#hex"` puts it in front of a new background, `pixelcanvas light` tints the figure toward
    the background's light and adds a rim light, and `pixelcanvas shadow figure` lays a contact shadow on the ground.
  - List options (`--minus`, `--plus`, `--regions`) accept repeats and comma lists.
- **Detail levels**: `pixelcanvas detail all`, then `pixelcanvas detail <region> --level 1` region by region in the ranked order,
  `pixelcanvas dot all`, `pixelcanvas sharpen`. The strokes follow the reference and take their colours from it along the stroke; every
  report prints the share retouched.
- **Drawing from a text request** (`pixelcanvas new pic --mode free --request "..."`): plan subjects and parts with boxes,
  then the same stages with construction tools (`sketch figure`, `face`, `line`, `flat auto`, `shade`, `eye`,
  `hair --flow`, `fold`, `text`, `band`, `surface`). `pixelcanvas polish` is the finish stage's default pass on the fx layer:
  strays painted out, streaks smoothed inside each colour area, line art coloured from the fills it borders, far things
  lighter and cooler under a sky, lights graded toward the light's colour, soft edges for painterly styles; each part
  has a `--no-<part>` switch, and the report gives numbers per part, a `VIEW:` and one `NEXT:`. The todo list is the
  same after a rebuild in the middle of a stage, brush seeds follow the command and its target, and sky, sea and cliffs
  are colour bands in the flat stage.
- **Projects survive being copied or moved**: a missing working-size or delivery-size reference file is rebuilt from
  the other.
- **Hand tools** for what a stage misses: `stroke` (with `--color pick --snap`), `fill`, `shape`, `gradient`,
  `erase`, `smooth`, `adjust`.
- **Export**: `pixelcanvas export final.png --psd --ora --timelapse film.gif --stages dir`.
- **Results**: on a private set of 93 reference pictures the redraw path reaches a judge mean of 9.42 of 10, with
  88 of 93 rated the same quality as the reference. In the redraw path the strokes follow the reference and take
  their colours from it along the stroke, and no picture is retouched pixel by pixel. A Qwen 3.8 Flash class model
  driving the skill in Claude Code or opencode finishes a redraw in about 19-45 minutes and 44-86 commands; its
  sessions now cover 23 anime pictures, with a judge mean of 9.52 and 21 of 23 rated the same quality as the
  reference (the scripted path on the same pictures: 9.48, 22 of 23).

### Install and packaging

- One console script, `pixelcanvas`; `python -m pixelcanvas` runs the same. `pixelcanvas install-skill` copies the skill into
  `~/.claude/skills/pixelcanvas`, `--opencode` for `~/.config/opencode/skills/pixelcanvas`, `--to DIR` for any skills folder;
  `--force` replaces an existing one).
- Dependencies: numpy, Pillow, opencv-python-headless, scikit-image, scipy.
- The wheel carries the skill as `pixelcanvas/skill/SKILL.md`.
- Removed: `pixelcanvas doctor`, `pixelcanvas render` and every other subcommand of the old command, the `fast` and `pixelcanvas` extras, `tools/install_skill.py` and the `Makefile`.

## 0.5.0 (2026-11-14)

A picture is now a living state your agent edits with tools. Until 0.4 the agent wrote a whole scene in one go and
fixed it once or twice; in 0.5 it works in **`pixelcanvas studio`**: one small drawing cell per object, your request
as a checklist, review notes and a revision history, and every command changes one thing, re-renders only that,
checks the picture against the request and prints one `NEXT:` line. Draft in one write, look, write down the weak
points, fix each with the smallest tool, deliver. A later "make the fox bigger" is one more command on the same
picture.

![First draft and delivered picture](https://raw.githubusercontent.com/treeleaves30760/pixelcanvas/v0.5.0/gallery/studio/campfire_fox_before_after.jpg)

### What you can now do

- **Draw in the studio**: `studio new NAME --prompt "REQUEST" --size WxH --out PATH.png` reads the request into a
  numbered checklist (each thing with its count and colour, each place, each quoted text) and writes `draft.py`;
  `studio add --from draft.py` turns its `# %% name` sections into layers and renders them all.
- **Follow one `NEXT:` line**: a failing cell first, then each open item with the command that settles it
  (`copy apple --total 3`, `put apple* in bowl`), then up to three review rounds, then `deliver`.
- **Review by looking**: `studio look` writes one sheet (the picture with a 0.1 grid and layer boxes, every object
  alone); `look L`, `look --zoom X0 Y0 X1 Y1`, and `look --text` for models without vision. `note add "..." "..."`
  records every weak point in one call and NEXT walks through them.
- **Fix with the smallest tool**: `move`, `scale`, `rotate`, `flip`, `fit`, `order`, `put A on|in|left-of|above.. B`,
  `copy --total N --row | --grid`, `recolor`, `crop`, `set L outline= shadow= glow= tint= blur=`, `set scene light=
  grade= paper=`, or rewrite one cell file and `render`; `batch "CMD" "CMD"` runs several with one report and `NAME*`
  names a cell with its copies.
- **Keep the picture safe**: a failing cell keeps its last good render and the report names the line (and `studio api
  NAME` for a wrong argument); `undo`, `redo`, `goto REV` restore code and parameters; `history` shows the request
  score of every revision; `compare REV REV` puts two side by side.
- **Keep the file current**: the `--out` file is rewritten after every command; follow-up requests run on the same
  state with `studio -p NAME ...`, and `set scene prompt="..."` updates the checklist for a changed request.
- **Real layers**: `deliver --layers` writes PSD and ORA with one layer per object; `deliver --scale 2` renders
  larger; `export NAME.py` writes a scene file for `tools/render.py`.
- **Charts, diagrams, drawings and UI** come in as one layer (`add base --scene FILE.py`) with text and stamps on top.
- **Deliver in a real art medium**: `studio set scene medium=render|flat|oil|watercolour|ink|pixel|off` (dials
  `medium_strength`, `medium_mark`, `medium_palette`, `medium_paper`). The layers become per-pixel buffers (local
  colour recorded apart from shading, a relief per shape, normals, depth, emission), one key light lights them (form
  shadow, cast shadows on the floor, contact occlusion), and the medium paints the result as a process of its own: oil
  strokes along the form, watercolour washes with pooled edges on paper, ink contours and hatching that wraps the form,
  a pixel grid in a small palette. `studio new` reads the medium from the request ("a watercolour of", "oil painting",
  "ink drawing", "pixel art", "3D render"); changing it runs no section again. Medium checks keep
  the drawing's value pattern, the requested colours, the contrast of each thing and legible text. `studio stages`
  writes one strip: flat silhouettes, relief clay, lit colour, the medium. New package `pixelcanvas/medium/`, page
  `references/medium.md`.
- **Three looks, then finished**: after the third look the picture is finished for good; a later change does not ask
  for another look, and only a failing section, an open item or an open note brings work back.
- **Every report names the file**: `picture file: <path> (WxH, written now)`; the installed SKILL.md names the
  reference folder (`references: <abs>/`).
- **Calls and options**: a wrong option reports every unknown option of the section at once with the call's signature;
  bare hex colours (`"3a4048"`) work in Python calls; `set` takes multi-word values (`grade=warm night`).
- **Placing**: `put` never shrinks a thing below 0.8 to satisfy a place; places are judged on the solid part of a layer;
  a 3D scene is one section with its named objects measured one by one.

### Tools

- `pixelcanvas studio` (`tools/studio.py`, package `pixelcanvas/studio/`): 34 verbs (`studio help`), a picture folder
  `./art/NAME/` with `picture.json`, `layers/`, `draft.py`, `preview.png`, `look.png`, a render cache and every
  revision in `history/`.
- `studio api NAME ..` prints any drawing call's exact arguments, first doc lines and an example; `studio api WORD`
  lists every call with that word; `studio api` alone prints the short map of shapes, paint options, solids and kits.
- A cell sees `box("L")` and `anchor("L", "mouth")`: a lantern drawn at the fox's mouth follows the fox when it moves.
- SKILL.md is rewritten around the studio loop and its fix table; `references/studio.md` documents every command.
- Built and tuned with a local model (Qwen 3.8 Flash Next) driving the studio through Claude Code and opencode, and
  with Claude models; `docs/INSTALL.md` shows the `opencode.json` entry that lets a local vision model read the
  `look` sheet.
- The MCP server has four studio tools (15 in all): `studio` runs any command and returns the report with the
  preview image, `studio_draft` takes the whole first draft, `studio_layer` redraws one object, `studio_read` returns
  the draft or one cell. An MCP client works in the studio without a shell.
- `install-skill` records `studio.py` as the skill's tool; the `pixelcanvas` command lists `studio`.

### Fixes

- A call that gets the same argument twice explains whether it was passed by position and keyword, or as both
  `color=` and `colour=`.

## 0.4.0 (2026-11-13)

Your agent now draws almost anything you can describe: illustrations, posters, dashboards, architecture diagrams,
floor plans, app screens, 3D product shots and clay characters. It designs the picture, draws it freehand in
Python with the kits as ingredients, checks it against your request and looks at it before handing it over.

First release on PyPI: `pip install pixelcanvas`, or `uvx pixelcanvas doctor` to try it without installing.

![The 0.4 gallery](https://raw.githubusercontent.com/treeleaves30760/pixelcanvas/v0.4.0/gallery/wall.jpg)

### What you can now draw

- **Freehand illustration**: `illustrate.Sketch` places shapes in fractions of the canvas, in nested local frames,
  with blobs, tapered strokes, shading, highlights, outlines, glow, stickers and washes; `cv.draw()` adds a
  supersampled Pillow layer.
- **Lit solids by default**: a photo, a 3D render or a request with no named style gets lit, shaded objects on a
  receding ground (`Sketch.paint(spec=, shadow="ground")`, planar shading of polygon faces), without outlines.
- **Soft-cel and anime**: `Sketch.cel`, `sheen`, `eye`, `blush`, `ink`, `ramp` and `cel_tone`.
- **Posters, cards and type**: poster and card templates, themes, text effects and auto-fitting type in any
  language; 57 one-call backdrops that keep a calm area for the title, and 60 palettes.
- **Charts and dashboards** (`chart`), **diagrams** with automatic layout (`diagram`: flowcharts, architecture,
  trees, sequences, timelines, tables, callouts) and **UI screens** in device frames (`ui`), all also as SVG.
- **Engineering drawings and floor plans** (`drafting`): projected views, sections, dimensions, title blocks, walls,
  doors and plan symbols, saved as PNG, SVG and DXF.
- **3D** (`solid`): numpy ray-marching with prefabs, materials, studio light, perspective and isometric cameras.
- **Clay characters** (`clay`): faces, 22 hair styles, 13 outfits, 15 expressions and 31 poses; 64 animal species in
  15 poses; 33 props.
- **People and animals**: posed, dressed people with hands and faces (`figure`); 54 species with poses, gaits and
  coats (`creature`).
- **Nature and materials**: trees, flowers and plants (`flora`); fire, smoke, water, weather, skies and light
  (`elements`); 80 lit materials (`texture`); 2.5D objects and rooms with one light (`form`).
- **633 pictograms** in six styles (silhouette, duotone, line, shaded, sticker, cutout), also usable as masks.
- **43 art styles**, from oil and watercolour to ukiyo-e and risograph; `styles pick "REQUEST"` chooses one.
- **Text**: a list separator never starts or ends a line, and CJK lookup picks a font that covers the text.

### Tools

- `render SCENE --check` writes a check sheet and a report: image critique, a text audit, the objects drawn and
  where, and the request checked item by item, ending with a FIX block that names what to change.
- The report names a missing thing only when the request clearly asks for it, counts a thing drawn under another
  name, label or comment as present, and ends with "N requested items unconfirmed" or "every requested item is
  confirmed - deliver".
- **Match a reference image** (#5): `match.py REF.png` prints a style sheet (palette with roles, lines, tone bands,
  shadow shifts, highlights, paste-ready constants); `match.py REF.png DRAFT.png` lists the gaps, largest first;
  `render --check --ref REF.png` adds them as a REFERENCE block.
- `kit find "REQUEST"` searches every kit at once and answers with one call per subject.
- `new NAME --prompt "REQUEST"` starts from the matching one of 30 starter templates, filled in for the request.
- Spec mode: `compose --example poster` writes a JSON spec and `compose SPEC.json --check` draws it, in 16 kinds
  from posters to dashboards and 3D scenes. The new **pixelcanvas-lite** skill is built on it, so small local models
  draw finished pictures too.
- `--layers` (PSD / ORA / PNG), `--svg` and `--dxf` on every render; `tools/build_gallery.py` rebuilds the gallery.

### Packaging

- `pip install pixelcanvas`, with a `pixelcanvas` command for every tool: `kit`, `new`, `render`, `compose`, `look`,
  `match`, `styles`, `layers`, `fonts`, `install-skill`, `doctor`, `mcp`.
- `pixelcanvas install-skill` installs both skills for Claude Code, Codex, Cursor, opencode or any Agent Skills
  tool, for one user or one project; `npx skills add treeleaves30760/pixelcanvas` works as before.
- Claude Code plugin: `/plugin marketplace add treeleaves30760/pixelcanvas`, then `/plugin install
  pixelcanvas@pixelcanvas`.
- `pixelcanvas mcp` runs an MCP server with 11 tools and every reference page as a resource, for any MCP client.
- The skill's bootstrap finds or fetches the runtime by itself.
- CI runs the tests on Ubuntu, Windows and macOS with Python 3.9 and 3.13.

## 0.3.0 (2026-11-11)

Editable layers (PSD / ORA / PNG), media stylization, inspection views for vision models, typography and SDF
shapes; the skill installs with `npx skills add`.
