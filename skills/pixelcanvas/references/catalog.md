# Catalogue: the route and the call for anything (`tools/kit.py`, `pixelcanvas.catalog`)

One search box over every kit — pictograms, plants, effects, materials, 3D clay characters and prefabs, backdrops,
palettes, icons, UI components, diagram shapes, chart kinds, text effects, layout templates and scene kinds. Give
it the whole request in plain words (English, Traditional Chinese or Japanese). A picture (people, animals, a scene,
poster art, stickers, a named art style) is drawn freehand: `find` answers in a few lines, the start line and one
optional ingredient per subject the library draws. A chart, diagram, UI, drawing or 3D request comes back with the
exact calls that build it, and `new_scene.py --prompt` prints a starter holding the request's data.

```bash
python tools/kit.py find "A storybook picture of a girl and her dog by two fishing boats at the harbour"
```
```
draw it freehand (Sketch, cv.draw · references/illustration.md); starter: python tools/new_scene.py NAME --prompt "..."
  girl               pictograms.pictogram(cv, "child", at=(0.5, 0.5), size=0.3, color="#1d2633")    # optional pictogram
  dog                pictograms.pictogram(cv, "dog", at=(0.5, 0.5), size=0.3, color="#1d2633")    # optional pictogram
  fishing boats x2   pictograms.pictogram(cv, "fishing boat", ..., style="shaded", shadow=True, anchor="bm")    # optional
-> suggested kind: python tools/new_scene.py NAME --kind illustration (main subject: girl)
```

How a request is routed:

| the request asks for | kind | what `find` prints |
|---|---|---|
| people, animals, a scene, a character, poster art, a sticker sheet, a named style (ukiyo-e, pixel art, watercolour, paper cut, storybook, flat vector, comic, anime ...) | `illustration` | the start line, one optional ingredient per subject the library draws (pictogram, flora, elements effect, texture; else clay or solid as a 3D alternative) |
| 3D, clay, toy, isometric, a render, a product shot | `characters`, `pets`, `scene3d`, `isometric`, `product` | a clay / solid call per subject |
| a chart, dashboard, infographic, diagram, figure, UI, icons, logo, engineering drawing, floor plan, a text poster, card, banner, menu, a comic strip with panels | its own kind | the kit's call with the request's text and data |

`find --kits` prints the kit answer for a picture too: one call per subject to paste, colours and counts attached
(`x2` loops over positions), stand-ins and three-line shape recipes for subjects no kit names, and the kind of the
kit table in SKILL.md (`res.kit_kind`). Quoted strings, dates, times, prices, URLs and capitalised names are listed
under `text to draw:` and never searched. Every printed call runs as printed, given a canvas `cv` and, for `sc.`
lines, a solid Scene `sc = solid.Scene()`; a call on a kit object (`d.bus(...)`) comes with the lines that build it.

## The starter: `new_scene.py --prompt`

`python <root>/tools/new_scene.py NAME --prompt "<request>" --size WxH --dir ./art` stores the request in `PROMPT`,
picks the kind when `--kind` is absent, takes the size from `--size` or the request (square, story, A4, YouTube
thumbnail, README banner ...) and prints the file it wrote, numbered (plan, tunables and the request's blocks for a
file over 120 lines; `--quiet` prints the summary only). Without `--dir` a project folder gets `./art/`. Starters are
ASCII files; text the request carries (a CJK title) is kept as real characters.

- **illustration**: the plan docstring gets the request's art direction (mood words and time of day, palette words,
  the named style, the subjects, the exact text); `MOOD`, `STYLE` and `TITLE` are set; the `# >>> subjects` block
  lists `# draw: <subject>` for each subject with its optional ingredient call as a comment. The starter renders
  as it is; you replace its example subject and prop with the picture you designed, one `log()` step per subject.
- **format kinds** get the request as data: diagram nodes / steps / participants / milestones, chart type,
  categories and series, KPI tiles, UI screen and app, icon names, drawing sizes and holes, floor-plan rooms,
  wallpaper motif, comic panels, textbook callouts; the values are examples to replace.
- **kit kinds** (`--kind people`, `pets`, `isometric` ...) compose the kits' calls: things stand on the ground line
  at a depth, sized by real height, never overlapping; "an owl in a pine" sits in the tree, "a woman holding an
  umbrella" holds it, "a man on a bench" and "a boy riding a bike" sit on them (`seat="none"`), "a tent beside a
  campfire" stands next to it; "a cat on the armchair" is `.on(armchair, "seat")` in 3D. `--plain` keeps the example.

## More commands
```bash
python tools/kit.py show pictograms.harp  # complete snippet, aliases, keywords, signature, reference page
python tools/kit.py list flora            # every name in one kit, grouped by kind (list: every kit)
python tools/kit.py kinds                 # kinds of things with examples
python tools/kit.py doc chart             # print a kit's reference page
```

```python
from pixelcanvas import catalog                     # the same from Python
res = catalog.find('three oranges and a teapot on a table, a card that says "Sunday"')
print(res.text())                                   # what kit.py find prints (res.text(kits=True): --kits)
print(res.kind, res.kit_kind)                       # the route, and the kind of the kit table
print(res.content)                                  # ['Sunday']: the text to draw
best = [e.call for e in res.best()]                 # top call per subject
kind = catalog.suggest_kind("an engineering drawing of a flange")      # "drawing"
title, lines, register = catalog.recipe("gramophone", "poster", silhouette=True)
oaks = catalog.entries(kit="flora", kind="plant")   # Entry(kit, name, kind, call, summary, keywords, ref)
```

`find --json` adds `text_to_draw`, `main`, `kind`, `kit_kind`, `add_kits`, `new_scene` and, for a subject without an
entry, `fallback` with `standins`, `nearest`, `title`, `lines`, `register` and `ref`. Kits added to the package join
the catalogue on their own.
