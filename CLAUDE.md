# CLAUDE.md: developing pixelcanvas

pixelcanvas is an Agent Skill and a Python package that draws finished pictures with numpy + Pillow. Agents read
`skills/pixelcanvas/SKILL.md`, design the picture, draw it freehand (`illustrate.Sketch`, `cv.draw()`) with kits as
ingredients where structure is the hard part, and iterate with `tools/render.py --check` and `tools/look.py`. Small
local models use `skills/pixelcanvas-lite` and `tools/compose.py` (one JSON spec). To draw inside this repo, follow
SKILL.md.

## Project map

```
pixelcanvas/__init__.py   public exports and __version__
pixelcanvas/_args.py      strict options: reject_unknown / unknown_name / check_choice with did-you-mean
pixelcanvas/canvas.py     Canvas: float32 HDR buffer, paint / text / paste / gradients / glow / post, manifest hooks,
                          cv.draw() (supersampled Pillow ImageDraw layer) and cv.cairo() (pycairo layer)
pixelcanvas/illustrate.py freehand drawing: Sketch (shapes in fractions, local frames, blobs, tapered strokes, paint
                          with shade / highlight / volume / line / shadow / glow, pattern, sticker, wash), paper,
                          grade, limit_palette
pixelcanvas/color.py      colour parsing, lerp / mix, luma, clamp01, smoothstep
pixelcanvas/noise.py      Perlin, fBm (plain / ridged / billow), domain warp, 1D noise
pixelcanvas/masks.py      supersampled Pillow masks, polygons, circles, lines, Béziers, feather
pixelcanvas/brush.py      stamps and tapering strokes
pixelcanvas/sdf.py        signed-distance primitives, boolean operators, fill / outline / glow / bevel
pixelcanvas/filters.py    blur, sampling, bloom, light shafts, grading, soft_clip
pixelcanvas/text.py       font lookup (CJK fallback), wrapping, auto-fit, effects, audit_text
pixelcanvas/layout.py     boxes, grids, stacks, poster / card templates, themes
pixelcanvas/vision.py     inspection views (guides, zoom, values, edges, sheets), critique, stage snapshots
pixelcanvas/layers.py     layer recorder: one layer per log() step -> PSD / ORA / PNG stack
pixelcanvas/manifest.py   what was drawn where: note, summary, relations, prompt checks for --check
pixelcanvas/catalog.py    one search index over every kit registry (tools/kit.py)
pixelcanvas/scenery.py    landscape recipes on a Scene: sky, sun, clouds, mountains, rivers, pines, god rays, finish
pixelcanvas/elements.py   fire, smoke, clouds, water, weather, skies, light
pixelcanvas/flora.py      trees, shrubs, flowers, potted plants, grass, rocks
pixelcanvas/texture.py    80 lit materials (wood, stone, fabric, metal, fur ...)
pixelcanvas/form.py       2.5D objects with volume, rooms, one light, shadows
pixelcanvas/solid.py      numpy ray-marching 3D: prefabs, materials, lighting, cameras, isometric
pixelcanvas/clay/         3D clay / toy characters (rig, faces, hair, outfits, 31 poses), 64 animal species, props,
                          decals on solid; a package: character, animal, species, props, registry ...
pixelcanvas/pictograms/   633 pictograms in six styles, also masks: 100 x 100 path data files (PICTOGRAMS = [...],
                          merged automatically), pictogram / mask / drawing / find / names / sheet
pixelcanvas/figure.py     posable dressed people: skeleton, hands, faces, hair, clothes
pixelcanvas/creature.py   animals on a skeleton: species, poses, gaits, coats
pixelcanvas/stylize.py    43 art styles, pick() from a request, apply() on a mask or the whole picture
pixelcanvas/palette.py    named palettes
pixelcanvas/backdrop.py   one-call backgrounds and wallpapers with a calm area for text
pixelcanvas/vector.py     Drawing: shapes, paths, icons, groups -> raster and SVG
pixelcanvas/diagram.py    flowcharts, architecture diagrams, trees, sequences, timelines, tables, callouts
pixelcanvas/chart.py      charts, KPI tiles and dashboards on vector
pixelcanvas/ui.py         declarative UI screens in device frames with auto layout
pixelcanvas/drafting.py   engineering sheets: views, dimensions, sections, plans, SVG and DXF
pixelcanvas/spec.py       spec mode: a picture from a small JSON spec (pcscene), placed by words, drawn by the kits
pixelcanvas/cli.py        the `pixelcanvas` command
pixelcanvas/mcp_server.py MCP server (stdlib only) exposing the tools and reference pages
scenes/                   example scenes (gallery, illustration_*.py freehand) and 30 starter templates
                          _template_<kind>.py (illustration, poster, chart, diagram, ui, drawing, floorplan, ...)
tools/                    render, look, kit, new_scene, compose (JSON specs), layers, styles, fonts, pc (one entry),
                          install_skill (--skill pixelcanvas | pixelcanvas-lite | both), release_check
skills/pixelcanvas/       SKILL.md, scripts/bootstrap.py, references/ (one page per kit + illustration, api, spec ...)
skills/pixelcanvas-lite/  the small-model skill: SKILL.md, scripts/bootstrap.py, tools/compose.py
tests/                    pytest; test_skill_docs.py runs reference code blocks, renders every scene, checks links
.claude-plugin/           plugin.json and marketplace.json
docs/INSTALL.md, AGENTS.md, README.md, README-zh.md, gallery/
```

## Tests

```bash
python -m pytest -q                                   # everything
python -m pytest -q tests/test_flora.py               # one kit
python -m pytest -q tests/test_skill_docs.py tests/test_distribution.py   # docs, links, install, MCP
```

Every scene in `scenes/` (not starting with `_`) is rendered at 192x108 by the tests; every `python` block of
every reference page runs.

## A new kit

1. `pixelcanvas/<kit>.py`, or a package `pixelcanvas/<kit>/__init__.py` when it grows, importable on its own,
   numpy + Pillow only (scipy only as an identical speed-up).
2. A registry: `names(kind=None)`, `find(query)`, `describe(name)`. The catalogue discovers modules and package
   folders and indexes their registries (`python tools/kit.py list` shows them); a kit that draws nameable objects
   gets a handler in `catalog._HANDLERS` (pictograms has `_kit_pictograms`).
3. Strict options: pass keyword options through `pixelcanvas._args` (`reject_unknown`, `unknown_name`,
   `check_choice`) so a misspelling raises an error with a did-you-mean.
4. Manifest hooks: register drawing calls with `manifest.register(globals(), kind, names, arg=...)` or call
   `manifest.note(cv, kind, name, mask=...)`, so OBJECTS and PROMPT in `--check` see what the kit drew.
5. Draw through `cv.paint` / `cv.text` / `cv.paste` so the layer recorder and the manifest keep working.
6. `skills/pixelcanvas/references/<kit>.md`, at most 250 lines: a 5-line quick start, a table of calls, 2–4
   recipes. Every code block runs; `tests/test_<kit>.py` names the page and executes its blocks.
7. A starter template `scenes/_template_<kind>.py`: a docstring whose first line is the kind's description, a plan,
   `PROMPT`, `NAME`, `KIND`, `DEFAULT_SIZE`, UPPER_CASE tunables, `render(width, height, seed, log)`, one `log("step")` per
   object, `post` last, a `# Read skills/pixelcanvas/references/<kit>.md` line. `new_scene.py --list` finds it.
8. Example scenes `scenes/<kit>_<subject>.py` good enough for the gallery; a 960x540 draft renders in seconds.
9. A row in the SKILL.md kit table (For | Kit | Read | Kind | Start with; `catalog.kit_kinds()` reads the Kit and
   Kind columns) and in the index table of `references/api.md`.
10. English only in code, comments, output and docs.

Conventions: x right, y down, `arr[y, x]`; masks float32 (H, W) 0–1; positions as fractions; pixel sizes
× `S = H / 1080`; anti-aliased edges; vectorised numpy, no per-pixel Python loops.

## Documentation rule

This repository is a tool. Docs say what it does and how to use it, with confidence: no disclaimers, no
limitation sections, no hedging, no comparisons that put the tool down, no invented numbers. Show capabilities
through examples that run. SKILL.md stays within 180 lines (about 3,650 tokens): a new sentence replaces a weaker
one, and details go to the reference pages. SKILL.md is scored on public benchmarks against the no-skill agent; a
change to it is kept only when the score holds.

## Release checklist

1. Same version in `pyproject.toml`, `pixelcanvas/__init__.py` (`__version__`), SKILL.md `metadata.version`,
   `.claude-plugin/plugin.json` and `skills/pixelcanvas/scripts/bootstrap.py` (`VERSION`); the CHANGELOG top entry
   and the `ref` default in `.github/workflows/publish.yml` name the new version.
2. `python -m pytest -q` passes.
3. No `zz_review*` or other `zz_*` scratch files in `scenes/`, `outputs/` or the repo root.
4. `python tools/release_check.py` (or `make release-check`) builds the sdist and wheel, runs `twine check
   --strict`, installs the wheel into a clean venv and smoke-tests the command: every line must say PASS before
   tagging.
5. Tag `vX.Y.Z` on the release commit and push the tag; then Actions -> publish with `ref` set to the tag
   (Trusted Publishing, environment `pypi`).
