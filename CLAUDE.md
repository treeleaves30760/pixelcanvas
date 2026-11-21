# CLAUDE.md: developing pixelcanvas

**pixelcanvas** is a command-line paint program an agent drives stage by stage
(plan, sketch, line, flat, shade, detail, finish) on a project folder with state, layers, undo and views. Agents
read `skills/pixelcanvas/SKILL.md` and run `pixelcanvas` commands; every command prints a report (scores `before -> after`,
`VIEW:` / `VIEW-REF:` pictures, one `NEXT:` line). To draw inside this repo, follow SKILL.md.

## Project map

```
pixelcanvas/
  __init__.py                  __version__ only
  cli.py, __main__.py          the `pixelcanvas` command: argument parsing, the command table, reports, NEXT
  cli_draw.py                  commands for drawing without a reference, hooked into the command table
  cli_change.py                "redraw it, but ...": change add / drop / list / none, region refine, recolor, canvas,
                               palette map, graphic, cutout, background, light, shadow
  change.py                    the change list, the target (orig.png kept) and the change gates
  change_ops.py                pure operations on the target: recolour keeping shading, palette map, region refine
  change_scene.py              pure operations for a new setting: cut-out, background, light, contact shadow
  skill_install.py             `pixelcanvas install-skill`: copies the packaged skill into an agent's skills folder
  project.py                   the project folder <name>.pixelcanvas/: state, layers, regions, history / undo / redo, journal
  compose.py, coords.py        layer compositing; 0-1000 coordinates and lenient argument parsing
  view.py, metrics.py          observation renders (VIEW / VIEW-REF); scores per region and the worst spots
  regions.py                   region masks: marks, box, polygon, wand, plus / minus / rest
  workflow.py, workflow_draw.py  styles, stage plans, todo list, gates, `next` (with and without a reference)
  ops.py, brush.py             paint ops (Stroke / Fill / Streak) with exact JSON, and the raster brush engine
  trace.py, lines.py, masses.py, shapes.py   image analysis, line-art extraction, ink masses, residual shapes
  stage_*.py                   stage algorithms as pure functions: sketch, ink, flat, shade, paint, streaks, levels, local
  airbrush.py                  soft airbrush shading inside flat areas
  pipeline.py                  the scripted reference pipeline (no model in the loop): the engine's ceiling
  manual.py                    hand tools: stroke, fill, shape, gradient, erase, smooth, adjust, pick, palette
  draw_*.py                    drawing without a reference (plan, sketch, line, flat, shade, face, hair, wear,
                               scene, surface, detail, text, checks, canvas, finish) and hand mode (draw_ref.py);
                               draw_finish.py is `pixelcanvas polish`, the finishing pass of a drawn picture
  cli_figure.py                `pixelcanvas figure`: one line that expands into pose, dress, hair3d and face
  pose3d.py                    camera, ground plane, anime-proportioned figures, 73 pose presets, modifiers, balance,
                               --check
  figure3d.py, render3d.py     the dressed 3D figure (body, clothes, hair, hands); ray-cast z-buffer rendering
  face3d.py, toon3d.py         anime faces on a 3D head (expressions, gaze); toon line art and cel shading of 3D parts
  scene3d.py                   `pixelcanvas scene <kit>`: a finished 2D background in the camera's perspective
  draw_pose.py, draw_render.py `camera`, `pose`, `block`, `prop`, `form`; `dress`, `hair3d`, `render` (the guide)
  draw_figure2d.py, anime2d.py, shape2d.py, hand2d.py   `pixelcanvas draw`: 2D anime line art over the guide (body
                               template, clothes, hair locks, hands)
  detail_measure.py, detail_profiles.json   measures by scale and kind against professional pictures
  export.py                    PNG, layered PSD / ORA, time-lapse GIF / MP4, stage pictures
  fx.py, text.py, fonts.py     finishing effects; lettering; font lookup with CJK fallback
skills/pixelcanvas/SKILL.md    the skill agents read (shipped in the wheel as pixelcanvas/skill/); new-picture.md: the
                               plugin's coarse-to-fine run
agents/, commands/             the plugin's subagents (director, blocker, painter, critic) and /pixelcanvas:draw
tests/                         pytest
.claude-plugin/                plugin.json, marketplace.json and icon.png
docs/INSTALL.md, docs/PYPI.md  install guide; the PyPI page
docs/images/                   README pictures, drawn by pixelcanvas (their steps are in the README)
README.md, README-zh.md, AGENTS.md, CHANGELOG.md
```

Dependencies: numpy, Pillow, opencv-python-headless, scikit-image, scipy. Python 3.9+ syntax. No GPU, no ML models,
no network.

## Tests

```bash
pip install -e ".[test]"
python -m pytest tests -q -n auto            # everything, in parallel
python -m pytest tests -q -k "not draw"            # engine, shell, export, fx, text (faster)
python -m pytest tests/test_gate_smoke.py -q       # one tiny reference through every stage command
```

## Conventions

- Coordinates the model sees are 0-1000 on each axis (x right, y down); `coords.py` converts. Images in memory are
  numpy arrays `H x W` first; layers float32 `[H,W,4]`, straight alpha, sRGB 0..1.
- Stage functions are pure: same inputs, same ops; they never touch the disk.
- Every command reports what it changed in numbers and as a picture, and ends in one `NEXT:` line. A failed
  command prints an `EXAMPLE:`.
- A stage command with no options equals the scripted pipeline's stage for the style
  (`tests/test_shell_parity.py`).
- English only in code, comments, output and docs.

## Documentation rule

This repository is a tool. Docs say what it does and how to use it: no disclaimers, no limitation sections, no
hedging, no invented numbers. Show it through command sequences that run. A change to SKILL.md is kept only when
sessions with it score at least as well.

## Release checklist

1. Same version in `pyproject.toml`, `pixelcanvas/__init__.py` (`__version__`) and `.claude-plugin/plugin.json`;
   the CHANGELOG top entry and the `ref` default in `.github/workflows/publish.yml` name the new version.
2. `python -m pytest tests -q -n auto` passes.
3. Build and check: `python -m build`, then `python -m twine check --strict dist/*`; install the wheel in a clean
   venv and run `pixelcanvas --version`, `pixelcanvas help`, `pixelcanvas install-skill --to <temp dir>` (the `build` job of
   `.github/workflows/tests.yml` does the same).
4. Tag `vX.Y.Z` on the release commit and push the tag; then Actions -> publish with `ref` set to the tag
   (Trusted Publishing, environment `pypi`).
