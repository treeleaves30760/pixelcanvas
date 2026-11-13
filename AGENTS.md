# AGENTS.md: pixelcanvas for AI coding assistants

pixelcanvas is a canvas, a toolbox and a pair of eyes for drawing with code. The agent designs the picture and
draws it in Python (numpy + Pillow, no image model): illustrations, characters, posters, diagrams, charts, UI
mock-ups, technical drawings, 3D objects and exact text in any language. A **render -> check -> look -> fix** loop
turns a first draft into a picture worth delivering.

**Read [`skills/pixelcanvas/SKILL.md`](skills/pixelcanvas/SKILL.md) before drawing.** It holds the loop, freehand
drawing, art direction, the kit table and how to read the report.

## Two skills

| Skill | For | The model writes |
|---|---|---|
| `pixelcanvas` | capable coding agents (Claude Code, Codex, Gemini CLI, Cursor, opencode with a large model) | a Python scene, drawn freehand with kits as ingredients |
| `pixelcanvas-lite` | small and local models (opencode with a model served on your own machine) | one JSON spec; `tools/compose.py` draws it and its `NEXT:` line says what to do |

## Install

```bash
pip install pixelcanvas
pixelcanvas install-skill claude                 # or agents, codex, cursor, opencode; --skill pixelcanvas-lite | both
uvx pixelcanvas doctor                           # Python, numpy, Pillow, fonts (CJK included), write access
```

From a clone: `python tools/install_skill.py claude --skill both` (targets `agents`, `claude`, `codex`, `cursor`,
`opencode`; `--project` for this project only). Every way to install, MCP included: [docs/INSTALL.md](docs/INSTALL.md).

## The loop

```bash
python tools/new_scene.py fox --prompt "a fox with a lantern in the snow" --dir ./art   # starter -> ./art/fox.py
python tools/render.py ./art/fox.py --scale 0.5 --check           # draft + report -> ./art/outputs/
python tools/look.py zoom art/outputs/fox_x0.5.png 0.3 0.2 0.7 0.6   # look closer (guides, values, probe ...)
python tools/render.py ./art/fox.py --check --deliver ./art/final/  # full size, copied where the user wants it
python tools/match.py REF.png ./art/outputs/fox.png               # with a style reference: the gaps, largest first
```

A picture gets the freehand starter (kind `illustration`); a chart, diagram, engineering drawing or UI screen gets a
starter holding the request's data. `python tools/kit.py find "<request>"` names the kit and its calls. With a style reference,
`python tools/match.py REF.png` (`pixelcanvas/reference.py`) prints its palette, lines and tone bands as numbers. Every
tool runs from any folder (`python <root>/tools/...`), and `pixelcanvas <tool>` does the same after
`pip install pixelcanvas`.

## Small and local models

Install the lite skill into opencode and run opencode with a local model:

```bash
pixelcanvas install-skill opencode --skill pixelcanvas-lite
python tools/compose.py --example poster > ./art/poster.json      # what the model starts from
python tools/compose.py ./art/poster.json --check                 # draws, reports, ends with one NEXT: line
```

## Rules that keep pictures good

1. Plan first: subject and frame, a mood and its light, a palette of 4-6 colours with one accent, the exact text.
2. Draw freehand for illustration; use a kit where structure is the hard part (charts, diagrams, drafting, UI, 3D).
3. Never deliver without looking: open the preview and zoom, or use `look.py report` without vision.
4. Coordinates: x right, y down, `arr[y, x]`; positions are fractions of the canvas.
5. After changing the package or the docs, run `python -m pytest -q`.
