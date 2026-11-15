# AGENTS.md: pixelcanvas for AI coding assistants

pixelcanvas is a canvas, a toolbox and a pair of eyes for drawing with code. The agent designs the picture and
draws it in Python (numpy + Pillow, no image model): illustrations, characters, posters, diagrams, charts, UI
mock-ups, technical drawings, 3D objects and exact text in any language. The picture lives in a **studio**: a
persistent stack of layers in `./art/NAME/`, one `# %% name` section of one draft file per object, edited one command
at a time. Every command re-renders what changed, checks the picture against the inventory lines of the draft and
prints one `NEXT:` line; a look sheet (the picture, then each item close up) shows what the numbers cannot, and the
`--out` file is the finished picture. The finished drawing can be delivered in a real art medium (render, flat, oil,
watercolour, ink, pixel art): `studio set scene medium=oil`.

**Read [`skills/pixelcanvas/SKILL.md`](skills/pixelcanvas/SKILL.md) before drawing.** It holds the studio loop, the fix
table, freehand drawing, art direction and the kit table.

## Two skills

| Skill | For | The model writes |
|---|---|---|
| `pixelcanvas` | coding agents (Claude Code, Codex, Gemini CLI, Cursor, opencode with a mid-size or large model) | one Python draft cut into `# %% name` sections in the studio, then one command per change |
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

Write the whole first draft in one file, `./art/NAME/draft.py`: one script cut into `# %% name` sections, back to
front. The first comment line of a section is its inventory line (`# apple: 3, green, one bitten | in bowl`: count,
colours, 'quoted text', states, then `|` and places); those lines are the request checklist.

```bash
python tools/studio.py new fox --prompt "a fox with a lantern in the snow" --size 1920x1080 --out ./fox.png --from ./art/fox/draft.py
python tools/studio.py scale fox 1.3                     # one command per non-code change; ./fox.png is rewritten
python tools/studio.py add --from ./art/fox/draft.py     # after editing one section: only what changed is redrawn
python tools/studio.py note add "3 lantern glow too faint"   # keeps a list of faults for NEXT to walk through
python tools/studio.py set scene medium=watercolour     # deliver it in a medium: render flat oil watercolour ink pixel off
python tools/studio.py stages                            # flat silhouettes -> relief clay -> lit colour -> the medium
python tools/studio.py undo                              # when it got worse
python tools/studio.py -p fox move title --by 0 0.02     # a later request continues the same picture
```

Every report prints `picture file: <path> (WxH, written now)` and ends with one `NEXT:` line. It names the failing section or the open item with its fixing command; when
nothing measured is open it says `look 1 of 3: open <look.png>`: the sheet with the picture and a close-up per
checklist item. Fix a line that does not show at a glance, look again; after three looks NEXT says `finished: the
picture is <out path>` and the picture stays finished. A request to draw in a medium ("a watercolour of", "oil painting",
"ink drawing", "pixel art", "3D render") sets `medium` in `studio new`; its checks (value pattern, colours, contrast, text) join
the `NEXT:` line ([medium.md](skills/pixelcanvas/references/medium.md)). `deliver PATH [--scale 2] [--layers]` writes a larger copy, another path, PSD and ORA layers.
`python tools/studio.py api sk.paint` prints how any call is written; `batch "CMD" "CMD"` runs several commands with
one report, and `fish*` names a cell with all its copies. For a chart, diagram, engineering drawing or UI screen
`studio new` without `--from` prints the `new_scene.py` starter and `add base --scene` commands: the starter becomes
the bottom layer, annotations are layers above it. Every tool
runs from any folder (`python <root>/tools/...`), and `pixelcanvas <tool>` does the same after
`pip install pixelcanvas`. A one-shot scene file still renders with `python tools/render.py SCENE.py --check`.

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
3. Look before you finish: open `look.png`, fix each line that does not show at a glance with one section edit or one command.
4. Coordinates: x right, y down, `arr[y, x]`; positions are fractions of the canvas.
5. After changing the package or the docs, run `python -m pytest -q`.
