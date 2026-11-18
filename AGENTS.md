# AGENTS.md: pixelcanvas for AI coding assistants

pixelcanvas is a command-line paint program you drive. You plan the regions of a picture, then paint it stage by stage
like a painter (sketch, line, flat colour, shade, detail, finish) and look at the canvas after every command. Use
it whenever the user asks to draw, paint, illustrate, sketch, copy or redraw a picture, with or without a reference
image.

**Read [`skills/pixelcanvas/SKILL.md`](skills/pixelcanvas/SKILL.md) before drawing.** It holds the loop, the three modes, the
todo list, every stage, the hand tools and the section "Redraw it, but ...".

## Install

```bash
pip install pixelcanvas            # the pixelcanvas command
pixelcanvas install-skill                # Claude Code: ~/.claude/skills/pixelcanvas
pixelcanvas install-skill --opencode     # opencode: ~/.config/opencode/skills/pixelcanvas
pixelcanvas install-skill --to DIR       # any other skills folder (DIR/pixelcanvas)
```

Tools without skill support: put `skills/pixelcanvas/SKILL.md` in their rules or system prompt.

## The loop

1. Run ONE `pixelcanvas ...` command per shell call (or one `pixelcanvas do "...; ..."` batch).
2. Read the report: scores `before -> after`, `WARNING`, worst spots.
3. Open the `VIEW:` file with your file-read tool; open `VIEW-REF:` too when you compare.
4. Say in one sentence what changed and whether it helped.
5. Follow `NEXT:`, or `pixelcanvas undo` when a score fell or the picture looks worse.

Coordinates are 0-1000 on each axis, read off the rulers of the views. Quote colours: `--color "#202020"`.

## Start

```bash
pixelcanvas new pic --ref reference.jpg --mode trace                       # redraw a picture
pixelcanvas new pic --size 1000x1200 --request "a chibi fox mage holding a staff"   # no reference
pixelcanvas help                                                           # every command
pixelcanvas next                                                           # the next command, always
```

A redraw whose request asks for changes ("with mint-green hair", "as a poster", "in front of a lake") starts with the
request, `pixelcanvas new pic --ref reference.jpg --mode trace --request "..."`. After the regions, state each change and
close the plan; the change stage makes them, and the stages that follow trace the changed picture:

```bash
pixelcanvas change add recolor hair --to "#8fdcb4"       # or: change add canvas | palette map | graphic | cutout | background | light | text
pixelcanvas stage done
pixelcanvas recolor hair --to "#8fdcb4"                  # `pixelcanvas next` names each command
```

Finish = `pixelcanvas export final.png` and `pixelcanvas stage done` in the finish stage, with every todo line completed.

## In this repository

The code is `pixelcanvas/`, the tests `tests/` (`python -m pytest tests -q`). See
[CLAUDE.md](CLAUDE.md) for the project map and the release steps.
