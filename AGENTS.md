# AGENTS.md: easel for AI coding assistants

easel is a command-line paint program you drive. You plan the regions of a picture, then paint it stage by stage
like a painter (sketch, line, flat colour, shade, detail, finish) and look at the canvas after every command. Use
it whenever the user asks to draw, paint, illustrate, sketch, copy or redraw a picture, with or without a reference
image.

**Read [`skills/easel/SKILL.md`](skills/easel/SKILL.md) before drawing.** It holds the loop, the three modes, the
todo list, every stage and the hand tools.

## Install

```bash
pip install pixelcanvas            # the easel command
easel-install-skill                # Claude Code: ~/.claude/skills/easel
easel-install-skill --opencode     # opencode: ~/.config/opencode/skills/easel
easel-install-skill --to DIR       # any other skills folder (DIR/easel)
```

Tools without skill support: put `skills/easel/SKILL.md` in their rules or system prompt.

## The loop

1. Run ONE `easel ...` command per shell call (or one `easel do "...; ..."` batch).
2. Read the report: scores `before -> after`, `WARNING`, worst spots.
3. Open the `VIEW:` file with your file-read tool; open `VIEW-REF:` too when you compare.
4. Say in one sentence what changed and whether it helped.
5. Follow `NEXT:`, or `easel undo` when a score fell or the picture looks worse.

Coordinates are 0-1000 on each axis, read off the rulers of the views. Quote colours: `--color "#202020"`.

## Start

```bash
easel new pic --ref reference.jpg --mode trace                       # redraw a picture
easel new pic --size 1000x1200 --request "a chibi fox mage holding a staff"   # no reference
easel help                                                           # every command
easel next                                                           # the next command, always
```

Finish = `easel export final.png` and `easel stage done` in the finish stage, with every todo line completed.

## In this repository

The code is `pixelcanvas/easel/`, the tests `tests/easel/` (`python -m pytest tests/easel -q`). See
[CLAUDE.md](CLAUDE.md) for the project map and the release steps.
