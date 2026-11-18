# pixelcanvas: easel

easel is a command-line paint program an AI agent drives. The agent plans the regions of a picture, then paints it
stage by stage like a painter: sketch, line, flat colour, shade, detail, finish. After every command easel reports
what changed in numbers and writes a picture of the result; the agent opens it, judges it, and goes on or undoes.
Give it a reference picture to redraw, or a text request.

On a private set of 93 reference pictures (anime, landscape, promotional art), the redraw path reaches a judge
mean of 9.42 out of 10, with 88 of 93 rated the same quality as the reference. In the redraw path the strokes follow the reference and take their colours
from it along the stroke, and no picture is retouched pixel by pixel. A small
open-weight model (Qwen 3.8 Flash class) driving the skill in Claude Code or opencode finishes a redraw in about
22-45 minutes and 46-75 commands; on 14 anime pictures its redraws have a judge mean of 9.57, with 13 of 14 rated the
same quality as the reference (the scripted path on the same pictures: 9.50, 13 of 14).

## Install

```bash
pip install pixelcanvas
easel-install-skill                # the skill -> ~/.claude/skills/easel
easel-install-skill --opencode     # or -> ~/.config/opencode/skills/easel
```

Then ask your agent: *"Redraw reference.jpg with easel"*. The `pixelcanvas` command runs the same program as
`easel`.

## A redraw

```bash
easel new pic --ref reference.jpg --mode trace
easel region add hair --marks 4,5,9,13
easel region add face --wand 500,550 --focus 3
easel region rest background
easel stage done
easel todo --all
easel do "sketch face; sketch hair; sketch background"
easel stage done
easel ink all
easel stage done
easel flat all
easel stage done
easel shade all
easel stage done
easel detail all
easel detail face --level 1
easel dot all
easel stage done
easel sharpen
easel export final.png
easel stage done
```

Every command prints the scores `before -> after`, a `VIEW:` picture of the canvas, a `VIEW-REF:` picture of the
reference and a `NEXT:` line. The project folder `pic.easel/` keeps one PNG per layer, the regions, an undo history
and a journal; `easel export final.png --psd --ora --timelapse film.gif` writes the layers and a time-lapse.

Without a reference: `easel new pic --size 1000x1200 --request "a chibi fox mage holding a staff"`, then follow
`NEXT:`.

Documentation, the skill and the source: <https://github.com/treeleaves30760/pixelcanvas>. MIT licence.
