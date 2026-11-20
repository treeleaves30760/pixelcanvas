# pixelcanvas

pixelcanvas is a command-line paint program an AI agent drives. Give it a reference picture and the agent redraws it like a
painter: it plans the regions, then works stage by stage (sketch, line, flat colour, shade, detail, finish). After
every command pixelcanvas reports what changed in numbers and writes a picture of the result; the agent opens it, judges
it, and goes on or undoes. Ask for a redraw with changes (mint-green hair, a poster, a lake behind the figure), or
describe a picture in words.

On a private set of 93 reference pictures (anime, landscape, promotional art), the redraw path reaches a judge
mean of 9.42 out of 10, with 88 of 93 rated the same quality as the reference. In the redraw path the strokes follow
the reference and take their colours from it along the stroke, and no picture is retouched pixel by pixel. A small
open-weight model (Qwen 3.8 Flash class) driving the skill in Claude Code or opencode finishes a redraw in about
19-45 minutes and 44-86 commands; its sessions now cover 23 anime pictures, with a judge mean of 9.52 and 21 of 23
rated the same quality as the reference (the scripted path on the same pictures: 9.48, 22 of 23).

## Install

```bash
pip install pixelcanvas
pixelcanvas install-skill                # the skill -> ~/.claude/skills/pixelcanvas
pixelcanvas install-skill --opencode     # or -> ~/.config/opencode/skills/pixelcanvas
```

Then ask your agent: *"Redraw reference.jpg with pixelcanvas"*. Every command is `pixelcanvas <command>`.

## A redraw

```bash
pixelcanvas new pic --ref reference.jpg --mode trace
pixelcanvas region add hair --marks 4,5,9,13
pixelcanvas region add face --wand 500,550 --focus 3
pixelcanvas region rest background
pixelcanvas stage done
pixelcanvas todo --all
pixelcanvas do "sketch face; sketch hair; sketch background"
pixelcanvas stage done
pixelcanvas ink all
pixelcanvas stage done
pixelcanvas flat all
pixelcanvas stage done
pixelcanvas shade all
pixelcanvas stage done
pixelcanvas detail all
pixelcanvas detail face --level 1
pixelcanvas dot all
pixelcanvas stage done
pixelcanvas sharpen
pixelcanvas export final.png
pixelcanvas stage done
```

Every command prints the scores `before -> after`, a `VIEW:` picture of the canvas, a `VIEW-REF:` picture of the
reference and a `NEXT:` line. The project folder `pic.pixelcanvas/` keeps one PNG per layer, the regions, an undo history
and a journal; `pixelcanvas export final.png --psd --ora --timelapse film.gif` writes the layers and a time-lapse.

## Redraw it, but ...

State the changes a request asks for after the regions; a change edits the target the stages trace, and the
original stays as `orig.png`:

```bash
pixelcanvas new pic --ref reference.jpg --mode trace --request "redraw it with mint-green hair"
pixelcanvas region add hair --marks 4,5,9,13
pixelcanvas region add face --wand 500,550 --focus 3
pixelcanvas region rest background
pixelcanvas change add recolor hair --to "#8fdcb4"
pixelcanvas stage done
pixelcanvas recolor hair --to "#8fdcb4"
```

A poster takes `change add canvas`, `palette map`, `graphic` and `text`; a new setting takes `cutout`, `background`,
`light` and `shadow`. `pixelcanvas next` names each command.

## Without a reference

People in a place: a `scene` kit draws the place from the camera, one `figure` line poses, dresses and gives hair and
a face to each figure on a 3D skeleton, and the picture is drawn in 2D over the faded guide.

```bash
pixelcanvas new pic --mode free --size 1000x834 --request "two children in a school corridor"
pixelcanvas do @steps.txt                          # camera, scene corridor ..., figure boy ..., figure girl ...
pixelcanvas render
pixelcanvas do "draw all; flat auto; shade all"
pixelcanvas polish --focus boy --motion boy:toward
pixelcanvas export final.png --stages stages
pixelcanvas stage done
```

![corridor](https://raw.githubusercontent.com/treeleaves30760/pixelcanvas/main/docs/images/corridor.png)

Kits: corridor, room, classroom, street, platform, outdoor, field, beach, rooftop, stairs. 73 pose presets with
modifiers, 18 expressions, hair styles with accessories; `pixelcanvas help pose,figure,scene` lists the options.

Documentation, the skill and the source: <https://github.com/treeleaves30760/pixelcanvas>. MIT licence.
