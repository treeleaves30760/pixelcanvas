---
name: pixelcanvas
description: >-
  Draw, paint, illustrate or redraw a picture with pixelcanvas, a command-line paint program: you plan regions, then
  work stage by stage like a painter (sketch, line, colour, shade, detail, finish) and look at the canvas after
  every command. Use whenever the user asks to draw, paint, illustrate, sketch, copy or redraw a picture, with or
  without a reference image: an anime character, a landscape, a poster, a portrait, "redraw reference.jpg".
---

# pixelcanvas

You paint with `pixelcanvas` commands in the shell. The picture lives in a project folder `<name>.pixelcanvas`; every
command changes it, reports numbers and writes a picture of the result.

## The loop: do this after EVERY command

1. Run ONE `pixelcanvas ...` command per shell call (or one `pixelcanvas do "...; ..."` batch). Never join pixelcanvas commands
   with `&&` or `;` in the shell: you would miss the VIEW of the first one.
2. Read the text report: scores `before -> after`, `WARNING`, `worst spots`.
3. Open the `VIEW:` file with your file-read tool (Read / read). Open `VIEW-REF:` too when you compare details.
4. Write ONE sentence: what changed, and did it help (numbers up, picture closer to the reference)?
5. Follow `NEXT:`, or do something else and say why in one sentence.

Undo at once when a score fell or the picture looks worse: `pixelcanvas undo` (or the `pixelcanvas undo N` that NEXT gives).
Never run a second command before you have looked at the first one's VIEW.
Coordinates are always 0-1000 on each axis (x right, y down), read off the rulers of the views. Never pixels.
Colours: always quote them, `--color "#202020"`. In the shell an unquoted `#` starts a comment.

## Which mode

- A picture to redraw or copy (`reference.jpg`): **trace mode**, on a working canvas 2x to 4x the picture you
  export (4x for a small picture, set by its size; `--scale N` on `pixelcanvas new` overrides): the strokes follow the reference and take their colours from it along the stroke; you plan the regions, order the
  work, check each command's effect and decide where to refine. `pixelcanvas new pic --ref reference.jpg --mode trace`,
  then "Start" and the sections after it.
- Only when the user asks you to draw every mark yourself: **hand mode** (experimental),
  `pixelcanvas new pic --ref reference.jpg --mode hand`, then the section "Hand mode" below.
- No picture: **a new picture**. With the pixelcanvas plugin's subagents available, read `new-picture.md` in this
  skill's base directory (next to this SKILL.md; no search needed) and follow it: you direct, `pixelcanvas-director`, `pixelcanvas-blocker`, `pixelcanvas-painter` and
  `pixelcanvas-critic` build it coarse to fine. Without them: free mode, `pixelcanvas new pic --mode free --request
  "..."`, then "Without a reference picture".

## Start (trace mode)

```bash
pixelcanvas new pic --ref reference.jpg --mode trace   # makes pic.pixelcanvas, guesses the style, shows the numbered marks
```
Then open VIEW (the numbered marks) and VIEW-REF (the reference). Check the style line:
`cel` = anime with dark outlines · `ink` = black-and-white line art, manga · `paint` = painting, brushwork, no
outlines · `photo` = photograph · `poster` = flat shapes and big lettering. Wrong guess: `pixelcanvas style paint`.

## The todo list (required)

1. Plan the regions (next section), then `pixelcanvas stage done`.
2. Run `pixelcanvas todo --all`. Copy EVERY line it prints into your todo tool (TodoWrite / TaskCreate / todowrite),
   in the same order, with the same words. The plan stage is completed; the next line is in progress.
3. Work through the lines in order. When `pixelcanvas stage done` succeeds, its report says which line to mark
   completed: update your todo tool right then.
4. Add a line to both lists when you find extra work: `pixelcanvas todo add "fix the left eye"`.
5. Never say the picture is finished while `pixelcanvas status` shows open items or a stage before `finish`.

## Plan: regions

A region is a named part you will paint as one unit. Make 4-12 of them.

- Give its own region to: the face (focus 3), each eye if large, hands, hair, each piece of clothing, each big
  object, sky, ground, background. Merge tiny bits into the part they belong to.
- Focus 3 = where the viewer looks first (face, eyes, hands, lettering). Focus 2 = the main subject's other parts.
  Focus 1 = the rest. The detail stage refines focus 3 first, then focus 2, each with a close look.
- Pick mark numbers from the marks VIEW: every number inside the part, nothing outside it.

```bash
pixelcanvas region add hair --marks 4,5,9,13
pixelcanvas region add face --wand 500,550 --focus 3     # click inside a part that has no mark of its own
pixelcanvas region add sword --box 600,100,700,800       # a box around one object; the tool fits its outline
pixelcanvas region add coat --marks 11,12 --minus face   # cut another region out
pixelcanvas region rest background                       # everything not yet in a region
```
- A part has no mark of its own (a white face inside a white background mark): use `--wand x,y` inside it.
- `--box` is for an object that stands out from its surroundings. If the note says it kept a small share of the
  box, use `--wand` or `--box ... --no-refine`.
- After each `region add`, open VIEW: every region is tinted. The new one must cover its part and nothing else.
  Wrong? Run the same `region add <name> ...` again with other marks: it replaces the region.
- Batch the regions in one call when you are sure: `pixelcanvas do "region add hair --marks 4,5; region add bow --marks 1,2"`.

## Stages

Each stage paints on its own layer and has a gate the tool measures per region. `pixelcanvas next` always names
the next command. Every stage after sketch works the same way:

1. Run the stage tool on the whole picture: `pixelcanvas ink all` (the tool's best result; never start region by region).
2. Open VIEW and VIEW-REF. Compare.
3. `pixelcanvas stage done`. If the gate names short regions, fix only those (`NEXT:` gives the command), then
   `pixelcanvas stage done` again.

**sketch** - loose construction lines, one per region: this checks your regions.
```bash
pixelcanvas do "sketch face; sketch hair; sketch bow; sketch background"
```
Right: in VIEW the blue lines follow every silhouette over the faded reference. A region drawn in the wrong
place means the region is wrong: fix it with `region add` (same name), then sketch it again.

**line** (cel, ink, poster) - `pixelcanvas ink all`. Right: clean outlines on every edge the reference draws.
Region short of `lines >= 50`: `pixelcanvas ink hair --detail 0.8`. One missing line: draw it by hand (below).
Extra or doubled lines: `pixelcanvas erase 400,300,460,360 --layer line`.

**flat** (cel, ink, poster) - `pixelcanvas flat all`. Right: every area has its colour, no white gaps.
Gap: `pixelcanvas fill 520,640 --color pick`. Wrong colour in one region: `pixelcanvas flat hair --colors 8`.

**shade** (cel) - `pixelcanvas shade all`, then `pixelcanvas smooth all` once when the todo lists it (a soft-shaded reference: it
blends flat and shade, never the detail layer; paint and photo get it in finish). Right: shadows on the same side and with the same softness as the
reference. Too dark or blotchy: `pixelcanvas undo`, then `pixelcanvas shade all --levels 3 --soft 0.5`.
A flat reference gets almost no shapes here: that is right.

**block-in** (paint, photo) - `pixelcanvas paint all` (big strokes, two passes). Right: every area covered with
the right colour masses; no detail yet.

**forms** (paint, photo) - `pixelcanvas paint all` again (medium strokes). Right: light and shadow sides, main
edges in place. Region short: `pixelcanvas paint lake --size 8 --density 1.5`.

**detail** - the picture is painted in detail levels, coarse to fine: 5 block-in, 4 forms, 3 edges, 2 fine
strokes, 1 hairlines. Levels 2 and 1 are streaks: thin strokes whose colour and width follow the reference.
`pixelcanvas detail all` lays level 2 over the whole picture and lists the regions in the order to refine: focus first,
then the largest remaining difference; it first draws each thin line of the reference as one stroke along its
centre and reports how many and the error on those lines. Then every region, in that order (the todo lists them all):
```bash
pixelcanvas detail face --level 1   # hairlines on the face; the report: how far off it was and is now
pixelcanvas look face               # open VIEW and VIEW-REF; does it still differ?
```
Then the dots, once, after every region is at level 1:
```bash
pixelcanvas dot all                 # soft 1.5 px dots on the texture points a stroke cannot carry (glints, specks)
```
The report gives the number of dots, their share of the canvas, the error before and after, and VIEW / VIEW-REF
of the place the dots changed most: open both. The gate wants every region at level 1 and the dots laid. A long
level or dot command stops after about 40 s and says so; `NEXT:` repeats it and it continues where it stopped.
A hand fix that makes a score fall: `pixelcanvas undo`.
`pixelcanvas retouch <region>` is level 0: it copies the reference's pixels where they still differ, and every report
and export prints the share of the canvas copied. Use it only on a small region you name that still differs after
level 1 (an eye, a letter).

**finish** - compare and deliver.
```bash
pixelcanvas look               # whole canvas + reference
pixelcanvas look face          # the last close look at every focus-3 region (the gate asks for it)
pixelcanvas sharpen            # the finishing sharpen of the delivered picture, fitted to the reference
pixelcanvas export final.png   # then: pixelcanvas stage done
```
`pixelcanvas sharpen` reports the share of pixels it changes, the error before and after, and VIEW / VIEW-REF of the
place it changes most: open both; `pixelcanvas sharpen --off` takes it back. Export delivers with it, and every report
and export prints what the picture is made of: levels, share retouched, share dotted, working scale, delivery.
Glow, grain or paper texture in the reference: `pixelcanvas fx grain`. Lettering: `pixelcanvas text "TITLE" --box 100,50,900,150`.

## Hand tools: when a stage tool cannot fix a spot

Use them for anything the stage tools miss: a missing line, a gap, an eye highlight.
```bash
pixelcanvas stroke "480,590 510,585 540,590" --size 3 --color "#202020"     # a line through the points
pixelcanvas stroke "480,590 510,585 540,590" --size 3 --color pick --snap  # reference colour, pulled onto its edge
pixelcanvas fill 520,640 --color pick                                       # bucket fill inside the line art
pixelcanvas shape ellipse 470,440,500,470 --color "#ffffff"                   # an eye highlight
pixelcanvas erase 400,300,460,360 --layer line                              # remove marks in a box
```
Read the points off the rulers of a close view (`pixelcanvas look 450,550,600,650`) and open its VIEW before the fix.
After the fix open VIEW again. Keep it only when the score line (`before -> after`) rose; otherwise `pixelcanvas undo`.

## When numbers stall or a gate will not pass

1. Do the `NEXT:` command; each retry the tool proposes stronger settings.
2. Look at the spot (`pixelcanvas look <box from worst spots>`) and fix it by hand.
3. After 3 tries, if the picture matches the reference there: `pixelcanvas stage done --because "<what you see>"`.
   Give the real reason ("the reference has no lines on the background"), never just "done".

## "Redraw it, but ...": a picture that differs from its reference

When the request asks for something the reference does not have (another colour, a poster, a new setting), state
each change after the regions, before `pixelcanvas stage done`; the plan's NEXT asks for it. A plain copy:
`pixelcanvas change none`. Each change edits the target, the picture the stages trace (the original stays as
`orig.png`). After `pixelcanvas stage done` the change stage makes them: follow NEXT until it says `pixelcanvas stage done`, then
trace as usual. `pixelcanvas look --show changes` shows original and target side by side.

Recolour ("with mint-green hair and a dark red top"):
```bash
pixelcanvas change add recolor hair --to "#8fdcb4"
pixelcanvas change add recolor clothes --to "#7b2635"
pixelcanvas stage done
pixelcanvas recolor hair --to "#8fdcb4"            # NEXT gives each command; stray patches:  pixelcanvas region refine hair --by hue
pixelcanvas recolor clothes --to "#7b2635"
```
Poster ("a minimalist poster titled SLOW SUMMER, navy line art on cream, coral accents"):
```bash
pixelcanvas change add canvas --size 900x1200 --place 186,240,814,930 --color "#f3e9d8"
pixelcanvas change add palette map --to "#1d2a44,#f3e9d8" --accent "jacket=#e8644a"
pixelcanvas change add graphic disc 260,295,860,745 --color "#e97a62"
pixelcanvas change add text "SLOW SUMMER" --box 70,40,930,175 --font display --weight black --color "#1d2a44"
pixelcanvas stage done                             # then NEXT: canvas, palette map, graphic; the lettering comes in finish
```
New setting ("in front of a mountain lake at sunset"). Draw the setting first as a free picture in its own folder
(`pixelcanvas new lake --mode free --size 850x1511 --request "a mountain lake at sunset"`, through to `pixelcanvas export`), or
take a picture file, a gradient or one colour:
```bash
pixelcanvas change add cutout figure --regions face,ear,headwrap,hair,collar,cloak
pixelcanvas change add background --from ../lake/lake.pixelcanvas     # or --from sky.png, --gradient "#f2a65a,#5b5f8f", --color "#88aacc"
pixelcanvas change add light                                     # tint and rim light from the background's light
pixelcanvas stage done
pixelcanvas cutout figure --regions face,ear,headwrap,hair,collar,cloak
pixelcanvas background --from ../lake/lake.pixelcanvas
pixelcanvas light
```
The cut-out report names its three weakest edge spots (boxed in VIEW, close up below). Ground left in the figure:
`pixelcanvas region add stick --box <spot box>`, then the cutout command again with `--minus stick`; a part left out:
`--plus <region>`. A figure standing inside the picture gets a contact shadow: `pixelcanvas shadow figure`.

## Hand mode (experimental): draw it yourself, with the reference beside you

Every line, fill and shadow is a mark you place; the reference supplies edges, colours and shadow shapes.
No stage command works on the whole picture at once: one part or area at a time. Follow the stage order `pixelcanvas new` prints.

1. Plan. Say what the picture shows: `pixelcanvas plan subject girl --box 250,80,750,1000`. Parts from the numbered
   reference marks: `pixelcanvas plan part hair --of girl --marks 3,4,9`. `pixelcanvas palette from-ref`, then name what
   you use: `pixelcanvas palette name 2 hair 1 skin 3 line`.
2. Sketch. A head or big shape is enough: `pixelcanvas sketch head girl --box 400,120,600,360`.
3. Line. `pixelcanvas line candidates hair` numbers the reference contours you have not drawn yet; `pixelcanvas line take
   1,4,7` inks the ones that belong (up to 10). Any edge yourself, a few points snapped to the reference:
   `pixelcanvas line "420,300 380,420 400,560" --snap ref`. Close leaks: `pixelcanvas line close all`.
4. Flat. `pixelcanvas flat` numbers the areas; `pixelcanvas flat 1,2,3 --color pick` takes the reference's colour in each.
   An area that holds two colours: `pixelcanvas line candidates a7`, take its border, then flat again.
5. Shade. `pixelcanvas shade face --like ref --levels 2` copies the reference's shadow shapes in that part.
6. Detail. `pixelcanvas detail candidates face`, then `pixelcanvas detail take 2,5`.
7. Finish. `pixelcanvas export final.png --stages stages`, then `pixelcanvas stage done`.

Every report shows your scores against the reference and the worst spots; VIEW is your canvas, VIEW-REF the
reference over the same area; sketch and line views are a lightbox. Any colour can be `pick@x,y`.
Read coordinates off a zoomed lightbox view (your canvas over the faint reference): `pixelcanvas look hair --lightbox`.
A `LEAKS` suggestion: look at the spot first, run it only when you see the gap. Small features that `detail take`
drops (pupils, highlights, a mouth): draw them with `shape`, `stroke`, and clean stray ink with
`erase ... --layer line`.
The todo list, the loop and the stall rules above apply as in trace mode.

## Without a reference picture

Free mode draws from a text request. You say what the picture is, the tool builds it stage by stage, on a working canvas 3x the
picture you export (coordinates stay 0-1000; `--scale 1|2|3` on `pixelcanvas new` sets it). Follow `NEXT:`. When
`stage done` refuses, run the command on its `NEXT:` line: every refused line names the place and the fix.
```bash
pixelcanvas new pic --size 1000x1200 --request "a chibi fox mage holding a staff"
```
1. **plan** - one box per subject, parts with boxes (focus 3 for face, eyes, hands, title), light, palette:
   `pixelcanvas plan subject fox --box 170,60,760,960` · `pixelcanvas plan part head --of fox --box 200,90,720,500 --focus 3`
   · `pixelcanvas plan light top-left` · `pixelcanvas palette make --base "#e8803a"`, look, `pixelcanvas palette pick 2`,
   `pixelcanvas palette set robe "#4a3c86"`. Scenes back to front, sky first; a farther thing drawn later still goes
   behind nearer ones; change the order with `pixelcanvas plan order sky,mountains,lake,pines`.
2. **sketch** - people: `pixelcanvas sketch figure fox --pose stand|walk|hold|wave|sit|bust` (`--heads 2.5` for
   chibi), then `pixelcanvas sketch head fox --yaw 15`; `pixelcanvas face <head id>` lays brows, nose and mouth (an animal
   gets a muzzle with `--kind animal`). Animals: `pixelcanvas sketch ellipse 320,500,660,710` for the body and
   `pixelcanvas sketch head cat`; a dressed animal (a fox mage in a robe) stands on a mannequin. Things:
   `pixelcanvas sketch curve "x,y x,y ..."`. Landscapes: `pixelcanvas sketch horizon 560`.
3. **line** (cel, ink) - 3-8 points per line: `pixelcanvas line "330,190 300,280 305,350"`; the tool smooths, tapers
   and joins ends. Eyes: `pixelcanvas line eye eye-l,eye-r`. Leaks: `pixelcanvas line gaps`, then all closes in ONE
   command: `pixelcanvas line close G1,G3` (ids renumber after a close). Sky, sea and cliffs need no lines.
4. **flat** - `pixelcanvas flat auto`, look at the numbered VIEW, fix one area: `pixelcanvas flat 8 --color fur --part paw`;
   the area under a point: `pixelcanvas flat 500,700 --color robe`; background showing inside a figure:
   `pixelcanvas flat hole 230,640`. Sky, sea and cliffs are colour bands: `pixelcanvas band sea "0,700 1000,700" --below
   --color sea`.
5. **shade** - `pixelcanvas shade all --soft 0.3`, cast shadows `pixelcanvas shadow --from hair --onto face`, `pixelcanvas rim hair`.
6. **detail** - `pixelcanvas eye eye-l,eye-r --color iris` · `pixelcanvas hair hair --flow "x,y x,y x,y"` ·
   `pixelcanvas fold "x,y x,y x,y" --region robe` · `pixelcanvas blush face` ·
   `pixelcanvas scatter robe --kind star --count 14` · `pixelcanvas text "BLUE NOTE" --box 60,50,940,240 --color "#f1e3c3"`.
   Then `pixelcanvas surface auto` and look at the view; change one area with `pixelcanvas surface coat --material
   cloth-rough`. If the gate says detail is short, run the command it prints: it asks for detail where the
   viewer looks and holds a picture that turned busy; plain skies, water and flat styles stay plain. For anime
   and cel pictures the surface keeps the flats clean: a few cel-shadow shapes, grouped hair light, no texture.
7. **finish** - `pixelcanvas look`, fix, `pixelcanvas polish` (one light over the picture, far things lighter and cooler,
   line art coloured from its fills, soft edges in painterly styles, strays out; leave a part out with
   `--no-grade`, `--no-depth`, `--no-lines`, `--no-soften`, `--no-strays`, `--no-clean`), look, then
   `pixelcanvas export final.png --stages stages`, `pixelcanvas stage done`.

Painterly (`paint` style): block-in with bands, `pixelcanvas band sky "0,560 1000,560" --above --color sky`,
`pixelcanvas band pines "x,y x,y ..." --closed --color pines --rough 0.15`. A plain band is a placeholder: draw the
trees over it, `pixelcanvas band pines "100,900 150,600" --tiers 5 --color pines` (a band named `pines-2` belongs to
`pines` in the depth order); move a band with
`--shift dx,dy`; a sunset or night sky: `--color sky --color2 sky-low`. Forms `pixelcanvas paint pines --dir 80`;
detail `pixelcanvas haze --amount 0.4`, `pixelcanvas rim`, `pixelcanvas scatter`. Poster style: flat shapes with
`band ... --closed`, then `text`; no outlines.

## Rules

- Run every command from the folder that holds `pic.pixelcanvas`; pixelcanvas finds the project itself.
- One stage at a time, in the order of your todo list. Look after every command.
- Hand fixes only where you SEE a difference in VIEW against VIEW-REF, and only when the score rises. A spot
  the numbers flag that looks identical in both pictures needs no fix.
- When a command fails, the report gives `EXAMPLE:`: fix your command from it; do not repeat the same one.
- `pixelcanvas status` shows the stage, the todo list and the scores; use it when unsure.
- Finish = `pixelcanvas export final.png` and `pixelcanvas stage done` in the finish stage, todo lists all completed.
