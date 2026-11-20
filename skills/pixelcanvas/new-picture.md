# A new picture, coarse to fine

You direct; subagents draw. The figures are posed, dressed and given hair on a 3D skeleton; `render` turns that into
a faded guide that is never exported. The picture itself is drawn in 2D over the guide: line art, flat colour, cel
shadows, faces. A problem is fixed at the level that causes it: a wrong pose is posed again, never painted over.

| level | what is decided | owner | shell calls |
|---|---|---|---|
| 1 brief | shot, camera, place, figures, light, palette; `brief.md` + `steps.txt` | `pixelcanvas-director` | 1-2 |
| 2 guide | camera, scene kit, blocks, posed and dressed figures, the guide | `pixelcanvas-blocker` | 4-6 |
| 3-4 picture | 2D drawing, finish, `final.png` | `pixelcanvas-painter` | 4-8 |
| checks | the guide, then the final picture, against the brief | `pixelcanvas-critic` | 1 each |

## The run: five subagent calls

1. `pixelcanvas-director`: give it the request, the canvas size and the folder. It writes `brief.md` and
   `steps.txt` (camera, `scene`, blocks, `figure` lines) and answers in five lines.
2. `pixelcanvas-blocker`: give it the folder. It runs `steps.txt`, renders and exports `guide.png`.
3. `pixelcanvas-critic`, check `guide`. Issues tagged `blocker` go to the blocker word for word with `SendMessage`
   (it keeps its context); one round.
4. `pixelcanvas-painter`: give it the folder and the brief's focus. It draws, runs `polish`, exports `final.png`
   and runs `stage done`.
5. `pixelcanvas-critic`, check `final`. On issues: one round, each to its owner with `SendMessage` (a blocker fix
   is followed by the painter's `draw <who>; flat auto; shade all`, `polish` and export). No third check.

The critic answers `ok` or at most five issues, each with its level, owner and command. Stop when `final.png` was
exported after the last change and `pixelcanvas status` says FINISHED. Wait for every subagent's answer before you
end your turn. You open one picture yourself: `final.png`.

## Commands a session needs

```bash
pixelcanvas new pic --mode free --size 1000x1400 --request "<request>"
pixelcanvas do @steps.txt              # a file of commands, one per line, run as one batch
pixelcanvas render
pixelcanvas do "draw all; flat auto; shade all"
pixelcanvas polish --focus girl --motion boy:left
pixelcanvas export final.png --stages stages
pixelcanvas stage done                 # status FINISHED
pixelcanvas help pose,figure,scene     # options and values of several commands in one call
```

`pixelcanvas scene corridor --time day --wall "#e6dcc4" --floor "#b07a4a" --lamps 4` draws the place from the
camera under the figures (kits: corridor room classroom street platform outdoor field beach rooftop stairs).

`pixelcanvas figure` sets a whole figure in one line:

```bash
pixelcanvas figure girl --pose run --at 420,900 --heads 6.5 --yaw 30 --set "head: turn 20; arm-l: raise 40" \
  --hand-shape-r fist --wear "blouse:white,skirt:#2c3460:pleated,thigh-highs:black,shoes:brown" \
  --hair "twintails,blunt,#f09ac0,bows:black" --face "smile,#3a78c8"
```

- `--wear`: pieces separated by commas, each `piece[:colour][:style|length|sleeve|fit]`. Pieces: shirt blouse
  blazer jacket hoodie coat skirt dress pants shorts socks thigh-highs shoes boots gloves tie ribbon scarf.
- `--hair`: style (long medium bob short spiky ponytail twintails braid bun buns), bangs (blunt side parted
  none), colour, accessories (hairband hairpin bow ribbon), `bows:<colour>`, `wind:left|right|dx,dy`.
- `--face`: expression (neutral smile open grin laugh shout angry sad crying teary surprised panicked smug
  embarrassed sleepy closed shut wink), eye colour, gaze (center left right up down up-left up-right ...).
- Every other option is a `pose` option: `--pose <preset>` (`pixelcanvas pose list`: 73 presets in groups),
  `--at x,y` (ground point, 0-1000), `--heads` or `--height`, `--yaw`, `--lean --twist --bend <deg>`, `--look
  camera|left|right|up|down|back|at:x,y`, `--energy 0..1`, `--arm-r/-l raised|forward|back|down|bent|out|hip|
  behind|crossed|face|chin|hair`, `--leg-r/-l forward|back|bent|raised|kick|out|tucked|relaxed`, `--hand-shape-r/-l
  relaxed|fist|open|point|grip|peace|cup|spread|flat|reach`, `--hand-r-to hip|chest|x,y|<block>`, `--foot-r-to
  x,y`, `--on <block>`, `--check`, and `--set "<joint>: <motion> <deg>; ..."` for any joint.

## What a good brief holds

- **Shot**: framing, camera height and angle, horizon height on the canvas, lens (wide 35 / normal 50 / long 85).
- **Each figure**: the action in one verb phrase, the preset and the joint edits, yaw, ground point, height,
  contact, hand shapes, clothes, hair, expression, eye colour: the `figure` line says all of it.
- **Scene**: one `scene <kit>` line for the place, blocks for what a figure touches or stands behind.
- **Light** (direction, key and fill colours), **palette** (5-8 named colours), **focus**.
