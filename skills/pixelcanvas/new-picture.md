# A new picture, coarse to fine

You direct; subagents draw. The figures are posed, dressed and given hair on a 3D skeleton; `render` turns that into
a faded guide that is never exported. The picture itself is drawn in 2D over the guide: line art, flat colour, cel
shadows, faces. A problem is fixed at the level that causes it: a wrong pose is posed again, never painted over.

| level | what is decided | owner | shell calls |
|---|---|---|---|
| 1 brief | shot, camera, figures, scene, light, palette; `brief.md` + `steps.txt` | `pixelcanvas-director` | 1-2 |
| 2 guide | camera, blocks, posed and dressed figures, the guide | `pixelcanvas-blocker` | 3-6 |
| 3-4 picture | 2D drawing, finish, `final.png` | `pixelcanvas-painter` | 4-8 |
| check | the finished picture against the brief | `pixelcanvas-critic` | 2-3 |

## The run: four subagent calls

1. `pixelcanvas-director`: give it the request, the canvas size and the folder. It writes `brief.md` and
   `steps.txt` (the level-2 commands) and answers in five lines.
2. `pixelcanvas-blocker`: give it the folder. It runs `steps.txt`, renders the guide, checks it once against the
   brief and answers with the guide path.
3. `pixelcanvas-painter`: give it the folder. It draws, finishes, exports `final.png` and runs `stage done`.
4. `pixelcanvas-critic`: give it the folder. It answers `verdict: pass` or up to five issues.

On `verdict: fix`: one round. Send pose, clothes and hair issues to the blocker, drawing issues to the painter, word
for word, with `SendMessage` to the agent that already ran (it keeps its context); the painter then redraws what
changed (`draw <who>`, `flat auto`, `shade all`) and exports again. No second critic round.

Stop when `final.png` was exported after the last change and `pixelcanvas status` says FINISHED. Wait for every
subagent's answer before you end your turn. You open one picture yourself: `final.png`.

## Commands a session needs

```bash
pixelcanvas new pic --mode free --size 1000x1400 --request "<request>"
pixelcanvas do @steps.txt              # a file of commands, one per line, run as one batch
pixelcanvas do "draw all; flat auto; shade all"
pixelcanvas help pose,figure,block     # options and values of several commands in one call
```

`pixelcanvas figure` sets a whole figure in one line:

```bash
pixelcanvas figure girl --pose run --at 420,900 --heads 6.5 --yaw 30 --set "head: turn 20; arm-l: raise 40" \
  --hand-shape-r fist --wear "blouse:white,skirt:#2c3460:pleated,thigh-highs:black,shoes:brown" \
  --hair "twintails,blunt,#f09ac0,bows:black" --face "smile,#3a78c8"
```

- `--wear`: pieces separated by commas, each `piece[:colour][:style|length|sleeve|fit]`. Pieces: shirt blouse
  blazer jacket hoodie coat skirt dress pants shorts socks thigh-highs shoes boots gloves tie ribbon scarf.
- `--hair`: style (long medium bob short spiky ponytail twintails braid bun), bangs (blunt side parted none),
  colour, `bows:<colour>`, `wind:left|right|dx,dy`.
- `--face`: expression (neutral smile open grin angry sad crying surprised panicked sleepy), eye colour.
- The pose options: `--pose <preset>` (`pixelcanvas pose list`), `--at x,y` (ground point, 0-1000), `--heads` or
  `--height`, `--yaw`, `--set "<joint>: <motion> <deg>; ..."`, `--on <block>`, `--hand-r/-l <block|x,y>`,
  `--foot-r/-l <block|x,y>`, `--hand-shape-r/-l relaxed|fist|open|point|grip|peace|cup`.
- Joints: body pelvis spine chest neck head arm-r/-l elbow-r/-l hand-r/-l leg-r/-l knee-r/-l foot-r/-l. Motions:
  arm raise|forward|twist, elbow bend, leg forward|raise|twist, knee bend, spine and chest bend|side|twist, head
  turn|tilt|nod, body lean.

## What a good brief holds

- **Shot**: framing, camera height and angle, horizon height on the canvas, lens (wide 35 / normal 50 / long 85).
- **Each figure**: the action in one verb phrase, the preset and the joint edits, yaw, ground point, height,
  contact, hand shapes, clothes, hair, expression, eye colour: the `figure` line says all of it.
- **Scene**: blocks: the floor, the walls, what a figure touches or stands behind; `draw all` draws them in 2D.
- **Light** (direction, key and fill colours), **palette** (5-8 named colours), **focus**.
