# A new picture, coarse to fine

You direct; subagents draw. A new picture (no reference to trace) is built from the biggest decision to the
smallest mark. The figures live in 3D: posed, dressed and with hair on a skeleton, then rendered as line art, flat
colours and cel shading. Fix every problem at the coarsest level that causes it: a wrong pose is fixed by posing
again (the picture renders again by itself), never by painting over it.

| level | what is decided | tools | owner |
|---|---|---|---|
| 1 brief | shot, camera, who does what where, clothes, hair, light, palette | text only: `brief.md` | `pixelcanvas-director` |
| 2 blockout | camera, figures posed in 3D, scene blocks, props | `camera`, `pose`, `block`, `prop` | `pixelcanvas-blocker` |
| 3 dressed render | clothes, hair, hands, faces; line, flat and cel shading in one step | `dress`, `hair3d`, `pose --hand-shape-*`, `render`, `face` | `pixelcanvas-blocker` |
| 4 scene + finish | background bands, eyes, blush, folds, props' details, polish, export | `band`, `eye`, `blush`, `fold`, `polish`, `export` | `pixelcanvas-painter` |

`pixelcanvas-critic` checks after level 2 and on the finished picture. It never draws.

## The run

1. Ask `pixelcanvas-director` for the brief. Give it the request and the canvas size. It writes `brief.md` next to
   the project and answers in five lines. Read the summary, not the whole brief.
2. Ask `pixelcanvas-blocker` for levels 2 and 3 from `brief.md`: project, camera, blocks, poses, props, then
   dress, hair, render and faces. It answers with the picture path and what it could not do.
3. Ask `pixelcanvas-critic` to check that picture against `brief.md` at level 3 (pose, gesture, contact, scale,
   camera, clothes, hair, faces). Send its issues to the blocker word for word. One round of fixes.
4. Ask `pixelcanvas-painter` for level 4. It finishes, exports `final.png` and runs `pixelcanvas stage done`.
5. Critic on the finished picture. If it says `fix`, one round: pose, clothes, hair and face issues go to the
   blocker (it re-poses; the render follows), the rest to the painter; then the painter exports again.

Keep your own context small: you do not open views yourself except the final picture. Pass each subagent the
project folder, the level it owns and the critic's issues for that level, word for word. Four to six subagent
calls make a picture; do not add more.

## What a good brief holds

- **Shot**: framing (full body, knees up, bust), camera height and angle (eye level, low, high), horizon height on
  the canvas, lens (wide 35 / normal 50 / long 85).
- **Each figure**: the action in one verb phrase, the pose preset nearest to it and the joint edits that make it
  this pose, facing (yaw), ground point (x, y on the canvas), height, contact with the scene, hand shapes, what it
  holds, the garments by `dress list` names with colours, the `hair3d` style, the expression and eye colour.
- **Scene**: the blocks a figure touches or stands behind, sizes in metres; the far shapes as bands back to front.
- **Light**: direction, time of day, key and fill colours. **Palette**: 5-8 named colours. **Focus**: where the
  eye goes first.

## Checks that need no eyes

`pixelcanvas pose <who> --show joints` lists every joint; `pixelcanvas status` lists the open todo items; every
command prints scores and `WARNING` lines. A subagent reads these before it opens a picture.
