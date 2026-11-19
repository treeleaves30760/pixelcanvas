# A new picture, coarse to fine

You direct; subagents draw. A new picture (no reference to trace) is built from the biggest decision to the
smallest mark. The figures are posed, dressed and given hair on a 3D skeleton; `render` turns that into a faded
guide that is never exported. The picture itself is drawn in 2D over the guide: line art, flat colour, cel
shadows, faces. Fix every problem at the coarsest level that causes it: a wrong pose is fixed by posing again and
drawing that figure again, never by painting over it.

| level | what is decided | tools | owner |
|---|---|---|---|
| 1 brief | shot, camera, who does what where, clothes, hair, light, palette | text only: `brief.md` | `pixelcanvas-director` |
| 2 pose + guide | camera, blocks, 3D poses, props, clothes, hair, hand shapes; the guide | `camera`, `block`, `pose`, `prop`, `dress`, `hair3d`, `render` | `pixelcanvas-blocker` |
| 3 drawing | 2D line art over the guide, flats, cel shadows, faces | `draw`, `flat auto`, `shade`, `face`, `line redo` | `pixelcanvas-painter` |
| 4 finish | background, eyes, blush, folds, props' details, polish, export | `band`, `eye`, `blush`, `fold`, `polish`, `export` | `pixelcanvas-painter` |

`pixelcanvas-critic` checks the guide after level 2 and the finished picture. It never draws.

## The run

1. Ask `pixelcanvas-director` for the brief. Give it the request and the canvas size. It writes `brief.md` next to
   the project and answers in five lines. Read the summary, not the whole brief.
2. Ask `pixelcanvas-blocker` for level 2 from `brief.md`. It answers with the guide picture path.
3. Ask `pixelcanvas-critic` to check the guide against `brief.md` at level 2 (pose, gesture, contact, scale,
   camera, clothes, hair). Send its issues to the blocker word for word. One round of fixes.
4. Ask `pixelcanvas-painter` for levels 3 and 4 in one call. It draws, finishes, exports `final.png` and runs
   `pixelcanvas stage done`.
5. Critic on the finished picture. If it says `fix`, one round: pose, clothes and hair issues go to the blocker
   first (then the painter runs `draw <who>` for that figure), drawing issues to the painter; then the painter
   exports again.

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
  eye goes first. **Style**: cel (anime pictures are cel).

## Checks that need no eyes

`pixelcanvas pose <who> --show joints` lists every joint; `pixelcanvas status` lists the open todo items; every
command prints scores and `WARNING` lines. A subagent reads these before it opens a picture.
