# A new picture, coarse to fine

You direct; subagents draw. A new picture (no reference to trace) is built in six levels, from the biggest decision
to the smallest mark. Each level has one owner, one check and a clear "done". Fix every problem at the coarsest
level that causes it: a wrong pose is fixed by posing again, never by painting over it.

| level | what is decided | tools | owner |
|---|---|---|---|
| 1 brief | shot, camera, who does what where, light, palette, style | text only: `brief.md` | `pixelcanvas-director` |
| 2 blockout | camera, figures posed in 3D, scene blocks, props | `camera`, `pose`, `block`, `prop` | `pixelcanvas-blocker` |
| 3 construction | heads, faces, hands, clothes and hair on the posed figures; scene shapes | `sketch head`, `face`, `hand`, `wear`, `hair`, `band`, `sketch curve` | `pixelcanvas-painter` |
| 4 line + flat | clean line art and every area filled | `line`, `line gaps`, `flat auto`, `flat`, `band` | `pixelcanvas-painter` |
| 5 light | form shading from the 3D figures, cast shadows, rim, depth | `shade`, `shadow`, `rim`, `haze`, `form` | `pixelcanvas-painter` |
| 6 detail + finish | eyes, hair strands, folds, materials, text, polish, export | `eye`, `hair --flow`, `fold`, `surface`, `polish`, `export` | `pixelcanvas-painter` |

`pixelcanvas-critic` checks the picture after levels 2, 4 and 6 (and whenever you are unsure). It never draws.

## The run

1. Ask `pixelcanvas-director` for the brief. Give it the request and the canvas size. It writes `brief.md` next to
   the project and answers with a five-line summary. Read the summary, not the whole brief.
2. Ask `pixelcanvas-blocker` to build levels 1-2 from `brief.md`: it creates the project, sets the camera, poses
   every figure, places blocks and props, and answers with the blockout picture path and what it could not do.
3. Ask `pixelcanvas-critic` to check the blockout against `brief.md`, level 2 only: gesture, silhouette, scale,
   contact with the scene, camera. Each issue it returns names a level and a command. Send level-2 issues back to
   the blocker. Two rounds at most; then move on.
4. Ask `pixelcanvas-painter` for level 3, then level 4 (one call per level). It follows `NEXT:` and the todo
   list and answers with the stage it reached and the picture path.
5. Critic on level 4: wrong silhouette or proportion goes back to level 2 or 3; gaps and stray lines stay at 4.
6. Painter: level 5, then level 6. Critic on the finished picture; one round of fixes; the painter exports
   `final.png` and runs `pixelcanvas stage done` in the finish stage.

Keep your own context small: you do not open views yourself except the final picture. Subagent answers are short
on purpose. Pass each subagent the project folder, the level it owns and the critic's issues for that level,
word for word.

## What a good brief holds

- **Shot**: framing (full body, knees up, bust), camera height and angle (eye level, low, high, dutch), horizon
  height on the canvas, lens (wide 35 / normal 50 / long 85).
- **Each figure**: role, the action in one verb phrase, the pose preset nearest to it and the joint edits that
  make it this pose, facing (yaw), where it stands on the ground (x, y on the canvas), height, contact with the
  scene (sits on the ledge, left hand on the railing), what it holds, expression and gaze.
- **Scene**: the blocks a figure touches or stands behind (bench, wall, stairs, railing, pole) with sizes in
  metres; the far shapes (sky, buildings, trees, sea) as bands back to front.
- **Light**: direction, time of day, key and fill colours, where cast shadows fall.
- **Palette and style**: 5-8 named colours, cel or painterly, line weight.
- **Focus**: where the eye goes first (face, hands, the held object).

## Checks that need no eyes

`pixelcanvas pose <who> --show joints` lists every joint with its position; `pixelcanvas status` lists the open
todo items; every command prints scores and `WARNING` lines. A subagent reads these before it opens a picture.
