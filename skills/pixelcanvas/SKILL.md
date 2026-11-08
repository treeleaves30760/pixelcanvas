---
name: pixelcanvas
description: >-
  Draw good-looking images with pure code (numpy + Pillow) when you cannot generate images natively — landscapes,
  skies, night scenes, sea, cities, still lifes, posters, cards, thumbnails, banners, badges/icons, labelled
  diagrams, wallpapers — with exact, correctly spelled text in any language (incl. Chinese/Japanese), and retouch
  photos. Iterative loop built for vision models: write a scene script → render a draft → inspect it with labelled
  views (coordinate guides, zoom, value map, edges, stage-by-stage sheet, parameter sweeps; text reports for
  models without vision) → fix 2-4 things → re-render. Use whenever the user asks to draw, paint, render,
  illustrate, design a poster/graphic, or "generate a picture/wallpaper", or to retouch an image, and no image
  model is available. 觸發詞：畫一張、幫我畫、產生圖片、海報、桌布、插畫、風景圖、設計圖、修圖、調色。
---

# pixelcanvas: drawing with code + maths + looking

No assets, no image model: every pixel comes from noise, masks, signed distances, text rasterisation and blending.
You do **plan → code → render → look → fix**. A good image takes 4~7 rounds; the first version is almost always
over-exposed, flat or too clean. **Never hand over the first render.**

## 0. Setup

1. Find the project root (folder with `pixelcanvas/`, `tools/render.py`, `scenes/`): current directory → env var
   `PIXELCANVAS_HOME` → "Local install location" at the bottom of this file. If none, install:
   ```bash
   git clone https://github.com/treeleaves30760/pixelcanvas.git && cd pixelcanvas
   pip install -r requirements.txt          # numpy, Pillow (scipy optional: faster blur / sampling)
   ```
2. Run every command from the project root. `python -m pytest -q` should pass in a few seconds.

## 1. Plan first (write it as the scene file's docstring)

Answer before coding — this is also the checklist you will compare the render against:
- **Format**: 1920x1080 landscape by default; 1080x1350 poster; 1080x1920 phone; 1280x720 thumbnail.
- **Focal point** and where it sits (U, V fractions, near a third); what is biggest / brightest / most saturated.
- **Eye level / horizon** (V fraction): sky-dominant 0.6~0.75; ground-dominant 0.35~0.5.
- **Light**: position and direction — decides every lit side, shadow and rim light.
- **Far / middle / near** content, or for designs the **text hierarchy** (title → subtitle → info) and margins.
- **Palette**: 3~5 colours + haze colour (tables in [techniques.md](references/techniques.md)).
- **Exact text** to render, copied verbatim from the request.
- **Prompt checklist** — when drawing from a prompt, split it into checkable items before coding: every entity
  (with its count), every attribute (colour, shape, size, material, state), every relation (left of, on top of,
  holding, inside), the **medium / style** ("photo", "oil painting", "watercolour", "pencil sketch", "anime",
  "3D render", "flat icon") and any quoted text. Each item must be visible in the final picture to a stranger who
  never read the prompt. Unusual attributes keep every *other* canonical cue of the object (circular leaves are
  still flat, many, clustered on twigs with veins — not yellow balls that read as fruit; a square apple keeps its
  stem, leaf and apple colouring).

## 2. Create the scene, render a draft

```bash
python tools/new_scene.py my_scene             # landscape template (scenery recipes) → scenes/my_scene.py
python tools/new_scene.py my_poster --poster   # design template (text, SDF shapes, layout)
python tools/render.py my_scene --scale 0.5 --check
#   → outputs/my_scene_x0.5_preview.jpg   the draft
#   → outputs/my_scene_x0.5_check.jpg     guides | values | edges | thumbnail, + automatic warnings printed
```

A scene file provides `render(width, height, seed, log) -> Canvas` and `DEFAULT_SIZE`. Keep tunable numbers as
UPPER_CASE module constants — then `--set` / `--sweep` can change them without editing. Call `log("step name")`
**before** each drawing step — that is what `--stages` snapshots, and **each step becomes one editable layer**
(`--layers`, §5). So give every object a human may want to move, recolour or hide its own step with a clear name
(`log("sun")`, `log("clouds")`, `log("title")`, `log("logo")`), keep post-processing in a last `log("post")` step,
and draw each object through `cv.paint` / `cv.text` / `cv.paste` with its mask (that mask becomes the layer's alpha). **Start from the closest example**:

| To draw | Start from | Shows |
|---|---|---|
| mountains, rivers, forests, clouds, sunsets | `scenes/mountains_sunset.py` | high-level recipes `pixelcanvas.scenery` |
| stars, night, moon | `scenes/night_sky.py` | power-law scatter, domain-warped galaxy, 1D silhouettes |
| aurora, glowing night skies, a wide banner with a title | `scenes/aurora_lake.py` | ray curtains (stretched fBm), lake mirror, pine frame, title over a calmed sky |
| objects, still life, product shots | `scenes/still_life.py` | normals → diffuse / specular / cast shadow / AO |
| cities, buildings, geometry | `scenes/city_dusk.py` | height profiles, grid patterns (windows), layered haze |
| sea, lakes, water | `scenes/sea_sunset.py` | perspective world coords, mirrored reflection, Fresnel, glitter |
| posters, cards, thumbnails, logos, UI, text | `scenes/neon_poster.py` | `cv.text` (fit, outline, glow, CJK), `pixelcanvas.sdf` shapes, layout |
| editing a photo | `Canvas.load("photo.jpg")` | filters + masks, techniques.md "Photo editing" |

**Medium is a requirement, not a mood.** If the prompt names one, finish with `pixelcanvas.stylize` in its own
`log("medium")` step (techniques.md §13), then check that the picture would be called that medium:

| Prompt says | Last step |
|---|---|
| painting, oil, acrylic, impressionist, brushstrokes, "in the style of <painter>" | `cv.apply(stylize.oil_paint, brush=6 * S)` + `stylize.canvas_texture` |
| watercolour, ink wash | `cv.apply(stylize.watercolor)` |
| pencil, sketch, charcoal, drawing, line art | `cv.apply(stylize.pencil_sketch)` |
| anime, manga, cartoon, comic, cel-shaded | flat colour + `cv.apply(stylize.cel_shade, outline=2 * S)` |
| photo, photograph, realistic | no filter: soft lighting, cast shadows, depth of field, `finish()` / bloom + grain |

References: [api.md](references/api.md) (all functions) · [techniques.md](references/techniques.md)
(how to draw X + palettes) · [typography.md](references/typography.md) (text, SDF shapes, layout rules) ·
[review.md](references/review.md) (how to look, checklist, symptom → fix).

## 3. Look (the most important step)

**With vision** — follow the protocol in [review.md](references/review.md#protocol-for-vision-models):

| Question | Tool (writes an image to look at) |
|---|---|
| Overall impression + objective warnings | the preview, then `python tools/look.py critique IMG` (also printed by `--check`) |
| Where exactly is X? | `look.py guides IMG` — labelled fraction grid + thirds. Read coordinates, don't guess |
| Are details / edges / text clean? | `look.py zoom IMG u0 v0 u1 v1` — enlarged crop, labels in full-image coords |
| Does the composition read? | `look.py values IMG` — 4-level squint test; `look.py edges IMG` |
| Which step broke it? | `render.py my_scene --scale 0.5 --stages` → `_stages.jpg`, the canvas after every step |
| Which value is best? | `render.py my_scene --sweep SUN_X 0.4 0.55 0.7` or `--seeds 1 2 3 4` → one labelled sheet |
| Exact colour / contrast? | `look.py probe IMG 0.5,0.3 0.5,0.8` |
| Before vs after | `look.py sheet a.png b.png c.png --labels v1,v2,v3` or `look.py diff A B` |

**Without vision** — `python tools/look.py report IMG` turns the image into text (3x3 colours / luma / contrast /
texture, dominant colours, brightest-region centroid, clipping, vertical & horizontal profiles, strongest boundaries
per column, ASCII luminance map); add `look.py critique IMG`, and `crop` + `report` for regions.

Either way, go through the [review.md](references/review.md) checklist **and tick the prompt checklist item by item**
(is each entity there, countable, the right colour / shape, in the right relation; does the picture read as the
requested medium?), then **write down the 2~4 worst problems with coordinates** before changing anything. Typical first round: blown highlights (> 3%), layers not separated,
marble-like clouds, flat foreground, jagged edges, text crowding the edge.

## 4. Fix → re-render

- Change only 2~4 things per round; keep the seed fixed so renders are comparable.
- Try without editing: `--set NAME=EXPR` (e.g. `--set "LAYERS[0]['fog']=0.6"`); compare options with `--sweep`.
- Back to step 3 until the checklist passes.

## 5. Full size, details, deliver

```bash
python tools/render.py my_scene --check                      # 1920x1080 (or --size 2560x1440)
python tools/look.py zoom outputs/my_scene.png 0.4 0.3 0.6 0.5   # inspect texture, edges, text at full res
python tools/look.py grid outputs/my_scene.png               # 3x3 full-resolution tiles
```
Check fine detail (fur-like texture, hard edges, glitter density, text edges), then copy PNG + preview to
`gallery/` and report: the preview path, what you changed and why. No long description of the image.

**Editable layers** (deliver these whenever a human will touch the picture up, or when asked for layers / PSD):
```bash
python tools/render.py my_scene --layers      # + outputs/my_scene.psd (Photoshop / Photopea / GIMP / Affinity),
                                              #   .ora (Krita / GIMP), my_scene_layers/ (PNG per layer + layers.json)
python tools/layers.py sheet outputs/my_scene_layers     # look at every layer on a checkerboard → check the split
python tools/layers.py compose outputs/my_scene_layers --hide post --out raw.png   # re-composite (after edits)
```
One layer per `log()` step, bottom → top. Painted objects are solid where their mask was (move the mountain and the
sky is still behind it); glows and haze are see-through. The final `post` step is folded into every layer (a fitted
colour LUT + a smooth vignette gain), so grading never leaves a ghost copy of objects; only what that can't express
(grain, global bloom halos) stays in a thin `post (residual)` layer. The stack re-composites to the final image.

When the picture is meant to be edited (or layers are requested):
- **One step per requested layer, named exactly as asked**, in the requested order; nothing drawn outside them except
  `post`. If two things a person would edit separately share a step, split it.
- **Draw every layer complete, including what is hidden behind layers above it**: the whole background under the
  product, the whole title under the player, the road under the car. Never cut holes for objects in front.
- **An object's own light stays with it**: its glow (`cv.glow`), rim light and contact shadow go in its step — unless
  the request lists them as separate effect layers, then give each its own step. Avoid a global `bloom` in `post`
  (it can't fold into layers); grading, `soft_clip`, `vignette` and light `grain` are fine.
- **Check by hiding**: `python tools/layers.py compose outputs/x_layers --hide <object> --out hidden.png` for each
  object → the scene behind must be complete, with no hole and no faint copy of the object.

## Drawing principles (break these and it won't look good)

1. **Mask / density first, colour second**: compute "where is it, how thick" (float32 (H, W) 0~1), then choose lit /
   shadow / haze colours, then `cv.paint(color, mask)`. Colour can be a full (H, W, 3) map.
2. **Far to near**: sky → far → middle → water (reflections sample the current image, so after banks and mountains)
   → foreground → light → text / UI → post-processing.
3. **One light**: every lit side faces the same light vector; low sun → lit cloud bottoms, rim light on silhouettes.
4. **Atmospheric perspective**: farther = paler, closer to haze colour, lower contrast; near = dark, high contrast.
5. **HDR, then compress**: light sources may exceed 1; finish with `bloom` + `soft_clip` (or `finish()`).
   Surfaces that need texture (moon, clouds, text you want readable) should stay ≤ 1.
6. **Resolution independent**: positions as fractions; pixel constants at 1080p × `S = H / 1080` (or `sc.px()`),
   so drafts and full size match.
7. **Anti-aliasing**: silhouettes `clamp01(Y - ridge + 0.5)`; shapes `sdf.fill(d)`; complex shapes `shape_mask`
   (supersampled); text `cv.text` / `text_mask` (supersampled). Never hard thresholds like `d < 0`.
8. **Texture scale**: noise scale too small → fur / noise; too large → plastic. Round things (forest, clouds) plain /
   billow fBm; rock ridged. Different seeds for neighbouring things.
9. **Vectorise**: no per-pixel Python loops; many small objects via `np.add.at` or one Pillow `shape_mask`.
10. **Text is exact**: render the user's words verbatim; omit `font=` for automatic CJK support; fit with `box=`;
    keep ≥ 30% calm space and one margin; verify spelling and edges with `look.py zoom`.

## Local install location

(Filled in by `tools/install_skill.py`; if empty, find or clone the project as in step 0.)
