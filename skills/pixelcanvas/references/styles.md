# Art styles (`pixelcanvas.stylize`)

Forty-three finished looks, one call each: paintings, drawings, prints, crafts and photographic grades. Every style
reads the picture's real edges and regions, so outlines follow the subject, glass is cut along its shapes and
mosaic rows wrap around its contours.
Options are checked: an unknown option name raises a TypeError that names the closest valid one (`tags` -> "did you mean 'tag'?"), and both spellings are accepted (`colour` / `color`, `centre` / `center`, `w` / `width`, `alpha` / `opacity`, ...). Unknown names of styles raise a ValueError with the closest valid names.


```python
from pixelcanvas import Canvas
from pixelcanvas.stylize import apply, pick, names, describe

cv = Canvas(480, 270, "#406080")
cv.paint("#e0a040", cv.uv()[1] > 0.6)                # ... your picture ...
apply(cv, "ukiyoe", seed=1)                         # in place; returns cv
style = pick("a risograph poster of a fox")          # -> "risograph"
```

## The rule

**When a request names a medium or style, it is a requirement.** Draw the picture as usual, then finish with that
style in its own step and look at the result:

```python
from pixelcanvas import Canvas
from pixelcanvas.stylize import apply

cv = Canvas(480, 270, "#406080")                       # ... your picture ...
prompt = "a watercolor of a lighthouse at dusk"
medium = apply(cv, prompt=prompt, seed=1)              # picks the style from the sentence and applies it
assert medium is cv                                    # (a prompt that names no medium returns None, picture untouched)
```

`apply(cv, prompt=...)` understands English, Traditional Chinese and Japanese keywords, so one line finishes every
scene; call `log("medium")` before it to keep the style as its own layer in `--layers` output.

Then check: would someone glancing at the preview call it *that* medium? Name the cues you see (brush marks,
paper, halftone dots, lead lines, stitches...). If the answer is "a filtered render", raise the style's size
parameter (`brush`, `cell`, `tile`, `pixel`, `stitch`) or simplify the picture underneath — big clear shapes make
every medium read better. Everything else in the picture (composition, light, subject) is drawn exactly as before.

Sizes are in 1080p pixels and scale with the image height, so a 960x540 draft and the full-size render match.
Every style is deterministic for a given `seed`.

## The picker

`pick(text)` scans the request for keywords in English, Traditional Chinese and Japanese and returns the most
specific match (the longest keyword wins: "ink wash" beats "ink", "pen and ink" beats "drawing"):

```python
from pixelcanvas.stylize import pick

assert pick("in the style of a ukiyo-e print") == "ukiyoe"
assert pick("a pixel-art sprite") == "pixel_art"
assert pick("水墨畫的山水") == "ink_wash"
assert pick("ステンドグラスの窓") == "stained_glass"
assert pick("a cozy cabin at dusk") is None          # no medium named: keep the house look
```

`python tools/styles.py pick "a linocut of a heron"` prints the same answer from the shell.

Style names are matched ignoring case, hyphens, spaces and underscores, everywhere a name is taken (`apply`,
`options`, `resolve`, the `tools/styles.py` commands): "ukiyo-e", "Ukiyo E", "pixel-art", "water colour" and "Ink Wash"
all land on the canonical name, and a real miss raises an error naming the closest styles. A sticker or layer drawn as
an array keeps the report informed with `apply(array, "sticker", cv=main)`; a styled scratch Canvas that is pasted
brings its styles along.

## Styles

`apply(cv_or_array, name=None, strength=1.0, seed=0, prompt=None, **params)` — params override the defaults
below. `strength` < 1 mixes the original back in. `options("oil")` lists a style's params (`python tools/styles.py options oil` prints them). Options are checked
(an unknown name raises a TypeError that names the closest valid one) and both spellings are accepted
(`watercolour` / `watercolor`, `colours` / `colors`).

### Painting

| style | what it looks like | key parameters |
|---|---|---|
| `oil` | blended colour blocks laid with loaded-brush strokes along the forms, bristle grooves, lit impasto, canvas weave | `brush=7`, `impasto=0.6`, `canvas=0.06` |
| `impressionist` | short thick dabs of broken colour following the forms, warm ground glinting between | `brush=20`, `jitter=0.08`, `ground` |
| `pointillism` | round dots of pure varied colour that mix in the eye, white ground | `dot=8` |
| `watercolor` | transparent washes, pigment pooled at wet edges, granulation, ragged white border | `wash`, `bleed`, `deckle` |
| `gouache` | flat opaque matte shapes, faint brush texture, paper slivers, taped margin | `colors=12`, `gap=1`, `border=0.03` |
| `storybook` | picture-book illustration: soft painterly shading, gentle brown outlines, warm paper tone and tooth (also answers "children's book", "picture book") | `softness=1`, `outline=1`, `warmth=1`, `paper` |
| `ink_wash` | sumi-e: graded ink tones with blooming edges, dry-brush contours, rice paper, red seal | `tones=3`, `dry=1`, `seal=True` |

### Drawing

| style | what it looks like | key parameters |
|---|---|---|
| `pencil` | graphite drawing on white paper: values lifted to the paper, contours, hatching in the shadows | `line`, `hatching` |
| `colored_pencil` | hatched colour strokes on white paper, tooth sparkling through, pencil outlines | `stroke=1`, `pressure=1` |
| `charcoal` | smudged broad darks on toned paper, bold broken contours, chalk-lifted lights | `contrast=1.35`, `smudge=1` |
| `pastel` | powdery strokes on toned paper, the paper's grain showing | `paper="#9a8f7e"`, `coverage=1` |
| `crayon` | bright scribbled wax colour, white paper tooth, waxy outlines | `colors=14`, `pressure=1` |
| `pen_ink` | contour lines + cross-hatching in tonal tiers, black ink on white | `spacing=7`, `line=1` |
| `chalkboard` | chalk lines and side-of-chalk shading on slate, wooden frame | `color=0.6`, `frame=True` |
| `blueprint` | white technical lines on Prussian blue, drafting grid, frame, title block | `grid_size=24` |

### Comics and animation

| style | what it looks like | key parameters |
|---|---|---|
| `comic` | bold black outlines, flat saturated fills, Ben-Day dots; highlights kept clean white for speech bubbles | `colors=8`, `dots=9`, `whites=True` |
| `manga` | black-and-white ink lines, cut screentone, solid blacks | `tone=6`, `outline=1` |
| `cel` | anime soft-cel: base colours with a crisp cool shadow band and a highlight band, thin colour outlines; to draw anime freehand use `sk.cel` / `sk.sheen` / `sk.eye` ([techniques.md](techniques.md), [reference.md](reference.md)) | `colors=12`, `shadow="#6e6290"` |

### Prints

| style | what it looks like | key parameters |
|---|---|---|
| `ukiyoe` | woodblock: flat traditional pigments, Prussian-blue bokashi sky, key-lines, washi, title cartouche | `colors=9`, `bokashi=1`, `cartouche="富嶽景"` |
| `risograph` | 2–3 fluorescent spot inks, halftone grain, misregistered overprint on off-white | `inks=("#ff48b0", "#0078bf")`, `misregister=1` |
| `linocut` | carved two-tone: black masses, white gouge lines following the forms, rough edges | `threshold=0.5`, `gouge=1`, `color=None` |
| `engraving` | banknote line work: parallel lines that swell with darkness and bow over the forms | `spacing=8`, `angle=-28`, `ink` |
| `cyanotype` | Prussian-blue sun print with brushed emulsion edges | `contrast=1.2` |
| `duotone` | values mapped onto two (or three, `mid=`) ink colours | `dark`, `light`, `mid` |

### Crafts and geometry

| style | what it looks like | key parameters |
|---|---|---|
| `stained_glass` | glowing glass pieces cut along the picture's regions, dark lead cames | `cell=40`, `lead=1`, `glow=1` |
| `mosaic` | tesserae laid in rows that wrap around the outlines, bevelled tiles, grout | `tile=17`, `grout` |
| `low_poly` | flat-coloured triangles whose corners snap to real edges | `cell=46`, `facet=0.06` |
| `pixel_art` | crisp coarse pixels, a small palette that keeps every highlight, ordered dither, sprite outlines | `pixel=12`, `colors=16`, `dither=0.35`, `outline=1`, `palette="pico8"` |
| `paper_cut` | each colour region a piece of card, stacked in depth, knife-cut edges, soft cast shadows | `layers=6`, `shadow=1` |
| `collage` | torn paper pieces (halftone print, newsprint, plain stock), white fibrous edges, kraft card | `pieces=7`, `offset=1` |
| `cross_stitch` | X stitches of twisted thread on Aida cloth, pale cells left bare | `stitch=15`, `colors=20` |
| `clay` | claymation: plasticine lumps cut by colour, rounded seams, soft form light, waxy sheen, fingerprints | `colors=10`, `softness=1` |
| `sticker` | die-cut white border and drop shadow around `mask=` (one mask or a list); the rest of the picture stays (whole picture as one sticker if no mask) | `mask`, `border=14`, `background` |

### Photographic looks

| style | what it looks like | key parameters |
|---|---|---|
| `photo_finish` | the last step for realistic renders: lens blur by `depth=`, chromatic aberration, bloom, grain, vignette | `depth`, `focus`, `aperture` |
| `vintage` | faded film: lifted blacks, warm cast, halation, light leak, grain, dust | `fade`, `leak`, `grain` |
| `sepia` | daguerreotype plate: toned monochrome, soft focus, tarnished rim, scratches | `tone="sepia" / "silver"` |
| `noir` | hard black and white, crushed blacks, venetian-blind light, grain | `blinds=0.5`, `contrast=1.5` |
| `tilt_shift` | miniature: sharp band, graded blur, toy colour | `focus=0.62`, `band=0.1`, `blur=12` |
| `synthwave` | neon grade: indigo-magenta-orange map, glowing edges, bloom, scanlines | `glow=1` |
| `vaporwave` | pastel pink-teal map, VHS chroma bleed, RGB split, scanlines | |
| `glitch` | displaced slices, RGB split, corrupted blocks, pixel sorting | `amount=1` |
| `thermal` | ironbow heat palette, sensor softness, crosshair and scale bar | `hud=True` |
| `infrared` | Aerochrome false colour: pink-red foliage, deep blue sky | `glow=1` |

## Recipes

**Try several looks at once.** The contact sheet shows the picture in every style (or a chosen few) with timings:

```
python tools/styles.py sheet scenes/my_scene --styles oil,gouache,ukiyoe,linocut --cols 4
python tools/styles.py sheet photo.png                      # all styles
python tools/styles.py apply photo.png stained_glass --set cell=60 --out glass.png
python tools/styles.py list                                 # every style with one line
python tools/styles.py options ukiyoe                       # a style's options: default, range, meaning
```

**Sticker of a subject.** Pass the subject's mask; the border and shadow grow around it and everything else in the
picture stays as it is, so a sticker cut is safe to apply to part of a scene (it is recorded in `cv.styles_applied`):

```python
import numpy as np
from pixelcanvas import Canvas, sdf
from pixelcanvas.stylize import apply

cv = Canvas(480, 270, "#d8e4ec")
X, Y = cv.xy()
cat = sdf.fill(sdf.sd_circle(X, Y, 240, 140, 70))
cv.paint("#f08a3c", cat)
apply(cv, "sticker", mask=cat, border=20)
```

**A sheet of stickers.** Draw each subject, then die-cut it: one `apply` per subject, or one call with a list of
masks. Add `background="#e8e4dc"` to replace everything outside the cuts with a plain sheet instead.

```python
from pixelcanvas import Canvas, sdf
from pixelcanvas.stylize import apply

cv = Canvas(480, 270, "#f6e3e8")
X, Y = cv.xy()
cuts = []
for i, (cx, cy, r, col) in enumerate([(100, 80, 44, "#f08a3c"), (240, 80, 44, "#5aa469"), (380, 80, 44, "#4a78c2"),
                                      (100, 190, 44, "#c25a8a"), (240, 190, 44, "#e0c040"), (380, 190, 44, "#7a5ac2")]):
    m = sdf.fill(sdf.sd_circle(X, Y, cx, cy, r))
    cv.paint(col, m)                                  # the subject
    cuts.append(m)
apply(cv, "sticker", mask=cuts, border=24, shadow=0.8)      # all six at once; or apply(cv, "sticker", mask=m) in the loop
```

**Photographic finish with depth.** Give a depth map (0 near, 1 far) and the distance to keep sharp:

```python
import numpy as np
from pixelcanvas import Canvas
from pixelcanvas.stylize import apply

cv = Canvas(480, 270, "#7a8fa6")
depth = np.tile(np.linspace(1.0, 0.0, 270, dtype=np.float32)[:, None], (1, 480))   # far at the top
apply(cv, "photo_finish", depth=depth, focus=0.3, aperture=1.5)
```

**Pixel-art sprite scene in a console palette.** Draw big simple shapes, then pick the grid and the palette:

```python
from pixelcanvas import Canvas, sdf
from pixelcanvas.stylize import apply, PALETTES

cv = Canvas(512, 512, "#2b2f4a")
X, Y = cv.xy()
cv.paint("#d23a2e", sdf.fill(sdf.sd_circle(X, Y, 256, 260, 90)))
apply(cv, "pixel_art", pixel=16)                         # palette taken from the picture, 16 colours
apply(cv.copy(), "pixel_art", palette="pico8")          # or a preset: pico8, gameboy, nes, c64
print(sorted(PALETTES))
```

`palette=` also takes your own list of colours, e.g. `palette=["#0f380f", "#306230", "#8bac0f", "#9bbc0f"]`.

**Custom inks and a second print colour.**

```python
from pixelcanvas import Canvas
from pixelcanvas.stylize import apply

cv = Canvas(320, 180, "#5080b0")
apply(cv.copy(), "risograph", inks=("#ff6c2f", "#3255a4", "#ffe800"))   # orange, blue, yellow
apply(cv.copy(), "linocut", color="#c8432b")                            # black key block + red block
apply(cv.copy(), "duotone", dark="#0b1d3a", light="#ffd166", mid="#ef476f")
```

**Use a style function directly** — every style is a plain array function, handy inside `cv.apply`:

```python
from pixelcanvas import Canvas
from pixelcanvas import stylize

cv = Canvas(320, 180, "#5080b0")
cv.apply(stylize.engraving, spacing=10, ink="#1d3b2a")    # banknote green
print(stylize.describe("engraving"))
```
