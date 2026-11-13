# Illustration: design the picture, draw it freehand

Characters, animals, a named art style, poster art, a storybook scene: design it as an illustrator would and draw
it from shapes. `Sketch` gives shapes in canvas fractions, local frames for characters, smooth blobs and tapered
strokes, one-call shading, outlines, shadows, stickers and washes. `cv.draw()` gives a supersampled Pillow
ImageDraw layer when you would rather draw the way you already know. Both feed layers and `--check`.

```python
from pixelcanvas import Canvas
from pixelcanvas.illustrate import Sketch, paper, grade
cv = Canvas(960, 540, "#f3e3c3")                       # positions: x of the width, y of the height
sk = Sketch(cv, light=(-0.6, -0.8))                    # lengths: fractions of the height
head = sk.ellipse(0.5, 0.42, 0.16, 0.15)               # every shape is a float mask (H, W)
sk.paint(head, "#f6c39a", shade=0.4, highlight=0.3, line=True, shadow=True, name="head")
sk.spot(0.44, 0.48, 0.035, 0.022, "#ff8a8a", 0.7, clip=head)              # blush, clipped to the face
sk.paint(sk.stroke([(0.3, 0.8), (0.5, 0.7), (0.7, 0.8)], 0.03, taper="both"), "#3d3350")
with cv.draw(ss=2, name="sun", highlight=0.4, glow="#ffcf70") as d:      # Pillow, same picture
    d.ellipse([*d.p(0.8, 0.1), *d.p(0.9, 0.28)], fill="#ffd36e")
paper(cv); grade(cv, "golden", 0.6)
```

## Art direction first

1. **Pick a mood and light for it**: `grade(cv, "warm night")` (also golden, cosy, cool day, moody, print) as the
   last step, `paper(cv)` for print grain; one light direction in `Sketch(cv, light=...)` drives every shade.
2. **Light sources glow**: `cv.glow(x, y, sigma, "#ffcf70", 0.5)` at a lamp, a moon, a window;
   `sk.paint(m, c, glow="#ffd27a")` for a shape that shines.
3. **Fill the frame**: the subject takes 40–70 % of the height; `python tools/look.py guides IMG` and the OBJECTS
   size column show it. Background and props fill the rest: shelves, a floor, a sky.
4. **Lock the palette**: 4–6 colours plus one saturated accent; outlines are a darker tone of the fill
   (`line=True`); near-black ink (`line="ink"`) belongs to comics. `limit_palette(cv, colors, 0.5)` pulls stray colours in.
5. **Dark ground, bright subject** (or the reverse): a saturated accent against a deep ground reads at thumbnail
   size; `cv.vignette(0.3)` holds the eye in the middle.

## Things a stranger can name

When the picture lists objects, every one must be nameable at thumbnail size by someone who never read the prompt.

1. **One flat colour per object, a clear value step to the ground**: shading on small things muddies their hue,
   so `Sketch` keeps shapes under 3 % of the canvas flat (`finish="auto"`); `finish="full"` models everything.
2. **Draw the part that names it**: a plain circle names nothing; a pear needs its narrow top and stalk.
3. **Big and apart**: each object at least a tenth of the height, side by side with gaps (`sk.row`, `sk.grid`);
   a dominant subject in front of small things hides them.
4. **See-through is a whole shape**: `sk.glass(m)` plus `sk.level(m, 0.6)` for liquid; dashes read as nothing.
5. **State sits on the object**: `sk.crack`, `shards`, `spill`, `steam`, `drops`, `stain`; `sk.label` look-alikes.

| Object | Parts that name it | Object | Parts that name it |
|---|---|---|---|
| teapot | round body, curved spout, loop handle, knob on top | hourglass | two see-through bulbs meeting at a waist, sand in the lower one |
| padlock | boxy body, U-shaped shackle on top, keyhole | whisk | straight handle, a teardrop of thin crossing loops |
| pear | narrow top, wide bottom, short stalk | snail | spiral on its back, soft body, two eye stalks |
| drawing compass | two legs hinged at a top knob, a point and a pencil tip | bell | flared dome, wide rim, clapper hanging below |
| wrench | long flat shaft, open jaw at one end | mitten | rounded hand shape, thumb lobe, wide band at the wrist |
| rubber duck | round body, smaller head, flat orange beak, one eye | fishbowl | see-through globe (`sk.glass`), a level line (`sk.level`), a goldfish |
| acorn | smooth nut, scaly cap, stub on top | accordion | pleated bellows between two end blocks with buttons |
| flask | see-through round bottom, long thin top, liquid level |  |  |

```python
from pixelcanvas import Canvas
from pixelcanvas.illustrate import Sketch
cv = Canvas(768, 432, "#d8d2c6")
sk = Sketch(cv)                                                   # small shapes stay flat
f = sk.row(3, box=(0.05, 0.2, 0.95, 0.8))                         # [(x, y, size), ...] side by side with gaps
with sk.at(*f[0]):                                                # fishbowl: draw in -0.5..0.5
    bowl = sk.circle(0, 0.05, 0.4); sk.level(bowl, 0.7, "#5ea6e6")
    sk.paint(sk.ellipse(0.02, 0.15, 0.1, 0.06), "#f08a24", line=True); sk.glass(bowl, name="fishbowl")
with sk.at(*f[1]):                                                # cracked egg
    sk.crack(sk.paint(sk.ellipse(0, 0.05, 0.28, 0.38), "#f3ead8", line=True, name="egg"), start=(0, -0.3))
with sk.at(*f[2]):                                                # kettle on the boil
    sk.paint(sk.stroke([(-0.28, 0.2), (-0.48, -0.08)], [0.1, 0.05]), "#3f6fb0", line=True)
    sk.paint(sk.blob([(-0.3, 0.42), (-0.32, -0.05), (0, -0.18), (0.32, -0.05), (0.3, 0.42)], 0.3), "#3f6fb0", line=True, name="kettle")
    sk.paint(sk.arc(0, -0.12, 0.2, 200, 340, 0.04), "#2b2b33")
    sk.steam((-0.5, -0.12), size=0.4)
sk.shards((0.5, 0.9), size=0.08); sk.spill((0.8, 0.9), size=0.08, on=0.9)
```

## Build the style, never filter it

| Style | How to build it |
|---|---|
| woodblock | flat colour areas, `line=INK` on every shape, `sk.pattern(sea, "seigaiha", ...)`, pale stripes following a form (offset copies of `sk.curve`, multiplied by the shape's mask), a cartouche with `vertical=True` text, `limit_palette`, a 1 px shifted red channel, `paper` |
| pixel art | draw on a small `Canvas` (e.g. 1/6 size) with hard shapes, then `small.resize(W, H, "nearest")` and `cv.paste(small)` |
| paper cut | far-to-near silhouettes, each `sk.paint(m, c, shadow=dict(offset=(0.004, 0.008), blur=0.01, opacity=0.45), highlight=0.3)` |
| watercolour | `sk.wash(m, c)` light washes first, darker on top; edges pool, pigment granulates; `paper(cv, blotch=0.04)` |
| comic | `Sketch(cv, finish="flat")`, `line="ink"` on every shape, flat bright fills, `sk.pattern(m, "dots", ...)` halftone, a jagged burst behind a sound word (recipe below) |
| anime / soft-cel | defined bands, never flat fill: `sk.cel(m, c, bands=2, line=True)` on skin, hair, cloth; `sk.sheen(hair)`, `sk.eye(...)`, `sk.blush`, `sk.ink` coloured lines, `sk.ramp` sky ([techniques.md](techniques.md) §13); with a reference image, its numbers first ([reference.md](reference.md)) |
| storybook | soft `shade` and `highlight` on every part, `line=True`, cheeks with `spot`, `grade(cv, "cosy")` |

## Calls

| Need | Call (positions in the current frame) |
|---|---|
| shape | `sk.circle(x, y, r)` · `ellipse(x, y, rx, ry, angle)` · `box(x0, y0, x1, y1, radius)` · `polygon(pts)` · `blob(pts, tension=0)` closed smooth shape through points |
| line | `sk.stroke(pts, width, taper=None, smooth=True, cap="round")`: width per point for a variable line; `taper="end"/"start"/"both"` · `sk.arc(x, y, r, start, end, width)` · `sk.curve(pts)` dense points to join into a `polygon` |
| fill | `sk.paint(m, color)`; gradient: `color=[(0, c0), (1, c1)]`, `angle=90` (degrees) or `"radial"` |
| shade | `paint(..., shade=0.4, highlight=0.3)`: form shadow away from the light, rim light towards it · `volume=0.6`: a rounded body that keeps whites white (cloth, clouds, snow), on a `polygon` face a flat plane darker away from the light · `spec=0.8`: a glossy hot spot on the lit side · `sk.spot(x, y, rx, ry, color, opacity, clip=m)` soft oval · soft-cel: `sk.cel(m, c, bands=1, shift=(-0.12, None), highlight=, rim=, bounce=, line=)`, `cel_tone(c, light, hue)`, `sk.ramp(m, colors, steps=0)` |
| pattern | `sk.pattern(m, "seigaiha" / "scales" / "stripes" / "dots" / "speckle" / "checker" / "tiles", color, size=0.05, width=None, angle=0)`: a motif clipped to the shape; stripes run along `angle`, speckle = soil, sand, stone |
| see-through | `sk.glass(m, tint="#cfe8ff", alpha=0.22, line="#5b7a99")` · `sk.level(m, frac=0.5, color)` liquid inside |
| state | `sk.crack(m, start=None, branches=2)` · `shards(at, size, glass=True)` · `spill(at, color, size, on=y)` · `steam(at, size, n=3)` · `drops(at, n=5, spread)` · `stain(m, color, amount)` rust, mud, dust |
| layout | `sk.row(n, box, gap)` · `sk.grid(n, box, cols=None, gap)`: frames `(x, y, size)` for `with sk.at(*f):`, draw in -0.5..0.5 |
| outline | `paint(..., line=True or colour or "ink", line_width=0.005)`: `"ink"` is a bold near-black comic line · `sk.outline(m, width, where="outside")` · `sk.grow(m, by)` |
| shadow, glow | `paint(..., shadow=True or dict(offset, blur, opacity, color), glow=colour or dict(color, radius, strength))` · `shadow="ground"`: a contact shadow on the floor under the shape's bottom edge |
| arrow | `sk.arrow(p0, p1, via=None, width=0.006, head=None, style="filled" / "open", color, outline=None, both=False)`: straight, or curved through `via`; `outline` makes it read on any ground |
| label | `sk.label("Answer: ice melts", at=(0.5, 0.1), to=(0.5, 0.45), box="pill", arrow=False)`: a caption on a contrasting plate with a leader line, kept inside the canvas, recorded as text |
| text | `sk.text(s, x, y, size, color, anchor="mm", tracking=0.2, outline=3, shadow=True, font="serif", case="upper", vertical=True)` |
| clip, combine | masks are arrays: `a * b` clip, `np.maximum(a, b)` union, `a * (1 - b)` cut |
| repeat, mirror | `with sk.at(x, y, size, rotate=deg, mirror=True):` then local coordinates; loop it for rows, pairs, crowds |
| group | `with sk.layer(name="cat") as g:` → `g.mask` · `with sk.sticker(...):` · `paint(..., name="cat", kind="creature")` records it |
| Pillow / cairo | `with cv.draw(ss=2, name=..., shade=, highlight=, volume=, spec=, line=, shadow=, glow=) as d:` `d.p(u, v)` `d.k(len)` `d.font(size)` · `with cv.cairo() as d: d.ctx` |
| finish | `Sketch(cv, finish="auto" / "full" / "flat")` modelling by size · `paper(cv)` · `grade(cv, mood)` · `limit_palette(cv, colors, amount)` · `cv.vignette(0.3)` |

## Characters that match

Design one character in its own frame and reuse the parts: the same head ellipse, the same eye function, the same
outline colour and width. Head at (0, -0.12), body below, feet at y 0.62; place it with `sk.at`.

```python
import numpy as np
from pixelcanvas import Canvas
from pixelcanvas.illustrate import Sketch
INK = "#3d3350"
cv = Canvas(720, 300, "#eaf2ff")
sk = Sketch(cv)
def eyes(y=-0.06, gap=0.16):
    for s in (-1, 1):
        sk.paint(sk.circle(s * gap, y, 0.055), INK)
        sk.paint(sk.circle(s * gap - 0.017, y - 0.02, 0.02), "#ffffff")
        sk.spot(s * (gap + 0.09), y + 0.11, 0.075, 0.045, "#ff8fa8", 0.7)
def critter(color, ears):
    for s in (-1, 1):
        sk.paint(sk.ellipse(s * 0.17, 0.62, 0.11, 0.06), color, shade=0.3)
        sk.paint(sk.blob([(s * 0.1, -0.4), (s * 0.36, -0.72), (s * 0.42, -0.3)], tension=0.4) if ears == "pointed"
                 else sk.ellipse(s * 0.3, -0.42, 0.12, 0.12), color, shade=0.3)
    sk.paint(sk.ellipse(0, 0.38, 0.32, 0.27), color, shade=0.32, highlight=0.35)
    sk.paint(sk.ellipse(0, -0.12, 0.46, 0.38), color, shade=0.32, highlight=0.35)
    eyes()

for i, (color, ears) in enumerate([("#f4943f", "pointed"), ("#aeb7cc", "pointed"), ("#e8b57e", "round")]):
    with sk.at(0.2 + 0.3 * i, 0.47, 0.32, rotate=(i - 1) * 4):
        with sk.sticker(border=0.1, ink=INK, ink_width=0.025, name="critter %d" % i, kind="creature"):
            critter(color, ears)
```

People: same frame idea, arms `stroke`s with a width per point, `sk.at(0, 0, 1, mirror=True)` for the second eye.

## Check your work

`python tools/render.py SCENE --scale 0.5 --check`, open the preview, zoom into every face, hand and edge
(`python tools/look.py zoom IMG 0.3 0.2 0.7 0.5`). Read OBJECTS (sizes, what hides what), `object_tiny`,
`subject_cropped`, `blown_highlights` (`cv.soft_clip(0.9)`), TEXT and PROMPT. Record each requested subject once
with `name=` (or `manifest.note(cv, "object", "sail", mask=m)`), a hand-built style with `manifest.note(cv, "style", "woodblock")`.

## Recipes

**Objects with counts, colours and positions** ("three green pears left of a red teapot, a lemon above it"):
exact counts, a plain background, no overlaps, every object recorded with `name=` so `--check` counts them and
reports left / right / above / below.

```python
from pixelcanvas import Canvas, manifest
from pixelcanvas.illustrate import Sketch
cv = Canvas(512, 512, "#f4f1ea")
sk = Sketch(cv)
for x, y in [(0.12, 0.6), (0.34, 0.6), (0.23, 0.82)]:                        # three pears, left half
    with sk.at(x, y, 0.09):
        sk.paint(sk.blob([(-0.35, -0.6), (0.35, -0.6), (0.85, 0.4), (0, 0.9), (-0.85, 0.4)]), "#8cbf3f", line=True, name="pear")
        sk.paint(sk.stroke([(0, -0.5), (0.1, -0.85)], 0.08), "#5a3a1a")
with sk.at(0.72, 0.7, 0.18):                                                  # one red teapot, right
    sk.paint(sk.stroke([(0.5, 0.1), (0.95, -0.4)], [0.2, 0.1]), "#d0402c", line=True)
    sk.paint(sk.arc(-0.6, 0.05, 0.3, 90, 270, 0.1), "#d0402c", line=True); sk.paint(sk.circle(0, -0.45, 0.1), "#8a2a1c")
    sk.paint(sk.ellipse(0, 0.1, 0.65, 0.5), "#d0402c", line=True, name="teapot")
sk.paint(sk.ellipse(0.73, 0.33, 0.07, 0.045), "#f2d21e", line=True, name="lemon")                   # above the teapot
print(manifest.relations(cv))         # ['pear is left of pear #2', ..., 'teapot is right of pear #2', 'teapot is below lemon']
```

**Woodblock wave**: flat fill, stripes along the form, foam fingers with ink line.

```python
import numpy as np
from pixelcanvas import Canvas
from pixelcanvas.illustrate import Sketch, paper, limit_palette
cv = Canvas(640, 360, "#ecdcb8")
sk = Sketch(cv)
back = [(0.0, 1.0), (0.12, 0.55), (0.3, 0.3), (0.5, 0.25), (0.62, 0.38)]
hollow = [(0.56, 0.36), (0.46, 0.4), (0.42, 0.6), (0.48, 1.0)]
wave = sk.polygon(sk.curve(back) + sk.curve(hollow))
sk.paint(wave, "#1d3f78", line="#1a1d2c", name="wave")
stripes = np.maximum.reduce([sk.stroke([(x + k * 0.01, y + k * 0.02) for x, y in sk.curve(back)], 0.003, smooth=False)
                             for k in range(1, 20)])
sk.paint(stripes * wave, "#9cb8cc")
for i, (x, y) in enumerate(sk.curve(back)[30::6]):
    with sk.at(x, y, 1):
        a = np.radians(30 + 12 * i) + 1.8 * np.linspace(0, 1, 6) ** 2
        p = np.cumsum(np.stack([np.cos(a), np.sin(a)], 1) * 0.012, 0)
        sk.paint(sk.stroke([(0, 0)] + [tuple(q) for q in p], 0.02, taper="end"), "#f4ecd8", line="#1a1d2c",
                 line_width=0.002)
limit_palette(cv, ["#1d3f78", "#9cb8cc", "#f4ecd8", "#ecdcb8", "#1a1d2c"], 0.5)
paper(cv)
```

**Pixel art**: a native low-resolution grid, scaled up without smoothing.

```python
from pixelcanvas import Canvas
from pixelcanvas.illustrate import Sketch, limit_palette
cv = Canvas(640, 360)
small = Canvas(160, 90)
small.gradient([(0, "#2a1b4a"), (0.6, "#e0607a"), (1, "#ffc56b")])
px = Sketch(small)
px.paint(px.polygon([(0, 0.8), (0.3, 0.55), (0.55, 0.75), (0.8, 0.5), (1, 0.7), (1, 1), (0, 1)]), "#3a2350")
px.paint(px.circle(0.62, 0.42, 0.12), "#ffe08a")
limit_palette(small, ["#2a1b4a", "#6a2f6a", "#e0607a", "#ffc56b", "#ffe08a", "#3a2350"])
cv.paste(small.resize(640, 360, "nearest"))
```

**Paper cut**: layers far to near, each casting a soft shadow on the one behind.

```python
from pixelcanvas import Canvas
from pixelcanvas.illustrate import Sketch, paper
cv = Canvas(640, 360, "#f2ead8")
sk = Sketch(cv, light=(-0.5, -0.9))
sk.paint(sk.circle(0.7, 0.35, 0.18), "#fbefc8", glow="#ffe6a0")
for i, (top, color) in enumerate([(0.5, "#9cc5b0"), (0.62, "#5f9a86"), (0.75, "#2f6658"), (0.88, "#173c35")]):
    hills = [(-0.05, 1.05)] + [(x / 6, top + 0.06 * ((x * 7 + i * 3) % 5 - 2) / 2) for x in range(-1, 8)] + [(1.05, 1.05)]
    sk.paint(sk.blob(hills, tension=0.3), color, highlight=0.25,
             shadow=dict(offset=(0.004, -0.01), blur=0.012, opacity=0.45))
paper(cv, grain=0.04, fibres=0.04)
```

**Comic panel**: bold ink lines, flat bright fills, halftone dots, a jagged burst behind the sound word.

```python
import math
from pixelcanvas import Canvas
from pixelcanvas.illustrate import Sketch
cv = Canvas(640, 400, "#fff4d6"); sk = Sketch(cv, finish="flat")
sk.pattern(sk.box(0, 0, 1, 1), "dots", "#f6b8c8", size=0.03)                  # halftone ground
with sk.at(0.3, 0.5, 0.7):                                                    # a rocket bursting upwards
    for s in (-1, 1):
        sk.paint(sk.polygon([(s * 0.12, 0.2), (s * 0.3, 0.38), (s * 0.12, 0.34)]), "#2f6fd0", line="ink")
    sk.paint(sk.blob([(0, -0.5), (0.13, -0.2), (0.13, 0.32), (-0.13, 0.32), (-0.13, -0.2)], 0.4), "#f2f2f2",
             line="ink", name="rocket")
    sk.paint(sk.circle(0, -0.12, 0.06), "#7fd0f0", line="ink")
    sk.paint(sk.stroke([(0, 0.34), (0, 0.62)], [0.16, 0.0]), "#ff8a1a", line="ink")
r = [0.36 if i % 2 else 0.22 for i in range(18)]                              # jagged burst: alternating radii
sk.paint(sk.polygon([(0.72 + 0.6 * r[i] * math.cos(i * math.pi / 9), 0.4 + r[i] * math.sin(i * math.pi / 9))
                     for i in range(18)]), "#ffd21f", line="ink", name="burst")
sk.text("WHOOSH!", 0.72, 0.4, size=0.08, color="#e8262a", anchor="mm", outline=4, font="sans black")
```
