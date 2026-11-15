# Medium: deliver the picture as a render, an oil, a watercolour, ink or pixel art

The layers you draw become per-pixel buffers (local colour, a relief per shape, normals, depth, emission), one key
light lights them, and a medium paints the result as a process of its own: strokes, washes, hatching, a pixel grid.
The drawing stays the layers you edit; the medium is the last stage.

## Quick start

```
studio new pears --prompt "an oil painting of three pears on a cloth" --from draft.py
#   -> medium: oil from the request (painted over the drawing after every render)
studio set scene medium=watercolour          # another medium: no section runs again
studio set scene medium_mark=2 medium_strength=0.7
studio stages                                # flat silhouettes -> relief clay -> lit colour -> the medium
studio set scene medium=off                  # the drawn picture again
```

`new` reads the medium from the request: "a watercolour of", "oil painting", "gouache", "ink drawing", "etching",
"pixel art", "8-bit", "flat vector"; "3D render" gives `render`. A photo or realistic request keeps the drawing
(lit solids, no outlines); `set scene medium=render` relights it with one key light and cast shadows.

## The six media

| medium | what it paints |
|---|---|
| `render` | the key light, soft shadows on a floor plane, contact occlusion, specular, bloom: a 3D-render look |
| `flat` | local colour posterised into value bands, cool shadow shapes, crisp edges: a flat graphic |
| `oil` | bristle strokes along the form, coarse to fine, on a toned canvas with impasto |
| `watercolour` | washes in value bands on paper: pooled edges, granulation, reserved whites, pencil under it |
| `ink` | contours and hatching that wraps the form, spot blacks, on warm paper |
| `pixel` | a low-resolution grid in a small palette, value bands and a dark outline |

Text layers stay sharp over every medium (inked in `ink`).

## Dials

| scene key | values | what changes |
|---|---|---|
| `medium` | `render flat oil watercolour ink pixel off` | the medium |
| `medium_strength` | 0 to 2, default 1 | how much of the medium: simplification, marks, texture, colour shift |
| `medium_mark` | 0.5 (fine) to 3 (broad), default 1 | stroke, wash, hatch or pixel size |
| `medium_palette` | 2 to 64 | the number of colours (`flat`, `pixel`, any medium) |
| `medium_paper` | a colour | the canvas or paper colour |

```
studio set scene medium=pixel medium_mark=2 medium_palette=12
studio set scene medium=oil medium_paper=3a2a20 medium_mark=1.5
```

## What the medium knows

Every section records what it paints while it runs, so the buffers are exact where the drawing knows them:

| drawn with | recorded |
|---|---|
| `sk.paint(mask, colour, volume=0.6)` | the mask, the local colour (without the painted shading), a rounded relief |
| `sk.paint(..., volume=-0.5)` | a hollow: the shape dips (a bowl's inside, a dish, a cave mouth) |
| `sk.paint(..., shade=0.4)` | a gentler relief from the shade amount |
| `sk.paint(...)` with no volume or shade | a flat shape lying on what is under it (labels, patterns, stripes) |
| `sk.paint(..., spec=0.6)` | gloss: a sharp highlight from the key light |
| `sk.paint(..., shadow="ground")` | the thing stands on the floor: the medium casts its shadow from the key light |
| `sk.paint(..., glow=...)` | emission: the glow lights its neighbours, the thing keeps its colour |
| form solids (`lathe`, `cylinder`, `ball`, `cone`, `cuboid` ...) | true normals; an open lathe records its inside |
| `creature(...)` animals | true normals and the coat's own colours; eyes and nose as drawn |
| `Figure(...)` people | every part: skin and cloth rounded, hair gently, details flat, each in its own colour |
| kits named fire, campfire, lamp, lantern, candle, neon | their flames are emission |

A later shape sits on the earlier one, so a nose stands on a face and a label on a bottle. Paint the records do
not know (a kit drawn inside a section) keeps its drawn colour. A layer with no records at all is estimated from its
pixels with a gentle relief; the report says so (`rock: volume estimated`), and `studio stages` shows the relief it
got. Flat things stay flat: a layer named like a puddle, rug, petal, shard, painting, poster, decal or pattern, a
low wide shape on the floor, and many small fragments. A layer that should read rounder: give its shapes
`volume=`, or draw it with a form solid.

Every layer keeps its drawn colour: the light changes value and adds a cool shadow or a highlight, and each
layer's hue, saturation and overall lightness stay the drawing's, so a grey elephant stays grey and a blue sky
blue. Lamps, flames and glows look as drawn and light their neighbours.

The floor comes from what the picture knows: a table, tabletop, ground or floor section sets the floor line; the
scene `light` is the key light.

## How a picture with a medium runs

1. The sections draw the layers; the measured checks (counts, colours, places, text) run on the layers.
2. The medium paints the picture after every render, over every layer that draws; what is still open (an item,
   a note, a failing section) is said on the medium line. When it cannot paint at all, NEXT says so first.
3. The picture file (`--out`), the preview and the look sheet show the medium picture.
4. The medium checks run; a failing one is the NEXT line, with its fix:

| check | measures | fix |
|---|---|---|
| value pattern | the medium picture's lights and darks follow the drawing (correlation 0.4 or more) | lower `medium_strength` |
| colours | every requested colour that reads on the layers still reads | raise `medium_palette` (`flat`, `pixel`) or lower `medium_strength` |
| contrast | every requested thing keeps at least 40 % of its contrast against its ground | lower `medium_strength` |
| text | text keeps 60 % of its contrast against what is behind it | `set TEXT outline=on` |

Buffers are cached by the picture state, so changing the medium or a dial re-runs only the light and the medium;
the report gives the seconds (`buffers 0.0, oil 8.4`).

## stages

`studio stages [MEDIUM]` writes `art/NAME/stages.png`: the flat silhouettes, the relief clay (grey, lit), the lit
colour and the medium side by side. The clay is the quickest way to see a flat or a wrong shape.

## Outside the studio

```python
from pixelcanvas import Canvas
from pixelcanvas.illustrate import Sketch
from pixelcanvas.medium import record, layer, buffers_from_layers, paint, stages

bg = Canvas(480, 320)
bg.gradient([(0, "#d9c7a8"), (0.6, "#c9b38f"), (0.6, "#7a4e2c"), (1, "#5e3a20")])
cv = Canvas(480, 320)                       # the objects, on black
sk = Sketch(cv)
rec = record.start(cv)
sk.paint(sk.circle(0.32, 0.6, 0.15), "#c8402f", volume=0.6, spec=0.6, shadow="ground")
sk.paint(sk.ellipse(0.68, 0.62, 0.17, 0.09), "#f0ece4", volume=-0.6)
record.stop(rec)

b = buffers_from_layers([layer(bg, role="backdrop"), layer(cv, rec, "still life")], light=(-0.6, -0.8))
oil = paint(b, "oil", mark=1.2)             # sRGB float H x W x 3
ink = paint(b, "ink")
strip = stages(b, "watercolour")            # [(label, image)] x 4
assert oil.shape == (320, 480, 3) and [s[0] for s in strip][-1] == "watercolour"
```

```python
from pixelcanvas.medium import detect, style
assert detect("a watercolour of a harbour at dawn") == "watercolour"
assert detect("a 3D render of a red bicycle") == "render" and detect("a photo of a red bicycle") is None
st = style.style_for("pixel", mark=2.0, palette=12)      # the dials as a Style
assert st.palette == 12
```
