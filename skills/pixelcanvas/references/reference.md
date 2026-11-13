# Match a reference: probe its style, draw with its numbers, close the gaps

When the request comes with a style reference ("in the style of this image", "match this look"), read the
reference as numbers before drawing: its palette, line colour and weight, how many tone bands each colour
region has and how the shadow tone shifts, where it grades softly, its highlights. Draw with those numbers, then
let the gap check say what still differs, largest gap first.

## The protocol

1. **Probe first**: `python <root>/tools/match.py REF.png` prints a style sheet. Open the reference too when you
   have vision; the sheet tells you what to look for.
2. **Write the numbers into the scene docstring** and paste the `PASTE` lines as constants: `PALETTE`, `LINE,
   LINE_WIDTH`, `SHIFT, BANDS`, `LIFT`.
3. **Draw with them**: regions the sheet calls *hard bands* get `sk.cel(mask, base, bands=BANDS, shift=SHIFT,
   line=LINE, line_width=LINE_WIDTH)`; *soft gradient* regions get `sk.ramp(mask, colors)`; *flat* regions get
   `sk.paint`. Hair sheen and eye highlights come from `sk.sheen` and `sk.eye` when the sheet lists highlight
   bands and specular spots.
4. **Run the gap check** in the render loop: `python <root>/tools/render.py ./art/NAME.py --check --out OUT.png
   --ref REF.png` (or `REFERENCE = "REF.png"` in the scene) adds a REFERENCE block just before FIX.
5. **Fix the largest gaps** in one edit and render once more. A match in the 80s with no `bands`, `lines` or
   `palette` line left is a faithful match; the rest is your eye.

## The two commands

```bash
python tools/match.py REF.png               # style sheet + ready-to-paste lines
python tools/match.py REF.png OUT.png       # gaps of a draft, largest first, and one match figure 0-100
python tools/match.py REF.png OUT.png --json
```

A style sheet reads like this:

```text
palette: #97c4ef 32%  #5593e0 27%  #e9ecec 25%  #d65a61 6%  #a62250 4% ...
roles: background #5593e0, skin #ebafa9, line #5d313a, accent #a62250
lines: #5d313a (coloured), width 0.0036 of the height (~4 px at 1080)
shading by region (largest first):
  #91c4f0 61%: soft gradient #c0e5f8 -> #4484da
  #f6eada 22% (skin): hard bands, 3 tones on a graded base #fffeff -> #f6eada; shadow #cbc9e2 = L -0.10, hue +0 deg
  #d2555f 11%: hard bands, 3 tones; shadow #b12854 = L -0.12, hue -13 deg; highlight band #f88f96 (L +0.17)
structure: 35% banded, 65% gradient, 0% flat
PASTE (scene docstring / constants):
  LINE, LINE_WIDTH = "#5d313a", 0.0036
  SHIFT, BANDS = (-0.108, -4.5), 2
  SHIFTS = {"#f6eada": (-0.1, 0.0), "#d2555f": (-0.122, -13.1)}   # per material
```

## Reading the output

Bands are read from crisp tone-step edges: same hue on both sides, lightness 0.06-0.32 apart, no drawn line
between. Soft gradients have no such edges; lines and boundaries between different things are set apart.

| Line | Means | Draw it with |
|---|---|---|
| `soft gradient A -> B` | the region changes smoothly (sky, hair, glow) | `sk.ramp(mask, [A, B])` |
| `hard bands, n tones` | base plus n-1 crisp shadow / highlight tones | `sk.cel(mask, base, bands=1 or 2)` |
| `... on a graded base` | crisp bands over a gradient fill (anime hair) | `sk.cel(mask, [(0, c0), (1, c1)], bands=...)` |
| `SHIFTS` | the shadow shift of each material (skin, hair, cloth) | `sk.cel(m, c, shift=SHIFTS[c])` |
| `shadow ... L -0.13, hue -12 deg` | the shadow tone is the base 0.13 darker (OKLab) and turned 12° | `shift=(-0.13, -12)` or `cel_tone(base, -0.13, -12)` |
| `highlight band` / `LIFT` | a light band inside the lit edge | `sk.cel(..., highlight=0.8)`, `sk.sheen(hair)` |
| `lines: #... (coloured)` | outlines in a dark tone of the fills, not black | `line=LINE, line_width=LINE_WIDTH`, `sk.ink(pts)` |
| `no outlines` | shapes meet colour to colour | leave `line=` off |
| `specular spots yes` | small white glints (eyes, metal, wet) | `sk.eye(...)`, small white `sk.circle` |
| `values`, `saturation` | the overall key and colour strength | pick fills in that range |

The gap list names the measured difference and the change, e.g. `[bands gap 0.71] flat where the reference has
2-3 tones: banded area 3% vs 31% - paint skin, hair and cloth with sk.cel(mask, base, bands=2, shift=(-0.128,
-12.5))`. Gaps: `palette` (OKLab distance, the reference colours to use), `bands`, `gradient`, `lines`,
`contrast`, `key`, `saturation`, `highlights`.

## Worked example

A banded reference, a flat draft of it, and the check that sends the draft back to soft-cel:

```python
from pixelcanvas import Canvas, reference
from pixelcanvas.illustrate import Sketch

def picture(cel):
    cv = Canvas(640, 360)
    sk = Sketch(cv, light=(-0.6, -0.8))
    sk.ramp(sk.box(0, 0, 1, 1), ["#4f8fe0", "#cdeefa"])                 # soft gradient sky
    for x, c in ((0.3, "#e0645a"), (0.7, "#f6dcc8")):
        m = sk.circle(x, 0.55, 0.3)
        if cel:
            sk.cel(m, c, bands=2, shift=(-0.13, -12), highlight=0.8, line=True)
        else:
            sk.paint(m, c)
    return cv

ref, draft = picture(True), picture(False)
sheet = reference.probe(ref)
print(reference.sheet_text(sheet, "ref"))
gaps = reference.compare(sheet, draft)
print("\n".join(reference.gap_lines(gaps)))
assert gaps["gaps"][0]["key"] in ("bands", "lines")
assert reference.compare(sheet, ref)["match"] == 100
```

In Python: `reference.probe(img)` -> dict, `reference.sheet_text(p)`, `reference.compare(ref, draft)` ->
`{"match", "gaps", "ref", "draft"}`, `reference.gap_lines(c)`. Pictures are paths, PIL images, arrays or a
`Canvas`. The anime recipe that uses these numbers is in [techniques.md](techniques.md).
