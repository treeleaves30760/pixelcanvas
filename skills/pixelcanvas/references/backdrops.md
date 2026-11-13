# Palettes and backdrops (`pixelcanvas.palette`, `pixelcanvas.backdrop`)

Start every poster, card, banner, slide and wallpaper here: pick a palette, paint a backdrop in one call, and put
the text where the backdrop says it will read.
Options are checked: an unknown option name raises a TypeError that names the closest valid one (`tags` -> "did you mean 'tag'?"), and both spellings are accepted (`colour` / `color`, `centre` / `center`, `w` / `width`, `alpha` / `opacity`, ...). Unknown names of backdrops, palettes and families raise a ValueError with the closest valid names.


```python
from pixelcanvas import Canvas
from pixelcanvas.backdrop import backdrop

cv = Canvas(1920, 1080)
bd = backdrop(cv, "hills", palette="sunset", seed=3)        # or backdrop(cv, "warm sunset hills")
cv.text("Golden Hour", box=bd.calm_px(cv, pad=0.12), color=bd.ink, bold=True)
```

`bd.calm` is the largest quiet box (x0, y0, x1, y1 as fractions), `bd.ink` a colour that passes WCAG 4.5 there,
`bd.palette` the Palette used; `bd.busy` is True when the pattern is even everywhere (stripes, fluid, voronoi ...):
`calm` is then a centred box, so run `quiet(cv, bd.calm_px(cv))` before the text. Every backdrop is one `cv.paint` (one layer), deterministic by `seed`, and sized in
units of the canvas short side, so a draft and a 4K render look the same.

## Palettes

60 hand-tuned palettes of 5-7 colours, each with a usable dark, a usable light and one or two accents.

| Family | Names |
|---|---|
| mood | calm energetic luxurious playful corporate earthy dreamy retro cyber minimal romantic moody |
| subject | sunset ocean forest desert autumn spring summer winter night_city candy coffee tropical lavender aurora sakura deep_space |
| era | bauhaus memphis art_deco mid_century ukiyo_e pop_art vaporwave synthwave nordic terracotta swiss risograph art_nouveau y2k |
| ui | indigo emerald rose amber slate sky violet teal orange, each also `<name>_dark` |

| Call | What it gives |
|---|---|
| `palette("sunset")`, `palette("warm evening sky")`, `palette("calm, sage green, cream")`, `palette(["#123", ...])`, `palette("#3a6ea5")`, `palette(p, n=9)` | a Palette (name, mood description, **named colours** with shade words — `soft blue and peach`, `dark navy and gold` — colour list, one colour → analogous harmony; `n` resamples) |
| `p.colors` `p.bg` `p.fg` `p.accent` `p.muted` `p.dark` `p.light` `p.accents` `p.mode` | roles: `fg` reads on `bg` (≥ 4.5), `accents` most vivid first, `mode` light / dark |
| `p.ramp(n)` · `p.stops()` · `p.sorted()` | OKLab ramp dark → light · `[(0, c), ..., (1, c)]` for `cv.gradient` |
| `p.text_on(colour)` | palette ink with ≥ 4.5 contrast on any colour |
| `p.lighter()` `p.darker()` `p.muted(0.5)` `p.saturated()` `p.inverted()` `p.shuffle(seed)` | variants (all return Palettes) |
| `find("warm sunset over the sea")`, `find("夢幻粉彩")`, `find(q, n=3)` · `names(tag)` · `describe(name)` | search (English + Traditional Chinese keywords) |
| `harmony(base, kind, n=5)` | kind: analogous complementary split triadic tetradic monochrome |
| `tints_shades(base, n=9)` · `from_image(img, n=5)` | one hue dark → light · dominant colours (k-means in OKLab) |
| `contrast_ratio(a, b)` · `ensure_contrast(fg, bg, 4.5)` · `mix_oklab(a, b, t)` · `oklch(L, C, h)` | colour maths |
| `categorical(n, scheme)` | colour-blind-safe sets: `okabe_ito` `tol_bright` `tol_vibrant` `tol_muted` `ibm` |
| `srgb_to_oklab` `oklab_to_srgb` `oklab_to_oklch` `oklch_to_oklab` | conversions on arrays (..., 3) |

## Backdrops

| Call | What it does |
|---|---|
| `backdrop(cv, name_or_description, palette=None, seed=0, strength=1, box=None, **opts)` | paint → `Backdrop(name, palette, calm, ink, busy)`; `box=(x0, y0, x1, y1)` fractions or px paints only there |
| `bd.calm_px(cv, pad=0.1)` | the calm box in pixels, shrunk by `pad` × its short side |
| `quiet(cv, box, amount=0.6, tone=None)` | blur + tone a busy area toward its average behind text, feathered (no box edge) |
| `render(name, w, h, palette, seed, **opts)` | the backdrop as an (h, w, 3) array + Palette, without painting |
| `names(family)` · `find("soft pastel blobs")` · `describe(name)` | catalogue; `describe` lists the options and the suggested palette |

| Family | Backdrops |
|---|---|
| gradient | mesh linear radial conic aurora grainy spotlight duotone glass |
| geometric | dot_grid line_grid isometric triangles hexagons circles sunburst stripes chevrons checker waves topographic ribbons low_poly voronoi memphis bauhaus art_deco truchet halftone circuit blueprint origami icons |
| scenic | hills mountains dunes ocean synthwave starry clouds skyline forest underwater |
| organic | watercolor ink marble fluid smoke bokeh light_leaks paper warm_paper kraft crumpled concrete fabric wood_desk |

Useful options: `linear(angle, colors)`, `radial(center, radius)`, `mesh(points, flow, grain)`, `glass(panels, blur)`,
`waves(layers, height)`, `hills(layers, sun, sun_x)`, `mountains(layers, sun_x, sun_y, mist)`, `ribbons(lines, ribbons)`, `ocean(horizon, sun_x)`, `memphis(density, clear)`,
`icons(icons=[...], size, clear)`, `halftone(spacing, corner)`, `fabric(weave="denim")`, `paper(tint)`,
`warm_paper(tint="#f3e5c8", fibres=1.0, edges=1.0)`.

**Warm paper** for flyers, menus and invitations is one call — cream, lighter at the top, soft fibres and warm
edges (`"cream paper"`, `"parchment"` find it too):

```python
from pixelcanvas import Canvas
from pixelcanvas.backdrop import backdrop
cv = Canvas(1080, 1350)
bd = backdrop(cv, "warm_paper")                   # bd.ink is a dark brown that reads on it
cv.text("Weekend Market", box=bd.calm_px(cv, pad=0.2), color=bd.ink, font="serif bold")
```
Without `palette=` each backdrop uses the palette it was designed with (`describe(name)` shows it).

## Recipes

**Poster with a headline in the calm area** — a busy pattern gets `quiet()` behind the text first.

```python
from pixelcanvas import Canvas
from pixelcanvas.backdrop import backdrop, quiet

cv = Canvas(1080, 1350)
bd = backdrop(cv, "memphis", palette="memphis", seed=4)
box = bd.calm_px(cv, pad=0.08)
quiet(cv, box, amount=0.7)
cv.text("SUMMER\nPARTY", box=box, color=bd.palette.fg, bold=True, align="center")
```

**Cards in a row, each with its own backdrop** — `box=` paints into part of the canvas; the result's ink fits the card.

```python
from pixelcanvas import Canvas
from pixelcanvas.backdrop import backdrop
from pixelcanvas.palette import palette

cv = Canvas(1920, 1080, palette("slate").bg)
for i, (name, pal) in enumerate([("mesh", "dreamy"), ("topographic", "earthy"), ("aurora", "aurora")]):
    x0 = 0.05 + i * 0.31
    bd = backdrop(cv, name, palette=pal, seed=i, box=(x0, 0.2, x0 + 0.28, 0.8))
    cv.text(name.title(), box=bd.calm_px(cv, pad=0.15), color=bd.ink, bold=True)   # auto-fits the calm box
```

**Palette roles in a layout** — background, accent shape, text and a quiet rule all from one palette.

```python
from pixelcanvas import Canvas, sdf
from pixelcanvas.palette import palette, harmony

p = palette("mid_century")                  # or harmony("#2c7873", "split")
cv = Canvas(1920, 1080, p.bg)
X, Y = cv.xy()
cv.paint(p.accent, sdf.fill(sdf.sd_circle(X, Y, 1450, 540, 300)))
cv.paint(p.muted, sdf.fill(sdf.sd_circle(X, Y, 1450, 540, 150)))
cv.text("Atomic Age", (160, 540), 120, p.fg, bold=True, anchor="ls")
cv.paint(p.accent, sdf.fill(sdf.sd_box(X, Y, 420, 600, 260, 5)))
cv.text("Design that lasts", (160, 690), 44, p.text_on(p.bg), anchor="ls")
```

**A social card from three words** — the palette comes from the words, the calm box holds the type.

```python
from pixelcanvas import Canvas
from pixelcanvas.backdrop import backdrop, quiet
from pixelcanvas.layout import stack

cv = Canvas(1200, 630)
bd = backdrop(cv, "tasteful abstract", palette="calm, sage green, cream", seed=1)   # soft mesh, sage and cream
box = bd.calm_px(cv, pad=0.06)                      # never closer than 7% of the short side to the canvas edge
if bd.busy:
    quiet(cv, box)
stack(cv, box, [("Designing for Calm", "title"), ("R. Okonkwo", "subtitle")], valign="middle", color=bd.ink)
```

**A wallpaper from one sentence** — colours named in the description become the palette.

```python
from pixelcanvas import Canvas
from pixelcanvas.backdrop import backdrop

cv = Canvas(2560, 1440)
backdrop(cv, "misty mountains at dawn in soft blue and peach", seed=3)
```

**A wallpaper in two lines, then a finish**

```python
from pixelcanvas import Canvas
from pixelcanvas.backdrop import backdrop

cv = Canvas(1920, 1080)
backdrop(cv, "synthwave", seed=2)
cv.grain(0.008, seed=2)
cv.save("outputs/night_drive.png")
```

`scenes/backdrop_sheet.py` shows every backdrop, `scenes/backdrop_wallpapers.py` six finished wallpapers
(`--set WALL=3 --size 1170x2532` renders one alone, e.g. for a phone) and `scenes/palette_sheet.py` every palette.
