# Textures and materials (pixelcanvas.texture)

80 ready materials — wood, stone, brick, fabric, metal, tiles, grass, fur, feathers, water, lava, fire — each built
from the structure that makes the real thing recognisable (stones with mortar, blades with a direction, threads going
over and under, cracks that glow) and lit through a height map, so every bump answers your scene light.
Options are checked: an unknown option name raises a TypeError that names the closest valid one (`tags` -> "did you mean 'tag'?"), and both spellings are accepted (`colour` / `color`, `centre` / `center`, `w` / `width`, `alpha` / `opacity`, ...). Unknown names of textures raise a ValueError with the closest valid names.


```python
from pixelcanvas import Canvas, sdf
from pixelcanvas.texture import texture, fill, find

W, H = 960, 540
cv = Canvas(W, H)
cv.paint(texture("brick", H, W, seed=3))                        # whole canvas: a lit colour map
X, Y = cv.xy()
disc = sdf.fill(sdf.sd_circle(X, Y, W / 2, H / 2, H / 3))
fill(cv, disc, find("wooden floor"), shadow=0.6)                # into a mask
```

## Conventions

- **Units**: sizes are texture units = 1080p pixels at `scale=1` (`scale=2` doubles every feature). The same call
  looks the same at any canvas size.
- **Light**: `light=(x, y)` points towards the light on screen (y down), default `(-0.6, -0.7)` = upper left.
  Pass the same value to every call in a scene. `bump=` scales the relief (0 = flat colour).
- **Mask**: `mask=` computes only the mask's bounding box; the pattern stays continuous across separate masks.
- **uv**: `uv=(U, V)` arrays (H, W) in texture units replace the pixel grid → perspective floors (`plane_uv`),
  arches (`polar_uv`), cylinders, any warp. `ss=2` supersamples (use it for far-away perspective ground).
- **Paper**: `texture("paper", h, w)` is a light cream sheet; painting it straight onto the canvas
  (`cv.paint(texture("paper", cv.h, cv.w, seed=1))`) gives a finished background. Paper-type textures take
  `color=` for the sheet colour (`color="#f3e6c8"` for parchment, `"#e8f0fa"` for blue stock).
- **Colours**: `tint="#3060c0"` recolours **any** material (it keeps the light / dark structure; `tint=(c, 0.5)` for half).
  Each texture also has its own options: `palette=[...]` (dark → light, or per-piece colours), `color=`, `mortar=`,
  `grout_color=` ... `describe("linen")` prints them with their defaults; a wrong name raises an error that lists them.
- **Object size**: a feature is the same size at any canvas size, so a napkin or a coin gets the same weave as a wall.
  For a finer or coarser material pass `scale=0.5` / `scale=2`.
- **Maps**: `maps=True` returns `Maps(color, height, alpha, shine, emit, lit)`: albedo, 0~1 height, alpha (holes,
  log disc, flames), HDR emission (lava, fire, stars, caustics) and the lit result. Use the height to place moss in
  joints, dirt in cracks, water in hollows.

## Calls

| Call | What it does |
|---|---|
| `texture(name, h, w, scale=1, seed=0, **kw)` | a texture by name or by description (`"red brick wall"` → brick) → lit (H, W, 3) |
| `fill(cv, mask, name, light=..., shadow=0, opacity=1, mode="normal", **kw)` | paint a texture into a mask; `shadow` adds a soft contact shadow |
| `find("wooden floor")` → `"planks"`, `find(q, n=3)` → best 3 | keyword search |
| `names(family=None)`, `FAMILIES`, `TEXTURES`, `describe(name)` | the catalogue |
| `<name>(h, w, ...)` e.g. `brick(h, w, bond="flemish")` | every texture is also a function in `TEXTURES` |
| `plane_uv(h, w, horizon=0.4, near=1, center=0.5, shift=(0, 0))` | (U, V) of a ground plane in perspective |
| `polar_uv(h, w, cx, cy, radius)` | (U, V) along circles: U = arc length, V = distance from `radius` |
| `cellular(U, V, jitter=1, seed=0, round=0)` | Worley / Voronoi: `f1, f2, edge, id, rand, cx, cy, dx, dy`, `r(k)` |
| `voronoi_shade(cells, palette, gap=0.06)` | one colour per cell with dark borders |
| `strokes(h, w, angle=-90, length=30, width=2, density=1, spread=20, bend=0.3, layers=3)` | oriented strokes following a direction field → `Strokes(cov, tone, along, across, height)` |
| `relief(height, light, depth=4, color=None, shine=0)` | light any height map (flat = 1) |
| `patches(h, w, coverage=0.4, size=120, bias=None, mask=None)` | organic patch mask for moss / snow / rust / dirt |
| `weather(cv, mask, amount=0.5)` | grime blotches and rain streaks on what is already painted |

## The catalogue (key options)

| Family | Textures |
|---|---|
| wood | `planks` (width, length, angle=90 vertical, nails, knots), `wood_grain` (spacing, knots), `end_grain` (log slice: center, radius, cracks, bark), `bark` (lichen) |
| stone | `cobblestone` (size, gap, wet), `pebbles`, `flagstone`, `brick` (size=(215, 65), bond=running/stack/english/flemish, mortar, weathering), `stone_wall` (style=rubble/ashlar), `marble` (vein), `granite`, `concrete` (formwork, stains), `gravel`, `sand` (wavelength, angle), `soil`, `dry_earth`, `asphalt` |
| fabric | `linen`, `denim`, `tartan` (sett=[(colour, threads)…]), `stripes` (widths, palette), `polka`, `houndstooth`, `chevron`, `knit` (color), `felt`, `velvet`, `leather`, `carpet`, `rug`, `quilted` |
| metal | `brushed_metal` (angle), `hammered_metal`, `rust` (amount), `galvanized`, `chrome`, `gold_leaf`, `carbon_fiber`, `perforated` (alpha = holes), `corrugated` (color, rust) |
| man-made | `paper` (color; a light warm cream sheet with fine grain, ready as a background), `cardboard`, `cork`, `crackle_glaze`, `glass_tiles`, `terrazzo`, `damask`, `wallpaper`, `tiles` (checker), `subway_tiles`, `hex_tiles`, `herringbone` (wood=False + palette for tiles), `circuit`, `lined_paper` (color = the sheet), `graph_paper` (paper = the sheet, color = the grid), `film_grain` (use `mode="overlay"`), `brushstrokes` |
| organic | `lawn` (stripes), `meadow` (wind, flowers), `moss`, `fur` (kind=short/long/curly, flow), `fur_long`, `fur_curly`, `feathers` (bars), `fish_scales`, `reptile_scales`, `leaves`, `hay`, `wicker`, `rattan`, `honeycomb` |
| liquid | `water` (sky, deep), `caustics`, `ice` (frost), `snow`, `lava` (heat), `fire` (height), `mud` (wet), `clouds` (coverage), `starfield` (density) |

Each tile of `scenes/texture_swatches.py` is one of these calls; `scenes/texture_wall.py` builds a whole scene from
them.

## Recipes

**Objects on a table** — one `fill` per object; `shadow` grounds it, `tint` or the texture's colour option sets the
colour, `angle=` turns directional materials (brushed metal, planks) to follow the object.

```python
from pixelcanvas import Canvas, sdf
from pixelcanvas.texture import fill, texture

W, H = 960, 540
cv = Canvas(W, H)
X, Y = cv.xy()
cv.paint(texture("planks", H, W, angle=0, width=220))                       # table top
mat = sdf.fill(sdf.sd_box(X, Y, 0.5 * W, 0.5 * H, 0.3 * W, 0.4 * H, radius=14))
fill(cv, mat, "cork", shadow=0.5)
plate = sdf.fill(sdf.sd_circle(X, Y, 0.5 * W, 0.5 * H, 0.28 * H))
fill(cv, plate, "crackle_glaze", color="#eceff1", shadow=0.6)                # ceramic
coaster = sdf.fill(sdf.sd_circle(X, Y, 0.78 * W, 0.3 * H, 0.08 * H))
fill(cv, coaster, "marble", shadow=0.6)
blade = sdf.fill(sdf.sd_box(X, Y, 0.69 * W, 0.5 * H, 11, 80, radius=8))
fill(cv, blade, "brushed_metal", angle=90, shadow=0.6)
fill(cv, sdf.fill(sdf.sd_circle(X, Y, 0.2 * W, 0.5 * H, 90)), "leather", tint="#2d3f6b", shadow=0.5)
```

**A floor in perspective** — the ground plane's coordinates go in as `uv`; far rows get `ss=2` for clean detail.

```python
import numpy as np
from pixelcanvas import Canvas
from pixelcanvas.color import clamp01
from pixelcanvas.texture import fill, plane_uv

W, H = 960, 540
cv = Canvas(W, H, "#2a2f3a")
X, Y = cv.xy()
floor = clamp01(Y - 0.45 * H + 0.5)
uv = plane_uv(H, W, horizon=0.3, near=1.5)
fill(cv, floor, "herringbone", uv=uv, ss=2)
cv.paint("#000000", floor * np.exp(-(Y - 0.45 * H) / (0.07 * H)) * 0.6)   # contact shadow at the wall
```

**A brick arch with radial joints** — `polar_uv` turns circles into rows; swap U and V so the brick length runs
along the radius.

```python
import numpy as np
from pixelcanvas import Canvas, sdf
from pixelcanvas.texture import fill, polar_uv

W, H = 960, 540
cv = Canvas(W, H, "#3a3530")
X, Y = cv.xy()
cx, cy, R, T = 0.5 * W, 0.6 * H, 0.28 * H, 0.15 * H
r = np.hypot(X - cx, Y - cy)
ring = sdf.fill(np.maximum(R - r, r - (R + T))) * (Y < cy)
U, V = polar_uv(H, W, cx, cy, radius=R)
brick_len = T / (H / 1080) / 219                          # one brick (215 + mortar) spans the ring
fill(cv, ring, "brick", uv=(V, U), scale=brick_len, bond="stack", shadow=0.5)
```

**Moss in the joints, not on the stones** — use the height map from `maps=True` to steer an effect.

```python
from pixelcanvas import Canvas
from pixelcanvas.color import smoothstep
from pixelcanvas.texture import texture, fill, patches

W, H = 960, 540
cv = Canvas(W, H)
m = texture("stone_wall", H, W, seed=2, maps=True)
cv.paint(m.lit)
joints = 1 - smoothstep(0.3, 0.7, m.height)
U, V = cv.uv()
moss = patches(H, W, coverage=0.45, bias=V - 0.6) * joints
fill(cv, moss, "moss")
```

**Fur that follows the body** — `flow` takes a per-pixel direction map (degrees, 0 = right, 90 = down).

```python
import numpy as np
from pixelcanvas import Canvas, sdf
from pixelcanvas.texture import fill

W, H = 960, 540
cv = Canvas(W, H, "#20242c")
X, Y = cv.xy()
body = sdf.fill(sdf.sd_ellipse(X, Y, 0.5 * W, 0.53 * H, 0.34 * W, 0.33 * H))
flow = np.degrees(np.arctan2(Y - 0.33 * H, X - 0.5 * W))  # hair combed away from the head
fill(cv, body, "fur", flow=flow, palette=["#1a1008", "#6b4424", "#c48a52", "#f0d2a0"], shadow=0.6)
```

**Glowing materials** — lava, fire, stars and caustics carry HDR emission; finish with `soft_clip`.

```python
from pixelcanvas import Canvas
from pixelcanvas.texture import texture

W, H = 960, 540
cv = Canvas(W, H)
cv.paint(texture("lava", H, W, seed=4, heat=1.2))
cv.bloom(0.9, 10, 0.3).soft_clip(0.85)
```
