# Plants: trees, shrubs, flowers, potted plants, leaves, rocks (pixelcanvas.flora)

Every plant is one call and is built from the structure that makes it recognisable: trees grow a tapering trunk with
root flare and a seeded branch skeleton, and their canopy is hundreds of leaves grouped into clusters that are shaded
as volumes (lit side, shadow side, cast shadows between clumps, sky holes, a leafy edge). Flowers are petals in polar
layers with veins and a real centre; rocks are faceted solids with cracks; grass is tapered blades with direction.
Options are checked: an unknown option name raises a TypeError that names the closest valid one (`tags` -> "did you mean 'tag'?"), and both spellings are accepted (`colour` / `color`, `centre` / `center`, `w` / `width`, `alpha` / `opacity`, ...). Unknown names of plants, species, flower, leaf, rock and potted kinds raise a ValueError with the closest valid names.


```python
from pixelcanvas import Canvas
from pixelcanvas.flora import plant, tree, flower, names, find

cv = Canvas(960, 540, "#cfe0ea")
cv.paint("#7f9a58", cv.xy()[1] > 470)                       # a strip of lawn
tree(cv, "oak", at=(0.3, 0.92), height=0.85, seed=3, shadow=True)
plant(cv, "sunflower", at=(0.72, 0.94), height=0.45, seed=1)
plant(cv, "potted monstera", at=(0.88, 0.94), height=0.3)
```

**Units.** `at=(u, v)` is the base point in canvas fractions (where the plant touches the ground); pixels divided by the
canvas size give fractions, and a position that is clearly pixels raises a ValueError that says so. `height`, `size`,
`width` are fractions of the canvas height. Texture sizes are 1080p pixels scaled by `H / 1080`, so a 0.5 draft
looks like the full render. Same `seed` → same plant.

**Every call** paints through `cv.paint` (one layer per `log` step) and returns a `Plant`: `.mask` (H, W),
`.base`, `.top`, `.canopy` (centre of the foliage / flower head), `.bbox`, all in canvas pixels. Common options:
`seed`, `light=(dx, dy)` (direction towards the light, default upper left `LIGHT`), `shadow=True` or a strength
(ground shadow cast away from the light; on by default for potted plants, rocks, pumpkins, mushrooms and topiary), `clip=mask`, `haze=(colour, amount)`, `paint=False` (build only;
`p.patch` holds the RGBA for compositing).

## The keyword registry (start here)

| Call | What it does |
|---|---|
| `plant(cv, name, at=(u, v), height=h, seed=0, **options)` | paint any preset; free text works: `"weeping willow"`, `"mossy boulder"`, `"potted snake plant"` |
| `names(kind=None)` | list presets; kinds: `tree`, `shrub`, `ground`, `flower`, `potted`, `leaf`, `rock` |
| `find("a big oak tree")` → `"oak"` | best preset for a phrase (None if nothing matches) |

Presets — trees: oak, old oak (thick gnarled trunk, wide crown), maple, birch, poplar, cypress, spruce, fir, pine, willow, cherry, apple, olive, baobab, bonsai,
palm, coconut palm, winter (bare), snowy tree, dead, autumn maple / oak / birch, cute, bamboo, saguaro, prickly pear,
banana. Shrubs and ground: bush, flowering bush, berry bush, hedge, topiary ball / cone / spiral / tiers / poodle,
grass, dry grass, reeds, cattails, fern, mushroom, mushrooms, pumpkin, lily pads. Flowers: rose, tulip, sunflower,
daisy, poppy, cosmos, lily, orchid, lavender, dandelion (puff), dandelion flower, lotus, cherry blossom, cornflower,
marigold, hibiscus, peony, bouquet, vase. Potted: potted monstera, snake plant, succulent, ficus, pothos (hanging),
fern, cactus, aloe, flowering. Leaves: ovate, oak, maple, ginkgo, monstera, palm, birch, heart (`"maple leaf"`).
Rocks: rock, mossy rock, sandstone, rock pile.

## Calls

| Function | Key options |
|---|---|
| `tree(cv, species, at, height, seed, season=None, fullness=1, age=1, lean=0, palette=None, bark=None, style="painted", shadow=False, log=None, haze=None, fruit=None, snow=None, params=None)` | season `"spring"` (blossom on cherry and apple), `"autumn"`, `"winter"`: every deciduous species (oak, maple, birch, poplar, cherry, apple, willow) honours it, winter is bare branches with a little snow (`snow=False` turns it off), pine / spruce / fir / cypress / olive stay green; a misspelt species such as `"oakk"` raises a did-you-mean error; style `"painted"`, `"flat"`, `"cute"`; `lean` −1~1 wind; `palette` = name in `PALETTES` or a 5-colour ramp (dark → light) or a list of ramps; `log=log` paints trunk and canopy as two layers; `params` overrides preset numbers (`trunk_h`, `spread`, `cluster`, `depth`, ...) |
| `forest(cv, region=(u0, v0, u1, v1), n=30, species=("oak", "maple", "birch"), height=(far, near), haze=(colour, amount))` | depth-sorted trees: far ones small and hazy, near ones big; `avoid=mask` |
| `palm`, `bamboo(width=)`, `saguaro(arms=)`, `prickly_pear(fruit=)`, `banana(fruit=)` | structured non-branching plants (same `at`, `height`, `seed`) |
| `bush(cv, at, width, height=None, flowers=None, berries=None)` | dome of leaf clusters; `flowers="#e85a8a"` → flowering shrub |
| `hedge(cv, box=(u0, v0, u1, v1), top=0.05, side=0.0)` | clipped box hedge: front face `box`, visible top depth, visible side width |
| `topiary(cv, at, height, shape="ball")` | ball, cone, spiral, tiers, poodle — in a terracotta pot |
| `grass(cv, at, width, height, blades=40)` / `grass(cv, region=..., tufts=60)` | tapered curved blades, dark base → light tip; `wind`, `dry=True` |
| `reeds`, `cattails`, `fern(fronds=9)`, `mushroom(kind="amanita", cluster=1)`, `pumpkin(size)`, `wheat(region)` | ground plants; mushroom kinds: amanita, porcini, chanterelle, button |
| `ivy(cv, path=[(u, v), ...])` / `ivy(cv, along=mask, hang=0.12)` | vine with lobed leaves along a path, or along a mask's top edge with tendrils down its face |
| `moss(cv, mask=m)` / `moss(cv, region=...)` | cushion clumps with a fuzzy lit surface |
| `lily_pads(cv, at, n=7, size=0.07, flowers=2)` | pads in perspective with notch and veins, water lilies |
| `flower(cv, kind, at, height, color=None, tilt=None, stem=True, leaves=None, spec=None)` | one flower; `stem=False` paints only the head at `at`; `spec=dict(petals, layers, shape, color, tip, base, width, cup, ruffle, center, center_r, center_kind)` builds any flower; shapes `round`, `obovate`, `pointed`, `spatula`, `notched`, `toothed`, `frilly`, `heart` |
| `flower_bed(cv, region, kinds=(...), n=40, height=(back, front))` / `meadow(cv, region, flowers=(...), density=1, grass_height=0.07)` | depth-sorted scatters (smaller at the back); in a meadow `grass_height` also sets the flower size (0.12~0.15 for a foreground meadow) |
| `bouquet(cv, at, size, kinds, n=11)` / `vase(cv, at, height, kinds, shape="vase", color=...)` | hand-tied bouquet with wrap and ribbon / arrangement in a glazed vase |
| `potted(cv, kind, at, height, pot="classic", pot_color=None)` | monstera, snake plant, succulent, ficus, pothos, fern, cactus, aloe, flowering; pots: classic, cylinder, bowl, urn, square, hanging |
| `leaf(cv, kind, at, size, angle=-90, color=None, tip=None, flip=1)` | one leaf with midrib and veins; `at` = leaf centre; `flip` < 0 shows the back |
| `falling_leaves(cv, n, region)` / `leaf_pile(cv, at, width, height)` | tumbling autumn leaves / a mound of fallen leaves |
| `branch_frame(cv, corner="tl", size=0.45, kind="leafy", dark=0, blur=0)` | foreground branch entering from a corner of the frame (not aimable); kinds: leafy, autumn, blossom, pine, bare |
| `perch(cv, a, b, seed, thickness=0.016, kind="leafy", twigs=3, droop=0.015)` | a plain branch from point `a` to point `b` (canvas fractions) that a bird sits on: bowed tapering limb, twigs and leaves below and at the ends, the top left clear. `seat = p.extra["seat"]` is a function: `seat(0.5)` returns the pixel (x, y) on top of the limb halfway along (0 at `a`, 1 at `b`); kinds: leafy, autumn, blossom, bare |
| `climber(cv, kind="bougainvillea", box=(u0, v0, u1, v1), drape="frame", density=1, size=0.022, color=None)` | a flowering climber draped over a wall or doorway: bougainvillea, wisteria (hanging racemes), ivy, jasmine; `drape` = `frame` (up one side, over the top, down the other: a door or window), `top` (a curtain from the top edge: a pergola), `wall` (vines wandering across the box); `along=mask` hangs it from the top edge of a mask; `color=` recolours the blooms |
| `rock(cv, at, size, kind="granite", moss=0)` / `rock_pile(cv, at, width, n=7)` | faceted boulder (granite, sandstone, slate, basalt, limestone, mossy) |
| `pebbles(cv, region, n=120, wet=0)` / `stepping_stones(cv, path, n=7, size)` / `cliff(cv, region, strata=12, edge="right", kind="sandstone", palette=None)` | pebbles; flat slabs in perspective; rock face with strata, ledges and joints. `palette` is a list of colours from dark to light (`palette=["#3a2a1c", "#7a5a3a", "#b8905e", "#e6c898"]`), or leave it out and pick `kind` (granite, sandstone, slate, basalt, limestone, mossy) |
| `cellular(h, w, size, seed, X=None, Y=None)` → `(f1, f2, cell_id)` | Worley cells: `f2 - f1` ≈ distance to the cell edge (mortar, cracks, scales) |
| `strokes(X, Y, angle, length, width, seed)` → `(coverage, id)` | short dashes aligned with `angle` (scalar or map): grass, needles, fur, rain |

## Recipes

**A park in depth.** Background forest first, hero tree last; one light direction for everything.

```python
from pixelcanvas import Canvas
from pixelcanvas.flora import forest, tree, bush, grass, LIGHT

cv = Canvas(960, 540, "#bcd3e4")
cv.paint("#6f8a44", cv.xy()[1] > 280)
forest(cv, region=(0.0, 0.5, 1.0, 0.58), n=30, species=("oak", "poplar", "birch"), height=(0.1, 0.2),
       haze=("#bcd3e4", 0.6), seed=1)
tree(cv, "maple", at=(0.7, 0.9), height=0.7, season="autumn", seed=4, shadow=True, light=LIGHT)
bush(cv, at=(0.3, 0.9), width=0.25, flowers="#f0a0c0", seed=2, shadow=True)
grass(cv, region=(0.0, 0.88, 1.0, 1.0), tufts=40, height=0.05, seed=3)
```

**Hill, old oak, wildflowers.** Ground first, the hero tree on the hill, a meadow across the foreground.

```python
import numpy as np
from pixelcanvas import Canvas, clamp01
from pixelcanvas.flora import plant, meadow, grass

cv = Canvas(960, 540)
X, Y = cv.xy()
cv.gradient([(0.0, "#6fa5da"), (0.6, "#bcd8ee"), (1.0, "#efe6cc")])
hill = 0.58 + 0.2 * np.exp(-((X / cv.w - 0.35) / 0.4) ** 2)          # a hill peaking at 35% of the width
cv.paint("#6f9a46", clamp01((Y / cv.h - (1.15 - hill)) * cv.h / 1.5))
plant(cv, "old oak", at=(0.35, 0.72), height=0.62, seed=3, shadow=True)
meadow(cv, region=(0.0, 0.78, 1.0, 1.0), grass_height=0.13, density=1.2, seed=2)
grass(cv, region=(0.0, 0.94, 1.0, 1.02), tufts=60, height=0.1, seed=4)
```

**Deck with palms, pots and a corner bloom.** Pots stand on the deck (they cast their own shadow), a bush with
head-only blooms fills the foreground corner.

```python
import numpy as np
from pixelcanvas import Canvas, clamp01
from pixelcanvas.flora import tree, potted, flower, bush

cv = Canvas(960, 540)
X, Y = cv.xy()
cv.gradient([(0.0, "#4fb6e8"), (0.5, "#a8e0f0"), (0.55, "#e8f4f4")])
cv.paint("#1fa3b8", clamp01((Y - 0.5 * cv.h) / 2) * (Y < 0.62 * cv.h))                    # sea
cv.paint("#e6cf9c", clamp01((Y - 0.62 * cv.h) / 2))                                       # sand
cv.paint("#9a6b3e", clamp01((Y - 0.74 * cv.h) / 2))                                       # deck
tree(cv, "palm", at=(0.2, 0.8), height=0.7, seed=1, shadow=True)
tree(cv, "coconut palm", at=(0.78, 0.78), height=0.65, seed=2, shadow=True)
potted(cv, "monstera", at=(0.38, 0.9), height=0.25, seed=1)
potted(cv, "cactus", at=(0.5, 0.9), height=0.22, seed=2, pot="cylinder")
potted(cv, "snake plant", at=(0.62, 0.9), height=0.28, seed=3, pot="square")
bush(cv, at=(0.97, 1.08), width=0.5, height=0.42, seed=3, shadow=False)                   # foliage behind the blooms
flower(cv, "hibiscus", at=(0.9, 0.84), height=0.3, stem=False, seed=4)                    # stem=False: head only, at = centre
flower(cv, "hibiscus", at=(0.8, 0.95), height=0.2, stem=False, seed=5)
```

**Editable layers.** Pass the scene's `log` so trunk and canopy become separate layers in `render.py --layers`.

```python
from pixelcanvas import Canvas
from pixelcanvas.flora import tree

steps = []
cv = Canvas(480, 270, "#dfe8ee")
p = tree(cv, "cherry", at=(0.5, 0.95), height=0.9, seed=2, log=steps.append)
print(steps, p.canopy)                       # ['cherry trunk', 'cherry canopy'] (x, y) of the blossom cloud
```

**Styles and seasons.** The same species in three looks and four seasons.

```python
from pixelcanvas import Canvas
from pixelcanvas.flora import tree

cv = Canvas(960, 360, "#f3efe6")
for i, (style, season) in enumerate([("painted", "spring"), ("painted", "autumn"), ("flat", None),
                                     ("cute", None), ("painted", "winter")]):
    tree(cv, "maple", at=(0.1 + i * 0.2, 0.95), height=0.8, seed=5, style=style, season=season)
```

**Your own flower and a framed picture.** Build a flower from parts, then frame the picture with a branch.

```python
from pixelcanvas import Canvas
from pixelcanvas.flora import flower, branch_frame, rock

cv = Canvas(800, 450, "#e8efe4")
flower(cv, spec=dict(petals=9, layers=2, shape="pointed", color=["#7a3cc8", "#9a5ae0"], tip="#e0c8ff",
                     center="#f2c234", center_r=0.2, center_kind="dots"), at=(0.5, 0.95), height=0.6, seed=2)
rock(cv, at=(0.25, 0.97), size=0.3, moss=0.6, seed=3, shadow=True)
branch_frame(cv, corner="tl", size=0.5, kind="blossom", seed=4)
```

**A bird on a branch, a doorway in bougainvillea.** `perch` returns the seat point for the bird; `climber` drapes the
frame of a door.

```python
from pixelcanvas import Canvas, sdf
from pixelcanvas.flora import perch, climber
from pixelcanvas.creature import creature

cv = Canvas(960, 540, "#efe6d2")
X, Y = cv.xy()
cv.paint("#5a3a2a", sdf.fill(sdf.sd_box(X, Y, 0.22 * cv.w, 0.62 * cv.h, 0.1 * cv.w, 0.34 * cv.h)))      # the door
climber(cv, "bougainvillea", box=(0.12, 0.2, 0.32, 0.96), drape="frame", seed=2)
limb = perch(cv, (0.5, 0.62), (0.97, 0.55), seed=4, thickness=0.022)
seat = limb.extra["seat"]
x, y = seat(0.55)                                                        # a point on top of the limb
creature(cv, "sparrow", at=(x / cv.w, y / cv.h), size=0.13, pose="perch", shadow=False)
```

The botanical plate `scenes/flora_sheet.py` shows every preset; `scenes/flora_garden.py` is a full garden scene
(perspective flagstone path, hedge, beds, steps with pots) built from these calls.
