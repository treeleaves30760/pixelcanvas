# Elements: fire, smoke, clouds, water, weather, sky, light

`pixelcanvas.elements` draws natural phenomena with one call each. Every call paints through `cv.paint` with its
own alpha (one editable layer per `log` step), is deterministic by `seed`, and scales with the canvas
(`at=(u, v)` fractions, `size` in 1080p pixels). Emissive effects go HDR, so finish with bloom + `soft_clip`.
Options are checked: an unknown option name raises a TypeError that names the closest valid one (`tags` -> "did you mean 'tag'?"), and both spellings are accepted (`colour` / `color`, `centre` / `center`, `w` / `width`, `alpha` / `opacity`, ...). Unknown names of elements and sky kinds raise a ValueError with the closest valid names.


```python
from pixelcanvas import Canvas
from pixelcanvas.elements import sky, stars, moon, lake, campfire, illuminate
cv = Canvas(480, 270)
sky(cv, "night", horizon=0.55); stars(cv, horizon=0.55); moon(cv, at=(0.75, 0.25), size=40)
lake(cv, horizon=0.55)
fire = campfire(cv, at=(0.4, 0.88), size=200, seed=1)
cv.bloom(0.9, 12, 0.3).soft_clip(0.82)
```

Every call returns an `FX`: `fx.mask` (what it covered), `fx.lights` / `fx.light` (point lights). Pass lights to
`illuminate(cv, lights, mask=...)` to warm the rocks next to a fire or tint a wall under a lamp. Masks feed other
calls: `storm_clouds(...).mask` → `lightning(clouds=)`, `fog(...).mask` → `rainbow(mask=)`.

## Keyword registry (the shortest path)

```python
from pixelcanvas import Canvas
from pixelcanvas.elements import element, names, find
cv = Canvas(320, 180)
element(cv, "sky", kind="dusk")
element(cv, "fireworks", at=(0.5, 0.35), size=160, seed=2)
element(cv, "aurora", at=(0.5, 0.3), size=200)      # at + size place full-frame effects too
print(find("rain on a window"))     # ['droplets', 'rain', ...]
print(len(names()))                  # every keyword
```

`element(cv, name, at=(u, v), size=px, seed=n, **options)` accepts a registry key or free text (best match).

## Calls

`at` is a point as (u, v) fractions; `region` / `box` = (u0, v0, u1, v1) fractions; sizes are 1080p px. A position given
in pixels (`at=(640, 360)`) raises a ValueError that says positions are canvas fractions.

| Group | Call | `at` / `size` mean | What you get |
|---|---|---|---|
| sky | `sky(cv, kind, horizon=1)` | — | gradient: noon day morning golden sunset dawn dusk twilight night storm overcast underwater |
| | `stars(cv, density, milky_way=True, horizon=None)` | — | power-law stars, spikes on the brightest, Milky Way with dust lanes |
| | `moon(cv, at, size, phase=0.5)` | centre / radius | maria, craters, terminator (0 new · 0.25 first quarter · 0.5 full) |
| | `sun(cv, at, size, rays=0.35, flare=False)` | centre / radius | HDR disc, layered glow, streak rays, optional lens flare |
| | `aurora(cv, v_range, colors, curtains=2, height=300)` | — / ray height | folded curtains, bright folds, rays fading green → violet |
| | `rainbow(cv, center, radius=0.8, mask=None)` | radius as fraction of height | spectral bow + Alexander band; `mask` keeps it in the spray (`waterfall(...).spray`) |
| | `shooting_star(cv, start, end)` · `god_rays(cv, at, occluder=mask)` · `heat_haze(cv, region=)` | | meteor · beams through gaps · shimmer of what is painted |
| clouds | `cumulus(cv, at, size, light=(dx, dy))` | base centre / width | cauliflower puffs, flat shaded base, lit side and silver rim |
| | `storm_clouds(cv, region, layers=3, backlight=(u, v))` | — | dark layered banks, lumpy undersides, glowing edges |
| | `cirrus(cv, region, angle)` · `cloud_bands(cv, region, colors)` | — | combed wisps · sunset stratus lit from below |
| vapour | `smoke(cv, at, size, wind, spread, angle, curl, softness, phase)` | source / column length | laminar at the source, billowing higher up, thins as it spreads |
| | `steam(cv, at, size, wisps=3, color=None, opacity=None)` · `chimney_smoke(cv, at, wind=0.9)` · `breath(cv, at, direction=70)` | source / length | cup steam · wind-bent smoke · cold breath puff; steam and breath take their tint from what is behind them (white on dark grounds, a soft darker edge and more body on cream walls and white jackets); `color=` and `opacity=` override |
| | `fog(cv, v_range, density, layers=3)` · `dust_motes(cv, n, mask=beam)` | — | depth-layered banks · specks only where light falls |
| fire | `flame(cv, at, size, lean, tongues=5, blue_base=0)` | base / height | licking tongues, white core → orange → red tips (`FIRE` palette) |
| | `campfire(cv, at, size)` | base / flame height | teepee logs, stone ring, coals, flames, embers, ground glow + light |
| | `candle(cv, base=(u, v), size, body_height, light_on=mask)` | foot of the wax (or `at` = wick) / flame height | teardrop flame, melted pool, glowing wax, contact shadow; `light_on` warms the sill or table |
| | `torch(cv, at, size, angle)` · `fireplace(cv, box)` · `lantern(cv, at, size, style)` | | wrapped torch · brick hearth · glass or paper lantern |
| | `explosion(cv, at, size)` · `sparks(cv, at, n, direction, gravity)` · `embers(cv, at, n)` · `fireworks(cv, at, size, kind)` | | fireball · streaks · rising specks · peony / willow / ring / chrysanthemum |
| | `stone(cv, at, size, lit_by=fire)` · `firewood(cv, start, end, width, lit_by=fire)` | centre / radius | faceted rock · bark-grained log with ringed cut end, both warmed by a fire |
| water | `ocean(cv, horizon, sun=(u, v), swell, chop, foam, camera)` | — | perspective sea: sky reflections, lacy whitecaps, glitter path |
| | `lake(cv, horizon, ripple, streak)` · `puddle(cv, at, size)` · `wet_ground(cv, mask, axis, smear)` | | mirror with glints · pool · streaky wet street |
| | `reflect(cv, of, water=None, axis=None, strength, ripple, streak, reach, glow)` | `of` = FX, mask, box or a list | mirror a light or object painted after the lake (campfire, tent, boat, lantern) into the water, with ripples, a glitter path and fade with distance |
| | `breaking_wave(cv, at, size, direction=1)` | foot of the face / wave height | curling lip over a shadowed tube, white water, spray |
| | `waterfall(cv, box, mist_height=0.35, fall=True)` | | glassy brink, aerating ropes, white water and mist; `.spray` = the mist mask |
| | `splash(cv, at, size)` · `ripples(cv, at, size)` | impact point / crown radius | crown wall breaking into beaded fingers · rings |
| | `river_flow(cv, mask, angle)` · `underwater(cv)` · `caustics(cv, mask)` | | flow lines · shafts, sand, caustics, bubbles · light web |
| | `droplets(cv, region, density=1, size=16, fog=0.35, trails=True)` | size = drop radius | fine drizzle beads, lens drops showing the scene upside down, runs clearing trails |
| | `rain(cv, density, angle, ground=v, splashes=60, sheets=n)` | | streak layers, crown splashes on the ground, distant curtains |
| weather | `snowfall(cv, density, wind, bokeh=True)` · `snow_cap(cv, mask, depth)` | | depth-layered flakes · snow on every top edge of a mask |
| | `lightning(cv, start, end, clouds=mask, branches=7)` | | jagged forked bolt, glow, lights the clouds |
| light | `glow(cv, at, radius, color)` · `lamp(cv, at, size)` · `neon(cv, text="OPEN" or points=[...])` | | HDR glow · pendant lamp with cone · glass tubes with white core |
| | `window_light(cv, box, floor=(v0, v1, shift))` · `bokeh(cv, n, region, size, shape="hex")` | size = disc radius | lit window + floor patch · out-of-focus discs |
| | `spotlight(cv, at, to, angle)` · `lens_flare(cv, at)` · `volumetric_fog(cv, lights)` | | beam with dust and pool · ghosts + streak · hazy halos |
| tools | `illuminate(cv, lights, mask)` · `light_map(cv, lights)` · `worley(X, Y)` · `voronoi(cv, scale)` · `curl(X, Y)` · `advect(X, Y)` | | firelight on surroundings · cells + edge distance · flow noise |

Draw order matters for anything that reflects or refracts: `lake`, `ocean`, `puddle`, `wet_ground`, `droplets`,
`ripples` and `heat_haze` read the current canvas, so paint the sky, moon and shore first. A light or an object painted
after the lake gets its reflection from `reflect(cv, of=..., water=lake_fx, axis=v)` (see the campfire recipe).

## Recipes

Candle on a windowsill, rain running down the glass, city lights outside:

```python
from pixelcanvas import Canvas
from pixelcanvas.elements import bokeh, droplets, candle
cv = Canvas(270, 338)
cv.gradient([(0.0, "#0b1020"), (0.55, "#1d2140"), (0.8, "#5a3a40")])
bokeh(cv, n=60, region=(0, 0.35, 1, 0.78), size=50, colors=["#ffb460", "#a0c8ff", "#ffe0a0"])
cv.blur(4)                                               # the city is out of focus
droplets(cv, region=(0, 0, 1, 0.78), seed=2)             # beads, drops and trails on the glass
X, Y = cv.xy()
sill = (Y > 0.78 * cv.h).astype("float32")
cv.paint("#2a1e16", sill)
candle(cv, base=(0.62, 0.82), size=80, body_height=180, light_on=sill)   # stands on the sill, lights it
cv.bloom(0.9, 12, 0.3).soft_clip(0.82)
```

Waterfall into a pool with mist and a rainbow in the spray:

```python
from pixelcanvas import Canvas
from pixelcanvas.elements import sky, waterfall, lake, fog, rainbow, god_rays
cv = Canvas(480, 270)
sky(cv, "morning")
cv.paint("#2c3028", (cv.xy()[1] > 0.15 * cv.h).astype("float32"))   # the cliff
box = (0.42, 0.15, 0.56, 0.76)
waterfall(cv, box=box, seed=1)
lake(cv, horizon=0.76, color="#0c2a26", tint="#bfe0d8")              # reflects the fall
fall = waterfall(cv, box=box, seed=1, fall=False)                    # white water on top of the pool
fog(cv, v_range=(0.55, 0.86), color="#e6efee", density=0.45)
rainbow(cv, center=(0.5, 1.05), radius=0.36, mask=fall.spray, secondary=False)   # the bow lives in the mist
god_rays(cv, at=(0.6, 0.0), strength=0.5)
cv.soft_clip(0.82)
```

Campfire by a lake (the full version is `scenes/elements_campfire.py`):

```python
import numpy as np
from pixelcanvas import Canvas
from pixelcanvas.elements import sky, stars, moon, lake, campfire, stone, illuminate
cv = Canvas(480, 270)
sky(cv, "night", horizon=0.5); stars(cv, horizon=0.5, seed=1); moon(cv, at=(0.75, 0.32), size=40)
lake(cv, horizon=0.5, ripple=1.5, streak=2.5)
X, Y = cv.xy()
ground = np.clip(Y - 0.74 * cv.h + 0.5, 0, 1)
cv.paint("#0e0d0d", ground)
fire = campfire(cv, at=(0.36, 0.88), size=300, seed=4)
stone(cv, at=(0.62, 0.92), size=40, lit_by=fire, seed=5)
illuminate(cv, fire, mask=ground, strength=2.0)
cv.bloom(0.9, 12, 0.3).soft_clip(0.8)
```

A campfire and a tent painted after the lake still reflect: `reflect` mirrors them about the waterline `axis` they
stand on, into the `water` the lake returned:

```python
import numpy as np
from pixelcanvas import Canvas, sdf
from pixelcanvas.elements import sky, stars, lake, campfire, reflect
cv = Canvas(480, 270)
sky(cv, "night", horizon=0.55); stars(cv, horizon=0.55)
water = lake(cv, horizon=0.55)
X, Y = cv.xy()
cv.paint("#10151c", (Y > 0.78 * cv.h).astype("float32"))                       # the near shore
fire = campfire(cv, at=(0.3, 0.8), size=180, seed=2)
tent = sdf.fill(sdf.sd_polygon(X, Y, [(0.6 * 480, 0.8 * 270), (0.82 * 480, 0.8 * 270), (0.71 * 480, 0.6 * 270)]))
cv.paint("#d8883a", tent * 0.8)
reflect(cv, of=[fire, tent], water=water, axis=0.8)                             # flames and tent in the water
cv.bloom(0.9, 12, 0.3).soft_clip(0.8)
```

Thunderstorm at sea (`scenes/elements_storm.py`):

```python
from pixelcanvas import Canvas
from pixelcanvas.elements import sky, storm_clouds, lightning, ocean, rain
cv = Canvas(480, 270)
sky(cv, "storm", horizon=0.56)
clouds = storm_clouds(cv, region=(0, 0, 1, 0.5), seed=7, backlight=(0.62, 0.3))
lightning(cv, start=(0.6, 0.2), end=(0.65, 0.56), clouds=clouds.mask, seed=8)
ocean(cv, horizon=0.56, swell=2.2, chop=1.4, foam=0.85, camera=2.2, crest=0.42, seed=9)
rain(cv, density=2.0, angle=14, opacity=0.3, sheets=6, sheet_band=(0.34, 0.56), sheet_color="#262b33")
cv.soft_clip(0.8)
```

Snowy night village — snow on every roof edge, falling snow in front:

```python
from pixelcanvas import Canvas, sdf
from pixelcanvas.elements import sky, snow_cap, snowfall, window_light
cv = Canvas(480, 270)
sky(cv, "twilight")
X, Y = cv.xy()
house = sdf.fill(sdf.sd_polygon(X, Y, [(140, 150), (240, 80), (340, 150), (320, 150), (320, 260), (160, 260), (160, 150)]))
cv.paint("#3a2a24", house)
window_light(cv, box=(0.45, 0.62, 0.55, 0.8))
snow_cap(cv, house, depth=14)
snowfall(cv, density=1.2, seed=3)
```

## Tips

- Sizes are 1080p pixels: `size=240` is a quarter of the picture height at any resolution, and particle counts
  (stars, rain, snow, drops) follow the frame, so a `--scale 0.5` draft looks like the final.
- Fire, lamps, neon, lightning and the sun are HDR: add `cv.bloom(0.9, 12, 0.3)` and `cv.soft_clip(0.8)` at the end
  (or `scenery.finish`).
- `campfire`, `flame`, `candle`, `lamp`, `lantern`, `window_light`, `lightning`, `glow` return lights: feed them to
  `illuminate` with the mask of the ground, rocks or walls they should light, or to `stone(lit_by=)` / `firewood(lit_by=)`.
  `candle(light_on=mask)` does it in one step and leaves the wax itself untouched.
- `stars(milky_way=True)` records two entries, "stars" and "Milky Way" (alias stars, night sky); every call in this
  module records itself, so `--check` lists what you drew without extra `manifest.note` calls. Sky, stars and aurora
  are marked as backdrops, water surfaces and vapour as see-through, so they never count as the subject or hide
  what lies under them.
- `smoke(softness=3)` gives a soft vapour puff; `curl` adds turbulence; `opacity` is the density at the source.
- Animate smoke with `smoke(phase=T)` (T from 0 to 1 drives the turbulence once around a circle, so a `--loop` animation
  closes seamlessly) or slide the noise with `offset=(ox, oy)`; the column, width and wind stay where they are.
- `voronoi`, `worley` and `curl` are the building blocks behind cells (cobbles, caustics, craters) and flow (smoke,
  flame licks); use them for your own textures.
