# solid — 3D renders in a few lines

`pixelcanvas.solid` is a ray-marching 3D renderer written in numpy. Describe objects in 3D and you get perspective,
soft shadows, ambient occlusion, reflections, glass, depth of field and anti-aliased edges. Every object is painted
as its own layer, and the render hands back depth / normal / id buffers and per-object masks.

Use it for anything that should look like a 3D render or a product photo: products, rooms, vehicles, toys,
characters, isometric dioramas, 3D logos, still lifes — and as a base layer for 2D illustration on top.

## Quick start

```python
from pixelcanvas import Canvas
from pixelcanvas.solid import Scene, Camera, car, sphere

sc = Scene(lighting="studio", ground="checker")
sc.add(car("#e8452c").rotate_y(-30), "car")
sc.add(sphere(0.6).place(x=3.4, z=-0.6), "ball", "chrome")
sc.camera = Camera.frame(sc, azimuth=30, elevation=15)
cv = Canvas(960, 540)
res = sc.render(cv)                     # one layer per object; res.mask("car"), res.depth, res.normal
cv.save("car.png")
```

Units are metres, y is up, the floor is y = 0, the default camera looks from +z. Build shapes around the origin,
then move them: `box(1, 0.5, 2).rotate_y(30).at(1, 0.25, 0)`, or `.place(x, z)` to stand anything on the floor.

## Calls

| Call | What it does |
|---|---|
| `sphere(r)` `ellipsoid(rx, ry, rz)` `box(w, h, d, r=0)` `rounded_box(w, h, d, r)` | basic solids (full sizes, centred) |
| `cylinder(r, h, round=0)` `cone(r1, r2, h)` `capsule(a, b, r)` `torus(R, r)` | upright along y; capsule between two points |
| `prism(points, depth, smooth=False)` `wedge(w, h, d)` | extruded 2D polygon (xy, along z); ramp |
| `lathe([(r, y), ...], smooth=True, thickness=None)` | profile revolved around y: bottles, vases, lamps, chess pieces |
| `extrude(f2d, depth, size)` `extrude_text("Hi", height, depth)` | any 2D SDF (e.g. `pixelcanvas.sdf`) or text, with depth |
| `a | b` `a - b` `a & b` `union(...)` `subtract(a, b)` `intersect(a, b)` | booleans |
| `a.blend(b, k)` `smooth_union(a, b, ..., k=)` `a.carve(b, k)` | organic joins and rounded cuts |
| `.round(r)` `.shell(t)` `.cut(normal, offset)` | inflate, hollow, half-space cut |
| `.mirror("x")` `.repeat(spacing, count)` `.twist(deg)` `.bend(deg)` | symmetry, arrays, deformations |
| `.at(x, y, z)` `.rotate_x/y/z(deg)` `.rotate(x, y, z)` `.scale(s)` `.place(x, z, y=0)` | chainable transforms (shapes and models) |
| `.on(model, part=None, at=(0.5, 0.5), level=None, xz=None)` | rest on top of another placed model or one of its parts, touching it: `cat.on(chair, "armrest_r")` |
| `material("chrome")` `material("plastic", "#2a9df4")` `material("#e33")` | presets + colour + overrides (`roughness=`, …) |
| presets | matte plastic glossy paint ceramic metal chrome gold copper glass rubber clay wood marble stone skin fabric leaf emissive |
| `material("wood", "#7a4a2c")` `("marble", "#eee")` `("fabric", "#345")` `("stone", "#888")` | grain / veins / weave / speckle in your colour |
| `material("stone", moss=0.6)` `material("wood", moss=0.4, moss_color="#5a7a30")` | moss on the upward-facing sides of any material (0 bare ~ 1 covered) |
| `wood()` `marble()` `stone()` `moss(base, amount)` `planks()` `weave()` `stripes()` `checker()` `noise3()` `fbm3()` | procedural 3D colour; any `f(P, N) -> colours` works |
| `Scene(lighting, ground, camera, background)` | ground: colour, preset (`("wood", "#d9b48c")`), `"checker"`, `"backdrop"`, None |
| `sc.add(obj, name, material)` | add a Shape or Model → its own layer |
| `sc.floor(color, checker=, size=, texture=, reflect=)` `sc.backdrop(color, distance, radius)` | floor / seamless studio sweep |
| `Lighting(preset, **options)` `sc.light("sunset")` | studio soft_box sunny sunset overcast night dramatic interior |
| `lit.point(pos, color, power, radius)` `sc.light_from(part)` | point lights (lamp bulbs, candles, windows) |
| `Camera(position, look_at, fov)` `Camera.orbit(target, distance, azimuth, elevation)` | perspective |
| `Camera.frame(obj, azimuth, elevation, fill)` `Camera.isometric(sc)` | fit to the real silhouette of any canvas shape; isometric fitted to the scene |
| `sc.render(cv, log, aa, box, quality, background)` | paint into the canvas; returns `RenderResult` |
| `res.mask(name)` `res.depth` `res.normal` `res.ids` `res.position` | AA object masks ("car", "car.wheels"), buffers |
| `res.window_mask()` | on-screen mask of the visible window glass of a `room()`: clip 2D rain, reflections, a painted view |
| `names(kind)` `find("reading lamp")` `describe("armchair")` | prefab registry (kinds furniture product light outdoor vehicle tableware ...) |
| `sc.project((x, y, z))` / `res.project(...)` | 3D point → canvas pixel for labels and 2D detail |
| `res.depth_of_field(cv, focus="speaker", blur)` `res.fog(cv, color, density, start)` | post steps from the depth buffer (focus: a name, distance or point) |

Lighting angles are relative to the camera: azimuth 0 = light from behind the camera, -90 = from the left, 180 =
backlight. `Lighting(..., relative=False)` uses world angles (0 = from +z, 90 = from +x). Options: `sun=(az, el)`,
`sun_color`, `sun_power`, `softness` (0 hard → 1 very soft shadows), `ambient`, `zenith`, `horizon`,
`ground_color`, `fill_power`, `rim_power`, `exposure`. `"interior"` is the warm, bright room light: its sun comes
through the window on the left wall of a `room()`, whatever the camera does.

`render(..., background=None)` shows the sky / environment; pass a colour, a vertical gradient `[(0, top), (1,
bottom)]`, or `False` to keep what is already on the canvas. `quality="draft"` (auto below 700 px tall) or
`"final"`; `aa` = samples per silhouette pixel (4 draft, 8 final). `box=(x0, y0, x1, y1)` frames the render in a
canvas region; a box that reaches past the canvas edge renders the part that is on the canvas.

## Prefabs (each returns a Model with named parts, standing on the floor, facing +z)

| Prefab | Parts / options |
|---|---|
| `mug(color, coffee=True)` `cup(color, saucer=True)` `bottle(color, liquid, level, label, cap, height)` `vase(color, height, shape)` | tableware |
| `speaker(color, fabric, kind="round"/"pill"/"box")` `laptop(screen)` `coffee_machine(color)` | products |
| `table(width, depth, height, round_top)` `chair(color, seat)` `sofa(color, width)` `bed(width, length, duvet, throw)` | furniture |
| `armchair(color, width)` `side_table(height, round_top, shelf)` | parts seat, back, armrest_l, armrest_r / top, legs, shelf |
| `desk_lamp(color, glow)` `pendant_lamp(height, drop, shade)` `floor_lamp(shade, glow, style="drum"/"cone"/"globe")` | lights: `sc.light_from(lamp)` lights from the bulb |
| `bookshelf(width, height, shelves)` `book(...)` `rug(width, depth, shape="rect"/"round"/"oval", pattern)` `potted_plant()` | room props |
| `room(width, depth, height, walls, walls2, floor, window="left", door="back", view=)` `counter(width)` | interiors; `view` = rain, night city, city, garden, snow, sunset, a colour or `f(u, v)` |
| `earbuds()` `charging_case(open=100, buds=True)` `smartphone(screen=, lying=)` `smartwatch(face=)` | small electronics (screens take a colour or `f(u, v)`) |
| `tent(lit=True)` `stone_lantern(lit=)` `arched_bridge(length, height)` `rock(size, color, moss=0.5)` | outdoor: campsite, Japanese garden (`lit` adds a part to `light_from`), boulders |
| `wall(width, height, thickness, openings=[(x, bottom, w, h)])` `window_frame(width, height, glow)` | walls with holes |
| `house(walls, roof, door, windows)` `tree(height, kind="round"/"pine"/"bush")` | architecture, nature |
| `tree(height, kind="maple"/"oak", season="summer"/"autumn", leaves=, seed=)` | limbs under a canopy of leaf clumps, each its own tone; `leaves` = a colour or a list |
| `car(color, style="hatch"/"sport"/"van")` `truck(color, cargo)` `rocket(body, accent, flame)` | vehicles (point along +x) |
| `robot(color, accent, eyes)` `character(skin, shirt, pants, hair, pose="wave")` `quadruped(color, kind="dog"/"cat"/"bear"/"sheep")` | characters |
| `toy_block(size, color, letter)` `model(name, part=(shape, material), ...)` | toys; your own models |

Models chain like shapes (`tree().scale(1.5).place(x=3)`), and `m.recolor(body="#3a7")` (wood, marble and fabric
keep their pattern), `m.material(body="gold")`, `m["wheels"]`, `a + b` customise them. `pendant_lamp()` hangs from
its ceiling height, so `.place(x, z)` keeps it up there.

**Facing and turning.** Every prefab and clay figure faces +z. `rotate_y(a)` turns it to face `(sin a, 0, cos a)`:
90 faces +x (to the right in the default view), -90 faces -x, 180 faces away; clay's `.face(a)` uses the same
angle. Part names ending `_l` / `_r` are the object's own left and right: a chair's `armrest_l` is at +x, on the
viewer's right in a front view.

**Putting one thing on another.** Add the base first, then rest the next object on what `sc.add` returned:
`chair = sc.add(armchair().rotate_y(-35).place(x=0.1, z=0.4), "armchair")`, then
`sc.add(animal("cat", pose="sit").face(-35).on(chair, "armrest_r"), "cat")`. `on` finds the surface under the object
and lifts it until its lowest point touches it: nothing sinks in, nothing floats. Surfaces: any part name, plus
`seat`, `top`, `armrest`, `shelf` (`level=1` is the highest shelf), `deck`; `at=(u, v)` slides along the surface.

## Recipe: product shot on a wooden table

```python
from pixelcanvas import Canvas
from pixelcanvas.solid import Scene, Camera, Lighting, speaker, cup

sc = Scene(lighting=Lighting("studio", softness=0.8), ground=("wood", "#d9b48c"))   # a light wooden tabletop
sc.add(speaker("#1c1c1e").rotate_y(20).place(x=-0.06), "speaker")
sc.add(cup("#f6f4f0").rotate_y(-40).place(x=0.1, z=0.06), "cup")
sc.camera = Camera.orbit(target=(0.02, 0.08, 0.02), distance=0.7, azimuth=12, elevation=14, fov=26)
cv = Canvas(1920, 1080)
res = sc.render(cv, background=[(0, "#e9e4dc"), (1, "#cfc4b6")])
res.depth_of_field(cv, focus="speaker", blur=14)            # shallow focus on the speaker
cv.save("product.png")
```

For a seamless studio sweep instead of a table: `Scene(ground=None)` then `sc.backdrop("#c3cdb9", distance=0.6,
radius=0.35)`.

## Recipe: isometric café (cut-away room)

```python
from pixelcanvas import Canvas
from pixelcanvas.solid import (Scene, Camera, room, counter, coffee_machine, pendant_lamp, table, chair,
                               potted_plant, cup)

sc = Scene(lighting="interior", ground=None)                 # warm sun through the left-wall window
sc.add(room(5, 5, walls="#f1dcc0", walls2="#e3b98e", window_size=(2.4, 1.5), door=("back", 1.5)), "room")
sc.add(counter(2.6).place(x=-0.8, z=-2.1), "counter")
sc.add(coffee_machine().place(x=-1.4, z=-2.15, y=1.0), "espresso machine")   # y = counter height
sc.add(cup().place(x=-0.5, z=-1.95, y=1.0), "cup")
for i, (x, z) in enumerate([(-0.9, 0.5), (1.1, 1.0)]):
    sc.add(table(0.8, round_top=True).place(x=x, z=z), f"table {i + 1}")
    sc.add(chair().rotate_y(90).place(x=x - 0.6, z=z), f"chair {i + 1}a")
    sc.add(chair().rotate_y(-90).place(x=x + 0.6, z=z), f"chair {i + 1}b")
    lamp = sc.add(pendant_lamp(drop=1.0).place(x=x, z=z), f"lamp {i + 1}")
    sc.light_from(lamp["bulb"], power=1.2, radius=1.0)
sc.add(potted_plant(1.1).place(x=-2.2, z=2.1), "plant")
sc.camera = Camera.isometric(sc)                             # centred and sized to fit the canvas
cv = Canvas(1600, 1200)
sc.render(cv, background=[(0, "#f6e6d0"), (1, "#e9cfb0")])
cv.save("cafe.png")
```

`room()`'s open interior spans x and z in ±width/2, ±depth/2 with the floor at y = 0; walls stand just outside, so
`.place(x=-width/2 + d/2)` puts furniture of depth `d` against the left wall.

## Recipe: reading nook with a cat on the armrest and rain at the window

```python
from pixelcanvas import Canvas, elements
from pixelcanvas.clay import animal
from pixelcanvas.solid import Scene, Camera, Lighting, room, armchair, floor_lamp, bookshelf, rug, side_table, mug

sc = Scene(lighting=Lighting("interior", sun_color="#cfdcec", ambient=0.6), ground=None)
sc.add(room(3.4, 3.4, 2.6, walls="#efe2cf", walls2="#c9d6dc", door=None, view="rain"), "room")
sc.add(rug(2.1, shape="round", color="#6f8f8a").place(x=0.1, z=0.35), "round rug")
sc.add(bookshelf(1.1, 1.9, 0.32).place(x=0.4, z=-1.53), "bookshelf")
chair = sc.add(armchair("#c4623f").rotate_y(-35).place(x=0.1, z=0.4), "armchair")
sc.add(animal("cat", coat="ginger", pose="sit").face(-35).on(chair, "armrest_r"), "cat")
sc.light_from(sc.add(floor_lamp().place(x=-1.05, z=0.0), "floor lamp"), power=1.4, radius=1.3)
table = sc.add(side_table().place(x=1.0, z=0.95), "side table")
sc.add(mug("#2f5d8a").on(table, "top"), "mug")
sc.camera = Camera.isometric(sc)
cv = Canvas(1600, 1200)
res = sc.render(cv, background=[(0, "#cdd8e0"), (1, "#a3b5c3")])
elements.rain(cv, density=3.0, angle=10, length=50, mask=res.window_mask(), splashes=0)   # stays on the glass
cv.save("nook.png")
```

The renderer records what it drew for lighting words in a request (`lighting`, `shadows`, `reflections`, the
`window` and its view), so `--check` counts "soft studio lighting with shadows" as answered.

## Recipe: autumn park with maples, an oak and a mossy rock

```python
from pixelcanvas import Canvas
from pixelcanvas.solid import Scene, Camera, tree, rock

sc = Scene(lighting="sunny", ground="#9fb07a")
sc.add(tree(4.0, kind="maple", season="autumn", seed=5), "red maple")
sc.add(tree(3.0, kind="oak", seed=3).place(x=-4.5, z=-3), "oak")                  # summer greens
sc.add(tree(2.6, kind="maple", leaves=["#e8b030", "#d98a2a"], seed=7).place(x=4.2, z=-2.5), "gold maple")
sc.add(rock(1.1, moss=0.7, seed=1).place(x=1.8, z=1.4), "mossy rock")
sc.camera = Camera.frame(sc, azimuth=-15, elevation=8)
cv = Canvas(960, 540)
sc.render(cv)
cv.save("park.png")
```

## Recipe: 3D objects over a 2D painting, with labels

```python
from pixelcanvas import Canvas
from pixelcanvas.solid import Scene, Camera, rocket, extrude_text, material

cv = Canvas(960, 540)
cv.gradient([(0, "#0b1a3a"), (1, "#3a2a5a")])              # any 2D background you painted
sc = Scene(lighting="night", ground=None)
sc.add(rocket(flame=True), "rocket")
sc.add(extrude_text("GO!", height=0.4, depth=0.12).rotate_y(-15).at(1.3, 1.5, 0), "title",
       material("plastic", "#ffcf6b", emissive=0.8))
sc.camera = Camera.frame(sc, azimuth=10, elevation=5)
res = sc.render(cv, background=False)                     # keep the painting behind the objects
x, y = sc.project((0, 2.5, 0))                            # nose of the rocket → pixel
cv.text("liftoff", (x + 24, y), 28, color="#ffe9b0", anchor="lm")
cv.save("poster.png")
```

## Tips

- **Ground contact**: `.place(x, z)` stands objects exactly on the floor, so contact shadows and AO look right.
- **Scale**: real sizes (a mug is 0.095 m, a car 4.2 m) keep prefabs consistent; scale whole models for toys.
- **Framing**: `Camera.frame(sc, azimuth, elevation)` fits the real silhouette of everything to whatever canvas
  it renders into (`fill=` for more or less margin); `Camera.isometric(sc)` does the same for isometric views.
- **Surfaces**: anything lying on the ground (a tabletop slab, a rug) shows cleanly on top of it; for a whole
  tabletop use the ground itself: `Scene(ground=("wood", "#d9b48c"))`.
- **Organic forms**: build from spheres / ellipsoids / capsules joined with `smooth_union(..., k=)`.
- **Speed**: drafts at 960x540 take 2~6 s; finals take 15~35 s for a full room at 1920x1080.
- **Editing**: run with `tools/render.py --layers`; every `sc.add` name is a layer, plus background and ground.

Options are checked: a misspelt option, material parameter, lighting option or part name (`mug().recolor(bodyy=...)`)
raises an error that names the closest valid one, and both spellings are accepted (`colour` / `color`, `radius` / `r`);
unknown presets, lighting names and enumerated values (`tree(kind=...)`, `render(quality=...)`) list the nearest names,
and so do part names (`chair["armrest_x"]`, `.on(chair, "armrst")`), `res.mask(...)`, window views and prefab names
(`solid.armchiar` -> did you mean `solid.armchair`).
