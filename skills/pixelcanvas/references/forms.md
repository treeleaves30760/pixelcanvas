# Objects with volume, shadows and perspective (pixelcanvas.form)

`pixelcanvas.form` turns flat shapes into solid objects that stand in a lit space: a mug on a table, oranges on a
shelf, a toy car on a checkered floor, a sofa in a room. Every object is one call; one light drives the shading and
the shadows of the whole picture.

```python
from pixelcanvas import Canvas
from pixelcanvas.form import set_light, tabletop, ball, cylinder, lathe

cv = Canvas(1920, 1080, "#cbb9a2")
set_light(direction=(-0.8, -0.5), elevation=32)                       # window on the upper left
table = tabletop(cv, top=0.5, front=0.95, horizon=0.38)               # wooden table in perspective
cylinder(cv, table.place(0.45, 0.4), 0.1, 0.21, base="#e9e1d3", material="ceramic",
         open=True, fill=0.86, fill_color="#3b1f12", handle=True, cast=True)   # a mug of coffee
ball(cv, table.place(0.65, 0.5), 0.09, base="#f07a18", material="citrus", cast=True)  # an orange
lathe(cv, table.place(0.3, 0.7), [(0.085, 0), (0.094, 0.03), (0.094, 0.26), (0.036, 0.4), (0.028, 0.53)],
      base="#5f9a72", material="glass", cast=True)                    # a glass bottle
cv.save("still_life.png")
```

Units: positions are canvas fractions `(u, v)`; sizes are fractions of the canvas height. `table.place(u, depth)`
returns a **Spot** (depth 0 = front edge, 1 = back edge) that carries the perspective scale, so the same size gets
smaller further back and every solid stands on the surface. A Spot also knows its surface: cast shadows stay on the
table top, and on a shelf or in a room they climb the wall behind. Masks and anchor points come back in pixels.

## Calls

| Call | What it gives you |
|---|---|
| `Frame(cv, center, size, rotate=0, flip=False, squash=(1,1), extent=1.5, flat=None)` | local grids `fr.X, fr.Y` (unit = size × height) for any `sdf` primitive; `fr.fill(d)`, `fr.paint(color, d)`, `fr.form(d, base, material=...)`, `fr.px(d)`, `fr.to_canvas(x, y)`, `fr.from_canvas(px, py)`, `fr.child(at, size, rotate)`; `flat=tilt` lays it on the table |
| `P(cv, u, v)`, `px(cv, f)`, `d_circle / d_ellipse / d_box / d_segment / d_polygon(cv, ...)` | fraction → pixel helpers; full-frame px distances from fraction arguments |
| `Light(direction, elevation, color, intensity, sky, ambient, bounce, shadow, front)` / `set_light(...)` / `get_light()` / `resolve_light(light)` | the scene light; every call with `light=None` uses it |
| `paint_form(cv, d, base, material="matte", profile="round", contact=0, **overrides)` | paint any distance field as a lit volume → mask |
| `shade(d, base, ...)` / `normals(d, profile, width, axis, bevel, height)` | the colour map / normal field without painting |
| `cast_shadow(cv, shape, ground_y=None, length=1, softness=1, opacity=0.7, surface=None)` | silhouette projected on the ground away from the light, widening penumbra, dark contact core; stays on its surface and climbs the wall behind |
| `contact_shadow(cv, at, width)` / `contact_shadow_px(cv, x, y, rx, ry, strength, softness)` | soft contact ellipse (fraction or pixel units), fades to zero at any softness |
| `drop_shadow(cv, shape, offset, blur)` / `occlusion_between(d_a, d_b)` / `multiply_shadow(cv, a, box)` | flat-design shadow / darkening where shapes meet / multiply any strength crop in the shadow tint |
| `reflection(cv, solid, strength=0.25, fade=0.6, blur=0)` | mirror a solid in a glossy table or floor (call right after drawing it) |
| `ball(cv, at, radius)` | sphere, or ellipsoid with `radius=(rx, ry)` |
| `cylinder(cv, at, radius, height, open=False, fill=None, handle=None)` | cans, mugs, glasses, pillars, table legs |
| `cone(cv, at, radius, height, top_radius=0)` | cones and frustums (lamp shades, pots) |
| `lathe(cv, at, [(radius, height), ...], open, inside, rim, fill, fill_color, top, handle)` | any round object from its profile: bottles, vases, bowls, lamps, pins |
| `cuboid(cv, at, (w, d, h), yaw=30, top=None, round=0, colors=None)` | three-quarter box; `round` makes cushions, toy blocks and car bodies; `colors` per face (`front`, `left`, `right`, `top`, `back`) |
| `book(cv, at, (w, d, thickness), cover, pages, spine="front", yaw=8)` | closed book lying flat: covers, page block, spine bands; stack with `spot.up(thickness)` |
| `disc(cv, at, radius, width, yaw=0, base, hub)` | a disc on its edge: wheels and tyres (same `yaw` as the car body), coins, tape rolls |
| `capsule(cv, a, b, radius)` / `torus(cv, at, radius, thickness)` | rods, stems and limbs / rings and donuts |
| `Spot.up(h)` / `Spot.move(dx, dz=0, yaw=0)` | raise a spot (stacking) / move it on the ground in size units, turned like a cuboid (wheel positions) |
| `Ground(cv, horizon, vanishing, camera_height)` | floor plane: `floor("checker", size, colors)` paints it in one call, melting into the sky at the horizon; `place(x, depth)` / `spot(u, v)` → Spot; `at`, `project(x, y, z)`, `uv()`; `checker()`, `tiles()`, `planks()`, `grid_mask()`, `mask()`; `rug(x0, x1, z0, z1, color, border, pattern)`; `box(x, depth, (w, d, h), yaw, lift, round)` |
| `Room(cv, horizon, vanishing, back, width, height, eye)` | one-point interior: `draw(floor="planks", window="left")`, `sun_beam()`, `place(x, z)`, `box(...)`, `rug(...)`, `picture(wall, center, size)`, `panel(wall, center, size, color, frame)` → 4 corners, `wall_quad(...)`, `wall_point(...)`, `ground`, `zb` (back-wall depth) |
| `tabletop(cv, top, front, horizon, kind="wood", wall=False)` / `shelf(cv, y, x0, x1, depth)` / `counter(cv, ...)` | surfaces for still lifes and product shots → `Surface` with `place(u, depth)` |
| `row`, `grid_positions`, `scatter_positions(n, region, min_gap, seed)`, `ring_positions` | `(points, order)`; draw `for i in order:` (back to front) |
| `Room.rug(x0, x1, z0, z1, color, border=None, pattern=None)` / `Ground.rug(...)` | a rug (a floor rectangle in perspective) with an optional border band and `diamonds` / `stripes` pattern → full-frame mask; lit by the room's window |
| `Room.panel(wall, center=(a, y), size=(w, h), color, frame=None)` | a wall-mounted rectangle (sign, menu board, screen, whiteboard) → its 4 pixel corners (top-left, top-right, bottom-right, bottom-left) for `cv.text` |

Every solid and surface takes `name="vase"` (and `alias="flower vase"`): it is the name the scene manifest (`--check`, `manifest.summary`) lists
the object under; without it the call name is used (`lathe`, `ball`). Options are checked: a misspelt option or
material parameter raises an error that names the closest one, and both spellings are accepted (`colour` / `color`,
`r` / `radius`, `w` / `width`); an unknown material, floor texture or profile lists the nearest valid names. Positions
(`at`, `center`, `box` ...) are canvas fractions; a value that is clearly pixels (`at=(640, 360)`) raises a ValueError that
says so (`form.check_frac` is the same check for your own helpers).

## Building blocks

The helpers `figure.py` and `creature.py` use are public, so your own solids can use them too (the old underscore
names stay as aliases):

| Name | What it does |
|---|---|
| `resolve_light(light=None)` | a `Light` from `None` (the scene light), a `Light`, a direction tuple or a dict |
| `clip_box(cv, x0, y0, x1, y1)` | pixel box clipped to the canvas as ints, or `None` |
| `full_frame(cv, a, box, fill=0)` (`full_mask`) | place a crop-sized array into a full-frame `(H, W)` array |
| `bbox_of(mask, thresh)` | tight pixel box of a mask, or `None` |
| `grid_box(x0, y0, x1, y1)` | pixel coordinate grids `(X, Y)` of a box |
| `sd_ellipse(X, Y, cx, cy, rx, ry, angle)` | signed pixel distance to an ellipse on those grids |
| `contact_px(cv, x, y, rx, ry, strength, light, color, softness)` | soft contact ellipse at a pixel position |
| `multiply(cv, a, box, color)` (`multiply_shadow`) | multiply a crop-sized strength map into the canvas in the shadow tint |
| `shadow_color(light, color)` | the shadow tint as an rgb array |

Every solid takes `base` (colour or full-frame colour map / texture), `material`, `light`, `shadow=True`
(contact shadow), `cast=False` (`True` or a dict of `cast_shadow` options) and returns a **Solid**: `s.mask`,
`s.box`, `s["base"]`, `s["top"]`, `s["center"]`, `s["left"]`, `s["right"]`, `s.uv("top")` (as fractions),
`s.point(t, side)` on lathes, `s.band(t0, t1, colour)` (labels, stripes), `s.recolor(base, where)`; cuboids add
`s["face_front"]`, `s["face_top"]` … (four pixel corners each, for labels).

**Materials** (`MATERIALS`): `matte`, `plastic`, `glossy`, `ceramic`, `metal`, `chrome`, `gold`, `glass`,
`rubber`, `skin`, `fabric`, `wood`, `clay`, `fruit`, `citrus` (pitted peel), `velvet`, `paper`, `leather`.
Override any parameter inline: `material="skin", sss=0.6` or `material=dict(preset="plastic", ks=0.9)`;
`emit=0.4` makes a lamp shade glow; `bump=0.3, bump_size=0.01` adds a fine surface relief, `mottle=0.08` a gentle
colour variation, `seed=` varies both between objects.

**Profiles** for `paint_form`: `round` (sphere-like; exact for circles and capsules), `dome`, `flat` (bevelled
rim, `bevel=` px), `cushion`, `ridge` (leaves, blades), `cylinder` (curved across `axis` degrees only: tubes, limbs,
bottles drawn as outlines). `height=-0.8` turns any of them into a dent: spoon bowls, dimples, wells.

## Recipe: a shelf against the wall

```python
from pixelcanvas import Canvas
from pixelcanvas.form import set_light, shelf, ball, lathe, book

cv = Canvas(1920, 1080)
cv.gradient([(0, "#f6efe4"), (1, "#e4d6c2")])                          # bright wall
set_light(direction=(-0.8, -0.5), elevation=35)
sh = shelf(cv, 0.7, 0.08, 0.92, depth=0.07)                            # board ending at row 0.7
s = sh.place(0.22, 0.5)
book(cv, s, (0.34, 0.24, 0.05), cover="#7a2a2a", cast=True)
book(cv, s.up(0.05), (0.3, 0.22, 0.045), cover="#2f5a7a", yaw=-4)     # stacked on the first one
vase = lathe(cv, sh.place(0.5, 0.5), [(0.05, 0), (0.08, 0.06), (0.085, 0.14), (0.04, 0.26), (0.045, 0.34)],
             base="#2f6fc0", material="ceramic", open=True, cast=True)  # shadow runs up the wall
for u, depth in [(0.72, 0.65), (0.82, 0.55), (0.77, 0.3)]:
    ball(cv, sh.place(u, depth), 0.06, base="#f07a18", material="citrus", seed=int(u * 100), cast=True)
```

## Recipe: toys on a checkered floor

```python
from pixelcanvas import Canvas
from pixelcanvas.form import Ground, set_light, ball, cuboid, disc

cv = Canvas(1920, 1080)
cv.gradient([(0, "#cfe0ee"), (0.45, "#eef2f4")])
set_light(direction=(-0.6, -0.7), elevation=40)
g = Ground(cv, horizon=0.4, camera_height=1.0)                         # sizes at its Spots are metres
g.floor("checker", size=0.4, colors=("#ece6da", "#3a3a44"))
cuboid(cv, g.place(0.9, 4.0), (0.4, 0.4, 0.4), yaw=25, base="#3fae4a", material="plastic", round=0.1, cast=True)
duck = g.place(0.0, 2.6)
ball(cv, duck, (0.22, 0.16), base="#f5c518", material="rubber", cast=True)
ball(cv, duck.move(-0.1).up(0.24), 0.12, base="#f5c518", material="rubber", shadow=False)
car, yaw = g.place(-0.75, 2.0), 25
wheels = [(-0.2, -0.14), (0.2, -0.14), (-0.2, 0.14), (0.2, 0.14)]
for dx, dz in wheels[:2]:                                              # far wheels first
    disc(cv, car.move(dx, dz, yaw), 0.07, 0.05, yaw=yaw)
cuboid(cv, car.up(0.05), (0.6, 0.3, 0.14), yaw=yaw, base="#d42a2a", material="glossy", round=0.3,
       shadow=False, cast=dict(ground_y=car.v))
cuboid(cv, car.move(-0.03, 0, yaw).up(0.19), (0.32, 0.26, 0.12), yaw=yaw, base="#d42a2a", material="glossy",
       round=0.3, shadow=False)
for dx, dz in wheels[2:]:                                              # near wheels last
    disc(cv, car.move(dx, dz, yaw), 0.07, 0.05, yaw=yaw)
```

## Recipe: any shape as a volume, placed with a frame

```python
from pixelcanvas import Canvas, sdf
from pixelcanvas.form import Frame, set_light, cast_shadow

cv = Canvas(1920, 1080, "#2b3440")
set_light(direction=(-0.7, -0.6))
for i, (u, rot, color) in enumerate([(0.25, -20, "#e05a3a"), (0.5, 0, "#3a8fe0"), (0.75, 25, "#f0c040")]):
    fr = Frame(cv, (u, 0.5), 0.12, rotate=rot)
    d = sdf.smooth_union(sdf.sd_circle(fr.X, fr.Y, 0, -0.5, 0.6),
                         sdf.sd_box(fr.X, fr.Y, 0, 0.6, 0.35, 0.9, 0.2), 0.3)   # written once, in local units
    cast_shadow(cv, fr.fill(d), ground_y=0.5 + 0.12 * 1.5)
    fr.form(d, color, material="plastic")
```

`fr.child(at=(0, 1.2), size=0.6, rotate=30)` hangs a part on a parent (an arm on a body, a petal on a stem);
the child moves, turns and mirrors with its parent. The left/right twin of a part is
`fr.child(at=(-x, y), rotate=same, flip=True)`: a mirror image about the parent's own vertical axis.

## Recipe: a room with furniture at three depths

```python
from pixelcanvas import Canvas
from pixelcanvas.form import Room, set_light, ball, lathe

cv = Canvas(1920, 1080)
set_light(direction=(-0.9, -0.4), elevation=30, front=-0.2)            # sunlight from the left window
room = Room(cv, horizon=0.44, vanishing=0.46, back=0.5, width=5.2, eye=1.35)
room.draw(floor="planks", window="left", sun_dir=(1.0, -0.7, -0.4))
z = room.zb                                                            # depth of the back wall
room.rug(-1.2, 1.9, z - 2.6, z - 0.9, color="#b8553d", pattern="diamonds")
room.picture("back", center=(0.2, 1.8), size=(1.2, 0.8))
board = room.panel("right", center=(z - 1.2, 1.6), size=(0.9, 0.6), color="#26302b")  # 4 corners for text
room.box(0.2, z - 0.2, (2.3, 0.26, 0.86), base="#4f7f86", material="fabric", round=0.35)  # sofa back
room.box(0.2, z - 0.5, (2.3, 0.92, 0.42), base="#4f7f86", material="fabric", round=0.35, cast=True)  # seat
lathe(cv, room.place(1.8, z - 0.6), [(0.2, 0), (0.035, 0.05), (0.035, 0.52), (0.3, 0.54), (0.3, 0.58)],
      base="#8b5a36", material="wood", smooth=False, cast=True)        # side table, sizes in metres
ball(cv, room.place(1.2, z - 2.6), 0.16, base="#2f6fb8", material="plastic", cast=True)
```

Draw furniture back to front; each `room.box` / solid is one `log("step")` layer if you want it editable.
Stack with `lift=` (cushions on a sofa base) or `spot.up(h)` (a vase on a table).

## Recipe: a perspective floor and a row of products

```python
from pixelcanvas import Canvas
from pixelcanvas.form import Ground, set_light, row, lathe, reflection

cv = Canvas(1920, 1080)
cv.gradient([(0, "#1d2330"), (0.45, "#3b4458")])
set_light(direction=(0.6, -0.7), elevation=40)
g = Ground(cv, horizon=0.45, camera_height=1.2)
g.floor("checker", size=0.5, colors=("#d8d2c6", "#2c2a2a"))           # any texture: g.uv() gives world x, depth
pts, order = row(6, (0.2, 0.78), (0.8, 0.78))
for i in order:
    bottle = lathe(cv, g.spot(*pts[i]), [(0.09, 0), (0.1, 0.05), (0.1, 0.3), (0.04, 0.42), (0.035, 0.55)],
                   base=["#c0392b", "#2e86c1", "#27ae60", "#f39c12", "#8e44ad", "#16a085"][i],
                   material="glossy", cast=True)
    reflection(cv, bottle, 0.3)                                        # polished floor
```

## Tips

- One light per picture: call `set_light` once at the top; window light from the left = `direction=(-0.8, -0.5)`.
  `front` > 0 sends ground shadows backwards (up the screen), < 0 towards the viewer (back-lit scenes).
- `tilt` is how much we look down on an object (0 = side view, 0.3 = table view, 0.6 = top-down). A Spot sets it
  for you; pass `tilt=` for a free-standing product shot.
- Textures: pass any `(H, W, 3)` map as `base` (wood grain, gingham, a printed label) — the volume shading is
  applied on top. Labels: `bottle.band(0.2, 0.45, "#efe4cc", material="matte")`.
- Text on a wall board: `tl, tr, br, bl = room.panel(...)`, then `cv.text` inside that box.
- Speed: frames and solids evaluate only their own bounding box, so hundreds of objects on a 1080p canvas stay fast.
