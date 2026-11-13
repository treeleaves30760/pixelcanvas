# Engineering drawings (`pixelcanvas.drafting`)

A `Sheet` is a complete technical drawing: ISO border with zone marks, a real title block, views in model
millimetres, ISO 128 line types, ISO 129 dimensions that stack themselves, sections with material hatching,
GD&T, architectural plan symbols and isometric pictorials with hidden lines removed. One description gives a
crisp raster (150 dpi A3 = 2480 px wide), an SVG in millimetres and an ASCII DXF R12 that opens in any CAD program.

Options are checked: an unknown option, furniture, material or symbol name raises an error naming the closest valid one, and both spellings are accepted (`width` / `w`, `centre` / `center`).

```python
from pixelcanvas.drafting import Sheet, Profile
sh = Sheet("A4", title="SPACER PLATE", drawing_no="PC-0001", material="S235JR", author="me", date="2026-11-12")
v = sh.view("FRONT", origin=(80, 110))                    # model (0, 0) lands at sheet (80, 110) mm
v.profile(Profile(0, 0).to(100, 0, fillet=8).to(100, 60, fillet=8).to(0, 60).close())
v.hole(50, 30, 20)                                         # circle + centre lines
v.dim_h((0, 0), (100, 0)); v.dim_v((100, 0), (100, 60), side="right"); v.dim_diameter(50, 30, 20)
cv = sh.draw()                                             # Canvas; sh.save_svg(p) / sh.save_dxf(p)
```

**Coordinates**: millimetres, **y up** (like CAD). Sheet positions are measured from the paper's bottom-left
corner; view geometry is model millimetres, mapped by the view's `origin` and `scale`. Text heights, arrow sizes,
offsets and line widths are paper millimetres. Angles: degrees counter-clockwise from +x.
After `sh.draw(cv)`, `v.ox, v.oy` hold where model (0, 0) landed (auto-placed views too), `v.to_sheet(x, y)` gives
sheet mm and `v.to_px(x, y, cv)` the canvas pixel of a model point, ready for your own marks on the drawing.

## Calls

| Call | What it does |
|---|---|
| `Sheet(size="A3", orientation, scale=None, projection="first"/"third", title, drawing_no, author, date, material, units, revision, checked, approved, tolerances, finish, mass, fields=[(label, value)], dpi=150, line_width=0.5, style="white"/"paper"/"blueprint", arrows="auto"/"arrow"/"tick", fit="shrink"/"error")` | sheet + title block; `scale=None` gives a plan the largest standard scale that fits and other drawings 1:1, `"auto"` the largest that fits for any drawing; `arrows="auto"` gives oblique ticks at 1:20 and smaller |
| `sh.view(name, origin=(x, y), scale=None, label=None)` | a View; leave `origin` out and the view (with its dimensions and projected views) is centred in the free drawing area and always fits: when the sheet scale is too large the sheet uses the largest standard scale that fits and says so in `sh.fit_report` (`fit="error"` raises an error that names that scale) |
| `sh.projected(front, "top"/"right"/"left"/"bottom", gap=12)` | view aligned for the sheet's projection method, `gap` paper mm clear of `front` (or `distance=` between origins) |
| `sh.detail(view, cx, cy, r, "B", origin, scale="2:1")` | circle on `view` + enlarged clipped copy labelled "DETAIL B (2:1)" |
| `v.line / polyline / polygon / rect(x, y, w, h, r=, chamfer=) / circle / arc(cx, cy, r, a0, a1)` | geometry; `kind=` line type |
| `v.profile(Profile(x, y).to(x, y, fillet=, chamfer=).by(dx, dy).arc_to(x, y, r, cw=).close(fillet=))` | outlines with rounded / cut corners |
| `v.slot(x1, y1, x2, y2, w)` · `v.hole(cx, cy, d, thread="M8")` · `v.bolt_circle(cx, cy, pcd, n, d, start=90)` | features with centre lines |
| `v.centerline(...)` · `v.centre_mark(cx, cy, r)` · `v.thread(x1, y1, x2, y2, d, internal=False)` · `v.break_line(...)` | conventions |
| `v.dim_h(p1, p2, side="bottom"/"top")` · `v.dim_v(p1, p2, side="left"/"right")` · `v.dim(p1, p2)` · `v.dim_aligned(p1, p2, offset)` | linear dims, auto-stacked outwards |
| `v.dim_chain(points, axis="h")` · `v.dim_baseline(points, axis="v")` | chain / baseline dimensions |
| `v.dim_radius(cx, cy, r, angle)` · `v.dim_diameter(cx, cy, d, angle)` · `v.dim_angle(cx, cy, a0, a1, r=, ext_from=)` | radial / angular |
| `v.dim_chamfer(x, y, 2)` · `v.hole_callout(cx, cy, d, n=6, thru=True, thread=, depth=)` · `v.callout(x, y, text, angle, r=)` | notes on leaders |
| dimension options: `text="{} TYP"`, `prefix="Ø"`, `suffix=" H7"`, `tol=0.1` / `tol=(0.021, 0)`, `limits=True`, `offset=`, `decimals=` | tolerances and fits |
| `v.section(outline, gaps=(), material="steel", beyond=())` | the whole cut body: outline, hatching on the cut faces, bores and holes as open gaps with their edges and centre line, visible edges `beyond` the plane |
| `v.hatch(region, holes=(), material="general", outline=True, mirror="x")` | `general steel aluminium plastic concrete insulation earth`; `mirror="x"` adds the half below the axis (turned parts) |
| `v.cutting_plane([(x, y), ...], "A", direction="right")` · `v.section_label("A")` · `v.label(text)` | sections; arrows point the way you look: a section shown as the `"left"` view takes `direction="right"` |
| `v.leader(x, y, text, dx, dy, arrow="arrow"/"dot")` · `v.balloon(x, y, n)` · `v.text(x, y, s, h, anchor)` | annotations |
| `v.surface_finish(x, y, "Ra 1.6", angle=90)` · `v.datum(x, y, "A", angle=-90)` · `v.fcf(x, y, "position", "0.1", ["A", "B"], diameter=True, leader=(x, y))` | ISO 1302 / GD&T (`GDT_SYMBOLS`) |
| `sh.notes(lines, x, y, width=)` · `sh.parts_list(rows)` · `sh.table(rows, columns, x, y, widths, title)` · `sh.text(x, y, s)` | sheet text blocks; notes wrap inside the frame |
| `v.wall(points, thickness, closed)` · `v.door(x, y, width, toward=(x, y), hinge="left", style="swing"/"double"/"sliding")` · `v.sliding_door(x, y, width)` · `v.window(x, y, width)` · `v.opening(...)` | plans: clean joins, poche, swings; **balcony doors and any opening wider than 1.2 m: `sliding_door`**; a pair of 800 mm leaves: `style="double"`; wardrobes: `furniture("wardrobe")` already draws its doors, a plain 1.0 m door is `swing` |
| `v.room(name, x, y, fill=None)` | name + floor area measured from the walls around (x, y); the label moves to free floor, off furniture and door swings (`area=` m², `(w, d)` or points to override); `fill=` (or `tint=`) tints the floor: a colour, a palette name, `True` (palette by room type) or `False` |
| `v.furniture(kind, x, y, w, d, angle, seats=, sink=, hob=, color=, fill=)` · `v.stairs(x, y, w, l)` · `v.gridlines(xs, ys)` | `bed single_bed sofa armchair table chair toilet basin shower bathtub counter sink hob fridge wardrobe desk`; walls cover furniture that touches them; `color` = lines, `fill` = inside |
| `v.colors(floors="pastel", furniture=None, furniture_fill=None)` | colour the whole plan: every room floor gets the palette colour of its type (`FLOOR_PALETTES`: `pastel warm cool grey`, or a dict `{"bedroom": "#dbe6f4", "BATH": ...}`; types living kitchen dining bedroom bath hall balcony office storage other, read from the room name); furniture lines in `furniture`, filled with a pale mix of it. Plans stay black on white without it |
| `sh.north_arrow(x, y)` · `sh.scale_bar(x, y, length=5)` · `sh.room_schedule(view, x, y)` | plan sheet items; the schedule lists every `room()` with a net total |
| `v.iso_box / iso_extrude(profile, depth, plane="xy"/"xz"/"yz", at) / iso_cylinder / iso_hole / iso_slot` · `iso(x, y, z)` | isometric with hidden-line removal |
| `sh.draw(cv=None, log=None, style=None)` · `sh.save_svg(path)` · `sh.save_dxf(path)` · `sh.to_drawing()` · `parse_dxf(path)` | output |

Labels keep clear of each other: cutting-plane letters, room names and view titles move off dimension values, other
labels, lines and furniture, and a room name shrinks to fit a small room. Each sheet records what it drew (walls,
doors, windows, furniture, rooms, north arrow, scale bar, holes, bores, chamfers, dimensions, views) for
`render.py --check`.

**Line types** (`kind=`): `visible` (thick), `hidden`, `center`, `phantom`, `dim`, `thin`, `hatch`, `cutting`,
`wall`, `furniture`. Thick:thin = 2:1 (0.5 / 0.25 mm); dashes are fitted so lines start and end on a dash and
centre lines cross on long dashes. DXF layers are named after the line types (VISIBLE, HIDDEN, CENTER, DIM, ...).

**Projection**: model axes are front (X, Z), top (X, Y), right (Y, Z), left (-Y, Z), with Y pointing away from
the viewer of the front view. `projection="first"` (ISO) puts the top view below the front view and the right
view on its left; `"third"` (ASME) the opposite. The title block shows the matching symbol.

## Recipes

**Part with three views and a section** (first angle):

```python
from pixelcanvas.drafting import Sheet, Profile
sh = Sheet("A3", scale="1:1", title="CLAMP BLOCK", drawing_no="PC-2001", material="S355")
fr = sh.view("FRONT", origin=(150, 170))
fr.rect(0, 0, 80, 50, chamfer=3)
fr.hole(40, 25, 16)
fr.cutting_plane([(40, 0), (40, 50)], "A", direction="right")
fr.dim_h((0, 50), (80, 50), side="top"); fr.dim_v((0, 0), (0, 50)); fr.dim_diameter(40, 25, 16, tol=(0.018, 0))
tp = sh.projected(fr, "top", 90)                          # first angle: below the front view
tp.rect(0, 0, 80, 30)
for x in (32, 48):
    tp.line(x, 0, x, 30, "hidden")
tp.centerline(40, 0, 40, 30)
tp.dim_v((80, 0), (80, 30), side="right")
sc = sh.projected(fr, "left", 120).section_label("A")     # seen from the left -> placed on the right
sc.section((0, 0, 30, 50), gaps=[(0, 17, 30, 33)])          # the whole cut body, the Ø16 hole as a gap
sc.dim_v((30, 17), (30, 33), side="right", prefix="Ø", suffix=" H7")
cv = sh.draw()
```

**Block with a through bore and four mounting holes** (front, top and section A-A, fully dimensioned):

```python
from pixelcanvas.drafting import Sheet
L, H, D, BORE, HOLE = 120, 80, 40, 52, 9                   # block, bore diameter, mounting hole diameter
sh = Sheet("A3", title="BEARING HOUSING", drawing_no="PC-3100", material="EN-GJL-250")
fr = sh.view("FRONT", label="FRONT VIEW")                  # no origin: centred, and the sheet scale fits it
fr.rect(0, 0, L, H, chamfer=5)
fr.hole(L / 2, H / 2, BORE)                                # the bore
for x in (10, L - 10):
    for y in (10, H - 10):
        fr.hole(x, y, HOLE)                                # four mounting holes on 100 x 60 centres
fr.cutting_plane([(L / 2, -6), (L / 2, H + 6)], "A", direction="right")
fr.dim_h((0, H), (L, H), side="top"); fr.dim_v((0, 0), (0, H))
fr.dim_chain([(0, 0), (10, 0), (L - 10, 0), (L, 0)], side="bottom", offset=18)
fr.dim_chain([(L, 0), (L, 10), (L, H - 10), (L, H)], axis="v", side="right")
fr.dim_diameter(L / 2, H / 2, BORE, angle=-120); fr.hole_callout(L - 10, H - 10, HOLE, n=4, angle=45)
fr.dim_chamfer(0, H - 5, 5)
top = sh.projected(fr, "top", gap=30).label("TOP VIEW")    # first angle: below the front view
top.rect(0, 0, L, D)
for x in (5, L - 5):
    top.line(x, 0, x, D)                                   # chamfer edges
for x in (L / 2 - BORE / 2, L / 2 + BORE / 2, 5.5, 14.5, L - 14.5, L - 5.5):
    top.line(x, 0, x, D, "hidden")                         # bore and holes
top.centerline(L / 2, -4, L / 2, D + 4)
top.dim_v((0, 0), (0, D)); top.dim_h((0, 0), (L, 0))
sec = sh.projected(fr, "left", gap=60).section_label("A")  # seen from the left, placed on the right
sec.section((0, 0, D, H), gaps=[(0, H / 2 - BORE / 2, D, H / 2 + BORE / 2)])   # body, hatching, the bore as a gap
sec.dim_h((0, H), (D, H), side="top"); sec.dim_v((0, 0), (0, H))
sec.dim_v((D, H / 2 - BORE / 2), (D, H / 2 + BORE / 2), side="right", prefix="Ø")
cv = sh.draw()
```

`section` takes the outline of the cut body and the bores the plane cuts through as `gaps`; a plane through the
mounting holes lists them as more gaps, and `beyond=[(x1, y1, x2, y2)]` adds edges seen past the plane.

**Holes, patterns and GD&T**:

```python
from pixelcanvas.drafting import Sheet
sh = Sheet("A4", title="FLANGE", drawing_no="PC-3001")
v = sh.view("FRONT", origin=(110, 120))
v.circle(0, 0, 50); v.hole(0, 0, 30)
v.bolt_circle(0, 0, 76, 6, 9, start=90)                   # 6 holes, pitch circle, radial centre lines
v.hole_callout(0, 38, 9, n=6, angle=60)                   # "6x Ø9 THRU"
v.dim_diameter(0, 0, 100, angle=-35); v.dim_diameter(0, 0, 76, angle=200, leader=14)
v.dim_angle(0, 0, 90, 150, r=58, ext_from=42)
v.datum(0, -50, "A"); v.fcf(40, -52, "position", "0.2", ["A"], diameter=True, leader=(32.9, -23.5))
v.surface_finish(50, 0, "Ra 3.2", angle=0)                # on the rim, value readable from the right
sh.draw(style="blueprint")                                # white on blue
```

**Floor plan at 1:50** (model mm): wall centre lines, doors and windows find their wall, rooms measure
themselves, and short chain segments move their values aside so nothing collides.

```python
from pixelcanvas.drafting import Sheet
sh = Sheet("A3", scale="1:50", title="STUDIO", drawing_no="A-100")
p = sh.view("PLAN", label="FLOOR PLAN")                   # auto-placed in the free area
p.wall([(125, 125), (6875, 125), (6875, 4875), (125, 4875)], thickness=250, closed=True)
p.wall([(4500, 125), (4500, 4875)], thickness=100); p.wall([(4500, 2400), (6875, 2400)], thickness=100)
p.door(3000, 125, 1000, toward=(3000, 1500), hinge="left")
p.door(4500, 1800, 800, toward=(5600, 1800)); p.door(4500, 3700, 800, toward=(5600, 3700), hinge="right")
p.window(125, 1600, 1400); p.window(2600, 4875, 1800); p.window(5700, 4875, 900)
p.furniture("bed", 1300, 3700, angle=90); p.furniture("sofa", 2000, 1900)
p.furniture("shower", 6350, 1850); p.furniture("toilet", 5600, 470, angle=180)
p.room("STUDIO", 3200, 3300); p.room("BATH", 5700, 1300); p.room("STORE", 5700, 3600)
p.dim_chain([(0, 5000), (1700, 5000), (3500, 5000), (5250, 5000), (6150, 5000), (7000, 5000)], side="top")
p.dim_chain([(0, 0), (4500, 0), (7000, 0)]); p.dim_h((0, 0), (7000, 0))
p.dim_v((0, 0), (0, 5000)); p.gridlines(xs=[125, 4500, 6875], ys=[125, 4875])
sh.north_arrow(380, 250); sh.scale_bar(300, 230, length=5); sh.room_schedule(p, x=296, y=200)
sh.save_svg("outputs/studio.svg"); sh.save_dxf("outputs/studio.dxf")
```

**Coloured plan** (presentation sheet): one call tints every room by type and colours the furniture; walls,
labels and dimensions stay on top.

```python
from pixelcanvas.drafting import Sheet
sh = Sheet("A4", scale="1:50", title="FLAT", drawing_no="A-101")
p = sh.view("PLAN", label="FLOOR PLAN")
p.wall([(125, 125), (6875, 125), (6875, 4875), (125, 4875)], thickness=250, closed=True)
p.wall([(4200, 125), (4200, 4875)], thickness=100)
p.door(4200, 1200, 800, toward=(5200, 1200))
p.colors(floors="pastel", furniture="#7a5a40")            # or per room: p.room(..., fill="#dbe6f4")
p.furniture("bed", 2100, 3700, 1600, 2000); p.furniture("bathtub", 5900, 3900, 1600, 700)
p.room("BEDROOM", 2100, 1600); p.room("BATH", 5500, 2400)
cv = sh.draw()
x, y = p.to_px(2100, 3700, cv)                            # canvas pixel of the bed centre
```

**Section of a turned part** (flange): draw the upper half once, `mirror="x"` completes it with the same hatch.

```python
from pixelcanvas.drafting import Sheet
sh = Sheet("A3", title="PIPE FLANGE", drawing_no="PC-5001", material="S235JR")
fr = sh.view("FRONT", label="FRONT VIEW")
fr.circle(0, 0, 80); fr.circle(0, 0, 45); fr.hole(0, 0, 60)
fr.bolt_circle(0, 0, 125, 6, 14, start=90)
fr.cutting_plane([(0, -80), (0, 80)], "A", direction="right")
fr.dim_diameter(0, 0, 160, angle=-35); fr.dim_diameter(0, 0, 125, angle=200, leader=14)
fr.hole_callout(54.1, 31.25, 14, n=6, angle=40); fr.dim_angle(0, 0, 30, 90, r=100, ext_from=70)
sc = sh.projected(fr, "left", gap=25).section_label("A")
sc.hatch([[(0, 30), (0, 55.5), (18, 55.5), (18, 45), (48, 45), (48, 30)],      # disc + hub
          [(0, 69.5), (0, 80), (18, 80), (18, 69.5)]],                           # rim beyond the bolt hole
         material="steel", outline=True, mirror="x")
sc.centerline(0, 0, 48, 0); sc.centerline(0, 62.5, 18, 62.5); sc.centerline(0, -62.5, 18, -62.5)
sc.dim_h((0, 80), (18, 80), side="top"); sc.dim_h((0, 80), (48, 45), side="top")
sc.dim_v((0, -80), (0, 80), prefix="Ø")
for r in (30, 45):
    sc.dim_v((48, -r), (48, r), side="right", prefix="Ø")
sc.dim_v((18, -62.5), (18, 62.5), side="right", prefix="Ø")
sh.notes(["Untoleranced dimensions ISO 2768-m.", "Break sharp edges 0.5 x 45°."])
cv = sh.draw()
```

**Isometric pictorial** (true-length isometric, hidden lines removed):

```python
from pixelcanvas.drafting import Sheet, Profile
sh = Sheet("A4", title="BRACKET", drawing_no="PC-4001")
iv = sh.view("ISO", origin=(70, 110), scale="1:2", label="ISOMETRIC VIEW")
iv.iso_extrude(Profile(0, 0).to(120, 0, fillet=10).to(120, 70).to(0, 70).close(fillet=10), 10)      # base
iv.iso_extrude(Profile(0, 10).to(120, 10).to(120, 85, fillet=15).to(0, 85, fillet=15).close(), 10,
               plane="xz", at=60)                                                                   # upright
iv.iso_hole(60, 0, 50, 28, axis="y"); iv.iso_slot(20, 18, 0, 20, 34, 0, 11, axis="z")
iv.iso_cylinder(100, 30, 10, 8, 20)                                                                 # a boss
cv = sh.draw()
```

In a scene, `sh.draw(cv, log=log)` fits the sheet into the canvas; every view, the border, the title block and
each sheet block is its own layer. Full examples: `scenes/drafting_bracket.py` (A3 part drawing: three views,
section, isometric, GD&T) and `scenes/drafting_floorplan.py` (1:50 apartment plan).

Tips: dimensions on one side of a view stack outwards in order of length, so call them in any order. Arrows move
outside and the value moves past the extension line when the space is short; in a chain the value goes to
whichever side is free, or to a second row. Radius and diameter values slide along their leader until they are
clear of the drawing. Put `side=` on the side with room, `offset=` (paper mm from the part) pins a dimension
line, and `text="{} TYP"` keeps the measured value.
