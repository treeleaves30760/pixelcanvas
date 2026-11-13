# pixelcanvas API: every module, then the low-level layer

Every module of the package, then the low-level calls: the canvas, freehand drawing, masks, noise, SDFs, filters,
layers and the manifest behind `--check`. `python tools/kit.py find "..."` names the kit call for any subject.

| Module | Draws | Page |
|---|---|---|
| `canvas`, `color`, `noise`, `masks`, `brush`, `sdf`, `filters` | the low-level layer: `Canvas`, colours, noise, masks, strokes, signed distances, filters | [below](#canvas) |
| `illustrate` | freehand drawing: `Sketch` shapes in fractions, one-call shading / outline / shadow / glow, `pattern`, `sticker`, `wash`, `paper`, `grade`, `limit_palette`; `cv.draw()` / `cv.cairo()` layers | [illustration.md](illustration.md) |
| `figure` | posable, dressed people with hands, faces and hair, in flat 2D styles | [figures.md](figures.md) |
| `creature` | 2D animals: mammals, birds, sea life, insects, with poses and coats | [creatures.md](creatures.md) |
| `clay` | 3D clay / toy people (`cast`, `Character`, 31 poses), 64 animal species, 33 props, decals on `solid` | [clay.md](clay.md) |
| `pictograms` | 633 named pictograms of objects, signs and symbols in six styles; each one is also a mask | [pictograms.md](pictograms.md) |
| `flora` | trees, shrubs, flowers, potted plants, grass, rocks | [flora.md](flora.md) |
| `elements` | fire, smoke, clouds, water, weather, skies, light | [elements.md](elements.md) |
| `texture` | materials: wood, stone, brick, fabric, metal, fur, lava | [textures.md](textures.md) |
| `form` | 2.5D objects with volume: still lifes, rooms, one light | [forms.md](forms.md) |
| `solid` | ray-marched 3D: products, rooms, vehicles, isometric dioramas | [solid.md](solid.md) |
| `scenery` | landscape recipes: sky, mountains, rivers, pines, god rays | [below](#landscapes-pixelcanvasscenery) |
| `text` | exact text in any language, auto-fit, wrapping, effects | [typography.md](typography.md) |
| `layout` | boxes, grids, stacks, poster and card templates, themes; `THEME_KEYS`, `role_color` | [typography.md](typography.md) |
| `palette` | named palettes, palettes from a mood sentence, harmonies, colour-blind-safe categories | [backdrops.md](backdrops.md) |
| `backdrop` | one-call backgrounds and wallpapers with a calm area for text (`bd.calm_px`, `bd.ink`) | [backdrops.md](backdrops.md) |
| `stylize` | 43 art styles, on a mask or the whole picture; `pick(request)`, `options(name)` | [styles.md](styles.md) |
| `vector` | shapes, paths, icons and groups, raster + SVG; `app_icon`, `app_icon_sheet`, `APP_GLYPHS` | [vector.md](vector.md) |
| `diagram` | flowcharts, architecture diagrams (auto layout, tiers, buses), sequences, timelines, callouts | [diagrams.md](diagrams.md) |
| `chart` | line, bar, scatter, pie, heatmap, KPI tiles, dashboards | [charts.md](charts.md) |
| `ui` | app and web screens in device frames, declarative auto layout, templates, SVG; `window(cv, kind, content, box=)` (`WINDOW_KINDS`), `COMPONENT_NAMES` | [ui.md](ui.md) |
| `drafting` | engineering sheets, dimensions, sections, floor plans, DXF | [drafting.md](drafting.md) |
| `catalog` | one search over every kit: `find(request)`, `show`, `kit_kinds()`, `recipe`, `extract_content` | [catalog.md](catalog.md) |
| `manifest` | what was drawn where: `note`, `summary`, `relations`, `check(prompt=)` behind `--check` | [below](#manifest-what-the-report-knows-pixelcanvasmanifest) |
| `spec` | a picture from a small JSON spec: `loads`, `validate`, `render`, `example`; `tools/compose.py` | [spec.md](spec.md) |
| `vision`, `layers` | inspection views behind `look.py`; one layer per `log()` step → PSD / ORA / PNG | [below](#vision-helpers-pixelcanvasvision) |
| `cli`, `mcp_server` | the `pixelcanvas` command (`doctor`, `install-skill`, `render` ...); the MCP server | `pixelcanvas --help` |

```python
import numpy as np
from pixelcanvas import *            # Canvas, colour, noise, masks, brushes, filters, text
from pixelcanvas import sdf, manifest, layers
```

Conventions: x right, y down, arrays `arr[y, x]`; masks float32 (H, W) in 0–1; colours `'#rrggbb'`, `'#rgb'`, a CSS name,
a tuple or a full (H, W, 3) map. **Tuple rule: floats are 0–1 and may exceed 1 (HDR), integers are 0–255**:
`(1.6, 0.62, 0.16)` is a bright orange glow for `mode="add"`, `(255, 128, 0)` is orange, `(1, 1, 1)` is white, and
a single float anywhere in the tuple makes all three 0–1. Canvas methods return `self`; `cv.buf` is the float32
(H, W, 3) array.

## Canvas

| Method | Does |
|---|---|
| `Canvas(w, h, color=(0,0,0))`, `Canvas.load(path)`, `Canvas.from_array(arr)` | create, load, wrap an array |
| `cv.w`, `cv.h`, `cv.shape`, `cv.copy()`, `cv.to_pil()` | size, copy, PIL image |
| `cv.xy()` → `(X, Y)` pixels; `cv.uv()` → `(U, V)` 0–1 | coordinate grids (float32 (H, W)) |
| `cv.paint(color, mask=None, mode="normal", opacity=1)` | **the core call**: blend by mask; modes normal / add / multiply / screen / overlay / softlight / lighten / darken. A number in the mode position is the opacity: `cv.paint(c, mask, 0.9)` equals `opacity=0.9` (same for `fill`, `gradient`, `linear`, `radial`, `paste`) |
| `cv.fill(color)`, `cv.gradient(stops, axis="y")`, `cv.linear(p0, p1, stops)`, `cv.radial(cx, cy, r, stops)` | flat fill and gradients, `stops = [(t, colour), ...]` |
| `cv.glow(cx, cy, sigma, color, strength=1, mode="add")` | Gaussian glow (may go HDR) |
| `with cv.draw(ss=2, name=None, **finish) as d:` | supersampled Pillow ImageDraw layer: `d.p(u, v)`, `d.k(h)`, `d.font(size)`; finish keywords `shade`, `highlight`, `line`, `shadow`, `glow`; `cv.cairo()` the same for pycairo |
| `cv.text(text, xy, size, color, **style)` | exact text, see [typography.md](typography.md); bbox in `cv.last_bbox` |
| `cv.paste(other, x, y, mask=None, mode=..., opacity=...)` | paste a Canvas or array (clipped) |
| `cv.get(x, y)`, `cv.put(x, y, color)`, `cv.region(x0, y0, x1, y1)`, `cv.sample(X, Y)` | pixel access, sub-view, sampling at float coordinates |
| `cv.displace(dx, dy)`, `cv.resize(w, h)`, `cv.crop(...)`, `cv.flip("x" or "y")` | geometry |
| `cv.blur(s)`, `bloom(th, sigma, k)`, `vignette(k, r, soft)`, `grain(amt, seed)`, `levels(b, w, gamma)`, `contrast(k)`, `saturation(s)`, `soft_clip(knee)`, `filmic()`, `split_tone(sh, hi, k)`, `clip()` | post-processing |
| `cv.apply(fn, *args, **kw)` | `cv.buf = fn(cv.buf, ...)` |
| `cv.save(path, preview=None)`, `cv.stats()` | save (+ preview JPG), luminance statistics |

## Colour and maths (`pixelcanvas.color`)

`rgb(c)` / `as_color(c)` → float32 array · `lerp(a, b, t)` (t may be an (H, W) mask → (H, W, 3)) ·
`mix([(0, '#000'), (1, '#fff')], V)` multi-stop lookup · `hsv_to_rgb(h, s, v)` · `luma(img)` · `clamp01(a)` ·
`smoothstep(e0, e1, x)` (array edges; e0 > e1 reverses) · `normalize(a, lo, hi)` · `grid(h, w)` · `F32`.

## Noise (`pixelcanvas.noise`)

`scale` = pixel width of the lowest octave's cell (bigger = smoother); `aspect` > 1 stretches vertically;
`mode`: `"plain"` (≈ -1–1, terrain, clouds), `"ridged"` (0–1, sharp ridges), `"billow"` (0–1, puffy).

| Function | Does |
|---|---|
| `fbm2d(h, w, scale, octaves=6, lacunarity=2, gain=0.5, seed=0, offset=(0,0), aspect=1, mode="plain")` | full-frame fBm |
| `fbm2d_at(X, Y, octaves, ..., seed, mode)` | fBm at any coordinates (in cells): perspective, warps, polar |
| `perlin2d(h, w, scale, seed)`, `perlin2d_at(X, Y, seed)` | single octave |
| `warp2d(h, w, scale, strength=1, octaves=5, seed, aspect, mode)` | domain warping: smoke, marble, flow |
| `fbm1d(n, scale, octaves=5, ..., seed, mode)`, `perlin1d_at(X, seed)`, `smooth1d(a, sigma)` | 1D: ridgelines, shores, rooftops |

## Freehand (`pixelcanvas.illustrate`)

`Sketch(cv, light=(dx, dy))`: positions in fractions, every shape a float mask. Page: [illustration.md](illustration.md).

```python
from pixelcanvas import Canvas, Sketch, paper, grade, limit_palette, grow, edge, signed_distance, MOODS
cv = Canvas(480, 270, "#f3e3c3")
sk = Sketch(cv, light=(-0.6, -0.8))
head = sk.ellipse(0.5, 0.45, 0.16, 0.15)
sk.paint(head, "#f6c39a", shade=0.4, highlight=0.3, line=True, shadow=True, name="head")
sk.pattern(sk.box(0.05, 0.75, 0.95, 0.95), "seigaiha", "#2f5d8a", size=0.06)        # also stripes, dots, speckle, checker, tiles
sk.arrow((0.1, 0.6), (0.3, 0.5), via=(0.2, 0.45))
sk.label("warm", (0.25, 0.2), to=(0.42, 0.4))                  # a caption pill with a leader line, recorded as text
cv.text("Title", box=(0.3, 0.02, 0.7, 0.12), color="#3d3350")  # fractions when every value is within 0-1
with cv.draw(ss=2, name="sun", glow="#ffcf70") as d:
    d.ellipse([*d.p(0.8, 0.1), *d.p(0.9, 0.28)], fill="#ffd36e")
ring = edge(head, 3)                                  # a 3 px band on the edge; grow(mask, px) widens a mask
d_px = signed_distance(head)                          # px, negative inside
limit_palette(cv, ["#f3e3c3", "#f6c39a", "#2f5d8a", "#ffd36e", "#3d3350"], 0.5)
paper(cv); grade(cv, "golden", 0.6)                   # MOODS: warm night, golden, cosy, cool day, moody, print
```

## Masks, curves and brushes (`pixelcanvas.masks`, `pixelcanvas.brush`)

Every mask function takes **`h, w` first** (rows, then columns), then the shape. Results are float32 (H, W) in 0–1
with anti-aliased edges.

| Function (argument order) | Does |
|---|---|
| `shape_mask(h, w, draw_fn, aa=3)` | draw with Pillow ImageDraw; in `draw_fn(d, s)` multiply every coordinate by `s` |
| `polygon_mask(h, w, points, aa=3)` | polygon from `[(x, y), ...]` |
| `circle_mask(h, w, cx, cy, r, feather=1)` | disc |
| `ellipse_mask(h, w, cx, cy, rx, ry, angle=0)` | ellipse, `angle` in degrees (clockwise on screen) |
| `rect_mask(h, w, x0, y0, x1, y1, feather=0)` | rectangle by two corners |
| `band_mask(h, w, y0, y1, soft=0)` | horizontal band between two rows |
| `line_mask(h, w, points, width=2, aa=3, closed=False)` | anti-aliased polyline, round joins and caps |
| `bezier_points(p0, p1, p2[, p3 ...], n=64)` | sample a Bézier → list of (x, y) for `line_mask` / `polygon_mask` |
| `feather(mask, sigma)` | soften edges |
| `shift(mask, dx, dy)` | move a mask; the uncovered border is zero (no wrap-around), fractions allowed |
| `drop_shadow(mask, offset=(6, 10), blur=8)` | the mask moved by `offset` and blurred: a shadow mask to paint under the subject |
| `stamp(mask, cx, cy, r, hardness=0.5, opacity=1)`, `stroke(mask, pts, r, hardness, opacity, spacing=0.3, r_end=None, jitter=0, seed=0)` | soft dabs and tapering brush strokes (in place) |

Combine masks with plain numpy: union `np.maximum(a, b)`, intersection `np.minimum(a, b)`, inverse `1 - a`.

```python
import numpy as np
from pixelcanvas import Canvas, masks

H, W = 270, 480
cv = Canvas(W, H, "#dfe6ee")
disc = masks.ellipse_mask(H, W, 240, 140, 120, 60, angle=-15)
cv.paint("#202a44", masks.drop_shadow(disc, (8, 14), 6), opacity=0.35)      # shadow first
cv.paint("#e0674a", disc)
tri = masks.polygon_mask(H, W, [(60, 230), (110, 140), (160, 230)])         # h, w first, then the points
cv.paint("#3a6ea5", np.minimum(tri, masks.rect_mask(H, W, 0, 150, 200, 260)))
```

## Text (`pixelcanvas.text`)

`cv.text(text, xy, size, color, anchor="la", font=None, bold=False, max_width=None, box=None, outline=0,
shadow=None, glow=None, gradient=None, ...)`; `text_mask(h, w, text, xy, size)`, `text_bbox(...)`,
`fit_text_size(...)`, `wrap_text(...)`, `find_font(name, bold, style, text)`, `audit_text(cv)`. Omit `font=` and
CJK picks a CJK font. Everything else: [typography.md](typography.md).

## Signed distance fields (`pixelcanvas.sdf`)

Inputs are pixel grids `X, Y`; outputs are float32 (H, W) distances in px, negative inside, positive outside.
Conventions: `offset(d, k)` with **k > 0 grows the shape** and k < 0 shrinks it (a ring of thickness t around a shape
is `subtract(offset(d, t), d)`); `subtract(a, b)` cuts b out of a; `smooth_union(a, b, k)` fillets the join by k px;
`annular(d, t)` makes a shell of thickness 2t centred on the edge; `outline(d, width)` strokes centred on the edge
(fully outside: `outline(offset(d, width / 2), width)`); `glow(d, radius)` is 1 at the edge and fades outward.
Primitives `sd_circle(X, Y, cx, cy, r)`, `sd_ellipse(..., rx, ry, angle)`, `sd_box(..., half_w, half_h, radius=0,
angle=0)`, `sd_segment(X, Y, ax, ay, bx, by, r)`, `sd_polygon(X, Y, pts)`, `sd_star(..., r_outer, r_inner, n)`,
`sd_ring`, `sd_arc`; operators `union`, `intersect`, `subtract`, `smooth_union(a, b, k)`, `offset(d, k)`,
`annular(d, half_t)`; rendering `fill(d, aa=1)` → mask, `outline(d, width)`, `glow(d, radius)`,
`bevel_light(d, width, light)` → brightness multiplier.

## Filters (`pixelcanvas.filters`, array in → array out)

`blend(base, top, alpha, mode)`, `gaussian_blur(a, sigma)`, `sample(img, X, Y)`, `resize(arr, w, h)`,
`radial_blur(img, center, strength, samples)`, `light_shafts(source_mask, center, samples, decay, length)`, `bloom`,
`vignette`, `grain`, `levels`, `contrast`, `saturation`, `soft_clip`, `filmic`, `split_tone`.

```python
cv = Canvas(320, 180)
X, Y = cv.xy()
cv.gradient([(0, "#0b1a3a"), (1, "#f6c28f")])
ridge = (0.7 - 0.15 * fbm1d(320, 90, 5, seed=2, mode="ridged")) * 180
cv.paint("#1a1f2e", clamp01(Y - ridge[None, :] + 0.5))                     # anti-aliased silhouette
cv.paint("#ffe2a8", sdf.fill(sdf.sd_circle(X, Y, 230, 70, 14)))             # crisp disc
cv.glow(230, 70, 30, "#ff9a55", 0.5).soft_clip(0.85)
```

## Vision helpers (`pixelcanvas.vision`)

What `tools/look.py` uses: `guides(img, step=0.1)`, `zoom(img, box)`, `value_map(img, levels)` → (image,
fractions), `edge_map(img)`, `contact_sheet(images, labels, cols, tile_width)`, `critique(img)`,
`image_stats(img)`, `stage(cv, label)`. Inputs: path, PIL, Canvas or array; image outputs are PIL images.

## Editable layers (`pixelcanvas.layers`)

Each `log("step")` (or `rec.mark("step")`) starts a layer; its alpha is the coverage of normal-mode `paint` /
`text` / `paste` during the step. Compositing the layers bottom → top reproduces the final image. A trailing `post`
step is folded into every layer; an object's soft shadow or glow becomes a `"<name> (halo)"` layer grouped with it.

```python
cv = Canvas(640, 360); rec = layers.LayerRecorder(cv)
rec.mark("sky");  cv.gradient([(0, "#0b1a3a"), (1, "#f6c28f")])
rec.mark("sun");  cv.glow(400, 200, 60, "#fff0c0")
rec.mark("post"); cv.soft_clip(0.8)
stack = rec.finish()                              # [Layer(name, rgba, bbox, size, visible, opacity, kind)]
layers.save_all(stack, "out/pic", final=cv.buf)   # pic.psd, pic.ora, pic_layers/ (PNGs + layers.json)
```

Also `composite(stack, hidden={...})`, `save_psd`, `save_ora`, `save_pngs`, `load_pngs(folder)`, `push_post`,
`reconcile`; `LayerRecorder(post="layer")` keeps post separate, `halos="keep"` keeps halos in the object.

## Manifest: what the report knows (`pixelcanvas.manifest`)

Kits record every object they draw in `cv.manifest`; `render.py --check` turns it into OBJECTS and PROMPT. Note
hand-painted subjects yourself:

```python
cv = Canvas(640, 360, "#dfe7ef")
X, Y = cv.xy()
kite = sdf.fill(sdf.sd_polygon(X, Y, [(200, 60), (250, 130), (200, 200), (150, 130)]))
cv.paint("#d0483c", kite)
manifest.note(cv, "object", "kite", mask=kite)              # or bbox=(x0, y0, x1, y1) in px or fractions
manifest.note(cv, "object", "tree", bbox=(0.6, 0.3, 0.8, 0.95))
manifest.note(cv, "object", "fence post", bbox=(0.1, 0.6, 0.4, 0.9), count=5)   # also backdrop=True, see_through=True
rows = manifest.summary(cv)                                  # name, centre, size, visible %, clipped %, hidden_by
print(manifest.relations(cv))                                # ['kite is left of tree']
issues = manifest.check(cv, prompt="two kites and a tree")   # prompt_count: 1 kite drawn
```

Also `coverage`, `prompt_status`, `quoted_status`, `count_phrases`, `note_raw`, `extent_from_args`, `register`,
`pasted`, `reconcile_cropped`, `prompt_relax`, `is_backdrop`, `is_see_through`, `META_KEYS`, `CUE` (light a renderer records).

## Landscapes (`pixelcanvas.scenery`)

`Scene(width, height, seed, log=None)`: `sc.cv`, grids `sc.X/Y/U/V`, `sc.S` (H / 1080), `sc.px(v)` (1080p px →
current), `sc.set_horizon(v)`, `sc.set_sun(x, y, light_dir=(dx, dy))`, `sc.set_haze(stops)`, `sc.occlude(mask)`
(blocks god rays). Template: `new_scene.py NAME --kind landscape`.

| Recipe | Key parameters |
|---|---|
| `sky(sc, stops)`, `sun(sc, glows, radius=27, disc, horizon_glow)` → disc mask | `disc=None` hides the disc |
| `cirrus(sc, band, ...)`, `cumulus(sc, band, coverage=0.56, softness=0.2, scale=340, warp=0.35, shade, lit, rim=0.4)` → density | smaller coverage = more cloud |
| `mountain_layer(sc, dict(base, amp, scale, seed, shadow, lit, fog, peaks, snow, forest, ...))` → mask | `fog` 0–1: far 0.7–0.8, near 0–0.1 |
| `ridge(...)`, `mountain(...)`, `valley_floor(sc, top, fade, colors)`, `mist(sc, v_range, strength)`, `haze_band(sc, ...)` | the parts of a range, mist bands |
| `river_course(sc, start, meander, width_far, width_near)` → (center, half, t1); `water_mask(...)` → (water, e) | meandering river |
| `water_surface(sc, water, e, t1, colors, fresnel, ripple, glitter)`, `river_bank(sc, water, e, t1)` | draw after mountains and banks |
| `pine_mask(sc, [(x, y, h, w), ...])`, `scatter_trees(sc, n, x_range, y_range, height, avoid)` | pine silhouettes |
| `rim_light(sc, mask, color)`, `ground_edge(sc, y, side, reach)`, `birds(sc, [(x, y, size, tilt)])` | finishing touches |
| `god_rays(sc, disc, clouds, strength=0.24)`, `finish(sc, bloom, knee, contrast, saturation, split, vignette, grain)` | light and post (None skips a step) |

## Tools (`python <root>/tools/<tool>.py`)

| Tool | Flags |
|---|---|
| `render.py SCENE` | `--scale 0.5` draft · `--check` report + `_check.jpg` · `--size WxH` · `--seed N` · `--out p.png` · `--outdir DIR` · `--deliver PATH` · `--crop X0 Y0 X1 Y1` · `--stages` · `--round` · `--set NAME=EXPR` · `--sweep NAME A B C` (NAME: an UPPER_CASE constant of the scene) · `--seeds 1 2 3` · `--style NAME` · `--styles a,b,c` · `--prompt TEXT` · `--kind painting\|design\|diagram\|drawing` · `--layers` · `--svg` · `--dxf` · `--json` · `--list` |
| `render.py SCENE --animate NAME START END` | `--frames 24` · `--fps 12` · `--format gif\|webp\|apng` · `--loop` · `--bounce` · `--hold N` · `--ease linear\|inout` · `--dither` |
| `new_scene.py NAME` | `--kind KIND` · `--prompt TEXT` · `--plain` · `--dir DIR` · `--size WxH` · `--list` |
| `compose.py SPEC.json` | `--check` · `--deliver PATH` · `--example KIND` · `--kinds` · `--schema` · `--names WORD` · `--set PATH=VALUE` · `--add JSON` · `--remove NAME` |
| `kit.py` | `find "request" [--n 1] [--json]` · `show KIT.NAME` · `list [KIT]` · `kinds` · `doc KIT` |
| `look.py` | `zoom IMG U0 V0 U1 V1` · `guides` · `values` · `edges` · `check` · `sheet A B --labels` · `rounds SCENE [--gif]` · `compare` · `diff` · `crop` · `grid` · `critique` · `probe IMG U,V` · `stats` · `report` · `ask IMG "q"` · `describe` (numbers take `--json`) |
| `layers.py` | `info` · `sheet` · `compose DIR --hide NAME --only NAME --out p.png` · `export DIR --psd p.psd --ora p.ora` |
| `styles.py` | `list` · `pick "request"` · `sheet SCENE --styles a,b` · `apply IMG NAME --strength 0.9` |
| `fonts.py` | `[pattern]` · `--sheet` · `--text "..."` |
| `pc.py` | `doctor` · `version` · `new` · `render` · `look` · `mcp` (same as the `pixelcanvas` command) |
| `install_skill.py` | `[claude\|agents\|opencode\|cursor\|codex]` · `--skill pixelcanvas\|pixelcanvas-lite\|both` · `--project` · `--dest DIR` · `--list` · `--uninstall` |
