# pixelcanvas API reference

```python
import numpy as np
from pixelcanvas import *            # low level: Canvas, colour, noise, masks, brushes, filters, text
from pixelcanvas import sdf          # signed-distance shapes (vector-like, crisp at any resolution)
from pixelcanvas.scenery import *    # high level: Scene + landscape recipes
```

Conventions: x right, y down, arrays `arr[y, x]`; masks are float32 (H, W) in 0~1; colours may be `'#rrggbb'` /
`'#rgb'` / `(r, g, b)` 0~1 / `(255, 128, 0)` / a full (H, W, 3) colour map; HDR > 1 is allowed (finish with
`soft_clip`). Every Canvas method returns `self` (chainable); `cv.buf` is the float32 (H, W, 3) array — edit it freely.

## Canvas

| Method | Description |
|---|---|
| `Canvas(w, h, color=(0,0,0))` / `Canvas.load(path)` / `Canvas.from_array(arr)` | create / load / from array ((H,W) grey → 3 channels) |
| `cv.w, cv.h, cv.shape, cv.copy(), cv.to_pil()` | size, copy, PIL image |
| `cv.xy()` → `(X, Y)` pixel grids; `cv.uv()` → `(U, V)` 0~1 | coordinate grids (float32 (H, W)) |
| `cv.get(x, y)` / `cv.put(x, y, color)` | read / write pixels; x, y may be int arrays, color may be (N, 3) |
| `cv.region(x0, y0, x1, y1)` | sub-array view (edits apply) |
| `cv.paint(color, mask=None, mode="normal", opacity=1)` | **the core colouring call**: blend by mask. modes: normal / add / multiply / screen / overlay / softlight / lighten / darken |
| `cv.gradient(stops, axis="y")` | axis-aligned linear gradient, `stops = [(0~1, colour), ...]` |
| `cv.linear(p0, p1, stops, mask=None)` | linear gradient in any direction (0 at point p0 → 1 at p1) |
| `cv.radial(cx, cy, r, stops)` | radial gradient (0 = centre, 1 = radius r) |
| `cv.glow(cx, cy, sigma, color, strength=1, mode="add")` | Gaussian glow (can go HDR) |
| `cv.text(text, xy, size, color, **style)` | styled text, see **Text**; bbox in `cv.last_bbox` |
| `cv.paste(other, x, y, mask=None, mode=..., opacity=...)` | paste another Canvas / array (auto-clipped) |
| `cv.sample(X, Y)` | sample the current image at float coordinates (reflections, distortion) |
| `cv.displace(dx, dy)` / `resize(w, h)` / `crop(...)` / `flip("y" or "x")` | geometry |
| `cv.blur(s)` `bloom(th, sigma, k)` `vignette(k, r, soft)` `grain(amt, seed)` `levels(black, white, gamma)` `contrast(k)` `saturation(s)` `soft_clip(knee)` `filmic()` `split_tone(shadow, highlight, k)` `clip()` | post-processing |
| `cv.apply(fn, *args)` | `cv.buf = fn(cv.buf, *args)` |
| `cv.save(path, preview=None, preview_width=960)` / `cv.stats()` | save (optionally a preview) / print luminance stats (>1 fraction > 3% is usually too bright) |

## Colour & maths (pixelcanvas.color)

| Function | Description |
|---|---|
| `rgb(c)` / `as_color(c)` | to float32 ndarray |
| `lerp(a, b, t)` | linear interpolation; a, b colours or colour maps, t may be an (H, W) mask → (H, W, 3) |
| `mix(stops, t)` | multi-stop gradient lookup: `mix([(0,'#000'),(1,'#fff')], V)` → (H, W, 3) |
| `hsv_to_rgb(h, s, v)` / `luma(img)` | colour space / luminance |
| `clamp01(a)` / `smoothstep(e0, e1, x)` / `normalize(a, lo, hi)` | everyday maths (smoothstep accepts array edges; e0 > e1 reverses) |
| `grid(h, w)` / `F32` | pixel grids (X, Y) / `np.float32` |

## Noise (pixelcanvas.noise)

`scale` = pixel width of one cell of the lowest octave (bigger = smoother); `aspect` > 1 stretches vertically, < 1
horizontally; `mode`: `"plain"` (≈ -1~1, terrain / clouds), `"ridged"` (0~1, sharp ridges), `"billow"` (0~1, puffy).

| Function | Description |
|---|---|
| `fbm2d(h, w, scale, octaves=6, lacunarity=2, gain=0.5, seed=0, offset=(0,0), aspect=1, mode="plain")` | full-frame fBm (fast path; 1080p, 6 octaves ≈ 1 s) |
| `fbm2d_at(X, Y, octaves, ..., seed, mode)` | fBm at arbitrary coordinates (in cells) → perspective, warps, polar |
| `perlin2d(h, w, scale, seed)` / `perlin2d_at(X, Y, seed)` | single-octave Perlin |
| `warp2d(h, w, scale, strength=1, octaves=5, seed, aspect, mode)` | domain warping: flow / smoke / marble / galaxy |
| `fbm1d(n, scale, octaves=5, ..., seed, mode)` / `perlin1d_at(X, seed)` | 1D (ridgelines, shorelines, rooftops) |
| `smooth1d(a, sigma)` | 1D Gaussian smoothing |

## Masks, curves & brushes (pixelcanvas.masks / brush)

| Function | Description |
|---|---|
| `shape_mask(h, w, draw_fn, aa=3)` | draw anything with Pillow ImageDraw; in `draw_fn(d, s)` multiply every coordinate by s (supersampling) |
| `polygon_mask(h, w, points)` / `circle_mask(h, w, cx, cy, r, feather=1)` / `rect_mask(h, w, x0, y0, x1, y1, feather=0)` / `band_mask(h, w, y0, y1, soft=0)` | geometric masks |
| `line_mask(h, w, points, width=2, closed=False)` | anti-aliased polyline / curve of constant width, round joins and caps |
| `bezier_points(p0, p1, p2[, p3, ...], n=64)` | sample a quadratic / cubic / higher Bézier → list of (x, y) |
| `feather(mask, sigma)` | soften edges |
| `stamp(mask, cx, cy, r, hardness=0.5, opacity=1)` | soft round dab on a mask (in place) |
| `stroke(mask, points, r, hardness, opacity, spacing=0.3, r_end=None, jitter=0, seed=0)` | brush stroke along a polyline (tapering, jitter) |

## Text (pixelcanvas.text) — see [typography.md](typography.md)

| Function | Description |
|---|---|
| `cv.text(text, xy, size, color, anchor="la", font=None, bold=False, align=None, max_width=None, line_spacing=1.2, tracking=0, box=None, box_anchor="mm", padding=0, gradient=None, outline=0, outline_color, shadow=None, glow=None, opacity=1, mode="normal")` | styled text; same as `draw_text(cv, ...)` which returns the bbox |
| `text_mask(h, w, text, xy, size=40, font=None, bold=False, anchor="la", align, max_width, line_spacing, tracking, stroke=0)` | text coverage mask (paint it with any colour map / texture) |
| `text_bbox(text, xy, size, ...)` | `(x0, y0, x1, y1)` without drawing |
| `fit_text_size(text, box_w, box_h=None, font, bold, wrap=False)` | largest size that fits |
| `wrap_text(text, font, max_width)` | line list (word wrap for Latin, per-character for CJK, no line starts with closing punctuation) |
| `find_font(name=None, bold=False, style="sans", text=None)` / `list_fonts(pattern)` / `load_font(...)` / `has_cjk(text)` | font lookup; styles sans / serif / mono / cjk / cjk-serif; CJK text auto-selects a CJK font |

- Anchor = horizontal `l/m/r` + vertical `t/m/b` (also `a` ascender, `s` first baseline, `d` bottom). `"mm"` centres.
- `box=(x0, y0, x1, y1)` auto-fits the size and places by `box_anchor`; with `max_width=True` it wraps to the box (paragraph).
- `shadow=dict(dx, dy, blur, color, opacity)` or `True`; `glow=dict(radius, color, strength)` or `True`; colours may be HDR.
- Env: `PIXELCANVAS_FONT`, `PIXELCANVAS_CJK_FONT` (default font paths), `PIXELCANVAS_FONT_DIR` (extra search dirs).

## Signed distance fields (pixelcanvas.sdf) — see [typography.md](typography.md) §5

All take pixel grids `X, Y` and return float32 (H, W) distances in px (negative inside).

| Group | Functions |
|---|---|
| primitives | `sd_circle(X,Y,cx,cy,r)`, `sd_ellipse(X,Y,cx,cy,rx,ry,angle)`, `sd_box(X,Y,cx,cy,half_w,half_h,radius=0,angle=0)`, `sd_segment(X,Y,ax,ay,bx,by,r)` (capsule), `sd_polygon(X,Y,points)`, `sd_star(X,Y,cx,cy,r_outer,r_inner,n,angle)`, `sd_ring(X,Y,cx,cy,r,thickness)`, `sd_arc(X,Y,cx,cy,r,start_deg,end_deg,thickness)` |
| operators | `union(*ds)`, `intersect(*ds)`, `subtract(a, b)`, `smooth_union(a, b, k)`, `offset(d, k)` (grow > 0 / shrink < 0), `annular(d, half_t)` (hollow) |
| rendering | `fill(d, aa=1)` → mask, `outline(d, width)` → mask, `glow(d, radius, falloff=1.5)` → halo mask, `bevel_light(d, width, light=(-0.6,-0.7), strength=0.5)` → brightness multiplier (≈ 0.5~1.5) |

## Filters (pixelcanvas.filters, array in → array out)

`blend(base, top, alpha, mode)`, `gaussian_blur(a, sigma)` (2D or 3D), `sample(img, X, Y)`, `resize(arr, w, h)`,
`radial_blur(img, center, strength, samples)`, `light_shafts(source_mask, center, ...)`, `bloom`, `vignette`, `grain`,
`levels`, `contrast`, `saturation`, `soft_clip`, `filmic`, `split_tone`.

## Vision helpers (pixelcanvas.vision) — what `tools/look.py` uses

`guides(img, step=0.1)`, `zoom(img, box)`, `value_map(img, levels)` → (image, fractions), `edge_map(img)`,
`contact_sheet(images, labels, cols, tile_width)`, `stage(cv, label)` (mark a drawing step for `render.py --stages`;
no-op otherwise). Inputs: path / PIL / Canvas / array; outputs: PIL images.

## Media (pixelcanvas.stylize) — techniques.md §13

Array in → array out, use with `cv.apply(fn, ...)`: `oil_paint(img, brush=6, strength=1, impasto=0.35, seed)`,
`watercolor(img, wash=6, transparency=0.35, edge=0.8, granulation=1, bleed=6, paper=1, deckle=0.05, seed)`,
`pencil_sketch(img, line=1, hatching=0.8, colored=0, seed)`, `cel_shade(img, levels=3, softness=0.04, outline=2)`,
`canvas_texture(img, strength=0.06, weave=3)`; building blocks `kuwahara(img, radius)`,
`flow_field(img, sigma)` → (vx, vy, coherence), `lic(tex, vx, vy, length, steps)`; `MEDIA` maps prompt words to
the filter.

## Editable layers (pixelcanvas.layers) — what `render.py --layers` uses

Each `log("step")` / `rec.mark("step")` starts a layer; layer alpha = coverage of normal-mode `paint` / `text` /
`paste` masks during the step, raised to the minimum needed to reproduce the result exactly (glows / post stay
see-through). Layers are straight-alpha uint8 RGBA cropped to `bbox`; compositing them bottom → top = final image.

```python
from pixelcanvas import Canvas, layers
cv = Canvas(640, 360); rec = layers.LayerRecorder(cv)
rec.mark("sky");  cv.gradient([(0, '#0b1a3a'), (1, '#f6c28f')])
rec.mark("sun");  cv.glow(400, 200, 60, '#fff0c0')
rec.mark("post"); cv.soft_clip(0.8)
stack = rec.finish()                       # [Layer(name, rgba, bbox, size, visible, opacity, kind)], bottom → top
layers.save_all(stack, "out/pic", final=cv.buf)   # pic.psd, pic.ora, pic_layers/ (PNGs + layers.json)
```
`layers.enable(target_size)` / `mark(name)` / `take(cv)` (module-level recorder following the first canvas of that
size), `composite(stack, hidden={...})`, `save_psd`, `save_ora`, `save_pngs`, `load_pngs(folder)` (read an edited
stack back), `reconcile(stack, final)`, `push_post(stack, final)` (fold a trailing post step into every layer:
fitted colour LUT + smooth gain, leftovers → thin "post (residual)" layer; `LayerRecorder(post="layer")` keeps the
old separate post layer). Layer `kind`: `base` / `paint` (has painted coverage) / `effect` / `post`.

## High-level recipes (pixelcanvas.scenery) — landscapes

`Scene(width, height, seed, log=None)`: `sc.cv` canvas, `sc.X/Y/U/V` grids, `sc.S` scale (H/1080),
`sc.px(v)` converts 1080p px to the current resolution, `sc.terrain` union of opaque things, `sc.haze` haze colour map,
`sc.D` distance to the sun.

| Setting | Description |
|---|---|
| `sc.set_horizon(v)` | eye level (mirror axis for water) |
| `sc.set_sun(x, y, light_dir=(dx, dy))` | sun position (fractions) and the direction towards the light |
| `sc.set_haze(stops)` | haze colour by height (target of atmospheric perspective) |
| `sc.occlude(mask)` | add an opaque object to terrain (so god_rays are blocked) |

| Recipe | Returns | Key parameters |
|---|---|---|
| `sky(sc, stops)` | — | vertical multi-stop gradient |
| `sun(sc, glows, radius=27, disc=(1.8,1.6,1.3), horizon_glow)` | sun disc mask | glows = ((sigma px, strength, colour), ...); `disc=None` = no disc (set / behind clouds); `horizon_glow=None` disables the warm horizon haze |
| `cirrus(sc, band, scale=480, aspect=0.1, threshold, alpha, colors)` | alpha | high wispy clouds |
| `cumulus(sc, band=(fade-in start, fade-in end, fade-out start, fade-out end), coverage=0.56, softness=0.2, scale=340, warp=0.35, aspect=0.5, shade=(c, c), lit=(c, c), rim=0.4)` | cloud density (pass to god_rays as occluder) | **smaller coverage = more cloud**; larger softness = fluffier edges; `shade` / `lit` = (away from sun, towards sun); `rim` = edge highlight strength (0 = off) |
| `mountain_layer(sc, dict(base, amp, scale, seed, shadow, lit, fog, ...))` | mountain mask | `fog` 0~1 = blend towards `sc.haze` (**bigger = paler, further**; near 0~0.1, farthest 0.7~0.8); `base` ridge centre height, `amp` relief, `scale` ridge noise cell (1080p px). Optional: peaks=[(x, height, half-width)], dips, snow (snowline), forest, treeline, relief, side/rise/reach (near slopes), fog_down, tree_px |
| `ridge(...)` / `mountain(...)` | ridge (W,) / mask | the two halves of mountain_layer |
| `valley_floor(sc, top, fade, colors, fog_range)` | mask | valley meadow |
| `mist(sc, v_range, strength, scale, exclude)` / `haze_band(sc, ...)` | — | mist band / thin horizon haze |
| `river_course(sc, start, meander, width_far, width_near, ...)` | (center, half, t1) | meandering river (one value per row) |
| `water_mask(sc, center, half, t1)` | (water, e) | e = signed distance to the bank |
| `water_surface(sc, water, e, t1, colors, fresnel, ripple, glitter, ...)` | — | reflection + ripples + glitter (draw after mountains and banks) |
| `river_bank(sc, water, e, t1)` | — | pebbles along the banks |
| `pine_mask(sc, [(x, y_base, h, w), ...])` / `scatter_trees(sc, n, x_range, y_range, height, avoid)` | mask / tree list | pine silhouettes |
| `rim_light(sc, mask, color, offset=(3,-2), strength)` | — | lit edge on silhouettes |
| `ground_edge(sc, y_frac, side, reach)` | mask | dark ground in the bottom corners |
| `birds(sc, [(x, y, size px, tilt), ...])` | — | birds |
| `god_rays(sc, disc, clouds, strength=0.24, ...)` | — | volumetric light (blocked by terrain / clouds) |
| `finish(sc, bloom, knee, contrast, saturation, split, vignette, grain)` | — | standard post; pass None to skip a step |

## Tools

```bash
python tools/render.py <scene> [--scale 0.5] [--seed N] [--size WxH] [--out p.png] [--crop x0 y0 x1 y1]...
                       [--check] [--stages] [--layers] [--set NAME=EXPR]... [--sweep NAME EXPR EXPR...] [--seeds N N...]
python tools/layers.py info|sheet|compose|export <out>_layers [--hide NAME...] [--only NAME...]
python tools/render.py --list
python tools/new_scene.py <name> [--poster]
python tools/look.py check|guides|zoom|values|edges|sheet|compare|diff|crop|grid|critique|probe|stats|report ...
python tools/fonts.py [pattern] [--sheet] [--text "..."]
pixelcanvas info img.png / pixelcanvas preview img.png small.jpg    # after pip install -e .
```
