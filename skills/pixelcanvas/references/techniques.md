# Technique library: how to draw X with math

Each technique is a snippet you can paste straight into a scene's `render()`. Every `python` block below assumes this prelude
has already run (`tests/test_skill_docs.py` actually executes each block, so the examples can't silently break):

```python
import numpy as np
from pixelcanvas import *

W, H = 1920, 1080
S = H / 1080                      # pixel constants are based on 1080p → multiply by S
cv = Canvas(W, H)
X, Y = cv.xy()                    # pixel coordinates (H, W)
U, V = cv.uv()                    # 0~1 coordinates
```

Full examples live in `scenes/`: `mountains_sunset` (high-level recipes), `night_sky`, `still_life`, `city_dusk`, `sea_sunset`.

---

## 1. Silhouettes: 1D profile → mask (mountains, hills, skylines, forest edges, wave crests)

One y value per column; wherever `Y` is greater than it is the object. `+ 0.5` then `clamp01` = 1px anti-aliasing.

```python
cv.gradient([(0, "#24324f"), (1, "#d9a77c")])
for base, amp, col, seed in [(0.62, 0.12, "#7d7a92", 1), (0.72, 0.10, "#4b4e66", 2), (0.85, 0.08, "#1d2030", 3)]:
    ridge = (base - amp * fbm1d(W, 0.3 * W, 6, seed=seed, mode="ridged")) * H   # ridged = sharp peaks
    cv.paint(col, clamp01(Y - ridge[None, :] + 0.5))
```
- Farther layers: higher (smaller base), paler (color closer to the bottom of the sky), smaller relief.
- Hills use `mode="plain"`; rocky mountains use `"ridged"`; for skylines make it stepped (see `scenes/city_dusk.py`).

## 2. Volumetric clouds: density field + sampling toward the light

A cloud's lit side = where density thins out toward the light. Shift the density toward the light and subtract it for cheap self-shadowing.

```python
cv.gradient([(0, "#3a5f95"), (1, "#a9c4dd")])
n = warp2d(H, W, 0.35 * H, 0.5, seed=4, aspect=0.55) * 0.5 + 0.5
dens = smoothstep(0.42 + 0.25 * V, 0.75 + 0.25 * V, n)       # threshold rises with height → more clouds up top, fewer below
lx, ly = 25 * S, -18 * S                                  # offset pointing toward the light (upper right)
shifted = gaussian_blur(sample(dens, X + lx, Y + ly), 4 * S)                          # density at the point toward the light
lit = clamp01(1.0 - 1.6 * shifted + 0.6 * dens)           # thinner toward the light → brighter
col = lerp("#6d7896", "#fff6ea", lit)                      # shadow color → lit color
cv.paint(col, smoothstep(0.0, 0.5, dens))                  # wider edge smoothstep → fluffier
```
- Clouds look like marble → threshold range too narrow or warp too strong; clouds look like cotton candy → softness too high.
- For landscapes just use `cumulus()` (three scales of lighting and rim light already done).

## 3. Normals → lighting (spheres, objects, embossing, terrain)

Anything you can express as a height field h(x, y) or a surface normal can be lit:
diffuse `max(n·L, 0)`, specular `(r·v)^k`, ambient, bounce light.

```python
L = np.array([-0.5, -0.6, 0.62]); L /= np.linalg.norm(L)  # points toward the light (upper left, front)
h = fbm2d(H, W, 0.15 * H, 5, seed=7, mode="ridged") * 0.15 * H                # height field (in pixels)
gy, gx = np.gradient(h)
n = np.stack([-gx, -gy, np.ones_like(h)], -1)
n /= np.linalg.norm(n, axis=-1, keepdims=True)             # surface normal
ndl = np.clip(n @ L, 0, 1)
albedo = mix([(0, "#5b4a3c"), (1, "#c9b59a")], h / h.max())                    # dark low, light high
cv.paint(albedo * (0.25 + 0.9 * ndl[..., None]))
```
- Compute a sphere's normal directly: `dx, dy = (X-cx)/r, (Y-cy)/r; dz = sqrt(1-dx²-dy²)`; full example in `scenes/still_life.py`.
- To make an object "sit" on the ground: cast shadow attached to the contact point + a narrow band of AO (contact shadow).

## 4. Perspective ground / water: invert to world coordinates

The horizon is at `hz`. Each screen row maps to a depth `z ∝ 1 / (Y - hz)`; sample noise in world coordinates
and distant texture naturally gets finer and denser. Fade it out in the distance or you'll get moiré.

```python
hz = 0.5 * H
cv.gradient([(0, "#7fa6d6"), (0.5, "#e8eef2"), (1, "#e8eef2")])
dy = np.maximum(Y - hz, 0.5)
z = 0.5 * H / dy                                           # z ≈ 1 at the bottom of the frame
wx = (X - W / 2) / H * z
tex = fbm2d_at(wx * 14, z * 14, 5, seed=3) * (1 - smoothstep(3, 20, z))       # grass / sand texture (fades out in the distance)
ground = lerp("#3f5a2c", "#8aa05a", tex * 0.5 + 0.5)
ground = lerp(ground, "#c8d6dc", smoothstep(1, 12, z))                         # closer to fog color with distance
cv.paint(ground, smoothstep(hz - 0.5, hz + 0.5, Y))
```

## 5. Reflections (water, wetland, glass)

Mirror the current canvas about the eye-level line: `cv.sample(X + jitter, 2*hz - Y + jitter)`.
**Draw everything being reflected first.** Near-total reflection in the distance (Fresnel), deep water color showing through up close; glints only directly below the light.

```python
hz = 0.6 * H
cv.gradient([(0, "#1b2a4a"), (0.6, "#f0a070"), (1, "#f0a070")])
ridge = (0.6 - 0.2 * fbm1d(W, 0.25 * W, 6, seed=5, mode="ridged")) * H     # ridged 0~1 → ridge line always above the water
cv.paint("#2b2440", clamp01(Y - ridge[None, :] + 0.5))
ripple = fbm2d(H, W, 0.3 * W, 4, seed=6, aspect=0.04)                         # stretched horizontally → ripples
refl = cv.sample(X + ripple * 6 * S, 2 * hz - Y + ripple * 10 * S)
t = clamp01((Y - hz) / (H - hz))                                               # 0 far → 1 near
water = lerp(refl * 0.92, "#0f1a28", 0.15 + 0.6 * t ** 1.5)
cv.paint(water, smoothstep(hz - 0.5, hz + 0.5, Y))
```
- For landscape rivers just use `river_course` + `water_mask` + `water_surface`; for the sea see `scenes/sea_sunset.py`.

## 6. Scattering points: stars, snow, fireflies, bokeh, dust

Random coordinates + **power-law brightness** (most dim, a few bright) → `np.add.at` into a mask → slight blur.
Don't draw thousands of points in a Python loop; add glow one by one only to the few bright ones.

```python
cv.gradient([(0, "#03050d"), (1, "#1a2a4e")])
rng = np.random.default_rng(1)
n = 2500
px, py = rng.uniform(0, W, n), rng.uniform(0, H, n)
b = rng.random(n) ** 6 * 1.5 + 0.1
pts = np.zeros((H, W), F32)
np.add.at(pts, (py.astype(int), px.astype(int)), b)
cv.paint("#ffffff", gaussian_blur(pts, 1.2 * S) * 12 * S * S, "add")         # blur radius ∝ S → compensate brightness by S²
for i in np.argsort(-b)[:10]:
    cv.glow(px[i], py[i], 6 * S, "#cfe0ff", strength=0.4)
```
- Snow: uniform brightness, slightly larger radius, bigger near and smaller far; bokeh: large soft circles (`stamp` hardness 0.8), low opacity, `add` mode.

## 7. Grid patterns: windows, brick walls, floor tiles, checkerboards

Integer cells `gx = X // cell`; use one random value per cell (`perlin2d_at(gx*a, gy*b)` or a hash),
and `X % cell` inside the cell to leave a border.

```python
cv.paint("#2a2230")
cell = 24 * S
gx, gy = X // cell, Y // cell
on = perlin2d_at(gx * 0.71, gy * 0.37, seed=2) > 0.2                          # which windows are lit
pane = (X % cell > cell * 0.3) & (Y % cell > cell * 0.35)                       # window frame
glow = gaussian_blur((on & pane).astype(F32), 2 * S)
cv.paint("#ffcf7a", (on & pane) * 0.9).paint("#ff9a40", glow * 0.4, "add")
```
- Brick walls: offset odd rows by `X + cell/2`; ±10% random brightness per brick + fine fBm surface.

## 8. Light: glows, volumetric light, rim light

```python
cv.gradient([(0, "#0e1424"), (1, "#3b2e3e")])
sx, sy = 0.7 * W, 0.3 * H
cv.glow(sx, sy, 0.25 * H, "#ff9a55", strength=0.35)          # large glow (atmospheric scattering)
cv.glow(sx, sy, 0.04 * H, "#fff1d6", strength=0.9)           # small glow
disc = circle_mask(H, W, sx, sy, 0.025 * H, feather=1.5 * S)
cv.paint((1.8, 1.6, 1.3), disc)                               # HDR light source itself > 1
occl = clamp01(Y - ((0.75 - 0.1 * fbm1d(W, 0.2 * W, seed=9)) * H)[None, :] + 0.5)
cv.paint("#0a0c12", occl)
shafts = light_shafts(disc * (1 - occl) + 0.15 * (1 - occl) * np.exp(-((X - sx) ** 2 + (Y - sy) ** 2) / (0.2 * H) ** 2),
                      (sx, sy), samples=32, decay=0.96, length=0.9)
cv.paint("#ffc98a", shafts * 0.6, "add")
cv.bloom(0.85, 0.02 * H, 0.3).soft_clip(0.85)
```
- Rim light (bright edge on the light-facing side of a silhouette): `edge = mask - np.roll(np.roll(mask, dy, 0), dx, 1)`, take the positive part and paint a warm color;
  for landscapes use `rim_light(sc, mask)`.

## 9. Arbitrary shapes and strokes: Pillow supersampled masks, brushes

```python
cv.gradient([(0, "#bcd4e6"), (1, "#eef3f6")])
def house(d, s):                                              # multiply every coordinate by s (supersampling factor)
    x0, y0 = 0.4 * W, 0.55 * H
    w, h = 0.2 * W, 0.2 * H
    d.rectangle([x0 * s, y0 * s, (x0 + w) * s, (y0 + h) * s], fill=255)
    d.polygon([(x0 - 0.02 * W) * s, y0 * s, (x0 + w / 2) * s, (y0 - 0.12 * H) * s, (x0 + w + 0.02 * W) * s, y0 * s], fill=255)
cv.paint("#6b4b3a", shape_mask(H, W, house))
grass = np.zeros((H, W), F32)
rng = np.random.default_rng(3)
for _ in range(300):                                          # grass blades: thin curved strokes
    x, y = rng.uniform(0, W), rng.uniform(0.74 * H, H)
    hgt = rng.uniform(20, 60) * S
    lean = rng.uniform(-15, 15) * S
    stroke(grass, [(x, y), (x + lean * 0.4, y - hgt * 0.6), (x + lean, y - hgt)], 2.0 * S, r_end=0.5 * S, hardness=0.6)
cv.paint("#3d5a2a", band_mask(H, W, 0.75 * H, 2 * H, soft=4 * S))
cv.paint("#557a35", grass)
```
- For many objects (thousands of trees), draw them all in one `shape_mask` draw_fn; much faster than building a mask per object (`scenery.pine_mask`).

## 10. Text and posters

```python
cv.gradient([(0, "#1d1b3a"), (1, "#e2566e")], axis="x")
flow = warp2d(H, W, 0.3 * H, 1.2, seed=8) * 0.5 + 0.5
cv.paint("#ffb86b", smoothstep(0.55, 0.75, flow) * 0.5, "screen")
title = text_mask(H, W, "PIXEL", (W / 2, H / 2), size=int(0.22 * H), anchor="mm")
cv.paint("#000000", feather(title, 8 * S) * 0.5)               # soft shadow first
cv.paint("#fff6e8", title)
cv.vignette(0.3).grain(0.015)
```
- Fonts are found automatically and CJK text picks a CJK font (`find_font`, `python tools/fonts.py --sheet`);
  pass `font="georgia"` / a path to choose one. If you force a Latin-only font on CJK text you get empty boxes.
- For styled text (outline, shadow, glow, gradient, auto-fit into a box, wrapping) use `cv.text(...)`;
  layout, hierarchy and poster recipes are in [typography.md](typography.md). Crisp vector shapes (badges, buttons,
  icons, geometric posters) → `pixelcanvas.sdf`, also in typography.md.

## 11. Photo editing (photo → adjustments)

```python
from PIL import Image
Image.fromarray((np.random.default_rng(0).random((H, W, 3)) * 255).astype(np.uint8)).save("photo_demo.jpg")  # demo only
cv = Canvas.load("photo_demo.jpg")                             # ← replace with the user's photo
X, Y = cv.xy(); U, V = cv.uv()
cv.levels(0.03, 0.97, 1.05)                                    # black/white points + gamma
sky = smoothstep(0.55, 0.85, luma(cv.buf)) * (1 - smoothstep(0.3, 0.6, V))    # bright and near the top → sky mask
cv.paint(cv.buf * 0.85, sky)                                   # locally darken the sky
cv.split_tone("#3d4a66", "#f2c58a", 0.08).saturation(1.08).vignette(0.3).grain(0.01)
cv.save("photo_demo_edit.jpg")
```
- Local adjustment = build a mask first (luminance, color, position, hand-drawn `shape_mask`), then `cv.paint(adjusted buf, mask)`.

## 12. Snowfields: blue shadows + warm lit areas + subtle sparkles

Snow is almost white, so **all the depth comes from light**: lit faces take the light's color (sunset → pink-orange), shaded faces reflect the sky (blue-violet),
undulations use low-frequency, soft noise (snow buries detail, so no high-frequency / ridged), and sparkles are very sparse and only on lit faces.

```python
cv.gradient([(0, "#3a3f78"), (0.5, "#c99ab8"), (1, "#c99ab8")])
top = 0.55 * H
dy = np.maximum(Y - top, 0.5)
z = 0.45 * H / dy                                          # perspective depth (see section 4)
wx = (X - W / 2) / H * z
h = fbm2d_at(wx * 1.0, z * 0.8, 4, gain=0.4, seed=11) * (1 - smoothstep(3, 14, z))  # low frequency, small gain → soft snow drifts
slope = np.gradient(h, axis=1) * dy / (0.02 * H)            # left/right slope; light from the right → positive is lit
lit = clamp01(0.55 + 1.6 * slope)
snow = lerp("#7d7fb8", "#ffe6e0", lit)                       # shaded = sky blue-violet, lit = sunset pink
snow = lerp(snow, "#d9b4c8", smoothstep(3, 14, z) * 0.7)    # blends into the horizon haze in the distance
rng = np.random.default_rng(5)
spark = (rng.random((H, W)) > 0.9995) * smoothstep(0.6, 0.9, lit)             # sparse sparkles, lit faces only
snow = snow + rgb((1.2, 1.1, 1.0)) * gaussian_blur(spark.astype(F32), 0.7 * S)[..., None] * 3
cv.paint(snow, smoothstep(top - 0.5, top + 0.5, Y))
```
- Shadows of trees / rocks on snow: stretch them opposite the light vector, color them blue-violet (not black); add a narrow, dark ring of AO at tree bases.
- Don't use the same large blotchy noise as the sky on snow (it reads as water reflections or cloud shadows).

## 13. Media: painting, watercolour, sketch, anime (`pixelcanvas.stylize`)

Draw the subject normally (clear shapes, lighting), then convert the finished image to the medium as the last step.
The filters are array in → array out; pixel sizes × `S`.
```python
from pixelcanvas import stylize
cv.gradient([(0, "#5a86c8"), (0.6, "#cfe0f0"), (1, "#6d9a4a")])
cv.paint("#c83a2a", sdf.fill(sdf.sd_circle(X, Y, 0.5 * W, 0.55 * H, 0.18 * H)))      # the subject
cv.paint("#2c5a22", clamp01(Y - (0.8 * H + 20 * S * np.sin(X / (90 * S))) + 0.5))
cv.apply(stylize.oil_paint, brush=6 * S, seed=1)     # Kuwahara colour blocks + LIC strokes along edges + impasto
cv.apply(stylize.canvas_texture, strength=0.06)      # "on canvas"
# alternatives: stylize.watercolor(img) / stylize.pencil_sketch(img) / stylize.cel_shade(img, levels=3, outline=2*S)
```
- `oil_paint(brush=…)`: 4~6 px × S = careful painting, 8~14 px × S = loose impressionism. Strokes follow edges
  (structure-tensor flow), so silhouettes survive; very small details (text, thin lines) do not — draw them after.
- `watercolor()`: transparent desaturated washes, pigment pooling at wash borders, granulation, white deckled border.
- `pencil_sketch(colored=0~1)`: dodge-blend contours + hatching in the darks; `colored` for coloured pencil.
- `cel_shade(levels, outline)`: flat bands + ink outlines. For anime also draw simple shapes and big flat areas first;
  the filter cannot invent line art from noisy textures.
- The building blocks are public: `kuwahara(img, r)`, `flow_field(img, sigma)` → (vx, vy, coherence),
  `lic(tex, vx, vy, length)` for any directional texture (hair, fur, wood grain, wind-blown grass).

---

## Palette table (palette starting points, (V fraction, color))

| Mood | Sky gradient | Fog / distance | Foreground / silhouette | Accent |
|---|---|---|---|---|
| Golden-hour dusk | `#0c1c3e` → `#3f6f9f` → `#e6a86c` → `#f3c89c` | `#e2b48f` | `#10191a` | Sun `(1.8,1.6,1.3)` |
| Blue hour | `#0b1230` → `#2c3f78` → `#8a7fb0` → `#e7a3a0` | `#9a90b8` | `#141626` | Window light `#ffcf7a` |
| Starry night | `#02040c` → `#0b1838` → `#23385f` | `#141d33` | `#04060c` | Stars `#ffffff`, moon `#f2efe4` |
| Clear noon | `#2f6fc4` → `#7fb2e6` → `#d6e8f5` | `#c9dbe8` | `#2f4a2a` | Clouds `#ffffff` / cloud shadow `#a9b8cc` |
| Misty morning | `#9fb0bd` → `#d7dcd9` → `#eae6dc` | `#dcdcd4` | `#4a5550` | Low contrast, saturation 0.8 |
| Autumn | `#4f7fb5` → `#d9c3a0` | `#d8b892` | `#3a2416` | Maple red `#c2502a`, gold `#e0a43a` |
| Snowfield | `#8fa9c8` → `#e8eef4` | `#dfe6ee` | `#2a3442` | Bluish shadows `#9fb3cf` |
| Underwater | `#7fd0e0` → `#1c6c8c` → `#062438` | `#1c5a74` | `#03121c` | Light beams `#bff4ff` |
| Neon | `#0a0618` → `#2a0f4a` → `#5a1a6e` | `#3a1a5a` | `#07040f` | `#ff3cac` `#2bd2ff` `#ffe66d` |

Principle: cool shadows, warm lights (or vice versa), never the same hue family for both; save saturation for the focal point; keep large areas low-saturation.
