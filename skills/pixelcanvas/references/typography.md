# Typography, shapes & layout: posters, cards, thumbnails, UI mock-ups, labelled diagrams

Text and crisp vector shapes are where code beats image models: letters come out exactly right, in any language.
Every snippet below runs as-is after this prelude (`tests/test_skill_docs.py` executes them all):

```python
import numpy as np
from pixelcanvas import *
from pixelcanvas import sdf

W, H = 1920, 1080
S = H / 1080                      # 1080p-based pixel constants → multiply by S
cv = Canvas(W, H)
X, Y = cv.xy()
U, V = cv.uv()
```

Finished example: `scenes/neon_poster.py` (title block, striped SDF sun, perspective grid, info card).

---

## 1. Drawing text: `cv.text(...)`

`cv.text(text, (x, y), size, color, anchor=...)` paints anti-aliased text and stores its bounding box in
`cv.last_bbox` = `(x0, y0, x1, y1)`. Anchor = horizontal `l/m/r` + vertical `t/m/b` (`"mm"` = centred on the point).
`\n` makes new lines; `max_width=` wraps (spaces for Latin, between any characters for CJK).

```python
cv.gradient([(0, "#f4efe6"), (1, "#e6dccb")])
cv.text("Quiet Mornings", (W / 2, 0.40 * H), 110 * S, "#1f1a17", anchor="mm", font="georgia", bold=True)
title = cv.last_bbox
cv.text("A STUDY IN SOFT LIGHT", (W / 2, title[3] + 30 * S), 26 * S, "#8a6f55", anchor="mt", tracking=0.3)
cv.text("清晨的光，安靜地落在桌上。", (W / 2, title[3] + 90 * S), 40 * S, "#3b302a", anchor="mt")
```
- Fonts: `font=None` picks a good default (sans; CJK font automatically when the text contains CJK);
  `font="georgia"`, `font="msjh"`, `font="Noto Serif TC"` or a path selects one; `bold=True`.
  Run `python tools/fonts.py --sheet` to *see* the fonts on this machine before choosing.
- `tracking=0.2~0.4` (letter spacing as a fraction of size) for small all-caps labels; never on lowercase body text.
- Sizes: think in a scale — title : subtitle : body ≈ 4 : 1.6 : 1 (e.g. 110 / 44 / 26 at 1080p). Two or three sizes only.

## 2. Fit into a box, paragraphs, alignment

`box=(x0, y0, x1, y1)` auto-sizes the text to the box (minus `padding`) and places it by `box_anchor`.
`max_width=True` with a box = paragraph mode (wraps to the box width, then fits the height).

```python
cv.gradient([(0, "#10202a"), (1, "#1f3b47")])
col = (0.08 * W, 0.15 * H, 0.46 * W, 0.85 * H)                    # a text column
cv.text("FIELD NOTES", box=(col[0], col[1], col[2], col[1] + 0.12 * H), box_anchor="lm", color="#e8f1f2", bold=True)
body = ("Fog rolled in before dawn and stayed until noon. The ridge disappeared, then the pines, "
        "then the lake itself. 霧從黎明前開始，一直到中午才散。")
cv.text(body, box=(col[0], col[1] + 0.16 * H, col[2], col[3]), box_anchor="lt", max_width=True,
        color="#b8cdd2", line_spacing=1.45)
x0, y0, x1, y1 = text_bbox("Measure first", (0, 0), 40 * S)      # measure without drawing
size = fit_text_size("ONE LINE THAT MUST FIT", 0.4 * W, 0.1 * H)  # largest size that fits
cv.text("ONE LINE THAT MUST FIT", (0.75 * W, 0.5 * H), size, "#ffd27a", anchor="mm", bold=True)
```
- Line length for body text: 45~75 Latin characters or 20~35 CJK characters per line; `line_spacing` 1.3~1.5.
- Left-align paragraphs; centre only short titles. Keep a consistent margin (≥ 5% of the short side).

## 3. Text effects: outline, shadow, glow, gradient, texture fill

Back to front: glow → shadow → outline → fill. All optional.

```python
cv.gradient([(0, "#0b0620"), (1, "#3a1050")])
cv.text("NEON", (W * 0.27, H * 0.3), 170 * S, color=(1.5, 0.7, 1.3), anchor="mm", bold=True,
        glow=dict(radius=35 * S, color="#ff3cac"), outline=3 * S, outline_color="#fff0fa")       # neon (HDR fill)
cv.text("RETRO", (W * 0.73, H * 0.3), 170 * S, anchor="mm", bold=True,
        gradient=[(0, "#ffe66d"), (0.5, "#ff9f43"), (1, "#ff3c6e")], outline=4 * S, outline_color="#2a0030",
        shadow=dict(dx=8 * S, dy=10 * S, blur=0, color="#2bd2ff", opacity=1))                     # hard offset shadow
m = text_mask(H, W, "STONE", (W / 2, H * 0.72), 220 * S, bold=True, anchor="mm", font="georgia")
rock = lerp("#5d5750", "#b9b1a6", fbm2d(H, W, 40 * S, 5, seed=4) * 0.5 + 0.5)                   # texture as colour map
cv.paint("#000000", feather(m, 10 * S) * 0.6)
cv.paint(rock, m)
cv.soft_clip(0.85)
```
- Knockout / cut-out text: draw a panel, then paint the background colour back through the text mask, or
  `cv.paint(panel_color, panel_mask * (1 - text_mask(...)))`.
- Embossed letters: `sdf`-style shading from the mask: `bevel = 1 + 0.5 * (m - sample(m, X + 2*S, Y + 2*S))`.

## 4. Legible text over a busy picture

Text on a photo-like background needs **value contrast**, not just colour contrast. Three fixes, weakest to strongest:
shadow → gradient scrim → solid / blurred panel. Check with `python tools/look.py probe IMG U,V` (luma difference
between text and background ≥ 0.4) or `look.py values IMG` (the text must stay visible in the 4-level map).

```python
cv.gradient([(0, "#7fa6d6"), (0.6, "#e8c9a0"), (1, "#5a4a3a")])
cv.paint("#2d3a2a", clamp01(Y - ((0.6 - 0.15 * fbm1d(W, 0.2 * W, 5, seed=2)) * H)[None, :] + 0.5))
scrim = smoothstep(0.55, 1.0, V) * 0.75                          # dark gradient behind the bottom text
cv.paint("#000000", scrim)
cv.text("Into the Highlands", (0.06 * W, 0.86 * H), 80 * S, "#ffffff", anchor="lb", bold=True,
        shadow=dict(blur=8 * S, opacity=0.6))
cv.text("A field guide · 2026", (0.06 * W, 0.91 * H), 30 * S, "#e6e6e6", anchor="lt")
card = sdf.sd_box(X, Y, 0.82 * W, 0.18 * H, 0.12 * W, 0.06 * H, radius=0.06 * H)       # pill label
cv.paint("#ffffff", sdf.fill(card), opacity=0.92)
cv.text("NEW", box=(0.70 * W, 0.12 * H, 0.94 * W, 0.24 * H), padding=0.025 * H, color="#1b2a1b", bold=True)
```

## 5. Crisp vector shapes with SDFs: badges, buttons, icons, geometric art

`sdf.sd_*` gives the signed distance to a shape (negative inside). Combine like a vector editor, then render with
`sdf.fill` (1 px anti-aliased edge at any resolution), `sdf.outline`, `sdf.glow`, `sdf.bevel_light`.

```python
cv.gradient([(0, "#1b1f3a"), (1, "#0d0f1e")])
cx, cy = 0.3 * W, 0.5 * H
badge = sdf.sd_star(X, Y, cx, cy, 260 * S, 215 * S, n=16)                       # serrated seal
ring = sdf.sd_ring(X, Y, cx, cy, 175 * S, 8 * S)
cv.paint("#000000", sdf.glow(badge, 30 * S) * 0.5)                               # drop glow
cv.paint(lerp("#d9a441", "#f6d27a", 0.5) * sdf.bevel_light(badge, 14 * S)[..., None], sdf.fill(badge))
cv.paint("#7a5212", sdf.fill(ring))
cv.text("100%", (cx, cy), 120 * S, "#5a3a08", anchor="mm", bold=True)
# boolean ops: a rounded "app icon" with a cut-out circle and a smooth blob merge
icon = sdf.sd_box(X, Y, 0.7 * W, 0.5 * H, 200 * S, 200 * S, radius=60 * S)
hole = sdf.sd_circle(X, Y, 0.7 * W + 70 * S, 0.5 * H - 70 * S, 70 * S)
blob = sdf.smooth_union(sdf.sd_circle(X, Y, 0.7 * W - 60 * S, 0.5 * H + 40 * S, 70 * S),
                        sdf.sd_circle(X, Y, 0.7 * W + 30 * S, 0.5 * H + 80 * S, 55 * S), k=40 * S)
cv.linear((0.7 * W - 200 * S, 0.5 * H - 200 * S), (0.7 * W + 200 * S, 0.5 * H + 200 * S),
          [(0, "#2bd2ff"), (1, "#7a5cff")], mask=sdf.fill(sdf.subtract(icon, hole)))
cv.paint("#ffffff", sdf.fill(blob), opacity=0.9)
cv.paint("#ffffff", sdf.outline(icon, 3 * S), opacity=0.5)
```
- Primitives: `sd_circle, sd_ellipse, sd_box(radius, angle), sd_segment (capsule), sd_polygon, sd_star, sd_ring, sd_arc`.
- Ops: `union, intersect, subtract, smooth_union(k), offset(d, k)` (grow / shrink), `annular(d, t)` (hollow).
- Soft shapes: `sdf.fill(d, aa=20 * S)`; inner shadow: `sdf.fill(-sdf.offset(d, 6 * S), 8 * S)` inside the shape.

## 6. Curves, arrows, ribbons: Bézier + `line_mask`

```python
cv.gradient([(0, "#fbf7f0"), (1, "#efe6d6")])
path = bezier_points((0.1 * W, 0.7 * H), (0.35 * W, 0.1 * H), (0.6 * W, 0.95 * H), (0.88 * W, 0.35 * H), n=120)
cv.paint("#c9b89c", line_mask(H, W, path, 18 * S))                             # ribbon
cv.paint("#d0452f", line_mask(H, W, path, 6 * S))                              # line on top
tip, prev = np.array(path[-1]), np.array(path[-6])
d = (tip - prev) / np.linalg.norm(tip - prev)
n_ = np.array([-d[1], d[0]])
head = [tuple(tip + d * 30 * S), tuple(tip - d * 10 * S + n_ * 22 * S), tuple(tip - d * 10 * S - n_ * 22 * S)]
cv.paint("#d0452f", polygon_mask(H, W, head))                                  # arrow head
cv.text("route", (path[60][0], path[60][1] - 30 * S), 40 * S, "#5b4636", anchor="mb", font="georgia")  # label clear of the line
```
- Hand-drawn feel: `stroke(mask, path, r, r_end=..., jitter=0.05)` instead of `line_mask`.

## 7. Layout rules that make designs look intentional

1. **One focal element**, clearly the biggest / brightest / most saturated. Everything else supports it.
2. **Grid & margins**: pick a margin (6~8% of the short side) and align everything to it or to the centre line.
   Use `look.py guides` on the render to verify alignment numerically (edges on the same U / V value).
3. **Hierarchy by size, weight, colour — in that order**; ≤ 2 font families, ≤ 3 sizes.
4. **Group by proximity**: related lines close together (line_spacing 1.2~1.5), groups far apart (≥ 2x line height).
5. **Contrast**: text vs background luma difference ≥ 0.4 for body text (`look.py probe`); headlines can rely on size.
6. **Breathing room**: ≥ 30% of a poster should be calm (no text, low detail). Crowded = amateur.
7. **Thumbnail test**: `look.py check` shows a 96 px thumbnail — the title and focal element must still read.

| Format | Size | Notes |
|---|---|---|
| Poster | 1080x1350 (4:5) / 2480x3508 (A4 @300dpi) | title in top third, focal image middle, info bottom |
| Social square | 1080x1080 | keep text ≥ 50 px; centre-weighted |
| Story / phone wallpaper | 1080x1920 | keep top 12% and bottom 18% free of key content (UI overlays) |
| Video thumbnail | 1280x720 | ≤ 4 words, huge, outlined; one face / object; high saturation |
| Slide / banner | 1920x1080 | left-aligned title at 8% margin; one idea per slide |
