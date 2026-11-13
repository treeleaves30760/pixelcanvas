# Typography, layout and templates: posters, cards, menus, thumbnails, labelled pictures

Every glyph comes from a real font (fallback fonts fill in CJK and symbols) and every text draw is recorded for
`audit_text(cv)` and `--check`. Every snippet below runs as-is after this prelude:

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

Finished examples: `scenes/type_poster.py` (bilingual concert poster, arc-text seal, vertical Chinese title),
`scenes/type_specimen.py` (every effect, rich text, fallback), `scenes/neon_poster.py`. The fastest finished design
is `layout.template(...)` (section 9): content in, layout out.

## 1. Drawing text: `cv.text(...)`

`cv.text(text, (x, y), size, color, anchor=...)` paints anti-aliased, kerned text in pixels; the bbox lands in
`cv.last_bbox`. Anchor = horizontal `l/m/r` + vertical `t/m/b` (`"mm"` = centred). `\n` breaks lines; `max_width=`
wraps (spaces for Latin, between any characters for CJK, never a closing bracket at a line start).

```python
cv.gradient([(0, "#f4efe6"), (1, "#e6dccb")])
cv.text("Quiet Mornings", (W / 2, 0.40 * H), 110 * S, "#1f1a17", anchor="mm", font="serif", bold=True)
title = cv.last_bbox
cv.text("A STUDY IN SOFT LIGHT", (W / 2, title[3] + 30 * S), 26 * S, "#8a6f55", anchor="mt", tracking=0.3)
cv.text("清晨的光，安靜地落在桌上。 Café → ★", (W / 2, title[3] + 90 * S), 40 * S, "#3b302a", anchor="mt")
```
- **Fonts**: `font=None` picks a good default (CJK for CJK text); style words `"sans"`, `"serif"`, `"slab"`, `"mono"`,
  `"display"`, `"condensed"`, `"rounded"`, `"handwriting"`, `"script"` + `"bold"`, `"black"`, `"light"`, `"italic"`,
  or names and paths (`"Noto Serif TC"`). More: `python tools/fonts.py --install starter` / `noto-cjk`, `--sheet`.
- Tracking and leading follow the size (`tracking=`, `line_spacing=` override); title : subtitle : body ≈ 4 : 1.6 : 1.

## 2. Fit into a box, paragraphs, lists, tables

`box=(x0, y0, x1, y1)` auto-sizes the text to the box (minus `padding`), placed by `box_anchor`; with a box,
`max_width=True` is paragraph mode and `balance=True` headline mode (even lines, good break points).

```python
cv.gradient([(0, "#10202a"), (1, "#1f3b47")])
col = (0.08 * W, 0.12 * H, 0.46 * W, 0.88 * H)
cv.text("The quiet art of setting type by hand", box=(col[0], col[1], col[2], col[1] + 0.3 * H), box_anchor="lt",
        balance=True, color="#e8f1f2", font="serif bold")
body = ("Fog rolled in before dawn and stayed until noon. The ridge disappeared, then the pines, then the lake "
        "itself. 霧從黎明前開始，一直到中午才散。")
cv.text(body, (col[0], col[1] + 0.34 * H), 26 * S, "#b8cdd2", max_width=col[2] - col[0], align="justify",
        drop_cap=3)
cv.text("- Lists get hanging indents\n- Numbered ones too: start lines with 1. 2. 3.", (0.55 * W, 0.15 * H), 28 * S,
        "#e8f1f2", max_width=0.38 * W)
cv.text("Coffee\t2\t$4.50\nCroissant\t12\t$36.00", (0.55 * W, 0.45 * H), 28 * S, "#ffd27a",
        tabs=[(0.22 * W, "r"), (0.36 * W, ".")], tabular=True)
x0, y0, x1, y1 = text_bbox("Measure first", (0, 0), 40 * S)        # measure without drawing
size = fit_text_size("ONE LINE THAT MUST FIT", 0.38 * W, 0.1 * H)    # largest size that fits
cv.text("ONE LINE THAT MUST FIT", (0.74 * W, 0.8 * H), size, "#ffd27a", anchor="mm", bold=True)
```
- Body text: 45–75 Latin or 20–35 CJK characters per line. `align="left" | "center" | "right" | "justify"`;
  `list_style="bullet" | "number" | "dash" | "check"`; `tabs=[x, (x, "r"), (x, ".")]` with `\t`.

## 3. Rich text in one block

Light markup is detected automatically: `**bold**`, `*italic*`, `__underline__`, `~~strike~~`, `==highlight==`,
`{#ff5|colour}`, `{size=1.5|bigger}`, `{b u #e33|combined}`, `{font=georgia|face}`, `{sc|small caps}`,
`{hl=#aef|marker}`. Or pass a list of `(text, style)` spans. Different sizes share one baseline.

```python
cv.gradient([(0, "#faf7f0"), (1, "#efe8da")])
cv.text("**Rich** text: *italic*, {#c2412d|colour}, {size=1.6|bigger}, ==marker==, {sc|Small Caps}",
        (0.06 * W, 0.2 * H), 44 * S, "#1d1a17", font="serif")
cv.text([("SALE ", {"bold": True, "color": "#e63946"}), ("up to ", {}), ("70%", {"size": 1.8, "bold": True})],
        (0.06 * W, 0.45 * H), 50 * S, "#1d1a17")
cv.text("whole block", (0.06 * W, 0.7 * H), 50 * S, "#1d1a17", case="smallcaps", underline=True)
cv.text("rotated label", (0.75 * W, 0.6 * H), 50 * S, "#2a4d69", anchor="mm", rotate=15, bold=True)
```
Whole-block options: `case="upper" | "lower" | "title" | "smallcaps"`, `underline`, `strike`, `highlight="#ffe45c"`,
`rotate=deg` (counter-clockwise around `xy`), `letter_colors=[...]` (cycled) or stops (per-letter gradient).

## 4. Effects: one keyword each

`effect=` takes a name, a dict with options, or a list; effects compose with `outline`, `shadow`, `glow`,
`gradient`, and `texture=` (an image path, array or Canvas) fills the letters.

```python
cv.gradient([(0, "#211c2b"), (1, "#0e0c14")])
cv.text("GOLD", (0.25 * W, 0.25 * H), 170 * S, anchor="mm", font="serif bold", effect="gold")
cv.text("NEON", (0.75 * W, 0.25 * H), 170 * S, "#ff3cac", anchor="mm", bold=True, effect=dict(kind="neon", off=[2]))
cv.text("3D", (0.25 * W, 0.72 * H), 200 * S, "#ef476f", anchor="mm", font="display",
        effect=dict(kind="extrude", depth=24 * S, color="#7a1f3d"))
cv.text("WOW", (0.75 * W, 0.72 * H), 170 * S, "#f72585", anchor="mm", font="display", effect="sticker")
```
Effects: `long_shadow`, `extrude` (3D block letters; `depth`, `angle`, `color`), `emboss`, `letterpress`, metals
`gold` `silver` `chrome` `rose_gold` `foil` (`stops`, `streak`), `neon` (`off=[i]` unlit letters, `tube=True`),
`sticker`, `knockout` (letters cut out of a plate), `outline_only`, `stripes`, `glitch`, `chalk`, `stencil`.

## 5. Arcs, paths and vertical CJK

```python
cv.gradient([(0, "#f3eee4"), (1, "#e7dfcf")])
cx, cy, R = 0.25 * W, 0.5 * H, 0.32 * H
cv.paint("#1d3557", sdf.fill(sdf.sd_circle(X, Y, cx, cy, R * 1.15)))
text_on_arc(cv, "PIXELCANVAS TYPE FOUNDRY", (cx, cy), R, 0.12 * R, angle=-90, color="#f1faee", bold=True)
text_on_arc(cv, "EST. 2026", (cx, cy), R, 0.11 * R, angle=90, color="#e9c46a", bold=True)
cv.paint("#e63946", sdf.fill(sdf.sd_circle(X, Y, 0.62 * W, 0.2 * H, 0.14 * H)))          # round LIVE badge
text_on_arc(cv, "LIVE", (0.62 * W, 0.2 * H), 0.11 * H, color="#ffffff", bold=True, repeat=True)
cv.text("LIVE", (0.62 * W, 0.2 * H), 0.06 * H, "#ffffff", anchor="mm", bold=True)
pts = bezier_points((0.45 * W, 0.8 * H), (0.6 * W, 0.3 * H), (0.75 * W, 1.0 * H), (0.95 * W, 0.5 * H), n=100)
text_on_path(cv, "text follows any curve", pts, 44 * S, align="center", color="#5b4636")
cv.text("縦書きの日本語、「括弧」も回転。", (0.95 * W, 0.08 * H), 44 * S, "#222", anchor="ra", vertical=True,
        max_height=0.5 * H)
```
- `radius` runs through the middle of the letters: ring text at ~0.8 of the badge radius, the centre word at ~0.4;
  `repeat=True` fills the ring; `side="outside" | "inside" | "auto"`; `upright=True`; also `cv.text(..., arc=...)`.
- Vertical text runs top to bottom, columns right to left; brackets and dashes turn; `max_height` (or a `box`)
  wraps columns. A line down the right edge: `anchor="ra"` at `(L.x1, L.y0)` of `Layout(cv)`, `max_height=L.h`.

## 6. Legible text over a busy picture

`color="auto"` picks black or white ink for what lies under the text (with a soft shadow on busy backgrounds);
`plate="pill" | "box" | "scrim" | "blur"` or `dict(kind, color, opacity, pad, radius, border, shadow)` puts a plate
behind it. Text never leaves the canvas: it is moved or shrunk inside unless `clip_ok=True`.

```python
cv.gradient([(0, "#7fa6d6"), (0.6, "#e8c9a0"), (1, "#5a4a3a")])
cv.paint("#2d3a2a", clamp01(Y - ((0.6 - 0.15 * fbm1d(W, 0.2 * W, 5, seed=2)) * H)[None, :] + 0.5))
cv.text("Into the Highlands", (0.06 * W, 0.86 * H), 80 * S, "auto", anchor="lb", bold=True, plate="scrim")
cv.text("A field guide · 2026", (0.06 * W, 0.91 * H), 30 * S, "auto", anchor="lt")
cv.text("NEW", (0.85 * W, 0.15 * H), 40 * S, "#1b2a1b", anchor="mm", bold=True, plate="pill")
cv.text("frosted", (0.85 * W, 0.4 * H), 48 * S, "#ffffff", anchor="mm", plate=dict(kind="blur", color="#000000"))
```

## 7. Check your typography: `audit_text(cv)`

`audit_text(cv)` turns `cv.text_log` into issues with coordinates and a fix; `--check` prints them under TEXT.

```python
cv.gradient([(0, "#808080"), (1, "#707070")])
cv.text("grey on grey", (0.1 * W, 0.2 * H), 60 * S, "#8a8a8a")
cv.text("crowded", (0.1 * W, 0.22 * H), 60 * S, "#ffffff")
for issue in audit_text(cv):
    print(issue["kind"], issue["bbox"], issue["message"], "→", issue["fix"])
cv.text("fixed", (0.1 * W, 0.6 * H), 60 * S, "auto", plate="pill")
```
Issue kinds: `small`, `contrast` (4.5:1 body, 3:1 display, word by word), `clipped`, `margin`, `overlap`,
`missing_glyphs`, `adjusted`. Decorative text can pass `audit=False`.

## 8. Layout: boxes, grids and stacks

`Layout(cv, margin=0.06)` is the canvas minus a margin, as a `Box`: split it instead of guessing coordinates.
Boxes pass straight to `cv.text(box=...)`; `stack` sets a list of text items with consistent spacing.

```python
from pixelcanvas.layout import Layout, stack
cv.gradient([(0, "#151a36"), (1, "#0b0d1c")])
L = Layout(cv, margin=0.07)
strip, main = L.take("bottom", 0.12, gap=0.04)        # (bottom strip, the rest)
text_col, picture = main.golden("x")                  # 0.618 : 0.382
cv.paint("#ffb347", picture.inset(0.05).mask(cv, radius=24 * S), opacity=0.85)
boxes = stack(cv, text_col.inset(0, 0.1), [
    ("Summer series 2026", "kicker"),
    ("Night Market by the River", "title"),
    ("Street food, music and lanterns", "subtitle"),
    ("Every Friday from 7 pm at Pier 4. Free entry, all ages, bring friends.", "body"),
], theme="midnight", valign="middle")
for cell in strip.cols(3, gap=0.03):
    cv.text("DETAIL", box=cell.inset(0.2), color="#8d93b8", bold=True)
```
- Box helpers: `rows([2, 1], gap)`, `cols(3)`, `grid(r, c, gap)` / `cells(...)`, `split(0.4, "x")`, `golden()`,
  `thirds()`, `take("top", 0.2)` → (strip, rest), `inset(...)`, `sub(u0, v0, u1, v1)`, `fit(aspect)`,
  `mask(cv, radius)`. Numbers ≤ 1 are fractions of the box's shorter side, larger numbers are pixels.
- `stack` roles: `kicker`, `title`, `headline`, `subtitle`, `body`, `quote`, `caption`, `footer`, `meta` (or dicts
  of `cv.text` keywords); it shrinks together on overflow. Also `align_boxes`, `distribute`, `baseline_grid()`.

## 9. Templates: a finished design from content alone

```python
from pixelcanvas import layout
boxes = layout.template("poster", cv, title="Night Market", subtitle="Street food, music and lanterns",
                        kicker="Summer series", details=[("When", "Fri 7 pm"), ("Where", "Pier 4")], theme="midnight")
pic = boxes["image"]                                   # free area for a picture
cv.paint("#ffb347", sdf.fill(sdf.sd_circle(X, Y, pic.cx, pic.cy, 0.3 * pic.short)), opacity=0.9)
```
| Template | Content keywords |
|---|---|
| `poster` | title, subtitle, kicker, details=[(head, body)], footer · `frame=True`, `glow=True` |
| `flyer` | title, subtitle, date, details=[...], place, cta (full-width picture area, colour band) |
| `book_cover` · `magazine` | title, author, subtitle, series · title (masthead), headline, cover_lines=[(kicker, line)], issue |
| `business_card` · `quote_card` | name, role, company, details=[...] · quote, author, source |
| `thumbnail` | title (2–4 words), subtitle, badge · `layout="top"`: title across the top, `boxes["left"]` / `["right"]` / `["middle"]` for two subjects |
| `slide` · `certificate` | title, subtitle, kicker, details, footer, page · title, name, details, date, signer, org |
| `menu` | title, subtitle, sections=[(heading, [(item, price, note)])], footer |
| `badge` · `label` | title (top arc), subtitle (bottom arc), center · title, brand, subtitle, details, est |

Themes: `midnight`, `paper`, `neon`, `forest`, `sunset`, `mono`, `pastel`, `luxury`, `ocean`, or a dict
(`layout.theme("paper", accent="#0a7")`). Any theme key is also a template option, so colours change in the call:
`layout.template("poster", cv, title="Blue Hour", theme="ocean", title_color="#f0d078", accent="#d4a63a")`.

| Theme key | Colours / sets |
|---|---|
| `bg` | page gradient `[(0, top), (1, bottom)]` (with `background=True`) |
| `ink` | main text: item names, place line, the title unless `title` is set |
| `title` (`title_color=`) | the title (gradient + soft shadow by default) |
| `subtitle` · `soft` | subtitle line (default `soft`, which also colours secondary lines) |
| `kicker` · `accent` | kicker, rules, frame, date pill, menu headings (default `accent`) |
| `body` | details and body text (default `ink`) |
| `band` | flyer footer band (default `accent`); its text picks the readable of `panel` / `ink` |
| `muted` | captions, footer, menu leaders |
| `panel` | card / label surfaces, badge disc |
| `accent2` | second accent for your own decorations |
| `title_font` · `body_font` | font of titles · of everything else |

Lists with separators (`"Ana Reyes Trio · Kofi & The Lanterns · DJ Marlowe"`, also `•` `|` `/`) wrap between items,
on balanced lines. Menus spread their sections down the whole sheet. Draw the picture first and pass
`background=False` to set type over it. The result holds a Box per text item, `"image"` (the free area) and
`"layout"` (the page minus its margin).

## 10. Crisp vector shapes with SDFs: badges, buttons, icons

`sdf.sd_*` gives the signed distance to a shape (negative inside); combine like a vector editor, then render with
`sdf.fill` (anti-aliased at any resolution), `sdf.outline`, `sdf.glow`, `sdf.bevel_light`.

```python
cv.gradient([(0, "#1b1f3a"), (1, "#0d0f1e")])
cx, cy = 0.3 * W, 0.5 * H
badge = sdf.sd_star(X, Y, cx, cy, 260 * S, 215 * S, n=16)                       # serrated seal
cv.paint("#000000", sdf.glow(badge, 30 * S) * 0.5)
cv.paint(lerp("#d9a441", "#f6d27a", 0.5) * sdf.bevel_light(badge, 14 * S)[..., None], sdf.fill(badge))
cv.text("100%", (cx, cy), 120 * S, "#5a3a08", anchor="mm", bold=True)
icon = sdf.sd_box(X, Y, 0.7 * W, 0.5 * H, 200 * S, 200 * S, radius=60 * S)
hole = sdf.sd_circle(X, Y, 0.7 * W + 70 * S, 0.5 * H - 70 * S, 70 * S)
cv.linear((0.7 * W - 200 * S, 0.5 * H - 200 * S), (0.7 * W + 200 * S, 0.5 * H + 200 * S),
          [(0, "#2bd2ff"), (1, "#7a5cff")], mask=sdf.fill(sdf.subtract(icon, hole)))
```
Primitives `sd_circle, sd_ellipse, sd_box(radius, angle), sd_segment, sd_polygon, sd_star, sd_ring, sd_arc`; ops
`union, intersect, subtract, smooth_union(k), offset(d, k), annular(d, t)`; curves: `bezier_points` + `line_mask`.

## 11. Layout rules and formats

One focal element; one margin (`Layout(cv, 0.07)`); hierarchy by size, weight, colour; at most 2 font families and
3 sizes; about 30 % of a poster calm; the title still reads in the 96 px thumbnail of `look.py check`. Formats: poster 1080x1350 (`poster`, `flyer`), square 1080x1080 (`quote_card`, `badge`), story 1080x1920 with the
top 12 % and bottom 18 % free, thumbnail 1280x720 with at most 4 huge outlined words (`thumbnail`), slide 1920x1080
(`slide`), book cover 1200x1800, business card 1050x600.

A misspelt option, key, effect, template name or content field raises an error naming the closest valid one.
