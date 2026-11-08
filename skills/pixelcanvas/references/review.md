# Review checklist + symptom → fix

After each render, go through every item and write down the **2~4 worst problems**; the next round fixes only those.

## How to "look"

| What you can do | How |
|---|---|
| Can read images (VLM) | `render.py <scene> --scale 0.5 --check` → open `_preview.jpg` **and** `_check.jpg`; then targeted views below |
| Can't read images | `python tools/look.py report <png>` + `critique`; for detail `look.py crop` a region, then `report` that crop |
| Compare changes | `look.py sheet v1.png v2.png v3.png` / `compare A B` (side by side); `diff A B` (numbers + heat map) |

### Protocol for vision models

Vision models judge "does it look good" well, but misjudge **positions**, **small details** and **tonal structure**,
and tend to describe what they *intended* instead of what is there. Counter each one with a tool:

1. **First impression, then the numbers.** Look at the preview cold and write one sentence: what is the focal point,
   what feels wrong. Then run `look.py critique` — it catches what eyes forgive (clipping, flat range, centred horizon).
2. **Positions → `look.py guides IMG`.** Read coordinates off the labelled grid instead of guessing
   ("sun centre ≈ (0.62, 0.31); horizon at V 0.60; title spans U 0.08~0.92"). Compare with the plan in the docstring.
   Fix by changing the numbers, not by eyeballing again.
3. **Details → `look.py zoom IMG u0 v0 u1 v1`.** Zoom (≥ 3x, nearest-neighbour) into the focal point, every
   silhouette edge, text edges, and anywhere that "looks fine" at preview size. Previews hide aliasing, banding,
   seams, tofu boxes in text, stray pixels and moiré.
4. **Values → `look.py values IMG`.** In the 4-level map the picture should still read: a few big masses, the focal
   point at the extreme value, layers separated. If two layers merge into one grey, change their value, not hue.
5. **Edges → `look.py edges IMG`.** Silhouettes should be continuous lines; texture where you want detail, calm
   where you want rest. Hard edges attract the eye — make sure the strongest ones surround the focal point.
6. **Where did it break → `render.py --stages`.** The stage sheet shows the canvas after each step; find the first
   step where it goes wrong instead of guessing which parameter did it.
7. **Choosing a value → `render.py --sweep NAME v1 v2 v3`** (or `--seeds 1 2 3 4`): look at one labelled sheet and
   pick; cheaper and more reliable than serial tweaking.
8. **Colours → `look.py probe IMG U,V ...`.** Exact hex / luma / hue where you need certainty
   (text contrast, "is the snow shadow actually blue?", sky vs mountain luma difference).

Be honest in the write-up: list what is actually wrong, with coordinates ("seam at (0.41, 0.66)"), not what you
hoped to see. If you can't tell, zoom.

### Reading the numbers from `look.py report`

- **Overexposed (過曝) (>0.98)**: > 3% is too bright (usually the sun glow is too big or the sky too bright); > 0.3% but concentrated around the light source is normal.
- **Crushed black (死黑) (<0.02)**: > 5% means the foreground is too dark and loses detail; night scenes can go up to 15%, but make sure it's silhouettes rather than a solid black mass.
- **Overall contrast (std)** depends on the subject (see measured values from the gallery): daytime / dusk landscapes 0.18~0.28 (mountains_sunset 0.24,
  sea_sunset 0.21); night scenes and still lifes with large single-color backgrounds can drop to 0.06~0.14 (night_sky 0.09, still_life 0.06, city_dusk 0.14),
  in which case look at the **focal cell**'s contrast and p99 instead — the focal point needs a clear highlight. Daytime landscape < 0.12 → too gray; any subject > 0.32 → too harsh.
- **Centroid of the brightest 0.5% of pixels**: should land near the light source position you set. Widely spread → too many bright things (clouds, excessive glints);
  the exception is highlights that are naturally spread over the whole frame (starry sky, city windows).
- **3x3 grid (九宮格) brightness**: atmospheric perspective → the distance (usually the middle row) should be brighter and lower-contrast than the foreground (bottom row);
  brightness difference between the three rows < 0.05 → layers don't separate. The focal cell should be the highest-contrast or brightest cell.
- **Texture (質感) column**: flat (`平`) = almost no detail (plastic look / pure gradient); rough (`粗糙`) in the sky → noise scale too small or too much grain.
  The sky should be `平` or fine (`細緻`); foreground rock / forest should be `細緻` or `粗糙`.
- **Dominant colors**: top 2 together > 70% → monotonous image; check that the expected accent colors are present (sun, window lights, red ball).
- **Hue**: 0° red, 30° orange, 60° yellow, 120° green, 180° cyan, 240° blue, 280° violet, 320° pink-violet / magenta; s = saturation.
  "Pink-violet sunset glow" ≈ 300°~330° near the horizon, s 0.2~0.4; "warm orange sunset" ≈ 20°~40°; s < 0.1 is almost gray.
- **Vertical profile**: brightness along V. The sky usually brightens toward the horizon; ground / foreground should be noticeably darker than the sky; a sudden drop = mountains or forest begin.
- **Strongest horizontal boundaries per column**: each column lists its two strongest light/dark transitions (V position, bright above / dark below (上亮下暗) or dark above / bright below (上暗下亮), Δ).
  They should match the ridge / horizon heights you set; Δ < 0.05 → that boundary is too weak and the two layers will blur together.
- **Local checks**: `look.py crop image x0 y0 x1 y1 -o tmp.jpg` then `report tmp.jpg`, e.g. to look only at the snowfield or one mountain layer;
  the 3x3 grid's "texture" mixes trees and snow together, so crop out pure snow for an accurate reading.
- **ASCII brightness map**: confirm the big structure: horizon height, mountain outlines, light source position, whether light/dark zones match the composition.

## Checklist

1. **Exposure**: no large blown-out whites (sun glow, sky) or crushed blacks (foreground). A ">1 fraction" in `cv.stats()` > 3% usually means too bright.
   The sun should look like "a bright point + glow"; the surrounding sky should be darker than the sun.
2. **Atmospheric perspective**: the farther, the paler, closer to fog color, lower contrast; foreground high contrast, darker colors. Layers clearly separated.
3. **Consistent lighting**: lit sides of mountains, clouds, trees and objects all face the same way; shadows point the opposite way; low-angle light → silhouettes get rim light.
4. **Texture scale**: noise scale too small → fur / noisy look; too large → plastic look. Forest / clouds use plain / billow, rock uses ridged.
5. **Edges**: silhouettes have 1px anti-aliasing; cloud edges fluffy (wider threshold smoothstep); objects have contact shadows with the ground.
6. **Water**: reflections mirrored about the eye-level line; near-total reflection in the distance, deep water color showing up close; glints only directly below the light.
7. **Repetition**: adjacent objects don't share the same noise / seed; randomly scattered objects vary in size and spacing.
8. **Composition**: focal point near the thirds lines; foreground frames the image without stealing attention; horizon not dead center (unless deliberately symmetric).
9. **Resolution consistency**: draft (0.5) and full size look the same (all pixel constants multiplied by S / `sc.px()`).
10. **Text & design** (posters, cards, labels): every word spelled exactly as requested; no empty boxes (missing
    glyphs); text-to-background luma difference ≥ 0.4 (`look.py probe`); nothing touches the frame edge or another
    element; edges aligned to the same margin / centre line (`look.py guides`); one clear focal element; still
    readable in the `look.py check` thumbnail.

## Symptom → fix

| Symptom | Likely cause | Fix |
|---|---|---|
| Blown out overall, sun is a big white blob | Glow strength too high, sigma too large, sun disc too big | Lower glow strength (0.3 → 0.15), shrink the large glow's sigma; `finish(knee=0.75)`; darken the sky gradient near the sun |
| Image gray and hazy | Too much fog, foreground not dark enough | Near-layer fog → 0~0.1; darker foreground colors; `contrast(0.12)`; `levels(0.03, 1, 1)` |
| Layers don't separate | Adjacent layers' colors / fog amounts too similar | Spread fog values apart (e.g. 0.8 / 0.55 / 0.3 / 0.1); far layers toward fog color, near layers toward dark; add `mist` between layers |
| Clouds look like marble / smoke | Warp too strong, threshold too narrow | Lower warp (0.35 → 0.2); widen the threshold range; aspect < 1 to stretch clouds horizontally |
| Clouds look like cotton candy / too blurry | Softness too high, missing small scales | Lower softness; add one more high-frequency octave; add fine noise on lit sides |
| Foreground too flat, plastic look | Missing mid / small-scale texture | Add relief / high-frequency fBm to modulate brightness (±10%); add small objects like trees, grass, rocks |
| Fur-like, noisy look | Noise scale too small or too many octaves | scale ×2; 1~2 fewer octaves; lower grain to 0.01 |
| Jagged silhouettes | Used a hard threshold `Y > ridge` | Switch to `clamp01(Y - ridge + 0.5)`; for shapes use `shape_mask(aa=3)` |
| Objects look like they float | Cast shadow detached from contact point, missing AO | Center the shadow on the contact point; add a narrow, dark contact shadow |
| Contradictory light direction | Each object assumes its own light vector | Define one `L` / `light_dir` at the top of the file and share it across all objects |
| Water looks like a plastic sheet | Reflection too sharp, no ripples | Add horizontally stretched ripple displacement; mix in deep water color up close; bright far, dark near |
| Glints too dense, look like noise | Glint threshold too low, area too wide | Only in a vertical band directly below the light; raise the smoothstep threshold; narrow the band with distance |
| Repetitive, tiled look | Same noise / seed used in multiple places | Different seed per layer; stagger offsets; modulate density with low-frequency variation |
| Moiré (shimmering stripes in the distance) | Perspective texture frequency too high far away | Fade texture amplitude with `smoothstep` in the distance; or use fewer octaves |
| Draft and full size look different | Pixel constants not multiplied by S | Change every px constant to `x * S` / `sc.px(x)`; count-type parameters (number of stars) should not scale with resolution |
| Muddy or off colors | Complementary colors mixing, saturation too high | Cool shadows / warm lights but don't mix them; desaturate large areas; `split_tone` amount ≤ 0.1 |
| Empty boxes instead of letters ("tofu") | Font lacks those glyphs (e.g. CJK in a Latin font) | Omit `font=` (auto CJK) or pick a CJK font: `python tools/fonts.py --sheet` |
| Text hard to read on the picture | Too little value contrast behind it | Add a gradient scrim / panel behind it, outline or shadow; check with `look.py values` |
| Text too big / overflowing / crowded | Size guessed by eye | `box=(...)` auto-fit with `padding`; measure with `text_bbox`; leave ≥ 30% calm space |
| Jagged shapes / text | Hard thresholds or no supersampling | `sdf.fill(d)` for shapes, `cv.text` / `text_mask` (supersampled) for text, never `d < 0` |
| Design looks random / amateur | No alignment grid, too many sizes / fonts | One margin, align to it; ≤ 2 fonts, ≤ 3 sizes; see typography.md §7 |
