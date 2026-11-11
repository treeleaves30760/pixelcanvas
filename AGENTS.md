# AGENTS.md — for AI coding assistants (Codex, Cursor, Gemini CLI, Copilot, Aider, …)

This project lets **models that cannot generate images** draw good-looking ones: numpy + Pillow compute every pixel
from noise, masks, signed distance fields, rasterised text and blending, and a **render → look → fix** loop
improves the image. There are no image assets.

**Install as a skill:** `npx skills add treeleaves30760/pixelcanvas` (the skill's `scripts/bootstrap.py` fetches the renderer on first use).

**The full workflow is in [`skills/pixelcanvas/SKILL.md`](skills/pixelcanvas/SKILL.md). Read it before drawing.**
References: [API](skills/pixelcanvas/references/api.md) · [techniques & palettes](skills/pixelcanvas/references/techniques.md) ·
[typography, shapes, layout](skills/pixelcanvas/references/typography.md) · [how to look + checklist](skills/pixelcanvas/references/review.md).

## One-minute version

```bash
pip install -r requirements.txt
python tools/new_scene.py my_scene [--poster]            # create scenes/my_scene.py (or copy the closest example)
python tools/render.py my_scene --scale 0.5 --check      # draft + diagnostic sheet (guides | values | edges | thumbnail)
python tools/look.py zoom outputs/my_scene_x0.5.png 0.4 0.3 0.6 0.5    # inspect a region
python tools/render.py my_scene --stages                 # canvas after every step → find what broke
python tools/render.py my_scene --sweep NAME 0.3 0.5 0.7 # compare values on one sheet
python tools/look.py report outputs/my_scene_x0.5.png    # no vision? read the image as text
python tools/render.py my_scene                          # full size when satisfied; copy to gallery/
python -m pytest -q                                      # after changing the package or docs (~5 s)
```

1. Plan first (format, focal point, horizon, light, far / middle / near or text hierarchy, palette, exact text) and
   write it as the scene file's docstring.
2. A picture takes 4~7 rounds. Each round, write down the 2~4 worst problems **with coordinates**, change only
   those, and keep the seed fixed.
3. **Never deliver without looking.** Use `--check`, `guides`, `zoom` (vision) or `report` + `critique` (no vision).

## Conventions

- **Coordinates:** x goes right and y goes down (`arr[y, x]`). Masks are float32 (H, W) in 0~1. Positions are
  fractions. Pixel constants are based on 1080p and multiplied by `S = H / 1080`.
- **Drawing order:** sky → far → near → water (reflections sample the current image) → foreground → light →
  text → post-processing. Lights may exceed 1; finish with `soft_clip`.
- **Scene constants:** keep tunables as UPPER_CASE module constants so `--set` / `--sweep` can reach them. Call
  `log("step")` at each drawing step so `--stages` can snapshot it.
- **Performance:** vectorise everything; no per-pixel Python loops.
- **Where new code goes:** general recipes go in `pixelcanvas/scenery.py` (landscapes), `pixelcanvas/text.py` or
  `pixelcanvas/sdf.py` (design); one-off code stays in the scene file. Add new techniques to
  `references/techniques.md` or `typography.md`; their ```python blocks are executed by `tests/test_skill_docs.py`.
- **Outputs:** `outputs/` is not committed; finished pictures go in `gallery/`.
