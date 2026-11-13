#!/usr/bin/env python3
"""
Rebuild gallery/: one JPEG per picture (long side 1600 px) and the hero mosaic gallery/wall.jpg.

  python tools/build_gallery.py                    render every picture in PICTURES, then write the JPEGs and the wall
  python tools/build_gallery.py --only jazz_poster solid_reading_nook      only these (file names or scenes)
  python tools/build_gallery.py --wall-only        rebuild wall.jpg from the JPEGs already in gallery/
  python tools/build_gallery.py --fresh            re-render even when outputs/gallery_build/<scene>.png exists

Each picture is rendered at its scene's DEFAULT_SIZE with tools/render.py into outputs/gallery_build/ (git-ignored),
so a second run only converts. Edit the lists below to change the gallery.
"""
import argparse
import os
import subprocess
import sys
import time

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GALLERY = os.path.join(ROOT, "gallery")
BUILD = os.path.join(ROOT, "outputs", "gallery_build")

# (gallery file name, scene name or path relative to the repo, optional crop box in pixels of the full render)
PICTURES = [
    ("type_poster", "type_poster", None),
    ("farmers_market", "poster_farmers_market", None),
    ("jazz_poster", "illustration_jazz", None),
    ("cafe_menu", "layout_cafe_menu", None),
    ("ukiyoe_wave", "illustration_ukiyoe", None),
    ("animal_stickers", "illustration_stickers", None),
    ("meditation_app", "ui_meditation", None),
    ("landing_page", "ui_landing_hero", None),
    ("dashboard", "chart_subscriptions", None),
    ("architecture", "diagram_architecture", None),
    ("flowchart", "diagram_flowchart", None),
    ("sequence", "diagram_login_sequence", None),
    ("stone_lantern", "diagram_stone_lantern", None),
    ("bracket_drawing", "drafting_bracket", None),
    ("floor_plan", "drafting_apartment", None),
    ("app_icons", "vector_app_icons", None),
    ("readme_banner", "layout_readme_banner", None),
    ("pictograms", "pictogram_sheet", (0, 3745, 1920, 4600)),       # the ANIMALS and PEOPLE blocks of the full sheet
    ("art_styles", "style_showcase", None),
    ("reading_nook", "solid_reading_nook", None),
    ("earbuds", "solid_earbuds", None),
    ("studio_still_life", "solid_product", None),
    ("toy_diorama", "solid_toys", None),
    ("clay_campsite", "clay_campsite", None),
    ("clay_cast", "clay_cast", None),
    ("clay_zoo", "clay_zoo", None),
    ("campfire_lake", "elements_campfire", None),
    ("aurora_lake", "aurora_lake", None),
    ("sea_sunset", "sea_sunset", None),
    ("harbour_light", "gallery/src/harbour_light.py", None),        # the README code sample
]

# The hero mosaic: rows of gallery file names, each row scaled to fill the width (justified rows).
WALL = [
    ["type_poster", "clay_campsite", "architecture", "jazz_poster"],
    ["meditation_app", "bracket_drawing", "reading_nook", "dashboard", "ukiyoe_wave"],
    ["farmers_market", "art_styles", "pictograms", "clay_cast"],
]
WALL_WIDTH = 2400
WALL_GAP = 10
WALL_BG = (16, 18, 24)
WALL_MAX_BYTES = 900_000

LONG_SIDE = 1600
QUALITY = 88


def render(scene, fresh=False):
    """Render one scene at its DEFAULT_SIZE into outputs/gallery_build/ and return the PNG path."""
    os.makedirs(BUILD, exist_ok=True)
    stem = os.path.splitext(os.path.basename(scene))[0]
    out = os.path.join(BUILD, stem + ".png")
    if os.path.exists(out) and not fresh:
        return out
    target = os.path.join(ROOT, scene) if scene.endswith(".py") else scene
    t = time.time()
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "render.py"), target, "--out", out, "--outdir", BUILD],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"render failed for {scene}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    print(f"  rendered {scene:28s} {time.time() - t:6.1f} s", flush=True)
    return out


def to_jpeg(png, dest, crop=None):
    """Downscale to LONG_SIDE (never up) and save as a progressive JPEG."""
    im = Image.open(png).convert("RGB")
    if crop:
        im = im.crop(crop)
    scale = LONG_SIDE / max(im.size)
    if scale < 1:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    im.save(dest, "JPEG", quality=QUALITY, optimize=True, progressive=True)
    return im.size


def build_wall():
    """Justified rows of gallery JPEGs -> gallery/wall.jpg, under WALL_MAX_BYTES."""
    rows = []
    for names in WALL:
        ims = [Image.open(os.path.join(GALLERY, n + ".jpg")).convert("RGB") for n in names]
        aspect = sum(im.width / im.height for im in ims)
        h = round((WALL_WIDTH - WALL_GAP * (len(ims) + 1)) / aspect)
        rows.append((ims, h))
    height = sum(h for _, h in rows) + WALL_GAP * (len(rows) + 1)
    wall = Image.new("RGB", (WALL_WIDTH, height), WALL_BG)
    y = WALL_GAP
    for ims, h in rows:
        x = WALL_GAP
        for i, im in enumerate(ims):
            w = round(im.width * h / im.height)
            if i == len(ims) - 1:
                w = WALL_WIDTH - WALL_GAP - x                     # absorb rounding in the last tile
            wall.paste(im.resize((w, h), Image.LANCZOS), (x, y))
            x += w + WALL_GAP
        y += h + WALL_GAP
    dest = os.path.join(GALLERY, "wall.jpg")
    for q in (86, 82, 78, 74, 70):
        wall.save(dest, "JPEG", quality=q, optimize=True, progressive=True)
        if os.path.getsize(dest) <= WALL_MAX_BYTES:
            break
    print(f"  wall.jpg {wall.width}x{wall.height} q{q} {os.path.getsize(dest) / 1e3:.0f} KB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", help="gallery file names or scene names to rebuild")
    ap.add_argument("--fresh", action="store_true", help="re-render even when a render is cached")
    ap.add_argument("--wall-only", action="store_true", help="only rebuild wall.jpg")
    a = ap.parse_args()
    os.makedirs(GALLERY, exist_ok=True)
    if not a.wall_only:
        for name, scene, crop in PICTURES:
            if a.only and name not in a.only and scene not in a.only:
                continue
            size = to_jpeg(render(scene, a.fresh), os.path.join(GALLERY, name + ".jpg"), crop)
            kb = os.path.getsize(os.path.join(GALLERY, name + ".jpg")) / 1e3
            print(f"  {name + '.jpg':24s} {size[0]}x{size[1]}  {kb:5.0f} KB", flush=True)
    build_wall()
    total = sum(os.path.getsize(os.path.join(GALLERY, f)) for f in os.listdir(GALLERY)
                if os.path.isfile(os.path.join(GALLERY, f)))
    print(f"gallery/: {total / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
