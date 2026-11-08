# -*- coding: utf-8 -*-
"""Helpers that make images easier for a vision-language model (VLM) to *inspect*.

VLMs judge overall look well but are poor at reading exact positions, small details, and tonal structure.
These helpers turn an image into views that fix that:

* ``guides(img)``        coordinate grid with labelled fractions (+ rule of thirds) → read positions precisely
* ``zoom(img, box)``     crop a region (fractions or px) and enlarge it, with labelled edges → inspect details
* ``value_map(img, n)``  blurred, posterised luminance ("notan" / squint test) → is the composition readable?
* ``edge_map(img)``      gradient magnitude → are silhouettes crisp, are layers separated, where is detail?
* ``contact_sheet(...)`` labelled grid of many images → compare iterations, seeds, parameter sweeps, stages
* ``stage(cv, label)``   record intermediate canvases during a render (``tools/render.py --stages``)

All functions take a PIL image, a path, a Canvas, or an (H, W, 3) array, and return a PIL RGB image.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

_RES = getattr(Image, "Resampling", Image)

__all__ = ["to_pil", "label_font", "guides", "zoom", "value_map", "edge_map", "contact_sheet",
           "stage", "stages_enabled", "enable_stages", "take_stages"]


def to_pil(img):
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, str):
        return Image.open(img).convert("RGB")
    if hasattr(img, "to_pil"):
        return img.to_pil()
    a = np.asarray(img, np.float32)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, -1)
    if a.max() <= 1.0 + 1e-6:
        a = a * 255
    return Image.fromarray(np.clip(a + 0.5, 0, 255).astype(np.uint8), "RGB")


def label_font(size=14):
    from .text import load_font
    return load_font(None, size, bold=True)


def _tag(d, xy, text, font, anchor="la", fg=(255, 255, 255), bg=(0, 0, 0, 170)):
    """Text with a dark backing box so it stays legible on any image."""
    x0, y0, x1, y1 = d.textbbox(xy, text, font=font, anchor=anchor)
    d.rectangle([x0 - 3, y0 - 2, x1 + 3, y1 + 2], fill=bg)
    d.text(xy, text, font=font, fill=fg, anchor=anchor)


def _fit(im, max_w):
    if max_w and im.width > max_w:
        im = im.resize((max_w, max(1, round(im.height * max_w / im.width))), _RES.LANCZOS)
    return im


def guides(img, step=0.1, thirds=True, max_width=1280, labels=True, origin=(0.0, 0.0), span=(1.0, 1.0),
           color=(0, 255, 255), title=None):
    """Overlay a labelled coordinate grid.

    Lines every ``step`` (fraction of the full image); thirds in magenta. Labels are fractions of the *full*
    image (``origin`` / ``span`` let a zoomed crop keep the parent image's coordinates).
    """
    im = _fit(to_pil(img), max_width).convert("RGBA")
    W, H = im.size
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    font = label_font(max(11, min(16, W // 70)))
    ox, oy = origin
    sx, sy = span

    def ticks(o, s):
        k0 = int(np.ceil(o / step - 1e-9))
        k1 = int(np.floor((o + s) / step + 1e-9))
        return [k * step for k in range(k0, k1 + 1)]

    for u in ticks(ox, sx):
        x = (u - ox) / sx * (W - 1)
        major = abs(u * 2 - round(u * 2)) < 1e-6                # every 0.5 stronger
        d.line([(x, 0), (x, H)], fill=color + ((150 if major else 80),), width=1)
        if labels:
            _tag(d, (x + 2, 2), f"{u:.2f}".rstrip("0").rstrip(".") if u else "0", font)
    for v in ticks(oy, sy):
        y = (v - oy) / sy * (H - 1)
        major = abs(v * 2 - round(v * 2)) < 1e-6
        d.line([(0, y), (W, y)], fill=color + ((150 if major else 80),), width=1)
        if labels and v > oy + 1e-9:
            _tag(d, (2, y + 2), f"{v:.2f}".rstrip("0").rstrip("."), font)
    if thirds:
        for t in (1 / 3, 2 / 3):
            if ox <= t <= ox + sx:
                x = (t - ox) / sx * (W - 1)
                d.line([(x, 0), (x, H)], fill=(255, 0, 255, 140), width=1)
            if oy <= t <= oy + sy:
                y = (t - oy) / sy * (H - 1)
                d.line([(0, y), (W, y)], fill=(255, 0, 255, 140), width=1)
    if title:
        _tag(d, (W - 4, H - 4), title, font, anchor="rd")
    return Image.alpha_composite(im, ov).convert("RGB")


def _box_px(box, w, h):
    """(x0, y0, x1, y1) given as fractions (all ≤ 1) or pixels → integer pixels."""
    if all(0 <= float(v) <= 1.0 for v in box):
        x0, y0, x1, y1 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    else:
        x0, y0, x1, y1 = box
    x0, x1 = sorted((int(round(x0)), int(round(x1))))
    y0, y1 = sorted((int(round(y0)), int(round(y1))))
    return max(0, x0), max(0, y0), min(w, max(x1, x0 + 1)), min(h, max(y1, y0 + 1))


def zoom(img, box, out_width=1024, grid=True, step=None, nearest=None):
    """Crop ``box`` (fractions 0~1 or pixels) and enlarge to ``out_width``.

    Small crops are enlarged with nearest-neighbour so individual pixels (aliasing, banding, noise) stay visible.
    With ``grid`` the crop is overlaid with guides labelled in the *full image's* fractional coordinates.
    """
    im = to_pil(img)
    w, h = im.size
    x0, y0, x1, y1 = _box_px(box, w, h)
    crop = im.crop((x0, y0, x1, y1))
    k = out_width / crop.width
    nearest = (k >= 3) if nearest is None else nearest
    big = crop.resize((out_width, max(1, round(crop.height * k))), _RES.NEAREST if nearest else _RES.LANCZOS)
    if not grid:
        return big
    span = ((x1 - x0) / w, (y1 - y0) / h)
    if step is None:                                        # ~5-10 grid lines across the crop
        raw = span[0] / 6
        step = min((s for s in (0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2) if s >= raw), default=0.25)
    title = f"zoom x{k:.1f}  px ({x0},{y0})-({x1},{y1}) of {w}x{h}"
    return guides(big, step=step, thirds=True, max_width=None, origin=(x0 / w, y0 / h), span=span, title=title)


def _luma(im):
    a = np.asarray(im, np.float32) / 255
    return a @ np.array([0.299, 0.587, 0.114], np.float32)


def value_map(img, levels=4, blur=0.004, max_width=960):
    """Squint test: blur (fraction of width) then posterise luminance into ``levels`` grey steps.

    A strong picture still reads clearly here: a few big value masses, the focal point at the extreme value.
    Returns (image, fractions per level from dark to light).
    """
    im = _fit(to_pil(img), max_width)
    if blur:
        im = im.filter(ImageFilter.GaussianBlur(max(0.5, blur * im.width)))
    L = _luma(im)
    q = np.clip((L * levels).astype(int), 0, levels - 1)
    grey = (q / max(levels - 1, 1) * 255).astype(np.uint8)
    frac = np.bincount(q.ravel(), minlength=levels) / q.size
    return Image.fromarray(grey, "L").convert("RGB"), frac


def edge_map(img, max_width=960, gain=4.0):
    """Gradient magnitude of luminance (white = strong edge). Shows silhouettes, layer separation, texture."""
    im = _fit(to_pil(img), max_width)
    L = _luma(im)
    gx = np.zeros_like(L)
    gy = np.zeros_like(L)
    gx[:, 1:-1] = L[:, 2:] - L[:, :-2]
    gy[1:-1] = L[2:] - L[:-2]
    g = np.clip(np.hypot(gx, gy) * gain * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(g, "L").convert("RGB")


def contact_sheet(images, labels=None, cols=None, tile_width=480, pad=6, bg=(24, 24, 28), title=None):
    """Labelled grid of images (paths / PIL / arrays / Canvases). Labels go under each tile."""
    ims = [to_pil(i) for i in images]
    n = len(ims)
    if n == 0:
        raise ValueError("no images")
    wide = ims[0].width > 1.6 * ims[0].height
    cols = cols or (n if n <= 3 else 2 if n == 4 or (wide and n <= 6) else 4 if n in (7, 8) else 3)
    rows = -(-n // cols)
    tiles = [im.resize((tile_width, max(1, round(im.height * tile_width / im.width))), _RES.LANCZOS) for im in ims]
    th = max(t.height for t in tiles)
    font = label_font(15)
    lab_h = 24 if labels else 0
    top = 30 if title else 0
    sheet = Image.new("RGB", (cols * (tile_width + pad) + pad, top + rows * (th + lab_h + pad) + pad), bg)
    d = ImageDraw.Draw(sheet)
    if title:
        d.text((pad, 6), title, font=label_font(17), fill=(235, 235, 235))
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        x = pad + c * (tile_width + pad)
        y = top + pad + r * (th + lab_h + pad)
        sheet.paste(t, (x, y))
        if labels:
            d.text((x + 2, y + th + 3), str(labels[i])[:70], font=font, fill=(230, 230, 230))
    return sheet


# ------------------------------------------------------------------ render stages

_STAGES = {"on": False, "items": [], "pending": None, "width": 480, "target": None, "canvas": None}


def enable_stages(width=480, target_size=None):
    """Turn stage capture on. ``target_size=(w, h)``: the first Canvas created with that size is tracked, so
    ``stage(None, label)`` (what render.py's ``log`` does) knows which canvas to snapshot."""
    _STAGES.update(on=True, items=[], pending=None, width=int(width), target=target_size, canvas=None)


def _track(cv):
    """Called by Canvas.__init__: remember the main canvas of the render being captured."""
    if _STAGES["on"] and _STAGES["canvas"] is None and _STAGES["target"] == (cv.w, cv.h):
        _STAGES["canvas"] = cv


def stages_enabled():
    return _STAGES["on"]


def stage(cv, label):
    """Mark the start of a new drawing step called ``label``.

    When stage capture is on (``render.py --stages``), the canvas *as it is now* is saved under the previous
    step's label, so the sheet shows the result of each step. ``Scene.log()`` calls this automatically;
    low-level scenes can call ``stage(cv, "stars")`` themselves. No-op (and free) when capture is off.
    """
    if not _STAGES["on"]:
        return
    cv = cv if cv is not None else _STAGES["canvas"]
    prev = _STAGES["pending"]
    if prev is not None and cv is not None:
        im = cv.to_pil() if hasattr(cv, "to_pil") else to_pil(cv)
        w = _STAGES["width"]
        _STAGES["items"].append((prev, im.resize((w, max(1, round(im.height * w / im.width))), _RES.BILINEAR)))
    _STAGES["pending"] = label


def take_stages(cv=None, final_label="final"):
    """Close the last step (snapshotting ``cv``), return [(label, PIL image)], and turn capture off."""
    if cv is not None and _STAGES["on"]:
        if _STAGES["pending"] is None:
            _STAGES["pending"] = final_label
        stage(cv, None)
    items = _STAGES["items"]
    _STAGES.update(on=False, items=[], pending=None, canvas=None, target=None)
    return items
