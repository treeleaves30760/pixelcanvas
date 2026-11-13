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
* ``critique(img)``      structured warnings (rule, severity, where, value, threshold, fix) from cheap image checks
* ``image_stats(img)``   size, luminance percentiles, clipping as a dict

All functions take a PIL image, a path, a Canvas, or an (H, W, 3) array, and return a PIL RGB image.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

_RES = getattr(Image, "Resampling", Image)

__all__ = ["to_pil", "label_font", "guides", "zoom", "value_map", "edge_map", "contact_sheet",
           "stage", "stages_enabled", "enable_stages", "take_stages",
           "critique", "format_warning", "image_stats", "trace_event"]


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
    return Image.fromarray(np.clip(a + 0.5, 0, 255).astype(np.uint8))


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
    return Image.fromarray(grey).convert("RGB"), frac


def edge_map(img, max_width=960, gain=4.0):
    """Gradient magnitude of luminance (white = strong edge). Shows silhouettes, layer separation, texture."""
    im = _fit(to_pil(img), max_width)
    L = _luma(im)
    gx = np.zeros_like(L)
    gy = np.zeros_like(L)
    gx[:, 1:-1] = L[:, 2:] - L[:, :-2]
    gy[1:-1] = L[2:] - L[:-2]
    g = np.clip(np.hypot(gx, gy) * gain * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(g).convert("RGB")


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


# ------------------------------------------------------------------ numbers: stats, critique, trace

_LUMA = np.array([0.299, 0.587, 0.114], np.float32)
SEVERITIES = ("fail", "warn", "info")


def _rgb01(img, max_width=None):
    """Any image-like -> float32 (H, W, 3) in 0..1 (optionally box-downsampled to ``max_width``)."""
    pil = to_pil(img)
    if max_width and pil.width > max_width:
        pil = pil.resize((max_width, max(1, round(pil.height * max_width / pil.width))), _RES.BOX)
    return np.asarray(pil, np.float32) / 255


def image_stats(img):
    """Size, mean, luminance percentiles and clipping of an image as a plain dict (numbers, no prose)."""
    im = _rgb01(img)
    lum = im @ _LUMA
    return {
        "width": int(im.shape[1]), "height": int(im.shape[0]),
        "mean": round(float(im.mean()), 4),
        "luma_p1": round(float(np.percentile(lum, 1)), 4),
        "luma_p50": round(float(np.percentile(lum, 50)), 4),
        "luma_p99": round(float(np.percentile(lum, 99)), 4),
        "overexposed_pct": round(float(np.mean(lum > 0.98) * 100), 3),
        "crushed_pct": round(float(np.mean(lum < 0.02) * 100), 3),
        "mean_rgb": [round(float(v), 4) for v in im.reshape(-1, 3).mean(0)],
    }


def _warn(rule, severity, message, where, value, threshold, fix):
    def r(v):
        if isinstance(v, (list, tuple)):
            return [r(t) for t in v]
        if isinstance(v, (float, np.floating)):
            return round(float(v), 4)
        if isinstance(v, (np.integer,)):
            return int(v)
        return v
    if isinstance(where, (list, tuple)) and len(where) == 4 and (where[2] - where[0]) * (where[3] - where[1]) > 0.9:
        where = None                                       # a box covering the whole image says nothing
    return {"rule": rule, "severity": severity, "message": message, "where": r(where),
            "value": r(value), "threshold": r(threshold), "fix": fix}


def format_warning(w):
    """One human-readable line for a warning dict."""
    at = ""
    if w.get("where"):
        wh = w["where"]
        at = " @ (" + ", ".join(f"{v:.2f}" for v in wh) + ")" if len(wh) == 2 else \
            " @ [" + ", ".join(f"{v:.2f}" for v in wh) + "]"
    return f"[{w['severity']}] {w['rule']}{at}: {w['message']}  Fix: {w['fix']}"


def _bbox_frac(mask):
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    if not len(xs):
        return None
    return [xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h]


def _components(mask):
    """4-connected components of a small boolean grid -> list of (cells (N,2) array [y, x])."""
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    out = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        cells = []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        out.append(np.array(cells))
    return out


def _run_mask(mask, n, axis):
    """True where ``mask`` is True along a run of at least ``n`` cells in ``axis`` (centre-aligned)."""
    m = np.moveaxis(mask.astype(np.int32), axis, 0)
    c = np.cumsum(np.vstack([np.zeros((1,) + m.shape[1:], np.int32), m]), axis=0)
    ok = (c[n:] - c[:-n]) == n                              # window starting at i covers i..i+n-1
    out = np.zeros_like(m, bool)
    for k in range(n):
        out[k:k + ok.shape[0]] |= ok
    return np.moveaxis(out & m.astype(bool), 0, axis)


def _banding(L8):
    """Fraction of sampled scan lines covered by wide, monotone 1-level steps (visible gradient banding)."""
    h, w = L8.shape
    cover, lines = 0.0, 0
    where = None
    for axis in (0, 1):
        n = L8.shape[1 - axis]
        idx = np.linspace(0, L8.shape[axis] - 1, 48).astype(int)
        for i in idx:
            line = L8[i] if axis == 0 else L8[:, i]
            d = np.diff(line)
            st = np.nonzero(d)[0]
            lines += 1
            if len(st) < 4:
                continue
            runs = np.diff(st)
            sgn = np.sign(d[st])
            unit = np.abs(d[st]) == 1
            good = (runs >= 6) & unit[:-1] & unit[1:] & (sgn[:-1] == sgn[1:])
            # need 3+ consecutive good runs (a staircase), not a single step
            g3 = good[:-2] & good[1:-1] & good[2:] if len(good) >= 3 else np.zeros(0, bool)
            if g3.any():
                keep = np.zeros(len(good), bool)
                for k in np.nonzero(g3)[0]:
                    keep[k:k + 3] = True
                cover += runs[keep].sum() / n
                if where is None and axis == 0:
                    where = [st[np.nonzero(keep)[0][0]] / w, i / h]
    return cover / max(lines, 1), where


def critique(img):
    """Cheap, objective checks on an image -> list of warning dicts, most severe first.

    Each warning is ``{rule, severity ("info"|"warn"|"fail"), message, where, value, threshold, fix}``;
    ``where`` is ``[u, v]``, a box ``[u0, v0, u1, v1]`` (fractions of the image) or ``None`` for the whole image.
    """
    full = _rgb01(img)
    im = _rgb01(img, 960)
    h, w = im.shape[:2]
    lum = im @ _LUMA
    out = []
    add = lambda *a: out.append(_warn(*a))                  # noqa: E731

    # ---- exposure & tonal range
    over, crush = float(np.mean(lum > 0.98)), float(np.mean(lum < 0.02))
    if over > 0.03:
        fix = "lower glow / light strength, use soft_clip or finish(knee=0.75)"
        if over > 0.4:
            fix = "check colours: tuples above 1 are HDR (use '#rrggbb' or 0-1 floats); then " + fix
        add("blown_highlights", "fail", f"{over * 100:.1f}% of pixels are blown out (luma > 0.98).",
            _bbox_frac(lum > 0.98), over, 0.03, fix)
    if crush > 0.08:
        add("crushed_blacks", "info" if float(np.median(lum)) < 0.12 else "warn", f"{crush * 100:.1f}% of pixels are crushed black (luma < 0.02).",
            _bbox_frac(lum < 0.02), crush, 0.08, "lift shadows (levels gamma 1.1) or add rim / bounce light")
    p5, p95, std = float(np.percentile(lum, 5)), float(np.percentile(lum, 95)), float(lum.std())
    accent = float(np.percentile(lum, 99.9) - np.percentile(lum, 50))   # a small specular still counts as an accent
    if p95 - p5 < 0.25 and accent < 0.3:
        add("narrow_tonal_range", "fail", f"Narrow tonal range (p5..p95 = {p5:.2f}..{p95:.2f}) and no bright accent: looks flat.",
            None, p95 - p5, 0.25, "darken the darkest mass and brighten the focal point")
    elif p95 - p5 < 0.25:
        add("narrow_tonal_range", "info", f"Narrow tonal range (p5..p95 = {p5:.2f}..{p95:.2f}) with a bright accent: reads as low-key / high-key.",
            None, p95 - p5, 0.25, "widen it with a darker dark and a lighter light if that was not the intent")
    elif std > 0.33:
        add("high_contrast", "info", f"Very high contrast (luma std {std:.2f}).", None, std, 0.33,
            "soften the extremes if it was not intended")

    # ---- horizon & focal point
    rows = np.convolve(lum.mean(1), np.ones(5) / 5, mode="same")
    dif = np.abs(np.diff(rows[3:-3]))
    if dif.size and dif.max() > 0.04:
        yb = (np.argmax(dif) + 3.5) / h
        if 0.46 < yb < 0.54:
            add("horizon_centered", "warn", f"Strongest horizontal boundary at V={yb:.2f}, dead centre.", [0.5, yb], yb, [0.46, 0.54],
                "move the horizon towards V=0.33 or V=0.67 (set_horizon)")
    ys, xs = np.nonzero(lum >= np.percentile(lum, 99.5))
    cx, cy = xs.mean() / w, ys.mean() / h
    spread = max(xs.std() / w, ys.std() / h)
    if accent < 0.05:
        pass                                                   # flat image: no meaningful highlight
    elif spread > 0.22:
        add("scattered_highlights", "warn", f"Brightest pixels are scattered (spread {spread:.2f}): no single focal highlight.",
            None, spread, 0.22, "strengthen one bright area (fine for stars or city lights)")
    else:
        if abs(cx - 0.5) < 0.04 and abs(cy - 0.5) < 0.04:
            add("focal_centered", "info", f"Focal highlight at the exact centre ({cx:.2f}, {cy:.2f}).", [cx, cy],
                [cx, cy], [0.5, 0.5], "move it to a third line (U or V = 0.33 / 0.67)")
        if min(cx, cy, 1 - cx, 1 - cy) < 0.05:
            add("focal_at_edge", "warn", f"Brightest area sits at the frame edge ({cx:.2f}, {cy:.2f}).", [cx, cy],
                min(cx, cy, 1 - cx, 1 - cy), 0.05, "move the bright area inward or dim it")

    # ---- colour
    mx, mn = im.max(-1), im.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    hot = float(np.mean((sat > 0.85) & (mx > 0.5)))
    if hot > 0.25:
        add("oversaturated", "warn", f"{hot * 100:.0f}% of the image is near-max saturation.", _bbox_frac((sat > 0.85) & (mx > 0.5)),
            hot, 0.25, "keep strong colour for the focal point and desaturate large areas")
    if sat.mean() < 0.06:
        add("monochrome", "info", f"Almost monochrome (mean saturation {sat.mean():.3f}).", None, float(sat.mean()), 0.06,
            "add one accent hue at the focal point if colour was intended")
    lw, rw = float(lum[:, : w // 2].mean()), float(lum[:, w // 2:].mean())
    if abs(lw - rw) > 0.25:
        add("unbalanced_lr", "info", f"Left/right brightness is unbalanced ({lw:.2f} vs {rw:.2f}).", None, abs(lw - rw), 0.25,
            "add a counterweight (a darker or lighter element) on the lighter / darker side")
    flat_cells = []
    for r in range(3):
        for c in range(3):
            blk = lum[r * h // 3:(r + 1) * h // 3, c * w // 3:(c + 1) * w // 3]
            if np.abs(np.diff(blk, axis=0)).mean() + np.abs(np.diff(blk, axis=1)).mean() < 0.004:
                flat_cells.append((r, c))
    if len(flat_cells) >= 6:
        add("flat_cells", "info", f"{len(flat_cells)}/9 grid cells have almost no detail.", None, len(flat_cells), 6,
            "add subtle texture, grain or secondary shapes")

    # ---- composition (coarse grid)
    gw = 128
    gh = max(8, round(h * gw / w))
    g = np.asarray(Image.fromarray((np.clip(im, 0, 1) * 255 + 0.5).astype(np.uint8)).resize((gw, gh), _RES.BOX), np.float32) / 255
    gl = g @ _LUMA
    grad = np.zeros((gh, gw), np.float32)
    grad[:, :-1] += np.abs(np.diff(gl, axis=1))
    grad[:-1, :] += np.abs(np.diff(gl, axis=0))
    cgrad = np.zeros((gh, gw), np.float32)
    cgrad[:, :-1] += np.abs(np.diff(g, axis=1)).sum(-1)
    cgrad[:-1, :] += np.abs(np.diff(g, axis=0)).sum(-1)
    flat = (grad < 0.012) & (cgrad < 0.04)
    pad = np.pad(flat, 1)
    nb = sum(pad[1 + dy:1 + dy + gh, 1 + dx:1 + dx + gw] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    comps_flat = _components(nb >= 6)                           # smoothed: grain does not break up a flat area
    if comps_flat:
        big_flat = max(comps_flat, key=len)
        frac = len(big_flat) / flat.size
        if frac >= 0.35:
            fm = np.zeros(flat.shape, bool)
            fm[big_flat[:, 0], big_flat[:, 1]] = True
            what = ("The canvas is almost empty: " if frac >= 0.9 else "") + f"one flat, detail-free area fills {frac * 100:.0f}% of the canvas."
            add("sparse_composition", "warn" if frac >= 0.5 else "info", what[0].upper() + what[1:], _bbox_frac(fm), frac, [0.35, 0.5],
                "fill it with a secondary element, texture, light falloff or a text block; or crop the canvas tighter")

    med = np.median(g.reshape(-1, 3), axis=0)
    sal = np.linalg.norm(g - med, axis=-1) > 0.2
    comps = [c for c in _components(sal) if len(c) >= 0.004 * sal.size]
    if comps:
        big = max(comps, key=len)
        by0, bx0 = big.min(0)
        by1, bx1 = big.max(0) + 1
        area_f = len(big) / sal.size
        touches = {"left": bx0 == 0, "right": bx1 == gw, "top": by0 == 0, "bottom": by1 == gh}
        sides = [k for k, v in touches.items() if v]
        span = {"left": (by1 - by0) / gh, "right": (by1 - by0) / gh, "top": (bx1 - bx0) / gw, "bottom": (bx1 - bx0) / gw}
        opp = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
        cut = [k for k in sides if span[k] < 0.85 and not touches[opp[k]]]
        if cut and 0.02 < area_f < 0.4:
            add("subject_cropped", "warn", f"The main subject touches the {'/'.join(cut)} edge and is cut off by the frame.",
                [bx0 / gw, by0 / gh, bx1 / gw, by1 / gh], cut, 0, "move or shrink it so a margin of at least 4% remains, or crop on purpose with a larger overlap")
        row = [c for c in comps if len(c) >= 0.006 * sal.size]
        if len(row) >= 3:
            cyv = np.array([c[:, 0].mean() / gh for c in row])
            cxv = np.array([c[:, 1].mean() / gw for c in row])
            if cyv.std() < 0.04 and 0.3 < cyv.mean() < 0.7 and np.ptp(cxv) > 0.4:
                add("subjects_in_a_row", "info", f"{len(row)} elements line up in one horizontal row at V={cyv.mean():.2f}.",
                    [0.5, cyv.mean()], float(cyv.std()), 0.04, "stagger them in depth: vary size, height and overlap, or group in a triangle")

    # ---- focal-area contrast
    w8 = grad ** 2
    if w8.sum() > 1e-6:
        fy = float((w8.sum(1) * np.arange(gh)).sum() / w8.sum()) / gh
        fx = float((w8.sum(0) * np.arange(gw)).sum() / w8.sum()) / gw
        x0, x1 = max(0, int((fx - 0.125) * w)), min(w, int((fx + 0.125) * w))
        y0, y1 = max(0, int((fy - 0.125) * h)), min(h, int((fy + 0.125) * h))
        win = im[y0:y1, x0:x1]
        if win.size:
            wl = win @ _LUMA
            rng = float(np.percentile(wl, 95) - np.percentile(wl, 5))
            chroma = float((win.max(-1) - win.min(-1)).mean())
            val = rng + 0.5 * chroma
            if val < 0.12:
                add("low_focal_contrast", "warn", f"Very low local contrast around the focal area (range {rng:.2f}).",
                    [x0 / w, y0 / h, x1 / w, y1 / h], val, 0.12, "separate the subject from its surroundings: lighten or darken it, or shift its hue")

    # ---- low-contrast crisp edges: object blends into the background
    cand = np.zeros((h, w), bool)
    for ax in (0, 1):
        a = np.moveaxis(im, ax, 0)
        if a.shape[0] < 9:
            continue
        left = np.linalg.norm(a[2:-2] - a[:-4], axis=-1)           # |I[j+2] - I[j]|
        right = np.linalg.norm(a[4:] - a[2:-2], axis=-1)           # |I[j+4] - I[j+2]|
        total = np.linalg.norm(a[4:] - a[:-4], axis=-1)
        step = (total > 0.02) & (total < 0.07) & (np.maximum(left, right) > 0.8 * total) & (np.minimum(left, right) < 0.006)
        k = np.zeros(a.shape[:2], bool)
        k[2:-2] = step
        cand |= np.moveaxis(k, 0, ax)
    # long crisp edges only (run of >= 14 px along the edge direction)
    long_v = _run_mask(cand, 14, 0)
    long_h = _run_mask(cand, 14, 1)
    longm = long_v | long_h
    length = longm.sum() / float(w + h)
    if length > 0.5:
        add("blends_into_background", "warn" if length > 1.2 else "info",
            "Long, crisp edges with almost no colour step: neighbouring colours are near-duplicates, so objects blend into what is behind them.",
            _bbox_frac(longm), length, 0.5, "shift value or hue of the object by 10-15% against its background, or add a rim light / outline")

    # ---- gradient banding & compression noise (full resolution)
    f8 = np.round(full @ _LUMA * 255).astype(np.int16)
    if min(f8.shape) >= 64:
        band, bw = _banding(f8)
        if band > 0.06:
            add("banding", "warn" if band > 0.15 else "info", f"Visible banding in smooth gradients on {band * 100:.0f}% of sampled scan lines.",
                bw, band, 0.06, "add 0.5-1.5 levels of grain / dither (cv.grain or add_noise) after the gradient, or build it in float and quantise once at the end")
        ctr = full[max(0, full.shape[0] // 2 - 256): full.shape[0] // 2 + 256, max(0, full.shape[1] // 2 - 384): full.shape[1] // 2 + 384]
        cl = ctr @ _LUMA
        lap = cl[1:-1, 1:-1] * 4 - cl[:-2, 1:-1] - cl[2:, 1:-1] - cl[1:-1, :-2] - cl[1:-1, 2:]
        bh, bw_ = (lap.shape[0] // 8) * 8, (lap.shape[1] // 8) * 8
        if bh >= 64 and bw_ >= 64:
            blocks = np.abs(lap[:bh, :bw_]).reshape(bh // 8, 8, bw_ // 8, 8).mean((1, 3))
            floor = float(np.percentile(blocks, 10)) / 3.6           # noise std (0..1) of the quietest areas
            dx = np.abs(np.diff(f8.astype(np.float32) / 255, axis=1)).mean(0)
            col = dx[: (len(dx) // 8) * 8].reshape(-1, 8)
            block_ratio = float(col[:, 7].mean() / max(col[:, :7].mean(), 1e-6)) if col.size else 1.0
            if col.size and col[:, :7].mean() < 0.003:
                block_ratio = 1.0                                   # flat image: the ratio is meaningless
            if floor > 0.024 or block_ratio > 1.5:
                add("jpeg_noise", "warn", f"Heavy pixel noise or JPEG-style blocking (noise floor {floor * 255:.1f} levels, 8px block ratio {block_ratio:.2f}).",
                    None, floor if floor > 0.024 else block_ratio, 0.024 if floor > 0.024 else 1.5, "save as PNG (not JPEG) for inspection, lower grain / noise amounts, and blur the noise layer before adding it")

    order = {s: i for i, s in enumerate(SEVERITIES)}
    out.sort(key=lambda t: order[t["severity"]])
    return out


def trace_event(tool, command, args=None, outputs=None, seconds=None, warnings=None, **extra):
    """Append one JSON line to the file named by env ``PIXELCANVAS_TRACE`` (no-op when unset; never raises)."""
    import json
    import os
    import time
    path = os.environ.get("PIXELCANVAS_TRACE")
    if not path:
        return
    try:
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "tool": tool, "command": command, "arguments": args or {},
               "outputs": outputs or [], "seconds": None if seconds is None else round(float(seconds), 3),
               "warnings": warnings or []}
        rec.update(extra)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass
