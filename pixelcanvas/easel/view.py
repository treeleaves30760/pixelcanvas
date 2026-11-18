"""Observation renders (DESIGN.md section 9).

A view is one picture made of panels (CANVAS / REFERENCE / SKETCH / DIFF ...) over the same canvas area. After a
command the tool writes the canvas and the reference as two separate one-panel pictures (``VIEW:`` and
``VIEW-REF:``): the test model compares two separate images better than one side-by-side composite and reads
coordinates off each picture's own rulers (work/research/03-qwen38-probe.md, items 2 and 4). Every panel has a title
band and rulers labelled in 0-1000 canvas units on its top and left edges, outside the image. Sizes per view type:
``VIEW_SIDE`` (768) for canvas / reference views, ``MAP_SIDE`` (1024) for the marks and regions maps whose numbers
and names must stay readable; never above ``MAX_SIDE``; both sides multiples of 32; small areas are enlarged (up to
8 screen px per canvas px); the layout (side by side, stacked or a grid) is the one that shows the area largest.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import coords

MAX_SIDE = 1280
VIEW_SIDE = 768      # canvas / reference views: ~13 s per look for the test model, no accuracy loss against 1024+
MAP_SIDE = 1024      # marks and regions maps: numbers and names must stay readable
TINT_ALPHA = 0.38
MULT = 32
MAX_ZOOM = 8.0
MARGIN = 8
GAP = 14
TITLE_H = 30
RULER_T = 24
RULER_L = 42
BG = (255, 255, 255)
RULER_BG = (236, 236, 236)
INK = (20, 20, 20)
MARK_COLOURS = [(230, 25, 75), (0, 130, 200), (60, 180, 75), (245, 130, 48), (145, 30, 180), (0, 160, 160),
                (240, 50, 230), (128, 128, 0), (170, 110, 40), (0, 0, 128), (128, 0, 0), (70, 70, 70)]
SPOT_COLOUR = (255, 0, 200)


@lru_cache(maxsize=16)
def font(size: int, bold: bool = True):
    names = (["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"] if bold else
             ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"])
    dirs = ["", "C:/Windows/Fonts/", "/usr/share/fonts/truetype/dejavu/", "/usr/share/fonts/TTF/",
            "/Library/Fonts/", "/System/Library/Fonts/Supplemental/"]
    for n in names:
        for d in dirs:
            try:
                return ImageFont.truetype(d + n, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, f) -> Tuple[int, int]:
    b = draw.textbbox((0, 0), text, font=f)
    return b[2] - b[0], b[3] - b[1]


@dataclass
class Panel:
    img: np.ndarray                 # uint8 RGB, full canvas size
    title: str
    spots: List[Tuple[tuple, str]] = field(default_factory=list)    # (box_px, label) drawn as numbered boxes
    outlines: List[Tuple[np.ndarray, str, tuple]] = field(default_factory=list)  # (mask, label, colour)
    labels: Optional[np.ndarray] = None   # int label map: boundaries + numbers (marks)
    label_anchors: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    nearest: bool = False           # force nearest-neighbour scaling
    tints: List[Tuple[np.ndarray, tuple]] = field(default_factory=list)   # (mask, colour) translucent fills


# ----------------------------------------------------------------------------------------------- layout

def _layout(n: int, cw: int, ch: int, max_side: int = MAX_SIDE):
    """Best (rows, cols, scale) for n panels of a cw x ch area."""
    best = None
    for rows in range(1, n + 1):
        cols = int(math.ceil(n / rows))
        if (rows - 1) * cols >= n:
            continue
        fixed_w = 2 * MARGIN + cols * RULER_L + (cols - 1) * GAP
        fixed_h = 2 * MARGIN + rows * (TITLE_H + RULER_T) + (rows - 1) * GAP
        s = min((max_side - fixed_w) / (cols * cw), (max_side - fixed_h) / (rows * ch))
        s = min(s, MAX_ZOOM)
        if best is None or s > best[2] + 1e-9:
            best = (rows, cols, s)
    return best


def _nice_step(px_per_unit: float, min_px: float) -> int:
    for st in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000):
        if st * px_per_unit >= min_px:
            return st
    return 1000


def render(panels: Sequence[Panel], box_px, grid: bool = False, max_side: int = VIEW_SIDE,
           footer: str = "") -> np.ndarray:
    """Render panels over the canvas area ``box_px`` (x0, y0, x1, y1; x1/y1 exclusive)."""
    H, W = panels[0].img.shape[:2]
    x0, y0, x1, y1 = box_px
    cw, ch = x1 - x0, y1 - y0
    footer_h = 24 if footer else 0
    rows, cols, s = _layout(len(panels), cw, ch, max_side - footer_h)
    pw, ph = max(1, int(math.floor(cw * s))), max(1, int(math.floor(ch * s)))
    cell_w, cell_h = RULER_L + pw, TITLE_H + RULER_T + ph
    tot_w = 2 * MARGIN + cols * cell_w + (cols - 1) * GAP
    tot_h = 2 * MARGIN + rows * cell_h + (rows - 1) * GAP + footer_h
    out_w = min(max_side, int(math.ceil(tot_w / MULT) * MULT))
    out_h = min(max_side, int(math.ceil(tot_h / MULT) * MULT))
    canvas = Image.new("RGB", (out_w, out_h), BG)
    draw = ImageDraw.Draw(canvas)
    sx, sy = pw / cw, ph / ch
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        ox = MARGIN + c * (cell_w + GAP)
        oy = MARGIN + r * (cell_h + GAP)
        ix, iy = ox + RULER_L, oy + TITLE_H + RULER_T
        crop = p.img[y0:y1, x0:x1]
        interp = cv2.INTER_NEAREST if (p.nearest or s >= 2.0) else (cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        tile = cv2.resize(crop, (pw, ph), interpolation=interp)
        for mask, colour in p.tints:
            tile = _draw_tint(tile, mask[y0:y1, x0:x1], pw, ph, colour)
        if p.labels is not None:
            tile = _draw_label_bounds(tile, p.labels[y0:y1, x0:x1], pw, ph)
        for mask, _, colour in p.outlines:
            tile = _draw_outline(tile, mask[y0:y1, x0:x1], pw, ph, colour)
        canvas.paste(Image.fromarray(tile), (ix, iy))
        if grid:
            _draw_grid(draw, ix, iy, pw, ph, x0, y0, x1, y1, W, H)
        _draw_rulers(draw, ox, oy + TITLE_H, ix, iy, pw, ph, x0, y0, x1, y1, W, H)
        draw.text((ix, oy + 4), p.title, fill=INK, font=font(19))
        # label texts on top
        if p.labels is not None:
            for lab, (ax, ay) in p.label_anchors.items():
                if x0 <= ax < x1 and y0 <= ay < y1:
                    _tag(draw, ix + (ax - x0 + 0.5) * sx, iy + (ay - y0 + 0.5) * sy, str(lab), (255, 255, 255), INK)
        for mask, label, colour in p.outlines:
            if not label:
                continue
            from .regions import label_anchor
            sub = mask[y0:y1, x0:x1]
            if not sub.any():
                continue
            ax, ay = label_anchor(sub)
            _tag(draw, ix + (ax + 0.5) * sx, iy + (ay + 0.5) * sy, label, colour, (255, 255, 255))
        for bx, label in p.spots:
            _spot(draw, bx, label, ix, iy, x0, y0, sx, sy, pw, ph)
        # thin frame
        draw.rectangle([ix - 1, iy - 1, ix + pw, iy + ph], outline=(150, 150, 150), width=1)
    # separators between columns / rows
    for c in range(1, cols):
        x = MARGIN + c * (cell_w + GAP) - GAP // 2
        draw.line([(x, MARGIN), (x, MARGIN + rows * cell_h + (rows - 1) * GAP)], fill=(190, 190, 190), width=1)
    for r in range(1, rows):
        y = MARGIN + r * (cell_h + GAP) - GAP // 2
        draw.line([(MARGIN, y), (MARGIN + cols * cell_w + (cols - 1) * GAP, y)], fill=(190, 190, 190), width=1)
    if footer:
        draw.text((MARGIN, out_h - footer_h + 2), footer, fill=(90, 90, 90), font=font(15, bold=False))
    return np.asarray(canvas)


def _draw_rulers(draw, ox, oy, ix, iy, pw, ph, x0, y0, x1, y1, W, H):
    f = font(14)
    draw.rectangle([ix, oy, ix + pw - 1, iy - 1], fill=RULER_BG)
    draw.rectangle([ox, iy, ix - 1, iy + ph - 1], fill=RULER_BG)
    # x axis
    u0, u1 = x0 * 1000.0 / W, x1 * 1000.0 / W
    ppu = pw / max(1e-6, (u1 - u0))
    step = _nice_step(ppu, 58)
    minor = step / 5 if step * ppu / 5 >= 7 else (step / 2 if step * ppu / 2 >= 7 else None)
    _ticks(draw, u0, u1, step, minor, lambda u: ix + (u - u0) * ppu, axis="x", base=iy, f=f, lim=(ix, ix + pw))
    v0, v1 = y0 * 1000.0 / H, y1 * 1000.0 / H
    ppv = ph / max(1e-6, (v1 - v0))
    step = _nice_step(ppv, 34)
    minor = step / 5 if step * ppv / 5 >= 7 else (step / 2 if step * ppv / 2 >= 7 else None)
    _ticks(draw, v0, v1, step, minor, lambda v: iy + (v - v0) * ppv, axis="y", base=ix, f=f, lim=(iy, iy + ph))


def _ticks(draw, a, b, step, minor, pos, axis, base, f, lim):
    if minor:
        k = math.ceil(a / minor - 1e-9)
        while k * minor <= b + 1e-9:
            p = pos(k * minor)
            if lim[0] <= p <= lim[1]:
                if axis == "x":
                    draw.line([(p, base - 5), (p, base - 1)], fill=(110, 110, 110))
                else:
                    draw.line([(base - 5, p), (base - 1, p)], fill=(110, 110, 110))
            k += 1
    k = math.ceil(a / step - 1e-9)
    while k * step <= b + 1e-9:
        u = k * step
        p = pos(u)
        if lim[0] - 0.5 <= p <= lim[1] + 0.5:
            label = str(int(round(u)))
            tw, th = text_size(draw, label, f)
            if axis == "x":
                draw.line([(p, base - 9), (p, base - 1)], fill=INK, width=1)
                tx = min(max(p - tw / 2, lim[0] - RULER_L + 2), lim[1] - tw)
                draw.text((tx, base - RULER_T + 1), label, fill=INK, font=f)
            else:
                draw.line([(base - 9, p), (base - 1, p)], fill=INK, width=1)
                ty = min(max(p - th / 2 - 2, lim[0]), lim[1] - th - 3)
                draw.text((base - 11 - tw, ty), label, fill=INK, font=f)
        k += 1


def _draw_grid(draw, ix, iy, pw, ph, x0, y0, x1, y1, W, H):
    u0, u1 = x0 * 1000.0 / W, x1 * 1000.0 / W
    ppu = pw / max(1e-6, u1 - u0)
    step = _nice_step(ppu, 58)
    col = (255, 0, 255)
    k = math.ceil(u0 / step)
    while k * step <= u1:
        x = ix + (k * step - u0) * ppu
        draw.line([(x, iy), (x, iy + ph - 1)], fill=col, width=1)
        k += 1
    v0, v1 = y0 * 1000.0 / H, y1 * 1000.0 / H
    ppv = ph / max(1e-6, v1 - v0)
    step = _nice_step(ppv, 34)
    k = math.ceil(v0 / step)
    while k * step <= v1:
        y = iy + (k * step - v0) * ppv
        draw.line([(ix, y), (ix + pw - 1, y)], fill=col, width=1)
        k += 1


def _tag(draw, cx, cy, text, bg, fg, size: int = 15):
    f = font(size)
    tw, th = text_size(draw, text, f)
    x, y = cx - tw / 2 - 3, cy - th / 2 - 4
    draw.rectangle([x, y, x + tw + 6, y + th + 7], fill=bg, outline=fg if bg == (255, 255, 255) else (255, 255, 255))
    draw.text((x + 3, y + 1), text, fill=fg, font=f)


def _spot(draw, bx, label, ix, iy, x0, y0, sx, sy, pw, ph):
    a = ix + (bx[0] - x0) * sx
    b = iy + (bx[1] - y0) * sy
    c = ix + (bx[2] - x0) * sx
    d = iy + (bx[3] - y0) * sy
    a, c = max(ix, a), min(ix + pw - 1, c)
    b, d = max(iy, b), min(iy + ph - 1, d)
    if c <= a or d <= b:
        return
    draw.rectangle([a, b, c, d], outline=SPOT_COLOUR, width=2)
    f = font(14)
    tw, th = text_size(draw, label, f)
    draw.rectangle([a, b, a + tw + 5, b + th + 6], fill=SPOT_COLOUR)
    draw.text((a + 3, b), label, fill=(255, 255, 255), font=f)


def _boundary(lab_small: np.ndarray) -> np.ndarray:
    b = np.zeros(lab_small.shape, bool)
    b[:, 1:] |= lab_small[:, 1:] != lab_small[:, :-1]
    b[1:, :] |= lab_small[1:, :] != lab_small[:-1, :]
    return b


def _draw_label_bounds(tile, labels, pw, ph):
    lab = cv2.resize(labels.astype(np.float32), (pw, ph), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    b = _boundary(lab)
    thick = cv2.dilate(b.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    out = tile.copy()
    out[thick] = (out[thick] * 0.3).astype(np.uint8)
    out[b] = (255, 255, 255)
    return out


def _draw_tint(tile, mask, pw, ph, colour):
    m = cv2.resize(mask.astype(np.uint8), (pw, ph), interpolation=cv2.INTER_NEAREST) > 0
    if not m.any():
        return tile
    out = tile.astype(np.float32)
    out[m] = out[m] * (1 - TINT_ALPHA) + np.array(colour, np.float32) * TINT_ALPHA
    return out.astype(np.uint8)


def _draw_outline(tile, mask, pw, ph, colour):
    m = cv2.resize(mask.astype(np.uint8), (pw, ph), interpolation=cv2.INTER_NEAREST)
    if not m.any():
        return tile
    edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    out = tile.copy()
    halo = cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    out[halo & ~edge] = (255, 255, 255)
    out[edge] = colour
    return out


# ----------------------------------------------------------------------------------------------- panel helpers

def heat_panel(canvas_u8: np.ndarray, err: np.ndarray) -> np.ndarray:
    """Grey canvas with the error as a heat overlay (dark = right, yellow / white = wrong)."""
    grey = cv2.cvtColor(canvas_u8, cv2.COLOR_RGB2GRAY)
    grey = cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB).astype(np.float32) * 0.35
    e = np.clip(err, 0, 1)
    if e.shape[:2] != canvas_u8.shape[:2]:      # an error map measured at the delivered size, a working canvas
        e = cv2.resize(e.astype(np.float32), (canvas_u8.shape[1], canvas_u8.shape[0]), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap((e * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1].astype(np.float32)
    a = np.clip(e * 1.6, 0, 1)[..., None]
    return np.clip(grey * (1 - a) + heat * a, 0, 255).astype(np.uint8)


def checker(H: int, W: int, cell: int = 16) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    c = ((yy // cell + xx // cell) % 2).astype(np.float32)
    v = 0.86 + 0.08 * c
    return np.repeat(v[..., None], 3, axis=2)


def layer_on_checker(layer: Optional[np.ndarray], H: int, W: int) -> np.ndarray:
    bg = checker(H, W, max(8, max(H, W) // 60))
    if layer is None:
        return (bg * 255).astype(np.uint8)
    a = layer[..., 3:4]
    return (np.clip(layer[..., :3] * a + bg * (1 - a), 0, 1) * 255 + 0.5).astype(np.uint8)


def region_map(base: np.ndarray, regions: Sequence[Tuple[str, np.ndarray, int]], title: str,
               highlight: Optional[str] = None) -> Panel:
    """One panel with every region tinted in its own colour, outlined and labelled ``name f<focus>``; the
    ``highlight`` region (the one just made) gets a magenta outline on top."""
    pan = Panel(base, title)
    for i, (nm, mask, focus) in enumerate(regions):
        col = MARK_COLOURS[i % len(MARK_COLOURS)]
        pan.tints.append((mask, col))
        pan.outlines.append((mask, f"{nm} f{focus}", col))
    if highlight is not None:
        for nm, mask, _ in regions:
            if nm == highlight:
                pan.outlines.append((mask, "", SPOT_COLOUR))
    return pan


def region_overlay(base: np.ndarray, regions: Sequence[Tuple[str, np.ndarray, int]]) -> np.ndarray:
    """The regions tinted and outlined over ``base`` at canvas size (the plan stage's picture)."""
    H, W = base.shape[:2]
    out = base.copy()
    for i, (_, mask, _) in enumerate(regions):
        col = MARK_COLOURS[i % len(MARK_COLOURS)]
        out = _draw_tint(out, mask, W, H, col)
        out = _draw_outline(out, mask, W, H, col)
    return out


def size_ok(img: np.ndarray) -> bool:
    h, w = img.shape[:2]
    return max(h, w) <= MAX_SIDE and h % MULT == 0 and w % MULT == 0
