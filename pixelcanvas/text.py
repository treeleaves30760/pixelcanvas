"""``pixelcanvas text`` and ``pixelcanvas fonts``: lettering on the ``text`` layer.

Lettering is fitted to a box (largest size that fits, wrapped to the box width unless ``--fit`` asks for one line),
CJK included. The font machinery is ``pixelcanvas.fonts`` (font discovery with CJK fallback, layout,
anti-aliased masks). The project keeps the text items in ``state["texts"]``; the ``text`` layer is always the render of
those items, so ``--list`` and ``--remove`` are exact and every change is one undoable step.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import coords, compose
from .coords import UsageError

WEIGHTS = {"regular": False, "normal": False, "book": False, "light": "light", "thin": "light", "medium": "medium",
           "semibold": "bold", "bold": "bold", "black": "black", "heavy": "black"}
ALIGNS = ("left", "center", "right")


def _fonts():
    from . import fonts
    return fonts


# ----------------------------------------------------------------------------------------------- rendering

def _font_spec(name: Optional[str], text: str):
    """Resolve ``--font`` (a path, a family / file name or style words) to what the layout accepts."""
    if not name:
        return None
    lt = _fonts()
    try:
        spec = lt.find_font(name, text=text)
    except Exception:  # noqa: BLE001
        spec = None
    if spec is None:
        idx = [stem for stem, _ in lt.list_fonts("")]
        import difflib
        close = difflib.get_close_matches(re.sub(r"\s+", "", name.lower()), idx, n=3, cutoff=0.5)
        hint = f" (did you mean {', '.join(close)}?)" if close else ""
        raise UsageError(f"font {name!r} was not found{hint}; list the installed ones with  pixelcanvas fonts",
                         f'pixelcanvas text "TITLE" --box 100,50,900,150 --font {close[0] if close else "serif"}')
    return spec


def _item_masks(H: int, W: int, it: dict) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    lt = _fonts()
    font = _font_spec(it.get("font"), it["text"])
    kw = dict(font=font, bold=WEIGHTS.get(it.get("weight", "regular"), False), anchor="la", align=it.get("align", "center"),
              max_width=it.get("max_width"), vertical=bool(it.get("vertical")))
    if it.get("vertical") and it.get("max_height"):
        kw["max_height"] = it["max_height"]
    fill = lt.text_mask(H, W, it["text"], tuple(it["xy"]), size=it["size"], **kw)
    out = None
    if it.get("outline"):
        out = lt.text_mask(H, W, it["text"], tuple(it["xy"]), size=it["size"], stroke=float(it["outline"][0]), **kw)
        out = np.maximum(out, fill)
    return np.clip(fill, 0, 1), (None if out is None else np.clip(out, 0, 1))


def render_items(H: int, W: int, items: List[dict]) -> np.ndarray:
    """The text layer: RGBA float32, straight alpha, outline under fill, items in order."""
    rgb = np.zeros((H, W, 3), np.float32)
    a = np.zeros((H, W), np.float32)
    for it in items:
        fill, out = _item_masks(H, W, it)
        if out is not None:
            oc = np.asarray(coords.parse_colour(it["outline"][1]), np.float32) / 255.0
            src = np.zeros((H, W, 4), np.float32)
            src[..., :3] = oc
            src[..., 3] = out
            rgb, a = compose.composite_over(rgb, a, src)
        c = np.asarray(coords.parse_colour(it["color"]), np.float32) / 255.0
        src = np.zeros((H, W, 4), np.float32)
        src[..., :3] = c
        src[..., 3] = fill
        rgb, a = compose.composite_over(rgb, a, src)
    out = np.zeros((H, W, 4), np.float32)
    out[..., :3] = rgb
    out[..., 3] = np.round(a * 255.0) / 255.0
    out[out[..., 3] <= 0, :3] = 0.0
    return out


def _ink_box(mask: np.ndarray, thr: float = 0.04):
    m = mask > thr
    ys = np.flatnonzero(m.any(1))
    if len(ys) == 0:
        return None
    xs = np.flatnonzero(m.any(0))
    return float(xs[0]), float(ys[0]), float(xs[-1] + 1), float(ys[-1] + 1)


def layout_item(H: int, W: int, text: str, box_px, *, font, weight: str, align: str, vertical: bool, fit: bool,
                outline_px: float, size: Optional[float]) -> dict:
    """Size and position that fit ``text`` into ``box_px`` (ink inside the box, outline included)."""
    lt = _fonts()
    x0, y0, x1, y1 = box_px
    pad = float(outline_px)
    bw, bh = max(4.0, x1 - x0 - 2 * pad), max(4.0, y1 - y0 - 2 * pad)
    bold = WEIGHTS.get(weight, False)
    wrap = not fit
    if size:
        sz = float(size)
    else:
        kw = {"vertical": True} if vertical else {}
        sz = float(lt.fit_text_size(text, bw, bh, font=font, bold=bold, wrap=wrap, min_size=4, **kw))
    max_w = bw if (wrap and not vertical) else None
    max_h = bh if (wrap and vertical) else None

    # measure on a padded canvas: ink that wraps or runs past the canvas edge is still measured (and then
    # shrunk to fit), instead of being cut off and read as "fits"
    P = int(max(H, W))

    def render(s, xy):
        kw2 = dict(font=font, bold=bold, anchor="la", align=align, max_width=max_w, vertical=vertical)
        if max_h:
            kw2["max_height"] = max_h
        return lt.text_mask(H + 2 * P, W + 2 * P, text, (xy[0] + P, xy[1] + P), size=s, **kw2)

    def ink_of(s, xy):
        ib = _ink_box(render(s, xy))
        return None if ib is None else (ib[0] - P, ib[1] - P, ib[2] - P, ib[3] - P)

    xy0 = (x0 + pad, y0 + pad)
    ink = None
    for _ in range(14):
        ink = ink_of(sz, xy0)
        if ink is None:
            raise UsageError("the text came out empty (spaces only, or the font has none of these characters; "
                             "list fonts with  pixelcanvas fonts)", 'pixelcanvas text "TITLE" --box 100,50,900,150')
        if size or (ink[2] - ink[0] <= bw + 1.0 and ink[3] - ink[1] <= bh + 1.0) or sz <= 5:
            break                        # an explicit --size is kept; the overflow check reports it
        sz = max(5.0, sz * 0.95)
    if sz < 6 and not size:
        raise UsageError(f"the box is too small for this text (best size {sz:.0f} px); use a bigger box or fewer words",
                         'pixelcanvas text "TITLE" --box 100,50,900,150')
    if not size:
        # the layout metrics leave slack around the ink: grow until the ink itself fills the box
        r = min(bw / max(1.0, ink[2] - ink[0]), bh / max(1.0, ink[3] - ink[1]))
        if r > 1.03:
            lo, hi = sz, sz * min(r, 1.8)
            for _ in range(7):
                mid = (lo + hi) / 2.0
                ib = ink_of(mid, xy0)
                if ib is not None and ib[2] - ib[0] <= bw + 1.0 and ib[3] - ib[1] <= bh + 1.0:
                    lo, ink = mid, ib
                else:
                    hi = mid
            sz = lo
            ink = ink_of(sz, xy0) or ink
    ink_w, ink_h = ink[2] - ink[0], ink[3] - ink[1]
    cy = (y0 + y1) / 2.0
    dy = cy - (ink[1] + ink[3]) / 2.0
    if align == "left":
        dx = (x0 + pad) - ink[0]
    elif align == "right":
        dx = (x1 - pad) - ink[2]
    else:
        dx = (x0 + x1) / 2.0 - (ink[0] + ink[2]) / 2.0
    xy = (xy0[0] + dx, xy0[1] + dy)
    return {"size": float(np.floor(sz * 100.0) / 100.0), "xy": [round(xy[0], 2), round(xy[1], 2)], "max_width": max_w, "max_height": max_h,
            "ink": [round(ink_w, 1), round(ink_h, 1)]}


# ----------------------------------------------------------------------------------------------- commands

def _parse_outline(text: str, fill_rgb) -> list:
    parts = [t for t in re.split(r"[,\s]+", str(text).strip()) if t]
    px, col = None, None
    for t in parts:
        if re.fullmatch(r"\d+(?:\.\d+)?(?:px)?", t) and px is None:
            px = float(t.rstrip("px"))
        else:
            col = t if col is None else col + "," + t
    if px is None and col is None:
        raise UsageError("--outline needs a width and a colour like 3,#000000", 'pixelcanvas text "TITLE" --box 100,50,900,150 --outline 3,#000000')
    if px is None:
        px = 2.0
    if col is None:
        lum = 0.2126 * fill_rgb[0] + 0.7152 * fill_rgb[1] + 0.0722 * fill_rgb[2]
        col = "#101014" if lum > 140 else "#f6f6f6"
    rgb = coords.parse_colour(col, 'pixelcanvas text "TITLE" --box 100,50,900,150 --outline 3,#000000')
    return [float(max(0.5, min(60.0, px))), coords.hex_colour(rgb)]


def _describe(it: dict, n: int) -> str:
    t = it["text"] if len(it["text"]) <= 28 else it["text"][:25] + "..."
    bits = [f"{it['size']:g} px", it["color"], it.get("weight", "regular"), it.get("align", "center")]
    if it.get("font"):
        bits.append(it["font"])
    if it.get("outline"):
        bits.append(f"outline {it['outline'][0]:g}px {it['outline'][1]}")
    if it.get("vertical"):
        bits.append("vertical")
    return f"  {n}  \"{t}\"  box {coords.fmt_box(it['box'])}  " + "  ".join(bits)


def cmd_text(ctx, args: List[str]):
    from . import cli
    from .cli import Region, Report, Spec, parse
    from . import regions as regions_mod
    usage = cli.USAGE["text"]
    sp = Spec(usage, ["string*"],
              {"box": "value", "font": "value", "color": "value", "weight": "value", "align": "value",
               "outline": "value", "vertical": "flag", "fit": "flag", "size": "value", "list": "flag",
               "remove": "value"},
              alias={"colour": "color", "bold": "weight", "stroke": "outline", "at": "box", "area": "box",
                     "rm": "remove", "delete": "remove", "ls": "list", "font-size": "size", "alignment": "align"})
    a = parse(args, sp)
    p = ctx.project
    H, W = p.H, p.W
    items: List[dict] = [dict(x) for x in p.state.get("texts", [])]

    if a.get("list") or (not a.get("string") and not a.get("remove")):
        rep = Report()
        if not items:
            rep.add("no text on the picture yet")
        else:
            rep.add(f"{len(items)} text item(s) on the text layer:")
            for i, it in enumerate(items, 1):
                rep.add(_describe(it, i))
        rep.next = 'pixelcanvas text "TITLE" --box 100,50,900,150 --color #ffffff' if not items else \
            "pixelcanvas text --remove 1   (or add more lettering with  pixelcanvas text \"...\" --box ...)"
        return rep

    if a.get("remove"):
        if not items:
            raise UsageError("there is no text to remove", usage)
        m = re.search(r"\d+", str(a["remove"]))
        n = int(m.group(0)) if m else 0
        if not 1 <= n <= len(items):
            raise UsageError(f"there is no text item {a['remove']} (1-{len(items)}); see  pixelcanvas text --list",
                             "pixelcanvas text --remove 1")
        gone = items.pop(n - 1)
        layer = render_items(H, W, items)
        box_px = coords.box_to_px(gone["box"], W, H)
        region = Region("text", regions_mod.box_mask(H, W, box_px), box_px, f"text {n}", False)

        def make(arr):
            p.state["texts"] = items
            return layer
        rep = cli.paint_step(ctx, label=f"text remove {n} \"{gone['text'][:24]}\"", layer="text", build=lambda: [],
                             clip=None, area_px=box_px, region=region, pixel_fn=make, key_metric=None)
        return rep

    text = str(a["string"]).replace("\\n", "\n")
    if not a.get("box"):
        raise UsageError("text needs --box x0,y0,x1,y1 (0-1000) where the lettering goes",
                         f'pixelcanvas text "{text[:20]}" --box 100,50,900,150')
    box = coords.parse_box(a["box"], ctx.notes, f'pixelcanvas text "{text[:20]}" --box 100,50,900,150')
    box_px = coords.box_to_px(box, W, H)
    if box_px[2] - box_px[0] < 8 or box_px[3] - box_px[1] < 8:
        raise UsageError("the box is too small for lettering", f'pixelcanvas text "{text[:20]}" --box 100,50,900,150')
    weight = str(a.get("weight") or "regular").lower().replace(" ", "").replace("-", "")
    if weight not in WEIGHTS:
        raise UsageError(f"unknown --weight {a['weight']!r}; one of regular, light, bold, black",
                         f'pixelcanvas text "{text[:20]}" --box {coords.fmt_box(box)} --weight bold')
    align = str(a.get("align") or "center").lower()
    align = {"centre": "center", "middle": "center", "l": "left", "r": "right", "c": "center"}.get(align, align)
    if align not in ALIGNS:
        raise UsageError(f"unknown --align {a['align']!r}; one of left, center, right",
                         f'pixelcanvas text "{text[:20]}" --box {coords.fmt_box(box)} --align left')
    vertical = bool(a.get("vertical"))
    font_name = str(a["font"]) if a.get("font") else None
    font = _font_spec(font_name, text)
    size = None
    if a.get("size"):
        size = cli._num(a, "size", 0.0, 4, 2000, ctx.notes, usage)
    # colour: #hex / name, or pick = eyedropper on the reference under the glyphs, default = readable on the canvas
    canvas = p.composite_u8("score")
    outline = None
    outline_px = 0.0
    if a.get("outline"):
        o = _parse_outline(a["outline"], (255, 255, 255))
        outline_px = o[0]
    lay = layout_item(H, W, text, box_px, font=font, weight=weight, align=align, vertical=vertical,
                      fit=bool(a.get("fit")), outline_px=outline_px, size=size)
    probe = {"text": text, "font": font_name, "weight": weight, "align": align, "vertical": vertical, "size": lay["size"],
             "xy": lay["xy"], "max_width": lay["max_width"], "max_height": lay["max_height"], "outline": None}
    glyphs, _ = _item_masks(H, W, probe)
    col_arg = str(a["color"]).strip() if a.get("color") else "auto"
    how = ""
    if col_arg.lower() in ("pick", "ref", "reference", "eyedropper"):
        if p.ref is None:
            raise UsageError("--color pick needs a reference; give a colour", f'pixelcanvas text "{text[:20]}" '
                             f"--box {coords.fmt_box(box)} --color #ffffff")
        sel = glyphs > 0.5
        rgb = tuple(int(v) for v in np.median(p.ref[sel], axis=0)) if sel.any() else (0, 0, 0)
        how = " (picked from the reference under the letters)"
    elif col_arg.lower() == "auto":
        x0, y0, x1, y1 = box_px
        med = np.median(canvas[y0:y1, x0:x1].reshape(-1, 3), axis=0)
        lum = 0.2126 * med[0] + 0.7152 * med[1] + 0.0722 * med[2]
        rgb = (24, 24, 30) if lum > 128 else (246, 246, 246)
        how = " (chosen to read on the canvas under the box)"
    else:
        rgb = None
        if p.ref is None:          # a palette swatch name ("title", "ink", "sky") before colour words
            try:
                from . import draw_plan
                rgb = tuple(int(v) for v in draw_plan.swatch(p, col_arg))
            except Exception:  # noqa: BLE001
                rgb = None
        if rgb is None:
            rgb = coords.parse_colour(col_arg, f'pixelcanvas text "{text[:20]}" --box {coords.fmt_box(box)} --color "#ffffff"')
    if a.get("outline"):
        outline = _parse_outline(a["outline"], rgb)
    item = {"text": text, "box": [round(float(v), 1) for v in box], "font": font_name, "color": coords.hex_colour(rgb),
            "weight": weight, "align": align, "outline": outline, "vertical": vertical, "fit": bool(a.get("fit")),
            "size": lay["size"], "xy": lay["xy"], "max_width": lay["max_width"], "max_height": lay["max_height"]}
    try:
        lt = _fonts()
        miss = lt.missing_glyphs(text, font)
        if miss:
            ctx.notes.append("the font has no glyph for " + "".join(sorted(set(miss)))[:12])
    except Exception:  # noqa: BLE001
        pass
    new_items = items + [item]
    layer = render_items(H, W, new_items)
    region = Region("text", regions_mod.box_mask(H, W, box_px), box_px, "lettering", False)

    def make(arr):
        p.state["texts"] = new_items
        return layer
    label = f"text \"{text[:24]}{'...' if len(text) > 24 else ''}\" size {lay['size']:g}px {item['color']}"
    rep = cli.paint_step(ctx, label=label, layer="text", build=lambda: [], clip=None, area_px=box_px, region=region,
                         pixel_fn=make, key_metric=None)
    rep.lines.insert(1, f"text {len(new_items)}: ink {lay['ink'][0]:g}x{lay['ink'][1]:g} px in a box of "
                        f"{box_px[2] - box_px[0]}x{box_px[3] - box_px[1]} px{how}")
    return rep


def cmd_fonts(ctx, args: List[str]):
    from .cli import Report, Spec, parse
    a = parse(args, Spec("pixelcanvas fonts serif", ["pattern*"]))
    lt = _fonts()
    rep = Report()
    pat = str(a.get("pattern") or "")
    found = lt.list_fonts(pat)
    stems = [s for s, _ in found]
    rep.add("font styles you can name:  sans  serif  slab  mono  display  condensed  rounded  handwriting  script  "
            "cjk  cjk-serif   (add bold / black / light / italic as words, or use --weight)")
    try:
        lat = lt.find_font(None)
        cjk = lt.find_font(None, text="中文")
        rep.add(f"default: Latin {os.path.basename(lat) if lat else 'built-in'}   CJK "
                f"{os.path.basename(cjk) if cjk else 'none found'}")
    except Exception:  # noqa: BLE001
        pass
    if pat:
        rep.add(f"{len(stems)} installed font file(s) matching {pat!r}:")
    else:
        rep.add(f"{len(stems)} installed font file(s); narrow the list with  pixelcanvas fonts <word>:")
    shown = stems[:40]
    line = ""
    for s in shown:
        if len(line) + len(s) + 2 > 110:
            rep.add("  " + line.rstrip(", "))
            line = ""
        line += s + ", "
    if line:
        rep.add("  " + line.rstrip(", "))
    if len(stems) > len(shown):
        rep.add(f"  ... {len(stems) - len(shown)} more")
    ex = shown[0] if shown else "serif"
    rep.next = f'pixelcanvas text "TITLE" --box 100,50,900,150 --font {ex}'
    return rep
