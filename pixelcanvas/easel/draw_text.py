"""Lettering defaults and a fit check for ``easel text`` on drawings (no reference, or hand mode).

A title (in a planned text part, or a few words in a wide box) is set on one line fitted to the box; lettering in a
planned text part or on a poster is bold (black for a big / large / bold title); a missing ``--box`` takes the box of
the only planned text part. After the text is set, the new item is rendered on a canvas padded on every side, so ink
that runs past the box or off the canvas (a wrapped line pushed below the edge) is seen and reported as a FIX.
Projects with a reference that are not in hand mode get the plain ``easel text``.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional, Tuple

import numpy as np

from . import coords
from .coords import UsageError

_VALUE = {"box", "font", "color", "weight", "align", "outline", "size", "remove",
          "colour", "bold", "stroke", "at", "area", "rm", "delete", "font-size", "alignment"}
_FLAG = {"vertical", "fit", "list", "ls"}
_CANON = {"colour": "color", "bold": "weight", "stroke": "outline", "at": "box", "area": "box", "rm": "remove",
          "delete": "remove", "ls": "list", "font-size": "size", "alignment": "align"}
_BIG = re.compile(r"\b(big|bigger|large|huge|giant|bold|heavy|massive)\b", re.I)
_TITLE = re.compile(r"\b(title|headline|heading|lettering|logo|wordmark|name)\b", re.I)


def _scan(args: List[str]) -> Tuple[dict, List[str]]:
    """The options given (canonical names -> value / True) and the positional words."""
    opts, pos = {}, []
    i = 0
    while i < len(args):
        t = str(args[i])
        if t.startswith("--") and len(t) > 2:
            name, eq, val = t[2:].partition("=")
            name = name.lower()
            key = _CANON.get(name, name)
            if name in _VALUE and not eq:
                val = str(args[i + 1]) if i + 1 < len(args) else ""
                i += 1
            opts[key] = val if (eq or name in _VALUE) else True
        else:
            pos.append(t)
        i += 1
    return opts, pos


def _text_parts(p) -> List[Tuple[str, list]]:
    from . import workflow_draw
    parts = (p.state.get("draw") or {}).get("parts") or {}
    out = []
    for name, d in parts.items():
        if workflow_draw.is_text(str(name)) and isinstance(d, dict) and d.get("box"):
            out.append((str(name), [float(v) for v in d["box"]]))
    return out


def _inside(box, part) -> float:
    """Share of ``box`` that lies inside ``part`` (unit boxes)."""
    ix = max(0.0, min(box[2], part[2]) - max(box[0], part[0]))
    iy = max(0.0, min(box[3], part[3]) - max(box[1], part[1]))
    area = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
    return ix * iy / area


def overflow(H: int, W: int, it: dict) -> Optional[Tuple[str, float, bool]]:
    """Where the item's ink runs past its box or the canvas: (side, edge in px, off the canvas) or None.

    The item is rendered on a canvas padded on every side, so lines pushed past the picture edge are measured too."""
    from . import text as text_mod
    bx = coords.box_to_px(it["box"], W, H)
    pad = int(max(bx[3] - bx[1], bx[2] - bx[0], float(it.get("size") or 0) * 2.0)) + 8
    probe = dict(it)
    probe["xy"] = [float(it["xy"][0]) + pad, float(it["xy"][1]) + pad]
    fill, out = text_mod._item_masks(H + 2 * pad, W + 2 * pad, probe)
    m = out if out is not None else fill
    ink = text_mod._ink_box(m)
    if ink is None:
        return None
    x0, y0, x1, y1 = ink[0] - pad, ink[1] - pad, ink[2] - pad, ink[3] - pad
    tol = 3.0 + (float(it["outline"][0]) if it.get("outline") else 0.0)
    over = {"top": bx[1] - y0, "bottom": y1 - bx[3], "left": bx[0] - x0, "right": x1 - bx[2]}
    edge = {"top": y0, "bottom": y1, "left": x0, "right": x1}
    off = {"top": y0 < -1, "bottom": y1 > H + 1, "left": x0 < -1, "right": x1 > W + 1}
    side = max(over, key=lambda k: (off[k], over[k]))
    if over[side] <= tol and not off[side]:
        return None
    return side, float(edge[side]), bool(off[side])


def _fix_line(n: int, it: dict, H: int, W: int, side: str, edge: float, off: bool) -> Tuple[str, str]:
    axis = "y" if side in ("top", "bottom") else "x"
    units = int(round(edge * coords.UNITS / (H if axis == "y" else W)))
    where = {"top": "above the canvas edge", "bottom": "below the canvas edge", "left": "past the left canvas edge",
             "right": "past the right canvas edge"}[side]
    words = it["text"].replace("\n", " ")
    words = words if len(words) <= 40 else words[:37] + "..."
    box = coords.fmt_box(it["box"])
    again = f'easel text "{words}" --box {box} --fit'
    line = (f'FIX: text {n}: "{words}" did not fit: the lettering runs past the box at the {side} '
            f'(to {axis}={units}{", " + where if off else ""}); make the box bigger or the text shorter:  '
            f"easel text --remove {n}, then {again}")
    return line, f"easel text --remove {n}"


def wrap_text(orig: Callable) -> Callable:
    from . import cli_draw
    if getattr(orig, "_draw_text", False):          # already wrapped: never stack the defaults twice
        return orig

    def text(ctx, args):
        if not cli_draw._noref(ctx):
            return orig(ctx, args)
        args = [str(a) for a in args]
        opts, pos = _scan(args)
        if (any(t in ("--help", "-h", "-?", "help") for t in args) or opts.get("list") or opts.get("remove")
                or not pos):
            return orig(ctx, args)
        p = ctx.project
        H, W = p.H, p.W
        words = " ".join(pos)
        added: List[str] = []
        why: List[str] = []
        planned = _text_parts(p)
        extra: List[str] = []
        if not opts.get("box") and len(planned) == 1:
            extra += ["--box", coords.fmt_box(planned[0][1])]
            opts["box"] = coords.fmt_box(planned[0][1])
            why.append(f"box {opts['box']} of the planned part {planned[0][0]}")
        part = None
        box = None
        if opts.get("box"):
            try:
                box = coords.parse_box(opts["box"], [], f'easel text "{words[:20]}" --box 100,50,900,150')
            except UsageError:
                box = None
        if box is not None:
            best = max(planned, key=lambda pb: _inside(box, pb[1]), default=None)
            if best is not None and _inside(box, best[1]) >= 0.6:
                part = best[0]
            bx = coords.box_to_px(box, W, H)
            bw, bh = bx[2] - bx[0], bx[3] - bx[1]
            short_wide = len(words.split()) <= 4 and bw >= 2.5 * bh
            breaks = "\n" in words or "\\n" in words
            if (part or short_wide) and not opts.get("size") and not breaks and not opts.get("vertical"):
                if not opts.get("fit"):
                    extra.append("--fit")
                    added.append("one line fitted to the box")
            if not opts.get("weight") and (part or p.state.get("style") == "poster"):
                from . import text as text_mod
                req = str(p.state.get("request") or "")
                big = "black" in text_mod.WEIGHTS and _BIG.search(req) and _TITLE.search(req)
                w = "black" if big else "bold"
                extra += ["--weight", w]
                added.append(f"weight {w}")
        if part:
            why.append(f"planned part {part}")
        elif added and p.state.get("style") == "poster":
            why.append("poster")
        n_before = len(p.state.get("texts") or [])
        try:
            rep = orig(ctx, args + extra)
        except UsageError:
            if "--fit" not in extra:
                raise
            extra = [e for e in extra if e != "--fit"]          # too long for one line: let it wrap
            added = [a for a in added if a != "one line fitted to the box"]
            rep = orig(ctx, args + extra)
        if added or why:
            head = "title" if "one line fitted to the box" in added else "lettering"
            msg = f"{head}: " + ", ".join(added or ["defaults"]) + (f" ({'; '.join(why)})" if why else "")
            rep.lines.insert(min(1, len(rep.lines)), msg)
        if not getattr(rep, "mutated", False):
            return rep
        p = ctx.project
        items = p.state.get("texts") or []
        if len(items) <= n_before:
            return rep
        n = len(items)
        try:
            hit = overflow(p.H, p.W, items[-1])
        except UsageError:
            hit = None
        if hit:
            line, nxt = _fix_line(n, items[-1], p.H, p.W, *hit)
            rep.lines.insert(min(2, len(rep.lines)), line)
            rep.next = nxt
        return rep

    text.__name__ = "text"
    text._draw_text = True  # type: ignore[attr-defined]
    return text


WRAP = {"text": wrap_text}
