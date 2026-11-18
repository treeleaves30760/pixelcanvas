"""Coordinates and lenient argument parsing for easel.

Every conversion between the model's units and canvas pixels lives here, so the convention can change in one place.

Convention (DESIGN.md section 5): integers 0-1000 on each axis of the canvas, x right, y down. A point in units
maps to the continuous pixel coordinate ``u / 1000 * side``; a box ``x0,y0,x1,y1`` maps to the pixel slice
``[floor(x0*W/1000), ceil(x1*W/1000))``.

Parsers are lenient: ``x,y``, ``x y``, ``(x, y)``, ``[x, y]``, JSON lists, percentages (``50%`` = 500), fractions
(all values in 0..1 with a decimal point = share of the canvas) and floats are accepted. Out-of-range values are
clamped and a note is added instead of failing.
"""
from __future__ import annotations

import math
import re
from typing import List, Optional, Sequence, Tuple

UNITS = 1000

_NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?%?")


class UsageError(Exception):
    """A user-facing argument error. ``example`` is a corrected command shown to the model."""

    def __init__(self, message: str, example: str = ""):
        super().__init__(message)
        self.message = message
        self.example = example


# ----------------------------------------------------------------------------------------------- numbers

def _numbers(text) -> List[Tuple[float, bool, bool]]:
    """All numbers in ``text``: (value, was_percent, had_decimal_point)."""
    if isinstance(text, (list, tuple)):
        out = []
        for t in text:
            out.extend(_numbers(t))
        return out
    if isinstance(text, (int, float)):
        return [(float(text), False, isinstance(text, float) and not float(text).is_integer())]
    s = str(text)
    out = []
    for m in _NUM.finditer(s):
        tok = m.group(0)
        pct = tok.endswith("%")
        if pct:
            tok = tok[:-1]
        out.append((float(tok), pct, "." in tok))
    return out


def _to_units(nums, notes: Optional[list], what: str) -> List[float]:
    """Convert raw parsed numbers to 0-1000 units (percent and fraction rules), clamp with notes."""
    vals = []
    fractions = bool(nums) and all((not p) and 0.0 <= v <= 1.0 for v, p, _ in nums) and any(d for _, p, d in nums)
    if fractions and notes is not None:
        notes.append(f"read {what} as fractions of the canvas (x1000)")
    for v, pct, _ in nums:
        if pct:
            v = v * UNITS / 100.0
        elif fractions:
            v = v * UNITS
        vals.append(v)
    clamped = []
    bad = [v for v in vals if v < 0 or v > UNITS]
    if bad and notes is not None:
        worst = max(bad, key=lambda b: abs(b - UNITS / 2))
        notes.append(f"{_fmt(worst)} is outside 0-1000 (coordinates are 0-1000 canvas units, not pixels); clamped")
    for v in vals:
        clamped.append(min(float(UNITS), max(0.0, v)))
    return clamped


def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.1f}"


# ----------------------------------------------------------------------------------------------- parsers

def parse_point(text, notes: Optional[list] = None, example: str = "easel pick 500,400") -> Tuple[float, float]:
    """``"x,y"`` / ``"x y"`` / ``"(x, y)"`` / ``[x, y]`` -> (x, y) in units."""
    nums = _numbers(text)
    if len(nums) != 2:
        raise UsageError(f"a point needs 2 numbers (x,y in 0-1000), got {len(nums)} from {_short(text)}", example)
    x, y = _to_units(nums, notes, "the point")
    return x, y


def parse_points(text, notes: Optional[list] = None, min_points: int = 1,
                 example: str = 'easel stroke "100,200 300,250 500,240"') -> List[Tuple[float, float]]:
    """A path: ``"x,y x,y ..."``, ``"(x,y), (x,y)"``, ``[[x,y],[x,y]]`` -> list of (x, y) in units."""
    nums = _numbers(text)
    if len(nums) % 2:
        raise UsageError(f"a path needs pairs of numbers (x,y x,y ...), got an odd count ({len(nums)}) "
                         f"from {_short(text)}", example)
    if len(nums) // 2 < min_points:
        raise UsageError(f"need at least {min_points} points (x,y in 0-1000), got {len(nums) // 2}", example)
    vals = _to_units(nums, notes, "the path")
    return [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]


def parse_box(text, notes: Optional[list] = None, example: str = "easel look 200,100,600,500") -> Tuple[float, ...]:
    """``"x0,y0,x1,y1"`` (any brackets / separators) -> box in units with x0 < x1, y0 < y1."""
    nums = _numbers(text)
    if len(nums) != 4:
        raise UsageError(f"a box needs 4 numbers x0,y0,x1,y1 (0-1000), got {len(nums)} from {_short(text)}", example)
    x0, y0, x1, y1 = _to_units(nums, notes, "the box")
    if x0 > x1 or y0 > y1:
        if notes is not None:
            notes.append("box corners were swapped; reordered to x0,y0,x1,y1")
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
    if x1 - x0 < 1 or y1 - y0 < 1:
        raise UsageError(f"the box {_fmt(x0)},{_fmt(y0)},{_fmt(x1)},{_fmt(y1)} has no area "
                         "(x1 must be > x0 and y1 > y0)", example)
    return x0, y0, x1, y1


def looks_like_box(text) -> bool:
    return isinstance(text, str) and len(_numbers(text)) == 4 and not re.search(r"[a-zA-Z]{2,}", text)


def looks_like_point(text) -> bool:
    return isinstance(text, str) and len(_numbers(text)) == 2 and not re.search(r"[a-zA-Z]{2,}", text)


def parse_float(text, name: str, lo: Optional[float] = None, hi: Optional[float] = None,
                notes: Optional[list] = None, example: str = "") -> float:
    nums = _numbers(text)
    if len(nums) != 1:
        raise UsageError(f"--{name} needs one number, got {_short(text)}", example)
    v, pct, _ = nums[0]
    if pct:
        v = v / 100.0
    if lo is not None and v < lo:
        if notes is not None:
            notes.append(f"--{name} {_fmt(v)} below {_fmt(lo)}; using {_fmt(lo)}")
        v = lo
    if hi is not None and v > hi:
        if notes is not None:
            notes.append(f"--{name} {_fmt(v)} above {_fmt(hi)}; using {_fmt(hi)}")
        v = hi
    return float(v)


def parse_int(text, name: str, lo: Optional[int] = None, hi: Optional[int] = None,
              notes: Optional[list] = None, example: str = "") -> int:
    return int(round(parse_float(text, name, lo, hi, notes, example)))


def parse_int_list(text, example: str = "easel region add hair --marks 3,4,9") -> List[int]:
    nums = _numbers(text)
    if not nums:
        raise UsageError(f"expected numbers like 3,4,9, got {_short(text)}", example)
    # ranges "3-7"
    s = str(text) if not isinstance(text, (list, tuple)) else " ".join(map(str, text))
    out: List[int] = []
    for m in re.finditer(r"(\d+)\s*(?:-|\.\.|to)\s*(\d+)|(\d+)", s):
        if m.group(1):
            a, b = int(m.group(1)), int(m.group(2))
            out.extend(range(min(a, b), max(a, b) + 1))
        else:
            out.append(int(m.group(3)))
    seen, uniq = set(), []
    for n in out:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def _short(text, n: int = 40) -> str:
    s = text if isinstance(text, str) else repr(text)
    s = s.replace("\n", " ")
    return '"' + (s if len(s) <= n else s[: n - 3] + "...") + '"'


# ----------------------------------------------------------------------------------------------- conversions

def unit_to_px(u: float, side: int) -> float:
    """Continuous pixel coordinate of a unit value on an axis of ``side`` pixels."""
    return float(u) * side / UNITS


def px_to_unit(p: float, side: int) -> int:
    return int(round(float(p) * UNITS / max(1, side)))


def point_to_px(pt: Sequence[float], W: int, H: int) -> Tuple[float, float]:
    return unit_to_px(pt[0], W), unit_to_px(pt[1], H)


def points_to_px(pts, W: int, H: int):
    import numpy as np
    a = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    out = a.copy()
    out[:, 0] = a[:, 0] * W / UNITS
    out[:, 1] = a[:, 1] * H / UNITS
    # keep strictly inside the canvas
    out[:, 0] = np.clip(out[:, 0], 0.0, W - 0.001)
    out[:, 1] = np.clip(out[:, 1], 0.0, H - 0.001)
    return out


def box_to_px(box: Sequence[float], W: int, H: int) -> Tuple[int, int, int, int]:
    """Unit box -> integer pixel box (x0, y0, x1, y1), x1/y1 exclusive, at least 1 px each way."""
    x0 = int(math.floor(box[0] * W / UNITS))
    y0 = int(math.floor(box[1] * H / UNITS))
    x1 = int(math.ceil(box[2] * W / UNITS))
    y1 = int(math.ceil(box[3] * H / UNITS))
    x0 = min(max(0, x0), W - 1)
    y0 = min(max(0, y0), H - 1)
    x1 = min(max(x0 + 1, x1), W)
    y1 = min(max(y0 + 1, y1), H)
    return x0, y0, x1, y1


def px_box_to_units(box: Sequence[float], W: int, H: int) -> Tuple[int, int, int, int]:
    """Pixel box (x1/y1 exclusive) -> unit box (integers)."""
    return (px_to_unit(box[0], W), px_to_unit(box[1], H), px_to_unit(box[2], W), px_to_unit(box[3], H))


def fmt_box(box_units: Sequence[float]) -> str:
    return ",".join(str(int(round(v))) for v in box_units)


def fmt_point(pt_units: Sequence[float]) -> str:
    return ",".join(str(int(round(v))) for v in pt_units)


def mask_bbox_px(mask) -> Optional[Tuple[int, int, int, int]]:
    import numpy as np
    m = np.asarray(mask) > 0
    ys = np.flatnonzero(m.any(axis=1))
    if ys.size == 0:
        return None
    xs = np.flatnonzero(m.any(axis=0))
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1


def pad_box_px(box, W: int, H: int, frac: float = 0.05, min_side: int = 24) -> Tuple[int, int, int, int]:
    """Pad a pixel box by ``frac`` of its size (at least a few px) and make it at least ``min_side`` px wide."""
    x0, y0, x1, y1 = box
    pw = max(2, int(round((x1 - x0) * frac)))
    ph = max(2, int(round((y1 - y0) * frac)))
    x0, y0, x1, y1 = x0 - pw, y0 - ph, x1 + pw, y1 + ph
    if x1 - x0 < min_side:
        c = (x0 + x1) / 2
        x0, x1 = c - min_side / 2, c + min_side / 2
    if y1 - y0 < min_side:
        c = (y0 + y1) / 2
        y0, y1 = c - min_side / 2, c + min_side / 2
    x0, y0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
    x1, y1 = min(W, int(math.ceil(x1))), min(H, int(math.ceil(y1)))
    return x0, y0, x1, y1


def union_box(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


# ----------------------------------------------------------------------------------------------- colours

def parse_colour(text, example: str = "--color #c04030") -> Tuple[int, int, int]:
    """``#rrggbb``, ``rrggbb``, ``#rgb``, CSS / plain names (``red``, ``dark blue``, ``skin``), ``rgb(1,2,3)``,
    ``1,2,3`` -> (r, g, b) 0..255."""
    from PIL import ImageColor
    s = str(text).strip().strip("'\"").lower()
    if not s:
        raise UsageError("empty colour", example)
    hexs = s.lstrip("#")
    if re.fullmatch(r"[0-9a-f]{6}", hexs) or re.fullmatch(r"[0-9a-f]{3}", hexs):
        if len(hexs) == 3:
            hexs = "".join(c * 2 for c in hexs)
        return int(hexs[0:2], 16), int(hexs[2:4], 16), int(hexs[4:6], 16)
    nums = _numbers(s)
    if len(nums) == 3 and not re.search(r"[a-z]{4,}", s.replace("rgb", "")):
        vals = [v for v, _, _ in nums]
        if all(v <= 1.0 for v in vals) and any(d for _, _, d in nums):
            vals = [v * 255 for v in vals]
        return tuple(int(max(0, min(255, round(v)))) for v in vals)  # type: ignore
    name = re.sub(r"[\s_-]+", " ", s).strip()
    if name in _PLAIN_EXTRA:
        return _PLAIN_EXTRA[name]
    for cand in (name.replace(" ", ""), name.replace("grey", "gray").replace(" ", "")):
        try:
            r, g, b = ImageColor.getrgb(cand)[:3]
            return int(r), int(g), int(b)
        except ValueError:
            pass
    m = re.fullmatch(r"(dark|light|pale|deep|bright) (.+)", name)
    if m:
        base = parse_colour(m.group(2), example)
        f = {"dark": 0.6, "deep": 0.7, "light": 1.45, "pale": 1.7, "bright": 1.2}[m.group(1)]
        if f < 1:
            return tuple(int(round(c * f)) for c in base)  # type: ignore
        return tuple(int(round(255 - (255 - c) / f)) for c in base)  # type: ignore
    raise UsageError(f"unknown colour {text!r}: use a hex value like #c04030 or a plain name like red, skin, navy",
                     example)


_PLAIN_EXTRA = {
    "skin": (243, 214, 190), "pale skin": (250, 228, 210), "dark skin": (150, 100, 70), "cream": (250, 240, 215),
    "off white": (245, 242, 232), "ink": (25, 22, 28), "paper": (250, 248, 242), "sky": (135, 190, 235),
    "grass": (90, 150, 60), "blond": (240, 215, 140), "blonde": (240, 215, 140), "auburn": (150, 60, 30),
}


def hex_colour(rgb) -> str:
    r, g, b = (int(round(float(c))) for c in rgb[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def colour_name(rgb) -> str:
    """A plain-words name for an sRGB colour ("dark red", "light grey", "skin")."""
    import colorsys
    r, g, b = (float(c) / 255.0 for c in rgb[:3])
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    hd = h * 360.0
    if v < 0.13:
        return "black"
    if s < 0.12 or (v < 0.3 and s < 0.25):
        if v > 0.93:
            return "white"
        if v > 0.75:
            return "light grey"
        if v > 0.45:
            return "grey"
        return "dark grey"
    hues = [(12, "red"), (40, "orange"), (65, "yellow"), (80, "yellow-green"), (160, "green"), (185, "teal"),
            (200, "cyan"), (225, "sky blue"), (255, "blue"), (285, "purple"), (320, "magenta"), (345, "pink"),
            (361, "red")]
    base = next(nm for lim, nm in hues if hd < lim)
    if base == "sky blue" and v < 0.6:
        base = "blue"
    if base == "orange":
        if v < 0.55:
            return "brown" if v > 0.25 else "dark brown"
        if s < 0.45 and v > 0.8:
            return "skin" if s > 0.12 else "cream"
    if base == "red" and s < 0.5 and v > 0.75:
        base = "pink"
    if base in ("yellow",) and v < 0.55:
        return "olive"
    mod = ""
    if v < 0.4:
        mod = "dark "
    elif s < 0.3 and v > 0.8:
        mod = "pale "
    elif v > 0.85 and s < 0.6:
        mod = "light "
    elif s < 0.35:
        mod = "dull "
    return mod + base
