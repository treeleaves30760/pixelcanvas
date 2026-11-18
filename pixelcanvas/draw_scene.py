"""Painterly scene tools for pictures drawn without a reference: clouds, water reflections with the sun's path,
foliage, grass tufts, atmosphere, glows and wet-ground streaks, a floor with contact shadows, windows.

The model says where (points, a region, a box) and the tool lays the marks: soft dabs, tapered strokes and feathered
fills, each one an op in one undoable step (``cli_draw.draw_step``), clipped to the region the command names. No
object is pre-drawn: every shape comes from the model's numbers, the region's own outline, the plan's light and the
picture already on the canvas. Clouds, glows and the atmosphere stay behind the subjects (every region that is not
the setting: sky, ground, water, walls, hills, bands across the picture); ``--over`` paints over them too.

- ``cloud "x,y x,y" [--size 80]``: soft masses of dabs around the points (or along them, a cloud bank); the side
  toward the light is lit (the underside when the light is low or behind: sunset), soft edges, wisps.
- ``reflect <water> [--horizon y] [--glow x]``: horizontal broken strokes whose colour mirrors the scene above the
  horizon, smaller and fainter toward the horizon; ``--glow`` (or the light's ``at`` point) adds the sun's path.
- ``foliage <tree> [--kind needle|leaf]``: needles droop from the trunk line in dark, mid and lit layers; leaves
  grow in clumps, each lit on its side toward the light; marks poke out of the silhouette.
- ``tufts <ground> [--count n]``: grass tufts of 3-5 blades in the near part of the ground, bigger toward the viewer.
- ``atmosphere [--warm 0.3] [--vignette 0.3]``: warm light toward the light source, cool dark edges (fx layer).
- ``glow "x,y" [--radius 40]``: soft halos around lamps and windows (the lamp's own region glows too);
  ``wet <ground> --from "x,y"``: their streaks on wet ground below them.
- ``ground --y 880``: a floor band and a soft contact shadow under every planned subject that stands on it.
- ``window [--region window] [--box x0,y0,x1,y1] [--panes 2x2]``: frame, panes, glass in the window region's own
  colour with a lighter diagonal sheen; it stays behind every region painted after it (or after the wall it is on).
"""
from __future__ import annotations

import math
import re
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import cli, coords
from . import cli_draw as cd
from . import draw_core as dc
from . import draw_plan, draw_shade, workflow_draw
from .cli import Ctx, Report, Spec, parse
from .coords import UsageError
from .draw_detail import bend_path, painted_mark, stroke_widths
from .ops import Fill, Stroke

USAGE = {
    "cloud": 'pixelcanvas cloud "300,180 420,170 520,190" --size 80',
    "reflect": "pixelcanvas reflect lake --glow 600",
    "foliage": "pixelcanvas foliage pines --kind needle",
    "tufts": "pixelcanvas tufts shore --count 60",
    "atmosphere": "pixelcanvas atmosphere --warm 0.3 --vignette 0.3",
    "glow": 'pixelcanvas glow "300,420 640,400" --radius 40 --color "#ffd27a"',
    "wet": 'pixelcanvas wet street --from "300,420 640,400"',
    "ground": "pixelcanvas ground --y 880",
    "window": "pixelcanvas window --box 600,300,760,520 --panes 2x2",
}


# ============================================================================================ helpers

def _rng(seed) -> np.random.Generator:
    return np.random.default_rng(int(seed) & 0x7FFFFFFF)


def _num(a, key, default, lo, hi, ctx: Ctx, usage: str):
    if a.get(key) is None:
        return default
    return coords.parse_float(a[key], key, lo, hi, ctx.notes, usage)


def _len(p, units: float) -> float:
    """A length in 0-1000 units (of the longer side) in pixels."""
    return float(units) * max(p.W, p.H) / 1000.0


def _u(p, px: float, axis: int) -> float:
    return float(px) * 1000.0 / (p.W if axis == 0 else p.H)


def _layer(p, a, default: str) -> str:
    return p.layer_entry(a["layer"])["name"] if a.get("layer") else default


def _colour(p, text, usage: str):
    return draw_plan.swatch(p, text, usage)


def _palette(p, *names):
    sw = dc.dstate(p)["palette"]["swatches"]
    for n in names:
        if n in sw:
            return dc.from_hex(sw[n])
    return None


def _toward(h: float, target: float, k: float) -> float:
    dh = ((target - h + 540) % 360) - 180
    return (h + k * dh) % 360


def _mix(a, b, k: float) -> np.ndarray:
    return (1 - k) * np.asarray(a, float) + k * np.asarray(b, float)


def _rgb(c) -> Tuple[int, int, int]:
    return tuple(int(v) for v in np.clip(np.rint(np.asarray(c, float)), 0, 255))  # type: ignore


def _box_colour(img: np.ndarray, x: float, y: float, r: float) -> np.ndarray:
    """Median colour of the picture in a box of half-size ``r`` around (x, y)."""
    H, W = img.shape[:2]
    r = max(1, int(r))
    xi, yi = min(max(int(x), 0), W - 1), min(max(int(y), 0), H - 1)
    sub = img[max(0, yi - r):yi + r + 1, max(0, xi - r):xi + r + 1, :3].reshape(-1, 3)
    return np.median(sub, axis=0)


def _source_colour(img: np.ndarray, at, reach: float, above: Optional[float] = None,
                   fallback=(255, 200, 140)) -> Tuple[int, int, int]:
    """The light's own colour: the brightest tenth of the picture around the source (only above the row ``above``
    when given: the sky, not its reflection), made bright and warm. A sun set behind mountains takes the colour of
    the sky beside it, not of the rock in front of it."""
    H, W = img.shape[:2]
    x0, x1 = int(max(0, at[0] - reach)), int(min(W, at[0] + reach + 1))
    y0, y1 = int(max(0, at[1] - reach)), int(min(H, at[1] + reach + 1))
    if above is not None:
        y1 = min(y1, int(above))
        if y1 - y0 < 2:
            y0 = max(0, y1 - int(2 * reach) - 2)
    sub = img[y0:y1, x0:x1, :3].reshape(-1, 3).astype(float)
    if len(sub) < 4:
        return tuple(fallback)  # type: ignore
    lum = sub @ np.array([0.2126, 0.7152, 0.0722])
    col = sub[lum >= np.quantile(lum, 0.9)].mean(0)
    L, C, h = dc.rgb_to_lch(col)
    return dc.lch_to_rgb(max(L, 86.0), float(np.clip(C, 30.0, 70.0)), _toward(h, 70, 0.3) if C > 6 else 75.0)


def _light(p, text: Optional[str] = None) -> dict:
    """The plan's light (or ``text``): unit vector toward it, its 2D screen direction, the source point in px."""
    lt = dict(dc.dstate(p)["light"])
    if text:
        deg, z, _ = dc.parse_light(text)
        lt = {"deg": deg, "z": z}
    vec = dc.light_vec({"light": lt})
    at = lt.get("at")
    at_px = dc.u2px(p, [at])[0] if at else None
    v2 = np.array(vec[:2], float)
    n = float(np.hypot(*v2))
    v2 = v2 / n if n > 1e-6 else np.array([0.0, -1.0])
    return {"vec": vec, "dir": v2, "at": at_px, "z": float(lt.get("z", 0.6)), "deg": float(lt.get("deg", 135.0))}


def _toward_light(L: dict, pt) -> np.ndarray:
    """Unit 2D direction from ``pt`` toward the light (the source point when the plan has one)."""
    if L["at"] is not None:
        v = np.asarray(L["at"], float) - np.asarray(pt, float)
        n = float(np.hypot(*v))
        if n > 1e-3:
            return v / n
    return L["dir"]


def _keep_marks(ops) -> list:
    """Only marks a painter could make (DESIGN.md 10.1): a stroke at least 4 widths long or a dab of 3 px or more,
    never under 1 px; a fill polygon with 3+ points."""
    out = []
    for op in ops:
        if isinstance(op, Stroke):
            w = np.asarray(op.width, float)
            if float(w.min()) < 1.0 - 1e-6:
                continue
            if len(op.pts) == 1 and float(w.max()) < 3.0:
                continue
            if len(op.pts) > 1 and dc.path_length(op.pts) < 4.0 * float(w.mean()) and float(w.mean()) < 3.0:
                continue
        elif isinstance(op, Fill) and op.polygons is not None:
            if not op.polygons or any(len(q) < 3 for q in op.polygons):
                continue
        out.append(op)
    return out


def _dab(pt, width: float, rgb, opacity: float, seed: int, brush: str = "soft") -> Stroke:
    return Stroke(pts=np.array([pt], np.float32), width=max(3.0, float(width)), color=_rgb(rgb),
                  opacity=float(np.clip(opacity, 0.02, 1.0)), brush=brush, seed=int(seed))


def _mark(pts, width: float, rgb, opacity: float, seed: int, brush: str, land: float = 0.5, lift: float = 0.14,
          own: bool = False) -> Optional[Stroke]:
    """One tapered stroke along ``pts``; a short one is kept only as a dab of 3 px or more (DESIGN.md 10.1)."""
    path = dc.resample(np.asarray(pts, float), 1.25)
    if len(path) < 2:
        return None
    w = stroke_widths(path, width, land, lift)
    if dc.path_length(path) < 4.0 * float(w.mean()) and float(w.mean()) < 3.0:
        return None
    return Stroke(pts=path.astype(np.float32), width=w, color=_rgb(rgb), opacity=float(np.clip(opacity, 0.02, 1.0)),
                  brush=brush, seed=int(seed), own=own)


def _soft_mark(pts, width: float, rgb, opacity: float, seed: int, tip: str = "flat", land: float = 0.5,
               lift: float = 0.14, own: bool = False) -> list:
    """A painted mark (``draw_detail.painted_mark``): a soft body with the bristles' streaks inside it."""
    m = _mark(pts, width, rgb, opacity, seed, tip, land, lift, own)
    if m is None:
        return []
    return painted_mark(m.pts, m.width, m.color, m.opacity, m.seed, tip, own)


def _region_list(ctx: Ctx, spec: str, usage: str) -> list:
    return [cd.region_or_part(ctx, s, usage) for s in re.split(r"[,\s]+", str(spec)) if s]


def _fmt_u(p, pt) -> str:
    return dc.fmt_pt(p, pt)


def _clip_rect(poly, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """A convex polygon cut to an axis-aligned rectangle (Sutherland-Hodgman)."""
    pts = [tuple(map(float, q)) for q in np.asarray(poly, float)]

    def cut(pts, inside, inter):
        out = []
        for i in range(len(pts)):
            cur, prev = pts[i], pts[i - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(inter(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(inter(prev, cur))
        return out

    def at_x(a, b, x):
        t = (x - a[0]) / ((b[0] - a[0]) or 1e-9)
        return (x, a[1] + t * (b[1] - a[1]))

    def at_y(a, b, y):
        t = (y - a[1]) / ((b[1] - a[1]) or 1e-9)
        return (a[0] + t * (b[0] - a[0]), y)

    for inside, inter in ((lambda q: q[0] >= x0, lambda a, b: at_x(a, b, x0)),
                          (lambda q: q[0] <= x1, lambda a, b: at_x(a, b, x1)),
                          (lambda q: q[1] >= y0, lambda a, b: at_y(a, b, y0)),
                          (lambda q: q[1] <= y1, lambda a, b: at_y(a, b, y1))):
        if len(pts) < 3:
            break
        pts = cut(pts, inside, inter)
    return np.asarray(pts, np.float32).reshape(-1, 2)


def _rect(x0, y0, x1, y1) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)


# ============================================================================================ what stays in front

SETTING_WORDS = ("wall", "hill", "mountain", "back", "backdrop", "bg", "room", "street", "road", "horizon",
                 "distance", "land", "landscape", "paper", "haze", "fog", "mist", "ceiling", "canvas")
WINDOW_WORDS = ("window", "glass", "pane", "frame")


def _named(name: str, words) -> bool:
    """True when a word of ``name`` (split at - _ digits, plural s dropped) is one of ``words``."""
    for t in re.split(r"[^a-z]+", str(name).lower()):
        if t and (t in words or (t.endswith("s") and t[:-1] in words)):
            return True
    return False


def _region_mask(p, name: str) -> Optional[np.ndarray]:
    try:
        m = p.region_mask(name)
    except OSError:
        return None
    return m if m.shape == (p.H, p.W) else None


def _is_setting(p, name: str) -> bool:
    """The setting, not a subject: sky, ground, water, walls, hills (by name), or a band across the whole width that
    is no planned subject or part of one."""
    if workflow_draw.is_ground(name.lower()) or _named(name, SETTING_WORDS):
        return True
    d = dc.dstate(p)
    if name in d.get("subjects", {}) or d.get("parts", {}).get(name, {}).get("of") in d.get("subjects", {}):
        return False
    m = _region_mask(p, name)
    if m is None:
        return True
    cols = np.flatnonzero(m.any(0))
    return len(cols) > 0 and cols[-1] + 1 - cols[0] >= 0.9 * p.W


def _subjects(p, skip=()) -> Tuple[np.ndarray, List[str]]:
    """Every subject region (everything that is not the setting) except ``skip``: what clouds, glows and the
    atmosphere stay behind unless --over."""
    m = np.zeros((p.H, p.W), bool)
    names = []
    for n in p.region_names():
        if n in skip or _is_setting(p, n):
            continue
        mk = _region_mask(p, n)
        if mk is not None and mk.any():
            m |= mk
            names.append(n)
    return m, names


def _later_regions(p, name: str, skip_words=()) -> List[str]:
    """Regions made after ``name``: bands and regions painted later are in front of it."""
    names = p.region_names()
    if name not in names:
        return []
    return [n for n in names[names.index(name) + 1:] if not (skip_words and _named(n, skip_words))]


def _ops_box(ops, W: int, H: int) -> Optional[Tuple[int, int, int, int]]:
    """The pixel box the ops can touch (marks with their width, fills with their feather), cut to the canvas."""
    x0 = y0 = math.inf
    x1 = y1 = -math.inf
    for op in ops:
        if isinstance(op, Stroke):
            pts, pad = np.asarray(op.pts, float).reshape(-1, 2), float(np.max(op.width))
        elif isinstance(op, Fill) and op.polygons:
            pts, pad = np.concatenate([np.asarray(q, float).reshape(-1, 2) for q in op.polygons]), float(op.feather)
        else:
            continue
        x0, y0 = min(x0, pts[:, 0].min() - pad), min(y0, pts[:, 1].min() - pad)
        x1, y1 = max(x1, pts[:, 0].max() + pad), max(y1, pts[:, 1].max() + pad)
    if not math.isfinite(x0):
        return None
    b = (int(max(0, x0)), int(max(0, y0)), int(min(W, math.ceil(x1))), int(min(H, math.ceil(y1))))
    return b if b[2] > b[0] and b[3] > b[1] else None


def _in_the_way(p, names, ops) -> List[str]:
    """Of the regions ``names``, those the ops would reach."""
    b = _ops_box(ops, p.W, p.H)
    if b is None:
        return []
    out = []
    for n in names:
        mk = _region_mask(p, n)
        if mk is not None and mk[b[1]:b[3], b[0]:b[2]].any():
            out.append(n)
    return out


def _behind_line(names: List[str]) -> str:
    return (f"stays behind {', '.join(names)}: subjects are never covered (--over paints over them too)"
            if names else "")


# ============================================================================================ cloud

def _cloud_groups(pts: np.ndarray, size: float) -> List[np.ndarray]:
    """Points closer than 3 cloud sizes join into one cloud along them; a farther point starts a new cloud."""
    groups = [[pts[0]]]
    for q in pts[1:]:
        if float(np.hypot(*(q - groups[-1][-1]))) <= 3.0 * size:
            groups[-1].append(q)
        else:
            groups.append([q])
    return [np.asarray(g, float) for g in groups]


def _cloud_puffs(pts: np.ndarray, size: float, kind: str, rng) -> Tuple[np.ndarray, np.ndarray]:
    """(x, y, r) of the puffs of one cloud, and its centre line. A heap sits on a flat base and piles up; a streak
    is long and thin."""
    R = size / 2.0
    if len(pts) == 1:
        half = (1.15 if kind == "heap" else 1.9) * size
        c = pts[0]
        pts = np.array([[c[0] - half, c[1]], [c[0] + half, c[1]]])
    else:
        d0 = pts[0] - pts[1]
        d1 = pts[-1] - pts[-2]
        ext = 0.45 * size
        pts = np.vstack([pts[0] + d0 / (np.hypot(*d0) + 1e-9) * ext, pts[1:-1] if len(pts) > 2 else
                         np.zeros((0, 2)), pts[-1] + d1 / (np.hypot(*d1) + 1e-9) * ext])
    line = dc.resample(dc.catmull_rom(pts, spacing=2.0), max(2.0, (0.32 if kind == "heap" else 0.4) * R))
    n = len(line)
    t = np.linspace(0.0, 1.0, n)
    prof = 0.35 + 0.65 * np.sin(np.pi * t) ** 0.7
    puffs = []
    for q, f in zip(line, prof):
        if kind == "heap":
            r = R * f * rng.uniform(0.6, 1.0)
            base = q[1] + 0.42 * size
            cy = base - r * rng.uniform(0.8, 1.05)
            puffs.append((q[0] + rng.normal(0, 0.15 * R), cy, r))
            if rng.random() < 0.6 * f:
                r2 = r * rng.uniform(0.55, 0.85)
                puffs.append((q[0] + rng.normal(0, 0.3 * R), cy - r * rng.uniform(0.45, 0.85), r2))
        else:
            r = R * f * rng.uniform(0.4, 0.7)
            puffs.append((q[0] + rng.normal(0, 0.2 * R), q[1] + rng.normal(0, 0.2 * r), r))
    return np.asarray(puffs, float), line


def _cloud_colours(p, img, centre, size, low: bool, at_px, given, given_light, usage):
    """(body, shadow, lit, glint) colours of a cloud from the sky behind it and the light."""
    sky = _box_colour(img, centre[0], centre[1], 1.5 * size)
    Ls, Cs, hs = dc.rgb_to_lch(sky)
    if given:
        base = np.asarray(_colour(p, given, usage), float)
    else:
        pal = _palette(p, "cloud", "clouds")
        if pal is not None:
            base = np.asarray(pal, float)
        elif low:
            base = np.asarray(dc.lch_to_rgb(max(24.0, Ls - 18.0), 0.35 * Cs + 3.0, _toward(hs, 300, 0.6)), float)
        else:
            base = np.asarray(dc.lch_to_rgb(min(97.0, Ls + 14.0), 0.25 * Cs + 1.0, hs), float)
    Lb, Cb, hb = dc.rgb_to_lch(base)
    if low:
        shadow = np.asarray(dc.shadow_of(base, 0.6), float)
    else:
        shadow = np.asarray(dc.lch_to_rgb(max(30.0, Lb - 20.0), 0.35 * Cs + 4.0, _toward(hs, 265, 0.35)), float)
    if given_light:
        lit = np.asarray(_colour(p, given_light, usage), float)
    elif low:
        H, W = img.shape[:2]
        if at_px is not None and 0 <= at_px[0] < W and 0 <= at_px[1] < H:
            lit = np.asarray(_source_colour(img, at_px, 0.12 * max(W, H)), float)
        else:
            lit = np.asarray((255, 196, 140), float)
    else:
        lit = np.asarray(dc.lch_to_rgb(98.0, 3.0, 95.0), float)
    glint = np.asarray(dc.light_of(lit, 0.25), float)
    return base, shadow, lit, glint


def _lit_outline(puffs: np.ndarray, ld: np.ndarray, R: float) -> np.ndarray:
    """The cloud's outline on the side toward the light: in each slice across that direction the puff edge that
    reaches farthest toward the light."""
    nv = np.array([-ld[1], ld[0]])
    pts = puffs[:, :2] + ld[None, :] * (0.38 * puffs[:, 2:3])
    a = pts @ nv
    b = pts @ ld
    bins = np.floor((a - a.min()) / max(1e-6, 0.45 * R)).astype(int)
    out = [pts[sel[np.argmax(b[sel])]] for sel in (np.nonzero(bins == k)[0] for k in np.unique(bins))]
    out = np.asarray(out, float)
    return out[np.argsort(out @ nv)]


def cloud_ops(pts_px: np.ndarray, size: float, kind: str, L: dict, colours_fn, opacity: float,
              seed: int) -> Tuple[list, list]:
    """Soft dabs for every cloud, back to front: the body as clusters of dabs (darker on the side away from the
    light), small dabs that bump the outline, the lit side, one soft lit edge along the outline that faces the
    light, then wisps. Returns (ops, [(centre px, puffs, underside lit)])."""
    rng = _rng(seed)
    ops: list = []
    info: list = []
    for g, grp in enumerate(_cloud_groups(pts_px, size)):
        puffs, line = _cloud_puffs(grp, size, kind, rng)
        if not len(puffs):
            continue
        wts = puffs[:, 2] ** 2
        centre = (puffs[:, :2] * wts[:, None]).sum(0) / wts.sum()
        ld = _toward_light(L, centre)
        low = (L["at"] is not None and L["at"][1] > centre[1]) or (L["at"] is None and (L["z"] < 0 or ld[1] > 0.2))
        if low:
            ld = np.array([0.8 * ld[0], abs(ld[1]) + 0.5])
        else:
            ld = np.array([0.8 * ld[0], -abs(ld[1]) - 0.5])
        ld /= np.hypot(*ld)
        base, shadow, lit, glint = colours_fn(centre, low)
        R = size / 2.0
        proj = (puffs[:, :2] - centre) @ ld
        tone = (proj - proj.min()) / max(1e-6, float(np.ptp(proj)))     # 0 on the far side .. 1 toward the light
        sd = seed * 1000 + g * 997
        order = np.argsort(proj)                        # the shadow side first, the lit side on top
        for k in order:                                 # the body: a cluster of soft dabs in every puff
            x, y, r = puffs[k]
            for j in range(3):
                q = np.array([x, y]) + rng.normal(0, 0.25 * r, 2)
                col = _mix(shadow, base, float(np.clip(0.2 + 0.8 * tone[k] + rng.normal(0, 0.08), 0, 1)))
                ops.append(_dab(q, r * rng.uniform(1.0, 1.35), col, opacity * rng.uniform(0.55, 0.75), sd))
                sd += 1
        up = -ld if low else ld                         # the outline away from a heap's flat base bulges
        if kind == "heap":
            far = proj < np.quantile(proj, 0.6) if low else proj > np.quantile(proj, 0.4)
            for k in np.nonzero(far)[0]:
                x, y, r = puffs[k]
                for j in range(2):
                    ang = math.atan2(up[1], up[0]) + rng.normal(0, 0.75)
                    q = np.array([x, y]) + np.array([math.cos(ang), math.sin(ang)]) * 0.72 * r
                    col = _mix(shadow, base, float(np.clip(0.2 + 0.8 * tone[k], 0, 1)))
                    ops.append(_dab(q, 0.55 * r, col, opacity * 0.6, sd))
                    sd += 1
        for k in order:                                 # the lit side
            if tone[k] < 0.45:
                continue
            x, y, r = puffs[k]
            f = (tone[k] - 0.45) / 0.55
            q = np.array([x, y]) + ld * 0.45 * r + rng.normal(0, 0.1 * r, 2)
            ops.append(_dab(q, 0.8 * r, _mix(base, lit, 0.5 + 0.5 * f), opacity * (0.45 + 0.4 * f), sd))
            sd += 1
        rim = _lit_outline(puffs, ld, R)                # the lit edge: a broad soft band on the cloud's edge
        if len(rim) >= 2:
            path = dc.smooth_poly(dc.catmull_rom(rim, spacing=2.0), 4.0)
            m = _mark(path, 0.55 * R, _mix(lit, base, 0.05), opacity * 0.6, sd, "soft", 0.3, 0.1)
            if m is not None:
                ops.append(m)
            n = len(path)                               # and a few brighter touches along it, not one line
            for j in range(int(rng.integers(2, 4))):
                a0 = int(rng.uniform(0.05, 0.7) * n)
                seg = path[a0:a0 + max(3, int(rng.uniform(0.12, 0.3) * n))]
                m = _mark(seg - ld * 0.1 * R, 0.22 * R, glint, opacity * rng.uniform(0.35, 0.55), sd + 1 + j,
                          "soft", 0.3, 0.1)
                if m is not None:
                    ops.append(m)
            sd += 5
        tang = line[-1] - line[0]                       # wisps: thin soft trails off both ends
        tang = tang / (np.hypot(*tang) + 1e-9)
        for end, sgn in ((line[0], -1.0), (line[-1], 1.0)):
            for j in range(int(rng.integers(1, 3))):
                ln = size * rng.uniform(0.8, 1.7)
                ang = math.atan2(tang[1], tang[0]) + rng.normal(0, 0.12)
                dvec = np.array([math.cos(ang), math.sin(ang)]) * sgn
                start = end - dvec * 0.35 * size + np.array([0, rng.normal(0, 0.12 * size)])
                path = bend_path(np.array([start, start + dvec * ln]), rng.uniform(-0.08, 0.08))
                col = _mix(base, lit if low else shadow, rng.uniform(0.2, 0.5))
                m = _mark(path, size * rng.uniform(0.12, 0.22), col, opacity * rng.uniform(0.3, 0.45), sd, "soft",
                          0.35, 0.08)
                sd += 1
                if m is not None:
                    ops.append(m)
        info.append((centre, len(puffs), low))
    return ops, info


def cmd_cloud(ctx: Ctx, args: List[str]) -> Report:
    p = ctx.project
    u = USAGE["cloud"]
    a = parse(args, Spec(u, ["pts*"], {"size": "value", "color": "value", "light-color": "value", "kind": "value",
                                       "region": "value", "opacity": "value", "over": "flag", "layer": "value"},
                         alias={"colour": "color", "light-colour": "light-color", "lit": "light-color",
                                "in": "region", "clip": "region", "shape": "kind", "type": "kind"}, min_pos=0))
    if not a.get("pts"):
        raise UsageError("cloud needs where: one x,y per cloud, or a few x,y along a cloud bank (0-1000)", u)
    pts = dc.u2px(p, coords.parse_points(a["pts"], ctx.notes, 1, u))
    size = _len(p, _num(a, "size", 80, 10, 600, ctx, u))
    kind = str(a.get("kind") or "heap").lower()
    kind = {"puffy": "heap", "cumulus": "heap", "heaped": "heap", "stratus": "streak", "streaks": "streak",
            "flat": "streak", "long": "streak"}.get(kind, kind)
    if kind not in ("heap", "streak"):
        raise UsageError(f"unknown --kind {a['kind']!r}; kinds: heap (piled-up cumulus), streak (long and flat)",
                         'pixelcanvas cloud "300,180 520,190" --kind streak')
    reg = None
    rname = a.get("region")
    if rname is None:
        names = p.region_names()
        sky = "sky" if "sky" in names else next((n for n in names if "sky" in n), None)
        if sky:
            reg = cli.resolve_region(p, sky, ctx.notes, u)
    elif str(rname).strip().lower() not in ("none", "no", "off", "all", "canvas"):
        reg = cd.region_or_part(ctx, rname, u)
    L = _light(p)
    opacity = _num(a, "opacity", 0.9, 0.1, 1, ctx, u)
    img = p.composite_u8("export")
    excl, subj = (None, []) if a.get("over") else _subjects(p, skip={reg.name} if reg is not None else ())
    clip = None if reg is None else reg.mask
    if excl is not None and excl.any():
        clip = (np.ones((p.H, p.W), bool) if clip is None else clip) & ~excl
    lines: List[str] = []

    def build():
        def colours(centre, low):
            return _cloud_colours(p, img, centre, size, low, L["at"], a.get("color"), a.get("light-color"), u)
        ops, info = cloud_ops(pts, size, kind, L, colours, opacity, dc.stable_seed(p, ctx.cmdline, "cloud"))
        for k, (c, n, low) in enumerate(info[:4]):
            lines.append(f"cloud {k + 1}: {n} soft puffs around {_fmt_u(p, c)}, "
                         + ("underside lit (the light is low or behind)" if low else "top lit (the light is above)"))
        if len(info) > 4:
            lines.append(f"... and {len(info) - 4} more clouds")
        if reg is not None:
            lines.append(f"clipped to {reg.name} (--region none: anywhere behind the subjects)")
        lines.append(_behind_line(_in_the_way(p, subj, ops)))
        return _keep_marks(ops)
    return cd.draw_step(ctx, label=f"cloud {len(_cloud_groups(pts, size))} cloud(s) {kind} size {size:.0f}px",
                        layer=_layer(p, a, "paint"), build=build, clip=clip,
                        lines=lines, note="soft cloud masses, lit on the side toward the light")


# ============================================================================================ reflect

def _far_scene(p, img: np.ndarray, h: float, water: str) -> np.ndarray:
    """The picture above the horizon with near things (regions that reach down past the water's edge: trees,
    people on the near shore) replaced by the far scene of the same row: they stand in front of the water and are
    not mirrored in it."""
    H, W = img.shape[:2]
    hi = int(min(H, h + 0.01 * H + 1))
    top = int(max(0, min(H, h)))
    near = np.zeros((H, W), bool)
    for n in p.region_names():
        if n == water or n in ("background", "bg", "sky", "paper"):
            continue
        mk = p.region_mask(n)
        tot = int(mk.sum())
        if tot == 0 or not mk[:top].any():
            continue
        if mk[hi:].sum() > 0.03 * tot:
            near |= mk
    out = img.astype(np.float32).copy()
    xs = np.arange(W)
    for y in np.nonzero(near[:top].any(1))[0]:
        good = ~near[y]
        if good.sum() < 2:
            continue
        gx = xs[good]
        for c in range(3):
            out[y, ~good, c] = np.interp(xs[~good], gx, out[y, good, c])
    return out


def _row_runs(row: np.ndarray, gap: int = 3) -> List[Tuple[float, float]]:
    """The stretches ``[x_lo, x_hi)`` where a mask row is on; gaps narrower than ``gap`` pixels are bridged."""
    cols = np.nonzero(row)[0]
    if not len(cols):
        return []
    cut = np.nonzero(np.diff(cols) > gap)[0]
    starts = np.r_[cols[0], cols[cut + 1]]
    ends = np.r_[cols[cut], cols[-1]] + 1
    return [(float(a), float(b)) for a, b in zip(starts, ends)]


def reflect_ops(mask: np.ndarray, src: np.ndarray, h: float, water_rgb, near_w: float, strength: float,
                glow_x: Optional[float], glow_rgb, seed: int) -> Tuple[list, int, int]:
    """Horizontal broken strokes mirroring ``src`` about the row ``h``; thinner, shorter and fainter toward the
    horizon. With ``glow_x``: the light's path, a column of light dashes widening toward the viewer."""
    rng = _rng(seed)
    H, W = mask.shape
    ys = np.nonzero(mask.any(1))[0]
    y_bot = float(ys.max() + 1)
    span = max(1.0, y_bot - h)
    far_w = max(1.2, 0.22 * near_w)
    water = np.asarray(water_rgb, float)
    ops: list = []
    n_refl = n_path = 0
    y = h + 0.6 * far_w
    sd = seed * 10000
    while y < y_bot:
        t = float(np.clip((y - h) / span, 0, 1))
        th = far_w + (near_w - far_w) * t ** 1.15
        yi = min(H - 1, int(y))
        cover = 0.95 - 0.4 * t
        Lm = th * (11.0 - 5.0 * t)
        # each stretch of water on the row on its own: a curved shore leaves short stretches and bays, and a
        # stroke is never longer than the water it lies on
        for x_lo, x_hi in _row_runs(mask[yi]):
            n = int(cover * (x_hi - x_lo) / Lm) + 1
            for _ in range(n):
                Ld = min(Lm * rng.uniform(0.6, 1.4), x_hi - x_lo)
                x0 = rng.uniform(x_lo - 0.3 * Ld, x_hi - 0.7 * Ld)
                yy = y + rng.normal(0, 0.25 * th)
                xs3 = np.clip(np.array([x0 + 0.2 * Ld, x0 + 0.5 * Ld, x0 + 0.8 * Ld]), 0, W - 1).astype(int)
                ym = int(np.clip(2 * h - yy, 0, max(0, h - 1)))
                col = src[ym, xs3].mean(0)
                col = _mix(col, water, 0.28 + 0.14 * t) * (0.93 + rng.normal(0, 0.02))
                pts = np.array([[x0, yy], [x0 + Ld, yy + rng.normal(0, 0.05 * th)]])
                if rng.random() < 0.5:
                    pts = pts[::-1]
                pts = bend_path(dc.resample(pts, 1.5), rng.uniform(-0.025, 0.025))
                op = strength * (0.55 + 0.4 * t) * rng.uniform(0.85, 1.1)
                mk = _soft_mark(pts, th * rng.uniform(0.75, 1.2), col, op, sd, "flat", rng.uniform(0.3, 0.5),
                                rng.uniform(0.1, 0.2))
                sd += 1
                if mk:
                    ops += mk
                    n_refl += 1
        y += 0.75 * th
    if glow_x is not None:
        lc = np.asarray(glow_rgb, float)
        hi_c = np.asarray(dc.light_of(lc, 0.8), float)
        hw0, hw1 = 0.022 * W, 0.085 * W
        poly = np.array([[glow_x - hw0, h], [glow_x + hw0, h], [glow_x + hw1, y_bot], [glow_x - hw1, y_bot]])
        ops.append(Fill(polygons=[poly.astype(np.float32)], color=_rgb(lc), opacity=0.22 * strength,
                        feather=max(2.0, 0.9 * (hw0 + hw1))))
        y = h + 1.0
        while y < y_bot:
            t = float(np.clip((y - h) / span, 0, 1))
            th = 0.85 * (far_w + (near_w - far_w) * t ** 1.15)
            hw = hw0 + (hw1 - hw0) * t
            Lm = th * (6.0 - 2.0 * t)
            n = int(1.8 * 2 * hw / max(Lm, 1.0)) + 1
            for _ in range(n):
                dx = rng.normal(0, 0.5 * hw)
                if abs(dx) > 1.8 * hw:
                    continue
                fall = math.exp(-0.5 * (dx / (0.6 * hw)) ** 2)
                Ld = Lm * rng.uniform(0.6, 1.3)
                xc = glow_x + dx
                yy = y + rng.normal(0, 0.3 * th)
                pts = np.array([[xc - Ld / 2, yy], [xc + Ld / 2, yy + rng.normal(0, 0.04 * th)]])
                col = _mix(lc, hi_c, 0.7 * fall * rng.uniform(0.5, 1.0))
                op = strength * (0.35 + 0.6 * fall) * (0.7 + 0.3 * t) * rng.uniform(0.85, 1.1)
                m = _mark(dc.resample(pts, 1.25), th * rng.uniform(0.7, 1.1), col, op, sd,
                          "round" if fall > 0.6 and rng.random() < 0.5 else "flat", rng.uniform(0.3, 0.5),
                          rng.uniform(0.1, 0.2))
                sd += 1
                if m is not None:
                    ops.append(m)
                    n_path += 1
            y += 0.9 * th
    return ops, n_refl, n_path


def cmd_reflect(ctx: Ctx, args: List[str]) -> Report:
    p = ctx.project
    u = USAGE["reflect"]
    a = parse(args, Spec(u, ["region*"], {"horizon": "value", "glow": "value", "no-glow": "flag", "size": "value",
                                         "color": "value", "glow-color": "value", "strength": "value",
                                         "layer": "value"},
                         alias={"colour": "color", "sun": "glow", "path": "glow", "no-sun": "no-glow",
                                "y": "horizon", "glow-colour": "glow-color", "opacity": "strength",
                                "amount": "strength"}, min_pos=1))
    r = cd.region_or_part(ctx, a["region"], u)
    m = r.mask
    if m.sum() < 50:
        raise UsageError(f"{r.name} is too small to hold reflections", u)
    if a.get("horizon"):
        h = coords.parse_float(a["horizon"], "horizon", 0, 1000, ctx.notes, u) * p.H / 1000.0
    else:
        cols = m.any(0)
        h = float(np.median(np.argmax(m, axis=0)[cols]))
    ys = np.nonzero(m.any(1))[0]
    if ys.max() + 1 - h < 4:
        raise UsageError(f"no water below the horizon y {_u(p, h, 1):.0f}: give the water's top edge with --horizon",
                         f"pixelcanvas reflect {r.name} --horizon {_u(p, ys.min(), 1):.0f}")
    L = _light(p)
    gx = None
    gtxt = str(a.get("glow") or "").strip().lower()
    if a.get("no-glow") or gtxt in ("none", "no", "off"):
        gx = None
    elif gtxt:
        gx = coords.parse_float(a["glow"], "glow", 0, 1000, ctx.notes, u) * p.W / 1000.0
    elif L["at"] is not None and L["at"][1] < h + 2 and 0 <= L["at"][0] <= p.W:
        gx = float(L["at"][0])
    water = cd.colour_set(p, r, a.get("color"))[0]
    strength = _num(a, "strength", 0.75, 0.05, 1, ctx, u)
    near_w = _len(p, _num(a, "size", 10, 1, 60, ctx, u))
    img = p.composite_u8("export")
    lines: List[str] = []

    def build():
        src = _far_scene(p, img, h, r.name)
        src = cv2.GaussianBlur(src, (0, 0), max(0.8, 0.004 * max(p.W, p.H)))
        if gx is not None:
            if a.get("glow-color"):
                gcol = np.asarray(_colour(p, a["glow-color"], u), float)
            else:
                at = L["at"] if L["at"] is not None and L["at"][1] < h else (gx, max(0.0, h - 0.05 * p.H))
                gcol = np.asarray(_source_colour(img, at, 0.12 * max(p.W, p.H), above=h), float)
        else:
            gcol = None
        ops, n1, n2 = reflect_ops(m, src, h, water, near_w, strength, gx, gcol, dc.stable_seed(p, ctx.cmdline, "reflect"))
        lines.append(f"{n1} strokes mirror the scene above the horizon y {_u(p, h, 1):.0f}: small and faint far "
                     f"away, bigger and more broken toward the viewer")
        if gx is not None:
            lines.append(f"sun path at x {_u(p, gx, 0):.0f}: {n2} light strokes widening toward the viewer, "
                         f"{coords.hex_colour(gcol)}")
        else:
            lines.append("add --glow x to lay the sun's glitter path down the water")
        return _keep_marks(ops)

    def post():
        for key in ("painted", "detailed"):
            cd._mark_list(dc.dstate(p), key, r.name)
        return []
    return cd.draw_step(ctx, label=f"reflect {r.name}", layer=_layer(p, a, "paint"), build=build, clip=m,
                        post=post, lines=lines, area_extra=r.box_px, note="the water mirrors the scene; light path")


# ============================================================================================ foliage

NEEDLE_WORDS = ("pine", "fir", "spruce", "conifer", "evergreen", "cedar", "cypress", "larch", "needle", "hemlock",
                "yew", "juniper")


def _row_axis(m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per row: the middle x of the shape (the trunk line) and its half width, smoothed so zigzag tiers do not
    swing it."""
    from scipy.ndimage import uniform_filter1d
    H, W = m.shape
    ys, xs = np.nonzero(m)
    cnt = np.bincount(ys, minlength=H).astype(float)
    sx = np.bincount(ys, weights=xs, minlength=H)
    lo = np.full(H, np.inf)
    hi = np.full(H, -np.inf)
    np.minimum.at(lo, ys, xs)
    np.maximum.at(hi, ys, xs)
    rows = np.nonzero(cnt > 0)[0]
    axis = np.interp(np.arange(H), rows, sx[rows] / cnt[rows])
    half = np.interp(np.arange(H), rows, (hi[rows] - lo[rows] + 1) / 2.0)
    k = max(1, int(0.03 * (rows.max() - rows.min() + 1)))
    return uniform_filter1d(axis, 2 * k + 1), uniform_filter1d(half, 2 * k + 1)


def _spaced_points(xs, ys, gap: float, rng, limit: int = 6000) -> np.ndarray:
    """Points of the shape at least ``gap`` apart (dart throwing): where leaf clumps sit."""
    order = rng.permutation(len(xs))[:limit]
    out = np.zeros((0, 2))
    for i in order:
        q = np.array([xs[i] + 0.5, ys[i] + 0.5])
        if len(out) == 0 or float(((out - q) ** 2).sum(1).min()) >= gap * gap:
            out = np.vstack([out, q])
    return out


def foliage_ops(mask: np.ndarray, kind: str, light: np.ndarray, colours, S: float, density: float, seed: int,
                lit_k: float = 1.0) -> Tuple[list, int]:
    """Back to front over one tree shape. Needles: dark masses, the silhouette broken by marks that poke out
    (``own``: painted whole past the edge), mid tones on the middle and lit parts, light marks on the side toward
    the light; every mark droops away from the trunk line. Leaves: a dark underlayer in the crown's shadow, then
    clumps of leaf marks (each clump lit on its own side toward the light, shadowed clumps first, marks curving
    round the clump), the silhouette broken by leaves, a few bright touches on the most lit clumps."""
    deep, sh, base, li, hi = (np.asarray(c, float) for c in colours)
    rng = _rng(seed)
    m = np.asarray(mask, bool)
    ys, xs = np.nonzero(m)
    area = len(xs)
    if area < 30:
        return [], 0
    I, (y0, y1, x0, x1) = draw_shade.lighting(m, light, 1.0)
    full = np.zeros(m.shape, np.float32)
    full[y0:y1, x0:x1] = I
    vals = full[ys, xs]
    q40, q72, q90 = np.quantile(vals, [0.4, 0.72, 0.9])
    lo, top = np.quantile(vals, [0.05, 0.97])
    axis, half = _row_axis(m)
    ops: list = []
    sd = seed * 10000

    def tone(v):
        t = float(np.clip((v - lo) / max(top - lo, 1e-6), 0, 1))
        if t < 0.35:
            return _mix(deep, sh, t / 0.35)
        if t < 0.7:
            return _mix(sh, base, (t - 0.35) / 0.35)
        return _mix(base, li, (t - 0.7) / 0.3)

    def mark(q, scale, col, op, out_dir=None, own=False, angle=None):
        nonlocal sd
        sd += 1
        col = np.clip(np.asarray(col, float) * (1 + rng.normal(0, 0.035)), 0, 255)
        if kind == "needle":
            yi = int(np.clip(q[1], 0, m.shape[0] - 1))
            dx = q[0] - axis[yi]
            hw = max(1.0, half[yi])
            side = (1.0 if dx >= 0 else -1.0) if abs(dx) > 0.12 * hw else (1.0 if rng.random() < 0.5 else -1.0)
            rel = min(1.0, abs(dx) / hw)
            droop = math.radians(14 + 26 * rel + rng.normal(0, 8))
            d = np.array([side * math.cos(droop), math.sin(droop)])
            if out_dir is not None:
                d = np.asarray(out_dir, float) + np.array([0.0, 0.45])
                d /= np.hypot(*d) + 1e-9
                side = 1.0 if d[0] >= 0 else -1.0
            Lm = S * scale * rng.uniform(1.6, 2.6)
            w = S * scale * rng.uniform(0.26, 0.4)
            pts = np.array([q - d * 0.25 * Lm, q + d * 0.75 * Lm])
            pts = bend_path(dc.resample(pts, 1.25), side * rng.uniform(0.04, 0.12))
            return _mark(pts, w, col, op, sd, "flat", rng.uniform(0.55, 0.75), rng.uniform(0.08, 0.15), own)
        ang = rng.uniform(0, 2 * math.pi) if angle is None else angle
        d = np.array([math.cos(ang), math.sin(ang)])
        if out_dir is not None:
            d = np.asarray(out_dir, float) + rng.normal(0, 0.35, 2)
            d /= np.hypot(*d) + 1e-9
        Lm = S * scale * rng.uniform(0.85, 1.3)
        w = S * scale * rng.uniform(0.45, 0.65)
        if out_dir is not None:
            pts = np.array([q - d * 0.2 * Lm, q + d * 0.8 * Lm])
        else:
            pts = np.array([q - d * 0.5 * Lm, q + d * 0.5 * Lm])
        pts = bend_path(dc.resample(pts, 1.25), rng.uniform(-0.16, 0.16))
        return _soft_mark(pts, w, col, op, sd, "flat", rng.uniform(0.45, 0.65), rng.uniform(0.12, 0.22), own)

    def pick(sel, n):
        ii = np.nonzero(sel)[0]
        if len(ii) == 0 or n <= 0:
            return []
        return rng.choice(ii, size=min(n, len(ii)), replace=False)

    def add(op):
        if isinstance(op, list):
            ops.extend(op)
        elif op is not None:
            ops.append(op)

    def ramp(t):
        t = float(np.clip(t, 0, 1))
        for (t0, c0), (t1, c1) in zip(((0.0, deep), (0.3, sh), (0.55, base), (0.8, li)),
                                      ((0.3, sh), (0.55, base), (0.8, li), (1.0, hi))):
            if t <= t1:
                return _mix(c0, c1, (t - t0) / (t1 - t0))
        return hi

    if kind != "needle":
        return _leaf_clumps(m, xs, ys, full, vals, lo, top, light, S, density, rng, mark, ramp, add, ops, deep,
                            hi, lit_k)

    # 1 dark masses all over
    for i in pick(np.ones(area, bool), int(density * area / (S * S * 2.0)) + 3):
        q = np.array([xs[i] + rng.uniform(0, 1), ys[i] + rng.uniform(0, 1)])
        add(mark(q, 1.25, _mix(deep, sh, rng.uniform(0, 0.4)), rng.uniform(0.78, 0.92)))
    # 2 the silhouette broken by marks that poke out
    n_edge = _edge_marks(m, xs, ys, full, S, density, rng, lambda q, v, out: mark(q, 0.6, tone(v),
                                                                                  rng.uniform(0.75, 0.92),
                                                                                  out_dir=out, own=True), add)
    # 3 mid tones over the middle and lit parts
    sel = vals > q40
    for i in pick(sel, int(density * area / (S * S * 2.8)) + 2):
        q = np.array([xs[i] + rng.uniform(0, 1), ys[i] + rng.uniform(0, 1)])
        add(mark(q, 1.0, _mix(tone(vals[i]), base, 0.35), rng.uniform(0.7, 0.88)))
    # 4 the lit side
    sel = vals > q72
    for i in pick(sel, int(lit_k * density * area / (S * S * 4.5)) + 2):
        q = np.array([xs[i] + rng.uniform(0, 1), ys[i] + rng.uniform(0, 1)])
        col = hi if vals[i] > q90 and rng.random() < 0.5 else li
        add(mark(q, 0.72, _mix(col, base, rng.uniform(0, 0.25)), rng.uniform(0.75, 0.92)))
    return ops, n_edge


def _edge_marks(m, xs, ys, full, S, density, rng, make, add) -> int:
    """Marks on the silhouette band pointing outward (along the outward normal of the shape): ``make(q, v, out)``
    builds one at q with the lighting v there."""
    edge = m & ~dc.erode(m, max(1.0, 0.55 * S))
    pad = int(2 * S) + 3
    cy0, cy1 = max(0, ys.min() - pad), min(m.shape[0], ys.max() + 1 + pad)
    cx0, cx1 = max(0, xs.min() - pad), min(m.shape[1], xs.max() + 1 + pad)
    dt = cv2.distanceTransform(m[cy0:cy1, cx0:cx1].astype(np.uint8), cv2.DIST_L2, 5)
    dt = cv2.GaussianBlur(dt, (0, 0), max(1.0, 0.5 * S))
    gy, gx = np.gradient(dt)
    eys, exs = np.nonzero(edge)
    n_edge = 0
    if len(exs):
        n4 = int(density * len(exs) / (0.45 * S * S)) + 2     # the band is 0.55 S deep: one mark per 0.8 S of edge
        for i in rng.choice(len(exs), size=min(n4, len(exs)), replace=False):
            ey, ex = eys[i], exs[i]
            g = np.array([gx[ey - cy0, ex - cx0], gy[ey - cy0, ex - cx0]])
            ng = float(np.hypot(*g))
            if ng < 1e-6:
                continue
            out = -g / ng
            q = np.array([ex + 0.5, ey + 0.5]) - out * 0.15 * S
            op = make(q, full[ey, ex], out)
            if op:
                add(op)
                n_edge += 1
    return n_edge


def _leaf_clumps(m, xs, ys, full, vals, lo, top, light, S, density, rng, mark, ramp, add, ops, deep, hi,
                 lit_k) -> Tuple[list, int]:
    l2 = np.asarray(light[:2], float)
    nl = float(np.hypot(*l2))
    l2 = l2 / nl if nl > 1e-6 else np.array([0.0, -1.0])
    span = max(top - lo, 1e-6)
    centres = _spaced_points(xs, ys, 1.6 * S, rng)
    tc = np.array([float(np.clip((full[int(c[1]), int(c[0])] - lo) / span, 0, 1)) for c in centres])
    order = np.argsort(tc + rng.normal(0, 0.05, len(tc)))       # shadowed clumps first, lit ones over them
    # 0 a dark underlayer where the crown turns from the light
    for k in order:
        if tc[k] < 0.5:
            add(_dab(centres[k], 2.3 * S, _mix(deep, ramp(0.25), rng.uniform(0, 0.5)), rng.uniform(0.35, 0.55),
                     int(rng.integers(1 << 30))))
    # 1 the clumps: leaves round each clump, each clump lit on its own side toward the light
    H, W = m.shape
    for k in order:
        c = centres[k]
        rc = S * rng.uniform(1.05, 1.5)
        leaves = []
        for j in range(int(round(density * rng.uniform(7, 11)))):
            r = rc * math.sqrt(rng.random())
            a = rng.uniform(0, 2 * math.pi)
            off = np.array([math.cos(a), math.sin(a)]) * r
            q = c + off
            qi = (int(np.clip(q[1], 0, H - 1)), int(np.clip(q[0], 0, W - 1)))
            if not m[qi]:
                continue
            local = float(off @ l2) / rc
            t = 0.55 * tc[k] + 0.45 * (0.5 + 0.5 * local) + rng.normal(0, 0.07)
            leaves.append((t, q, a + math.pi / 2 + rng.normal(0, 0.5)))
        for t, q, ang in sorted(leaves, key=lambda x: x[0]):
            add(mark(q, rng.uniform(0.85, 1.05), ramp(t), rng.uniform(0.72, 0.9), angle=ang))
    # 2 the silhouette broken by leaves pointing out
    n_edge = _edge_marks(m, xs, ys, full, S, density, rng,
                         lambda q, v, out: mark(q, 0.75, ramp((v - lo) / span * 0.9 + 0.05), rng.uniform(0.72, 0.9),
                                                out_dir=out, own=True), add)
    # 3 bright touches on the most lit clumps
    for k in order:
        if tc[k] > 0.8 and rng.random() < 0.6 * lit_k:
            q = centres[k] + l2 * 0.5 * S
            add(mark(q, 0.6, _mix(hi, ramp(0.9), rng.uniform(0, 0.3)), rng.uniform(0.75, 0.9)))
    return ops, n_edge


def cmd_foliage(ctx: Ctx, args: List[str]) -> Report:
    p = ctx.project
    u = USAGE["foliage"]
    a = parse(args, Spec(u, ["region*"], {"kind": "value", "light": "value", "size": "value", "density": "value",
                                         "color": "value", "layer": "value"},
                         alias={"colour": "color", "type": "kind", "leaves": "kind"}, min_pos=1))
    regs = _region_list(ctx, a["region"], u)
    kinds = {"needle": "needle", "needles": "needle", "pine": "needle", "conifer": "needle", "fir": "needle",
             "leaf": "leaf", "leaves": "leaf", "broadleaf": "leaf", "round": "leaf", "bush": "leaf", "crown": "leaf"}
    if a.get("kind"):
        k = str(a["kind"]).strip().lower()
        if k not in kinds:
            raise UsageError(f"unknown --kind {a['kind']!r}; kinds: needle (pines, firs), leaf (round crowns, bushes)",
                             f"pixelcanvas foliage {regs[0].name} --kind leaf")
        kind = kinds[k]
    else:
        kind = "needle" if any(w in regs[0].name.lower() for w in NEEDLE_WORDS) else "leaf"
    L = _light(p, a.get("light"))
    density = _num(a, "density", 1.0, 0.2, 4, ctx, u)
    clip = np.zeros((p.H, p.W), bool)
    for r in regs:
        clip |= r.mask
    lines: List[str] = []

    def build():
        from scipy import ndimage as ndi
        ops: list = []
        for ri, r in enumerate(regs):
            base, sh, li, nm = cd.colour_set(p, r, a.get("color"))
            colours = (dc.shadow_of(sh, 0.7), sh, base, li, dc.light_of(li, 0.7))
            lab, n = ndi.label(r.mask)
            if n == 0:
                continue
            sizes = ndi.sum(r.mask, lab, range(1, n + 1))
            keep = [k for k in range(n) if sizes[k] >= max(40, 0.01 * max(sizes))]
            n_edge = 0
            sizes_px = []
            for j, k in enumerate(keep):
                comp = lab == k + 1
                S = _len(p, _num(a, "size", 0, 0, 100, ctx, u)) or float(np.clip(math.sqrt(sizes[k]) / 12.0, 3.0,
                                                                                     0.03 * max(p.W, p.H)))
                sizes_px.append(S)
                o, ne = foliage_ops(comp, kind, L["vec"], colours, S, density,
                                    dc.stable_seed(p, ctx.cmdline, "foliage", ri, j), 1.4 if L["z"] < 0 else 1.0)
                ops += o
                n_edge += ne
            side = dc.light_words(L["deg"])
            lines.append(f"{r.name}: {len(keep)} tree shape(s), marks {min(sizes_px or [0]):.0f}px: dark masses, "
                         f"mid tones, lit side toward the {side}"
                         + (" (back light: a lit rim)" if L["z"] < 0 else "")
                         + f"; {n_edge} marks break the silhouette")
        return _keep_marks(ops)

    def post():
        for r in regs:
            for key in ("painted", "detailed"):
                cd._mark_list(dc.dstate(p), key, r.name)
        return []
    return cd.draw_step(ctx, label=f"foliage {', '.join(r.name for r in regs)} ({kind})", layer=_layer(p, a, "paint"),
                        build=build, clip=clip, post=post, lines=lines,
                        note="foliage: dark masses, mid, lit side, broken edge")


# ============================================================================================ tufts

def tuft_ops(mask: np.ndarray, count: int, height: float, colours, spread_deg: float, seed: int) -> Tuple[list, int,
                                                                                                          float]:
    """Grass tufts of 3-5 blades from one base, only in the near two thirds of the region, more and bigger toward
    its bottom (the viewer)."""
    sh, base, li = (np.asarray(c, float) for c in colours)
    rng = _rng(seed)
    m = np.asarray(mask, bool)
    ys, xs = np.nonzero(m)
    y0, y1 = float(ys.min()), float(ys.max() + 1)
    yt = y0 + (y1 - y0) / 3.0
    sel = ys >= yt
    if sel.sum() < 10:
        return [], 0, yt
    ys, xs = ys[sel], xs[sel]
    t = (ys - yt) / max(1.0, y1 - yt)
    wt = t ** 1.5 + 0.02
    n = min(count, len(xs))
    # tufts grow in small clumps: clump centres follow the density, 1-3 tufts around each
    centres = rng.choice(len(xs), size=max(1, int(math.ceil(n / 2.0))), replace=len(xs) < n, p=wt / wt.sum())
    bases = []
    for i in centres:
        c = np.array([xs[i] + rng.uniform(0, 1), ys[i] + rng.uniform(0, 1)])
        tt = float(np.clip((c[1] - yt) / max(1.0, y1 - yt), 0, 1))
        ht = height * (0.3 + 0.7 * tt)
        for k in range(int(rng.integers(1, 4))):
            q = c if k == 0 else c + np.array([rng.normal(0, 0.45 * ht), rng.normal(0, 0.1 * ht)])
            qi = (int(np.clip(q[1], 0, m.shape[0] - 1)), int(np.clip(q[0], 0, m.shape[1] - 1)))
            if q[1] >= yt and m[qi]:
                bases.append(q)
    bases = sorted(bases[:n], key=lambda q: q[1])      # far tufts first, near ones over them
    spread = math.radians(spread_deg)
    dark, mid, light = _mix(sh, base, 0.3), base, _mix(base, li, 0.7)
    ops: list = []
    sd = seed * 10000
    made = 0
    for b in bases:
        tt = float(np.clip((b[1] - yt) / max(1.0, y1 - yt), 0, 1))
        ht = height * (0.3 + 0.7 * tt) * rng.uniform(0.75, 1.2)
        nb = int(rng.integers(3, 6))
        lean = rng.normal(0, 0.15)                      # the whole tuft leans a little (wind)
        angs = np.sort(np.linspace(-spread, spread, nb) * rng.uniform(0.7, 1.2) + rng.normal(0, 0.15, nb)) + lean
        tone = rng.random()
        blades = []
        for ang in angs:
            frac = min(1.0, abs(ang - lean) / max(spread, 1e-6))
            ln = max(5.5, ht * (1 - 0.35 * frac) * rng.uniform(0.6, 1.05))
            w = max(1.0, min(0.065 * ln, ln / 5.5))
            out = math.sin(ang)
            ctrl = b + np.array([0.15 * out * ln, -0.62 * ln])          # it rises, then arcs outward
            tip = b + np.array([1.05 * out * ln, -math.cos(ang) * ln * (0.9 - 0.15 * frac)])
            s = np.linspace(0, 1, max(6, int(ln / 1.5)))[:, None]
            path = (1 - s) ** 2 * b + 2 * (1 - s) * s * ctrl + s ** 2 * tip
            k = rng.random() + 0.35 * (tone - 0.5)
            col = dark if k < 0.3 else mid if k < 0.8 else light
            col = np.clip(np.asarray(col, float) * (1 + rng.normal(0, 0.04)), 0, 255)
            blades.append((0 if k < 0.3 else 1 if k < 0.8 else 2, path, w, col))
        blades.sort(key=lambda x: x[0])                 # dark blades behind, light ones in front
        group = []
        for _, path, w, col in blades:
            sd += 1
            path = dc.resample(path, 1.25)
            wid = stroke_widths(path, w, 1.0, rng.uniform(0.08, 0.14))
            if dc.path_length(path) < 4.0 * float(wid.mean()):
                continue
            group.append(Stroke(pts=path.astype(np.float32), width=wid, color=_rgb(col),
                                opacity=float(rng.uniform(0.82, 1.0)), brush="round", seed=sd, own=True))
        if len(group) >= 3:
            ops += group
            made += 1
    return ops, made, yt


def cmd_tufts(ctx: Ctx, args: List[str]) -> Report:
    p = ctx.project
    u = USAGE["tufts"]
    a = parse(args, Spec(u, ["region*"], {"count": "value", "size": "value", "color": "value", "spread": "value",
                                         "layer": "value"},
                         alias={"colour": "color", "n": "count", "height": "size", "number": "count"}, min_pos=1))
    r = cd.region_or_part(ctx, a["region"], u)
    m = r.mask
    ys = np.nonzero(m.any(1))[0]
    if len(ys) < 6 or m.sum() < 60:
        raise UsageError(f"{r.name} is too small for grass tufts", u)
    rh = float(ys.max() + 1 - ys.min())
    height = _len(p, _num(a, "size", 0, 0, 300, ctx, u)) or min(0.05 * max(p.W, p.H), 0.6 * rh)
    near_px = int((m & (np.arange(p.H)[:, None] >= ys.min() + rh / 3)).sum())
    count = int(_num(a, "count", 0, 1, 2000, ctx, u)) or int(np.clip(near_px / (height * height * 0.35), 8, 400))
    if a.get("color"):
        base = _colour(p, a["color"], u)
        cols = (dc.shadow_of(base), base, dc.light_of(base))
    elif _palette(p, "grass") is not None:
        base = _palette(p, "grass")
        cols = (dc.shadow_of(base), base, dc.light_of(base))
    else:
        base, sh, li, _ = cd.colour_set(p, r)
        cols = (dc.shadow_of(sh, 0.5), base, li)
    spread = _num(a, "spread", 32, 5, 80, ctx, u)
    ops, n, yt = tuft_ops(m, count, height, cols, spread, dc.stable_seed(p, ctx.cmdline, "tufts"))
    ops = _keep_marks(ops)
    if not ops:
        raise UsageError(f"no room for tufts in the near part of {r.name}", u)
    lines = [f"tufts of 3-5 blades, {0.3 * height:.0f}-{height:.0f}px tall from far to near; none in the far top "
             f"third (above y {_u(p, yt, 1):.0f}), more toward the bottom"]

    def post():
        cd._mark_list(dc.dstate(p), "detailed", r.name)
        return []
    return cd.draw_step(ctx, label=f"tufts {r.name}: {n} tufts", layer=_layer(p, a, "detail"), build=lambda: ops,
                        clip=m, post=post, lines=lines, area_extra=r.box_px, note="grass tufts, bigger toward the viewer")


# ============================================================================================ atmosphere

def _light_point(p, L: dict) -> np.ndarray:
    """The light source in px: the plan's point, else a point on the frame on the light's side."""
    W, H = p.W, p.H
    if L["at"] is not None:
        return np.array([np.clip(L["at"][0], 0, W), np.clip(L["at"][1], 0, H)], float)
    c = np.array([W / 2.0, H / 2.0])
    v = L["dir"]
    # walk from the middle toward the light until the frame
    tx = (W / 2.0) / abs(v[0]) if abs(v[0]) > 1e-6 else np.inf
    ty = (H / 2.0) / abs(v[1]) if abs(v[1]) > 1e-6 else np.inf
    q = c + v * min(tx, ty) * 0.92
    return np.array([np.clip(q[0], 0, W), np.clip(q[1], 0, H)], float)


def atmosphere_ops(W: int, H: int, lp, warm: float, vignette: float, warm_rgb, cool_rgb) -> list:
    ops: list = []
    diag = math.hypot(W, H)
    if warm > 0:
        R = 0.85 * diag
        N = 8
        for k in range(N):
            f = 1.0 - k / N
            rk = R * f
            ops.append(Fill(polygons=[dc.ellipse_pts(lp[0], lp[1], rk, 0.88 * rk, n=96).astype(np.float32)],
                            color=_rgb(warm_rgb), opacity=float(warm * 0.6 / N), feather=max(2.0, R / N)))
    if vignette > 0:
        c = np.array([W / 2.0, H / 2.0])
        vc = c + 0.25 * (np.asarray(lp, float) - c)
        outer = _rect(-W, -H, 2 * W, 2 * H)
        N = 6
        for k in range(N):
            f = 1.3 - 0.1 * k
            inner = dc.ellipse_pts(vc[0], vc[1], 0.5 * W * f, 0.5 * H * f, n=96).astype(np.float32)
            ops.append(Fill(polygons=[outer, inner], color=_rgb(cool_rgb), opacity=float(vignette * 0.5 / N),
                            feather=max(2.0, 0.2 * min(W, H) * 0.5)))
    return ops


def cmd_atmosphere(ctx: Ctx, args: List[str]) -> Report:
    p = ctx.project
    u = USAGE["atmosphere"]
    a = parse(args, Spec(u, [], {"warm": "value", "vignette": "value", "at": "value", "color": "value",
                                 "cool": "value", "over": "flag", "layer": "value"},
                         alias={"colour": "color", "vig": "vignette", "glow": "warm", "from": "at", "cold": "cool",
                                "warmth": "warm", "edges": "vignette"}))
    warm = _num(a, "warm", 0.3, 0, 1, ctx, u)
    vig = _num(a, "vignette", 0.3, 0, 1, ctx, u)
    if warm <= 0 and vig <= 0:
        raise UsageError("nothing to do: --warm and --vignette are both 0", u)
    L = _light(p)
    if a.get("at"):
        lp = dc.u2px(p, [coords.parse_point(a["at"], ctx.notes, u)])[0]
    else:
        lp = _light_point(p, L)
    img = p.composite_u8("export")
    if a.get("color"):
        warm_rgb = _colour(p, a["color"], u)
    else:
        warm_rgb = _source_colour(img, lp, 0.12 * max(p.W, p.H))
    if a.get("cool"):
        cool_rgb = _colour(p, a["cool"], u)
    else:
        L3, _, h3 = dc.rgb_to_lch(np.median(img.reshape(-1, 3), axis=0))
        cool_rgb = dc.lch_to_rgb(min(24.0, 0.35 * L3 + 4.0), 16.0, _toward(h3, 275, 0.7))
    excl, subj = (None, []) if a.get("over") else _subjects(p)
    lines = [f"warm light {coords.hex_colour(warm_rgb)} toward {_fmt_u(p, lp)}" if warm > 0 else "",
             f"cool dark edges {coords.hex_colour(cool_rgb)}" if vig > 0 else ""]

    def build():
        ops = _keep_marks(atmosphere_ops(p.W, p.H, lp, warm, vig, warm_rgb, cool_rgb))
        lines.append(_behind_line(_in_the_way(p, subj, ops)))
        return ops
    return cd.draw_step(ctx, label=f"atmosphere warm {warm:g} vignette {vig:g}", layer=_layer(p, a, "fx"),
                        build=build, clip=None if excl is None or not excl.any() else ~excl, lines=lines,
                        full_view=True, note="warmer and lighter toward the light, cooler and darker at the edges")


# ============================================================================================ glow / wet

def glow_ops(pts_px, radius: float, rgb, strength: float) -> list:
    ops: list = []
    fr = (1.0, 0.72, 0.5, 0.32, 0.18)
    opk = (0.1, 0.13, 0.18, 0.26, 0.4)
    for q in np.asarray(pts_px, float).reshape(-1, 2):
        for j, (f, o) in enumerate(zip(fr, opk)):
            c = rgb if j < 2 else dc.light_of(rgb, 0.5 * (j - 1))
            r = max(1.5, radius * f)
            ops.append(Fill(polygons=[dc.ellipse_pts(q[0], q[1], r, r, n=64).astype(np.float32)], color=_rgb(c),
                            opacity=float(np.clip(strength * o, 0.01, 1.0)), feather=max(1.0, 0.6 * r)))
    return ops


def cmd_glow(ctx: Ctx, args: List[str]) -> Report:
    p = ctx.project
    u = USAGE["glow"]
    a = parse(args, Spec(u, ["pts*"], {"radius": "value", "color": "value", "strength": "value", "over": "flag",
                                       "layer": "value"},
                         alias={"colour": "color", "size": "radius", "r": "radius", "opacity": "strength",
                                "amount": "strength", "at": "pts"}, min_pos=0))
    if not a.get("pts"):
        raise UsageError("glow needs the lights: one x,y per lamp or window (0-1000)", u)
    pts = dc.u2px(p, coords.parse_points(a["pts"], ctx.notes, 1, u))
    rad = _len(p, _num(a, "radius", 40, 2, 500, ctx, u))
    if a.get("color"):
        col = _colour(p, a["color"], u)
    else:
        col = _palette(p, "glow", "lamp", "light", "window-light") or (255, 210, 122)
    strength = _num(a, "strength", 0.8, 0.05, 1, ctx, u)
    lines = [f"halos {coords.hex_colour(col)}: rings fading outward, a light core"]
    excl, subj = None, []
    if not a.get("over"):
        # the lamp or window under a glow point is the light itself: it glows; every other subject stays in front
        sources = []
        for n in p.region_names():
            mk = _region_mask(p, n)
            if mk is not None and not _is_setting(p, n) and any(
                    mk[int(np.clip(q[1], 0, p.H - 1)), int(np.clip(q[0], 0, p.W - 1))] for q in pts):
                sources.append(n)
        excl, subj = _subjects(p, skip=set(sources))
        if sources:
            lines.append(f"{', '.join(sources)} under the light glows with it")

    def build():
        ops = _keep_marks(glow_ops(pts, rad, col, strength))
        lines.append(_behind_line(_in_the_way(p, subj, ops)))
        return ops
    return cd.draw_step(ctx, label=f"glow {len(pts)} light(s) radius {rad:.0f}px", layer=_layer(p, a, "fx"),
                        build=build, clip=None if excl is None or not excl.any() else ~excl, lines=lines,
                        note="soft halos around the lights")


def wet_ops(mask: np.ndarray, lights, width: float, colours, strength: float, length: Optional[float],
            seed: int) -> Tuple[list, List[int]]:
    """Broken vertical streaks under each light on the ground: the reflections on wet ground."""
    rng = _rng(seed)
    H, W = mask.shape
    ops: list = []
    done = []
    sd = seed * 10000
    for li, (q, rgb) in enumerate(zip(lights, colours)):
        x = float(q[0])
        xi = int(np.clip(x, 0, W - 1))
        rows = np.nonzero(mask[:, xi])[0]
        rows = rows[rows >= q[1] - 1]
        if len(rows) < 3:
            continue
        y_s, y_e = float(rows.min()), float(rows.max() + 1)
        span = (length if length else 0.9 * (y_e - y_s))
        y_end = min(y_e, y_s + span)
        if y_end - y_s < 3:
            continue
        done.append(li)
        rgb = np.asarray(rgb, float)
        core = np.asarray(dc.light_of(rgb, 0.8), float)
        ops.append(Fill(polygons=[dc.ellipse_pts(x, (y_s + y_end) / 2, 0.9 * width, (y_end - y_s) / 2,
                                                 n=48).astype(np.float32)],
                        color=_rgb(rgb), opacity=0.18 * strength, feather=max(1.5, 1.2 * width)))
        span = y_end - y_s
        y = y_s + rng.uniform(0, 0.02) * span
        while y < y_end - 2:                            # overlapping soft dashes down the column, broken by gaps
            t = (y - y_s) / max(1.0, span)
            w = width * (0.8 + 0.5 * t) * rng.uniform(0.8, 1.15)
            seg = min(max(4.6 * w, span * rng.uniform(0.07, 0.14) * (1 + 0.5 * t)), y_end - y)
            xx = x + rng.normal(0, 0.2 * width)
            pts = np.array([[xx, y], [xx + rng.normal(0, 0.04 * seg), y + seg]])
            op = strength * (0.95 - 0.55 * t) * rng.uniform(0.8, 1.0)
            sd += 1
            m = _mark(pts, w, rgb, op, sd, "soft", 0.5, 0.2)
            if m is not None:
                ops.append(m)
                if t < 0.5:                             # a brighter heart near the light's foot
                    sd += 1
                    c2 = _mark(pts, max(1.0, 0.4 * w), core, min(1.0, op * 0.9), sd, "soft", 0.4, 0.15)
                    if c2 is not None:
                        ops.append(c2)
            y += seg * rng.uniform(0.75, 1.25)
        for j in range(int(span / (1.3 * width)) + 1):   # soft ripples across the streak: the ground is wet
            yy = y_s + rng.uniform(0.03, 0.9) * span
            t = (yy - y_s) / max(1.0, span)
            ln = width * rng.uniform(1.4, 2.4)
            xx = x + rng.normal(0, 0.2 * width)
            sd += 1
            m = _mark(np.array([[xx - ln / 2, yy], [xx + ln / 2, yy + rng.normal(0, 0.5)]]),
                      max(1.0, width * rng.uniform(0.3, 0.45)), _mix(rgb, core, 0.5),
                      strength * (0.75 - 0.45 * t) * rng.uniform(0.7, 1.0), sd, "soft", 0.4, 0.2)
            if m is not None:
                ops.append(m)
    return ops, done


def cmd_wet(ctx: Ctx, args: List[str]) -> Report:
    p = ctx.project
    u = USAGE["wet"]
    a = parse(args, Spec(u, ["region*"], {"from": "value", "color": "value", "width": "value", "length": "value",
                                         "strength": "value", "layer": "value"},
                         alias={"colour": "color", "lights": "from", "at": "from", "size": "width",
                                "opacity": "strength"}, min_pos=1))
    r = cd.region_or_part(ctx, a["region"], u)
    if not a.get("from"):
        raise UsageError("wet needs --from \"x,y x,y\": the lights whose reflections fall on the wet ground",
                         f'pixelcanvas wet {r.name} --from "300,420 640,400"')
    pts = dc.u2px(p, coords.parse_points(a["from"], ctx.notes, 1, u))
    width = _len(p, _num(a, "width", 20, 1, 200, ctx, u))
    strength = _num(a, "strength", 0.8, 0.05, 1, ctx, u)
    length = _len(p, _num(a, "length", 0, 0, 1000, ctx, u)) or None
    img = p.composite_u8("export")
    if a.get("color"):
        cols = [_colour(p, a["color"], u)] * len(pts)
    else:
        cols = []
        for q in pts:
            L2, C2, h2 = dc.rgb_to_lch(_box_colour(img, q[0], q[1], 0.4 * width))
            cols.append(dc.lch_to_rgb(max(L2, 80.0), max(C2, 30.0), h2 if C2 > 8 else 80.0))
    lines: List[str] = []

    def build():
        ops, done = wet_ops(r.mask, pts, width, cols, strength, length, dc.stable_seed(p, ctx.cmdline, "wet"))
        lines.append(f"{len(done)} streak(s) of broken light under " +
                     ", ".join(_fmt_u(p, pts[i]) for i in done) if done else
                     f"no light stands over {r.name} (each streak starts where its light's column meets the ground)")
        return _keep_marks(ops)
    return cd.draw_step(ctx, label=f"wet {r.name}: {len(pts)} light(s)", layer=_layer(p, a, "fx"), build=build,
                        clip=r.mask, lines=lines, area_extra=r.box_px, note="streak reflections on wet ground")


# ============================================================================================ ground

def _standing(p, y_u: float) -> list:
    """Planned subjects that stand on a floor at ``y_u``: their box ends below the line (not sky, water, ground)."""
    out = []
    for name, s in dc.dstate(p)["subjects"].items():
        x0, y0, x1, y1 = s["box"]
        if workflow_draw.is_ground(name) or (x1 - x0) > 900 or y1 < y_u + 5:
            continue
        out.append((name, s["box"]))
    return out


def _subject_regions(p, names) -> Tuple[np.ndarray, List[str]]:
    d = dc.dstate(p)
    m = np.zeros((p.H, p.W), bool)
    used = []
    for n in p.region_names():
        owner = d["parts"].get(n, {}).get("of")
        if any(n == s or owner == s or re.fullmatch(re.escape(s) + r"-\d+", n) for s in names):
            m |= p.region_mask(n)
            used.append(n)
    return m, used


def contact_ops(p, standing, y_floor: float, colour_at, L: dict, strength: float) -> list:
    ops: list = []
    for name, box in standing:
        bx0, by0, bx1, by1 = dc.box_px(p, box)
        cx = (bx0 + bx1) / 2.0
        yc = min(by1, p.H - 2.0)
        yc = max(yc, y_floor + 1.0)
        rx = 0.36 * (bx1 - bx0)
        ry = max(2.5, 0.16 * rx)
        v = _toward_light(L, (cx, yc))
        dx = -v[0] * 0.35 * rx
        sc = dc.shadow_of(colour_at(yc), 1.6)
        ops.append(Fill(polygons=[dc.ellipse_pts(cx + dx, yc - 0.15 * ry, 1.3 * rx, 1.5 * ry, n=72).astype(np.float32)],
                        color=sc, opacity=0.38 * strength, feather=max(2.0, 1.4 * ry)))
        ops.append(Fill(polygons=[dc.ellipse_pts(cx + 0.35 * dx, yc, 0.62 * rx, 0.7 * ry, n=60).astype(np.float32)],
                        color=dc.shadow_of(sc, 0.5), opacity=0.55 * strength, feather=max(1.5, 0.8 * ry)))
    return ops


def cmd_ground(ctx: Ctx, args: List[str]) -> Report:
    p = ctx.project
    d = dc.dstate(p)
    u = USAGE["ground"]
    a = parse(args, Spec(u, ["at"], {"y": "value", "color": "value", "color2": "value", "line": "flag",
                                     "name": "value", "shadow": "value", "layer": "value"},
                         alias={"colour": "color", "colour2": "color2", "floor": "y", "horizon": "y",
                                "strength": "shadow"}))
    ytxt = a.get("y") or a.get("at")
    if ytxt:
        y_u = coords.parse_float(ytxt, "y", 0, 1000, ctx.notes, u)
    else:
        hz = next((el for el in d["sketch"] if el.get("kind") == "horizon"), None)
        y_u = float(hz["y"]) if hz else 820.0
        ctx.notes.append(f"floor line at y {y_u:g} ({'the sketch horizon' if hz else 'default'}; set it with --y)")
    y = y_u * p.H / 1000.0
    name = cd.from_regions_clean(a.get("name") or ("floor-line" if a.get("line") else "ground"))
    img = p.composite_u8("export")
    if a.get("color"):
        c1 = _colour(p, a["color"], u)
    else:
        pal = _palette(p, "floor", "ground")
        if pal is not None:
            c1 = pal
        else:
            above = img[max(0, int(y - 0.12 * p.H)):max(1, int(y)), :, :3].reshape(-1, 3)
            Lb, Cb, hb = dc.rgb_to_lch(np.median(above, axis=0))
            c1 = dc.lch_to_rgb(0.78 * Lb, 0.8 * Cb + 3.0, _toward(hb, 55, 0.25))
    c2 = _colour(p, a["color2"], u) if a.get("color2") else dc.shadow_of(c1, 0.45)
    standing = _standing(p, y_u)
    excl, used = _subject_regions(p, [s for s, _ in standing])
    strength = _num(a, "shadow", 1.0, 0, 1, ctx, u)
    L = _light(p)
    painterly = (p.state.get("style") or "paint") in ("paint", "photo")
    layer = _layer(p, a, "paint" if painterly else "flat")
    poly = np.array([[-1, y], [p.W + 1, y], [p.W + 1, p.H + 1], [-1, p.H + 1]], np.float32)
    floor = dc.poly_mask(p.H, p.W, poly) & ~excl
    others = [n for n in d["bands"] if n != name and n not in used]
    files = [] if a.get("line") else [f"regions/{name}.png"] + [f"regions/{n}.png" for n in others]

    def colour_at(yy):
        k = float(np.clip((yy - y) / max(1.0, p.H - y), 0, 1))
        return _rgb(_mix(c1, c2, k))

    def build():
        ops: list = []
        if a.get("line"):
            line_c = dc.shadow_of(_box_colour(img, p.W / 2, y, 0.02 * max(p.W, p.H)), 1.2)
            ops.append(Fill(polygons=[_rect(-1, y, p.W + 1, y + 0.03 * p.H)], color=line_c, opacity=0.25,
                            feather=max(2.0, 0.03 * p.H)))
            m = _mark(np.array([[-2.0, y], [p.W + 2.0, y]]), max(1.5, 0.004 * max(p.W, p.H)), line_c, 0.75,
                      dc.stable_seed(p, ctx.cmdline, "ground"), "round", 0.8, 0.8)
            if m is not None:
                ops.append(m)
        else:
            ops.append(Fill(polygons=[poly], color=_rgb(c1), color2=_rgb(c2), p0=(p.W / 2.0, y),
                            p1=(p.W / 2.0, float(p.H)), feather=1.0))
        ops += contact_ops(p, standing, y, colour_at, L, strength)
        return _keep_marks(ops)

    def post():
        out = []
        if standing:
            sx = -_toward_light(L, (p.W / 2, y))[0]
            out.append(f"contact shadow under {', '.join(s for s, _ in standing)}"
                       + (f" (pushed {'right' if sx > 0 else 'left'}, away from the light)" if abs(sx) > 0.15 else ""))
        else:
            out.append(f"no planned subject stands on the floor (a subject's box must end below y {y_u:g})")
        if used:
            out.append(f"the floor stays behind {', '.join(used)}")
        if not a.get("line"):
            d["bands"][name] = {"color": coords.hex_colour(c1), "color2": coords.hex_colour(c2), "depth": 1,
                                "how": "floor band", "depth_set": False}
            p.save_region_mask(name, floor)
            box = coords.px_box_to_units(coords.mask_bbox_px(floor) or (0, int(y), p.W, p.H), p.W, p.H)
            p.state["regions"][name] = {"focus": p.state["regions"].get(name, {}).get("focus", 1), "box": list(box),
                                        "area": round(float(floor.mean()), 4), "how": "floor band"}
            cut = []
            for n in others:
                if n in p.state["regions"]:
                    om = p.region_mask(n)
                    if (om & floor).any():
                        nm2 = om & ~floor
                        if nm2.any():
                            p.save_region_mask(n, nm2)
                            p.state["regions"][n]["area"] = round(float(nm2.mean()), 4)
                            cut.append(n)
            out.append(f"region {name}: the floor from y {y_u:g} down, {coords.hex_colour(c1)} -> "
                       f"{coords.hex_colour(c2)}" + (f"; {', '.join(cut)} now end at the floor" if cut else ""))
        return out
    label = (f"ground line at y {y_u:g}" if a.get("line") else
             f"ground from y {y_u:g}" + ("" if name == "ground" else f" (region {name})"))
    return cd.draw_step(ctx, label=label, layer=layer, build=build, clip=None if not excl.any() else ~excl,
                        extra_files=files, post=post, full_view=True, note="the floor and the contact shadows")


# ============================================================================================ window

def _panes(text: str, usage: str) -> Tuple[int, int]:
    s = str(text).strip().lower()
    m = re.fullmatch(r"(\d+)\s*[x*,\s]\s*(\d+)", s)
    if m:
        c, r = int(m.group(1)), int(m.group(2))
    elif re.fullmatch(r"\d+", s):
        c = r = int(s)
    else:
        raise UsageError(f"--panes is columns x rows like 2x2 or 1x3, got {text!r}", usage)
    if not (1 <= c <= 12 and 1 <= r <= 12):
        raise UsageError(f"--panes {c}x{r}: use 1-12 columns and rows", usage)
    return c, r


def window_ops(box_px, cols: int, rows: int, ft: float, frame_rgb, glass_rgb, wall_rgb, L: dict,
               lit: bool, sheen_rgb=None) -> list:
    X0, Y0, X1, Y1 = (float(v) for v in box_px)
    ix0, iy0, ix1, iy1 = X0 + ft, Y0 + ft, X1 - ft, Y1 - ft
    iw, ih = ix1 - ix0, iy1 - iy0
    mt = max(1.5, 0.45 * ft)
    v = _toward_light(L, ((X0 + X1) / 2, (Y0 + Y1) / 2))
    frame = np.asarray(frame_rgb, float)
    glass = np.asarray(glass_rgb, float)
    ops: list = []
    # the frame's shadow on the wall, away from the light
    off = -v * 0.45 * ft
    ops.append(Fill(polygons=[_rect(X0 + off[0], Y0 + off[1], X1 + off[0], Y1 + off[1])],
                    color=dc.shadow_of(wall_rgb, 1.2), opacity=0.45, feather=max(1.5, 0.6 * ft)))
    # glass: lighter where it reflects the sky (top), or the warm room light
    top, bot = (glass, dc.light_of(glass, 0.5)) if lit else (dc.light_of(glass, 0.45), dc.shadow_of(glass, 0.35))
    ops.append(Fill(polygons=[_rect(ix0, iy0, ix1, iy1)], color=_rgb(top), color2=_rgb(bot),
                    p0=((ix0 + ix1) / 2, iy0), p1=((ix0 + ix1) / 2, iy1), feather=0.8))
    # the frame's shadow on the glass, on the side the light comes from
    sw = 0.7 * ft
    strips = []
    if v[1] < -0.2:
        strips.append(_rect(ix0, iy0, ix1, iy0 + sw))
    elif v[1] > 0.2:
        strips.append(_rect(ix0, iy1 - sw, ix1, iy1))
    if v[0] < -0.2:
        strips.append(_rect(ix0, iy0, ix0 + sw, iy1))
    elif v[0] > 0.2:
        strips.append(_rect(ix1 - sw, iy0, ix1, iy1))
    for s_ in strips:
        ops.append(Fill(polygons=[s_], color=dc.shadow_of(glass, 0.9), opacity=0.4, feather=max(1.0, 0.5 * sw)))
    # the sheen: two soft parallel bands from the lower left to the upper right, across every pane
    pw = (iw - (cols - 1) * mt) / cols
    ph = (ih - (rows - 1) * mt) / rows
    m0 = min(iw, ih)
    dvec = np.array([1.0, -1.25])
    dvec /= np.hypot(*dvec)
    nvec = np.array([-dvec[1], dvec[0]])
    half_len = math.hypot(iw, ih)
    white = _rgb(sheen_rgb) if sheen_rgb is not None else dc.lch_to_rgb(98.5, 2.0, dc.rgb_to_lch(glass)[2])
    c0 = np.array([ix0 + 0.36 * iw, iy0 + 0.36 * ih])
    for shift, hw, op in ((0.0, 0.09 * m0, 0.32), (0.2 * m0, 0.035 * m0, 0.24)):
        cc = c0 + nvec * shift
        band = np.array([cc - dvec * half_len - nvec * hw, cc + dvec * half_len - nvec * hw,
                         cc + dvec * half_len + nvec * hw, cc - dvec * half_len + nvec * hw])
        for i in range(cols):
            for j in range(rows):
                px0 = ix0 + i * (pw + mt)
                py0 = iy0 + j * (ph + mt)
                q = _clip_rect(band, px0, py0, px0 + pw, py0 + ph)
                if len(q) >= 3:
                    ops.append(Fill(polygons=[q], color=white, opacity=op * (0.45 if lit else 1.0),
                                    feather=max(1.0, 0.45 * hw)))
    # bars between the panes, then the frame on top
    for i in range(1, cols):
        x = ix0 + i * (pw + mt) - mt / 2
        ops.append(Fill(polygons=[_rect(x - mt / 2, iy0 - 0.5, x + mt / 2, iy1 + 0.5)], color=_rgb(frame),
                        feather=0.6))
    for j in range(1, rows):
        yy = iy0 + j * (ph + mt) - mt / 2
        ops.append(Fill(polygons=[_rect(ix0 - 0.5, yy - mt / 2, ix1 + 0.5, yy + mt / 2)], color=_rgb(frame),
                        feather=0.6))
    ops.append(Fill(polygons=[_rect(X0, Y0, X1, Y1), _rect(ix0, iy0, ix1, iy1)], color=_rgb(frame), feather=0.7))
    # bevel: the frame's outer edges catch the light on the light's sides, shade on the others
    bw = max(1.0, 0.32 * ft)
    lite, dark = dc.light_of(frame, 0.8), dc.shadow_of(frame, 0.6)
    sides = {"top": (_rect(X0, Y0, X1, Y0 + bw), -v[1]), "bottom": (_rect(X0, Y1 - bw, X1, Y1), v[1]),
             "left": (_rect(X0, Y0, X0 + bw, Y1), -v[0]), "right": (_rect(X1 - bw, Y0, X1, Y1), v[0])}
    for _, (poly, facing) in sides.items():
        if facing > 0.2:
            ops.append(Fill(polygons=[poly], color=lite, opacity=0.6, feather=max(0.8, 0.4 * bw)))
        elif facing < -0.2:
            ops.append(Fill(polygons=[poly], color=dark, opacity=0.5, feather=max(0.8, 0.4 * bw)))
    return ops


def _region_box_u(p, name: str) -> Optional[Tuple[float, ...]]:
    """A region's box in units: the one recorded when it was made (before later bands cut it), else its mask's."""
    b = p.state["regions"].get(name, {}).get("box")
    if b and len(b) == 4:
        return tuple(float(v) for v in b)
    mk = _region_mask(p, name)
    bb = coords.mask_bbox_px(mk) if mk is not None else None
    return tuple(float(v) for v in coords.px_box_to_units(bb, p.W, p.H)) if bb else None


def _box_match(box, rb) -> float:
    """How well a window box matches a region's box: their overlap over union, or 1 when the box lies inside it."""
    ix = max(0.0, min(box[2], rb[2]) - max(box[0], rb[0]))
    iy = max(0.0, min(box[3], rb[3]) - max(box[1], rb[1]))
    inter = ix * iy
    a_box = (box[2] - box[0]) * (box[3] - box[1])
    union = a_box + (rb[2] - rb[0]) * (rb[3] - rb[1]) - inter
    if a_box > 0 and inter / a_box >= 0.8:
        return 1.0
    return inter / union if union > 0 else 0.0


def _window_region(ctx: Ctx, rtxt, box, usage: str):
    """The window's own region: --region, else a region named window / glass (under --box when one is given)."""
    p = ctx.project
    if rtxt:
        if str(rtxt).strip().lower() in ("none", "no", "off"):
            return None
        return cd.region_or_part(ctx, rtxt, usage)
    cands = [n for n in p.region_names() if _named(n, ("window", "glass"))]
    if box is not None:
        scored = [(_box_match(box, rb), n) for n in cands for rb in [_region_box_u(p, n)] if rb]
        scored = [s for s in scored if s[0] >= 0.4]
        return cli.resolve_region(p, max(scored)[1], ctx.notes, usage) if scored else None
    if len(cands) > 1:
        cands = [n for n in cands if n in ("window", "glass")] or cands
    if len(cands) > 1:
        raise UsageError(f"there are several window regions ({', '.join(cands)}): name one with --region",
                         f"pixelcanvas window --region {cands[0]} --panes 2x2")
    return cli.resolve_region(p, cands[0], ctx.notes, usage) if cands else None


def cmd_window(ctx: Ctx, args: List[str]) -> Report:
    p = ctx.project
    u = USAGE["window"]
    a = parse(args, Spec(u, ["where*"], {"box": "value", "region": "value", "panes": "value", "color": "value",
                                        "glass": "value", "frame": "value", "lit": "flag", "over": "flag",
                                        "layer": "value"},
                         alias={"colour": "color", "pane": "panes", "grid": "panes", "glass-color": "glass",
                                "glass-colour": "glass", "frame-color": "color", "width": "frame", "in": "region"},
                         min_pos=0))
    btxt, rtxt = a.get("box"), a.get("region")
    if a.get("where"):
        if coords.looks_like_box(a["where"]) and not btxt:
            btxt = a["where"]
        elif not rtxt:
            rtxt = a["where"]
    box = coords.parse_box(btxt, ctx.notes, u) if btxt else None
    reg = _window_region(ctx, rtxt, box, u)
    if box is None:
        if reg is None:
            raise UsageError("window needs --box x0,y0,x1,y1 (0-1000), the outside of the frame, or a window "
                             "region to fill (--region NAME)", u)
        box = _region_box_u(p, reg.name) if reg.name in p.state["regions"] else \
            coords.px_box_to_units(reg.box_px, p.W, p.H)
    bp = dc.box_px(p, box)
    cols, rows = _panes(a.get("panes") or "2x2", u)
    bw, bh = bp[2] - bp[0], bp[3] - bp[1]
    ft = _len(p, _num(a, "frame", 0, 0, 100, ctx, u)) or max(2.0, 0.07 * min(bw, bh))
    if min(bw, bh) - 2 * ft < 4 * max(cols, rows):
        raise UsageError(f"the box {coords.fmt_box(box)} is too small for a window with {cols}x{rows} panes", u)
    sheen = None
    if a.get("glass"):
        glass = _colour(p, a["glass"], u)
        how = "given"
    elif reg is not None:
        glass = cd.colour_set(p, reg)[0]                # the window region's own colour, with a lighter sheen
        Lg, Cg, hg = dc.rgb_to_lch(glass)
        sheen = dc.lch_to_rgb(min(99.0, Lg + 0.85 * (100 - Lg)), 0.3 * Cg, hg)
        how = f"the colour of {reg.name}"
    else:
        glass = _palette(p, "glass") or ((255, 214, 130) if a.get("lit") else (150, 172, 196))
        how = "palette glass" if _palette(p, "glass") is not None else "default"
    if a.get("color"):
        frame = _colour(p, a["color"], u)
    else:
        frame = _palette(p, "frame", "window-frame", *(() if reg is not None else ("window",))) or (236, 230, 218)
    img = p.composite_u8("export")
    wall = _box_colour(img, bp[0] - 0.05 * bw, bp[1] - 0.05 * bh, 0.03 * max(p.W, p.H))
    L = _light(p)
    # what stays in front: regions painted after the window's region (or after the region it sits on)
    clip, front = None, []
    if not a.get("over"):
        x0, y0, x1, y1 = (int(round(v)) for v in bp)
        H, W = p.H, p.W
        hold = reg
        if hold is None:
            share = []
            for n in p.region_names():
                mk = _region_mask(p, n)
                if mk is not None and y1 > y0 and x1 > x0:
                    share.append((float(mk[max(0, y0):y1, max(0, x0):x1].mean()), n))
            best = max(share) if share else (0.0, None)
            hold = cli.resolve_region(p, best[1], ctx.notes, u) if best[0] >= 0.3 else None
        view: List[str] = []
        if hold is not None:
            later = _later_regions(p, hold.name, WINDOW_WORDS)
            # what belongs to the window's own subject (the hills, sky or street seen through it, even drawn with
            # --front) is the view behind the glass: the frame goes over it
            from .draw_checks import region_subject
            own = region_subject(p, hold.name)
            if own is not None:
                view = [n for n in later if region_subject(p, n) == own]
                later = [n for n in later if n not in view]
        else:
            later = _subjects(p, skip=[n for n in p.region_names() if _named(n, WINDOW_WORDS)])[1]
        g = int(math.ceil(0.6 * ft)) + 2
        front_m = np.zeros((H, W), bool)
        for n in later:
            mk = _region_mask(p, n)
            if mk is not None and mk[max(0, y0 - g):y1 + g, max(0, x0 - g):x1 + g].any():
                front_m |= mk
                front.append(n)
        if reg is not None:
            ring = np.zeros((H, W), bool)
            ring[max(0, y0 - g):min(H, y1 + g), max(0, x0 - g):min(W, x1 + g)] = True
            ring[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = False
            seen = reg.mask.copy()
            for n in view:
                mk = _region_mask(p, n)
                if mk is not None:
                    seen |= mk
            clip = (seen | ring) & ~front_m
        elif front:
            clip = ~front_m
    lines = [f"frame {coords.hex_colour(frame)} {ft:.0f}px, glass {coords.hex_colour(glass)} ({how})"
             + (" lit from inside" if a.get("lit") else "") + ", a soft sheen from the lower left to the upper right"]
    if reg is not None:
        lines.append(f"in region {reg.name}")
    if front:
        lines.append(f"stays behind {', '.join(front)} (painted after the "
                     f"{'window' if reg is not None else 'wall'}: in front of it; --over paints over them)")

    def build():
        return _keep_marks(window_ops(bp, cols, rows, ft, frame, glass, wall, L, bool(a.get("lit")), sheen))
    return cd.draw_step(ctx, label=f"window {coords.fmt_box(box)} panes {cols}x{rows}", layer=_layer(p, a, "detail"),
                        build=build, clip=clip, lines=lines, area_extra=bp,
                        note="the window: frame, panes, glass and sheen")


# ============================================================================================ table

# ============================================================================================ band silhouettes

def shape_seed(name: str, step: int) -> int:
    """A seed from a band's name and the project step: the same command at the same step gives the same shape (a
    layer re-rendered from its records), two trees drawn one after the other differ."""
    import zlib
    return zlib.crc32(f"{name}:{int(step)}".encode()) & 0x7FFFFFFF


def tier_outline(base, top, width_px: float, tiers: int, seed: int = 1) -> np.ndarray:
    """Closed outline of a tiered silhouette (a conifer, a pagoda of branches) from the base point to the top point.

    Each tree differs by its seed: height and width vary by up to about 10 %, the top leans a little; tiers get
    closer and narrower toward the top, unevenly; each tier's two tips sit at their own heights and reaches (one
    side longer), the branch underside droops to the tip and the top edge carries a few notches of needle clumps;
    the trunk foot is uneven, with a root flare on one side."""
    rng = _rng(seed)
    b, t = np.asarray(base, float), np.asarray(top, float)
    ax = t - b
    L0 = float(np.linalg.norm(ax))
    if L0 < 1e-6:
        return np.array([b, b + [1, 0], b + [0, 1]], float)
    u = ax / L0
    nrm = np.array([-u[1], u[0]])
    # the tree's own proportions: height and width a little off the asked ones, the tip leaning
    L = L0 * (1 + rng.uniform(-0.09, 0.09))
    W = width_px * (1 + rng.uniform(-0.1, 0.1))
    lean = rng.normal(0, 0.035) * L
    trunk = rng.uniform(0.05, 0.09)

    def P(h: float, x: float) -> np.ndarray:
        # a point at height h (0 base .. 1 top, in L) and side offset x (px); the axis bends toward the lean
        return b + u * (h * L) + nrm * (x + lean * max(h, 0.0) ** 1.6)

    # tier bottoms: gaps shrink upward (a geometric run), each one jittered
    r = rng.uniform(0.78, 0.9)
    gaps = np.array([r ** k for k in range(tiers)]) * rng.uniform(0.82, 1.18, tiers)
    top_gap = rng.uniform(0.05, 0.1)
    span = 1 - trunk - top_gap
    hs = trunk + np.concatenate([[0.0], np.cumsum(gaps)[:-1]]) / gaps.sum() * span \
        + np.concatenate([[0.0], rng.uniform(-0.12, 0.12, tiers - 1)]) * span / tiers
    for k in range(1, tiers):                       # tiers never bunch up: the outline stays a simple polygon
        hs[k] = max(hs[k], hs[k - 1] + 0.55 * span / tiers)
    hi_ = 1 - top_gap - 0.6 * span / tiers
    if hs[-1] > hi_:
        hs = trunk + (hs - trunk) * (hi_ - trunk) / (hs[-1] - trunk)
    # half widths: widest low, shrinking faster toward the top, uneven; never wider than the tier below
    pw = rng.uniform(0.75, 1.05)
    hw = W / 2 * np.clip(1 - (hs - trunk) / (1 - trunk), 0.04, 1) ** pw * rng.uniform(0.86, 1.12, tiers)
    for k in range(1, tiers):
        hw[k] = min(hw[k], hw[k - 1] * rng.uniform(0.86, 0.97))
    side_bias = rng.uniform(-0.08, 0.08)          # the tree is fuller on one side
    step_h = span / tiers

    def side(sg: int) -> List[np.ndarray]:
        # (height, distance from the axis) pairs, converted at the end; every run keeps its height monotone, so
        # the outline never crosses itself
        q: List[Tuple[float, float]] = []
        tw = W * rng.uniform(0.035, 0.055)
        # trunk foot: uneven, a root flare on one side
        q.append((rng.uniform(-0.006, 0.004), tw * rng.uniform(1.0, 1.9)))
        q.append((trunk * rng.uniform(0.35, 0.6), tw * rng.uniform(0.85, 1.05)))
        q.append((trunk * rng.uniform(0.85, 1.0), tw * 0.95))
        prev_reach = float("inf")
        px_ = ph_ = None                                # the tier below's top edge (x rising), for the clamp
        margin = 0.05 * step_h
        for k in range(tiers):
            reach = min(hw[k] * (1 + sg * side_bias) * rng.uniform(0.9, 1.1), 0.97 * prev_reach)
            root_h, root_x = q[-1]
            reach = max(reach, 1.2 * root_x + 1.0)      # a bough always reaches out past its root
            prev_reach = reach
            tip_h = hs[k] - rng.uniform(0.05, 0.3) * step_h           # the tip droops below the tier's root
            tip_h = min(tip_h, root_h - 0.02 * step_h) if k == 0 else tip_h

            def above_below(x: float, h: float) -> float:
                # a bough stays above the top edge of the tier under it
                if px_ is not None and px_[0] <= x <= px_[-1]:
                    return max(h, float(np.interp(x, px_, ph_)) + margin)
                return h
            tip_h = above_below(reach, tip_h)
            # the underside: out from the root to the tip, sagging more toward the tip (a drooping bough), with a
            # vertex wherever the edge below bends
            xs_u = [root_x + (reach - root_x) * s for s in (0.3, 0.55, 0.78, 0.92)]
            if px_ is not None:
                xs_u += [float(x) for x in px_ if root_x < x < reach]
            xs_u = sorted(set(xs_u))
            under = []
            for x in xs_u:
                s = (x - root_x) / max(reach - root_x, 1e-9)
                under.append((above_below(x, root_h + (tip_h - root_h) * s ** 1.7), x))
            under.append((tip_h, reach))
            q += under
            ux = np.array([root_x] + [x for _, x in under])
            uh = np.array([root_h] + [h for h, _ in under])
            top_edge: List[Tuple[float, float]] = []
            # the top edge up and in from the tip to the notch above, with needle-clump notches
            if k + 1 < tiers:
                nh = hs[k + 1] + rng.uniform(0.05, 0.22) * step_h
                nx = (0.35 * hw[k] + 0.15 * hw[k + 1]) * rng.uniform(0.8, 1.15)
            else:
                nh = 1 - top_gap * rng.uniform(0.3, 0.7)
                nx = 0.22 * hw[k] * rng.uniform(0.6, 1.2)
            hk = (tip_h, reach)
            n_notch = int(rng.integers(1, 4))
            ss = np.linspace(0.2, 0.8, n_notch + 2)[1:-1] + rng.uniform(-0.06, 0.06, n_notch)
            gap_s = 0.6 / (n_notch + 1)
            for s in ss:
                h = hk[0] + (nh - hk[0]) * s ** 0.85
                x = hk[1] + (nx - hk[1]) * s
                dent = rng.uniform(0.25, 0.45) * gap_s * (hk[1] - nx)
                top_edge.append((h, x))
                top_edge.append((h + 0.01 * step_h, x - dent))         # a notch: in, then out again above
            # the top edge stays above the bough's underside and keeps rising: a vertex at every x where the
            # underside bends, so between vertices both are straight and the top one stays on top
            ex = [-reach] + [-x for _, x in top_edge] + [-nx]
            eh = [tip_h] + [h for h, _ in top_edge] + [nh]
            for gx in ux[1:-1]:
                if nx < gx < reach:
                    gh = float(np.interp(-gx, ex, eh))
                    top_edge.append((gh, float(gx)))
            top_edge.sort(key=lambda hx: -hx[1])
            lo_h = tip_h
            done: List[Tuple[float, float]] = [(tip_h, reach)]
            for h, x in top_edge:
                if ux[0] <= x <= ux[-1]:
                    h = max(h, float(np.interp(x, ux, uh)) + 0.06 * step_h)
                lo_h = max(lo_h + 1e-4, h)
                q.append((lo_h, x))
                done.append((lo_h, x))
            q.append((max(nh, lo_h + 0.02 * step_h), nx))
            done.append(q[-1])
            px_ = np.array([x for _, x in done[::-1]])
            ph_ = np.array([h for h, _ in done[::-1]])
        return [P(h, sg * x) for h, x in q]

    right = side(1)
    left = side(-1)
    apex = P(1.0, rng.normal(0, 0.01) * W)
    return np.array(right + [apex] + left[::-1], float)


def ridge_regularity(pts: np.ndarray) -> Optional[Tuple[float, float, int]]:
    """How regular the top edge of a closed band is: (spread of the peak spacing, spread of the peak heights,
    the number of peaks), each spread as a coefficient of variation; None with fewer than 4 peaks. A ridge drawn
    as an even zigzag or a sine wave has both spreads small."""
    P = np.asarray(pts, float)
    if len(P) < 7:
        return None
    # the top edge: the points from the leftmost to the rightmost going over the top (smaller y)
    i0, i1 = int(np.argmin(P[:, 0])), int(np.argmax(P[:, 0]))
    a = P[min(i0, i1):max(i0, i1) + 1]
    b = np.vstack([P[max(i0, i1):], P[:min(i0, i1) + 1]])
    top = a if a[:, 1].mean() < b[:, 1].mean() else b
    top = top[np.argsort(top[:, 0])]
    y = top[:, 1]
    pk = [k for k in range(1, len(y) - 1) if y[k] < y[k - 1] and y[k] < y[k + 1]]
    if len(pk) < 4:
        return None
    xs = top[pk, 0]
    gaps = np.diff(xs)
    heights = float(y.max()) - y[pk]                  # each peak's rise over the lowest point of the edge
    return (float(gaps.std() / max(gaps.mean(), 1e-6)), float(heights.std() / max(heights.mean(), 1e-6)), len(pk))


def irregular_ridge(box, seed: int, peaks: int = 0) -> List[Tuple[float, float]]:
    """A ridge-line suggestion (0-1000 units) inside ``box``: peaks at uneven spacing and heights, one dominant
    peak off centre, shoulders between; left to right, then the two bottom corners (a closed band)."""
    rng = _rng(seed)
    x0, y0, x1, y1 = [float(v) for v in box]
    w, h = x1 - x0, y1 - y0
    n = peaks or int(rng.integers(4, 7))
    cuts = np.sort(rng.uniform(0.04, 0.96, n))
    for _ in range(4):                               # keep peaks apart but uneven
        d = np.diff(cuts)
        if d.min() > 0.07:
            break
        cuts = np.sort(rng.uniform(0.04, 0.96, n))
    main = int(rng.integers(1, n - 1)) if n > 2 else 0
    base_y = y0 + h * rng.uniform(0.55, 0.7)
    out: List[Tuple[float, float]] = [(x0, base_y + rng.uniform(-0.08, 0.08) * h)]
    for k, c in enumerate(cuts):
        tall = 1.0 if k == main else rng.uniform(0.3, 0.75) * (1 - 0.25 * abs(k - main) / n)
        px = x0 + c * w
        py = base_y - tall * (base_y - y0)
        if k > 0:
            # the saddle before this peak: off the middle, at its own depth (a shoulder on the higher side)
            pxs = x0 + (cuts[k - 1] + (c - cuts[k - 1]) * rng.uniform(0.35, 0.65)) * w
            lo = max(py, out[-1][1])                 # below the lower of the two peaks, at its own depth
            out.append((pxs, lo + (base_y + 0.15 * h - lo) * rng.uniform(0.2, 0.6)))
        out.append((px, py))
    out.append((x1, base_y + rng.uniform(-0.1, 0.05) * h))
    out += [(x1, y1), (x0, y1)]
    return [(round(x), round(y)) for x, y in out]


COMMANDS = {
    "cloud": cmd_cloud, "clouds": cmd_cloud, "reflect": cmd_reflect, "foliage": cmd_foliage, "tufts": cmd_tufts,
    "atmosphere": cmd_atmosphere, "glow": cmd_glow, "wet": cmd_wet, "ground": cmd_ground, "window": cmd_window,
}
