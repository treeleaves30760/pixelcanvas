"""``easel polish``: the finishing pass of a picture drawn from a request, run in the finish stage.

Six parts, each reported in numbers and each one switchable (``--no-<part>``):

  strays   specks on the line layer that no line draws, and line ends that run on past a crossing, are painted out
  clean    streaks and patches inside each colour area are smoothed toward the area's own colour; edges stay
  lines    line art takes a dark shade of the colour it borders (coloured line art) instead of plain black
  depth    atmospheric perspective: farther subjects (plan order) lighter and cooler, toward the sky colour
  grade    one light over the whole picture: lights lean toward the light's colour, darks toward its cool
           complement
  soften   soft edges between colour areas for painterly and soft styles (cel and ink keep their crisp edges)

The result is stored on the ``fx`` layer as an ordinary RGBA layer that reproduces the pass over the layers below
(as ``easel fx`` does), so it is one undoable step, shows in layered exports and the time-lapse.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import coords
from . import draw_core as dc
from .coords import UsageError

WARM_WHITE = (255, 242, 222)
PARTS = ("strays", "clean", "lines", "depth", "grade", "soften")
USAGE = {"polish": "easel polish   [--no-strays --no-clean --no-lines --no-depth --no-grade --no-soften]   "
                   "[--clean 0-1 --grade 0-1 --depth 0-1 --soften 0-1]"}


# ============================================================================================ colour helpers

def _lab(rgb01: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(rgb01, np.float32), cv2.COLOR_RGB2Lab)


def _rgb(lab: np.ndarray) -> np.ndarray:
    return np.clip(cv2.cvtColor(np.ascontiguousarray(lab, np.float32), cv2.COLOR_Lab2RGB), 0, 1)


def _smooth(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def _lch_of(rgb255) -> Tuple[float, float, float]:
    return dc.rgb_to_lch(tuple(float(v) for v in rgb255))


# ============================================================================================ the parts

def strays_mask(p, line_a: np.ndarray) -> Tuple[np.ndarray, int]:
    """Specks: small bits of the line layer that no line record draws (left over from cuts, deletes and stray
    clicks). A record's own short marks (a nose dot, a lash tip) are never touched; scatter and detail marks are
    meant and stay."""
    from .draw_line import record_path
    s = dc.work_scale(p)
    m = (line_a > 0.25).astype(np.uint8)
    k, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if k <= 1:
        return np.zeros((p.H, p.W), bool), 0
    drawn = np.zeros((p.H, p.W), np.uint8)
    for r in dc.dstate(p)["lines"]:
        P = record_path(p, r)
        if len(P) >= 1:
            cv2.polylines(drawn, [np.round(P).astype(np.int32).reshape(-1, 1, 2)], bool(r.get("closed")), 1,
                          max(3, int(4 * s)))
    lim_area = int((6 * s) ** 2)                     # under about 6 x 6 delivered pixels
    ids = []
    for i in range(1, k):
        if st[i, cv2.CC_STAT_AREA] >= lim_area:
            continue
        x, y, w, h = st[i, :4]
        if (drawn[y:y + h, x:x + w][lab[y:y + h, x:x + w] == i]).any():
            continue
        ids.append(i)
    out = np.isin(lab, ids) if ids else np.zeros((p.H, p.W), bool)
    if out.any():
        out = cv2.dilate(out.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=max(1, s)) > 0
    return out, len(ids)


def _fill_colour_under(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The picture with ``mask`` painted from its surroundings (what a line or speck lies on)."""
    u8 = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)
    H, W = mask.shape
    f = 4 if max(H, W) > 1600 else 2 if max(H, W) > 800 else 1
    if f > 1:
        small = cv2.resize(u8, (W // f, H // f), interpolation=cv2.INTER_AREA)
        sm = cv2.resize(mask.astype(np.float32), (W // f, H // f), interpolation=cv2.INTER_AREA) > 0.0
        sm = cv2.dilate(sm.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        inp = cv2.inpaint(small, sm.astype(np.uint8), 3, cv2.INPAINT_TELEA)
        big = cv2.resize(inp, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        big = cv2.inpaint(u8, mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    return big.astype(np.float32) / 255.0


def coloured_lines(img: np.ndarray, line_a: np.ndarray, strength: float) -> Tuple[np.ndarray, float]:
    """Line pixels take a dark shade of the colour beside them: the hue of the fill, about a quarter of its
    lightness, a little less chroma. Returns (picture, share of the canvas recoloured)."""
    a = np.clip(line_a, 0, 1)
    on = a > 0.2
    if not on.any():
        return img, 0.0
    under = _fill_colour_under(img, cv2.dilate(on.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    lab = _lab(under)
    L = lab[..., 0]
    lab[..., 0] = np.clip(0.24 * L, 5.0, 22.0)
    lab[..., 1:] *= 0.85
    col = _rgb(lab)
    cur = _lab(img)[..., 0]
    # only where the line is dark already (an inked line, not a light rim or a white catch-light)
    w = strength * a * _smooth(-cur, -55.0, -25.0)
    out = img * (1 - w[..., None]) + col * w[..., None]
    return out, float((w > 0.05).mean())


def depth_weights(p) -> Dict[str, float]:
    """subject -> how far it is (0 nearest .. 1 farthest) from the plan's depth order, in a picture with open air
    (a sky): the sky and a backdrop that fills the frame are not hazed (they are the air itself). Indoors and for a
    figure with what it holds there is no air to see through: nothing."""
    from .draw_checks import depth_order
    from .workflow_draw import is_ground
    d = dc.dstate(p)
    if not (any("sky" in s for s in d["subjects"]) or any("sky" in n for n in d["parts"])):
        return {}
    order = [s for s in depth_order(p) if s in d["subjects"]]
    keep = []
    for s in order:
        b = d["subjects"][s]["box"]
        full = (b[2] - b[0]) * (b[3] - b[1]) >= 0.8e6
        if any(w in s for w in ("sky", "background", "backdrop", "wall")) or (full and is_ground(s)) or full:
            continue
        keep.append(s)
    if len(keep) < 2:
        return {}
    n = len(keep) - 1
    return {s: 1.0 - i / n for i, s in enumerate(keep)}


def subject_mask(p, s: str) -> np.ndarray:
    from .draw_checks import region_subject
    m = np.zeros((p.H, p.W), bool)
    for n in p.state.get("regions", {}):
        if n == s or region_subject(p, n) == s:
            try:
                m |= p.region_mask(n)
            except Exception:  # noqa: BLE001
                continue
    return m


def haze_colour(p, img: np.ndarray) -> Tuple[int, int, int]:
    """The air: the sky swatch near the horizon (or the picture's top), a little lighter, greyer and cooler."""
    sw = dc.dstate(p)["palette"]["swatches"]
    base = None
    for n in ("sky-low", "sky", "background"):
        if n in sw:
            base = dc.from_hex(sw[n])
            break
    if base is None:
        top = img[: max(1, img.shape[0] // 6)].reshape(-1, 3)
        base = tuple(float(v) for v in np.median(top, 0) * 255)
    L, C, h = _lch_of(base)
    from .draw_scene import _toward
    return dc.lch_to_rgb(min(92.0, L + 10.0), 0.75 * C, _toward(h, 250.0, 0.1))


def light_colour(p, img_u8: np.ndarray) -> Tuple[int, int, int]:
    """The light's colour: a sun, lamp or light swatch (or the sky low by the sun), else a warm white."""
    d = dc.dstate(p)
    sw = d["palette"]["swatches"]
    for n in ("light", "sun", "lamp", "glow", "sky-low"):
        if n in sw:
            return dc.from_hex(sw[n])
    return WARM_WHITE           # no sun or lamp in the picture: a plain warm light (a backdrop or a blue sky is
    #                             not the light's colour)


def grade(img: np.ndarray, light_rgb, amount: float) -> np.ndarray:
    """Split toning toward one light: the lights lean toward its colour, the darks toward its cool complement."""
    lab = _lab(img)
    L = lab[..., 0]
    Ll, Cl, hl = _lch_of(light_rgb)
    lt = np.array(_lab(np.array([[np.asarray(light_rgb, np.float32) / 255.0]]))[0, 0])
    from .draw_scene import _toward
    cool = dc.lch_to_rgb(30.0, 22.0, _toward((hl + 180.0) % 360, 255.0, 0.6))
    ct = np.array(_lab(np.array([[np.asarray(cool, np.float32) / 255.0]]))[0, 0])
    # a tint, not a mix: the light's own colour direction is added to the lights and the shadow's to the darks,
    # so saturated colours keep their chroma and a white light changes almost nothing
    w_hi = amount * 0.35 * _smooth(L, 40.0, 92.0)
    w_lo = amount * 0.3 * _smooth(-L, -45.0, -8.0)
    for c in (1, 2):
        lab[..., c] = lab[..., c] + lt[c] * w_hi + ct[c] * w_lo
    # the lights lift a touch toward the light, the darks sink a touch
    lab[..., 0] = L + amount * (1.5 * _smooth(L, 60.0, 95.0) - 1.5 * _smooth(-L, -40.0, -5.0))
    return _rgb(lab)


def soften(img: np.ndarray, protect: np.ndarray, S: float, amount: float) -> Tuple[np.ndarray, float]:
    """Soft edges between colour areas (not on lines, lettering or the lit rims the protect mask holds)."""
    g = cv2.cvtColor(np.clip(img * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    e = cv2.Canny(g, 18, 48) > 0
    e &= ~protect
    if not e.any():
        return img, 0.0
    r = max(1.0, 0.0018 * S)
    wmap = cv2.GaussianBlur(e.astype(np.float32), (0, 0), 1.5 * r)
    wmap = np.clip(wmap / max(1e-6, float(np.quantile(wmap[e], 0.5))), 0, 1) * amount * 0.85
    wmap[protect] = 0.0
    blur = cv2.GaussianBlur(img, (0, 0), r)
    out = img * (1 - wmap[..., None]) + blur * wmap[..., None]
    return out, float((wmap > 0.1).mean())


def spur_mask(p, line_a: np.ndarray) -> Tuple[np.ndarray, int]:
    """Line ends that run on past a crossing: from a free end back to the first junction, when that run is
    shorter than a short dash (an outline that overshoots the line it meets). The pixels of those runs, widened
    to the line's width, and how many."""
    from skimage.morphology import skeletonize
    s = dc.work_scale(p)
    m = line_a > 0.4
    if not m.any():
        return np.zeros_like(m), 0
    sk = skeletonize(m)
    k = np.ones((3, 3), np.float32)
    k[1, 1] = 0
    nb = np.rint(cv2.filter2D(sk.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)).astype(int)
    ends = np.argwhere(sk & (nb == 1))
    junction = sk & (nb >= 3)
    lim = max(6, int(0.012 * max(p.W, p.H)))
    width = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
    out = np.zeros(m.shape, np.uint8)
    n = 0
    H, W = m.shape
    for y, x in ends:
        path = [(int(y), int(x))]
        seen = {path[0]}
        cy, cx = path[0]
        hit = False
        for _ in range(lim):
            nxt = None
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = cy + dy, cx + dx
                    if (dy or dx) and 0 <= yy < H and 0 <= xx < W and sk[yy, xx] and (yy, xx) not in seen:
                        if nxt is None or junction[yy, xx]:
                            nxt = (yy, xx)
            if nxt is None:
                break
            if junction[nxt]:
                hit = True
                break
            path.append(nxt)
            seen.add(nxt)
            cy, cx = nxt
        if not hit or len(path) < 2 * s:
            continue
        # the run up to (not into) the junction, as wide as the line there
        r = int(max(1.0, 1.4 * float(np.median([width[q] for q in path]))))
        keep = path[: max(1, len(path) - r)]
        for (py, px) in keep:
            cv2.circle(out, (int(px), int(py)), r, 1, -1)
        n += 1
    return (out > 0) & m, n


def clean(p, img: np.ndarray, protect: np.ndarray, amount: float) -> Tuple[np.ndarray, int]:
    """Streaks and patches inside each colour area are smoothed toward the area's own local colour (the colour is
    averaged inside the area only, so area edges stay where they are); lines, eyes and lettering are kept."""
    from .workflow_draw import is_text
    S = float(max(p.W, p.H))
    sig = max(2.0, 0.006 * S)
    out = img.copy()
    n = 0
    regs = [r for r in p.state.get("regions", {}) if not is_text(r) and not r.startswith(("eye", "iris", "pupil"))]
    for r in regs:
        try:
            m = p.region_mask(r) & ~protect
        except Exception:  # noqa: BLE001
            continue
        if m.sum() < 400:
            continue
        ys, xs = np.nonzero(m)
        pad = int(3 * sig) + 2
        y0, y1 = max(0, ys.min() - pad), min(p.H, ys.max() + 1 + pad)
        x0, x1 = max(0, xs.min() - pad), min(p.W, xs.max() + 1 + pad)
        mm = m[y0:y1, x0:x1].astype(np.float32)
        sub = out[y0:y1, x0:x1]
        num = cv2.GaussianBlur(sub * mm[..., None], (0, 0), sig)
        den = cv2.GaussianBlur(mm, (0, 0), sig)[..., None]
        local = num / np.maximum(den, 1e-4)
        w = amount * mm[..., None]
        out[y0:y1, x0:x1] = sub * (1 - w) + local * w
        n += 1
    return out, n


# ============================================================================================ command

def soft_style(p) -> bool:
    from .draw_surface import style_mode
    return style_mode(p) in ("paint", "soft")


def cmd_polish(ctx, args: List[str]):
    from . import cli, compose, fx
    from .cli import Spec, parse
    p = ctx.project
    u = USAGE["polish"]
    flags = {f"no-{k}": "flag" for k in PARTS}
    a = parse(args, Spec(u, [], dict(flags, grade="value", depth="value", soften="value", lines="value",
                                     clean="value"),
                         alias={"no-haze": "no-depth", "no-air": "no-depth", "no-colour": "no-lines",
                                "no-color": "no-lines", "no-specks": "no-strays", "no-soft": "no-soften",
                                "no-smooth": "no-clean",
                                "haze": "depth"}))
    on = {k: not a.get(f"no-{k}") for k in PARTS}
    if not any(on.values()):
        raise UsageError("every part is off: leave at least one of " + ", ".join(PARTS), u)
    amt_grade = cli._num(a, "grade", 0.6, 0.0, 1.0, ctx.notes, u)
    amt_depth = cli._num(a, "depth", 0.6, 0.0, 1.0, ctx.notes, u)
    amt_lines = cli._num(a, "lines", 0.6, 0.0, 1.0, ctx.notes, u)
    amt_clean = cli._num(a, "clean", 0.5, 0.0, 1.0, ctx.notes, u)
    soft_default = 0.6 if soft_style(p) else 0.0
    amt_soft = cli._num(a, "soften", soft_default, 0.0, 1.0, ctx.notes, u)
    below = p.composite("score", upto="fx")                     # the picture under the fx layer
    cur = np.array(p.composite("score", upto="text"), np.float32)
    img = cur[..., :3].copy()
    S = float(max(p.W, p.H))
    lines_out: List[str] = []
    line_arr = p.layer("line", create=False)
    line_a = line_arr[..., 3] if line_arr is not None and p.layer_entry("line").get("visible", True) else None
    text_arr = p.layer("text", create=False)
    protect = np.zeros((p.H, p.W), bool)
    if line_a is not None:
        protect |= line_a > 0.15
    if text_arr is not None:
        protect |= text_arr[..., 3] > 0.1

    if on["strays"] and line_a is not None:
        sm, n = strays_mask(p, line_a)
        sp, n2 = spur_mask(p, line_a)
        sm = sm | sp
        if n or n2:
            filled = _fill_colour_under(img, sm)
            img = np.where(sm[..., None], filled, img)
            line_a = np.where(sm, 0.0, line_a).astype(np.float32)
            protect &= ~sm
        lines_out.append(f"strays: {n} specks and {n2} line ends past a crossing painted out")
    elif on["strays"]:
        lines_out.append("strays: no line art, nothing to clean")
    if on["clean"] and amt_clean > 0:
        img, nr = clean(p, img, protect, amt_clean)
        lines_out.append(f"clean: streaks and patches smoothed inside {nr} colour areas (amount {amt_clean:g}; "
                         "edges, lines and lettering kept)")
    if on["lines"] and line_a is not None and amt_lines > 0:
        img, share = coloured_lines(img, line_a, amt_lines)
        lines_out.append(f"lines: line art coloured from the fills it borders ({share * 100:.1f}% of the canvas), "
                         f"strength {amt_lines:g}")
    elif on["lines"]:
        lines_out.append("lines: no line art to colour")
    if on["depth"] and amt_depth > 0:
        far = depth_weights(p)
        if far:
            hc = np.asarray(haze_colour(p, img), np.float32) / 255.0
            parts_txt = []
            for s, t in sorted(far.items(), key=lambda kv: -kv[1]):
                if t <= 0.01:
                    continue
                m = subject_mask(p, s)
                if not m.any():
                    continue
                w = cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(1.0, 0.0015 * S)) * (0.32 * amt_depth * t)
                img = img * (1 - w[..., None]) + hc * w[..., None]
                parts_txt.append(f"{s} {0.32 * amt_depth * t:.2f}")
            lines_out.append(f"depth: toward the air {coords.hex_colour(tuple(int(v * 255) for v in hc))}, far to near: "
                             + (", ".join(parts_txt) if parts_txt else "nothing far enough"))
        else:
            lines_out.append("depth: no open air between the things in this picture: nothing to set back")
    if on["grade"] and amt_grade > 0:
        lc = light_colour(p, np.clip(np.rint(img * 255), 0, 255).astype(np.uint8))
        before = img
        img = grade(img, lc, amt_grade)
        dE = float(np.abs(_lab(img) - _lab(before)).sum(-1).mean())
        lines_out.append(f"grade: lights toward the light {coords.hex_colour(tuple(int(v) for v in lc))}, darks "
                         f"toward its cool complement, amount {amt_grade:g} (mean change {dE:.1f} Lab)")
    if on["soften"]:
        if amt_soft > 0:
            img, share = soften(img, protect, S, amt_soft)
            lines_out.append(f"soften: soft edges between colour areas on {share * 100:.1f}% of the canvas "
                             f"(amount {amt_soft:g})")
        else:
            lines_out.append("soften: crisp edges kept (cel and ink style; --soften 0.4 to soften them)")
    target = cur.copy()
    target[..., :3] = np.clip(img, 0, 1)
    layer = fx.to_fx_layer(np.asarray(below, np.float32), target)

    def make(arr: np.ndarray) -> np.ndarray:
        ent = p.layer_entry("fx")
        ent.update({"mode": "normal", "opacity": 1.0, "visible": True, "clip": False, "alpha_lock": False})
        return layer

    def tick(_ops):
        d = dc.dstate(p)
        d["polished"] = {k: bool(v) for k, v in on.items()}
        from . import workflow_draw
        workflow_draw.refresh(p)
    region = cli.resolve_region(p, None, ctx.notes, u)
    off = [k for k in PARTS if not on[k]]
    label = "polish" + (" " + " ".join(f"--no-{k}" for k in off) if off else "")
    rep = cli.paint_step(ctx, label=label, layer="fx", build=lambda: [], clip=None, area_px=(0, 0, p.W, p.H),
                         region=region, pixel_fn=make, extra_lines=lines_out, key_metric="overall", on_apply=tick)
    missing = [ln for ln in lines_out if ln not in rep.lines]
    for k, ln in enumerate(missing):
        rep.lines.insert(1 + k, ln)
    rep.lines.append("each part can be left out: " + " ".join(f"--no-{k}" for k in PARTS) + ";  undo with  "
                     "easel undo")
    return rep


COMMANDS = {"polish": cmd_polish}
