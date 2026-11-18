"""Hand mode: the model draws every mark itself; the reference is something to look at, snap to and take colours
from.

``pixelcanvas mode hand`` (right after ``pixelcanvas new pic --ref ref.jpg``) switches a reference project to the drawing
commands and stages. On top of them this module adds:

- ``line "x,y ..." --snap ref``: a live-wire along the reference's edge between the model's points; the weight
  comes from the reference's line width there; the report says how much of the line sits on an edge.
- ``line candidates <region|box>`` / ``line take 3,5``: reference contours not drawn yet, as numbered marks.
- ``palette from-ref`` / ``palette name 3 hair``: the reference's dominant colours as numbered swatches.
- ``pick@x,y`` as a colour anywhere (median of the reference in a small disc); ``flat 3 --color pick``: the
  reference's median colour inside each area.
- ``shade <region> --like ref [--levels 2]``: shadow (and light) shapes copied from the reference in one area.
- ``detail candidates <region>`` / ``detail take 2,4``: small shapes the reference has and the canvas lacks.
- ``plan part <name> --marks 3,4``: a part box from the shell's numbered marks.
- ``look --lightbox``: the canvas over the faint reference; sketch and line steps show it automatically.
- scores against the reference and worst spots in every report, and gates on them (``gate``).
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from . import cli, coords, regions, trace, view, workflow
from . import draw_core as dc
from . import draw_line, draw_plan, draw_shade
from .cli import Ctx, Report, Spec, parse
from .coords import UsageError

MAX_TAKE = 10
GATES = {"line": {"lines": 40}, "flat": {"colour": 62}, "shade": {"colour": 66}, "detail": {"overall": 58}}
FIXES = {"line": "pixelcanvas line candidates {box}", "flat": "pixelcanvas flat --at {pt} --color pick",
         "shade": "pixelcanvas shade {region} --like ref", "detail": "pixelcanvas detail candidates {box}"}


def is_hand(obj) -> bool:
    st = obj if isinstance(obj, dict) else getattr(obj, "state", {}) or {}
    return st.get("mode") == "hand"


def _hand(ctx: Ctx) -> bool:
    try:
        return is_hand(ctx.project)
    except UsageError:
        return False


def _hs(p) -> dict:
    return dc.dstate(p).setdefault("hand", {"ref_swatches": [], "cands": {}})


# ============================================================================================ reference maps

def _cached(p, name: str, make):
    f = p.root / "cache" / f"hand-{name}.npy"
    if f.exists():
        try:
            return np.load(f)
        except (OSError, ValueError):
            pass
    a = make()
    f.parent.mkdir(exist_ok=True)
    np.save(f, a)
    return a


def edge_cost(p) -> np.ndarray:
    return _cached(p, "cost", lambda: regions.edge_cost(p.require_ref("line --snap ref")))


def ref_lines(p) -> np.ndarray:
    """The reference's dark line art (bool)."""
    return _cached(p, "lines", lambda: trace.line_map(p.require_ref("hand mode")).mask)


def ref_edges(p) -> np.ndarray:
    """Thin strong colour edges of the reference (the crest of the live-wire's cheap ground): black hair against a
    red bow counts, not only light against dark."""
    return _cached(p, "edges", lambda: skeletonize(edge_cost(p) < 0.25))


def edge_dist(p) -> np.ndarray:
    """Distance (px) to the nearest strong reference edge or line."""
    return _cached(p, "edist", lambda: ndi.distance_transform_edt(~(ref_edges(p) | ref_lines(p))).astype(np.float32))


def line_widths(p) -> np.ndarray:
    """Width (px) of the reference line art on its skeleton, 0 elsewhere."""
    def make():
        m = ref_lines(p)
        sk = skeletonize(m)
        return np.where(sk, 2 * ndi.distance_transform_edt(m) - 1, 0).astype(np.float32)
    return _cached(p, "widths", make)


def ref_weight_global(p) -> float:
    """The reference's usual line width, as a draw_line weight."""
    w = line_widths(p)
    px = float(np.median(w[w > 0])) if (w > 0).any() else draw_line.base_width(p)
    return float(np.clip(px / draw_line.base_width(p), 0.3, 6.0))


WEIGHT_STEP = 1.25          # weights from the reference sit on a ladder: the usual line, then x1.25 steps


def steady_weight(p, w: float) -> float:
    """A weight from the reference put on the ladder around its usual line: noise within a step reads as the
    usual line, so pieces of one contour get the same weight."""
    g = ref_weight_global(p)
    k = round(float(np.log(max(w, 1e-3) / g) / np.log(WEIGHT_STEP)))
    return round(float(np.clip(g * WEIGHT_STEP ** k, 0.3, 6.0)), 2)


def cross_widths(p, path):
    """(width px, near) per sample of ``path``: the reference line art's width measured across the path (along its
    normal, so diagonal lines measure true), and whether there is line art within 2 px of the sample."""
    P = np.asarray(path, np.float64)
    t = np.gradient(P, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    step = 0.5
    off = np.arange(-max(6.0, 0.03 * max(p.W, p.H)), max(6.0, 0.03 * max(p.W, p.H)) + 1e-6, step)
    X = P[:, None, 0] - off[None] * t[:, None, 1]
    Y = P[:, None, 1] + off[None] * t[:, None, 0]
    v = ndi.map_coordinates(ref_lines(p).astype(np.float32), [Y.ravel() - 0.5, X.ravel() - 0.5], order=1,
                            mode="constant").reshape(X.shape) >= 0.5
    c, k = len(off) // 2, int(2 / step)
    win = v[:, c - k:c + k + 1]
    near = win.any(1)
    start = c - k + np.argmin(np.where(win, np.abs(np.arange(-k, k + 1))[None], 99), axis=1)
    idx = np.arange(len(off))[None]
    lo = np.where(~v & (idx <= start[:, None]), idx, -1).max(1)
    hi = np.where(~v & (idx >= start[:, None]), idx, len(off)).min(1)
    return np.where(near, (hi - lo - 1) * step, 0.0), near


def ref_profile(p, path):
    """(weight, wprof) of the reference's dark line along ``path`` (px); (None, None) where there is none. The
    width samples are smoothed and put on the weight ladder: one weight for the whole line, unless the reference's
    own line is a step wider or thinner over a long stretch (then wprof holds that change)."""
    if len(path) < 2:
        return None, None
    w, near = cross_widths(p, path)
    if near.mean() < 0.3:
        return None, None
    i = np.arange(len(w))
    ws = np.interp(i, i[near], w[near])
    ws = ndi.median_filter(ws, size=min(len(ws), 9), mode="nearest")
    weight = steady_weight(p, float(np.median(ws)) / draw_line.base_width(p))
    # each sample's step on the ladder relative to the line's weight; short changes are noise
    lev = np.round(np.log(np.maximum(ws, 0.3) / (weight * draw_line.base_width(p))) / np.log(WEIGHT_STEP))
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(np.asarray(path, float), axis=0).T))])
    L = s[-1]
    if L <= 0 or not lev.any():
        return weight, None
    min_run = max(15.0, 0.2 * L)
    if L < 2 * min_run:
        return weight, None
    for _ in range(4):
        cuts = np.flatnonzero(np.diff(lev)) + 1
        bounds = np.r_[0, cuts, len(lev)]
        runs = [(a, b) for a, b in zip(bounds[:-1], bounds[1:])]
        short = [(a, b) for a, b in runs if s[b - 1] - s[a] < min_run]
        if not short:
            break
        for a, b in short:
            lev[a:b] = lev[a - 1] if a > 0 else (lev[b] if b < len(lev) else 0)
    if not lev.any():
        return weight, None
    if len(set(lev.tolist())) == 1:
        return steady_weight(p, weight * WEIGHT_STEP ** float(lev[0])), None
    t = s / L
    ramp = min(0.1, 6.0 / L)
    knots = [[0.0, WEIGHT_STEP ** float(lev[0])]]
    for c in np.flatnonzero(np.diff(lev)) + 1:
        knots += [[max(0.0, t[c] - ramp), WEIGHT_STEP ** float(lev[c - 1])],
                  [min(1.0, t[c] + ramp), WEIGHT_STEP ** float(lev[c])]]
    knots.append([1.0, WEIGHT_STEP ** float(lev[-1])])
    return weight, [[round(a, 3), round(b, 3)] for a, b in knots]


def ref_L(p) -> np.ndarray:
    return trace.to_lab(p.require_ref("hand mode"))[..., 0]


# ============================================================================================ colours

def pick_colour(p, spec: str, example: str = "pixelcanvas flat 3 --color pick@420,310"):
    """``x,y`` (0-1000) -> the reference's median colour in a small disc there."""
    ref = p.ref
    if ref is None:
        raise UsageError("pick@ takes colours from the reference, and this project has none", example)
    pt = coords.parse_point(spec, None, example)
    x, y = coords.point_to_px(pt, p.W, p.H)
    r = max(2.0, 0.006 * max(p.W, p.H))
    yy, xx = np.ogrid[:p.H, :p.W]
    m = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
    if not m.any():
        raise UsageError(f"{spec} is outside the picture", example)
    return tuple(int(v) for v in np.median(ref[m], axis=0))


def area_colour(p, mask: np.ndarray) -> str:
    """The reference's median colour in ``mask``, leaving out its line art."""
    ref = p.require_ref("--color pick")
    m = np.asarray(mask, bool)
    inner = m & ~cv2.dilate(ref_lines(p).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    use = inner if inner.sum() > 0.2 * m.sum() and inner.any() else m
    return coords.hex_colour(np.median(ref[use], axis=0))


def pick_area(p, mask: np.ndarray) -> str:
    """``flat 3 --color pick``: the palette name with the reference's colour in the area, else its hex."""
    return _named(p, dc.from_hex(area_colour(p, mask)))


def _named(p, rgb) -> str:
    """A palette name that has (almost) this colour, else the hex."""
    best, bd = None, 1e9
    for n, hx in dc.dstate(p)["palette"]["swatches"].items():
        c = dc.from_hex(hx)
        dd = sum((int(a) - int(b)) ** 2 for a, b in zip(c, rgb)) ** 0.5
        if dd < bd:
            best, bd = n, dd
    return best if best and bd < 18 else coords.hex_colour(rgb)


# ============================================================================================ views

def lightbox(p, img: np.ndarray, amount: float = 0.3) -> np.ndarray:
    """The reference, faint, under ``img`` (multiply: lines and paint stay as they are, paper shows the ref)."""
    ref = p.ref
    if ref is None:
        return img
    faded = ref.astype(np.float32) * amount + 255.0 * (1 - amount)
    return np.clip(faded * img.astype(np.float32) / 255.0, 0, 255).astype(np.uint8)


def hand_views(ctx: Ctx, rep, before: np.ndarray, after: np.ndarray, area):
    """Called by the draw steps: in the sketch and line stages both views get the lightbox; every step also gets
    the reference over the same area as VIEW-REF."""
    p = ctx.project
    rep.view_ref = cli._ref_view(ctx, area)
    if p.state.get("stage") in ("plan", "sketch", "line"):
        return lightbox(p, before), lightbox(p, after)
    return before, after


def _draw_polys(img: np.ndarray, polys, area, colours, width_view: float = 3.0, fill: bool = False) -> np.ndarray:
    from .cli_draw import _zoom
    w = max(1, int(round(width_view / max(_zoom(area), 1e-3))))
    im = Image.fromarray(img.copy())
    dr = ImageDraw.Draw(im)
    for pts, col in zip(polys, colours):
        q = [tuple(map(float, xy)) for xy in pts]
        if len(q) < 2:
            continue
        if fill:
            dr.polygon(q, outline=col, width=w)
        else:
            dr.line(q, fill=col, width=w, joint="curve")
    return np.asarray(im)


CAND_COLOURS = [(230, 30, 120), (20, 140, 230), (240, 140, 0), (30, 170, 60), (150, 60, 220), (0, 170, 170)]


def _cand_view(ctx: Ctx, rep, img: np.ndarray, cands, area, title: str, note: str, fill: bool = False):
    from .cli_draw import _zoom, tag_image
    p = ctx.project
    # a close view of a small canvas: draw on an enlarged copy so the number tags stay small beside the lines
    k = int(np.clip(np.floor(_zoom(area) / 1.4), 1, 4))
    big = cv2.resize(img, (p.W * k, p.H * k), interpolation=cv2.INTER_LINEAR) if k > 1 else img
    area_k = tuple(int(v * k) for v in area)
    polys = [dc.u2px(p, c["path"]) * k for c in cands]
    cols = [CAND_COLOURS[i % len(CAND_COLOURS)] for i in range(len(cands))]
    out = _draw_polys(big, polys, area_k, cols, 3.0 if not fill else 2.0, fill)
    tags = []
    for c, poly, col in zip(cands, polys, cols):
        q = poly[len(poly) // 2] if not fill else poly.mean(0) + np.array([0, -0.012 * p.H * k])
        tags.append((str(c["n"]), (float(q[0]), float(q[1])), col))
    out = tag_image(out, tags, area_k, 13)
    rep.view = cli._save_view(ctx, view.render([view.Panel(out, title)], area_k), f"cands-{cli._view_counter(p):04d}")
    rep.view_note = f"{note}, area {cli._area_note(p, area)}"
    rep.view_ref = cli._ref_view(ctx, area)


def cmd_look_lightbox(ctx: Ctx, args: List[str]) -> Report:
    from .cli_draw import DReport
    a = parse(args, Spec("pixelcanvas look hair --lightbox", ["where*"], {"lightbox": "flag", "amount": "value"}))
    p = ctx.project
    p.require_ref("look --lightbox")
    region = cli.resolve_region(p, a.get("where"), ctx.notes, "pixelcanvas look hair --lightbox")
    area = (0, 0, p.W, p.H) if region.name == "all" else cli._panel_box(p, region.box_px)
    amt = coords.parse_float(a["amount"], "amount", 0.05, 0.9, ctx.notes) if a.get("amount") else 0.3
    rep = DReport()
    rep.add(f"lightbox {region.label}: your canvas over the faint reference, area {cli._area_note(p, area)}")
    img = lightbox(p, p.composite_u8("view"), amt)
    rep.view = cli._save_view(ctx, view.render([view.Panel(img, "LIGHTBOX: CANVAS OVER FAINT REFERENCE")], area),
                              f"look-{cli._view_counter(p):04d}")
    rep.view_note = "the canvas over the faint reference"
    rep.view_ref = cli._ref_view(ctx, area)
    rep.next = cli._finish_next(ctx)
    return rep


# ============================================================================================ scores / gates

def _cmp(p):
    return cli._comparison(p, p.composite_u8("score"))


def score_lines(p, stage: Optional[str] = None) -> List[str]:
    """Scores against the reference (whole picture, weakest regions) and the worst spots, for every report."""
    stage = stage or p.state.get("stage")
    if stage in ("plan", "sketch") or p.ref is None:
        return []
    cmp = _cmp(p)
    emph = cli.KEY_METRIC.get(stage)
    out = ["vs reference:  " + cli.metrics.fmt_scores(cmp.scores())]
    rows = []
    for n in p.state.get("regions", {}):
        s = cmp.scores(p.region_mask(n))
        if s.get("overall") is not None:
            rows.append((s["overall"], n, s))
    rows.sort(key=lambda r: r[0])
    if rows:
        out.append("weakest regions: " + " | ".join(
            f"{n} {o} ({min((k for k in s if k != 'overall' and s[k] is not None), key=lambda k: s[k])} "
            f"{min(v for k, v in s.items() if k != 'overall' and v is not None)})" for o, n, s in rows[:3]))
    spots = cli._focus_spots(cmp.worst(None, n=5, emphasis=emph), emph)[:3]
    out.extend(cli._spots_lines(spots, 3, emph))
    return out


def hand_tail(p, stage: str, rep) -> None:
    """The end of every hand-mode drawing report: findings that fight the reference go (the reference sets the
    values), the scores against it come in."""
    rep.lines = [ln for ln in rep.lines if "read as one shape in grey" not in str(ln)]
    try:
        leaks = leak_line(line_leaks(p)) if stage in ("plan", "sketch", "line") else None
    except Exception:  # noqa: BLE001
        leaks = None
    for ln in (split_line(p), leaks):
        if ln:
            rep.lines.insert(min(1, len(rep.lines)), ln)
    rep.lines.extend(score_lines(p, stage))


def _region_at(p, box_u) -> Optional[str]:
    cx, cy = coords.point_to_px(((box_u[0] + box_u[2]) / 2, (box_u[1] + box_u[3]) / 2), p.W, p.H)
    xi, yi = int(np.clip(cx, 0, p.W - 1)), int(np.clip(cy, 0, p.H - 1))
    for n in p.state.get("regions", {}):
        if p.region_mask(n)[yi, xi]:
            return n
    return None


def gate(p, stage: str) -> List[dict]:
    """Unmet items for hand mode: the stage's score against the reference (whole picture, and per region for
    colour), each with a command that raises it; areas a line cut off a filled area come first."""
    out = []
    sf = split_fix(p) if stage in ("flat", "shade", "detail") else None
    if sf:
        out.append({"region": None, "metric": "areas without their colour", "value": sf[1].split()[2].count(",") + 1,
                    "need": 0, "fix": sf[1]})
    need = GATES.get(stage)
    if not need or p.ref is None:
        return out
    cmp = _cmp(p)
    emph = cli.KEY_METRIC.get(stage)
    spots = cli._focus_spots(cmp.worst(None, n=5, emphasis=emph), emph) or cmp.worst(None, n=3)
    s = cmp.scores()
    for metric, v in need.items():
        if s.get(metric) is None or s[metric] >= v:
            continue
        fix = "pixelcanvas look --lightbox"
        if spots:
            b = spots[0]["box"]
            reg = _region_at(p, b) or "<region>"
            fix = FIXES[stage].format(box=coords.fmt_box(b), pt=f"{(b[0] + b[2]) // 2},{(b[1] + b[3]) // 2}",
                                      region=reg)
        out.append({"region": None, "metric": f"{metric} vs reference", "value": s[metric], "need": v, "fix": fix})
    if stage in ("flat", "shade") and "colour" in need:
        for n in p.state.get("regions", {}):
            sc = cmp.scores(p.region_mask(n)).get("colour")
            if sc is not None and sc < need["colour"] - 15:
                fx = (f"pixelcanvas shade {n} --like ref" if stage == "shade" else
                      f"pixelcanvas flat --at {_anchor_u(p, p.region_mask(n))} --color pick")
                out.append({"region": n, "metric": "colour vs reference", "value": sc, "need": need["colour"] - 15,
                            "fix": fx})
    return out


def hand_next(p, why: str, cmd: str):
    """The draw-mode NEXT, worded for hand mode (the reference's tools instead of inventing)."""
    why = why.replace("no reference: ", "")
    c = cmd.strip()
    if c.startswith("pixelcanvas palette make"):
        c = "pixelcanvas palette from-ref"
    elif c.startswith("pixelcanvas flat auto"):
        c, why = "pixelcanvas flat", "see the numbered areas, then: pixelcanvas flat 3,5,8 --color pick"
    elif (c.startswith("pixelcanvas line \"x,y") or c.startswith("pixelcanvas line trace")) and "--snap" not in c:
        part = re.search(r"ink (\S+) \(box ([\d,]+)\)", why)
        c = (f"pixelcanvas line candidates {part.group(2)}" if part else
             'pixelcanvas line "x,y x,y x,y ..." --snap ref' if "trace" in c else c.replace('..."', '..." --snap ref', 1))
        why += '   (or trace an edge yourself: pixelcanvas line "x,y x,y x,y" --snap ref)'
    elif c.startswith("pixelcanvas shade ") and "--like" not in c:
        c = re.sub(r"\s+--levels \d", "", c.split("   (")[0]) + " --like ref --levels 2"
    elif c.startswith("pixelcanvas sketch figure"):
        why += "   (with a reference a head ellipse is enough: pixelcanvas sketch head <subject>)"
    return why, c


def _anchor_u(p, mask) -> str:
    ax, ay = regions.label_anchor(mask)
    return f"{round(ax * 1000 / p.W)},{round(ay * 1000 / p.H)}"


# ============================================================================================ mode

def cmd_mode(ctx: Ctx, args: List[str]) -> Report:
    from . import workflow_draw
    from .cli_draw import DReport
    a = parse(args, Spec("pixelcanvas mode hand", ["mode"]))
    p = ctx.project
    want = str(a.get("mode") or "show").lower()
    rep = DReport()
    if want in ("show", "status"):
        rep.add(f"mode {p.state.get('mode') or ('trace' if p.ref is not None else 'draw')}")
        rep.next = cli._finish_next(ctx)
        return rep
    if want not in ("hand", "trace"):
        raise UsageError(f"unknown mode {want!r}: hand (you draw every mark, the reference guides) or trace",
                         "pixelcanvas mode hand")
    p.require_ref("mode hand")
    step = p.begin_step([])
    st = p.state
    if want == "hand":
        st["mode"] = "hand"
        st["stage"], st["stages_done"], st["finished"] = "plan", [], False
        dc.dstate(p)
        _hs(p)
        workflow_draw.build_todo(st)
    else:
        st.pop("mode", None)
        st["stage"], st["stages_done"], st["finished"] = "plan", [], False
        workflow.build_todo(st)
    p.end_step(step, {"cmd": ctx.cmdline})
    rep.mutated, rep.step = True, step
    if want == "hand":
        rep.add(f"step {step}  mode hand: you draw every mark; the reference is there to look at, snap to and "
                "take colours from")
        rep.add("stages: " + " > ".join(workflow_draw.plan_of(st)))
        rep.add("plan with boxes or marks:  pixelcanvas plan part hair --of girl --marks 3,4,9   (marks: pixelcanvas look "
                "--show marks)")
        rep.add("colours: pixelcanvas palette from-ref, then pixelcanvas palette name 3 hair; any colour can be pick@x,y")
    else:
        rep.add(f"step {step}  mode trace: the stage commands follow the reference")
    rep.next = cli._finish_next(ctx)
    return rep


# ============================================================================================ plan --marks

def _marks_box(ctx: Ctx, spec: str, example: str) -> str:
    p = ctx.project
    nums = coords.parse_int_list(spec, example)
    labels = cli._marks_labels(p)
    m = np.isin(labels, nums)
    if not m.any():
        raise UsageError(f"no marks {spec}; marks go from 1 to {int(labels.max())}", example)
    return coords.fmt_box(coords.px_box_to_units(coords.mask_bbox_px(m), p.W, p.H))


def marks_outline(p, spec: str, example: str) -> list:
    """The union of the numbered marks, a little grown, as outline polygons (units): a part made from marks covers
    those marks only, not their bounding box."""
    m = dc.dilate(np.isin(cli._marks_labels(p), coords.parse_int_list(spec, example)), max(1.5, 0.004 * max(p.W, p.H)))
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [dc.rnd(dc.px2u(p, dc.rdp(c.reshape(-1, 2).astype(float) + 0.5, 0.75)), 1) for c in cs if len(c) >= 3]


def wrap_plan(orig):
    def run(ctx, args):
        if _hand(ctx) and "--marks" in args:
            ex = "pixelcanvas plan part hair --of girl --marks 3,4,9"
            i = args.index("--marks")
            if i + 1 >= len(args):
                raise UsageError("--marks needs mark numbers", ex)
            box = _marks_box(ctx, args[i + 1], ex)
            ctx.notes.append(f"marks {args[i + 1]} -> box {box} (the part is the marks' outline)")
            if args and args[0].lower() == "part":
                draw_plan.PENDING_OUTLINE["poly"] = marks_outline(ctx.project, args[i + 1], ex)
            args = args[:i] + ["--box", box] + args[i + 2:]
        try:
            return orig(ctx, args)
        finally:
            draw_plan.PENDING_OUTLINE.clear()
    return run


# ============================================================================================ palette

def wrap_palette(orig):
    def run(ctx, args):
        sub = args[0].lower() if args else ""
        if _hand(ctx) and sub in ("set", "add"):        # palette set hair pick@420,300
            args = [coords.hex_colour(pick_colour(ctx.project, t[5:], "pixelcanvas palette set hair pick@420,300"))
                    if t.lower().startswith("pick@") else t for t in args]
        if not _hand(ctx) or sub not in ("from-ref", "fromref", "ref", "name"):
            return orig(ctx, args)
        return palette_name(ctx, args[1:]) if sub == "name" else palette_from_ref(ctx, args[1:])
    return run


def palette_from_ref(ctx: Ctx, args: List[str]) -> Report:
    from .cli_draw import state_step
    a = parse(args, Spec("pixelcanvas palette from-ref --n 8", [], {"n": "value"}))
    p = ctx.project
    ref = p.require_ref("palette from-ref")
    k = coords.parse_int(a["n"], "n", 3, 14, ctx.notes) if a.get("n") else 9
    labels, _ = trace.cluster_colors(ref, k, samples=40000)
    rows = []
    for i in range(k):
        m = labels == i
        if m.sum() > 0.002 * p.W * p.H:
            rows.append((float(m.mean()), coords.hex_colour(np.median(ref[m], axis=0)), m))
    rows.sort(key=lambda r: -r[0])
    hs = _hs(p)

    def mut():
        hs["ref_swatches"] = [h for _, h, _ in rows]
        return [f"{len(rows)} colours from the reference, most area first"]
    rep, step = state_step(ctx, "palette from-ref", mut)
    for i, (share, hx, m) in enumerate(rows, 1):
        ax, ay = regions.label_anchor(m)
        rep.add(f"  {i}  {hx}  {coords.colour_name(dc.from_hex(hx)):<14} {share * 100:4.1f}%  e.g. at "
                f"{round(ax * 1000 / p.W)},{round(ay * 1000 / p.H)}")
    if not ctx.batch:
        img = draw_plan.palette_image([{"scheme": "from the reference (number: palette name <n> <name>)",
                                        "swatches": {str(i): h for i, (_, h, _) in enumerate(rows, 1)}}])
        rep.view = cli._save_view(ctx, img, f"step-{step:04d}")
        rep.view_note = "numbered swatches from the reference"
        rep.view_ref = cli._ref_view(ctx, (0, 0, p.W, p.H))
    need = list(dict.fromkeys(draw_plan.needed_swatches(dc.dstate(p)) + ["background", "line"]))
    rep.next = (f"pixelcanvas palette name 1 {need[0]} 2 {need[1 % len(need)]} ...   (a number from the list for each "
                f"name: {', '.join(need)})")
    return rep


def palette_name(ctx: Ctx, args: List[str]) -> Report:
    from .cli_draw import _plan_view, state_step
    p = ctx.project
    ex = "pixelcanvas palette name 3 hair 5 skin"
    sw = _hs(p)["ref_swatches"]
    if not sw:
        raise UsageError("no swatches from the reference yet", "pixelcanvas palette from-ref")
    toks = [t for t in re.split(r"[\s=:]+", " ".join(args)) if t]
    if len(toks) < 2 or len(toks) % 2:
        raise UsageError("palette name needs pairs: a swatch number and a name", ex)
    pairs = []
    for num, name in zip(toks[::2], toks[1::2]):
        if not num.isdigit():
            num, name = name, num
        if not num.isdigit() or not 1 <= int(num) <= len(sw):
            raise UsageError(f"swatch numbers go from 1 to {len(sw)}", ex)
        pairs.append((re.sub(r"[^a-z0-9\-]+", "-", name.lower()).strip("-"), sw[int(num) - 1]))

    def mut():
        d = dc.dstate(p)
        for n, hx in pairs:
            d["palette"]["swatches"][n] = hx
        return ["; ".join(f"{n} = {hx} ({coords.colour_name(dc.from_hex(hx))})" for n, hx in pairs)]
    rep, step = state_step(ctx, "palette name", mut)
    return _plan_view(ctx, rep, step)


# ============================================================================================ lines

def _pop_snap(args: List[str]):
    out, snap = [], False
    i = 0
    while i < len(args):
        t = args[i]
        if t.lower() in ("--snap=ref", "--snap-ref"):
            snap = True
        elif t.lower() == "--snap" and i + 1 < len(args) and args[i + 1].lower() == "ref":
            snap = True
            i += 1
        else:
            out.append(t)
        i += 1
    return snap, out


def wire_path(p, pts_u, closed: bool = False) -> np.ndarray:
    P = dc.u2px(p, pts_u)
    if closed:
        P = np.vstack([P, P[:1]])
    # a corridor wide enough for a curve between two far points (a quarter circle bulges ~0.3 of its chord)
    seg = float(np.max(np.hypot(*np.diff(P, axis=0).T))) if len(P) > 1 else 0.0
    path = regions.livewire(edge_cost(p), P, corridor=max(6.0, 0.32 * seg + 0.01 * max(p.W, p.H)))
    # the wire runs pixel to pixel: smooth out its stair steps (a few px; the edge's own curves stay)
    path = dc.resample(path.astype(np.float64), 1.0)
    sig = max(4.0, 0.004 * max(p.W, p.H))
    if closed and len(path) > 8 and np.hypot(*(path[0] - path[-1])) < 2:
        q = dc.smooth_poly(path[:-1], sig, closed=True)
        path = np.vstack([q, q[:1]])
    elif len(path) > 6:
        path = dc.smooth_poly(path, sig, closed=False)
    return dc.resample(path, 1.5)


def on_edge(p, path) -> float:
    dist = edge_dist(p)
    q = np.round(np.asarray(path)).astype(int)
    q[:, 0] = np.clip(q[:, 0], 0, p.W - 1)
    q[:, 1] = np.clip(q[:, 1], 0, p.H - 1)
    return float((dist[q[:, 1], q[:, 0]] <= 2.0).mean()) if len(q) else 0.0


def _ink(ctx: Ctx, make, label: str, lines: List[str]):
    """Ink line records made by ``make()`` (inside the step) as one undoable step on the line layer."""
    from . import cli_draw as cd
    p = ctx.project
    d = dc.dstate(p)
    new: List[str] = []
    grown: List[str] = []

    def mutate():
        ops, bb = [], None
        before = {id(r) for r in d["lines"]}
        for rec in make():
            if id(rec) in before:
                grown.append(rec["id"])         # an existing line that the new one runs on from
            elif rec["id"] not in new:
                new.append(rec["id"])
                if not any(r is rec for r in d["lines"]):
                    d["lines"].append(rec)
            ops += draw_line.render_ops(p, rec)
            bb = coords.union_box(bb, cd._path_box(p, draw_line.record_path(p, rec)))
        return ops, bb
    rep = cd.draw_step(ctx, label=label, layer="line", owned=cd._line_owned(p, mutate), lines=lines,
                       post=cd._line_post(p), overlay=cd._line_overlay(p), area_extra=cd._subject_box_px(p),
                       note="the line art over the faint reference (new ids in blue, gaps magenta)")
    if new or grown:
        rep.lines.insert(1, "; ".join(x for x in (f"new: {', '.join(new)}" if new else "",
                                                  f"longer: {', '.join(dict.fromkeys(grown))}" if grown else "") if x))
    return rep


TAPERS = {(True, True): "both", (True, False): "start", (False, True): "end", (False, False): "none"}


def _free(taper: str):
    return [taper in ("both", "start"), taper in ("both", "end")]


def _on_ref(p, pts) -> bool:
    """Whether a stretch (px) lies on the reference's lines or strong edges."""
    q = np.round(np.asarray(pts, np.float64).reshape(-1, 2)).astype(int)
    if not len(q):
        return True
    return bool((edge_dist(p)[np.clip(q[:, 1], 0, p.H - 1), np.clip(q[:, 0], 0, p.W - 1)] <= 2.0).mean() >= 0.8)


def _near_weight(p, piece, r: float) -> Optional[float]:
    """The weight of the drawn line nearest to an end of ``piece`` (within ``r`` px)."""
    best = None
    for o in dc.dstate(p)["lines"]:
        op = draw_line.record_path(p, o)
        if len(op):
            dd = min(float(np.hypot(*(op - piece[e]).T).min()) for e in (0, -1))
            if dd <= r and (best is None or dd < best[0]):
                best = (dd, float(o.get("weight", 1.0)))
    return best[1] if best else None


def _place(p, path, ctrl_u, profile, *, taper: str, color: str, src: str, fallback: float):
    """Ink ``path`` once, as records added to the line list (returned with any line it grew): stretches your lines
    ink already are left out; an end that runs on from the end of a line of the same weight becomes part of that
    line (one record, one smooth path); other ends meet lines with a tangent curve, and lose their taper there. Ends
    a little further from a line meet it too when the reference's own line or edge bridges the gap.
    ``profile(path) -> (weight or None, wprof)``; with no reference line the weight is that of the line it meets,
    else ``fallback``."""
    d = dc.dstate(p)
    bw = draw_line.base_width(p)
    r = draw_line.join_radius(p)
    r2 = 2.5 * r                                # a gap the reference bridges
    out, notes = [], []
    w_all = profile(path)[0] or fallback
    others = [(draw_line.record_path(p, o), 0.5 * bw * (w_all + o.get("weight", 1.0)) + 1.0) for o in d["lines"]]
    pieces = draw_line.unshared(path, others, max(4.0, 0.01 * max(p.W, p.H)))
    if not pieces:
        return out, [f"{src}: your lines ink it already; left out"]
    if len(pieces) > 1 or len(pieces[0]) != len(path):
        notes.append(f"{src}: the stretch your lines ink already is left out (no doubled line)")
    for piece in pieces:
        free = _free(taper)
        weight, wprof = profile(piece)
        weight = weight or _near_weight(p, piece, r2) or fallback
        # a short piece is too little evidence for a change of weight along a contour: up to half or double, it
        # takes the weight of the line it runs on from
        span = 2.05 if dc.path_length(piece) < max(30.0, 0.06 * max(p.W, p.H)) else 1.3
        hit = None
        for e in (0, -1):
            for o in d["lines"]:
                if o.get("closed") or o.get("wprof") or wprof or (o.get("color") or "line") != color or \
                        not 1 / span <= float(o.get("weight", 1.0)) / weight <= span:
                    continue
                op = draw_line.record_path(p, o)
                oe = draw_line.continues(piece, e, op, r2)
                if oe is not None and (np.hypot(*(op[oe] - piece[e])) < r or
                                       _on_ref(p, dc.resample(np.array([op[oe], piece[e]]), 1.0))):
                    hit = (e, o, op, oe)
                    break
            if hit:
                break
        if hit:
            e, o, op, oe = hit
            far = -1 if e == 0 else 0
            piece, jn = draw_line._join(p, piece, o["id"], ends=(far,))
            if jn:
                free[1 if far == -1 else 0] = False
            A = op if oe == -1 else op[::-1]
            B = piece if e == 0 else piece[::-1]
            of = _free(o.get("taper", "both"))
            o["path"] = dc.rnd(dc.px2u(p, dc.rdp(draw_line.bridge(A, B), 0.25)), 2)
            o["ctrl"] = dc.rnd(list(o["ctrl"]) + [list(q) for q in np.asarray(ctrl_u, float)])
            o["taper"] = TAPERS[(of[0] if oe == -1 else of[1], free[1] if e == 0 else free[0])]
            o["n"] = dc.next_order(d)
            o["part"] = draw_line.guess_part(p, o)
            out.append(o)
            notes.append(f"{src} runs on from {o['id']}: one line now")
            continue
        piece, jn = draw_line._join(p, piece, None)
        for e, word in ((0, "start"), (-1, "end")):
            if jn and jn[0].startswith("the ends met") or any(j.startswith(word) for j in jn):
                continue
            # a wider gap: joined only where the reference's line or edge runs across it
            longer, jn2 = draw_line._join(p, piece, None, ends=(e,), r=r2)
            if jn2:
                Q = dc.resample(longer, 1.0)
                gap = Q[np.hypot(*(Q[:, None, :] - piece[None, ::2, :]).transpose(2, 0, 1)).min(1) > 1.0]
                if _on_ref(p, gap):
                    piece, jn = longer, jn + jn2
        if any(j.startswith("the ends met") for j in jn):
            free = [False, False]
        for k, word in ((0, "start"), (1, "end")):
            if any(j.startswith(word) for j in jn):
                free[k] = False
        rec = draw_line.make_record(p, piece, ctrl_u, weight=weight, taper=TAPERS[tuple(free)], color=color,
                                    closed=False, src=src)
        if wprof:
            rec["wprof"] = wprof
        d["lines"].append(rec)
        out.append(rec)
    return out, notes


def line_snap_ref(ctx: Ctx, args: List[str]) -> Report:
    ex = 'pixelcanvas line "420,300 380,420 400,560" --snap ref'
    a = parse(args, Spec(ex, ["pts*"], {"weight": "value", "taper": "value", "closed": "flag", "color": "value"},
                         alias={"colour": "color", "close": "closed", "width": "weight", "w": "weight"}, min_pos=1))
    p = ctx.project
    pts = np.asarray(coords.parse_points(a["pts"], ctx.notes, 2, ex), float)
    taper = str(a.get("taper") or "both").lower()
    if taper not in ("both", "start", "end", "none"):
        raise UsageError("--taper is both, start, end or none", ex)
    colour = str(a.get("color") or "line")
    draw_plan.swatch(p, colour, ex + " --color line")
    closed = bool(a.get("closed"))
    lines: List[str] = []

    def make():
        path = wire_path(p, pts, closed)
        share = on_edge(p, path)
        if a.get("weight"):
            w = coords.parse_float(a["weight"], "weight", 0.3, 6, ctx.notes)
            how, prof = "given", None
        else:
            w0, prof = ref_profile(p, path)
            if w0:
                w, how = w0, "from the reference's line there" + ("; its width changes along it, so does this line"
                                                                  if prof else "")
            else:
                w, how = round(ref_weight_global(p), 2), "the reference's usual line: no dark line right there"
        lines.append(f"on the reference edge: {share * 100:.0f}% of the line within 2 px"
                     + ("" if share >= 0.7 else "  (low: put the points closer to the edge you see in VIEW-REF)"))
        lines.append(f"weight {w:g} ({how})")
        if closed:
            rec = draw_line.make_record(p, path, pts, weight=w, taper=taper, color=colour, closed=True,
                                        src="snap ref")
            if prof:
                rec["wprof"] = prof
            return [rec]

        def profile(q):
            return (w, None) if a.get("weight") else ref_profile(p, q)
        recs, notes = _place(p, path, pts, profile, taper=taper, color=colour, src="the line", fallback=w)
        lines.extend(notes)
        return recs
    return _ink(ctx, make, f"line {len(pts)} points --snap ref", lines)


def ref_paths(mask: np.ndarray, min_len: float) -> list:
    """Line art -> long polylines [(pts, (start free, end free))]: on each connected piece of its skeleton, the
    longest path, then the longest path of what is left, and so on (junction clutter never splits a contour into
    crumbs); a branch is joined to the path it leaves, and only free ends taper."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra
    sk = skeletonize(np.asarray(mask, bool))
    ys, xs = np.nonzero(sk)
    n = len(ys)
    if n < 2:
        return []
    idx = -np.ones((sk.shape[0] + 2, sk.shape[1] + 2), np.int64)
    idx[ys + 1, xs + 1] = np.arange(n)
    I, J, Wt = [], [], []
    for dy, dx, w in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.4142), (1, -1, 1.4142)):
        nb = idx[ys + 1 + dy, xs + 1 + dx]
        ok = nb >= 0
        I.append(np.flatnonzero(ok))
        J.append(nb[ok])
        Wt.append(np.full(int(ok.sum()), w))
    I, J, Wt = np.concatenate(I), np.concatenate(J), np.concatenate(Wt)
    G = coo_matrix((np.r_[Wt, Wt], (np.r_[I, J], np.r_[J, I])), shape=(n, n)).tocsr()
    out: list = []
    _, lab = connected_components(G, directed=False)
    work = [np.flatnonzero(lab == c) for c in np.unique(lab)]
    while work:
        nodes = work.pop()
        if len(nodes) < min_len * 0.7:
            continue
        sub = G[nodes][:, nodes]
        d0 = dijkstra(sub, indices=0)
        a = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
        da, pred = dijkstra(sub, indices=a, return_predecessors=True)
        b = int(np.argmax(np.where(np.isfinite(da), da, -1)))
        if da[b] < min_len:
            continue
        chain = [b]
        while chain[-1] != a and pred[chain[-1]] >= 0:
            chain.append(int(pred[chain[-1]]))
        gi = nodes[np.array(chain[::-1])]
        pts = np.stack([xs[gi] + 0.5, ys[gi] + 0.5], 1).astype(float)
        # a branch ends where it left a longer path: join it there, and only free ends taper
        free = [True, True]
        if out:
            allp = np.vstack([q for q, _ in out])
            for e in (0, -1):
                dist = np.hypot(allp[:, 0] - pts[e, 0], allp[:, 1] - pts[e, 1])
                k = int(np.argmin(dist))
                if dist[k] <= 5.0:
                    free[e] = False
                    pts = np.vstack([allp[k:k + 1], pts]) if e == 0 else np.vstack([pts, allp[k:k + 1]])
        out.append((pts, tuple(free)))
        # drop the path and the skeleton beside it, then look again at what is left
        dd = np.min(np.hypot(xs[nodes][:, None] - xs[gi][None, :], ys[nodes][:, None] - ys[gi][None, :]), 1) \
            if len(nodes) * len(gi) < 4e6 else np.zeros(len(nodes))
        rest = nodes[dd > 2.5]
        if len(rest) and len(out) < 400:
            _, l2 = connected_components(G[rest][:, rest], directed=False)
            work.extend(rest[l2 == c] for c in np.unique(l2))
    return out


def _no_all(spec, ex):
    if spec is None or str(spec).strip().lower() in ("", "all", "canvas", "whole", "everything", "*", "every"):
        raise UsageError("hand mode works one area at a time: name a region, a plan part, a box or a mark", ex)


def _area(ctx: Ctx, spec, ex):
    """A region, plan part, box, mark (m12) or flat area (a12, the numbers of  pixelcanvas flat)."""
    from .cli_draw import region_or_part
    _no_all(spec, ex)
    m = re.fullmatch(r"\s*a(?:rea)?\s*(\d+)\s*", str(spec), re.I)
    if m:
        from . import draw_flat
        p = ctx.project
        labels, table, _ = draw_flat.compute_areas(p)
        k = int(m.group(1))
        if not 1 <= k <= len(table):
            raise UsageError(f"there is no area {k}; areas go from 1 to {len(table)} (pixelcanvas flat shows them)", ex)
        mask = labels == k
        return cli.Region(f"a{k}", mask, coords.mask_bbox_px(mask), f"area {k}", True)
    return region_or_part(ctx, str(spec), ex, allow_box=True)


PAGE = 15           # tags per candidate view


def _cand_cover(a: dict, b: dict, tol: float):
    """(share of a's samples within ``tol`` of b, share of b's within ``tol`` of a) for two candidates (paths in
    units); (0, 0) when their kinds differ."""
    if a.get("kind") != b.get("kind") or not a.get("path") or not b.get("path"):
        return 0.0, 0.0
    A = dc.resample(np.asarray(a["path"], float), 2.0)
    B = dc.resample(np.asarray(b["path"], float), 2.0)
    d = np.hypot(A[:, None, 0] - B[None, :, 0], A[:, None, 1] - B[None, :, 1])
    return float((d.min(1) <= tol).mean()), float((d.min(0) <= tol).mean())


def _number(p, kind: str, found: list, mask: np.ndarray, tol: float, complete: bool = True, keep=None) -> list:
    """Fixed candidate numbers. ``found`` (in place order) is matched against the candidates listed before (the
    store _hs(p)["cands"][kind] holds number n at index n - 1):
    - the same shape keeps its number;
    - a piece of an earlier candidate (an area border cut it) shows that whole candidate, while ``keep(old)`` says it
      is still undrawn;
    - a shape that holds an earlier piece takes that piece's number;
    - anything else gets the next number after the highest ever used.
    Earlier candidates in this area that are not found again are gone (drawn, or the area changed), when ``found``
    is the complete list. Taken and gone numbers are never reused. Returns the listed candidates by number."""
    store = _hs(p)["cands"]
    old = list(store.get(kind) or [])
    live = [c for c in old if c.get("path") and not c.get("taken") and not c.get("gone")]
    cov = {(i, j): _cand_cover(f, c, tol) for i, f in enumerate(found) for j, c in enumerate(live)}
    got, used, shown = {}, set(), []
    for (i, j), (fa, fb) in sorted(cov.items(), key=lambda kv: -min(kv[1])):
        if min(fa, fb) >= 0.6 and i not in got and j not in used:
            got[i] = live[j]["n"]
            used.add(j)
    for (i, j), (fa, fb) in sorted(cov.items(), key=lambda kv: -kv[1][0]):
        if fa >= 0.9 and i not in got and (j in used or keep is None or keep(live[j])):
            got[i] = None                                   # a piece of live[j]: that one is shown
            if j not in used:
                used.add(j)
                shown.append(live[j])
    for (i, j), (fa, fb) in sorted(cov.items(), key=lambda kv: live[kv[0][1]]["n"]):
        if fb >= 0.9 and j not in used:
            used.add(j)
            if i not in got:
                got[i] = live[j]["n"]
            else:
                old[live[j]["n"] - 1] = {"n": live[j]["n"], "gone": True}
    for j, c in enumerate(live):
        if j in used or not complete:
            continue
        q = np.round(dc.u2px(p, c["path"])).astype(int)
        if mask[np.clip(q[:, 1], 0, p.H - 1), np.clip(q[:, 0], 0, p.W - 1)].mean() >= 0.5:
            old[c["n"] - 1] = {"n": c["n"], "gone": True}
    for c in old:
        if c.get("taken"):
            c.pop("path", None)
    for i, f in enumerate(found):
        if got.get(i, 0) is None:
            continue
        if i in got:
            f["n"] = got[i]
            old[f["n"] - 1] = f
        else:
            f["n"] = len(old) + 1
            old.append(f)
        shown.append(f)
    store[kind] = old
    return sorted(shown, key=lambda c: c["n"])


def _page(shown: list, page: int, cmd: str, ex: str):
    """(this page's candidates, report line or None) for a listing of more than PAGE candidates."""
    pages = max(1, -(-len(shown) // PAGE))
    if page > pages:
        raise UsageError(f"there {'is' if pages == 1 else 'are'} {pages} page{'s' * (pages > 1)} of candidates here",
                         ex)
    items = shown[(page - 1) * PAGE:page * PAGE]
    if pages == 1:
        return items, None
    more = (f"next page: {cmd} --page {page + 1}" if page < pages else f"first page: {cmd} --page 1")
    return items, (f"page {page} of {pages} ({PAGE} per page): numbers {items[0]['n']}-{items[-1]['n']}; {more}   "
                   "(or name a smaller area)")


def line_candidates(ctx: Ctx, args: List[str]) -> Report:
    from .cli_draw import DReport
    ex = "pixelcanvas line candidates hair"
    a = parse(args, Spec(ex, ["where*"], {"page": "value"}, min_pos=1))
    p = ctx.project
    page = coords.parse_int(a["page"], "page", 1, 99, ctx.notes) if a.get("page") else 1
    r = _area(ctx, a["where"], ex)
    S = max(p.W, p.H)
    zone = dc.dilate(r.mask, 0.01 * S)
    rl = ref_lines(p)
    # the line art, and the edges of flat masses (hair against skin) where there is no drawn line
    lm = (rl | (ref_edges(p) & ~dc.dilate(rl, 2))) & zone
    is_area = r.name.startswith("a") and r.name[1:].isdigit()
    if is_area:
        # a flat area that holds two reference colours: the border between them is where a line is missing
        minor, share, dist = colour_split(p, r.mask)
        if dist >= 6 and share >= 0.02:
            mn = minor.astype(np.uint8)
            border = (cv2.dilate(mn, np.ones((3, 3), np.uint8)) > 0) & ~minor & r.mask & ~dc.dilate(rl, 1)
            lm |= skeletonize(dc.dilate(border, 1))
    wid = line_widths(p)
    line = p.layer("line", create=False)
    drawn = (line[..., 3] > 0.3) if line is not None else np.zeros((p.H, p.W), bool)
    dd = ndi.distance_transform_edt(~drawn) if drawn.any() else np.full((p.H, p.W), 1e9)
    # only the parts not drawn yet: a contour half inked offers its missing half
    lm &= dd > max(3.0, 0.006 * S)
    # in a flat area (a12) even short pieces count: they are the leaks between its colours
    polys = ref_paths(lm, (0.005 if is_area else 0.015) * S)
    cands = []
    for pts, free in polys:
        q = np.round(pts).astype(int)
        xs, ys = np.clip(q[:, 0], 0, p.W - 1), np.clip(q[:, 1], 0, p.H - 1)
        if (dd[ys, xs] <= max(3.0, 0.006 * S)).mean() > 0.5 or r.mask[ys, xs].mean() < 0.5:
            continue
        w = wid[ys, xs]
        # an end tapers only where the reference's own line thins out there (marker lines and lettering do not)
        if (w > 0).sum() >= 6:
            k_ = max(2, len(w) // 6)
            mid = float(np.median(w[w > 0]))
            thin = [float(np.median(seg[seg > 0])) < 0.7 * mid if (seg > 0).any() else True
                    for seg in (w[:k_], w[-k_:])]
            free = (free[0] and thin[0], free[1] and thin[1])
        sm, _ = trace.smooth_polyline(pts, window=5)
        simp = dc.rdp(sm.astype(float), 0.6)
        cands.append({"path": dc.rnd(dc.px2u(p, simp), 1), "len": dc.path_length(pts),
                      "taper": {(True, True): "both", (True, False): "start", (False, True): "end"}.get(free, "none"),
                      "w": float(np.median(w[w > 0])) if (w > 0).any() else
                      ref_weight_global(p) * draw_line.base_width(p)})
    cands.sort(key=lambda c: -c["len"])
    complete = len(cands) <= 3 * PAGE
    cands = cands[:3 * PAGE]
    cands.sort(key=lambda c: (round(c["path"][0][1] / 60), c["path"][0][0]))
    for c in cands:
        c["len"] = round(c["len"], 1)
    # numbers stay fixed between listings: a contour shown before keeps its number, new ones count on

    def undrawn(c):
        q = np.round(dc.u2px(p, c["path"])).astype(int)
        return (dd[np.clip(q[:, 1], 0, p.H - 1), np.clip(q[:, 0], 0, p.W - 1)] > max(3.0, 0.006 * S)).mean() >= 0.8
    cands = _number(p, "line", cands, r.mask, 5.0, complete, undrawn)
    p.save_state()
    shown, page_note = _page(cands, page, f"pixelcanvas line candidates {a['where']}", ex + " --page 2")
    rep = DReport()
    rep.add(f"{len(cands)} reference contours in {r.label} not drawn yet (numbers stay fixed until taken)")
    if not cands:
        rep.add("nothing left there: every dark reference line in it is near a line of yours")
    for i in range(0, len(shown), 6):
        rep.add("  " + " | ".join(f"{c['n']} len {c['len'] * 1000 / S:.0f}" for c in shown[i:i + 6]))
    if page_note:
        rep.add(page_note)
    if not ctx.batch:
        area = cli._panel_box(p, r.box_px)
        _cand_view(ctx, rep, lightbox(p, p.composite_u8("view"), 0.25), shown, area,
                   "LINE CANDIDATES (numbered) OVER YOUR LINES", "numbered reference contours you have not drawn")
    cands = shown
    some = ",".join(str(c["n"]) for c in cands[:4]) or "1"
    rep.next = (f"pixelcanvas line take {some}   (only the ones that belong to the drawing; at most {MAX_TAKE})"
                if cands else cli._finish_next(ctx))
    return rep


def _take_nums(ctx, args, kind: str, have: int, ex: str, taken=None) -> List[int]:
    txt = " ".join(args).strip()
    if txt.lower() in ("all", "every", "*"):
        raise UsageError(f"choose the {kind} one by one (at most {MAX_TAKE} per command)", ex)
    nums = coords.parse_int_list(txt, ex) if txt else []
    if not nums:
        raise UsageError(f"say which {kind} to take, by number", ex)
    if len(nums) > MAX_TAKE:
        raise UsageError(f"at most {MAX_TAKE} per command; you gave {len(nums)}", ex)
    bad = [n for n in nums if not 1 <= n <= have]
    if bad:
        raise UsageError(f"no {kind} {', '.join(map(str, bad))}; they go from 1 to {have}", ex)
    done = [n for n in nums if taken and taken[n - 1].get("taken")]
    if done:
        raise UsageError(f"{', '.join(map(str, done))} already taken; take the others or list again", ex)
    gone = [n for n in nums if taken and (taken[n - 1].get("gone") or not taken[n - 1].get("path"))]
    if gone:
        raise UsageError(f"{', '.join(map(str, gone))} {'is' if len(gone) == 1 else 'are'} no longer a candidate "
                         "(drawn, or the area changed); list the area again for the current numbers", ex)
    return list(dict.fromkeys(nums))


def line_take(ctx: Ctx, args: List[str]) -> Report:
    ex = "pixelcanvas line take 3,5,8"
    p = ctx.project
    cands = _hs(p)["cands"].get("line") or []
    if not cands:
        raise UsageError("no line candidates yet: list them for one area first", "pixelcanvas line candidates hair")
    weight = None
    if "--weight" in args:
        i = args.index("--weight")
        weight = coords.parse_float(args[i + 1] if i + 1 < len(args) else "", "weight", 0.3, 6, ctx.notes,
                                    ex + " --weight 1.2")
        args = args[:i] + args[i + 2:]
    nums = _take_nums(ctx, args, "candidates", len(cands), ex, cands)
    lines: List[str] = []

    def make():
        out = []
        # the longest first: shorter pieces then run on from them or meet them
        for n in sorted(nums, key=lambda k: -float(cands[k - 1].get("len", 0))):
            c = cands[n - 1]
            c["taken"] = True
            path = dc.resample(dc.u2px(p, c["path"]), 1.5)

            def profile(q):
                return (weight, None) if weight else ref_profile(p, q)
            # ends meet drawn lines with a tangent curve (no hairline gaps for the fills, no kinks); stretches
            # already inked are left out; a piece that runs on from a line's end becomes part of that line
            recs, notes = _place(p, path, c["path"][:: max(1, len(c["path"]) // 8)], profile,
                                 taper=c.get("taper", "both"), color="line", src=f"candidate {n}",
                                 fallback=weight or steady_weight(p, ref_weight_global(p)))
            lines.extend(notes)
            out.extend(recs)
        return out
    rep = _ink(ctx, make, f"line take {','.join(map(str, nums))}", lines)
    left = [str(c["n"]) for c in cands if c.get("path") and not c.get("taken") and not c.get("gone")]
    rep.lines.insert(2, f"candidates not taken: {','.join(left[:20]) or 'none'} (numbers stay fixed; listing an "
                        "area again keeps them)")
    return rep


def wrap_line(orig):
    def run(ctx, args):
        if not _hand(ctx):
            return orig(ctx, args)
        sub = args[0].lower() if args else ""
        if sub in ("candidates", "cands", "candidate"):
            return line_candidates(ctx, args[1:])
        if sub == "take":
            return line_take(ctx, args[1:])
        snap, rest = _pop_snap(args)
        if snap:
            return line_snap_ref(ctx, rest)
        return orig(ctx, args)
    return run


# ============================================================================================ flat

def colour_split(p, mask: np.ndarray):
    """The reference inside ``mask`` as two colour groups: (minority bool mask, its share, Lab distance)."""
    lab = trace.to_lab(p.ref)
    m = np.asarray(mask, bool) & ~dc.dilate(ref_lines(p), 1)
    if m.sum() < 50:
        return np.zeros_like(m), 0.0, 0.0
    v = np.ascontiguousarray(lab[m], np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    cv2.setRNGSeed(1)
    # three groups, so stray dark line pixels take the third one; the two biggest are the area's colours
    k = 3 if len(v) >= 3 else 1
    _, lb, cen = cv2.kmeans(v[:: max(1, len(v) // 20000)], k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    d = ((v[:, None, :] - cen[None]) ** 2).sum(-1).argmin(1)
    order = np.argsort(-np.bincount(d, minlength=k))
    if k < 2:
        return np.zeros_like(m), 0.0, 0.0
    big, minor = int(order[0]), int(order[1])
    out = np.zeros_like(m)
    out[m] = d == minor
    return out, float((d == minor).mean()), float(np.sqrt(((cen[big] - cen[minor]) ** 2).sum()))


# ============================================================================================ leaks / split areas

def line_leaks(p) -> List[str]:
    """Where the line art is open (0-1000 points) and the one command that closes it: the gaps of ``line gaps``,
    and open ends where the reference's edge goes on (a snapped line from the end to the nearest line)."""
    if not dc.dstate(p)["lines"]:
        return []
    an = draw_line.analyze(p)
    u = lambda q: draw_line._u(p, q)  # noqa: E731
    out = [f"{g['id']} at {u(g['a']['pt'])}: pixelcanvas line close {g['id']}" for g in an["gaps"]]
    dist = edge_dist(p)
    for e in an["stray"]:
        P = an["paths"][e["line"]]
        tail = P[-6:] if e["which"] == "end" else P[:6][::-1]
        v = tail[-1] - tail[0]
        if e["near"] is None or len(tail) < 2 or np.hypot(*v) < 1e-6:
            continue
        q = tail[-1] + v / np.hypot(*v) * 0.015 * max(p.W, p.H)
        if dist[int(np.clip(q[1], 0, p.H - 1)), int(np.clip(q[0], 0, p.W - 1))] > 2.0:
            continue        # the reference's edge stops there too: a free end, not a leak
        out.append(f"{e['line']} {e['which']} at {u(e['pt'])}: pixelcanvas line \"{u(e['pt'])} {u(e['near'][1])}\" "
                   "--snap ref")
    return out


def area_leaks(p, n_max: int = 4) -> List[str]:
    """Areas that run into a neighbour or the background (two reference colours in big pieces), each with the
    snapped line along the reference's border between the two colours."""
    from . import draw_flat
    labels, table, _ = draw_flat.compute_areas(p)
    S = max(p.W, p.H)
    rl = ref_lines(p)
    out = []
    for r in sorted(table, key=lambda r: -r["share"]):
        if r["share"] < 0.004:
            continue
        m = labels == r["n"]
        minor, share, dist = colour_split(p, m)
        if not (dist >= 6 and share >= 0.015 and minor.sum() > 0.0015 * p.W * p.H):
            continue
        border = dc.dilate(minor, 1) & ~minor & m & ~dc.dilate(rl, 1)
        paths = ref_paths(dc.dilate(border, 1), 0.01 * S)
        if paths:
            q = max((q for q, _ in paths), key=dc.path_length)
            pts = [draw_line._u(p, q[i]) for i in (0, len(q) // 2, -1)]
            out.append(f"area {r['n']} at {pts[1]}: pixelcanvas line \"{' '.join(pts)}\" --snap ref")
        else:
            out.append(f"area {r['n']} at {_anchor_u(p, minor)}: pixelcanvas line candidates a{r['n']}")
        if len(out) >= n_max:
            break
    return out


def leak_line(items: List[str]) -> Optional[str]:
    if not items:
        return None
    return (f"LEAKS ({len(items)}, close them before flat colour): " + " | ".join(items[:6])
            + (f" | +{len(items) - 6} more" if len(items) > 6 else ""))


def split_fix(p) -> Optional[Tuple[str, str]]:
    """(why, command) when a line split filled areas and the cut-off pieces lost their colour."""
    from . import draw_flat
    if not dc.dstate(p).get("fills"):
        return None
    labels, table, _ = draw_flat.compute_areas(p)
    lost = draw_flat.split_pieces(p, labels, table)
    if not lost:
        return None
    nums = ",".join(str(r["n"]) for r in lost[:10])
    return (f"areas {nums} were cut off a filled area by a new line and have no colour of their own; give them "
            "their colour again", f"pixelcanvas flat {nums} --color pick")


def split_line(p) -> Optional[str]:
    try:
        sf = split_fix(p)
    except Exception:  # noqa: BLE001
        return None
    return f"FIX: {sf[0]}:  {sf[1]}" if sf else None


def wrap_flat(orig):
    def run(ctx, args):
        if not _hand(ctx):
            return orig(ctx, args)
        if args and args[0].lower() == "auto":
            raise UsageError("hand mode colours area by area: give the area numbers and a colour (pick = the "
                             "reference's colour inside each area)", "pixelcanvas flat 3,5,8 --color pick")
        rep = orig(ctx, args)
        if (rep.next or "").startswith("pixelcanvas flat auto"):
            rep.next = "pixelcanvas flat 1,2,3 --color pick   (area numbers from VIEW; pick = the reference's colour in each)"
        try:
            ln = leak_line(area_leaks(ctx.project))
            if ln:
                rep.lines.insert(1, ln)
        except Exception:  # noqa: BLE001
            pass
        sl = split_line(ctx.project)
        if sl and sl not in rep.lines:
            rep.lines.insert(1, sl)
        return rep
    return run


# ============================================================================================ shade --like ref

def shade_like(ctx: Ctx, args: List[str]) -> Report:
    from . import cli_draw as cd
    ex = "pixelcanvas shade hair --like ref --levels 2"
    a = parse(args, Spec(ex, ["region*"], {"like": "value", "levels": "value", "soft": "value", "depth": "value"},
                         alias={"level": "levels", "softness": "soft", "threshold": "depth"}, min_pos=1))
    if str(a.get("like") or "").lower() not in ("ref", "reference"):
        raise UsageError("--like takes ref", ex)
    p = ctx.project
    ref = p.require_ref("shade --like ref")
    r = _area(ctx, a["region"], ex)
    levels = coords.parse_int(a["levels"], "levels", 1, 2, ctx.notes) if a.get("levels") else 1
    soft = coords.parse_float(a["soft"], "soft", 0, 1, ctx.notes) if a.get("soft") else 0.0
    thr = coords.parse_float(a["depth"], "depth", 2, 40, ctx.notes) if a.get("depth") else 7.0
    L = ref_L(p)
    lines_ = cv2.dilate(ref_lines(p).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    m = r.mask
    body = m & ~lines_
    if body.sum() < 30:
        raise UsageError(f"{r.label} is too small to shade", ex)
    # each flat colour in the area is its own base: the reference darker than that base is shadow, lighter is light
    flat = p.composite_u8("export", upto="shade")      # the canvas up to and with the flats
    lab_ref = trace.to_lab(ref)
    Lf = trace.to_lab(flat)[..., 0]
    min_px = max(12, int(0.00004 * p.W * p.H))
    k = np.ones((3, 3), np.uint8)
    keys = flat[..., 0].astype(np.int32) << 16 | flat[..., 1].astype(np.int32) << 8 | flat[..., 2]
    vals, cnt = np.unique(keys[body], return_counts=True)
    groups = [keys == v for v, c in zip(vals, cnt) if c >= min_px]

    def shape(sel):
        s = cv2.morphologyEx(sel.astype(np.uint8), cv2.MORPH_CLOSE, k)
        s = cv2.morphologyEx(s, cv2.MORPH_OPEN, k).astype(bool) & m
        return dc.clean_mask(s, min_px)
    tones = []
    for g in groups:
        gb = body & g
        base = float(np.median(Lf[gb]))
        tones.append(("shadow", shape(gb & (L < base - thr))))
        if levels >= 2:
            tones.append(("light", shape(gb & (L > base + thr))))
    info = []
    fe = 1.0 + soft * 0.06 * np.sqrt(max(1, m.sum()))
    layer = cd._shade_layer(p)

    def build():
        from .ops import Fill, mask_to_rle
        ops = []
        old = p.layer(layer, create=False)
        if old is not None and (old[..., 3][m] > 0.02).any():
            ops.append(Fill(mask_rle=mask_to_rle(m), color=(0, 0, 0), erase=True, feather=1.0))
        got = {}
        for name, t in tones:
            if not t.any():
                continue
            # every shape takes the reference's colour inside it: the shadow pixels are grouped by colour first
            # (an iris and a blush that touch stay apart), then each group is one fill
            v = np.ascontiguousarray(lab_ref[t], np.float32)
            kk = int(min(5, max(1, len(v) // 40)))
            if kk > 1:
                crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
                cv2.setRNGSeed(1)
                _, _, cen = cv2.kmeans(v[:: max(1, len(v) // 20000)], kk, None, crit, 2, cv2.KMEANS_PP_CENTERS)
                gl = ((v[:, None, :] - cen[None]) ** 2).sum(-1).argmin(1)
            else:
                gl = np.zeros(len(v), int)
            cols = {}
            for gi in range(kk):
                cm = np.zeros_like(t)
                cm[t] = gl == gi
                cm = dc.clean_mask(cm, min_px)
                if not cm.any():
                    continue
                col = tuple(int(c_) for c_ in np.median(ref[cm], axis=0))
                cols[col] = cols.get(col, np.zeros_like(t)) | cm
            if not cols:
                continue
            for col, cm in sorted(cols.items(), key=lambda kv: -int(kv[1].sum())):
                ops.append(draw_shade.mask_fill(cm, col, 1.0, fe))
            big = max(cols.items(), key=lambda kv: int(kv[1].sum()))[0]
            got.setdefault(name, []).append(f"{t.sum() / m.sum() * 100:.0f}% in {len(cols)} shape(s), mostly "
                                            f"{coords.hex_colour(big)}")
        for name in ("shadow", "light")[:levels]:
            info.append(f"{name}: " + (", ".join(got[name]) if name in got else
                                       f"none in the reference there (try --depth {max(2, thr - 3):g})"))
        return ops
    return cd.draw_step(ctx, label=f"shade {r.name} --like ref levels {levels} soft {soft:g}", layer=layer,
                        build=build, clip=m, post=lambda: ["; ".join(info)],
                        note="shadow shapes copied from the reference")


def wrap_shade(orig):
    def run(ctx, args):
        if not _hand(ctx):
            return orig(ctx, args)
        if any(t.lower().startswith("--like") for t in args):
            return shade_like(ctx, args)
        pos = [t for t in args if not t.startswith("-")]
        _no_all(pos[0] if pos else None, "pixelcanvas shade hair --like ref")
        return orig(ctx, args)
    return run


# ============================================================================================ detail candidates

def piece_marks(cm: np.ndarray, skel: bool = True) -> list:
    """One small shape (bool mask) -> the smallest legal marks covering it, in px: a dab at least 3 px wide, or
    strokes at least 1 px wide and 4 times as long as wide; a round shape with room for an outline stays a fill.
    Strokes follow the skeleton; what they leave uncovered (a short arm of a letter) gets its own dab or stroke."""
    ys, xs = np.nonzero(cm)
    # line width: the thickest point, or area over skeleton length (a 2 px mouth line has no inner pixel)
    w = max(1.0, 2.0 * float(ndi.distance_transform_edt(np.pad(cm, 1)).max()) - 1.0,
            len(ys) / max(1, int(skeletonize(cm).sum())))
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32) + 0.5)
    long = max(rw, rh) + 1.0
    if long < 2 * w:
        if long > 6 and len(ys) >= 24:
            return [{"kind": "fill"}]
        return [{"kind": "dab", "pts": [[float(cx), float(cy)]], "w": round(max(3.0, long), 2)}]
    paths = [dc.rdp(path, 0.5) for path, _ in ref_paths(cm, 1.0)] if skel else []
    out = [{"kind": "stroke", "pts": q.tolist(), "w": round(w, 2)} for q in paths if dc.path_length(q) >= 4 * w + 0.2]
    if not out:      # too short a skeleton: one stroke along the long axis, lengthened to 4 widths
        t = np.radians(ang if rw >= rh else ang + 90)
        h = max(long - w, 4 * w + 0.2) / 2
        dx, dy = np.cos(t) * h, np.sin(t) * h
        return [{"kind": "stroke", "pts": [[cx - dx, cy - dy], [cx + dx, cy + dy]], "w": round(w, 2)}]
    line = np.zeros(cm.shape, np.uint8)
    for mk in out:
        cv2.polylines(line, [np.round(np.asarray(mk["pts"]) - 0.5).astype(np.int32)], False, 1)
    n, lab = cv2.connectedComponents((cm & ~dc.dilate(line, (w - 1) / 2)).astype(np.uint8), connectivity=8)
    for i in range(1, n):
        if (lab == i).sum() >= 2:
            out += piece_marks(lab == i, skel=False)
    return out


def group_pieces(stats: np.ndarray, ids) -> List[List[int]]:
    """Connected pieces (cv2 stats rows) -> glyph groups: a smaller piece joins a bigger one it sits over or under
    (the dot of an i, a broken stroke) or touches within 1 px."""
    ids = list(ids)
    root = {i: i for i in ids}

    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i
    for a_ in range(len(ids)):
        for b_ in range(a_ + 1, len(ids)):
            i, j = ids[a_], ids[b_]
            s, b = (i, j) if stats[i, 4] <= stats[j, 4] else (j, i)
            if stats[s, 4] > 0.5 * stats[b, 4]:
                continue
            x0, y0, w0, h0 = stats[s, :4]
            x1, y1, w1, h1 = stats[b, :4]
            ox = min(x0 + w0, x1 + w1) - max(x0, x1)
            gx, gy = -ox, max(y0, y1) - min(y0 + h0, y1 + h1)
            if (ox >= 0.5 * min(w0, w1) and gy <= 0.5 * h1) or max(gx, gy) <= 1:
                root[find(s)] = find(b)
    groups: dict = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def detail_shapes(p, region) -> List[dict]:
    """Small shapes the reference has in ``region`` and the canvas lacks, as candidate dicts ("path": outline in
    units, "kind", "hex", "area", "score", "marks": the legal marks that draw it, in px). Specks too small for a
    fill (a nose dot, a mouth line) become a dab or a short stroke; lettering comes one letter per candidate."""
    ref = p.require_ref("detail candidates")
    S = max(p.W, p.H)
    Lr = ref_L(p)
    Lc = trace.to_lab(p.composite_u8("export"))[..., 0]
    rad = max(3, int(0.012 * S))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
    missing = np.abs(Lr - Lc) > 12
    hi = int(0.004 * p.W * p.H)
    cands = []
    for kind, op in (("dark", cv2.MORPH_BLACKHAT), ("light", cv2.MORPH_TOPHAT)):
        sel = (cv2.morphologyEx(Lr, op, ker) > 12) & region.mask
        n, lab, stats, _ = cv2.connectedComponentsWithStats(sel.astype(np.uint8), connectivity=8)
        ids = [i for i in range(1, n) if 2 <= stats[i, 4] <= hi and max(stats[i, 2], stats[i, 3]) <= 0.1 * S]
        for grp in group_pieces(stats, ids):
            gm = np.isin(lab, grp)
            area = int(gm.sum())
            if area > hi or missing[gm].mean() < 0.5:
                continue
            col = np.median(ref[gm], axis=0)
            marks = []
            for i in grp:
                cm = lab == i
                for mk in piece_marks(cm):
                    if mk["kind"] == "fill":
                        cs, _ = cv2.findContours(cm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        mk["pts"] = (max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float) + 0.5).tolist()
                    mk["pts"] = dc.rnd(dc.px2u(p, mk["pts"]), 2)
                    marks.append(mk)
            ys, xs = np.nonzero(gm)
            corners = np.concatenate([np.stack([xs + dx, ys + dy], 1) for dx in (0, 1) for dy in (0, 1)])
            hull = cv2.convexHull(corners.astype(np.float32)).reshape(-1, 2)
            cands.append({"path": dc.rnd(dc.px2u(p, hull), 1), "kind": kind, "hex": coords.hex_colour(col),
                          "area": area, "score": float(max(area, 12) * abs(Lr[gm] - Lc[gm]).mean()),
                          "marks": marks})
    return cands


def detail_ops(p, c: dict, lab_ref=None) -> list:
    """A detail candidate -> ops in the reference's colour: its dabs and strokes, its fills cut to that colour."""
    from .ops import Stroke
    rgb = dc.from_hex(c["hex"])
    marks = c.get("marks") or [{"kind": "fill", "pts": c["path"]}]
    ops = []
    for mk in marks:
        P = dc.u2px(p, mk["pts"]).astype(np.float32)
        if mk["kind"] == "fill":
            # only the pixels of the shape itself (a mouth's outline, not the paper inside it)
            lab = trace.to_lab(p.ref) if lab_ref is None else lab_ref
            m = dc.poly_mask(p.H, p.W, P)
            target = trace.to_lab(np.array([[rgb]], np.uint8))[0, 0]
            m &= np.sqrt(((lab - target) ** 2).sum(-1)) < 22
            ops.append(draw_shade.mask_fill(m, rgb, 1.0, 1.0))
        else:
            ops.append(Stroke(pts=P, width=float(mk["w"]), color=tuple(int(v) for v in rgb), brush="liner"))
    return ops


def detail_candidates(ctx: Ctx, args: List[str]) -> Report:
    from .cli_draw import DReport
    ex = "pixelcanvas detail candidates face"
    a = parse(args, Spec(ex, ["where*"], {"page": "value"}, min_pos=1))
    p = ctx.project
    page = coords.parse_int(a["page"], "page", 1, 99, ctx.notes) if a.get("page") else 1
    p.require_ref("detail candidates")
    r = _area(ctx, a["where"], ex)
    cands = detail_shapes(p, r)
    cands.sort(key=lambda c: -c["score"])
    complete = len(cands) <= 3 * PAGE
    cands = cands[:3 * PAGE]
    cands.sort(key=lambda c: (round(c["path"][0][1] / 60), c["path"][0][0]))
    hs = _hs(p)
    # numbers stay fixed between listings: a shape shown before keeps its number, new ones count on
    cands = _number(p, "detail", cands, r.mask, 3.0, complete)
    hs["cands"]["detail_region"] = r.name if r.name in p.state.get("regions", {}) else None
    p.save_state()
    shown, page_note = _page(cands, page, f"pixelcanvas detail candidates {a['where']}", ex + " --page 2")
    rep = DReport()
    rep.add(f"{len(cands)} small shapes the reference has in {r.label} and your canvas lacks (numbers stay fixed "
            "until taken)")
    for i in range(0, len(shown), 5):
        rep.add("  " + " | ".join(f"{c['n']} {c['kind']} {c['hex']}" for c in shown[i:i + 5]))
    if page_note:
        rep.add(page_note)
    if not ctx.batch:
        area = cli._panel_box(p, r.box_px)
        _cand_view(ctx, rep, p.composite_u8("view"), shown, area, "DETAIL CANDIDATES (numbered outlines)",
                   "numbered small shapes from the reference", fill=True)
    cands = shown
    rep.next = (f"pixelcanvas detail take {','.join(str(c['n']) for c in cands[:3])}   (the ones that belong; at most "
                f"{MAX_TAKE})" if cands else cli._finish_next(ctx))
    return rep


def detail_take(ctx: Ctx, args: List[str]) -> Report:
    from . import cli_draw as cd
    ex = "pixelcanvas detail take 2,4"
    p = ctx.project
    hs = _hs(p)
    cands = hs["cands"].get("detail") or []
    if not cands:
        raise UsageError("no detail candidates yet: list them for one area first", "pixelcanvas detail candidates face")
    nums = _take_nums(ctx, args, "candidates", len(cands), ex, cands)
    reg = hs["cands"].get("detail_region")

    def build():
        lab = trace.to_lab(p.ref)
        return [op for n in nums for op in detail_ops(p, cands[n - 1], lab)]

    def post():
        d = dc.dstate(p)
        if reg:
            cd._mark_list(d, "detailed", reg)
        for n in nums:
            cands[n - 1]["taken"] = True
        return [f"took {len(nums)} shapes" + (f" in {reg}" if reg else "")]
    return cd.draw_step(ctx, label=f"detail take {','.join(map(str, nums))}", layer="detail", build=build,
                        post=post, note="small shapes from the reference")


def wrap_detail(orig):
    def run(ctx, args):
        if _hand(ctx) and args:
            sub = args[0].lower()
            if sub in ("candidates", "cands", "candidate"):
                return detail_candidates(ctx, args[1:])
            if sub == "take":
                return detail_take(ctx, args[1:])
        return orig(ctx, args)
    return run


# ============================================================================================ look

def wrap_look(orig):
    def run(ctx, args):
        if any(t.lower() == "--lightbox" for t in args):
            return cmd_look_lightbox(ctx, args)
        return orig(ctx, args)
    return run


# ============================================================================================ install

def wrap_pick(orig):
    """Shell commands that parse colours themselves (text, stroke, fill, shape, gradient): pick@x,y -> hex."""
    def run(ctx, args):
        if _hand(ctx) and any(t.lower().startswith("pick@") for t in args):
            args = [coords.hex_colour(pick_colour(ctx.project, t[5:], 'pixelcanvas text "Hi" --box 100,50,900,150 '
                                                                         '--color pick@420,300'))
                    if t.lower().startswith("pick@") else t for t in args]
        return orig(ctx, args)
    return run


COMMANDS = {"mode": cmd_mode}
USAGE = {"mode": "pixelcanvas mode hand"}
WRAP = {"plan": wrap_plan, "palette": wrap_palette, "line": wrap_line, "ink": wrap_line, "flat": wrap_flat,
        "shade": wrap_shade, "detail": wrap_detail, "look": wrap_look, "text": wrap_pick, "stroke": wrap_pick,
        "fill": wrap_pick, "shape": wrap_pick, "gradient": wrap_pick}
