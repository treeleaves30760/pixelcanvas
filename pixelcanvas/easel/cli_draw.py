"""Commands for drawing without a reference picture, and the hook that routes them.

``install(cli.COMMANDS)`` puts these commands into easel's command table: ``plan``, ``palette`` (make / pick /
set), ``sketch``, ``line`` / ``ink``, ``flat``, ``shade``, ``paint`` and ``detail`` run the versions here when the
project has no reference (and the reference versions otherwise); ``shadow``, ``rim``, ``crease``, ``blush``,
``highlight``, ``hair``, ``eye``, ``fold``, ``pattern``, ``scatter``, ``band`` and ``haze`` are new. It also routes
the workflow functions (todo, gates, next) to ``workflow_draw`` for such projects.

Until the hook is in ``cli.py``: ``python -m pixelcanvas.easel.cli_draw <command> ...`` runs easel with it installed.

Every mutating command is one undoable step: ops on the stage's layer (the sketch and line layers are rebuilt from
their records), journaled, with the canvas BEFORE and AFTER as two separate pictures (``VIEW-BEFORE:`` / ``VIEW:``),
measured findings (``FIX:``) and one yes/no question about the view (``CHECK:``).
"""
from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import cli, coords, view, workflow
from . import draw_core as dc
from . import draw_checks, draw_detail, draw_flat, draw_line, draw_plan, draw_shade, draw_sketch, workflow_draw
from .cli import Ctx, Report, Spec, parse
from .coords import UsageError
from .project import save_png

MIN_VIEW_FRAC = 0.3


# ============================================================================================ report

@dataclass
class DReport(Report):
    view_before: Optional[str] = None

    def text(self) -> str:
        body = [ln for ln in self.lines if ln is not None]
        tail = []
        if self.view_before:
            tail.append(f"VIEW-BEFORE: {self.view_before}   (the same area before this step)")
        if self.view:
            tail.append(f"VIEW: {self.view}" + (f"   ({self.view_note})" if self.view_note else ""))
        if self.view_ref:
            tail.append(f"VIEW-REF: {self.view_ref}   (the reference over the same area)")
        tail.append("NEXT: " + (self.next or "easel next"))
        room = cli.MAX_LINES - len(tail)
        if len(body) > room:
            cut = len(body) - room + 1
            body = body[: room - 1] + [f"... ({cut} more lines; see  easel status)"]
        return "\n".join(body + tail)


USAGE = {
    "plan": "easel plan subject girl --box 250,80,750,1000",
    "palette": 'easel palette make --base "#c83040"',
    "sketch": "easel sketch figure girl --pose stand",
    "line": 'easel line "420,300 380,420 400,560" --weight 1.2',
    "flat": "easel flat auto",
    "shade": "easel shade face --levels 2",
    "shadow": "easel shadow --from hair --onto face",
    "rim": "easel rim hair",
    "crease": "easel crease face",
    "blush": "easel blush face",
    "highlight": "easel highlight hair",
    "hair": 'easel hair hair --flow "450,150 420,300 400,480"',
    "eye": "easel eye eye-l --color iris",
    "fold": 'easel fold "420,600 450,680 470,760" --region coat',
    "pattern": "easel pattern scarf stripes --spacing 20 --angle 30",
    "scatter": "easel scatter grass --kind grass --count 120",
    "paint": "easel paint hills --dir 10 --size 14",
    "band": 'easel band hills "0,600 300,540 700,580 1000,560" --below --color "#5a7a50"',
    "haze": "easel haze --amount 0.5",
    "detail": "easel detail",
}


# ============================================================================================ views

def _zoom(area) -> float:
    x0, y0, x1, y1 = area
    return view.VIEW_SIDE * 0.9 / max(1, max(x1 - x0, y1 - y0))


def tag_image(img: np.ndarray, tags, area, size: int = 15) -> np.ndarray:
    """Draw small labelled tags on a full-size canvas image so they come out ~``size`` px in a view of ``area``."""
    if not tags:
        return img
    z = _zoom(area)
    fz = max(9, int(round(size / max(z, 1e-3))))
    im = Image.fromarray(img.copy())
    dr = ImageDraw.Draw(im)
    f = view.font(fz)
    for text, (x, y), col in tags:
        tw, th = view.text_size(dr, text, f)
        bx, by = x - tw / 2 - 3, y - th / 2 - 4
        dr.rectangle([bx, by, bx + tw + 6, by + th + 8], fill=col, outline=(255, 255, 255))
        dr.text((bx + 3, by), text, fill=(255, 255, 255) if dc.lum(col) < 60 else (0, 0, 0), font=f)
    return np.asarray(im)


def mark_points(img: np.ndarray, pts, area, colour=(255, 0, 200), r_view: float = 5) -> np.ndarray:
    z = _zoom(area)
    r = max(2.0, r_view / max(z, 1e-3))
    im = Image.fromarray(img.copy())
    dr = ImageDraw.Draw(im)
    for x, y in pts:
        dr.ellipse([x - r, y - r, x + r, y + r], outline=colour, width=max(1, int(r / 2.5)))
    return np.asarray(im)


def view_area(p, touched, extra=None, full: bool = False):
    if full or touched is None:
        return (0, 0, p.W, p.H)
    box = coords.union_box(touched, extra) if extra is not None else touched
    x0, y0, x1, y1 = box
    m = MIN_VIEW_FRAC * max(p.W, p.H)
    if x1 - x0 < m:
        c = (x0 + x1) / 2
        x0, x1 = c - m / 2, c + m / 2
    if y1 - y0 < m:
        c = (y0 + y1) / 2
        y0, y1 = c - m / 2, c + m / 2
    b = coords.pad_box_px((max(0, x0), max(0, y0), min(p.W, x1), min(p.H, y1)), p.W, p.H, 0.06)
    if (b[2] - b[0]) * (b[3] - b[1]) > 0.6 * p.W * p.H:
        return (0, 0, p.W, p.H)
    return b


def show_before_after(ctx: Ctx, rep: DReport, before: np.ndarray, after: np.ndarray, area, step: int,
                      note: str, overlay: Optional[Callable[[np.ndarray, tuple], np.ndarray]] = None) -> None:
    p = ctx.project
    if p.state.get("mode") == "hand":       # hand mode: lightbox in sketch / line, VIEW-REF always
        from . import draw_ref
        before, after = draw_ref.hand_views(ctx, rep, before, after, area)
    if overlay is not None:
        after = overlay(after, area)
    bpath = p.root / "views" / f"step-{step:04d}-before.png"
    save_png(bpath, view.render([view.Panel(before, f"BEFORE STEP {step}")], area))
    rep.view_before = p.rel(bpath)
    rep.view = cli._save_view(ctx, view.render([view.Panel(after, f"AFTER STEP {step}")], area), f"step-{step:04d}")
    rep.view_note = f"{note}, area {cli._area_note(p, area)}"


def show_one(ctx: Ctx, rep: Report, img: np.ndarray, area, title: str, note: str, tag: str,
             max_side: int = view.VIEW_SIDE) -> None:
    p = ctx.project
    rep.view = cli._save_view(ctx, view.render([view.Panel(img, title)], area, max_side=max_side), tag)
    rep.view_note = f"{note}, area {cli._area_note(p, area)}"


# ============================================================================================ steps

def _check_marks(ops) -> None:
    from . import manual
    manual.check_marks(manual.engine(), ops, stage_tool=True)


def drop_specks(ops) -> Tuple[list, int]:
    """Leave out marks too small to be marks (fills under 12 px^2, strokes shorter than 4 widths and thinner than a
    3 px dab): on a small canvas an eye's second catch-light simply is not drawn."""
    from .ops import Fill, Stroke, mark_area, stroke_length
    keep, n = [], 0
    for op in ops:
        if isinstance(op, Fill) and mark_area(op) < 12.0:
            n += 1
            continue
        if isinstance(op, Stroke):
            wm = float(np.mean(op.width))
            if stroke_length(op) < 4 * wm and wm < 3.0:
                n += 1
                continue
        keep.append(op)
    return keep, n


def _tail(ctx: Ctx, rep: DReport, stage: Optional[str] = None, question: bool = True) -> None:
    p = ctx.project
    stage = stage or p.state["stage"]
    for f in draw_checks.findings(p, stage):
        rep.add("FIX: " + f)
    if p.state.get("mode") == "hand":       # hand mode: scores against the reference and worst spots
        from . import draw_ref
        draw_ref.hand_tail(p, stage, rep)
    if question:
        q = draw_checks.question(p, stage)
        if q:
            rep.add("CHECK: " + q)
    rep.next = cli._finish_next(ctx)


def draw_step(ctx: Ctx, *, label: str, layer: str, build: Optional[Callable[[], list]] = None,
              clip: Optional[np.ndarray] = None, owned: Optional[tuple] = None, extra_files=(),
              post: Optional[Callable[[], List[str]]] = None, overlay=None, lines: List[str] = (),
              area_extra=None, full_view: bool = False, note: str = "the canvas after this step") -> DReport:
    """One undoable painting step on ``layer``; ``owned`` = (records getter, render_fn, mutate) for the record-owned
    sketch / line layers (mutate changes the records and returns (new ops, touched px box))."""
    from . import brush
    p = ctx.project
    rep = DReport(mutated=True)
    before = dc.composite_u8(p, "view")
    files = [f"layers/{layer}.png"] + list(extra_files)
    if owned is not None:
        files += dc.owned_files(layer)[1:]
    (p.root / "draw").mkdir(exist_ok=True)
    step = p.begin_step(list(dict.fromkeys(files)))
    notes: List[str] = []
    try:
        if owned is not None:
            records, render_fn, mutate = owned
            n = dc.owned_fold(p, layer, records(), render_fn)
            if n:
                notes.append(n)
            ops, touched = mutate()
            _check_marks(ops)
            dc.owned_write(p, layer, records(), render_fn)
            bbox = touched
        else:
            ops, n_small = drop_specks(build() if build else [])
            if n_small:
                notes.append(f"{n_small} marks too small to draw at this canvas size were left out")
            _check_marks(ops)
            arr = p.layer(layer).copy()
            entry = p.layer_entry(layer)
            bbox = brush.apply_ops(arr, ops, clip=None if clip is None else np.asarray(clip, np.float32),
                                   alpha_lock=bool(entry.get("alpha_lock"))) if ops else None
            p._write_layer_file(layer, arr)
        extra_lines = post() if post else []
        from .ops import ops_to_jsonable
        ops_file = p.save_ops(step, {"cmd": ctx.cmdline, "layer": layer, "ops": ops_to_jsonable(ops)})
        ticked = workflow_draw.refresh(p)
        units = coords.px_box_to_units(bbox, p.W, p.H) if bbox else None
        p.end_step(step, {"cmd": ctx.cmdline, "layer": layer, "ops": ops_file, "n_ops": len(ops),
                          "bbox": list(units) if units else None})
    except Exception:
        p.abort_step(step)
        raise
    rep.step = step
    rep.touched = bbox
    rep.add(f"step {step}  {label}  (layer {layer})  {len(ops)} marks  box {coords.fmt_box(units) if units else '-'}")
    if not bbox and not ops:
        rep.add("nothing changed: no marks were made (empty region or the points are outside it)")
    elif not bbox:
        rep.add("nothing changed on the layer: the marks fell outside the region")
    for ln in list(lines) + list(extra_lines or []):
        if ln:
            rep.add(ln)
    for n in notes:
        rep.add("note: " + n)
    if ticked:
        rep.add(f"todo ticked: {', '.join(map(str, ticked))}")
    if not ctx.batch:
        area = view_area(p, bbox, area_extra, full_view)
        show_before_after(ctx, rep, before, dc.composite_u8(p, "view"), area, step, note, overlay)
    _tail(ctx, rep)
    rep.summary = f"{len(ops)} marks"
    return rep


def state_step(ctx: Ctx, label: str, mutate: Callable[[], List[str]], files=()) -> Tuple[DReport, int]:
    p = ctx.project
    step = p.begin_step(list(files))
    try:
        lines = mutate() or []
        if p.state.get("has_ref") is False or p.ref is None or p.state.get("mode") == "hand":
            workflow_draw.build_todo(p.state)
        ticked = workflow_draw.refresh(p)
        p.end_step(step, {"cmd": ctx.cmdline})
    except Exception:
        p.abort_step(step)
        raise
    rep = DReport(mutated=True)
    rep.step = step
    rep.add(f"step {step}  {label}")
    for ln in lines:
        rep.add(ln)
    if ticked:
        rep.add(f"todo ticked: {', '.join(map(str, ticked))}")
    return rep, step


# ============================================================================================ helpers

def _num(a, key, default, lo=None, hi=None, ctx=None):
    if a.get(key) is None:
        return default
    return coords.parse_float(a[key], key, lo, hi, ctx.notes if ctx else None)


def _units_to_px(p, u: float) -> float:
    return float(u) * max(p.W, p.H) / 1000.0


def region_or_part(ctx: Ctx, spec: str, usage: str, allow_box: bool = False):
    """A region (named, box, mark). A planned part that has no coloured area yet resolves to its plan box only with
    ``allow_box`` (patterns, scatter); otherwise the error says how to make the area."""
    p = ctx.project
    d = dc.dstate(p)
    s = str(spec).strip()
    names = p.region_names()
    pm = draw_plan.part_mask(p, s) if s not in names and s in d["parts"] else None
    if pm is not None and pm.any():     # a part made from marks: its own outline, not its box
        ctx.notes.append(f"{s} is not a coloured region yet; used the outline of its marks")
        return cli.Region(s, pm, coords.mask_bbox_px(pm), s, True)
    if s not in names and s in d["parts"] and not coords.looks_like_box(s) and not allow_box:
        holder = _region_holding_part(p, s)
        raise UsageError(f"{s} has no coloured area of its own yet" + (f" (its box is mostly in {holder})" if holder
                                                                     else "") +
                         f": give it one with  easel flat <area numbers> --color <name> --part {s}, or use "
                         f"{holder or 'a region'} instead", "easel flat")
    if s not in names and s in d["parts"] and not coords.looks_like_box(s):
        b = coords.box_to_px(d["parts"][s]["box"], p.W, p.H)
        from .regions import box_mask
        ctx.notes.append(f"{s} is not a coloured region yet (flat makes it one); used its plan box")
        return cli.Region(s, box_mask(p.H, p.W, b), b, s, True)
    return cli.resolve_region(p, s, ctx.notes, usage)


def _region_holding_part(p, part: str) -> Optional[str]:
    d = dc.dstate(p)
    b = coords.box_to_px(d["parts"][part]["box"], p.W, p.H)
    best, val = None, 0.0
    for n in p.region_names():
        m = p.region_mask(n)[b[1]:b[3], b[0]:b[2]]
        v = float(m.mean()) if m.size else 0.0
        if v > val:
            best, val = n, v
    return best if val > 0.3 else None


def part_swatch(p, region_name: str) -> Optional[str]:
    d = dc.dstate(p)
    if region_name in d["parts"]:
        return d["parts"][region_name].get("swatch")
    if region_name in d["bands"]:
        return d["bands"][region_name].get("color")
    if region_name in d["palette"]["swatches"]:
        return region_name
    return None


def colour_set(p, region, given: Optional[str] = None):
    """(base, shadow, light) rgb for a region: its palette swatch, else the median flat colour under it."""
    name = given or part_swatch(p, region.name)
    if name:
        try:
            base = draw_plan.swatch(p, name)
            try:
                sh = draw_plan.swatch(p, f"{name}-shadow") if not str(name).startswith("#") else dc.shadow_of(base)
                li = draw_plan.swatch(p, f"{name}-light") if not str(name).startswith("#") else dc.light_of(base)
            except UsageError:
                sh, li = dc.shadow_of(base), dc.light_of(base)
            return base, sh, li, name
        except UsageError:
            if given:
                raise
    img = dc.composite_u8(p)
    m = region.mask
    base = tuple(int(v) for v in np.median(img[m], axis=0)) if m.any() else (128, 128, 128)
    return base, dc.shadow_of(base), dc.light_of(base), coords.hex_colour(base)


def _mark_list(d: dict, key: str, name: str) -> None:
    lst = d.setdefault(key, [])
    if name not in lst:
        lst.append(name)


def _parse_pts(ctx, text, n=2, example="") -> np.ndarray:
    pts = coords.parse_points(text, ctx.notes, n, example or USAGE["line"])
    return np.asarray(pts, float)


# ============================================================================================ plan

def cmd_plan(ctx: Ctx, args: List[str]) -> Report:
    p = ctx.project
    sub = args[0].lower() if args else "show"
    rest = args[1:]
    if sub in ("show", "list", "view"):
        return _plan_show(ctx)
    if sub in ("subject", "subjects", "object"):
        a = parse(rest, Spec(USAGE["plan"], ["name"], {"box": "value"}, min_pos=1))
        if not a.get("box"):
            raise UsageError("a subject needs --box x0,y0,x1,y1 (0-1000; the whole subject with room around it)",
                             f"easel plan subject {a['name']} --box 250,80,750,1000")
        box = coords.parse_box(a["box"], ctx.notes, USAGE["plan"])
        rep, step = state_step(ctx, f"subject {a['name']}", lambda: [
            f"subject {draw_plan.set_subject(p, a['name'], box)} box {coords.fmt_box(box)}"])
        return _plan_view(ctx, rep, step)
    if sub in ("part", "parts"):
        a = parse(rest, Spec("easel plan part hair --box 300,80,700,450 --focus 2", ["name"],
                             {"box": "value", "of": "value", "focus": "value", "color": "value"},
                             alias={"colour": "color", "subject": "of", "swatch": "color"}, min_pos=1))
        if not a.get("box"):
            raise UsageError("a part needs --box x0,y0,x1,y1", f"easel plan part {a['name']} --box 300,80,700,450")
        box = coords.parse_box(a["box"], ctx.notes, USAGE["plan"])
        focus = coords.parse_int(a["focus"], "focus", 1, 3, ctx.notes) if a.get("focus") else None
        sw = str(a["color"]).strip().lower() if a.get("color") else None
        rep, step = state_step(ctx, f"part {a['name']}", lambda: _part_line(p, a["name"], box, a.get("of"), focus, sw))
        return _plan_view(ctx, rep, step)
    if sub == "light":
        if not rest:
            raise UsageError("say where the light comes from", "easel plan light top-left")
        txt = " ".join(rest)
        m_ = re.search(r"(?:--(?:from|at)\s+|^)(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$", txt.strip())
        src = None
        if m_:
            # a light source in the picture (the sun, a lamp): direction from the picture centre toward it;
            # a source inside the frame lights the subjects from behind
            sx, sy = float(m_.group(1)), float(m_.group(2))
            word = txt[:m_.start()].strip().lower()
            deg = math.degrees(math.atan2(-(sy - 500) * p.H, (sx - 500) * p.W)) % 360
            inside = 0 <= sx <= 1000 and 0 <= sy <= 1000
            z = -0.4 if (inside or word in ("back", "behind")) and word not in ("front",) else 0.6
            words = f"a source at {sx:.0f},{sy:.0f}" + (" (behind the subjects: rim light)" if z < 0 else "")
            src = [sx, sy]
        else:
            deg, z, words = dc.parse_light(txt)

        def mut():
            d = dc.dstate(p)
            d["light"] = {"deg": deg, "z": z, "set": True}
            if src:
                d["light"]["at"] = src
            return [f"light {words}"]
        rep, step = state_step(ctx, "light", mut)
        return _plan_view(ctx, rep, step)
    if sub in ("order", "depth"):
        d = dc.dstate(p)
        names = [x for x in re.split(r"[,\s]+", " ".join(rest).lower()) if x]
        if not names:
            raise UsageError("say the subjects from far to near", "easel plan order sky,sun,mountains,lake,pines")
        bad = [x for x in names if x not in d["subjects"] and x not in d["parts"]]
        if bad:
            raise UsageError(f"not planned: {', '.join(bad)}; subjects: {', '.join(d['subjects'])}",
                             "easel plan order " + ",".join(d["subjects"]))

        def mut():
            d["order"] = names
            return [f"depth order, far to near: {', '.join(draw_checks.depth_order(p))} (a farther thing drawn later "
                    f"goes behind the nearer ones)"]
        rep, step = state_step(ctx, "plan order", mut)
        return _plan_view(ctx, rep, step)
    if sub in ("remove", "rm", "delete"):
        if not rest:
            raise UsageError("name the subject or part to remove", "easel plan remove hat")
        name = rest[0].lower()

        def mut():
            d = dc.dstate(p)
            if name in d["parts"]:
                del d["parts"][name]
                return [f"part {name} removed"]
            if name in d["subjects"]:
                del d["subjects"][name]
                gone = [k for k, v in d["parts"].items() if v["of"] == name]
                for k in gone:
                    del d["parts"][k]
                return [f"subject {name} removed" + (f" with its parts {', '.join(gone)}" if gone else "")]
            raise UsageError(f"no subject or part named {name}", "easel plan")
        rep, step = state_step(ctx, f"remove {name}", mut)
        return _plan_view(ctx, rep, step)
    raise UsageError(f"unknown plan action {sub!r}; use subject, part, light, order, remove, show", USAGE["plan"])


def _part_line(p, name, box, of, focus, sw) -> List[str]:
    n = draw_plan.set_part(p, name, box, of, focus, sw)
    prt = dc.dstate(p)["parts"][n]
    return [f"part {n} of {prt['of']}  box {coords.fmt_box(box)}  focus {prt['focus']}  colour name {prt['swatch']}"]


def _plan_view(ctx: Ctx, rep: DReport, step: int) -> DReport:
    p = ctx.project
    rep.lines.extend(draw_plan.plan_lines(p)[:4])
    if ctx.batch:
        rep.next = "easel plan   (look at the boxes, the light arrow and the palette before going on)"
        return rep
    if not ctx.batch:
        img = draw_plan.plan_overlay(p, p.composite_u8("view"))
        rep.view = cli._save_view(ctx, view.render([view.Panel(img, "PLAN: subjects, parts, light, palette")],
                                                   (0, 0, p.W, p.H), max_side=view.MAP_SIDE), f"step-{step:04d}")
        rep.view_note = "subject boxes (black), part boxes 'name f<focus>', light arrow, palette strip"
    _tail(ctx, rep, "plan")
    return rep


def _plan_show(ctx: Ctx) -> Report:
    p = ctx.project
    rep = DReport()
    rep.lines.extend(draw_plan.plan_lines(p))
    if not ctx.batch:
        img = draw_plan.plan_overlay(p, p.composite_u8("view"))
        rep.view = cli._save_view(ctx, view.render([view.Panel(img, "PLAN")], (0, 0, p.W, p.H),
                                                   max_side=view.MAP_SIDE), f"plan-{cli._view_counter(p):04d}")
        rep.view_note = "subject boxes, part boxes, light arrow, palette strip"
    rep.next = cli._finish_next(ctx)
    return rep


# ============================================================================================ palette

def cmd_palette(ctx: Ctx, args: List[str]) -> Report:
    p = ctx.project
    d = dc.dstate(p)
    sub = args[0].lower() if args else "show"
    rest = args[1:]
    if sub in ("show", "list"):
        rep = DReport()
        full = draw_plan.full_palette(d["palette"]["swatches"])
        if not full:
            rep.add("no palette yet")
        for n, hx in d["palette"]["swatches"].items():
            rgb = dc.from_hex(hx)
            rep.add(f"  {n:<12} {hx}  ({coords.colour_name(rgb)})   shadow {full.get(n + '-shadow', '-')}   light "
                    f"{full.get(n + '-light', '-')}")
        if not ctx.batch and d["palette"]["swatches"]:
            img = draw_plan.palette_image([], d["palette"]["swatches"])
            rep.view = cli._save_view(ctx, img, f"palette-{cli._view_counter(p):04d}")
            rep.view_note = "each swatch: base colour, shadow (left) and light (right) under it"
        rep.next = cli._finish_next(ctx)
        return rep
    if sub in ("make", "new", "propose", "suggest"):
        a = parse(rest, Spec(USAGE["palette"], [], {"base": "value", "names": "value"},
                             alias={"color": "base", "colour": "base"}))
        names = [x for x in re.split(r"[,\s]+", str(a.get("names") or "")) if x] or draw_plan.needed_swatches(d)
        for x in ("background", "line"):
            if x not in names:
                names.append(x)
        req = p.state.get("request", "")
        fixed = draw_plan.request_colours(req, names)
        if a.get("base"):
            base = draw_plan.swatch(p, a["base"])
        elif fixed:
            base = max(fixed.values(), key=lambda c: dc.rgb_to_lch(c)[1])
        else:
            base = (200, 80, 90)
        cands = draw_plan.make_candidates(d, base, req, names)

        def mut():
            d["palette"]["candidates"] = cands
            return [f"{len(cands)} palettes around {coords.hex_colour(base)}"
                    + (f"; from the request: {', '.join(f'{k} {coords.colour_name(v)}' for k, v in fixed.items())}"
                       if fixed else "")]
        rep, step = state_step(ctx, "palette make", mut)
        for i, c in enumerate(cands, 1):
            rep.add(f"  {i} {c['scheme']}: " + ", ".join(f"{n} {h}" for n, h in list(c["swatches"].items())[:7]))
        if not ctx.batch:
            img = draw_plan.palette_image(cands)
            rep.view = cli._save_view(ctx, img, f"step-{step:04d}")
            rep.view_note = "numbered palettes; under each swatch its shadow (left) and light (right)"
        rep.next = "easel palette pick <1-4>   (the one that fits the request; change one colour later with " \
                   "easel palette set <name> <colour>)"
        return rep
    if sub in ("pick", "choose", "use"):
        if not rest:
            raise UsageError("palette pick needs the palette number", "easel palette pick 2")
        k = coords.parse_int(rest[0], "n", 1, 99, ctx.notes)
        cands = d["palette"]["candidates"]
        if not cands:
            raise UsageError("no palettes proposed yet", USAGE["palette"])
        if not 1 <= k <= len(cands):
            raise UsageError(f"there are palettes 1-{len(cands)}", "easel palette pick 1")

        def mut():
            d["palette"]["swatches"] = dict(cands[k - 1]["swatches"])
            d["palette"]["picked"] = k
            return [f"palette {k} ({cands[k - 1]['scheme']}): " + ", ".join(
                f"{n} {h}" for n, h in d["palette"]["swatches"].items())]
        rep, step = state_step(ctx, f"palette pick {k}", mut)
        return _plan_view(ctx, rep, step)
    if sub in ("set", "add", "name"):
        pairs = _pairs(rest)
        if not pairs:
            raise UsageError("palette set needs a name and a colour", 'easel palette set hair "#d8dce8" scarf red')
        vals = [(n, draw_plan.swatch(p, c) if c.lower() != n else coords.parse_colour(c)) for n, c in pairs]

        def mut():
            out = []
            for n, rgb in vals:
                d["palette"]["swatches"][n] = coords.hex_colour(rgb)
                out.append(f"{n} = {coords.hex_colour(rgb)} ({coords.colour_name(rgb)})")
            return ["; ".join(out), "flats already painted keep their colour: easel flat refresh repaints them"
                    if d.get("fills") else ""]
        rep, step = state_step(ctx, "palette set", mut)
        rep.lines = [ln for ln in rep.lines if ln]
        return _plan_view(ctx, rep, step)
    raise UsageError(f"unknown palette action {sub!r}; use make, pick, set, show", USAGE["palette"])


def _pairs(tokens: List[str]) -> List[Tuple[str, str]]:
    toks = []
    for t in tokens:
        toks.extend([x for x in re.split(r"\s*[=:]\s*", t) if x] if ("=" in t or ":" in t) and " " not in t else [t])
    out = []
    i = 0
    while i < len(toks):
        name = re.sub(r"[^a-z0-9\-]+", "-", toks[i].strip().lower()).strip("-")
        if i + 1 >= len(toks):
            raise UsageError(f"no colour after {name}", f'easel palette set {name} "#c04030"')
        col = toks[i + 1]
        try:
            coords.parse_colour(col)
            i += 2
        except UsageError:
            if i + 2 < len(toks):
                col = toks[i + 1] + " " + toks[i + 2]
                coords.parse_colour(col, f'easel palette set {name} "#c04030"')
                i += 3
            else:
                raise
        out.append((name, col))
    return out


# ============================================================================================ sketch

SKETCH_SUBS = ("ellipse", "circle", "oval", "box", "rect", "line", "curve", "figure", "mannequin", "head", "horizon",
               "joint", "joints", "move", "scale", "rotate", "delete", "remove", "rm", "list", "show")


def _sketch_owned(p, mutate):
    return (lambda: dc.dstate(p)["sketch"], draw_sketch.render_ops, mutate)


def _sketch_overlay(p, ids=None):
    def ov(img, area):
        d = dc.dstate(p)
        tags = []
        for el in d["sketch"]:
            if ids is not None and el["id"] not in ids:
                continue
            for txt, xy in draw_sketch.label_points(p, el):
                tags.append((txt, xy, (40, 60, 150) if txt.startswith("s") and txt[1:].isdigit() else (200, 70, 0)))
        return tag_image(img, tags, area, 13)
    return ov


def _el_box_px(p, el):
    pts = []
    for poly, _, _ in draw_sketch.shapes(p, el):
        pts.append(poly)
    if not pts:
        return None
    a = np.vstack(pts)
    return (int(max(0, a[:, 0].min() - 4)), int(max(0, a[:, 1].min() - 4)), int(min(p.W, a[:, 0].max() + 5)),
            int(min(p.H, a[:, 1].max() + 5)))


def cmd_sketch(ctx: Ctx, args: List[str]) -> Report:
    p = ctx.project
    d = dc.dstate(p)
    if not args:
        raise UsageError("say what to sketch: ellipse, box, line, curve, figure, head, horizon (or move / scale / "
                         "rotate / delete / list / show an element)", USAGE["sketch"])
    sub = args[0].lower()
    rest = args[1:]
    if sub not in SKETCH_SUBS:
        if sub in d["subjects"]:
            sub, rest = "figure", args
        else:
            raise UsageError(f"unknown sketch kind {args[0]!r}; one of {', '.join(SKETCH_SUBS[:11])}", USAGE["sketch"])
    if sub in ("list", "show"):
        rep = DReport()
        for el in d["sketch"]:
            rep.add("  " + draw_sketch.describe(el))
            if el["kind"] == "figure":
                rep.add("    " + "; ".join(draw_sketch.figure_report(p, el)))
        if not d["sketch"]:
            rep.add("no sketch elements yet")
        if not ctx.batch:
            img = _sketch_overlay(p)(p.composite_u8("view"), (0, 0, p.W, p.H))
            show_one(ctx, rep, img, (0, 0, p.W, p.H), "SKETCH (ids, joints)", "sketch elements with their ids",
                     f"sketch-{cli._view_counter(p):04d}")
        rep.next = cli._finish_next(ctx)
        return rep
    lines: List[str] = []
    focus_ids: List[str] = []

    if sub in ("ellipse", "circle", "oval", "box", "rect"):
        a = parse(rest, Spec("easel sketch ellipse 300,200,500,420", ["box*"], {"rot": "value"},
                             alias={"rotate": "rot", "angle": "rot"}, min_pos=1))
        box = coords.parse_box(a["box"], ctx.notes, "easel sketch ellipse 300,200,500,420")
        kind = "ellipse" if sub in ("ellipse", "circle", "oval") else "box"

        def mutate():
            el = draw_sketch.new_element(p, kind, box=[round(v, 1) for v in box],
                                         rot=float(_num(a, "rot", 0.0, -180, 180, ctx)))
            focus_ids.append(el["id"])
            return draw_sketch.render_ops(p, el), _el_box_px(p, el)
        label = f"sketch {kind} {coords.fmt_box(box)}"
    elif sub in ("line", "curve"):
        a = parse(rest, Spec('easel sketch curve "300,200 360,260 380,340"', ["pts*"], {"closed": "flag"},
                             alias={"close": "closed"}, min_pos=1))
        pts = coords.parse_points(a["pts"], ctx.notes, 2, 'easel sketch curve "300,200 360,260 380,340"')

        def mutate():
            el = draw_sketch.new_element(p, sub, pts=[[round(x, 1), round(y, 1)] for x, y in pts],
                                         closed=bool(a.get("closed")))
            focus_ids.append(el["id"])
            return draw_sketch.render_ops(p, el), _el_box_px(p, el)
        label = f"sketch {sub} {len(pts)} points"
    elif sub in ("figure", "mannequin"):
        a = parse(rest, Spec("easel sketch figure girl --pose stand --heads 7", ["subject"],
                             {"pose": "value", "heads": "value", "box": "value"}, min_pos=1))
        subj = a["subject"].lower()
        if a.get("box"):
            box = coords.parse_box(a["box"], ctx.notes, USAGE["sketch"])
        elif subj in d["subjects"]:
            box = d["subjects"][subj]["box"]
        else:
            raise UsageError(f"no subject {subj!r} in the plan; plan it or give --box",
                             f"easel sketch figure {subj} --box 300,100,700,950")
        req = (p.state.get("request") or "").lower()
        pose = str(a.get("pose") or ("bust" if any(w in req for w in ("bust", "portrait", "headshot")) else
                                     "stand")).lower()
        if pose not in draw_sketch.POSES:
            raise UsageError(f"unknown pose {pose!r}; poses: {', '.join(draw_sketch.POSES)} (then move joints)",
                             f"easel sketch figure {subj} --pose stand")
        heads = _num(a, "heads", 2.5 if "chibi" in req else 7.0, 2.0, 9.0, ctx)

        def mutate():
            old = [el for el in d["sketch"] if el["kind"] == "figure" and el.get("subject") == subj]
            for el in old:
                d["sketch"].remove(el)
            el = draw_sketch.new_element(p, "figure", subject=subj, pose=pose, heads=heads,
                                         joints=draw_sketch.figure_joints(box, heads, pose, p.W, p.H))
            focus_ids.append(el["id"])
            lines.extend(draw_sketch.figure_report(p, el))
            lines.append("joints (l = picture left): " + " ".join(
                f"{draw_sketch.SHORT[j]} {coords.fmt_point(v)}" for j, v in el["joints"].items()
                if pose != "bust" or j in ("head", "chin", "lshoulder", "rshoulder")))
            if old:
                lines.append(f"replaced {', '.join(o['id'] for o in old)}")
            return draw_sketch.render_ops(p, el), _el_box_px(p, el)
        label = f"sketch figure {subj} ({pose}, {heads:g} heads)"
    elif sub == "head":
        a = parse(rest, Spec("easel sketch head girl --yaw 20", ["subject"],
                             {"box": "value", "yaw": "value", "pitch": "value", "turn": "value", "tilt": "value",
                              "shape": "value"},
                             alias={}))
        subj = (a.get("subject") or "").lower()
        yaw = _num(a, "yaw", _num(a, "turn", 0.0, -80, 80, ctx), -80, 80, ctx)
        pitch = _num(a, "pitch", _num(a, "tilt", 0.0, -60, 60, ctx), -60, 60, ctx)
        if a.get("box"):
            box = list(coords.parse_box(a["box"], ctx.notes, USAGE["sketch"]))
        else:
            box = _head_box(p, subj)
        if box is None:
            raise UsageError("give the head's box (top of the skull to the chin), or sketch a figure / plan a part "
                             "head or face first", "easel sketch head girl --box 400,120,600,360 --yaw 15")

        def mutate():
            old = [el for el in d["sketch"] if el["kind"] == "head" and subj and el.get("subject") == subj]
            for el in old:
                d["sketch"].remove(el)
            fig = next((e for e in d["sketch"] if e["kind"] == "figure" and (not subj or e.get("subject") == subj)), None)
            chibi = fig is not None and float(fig.get("heads", 7)) < 4.2
            el = draw_sketch.new_element(p, "head", subject=subj or None, box=[round(v, 1) for v in box],
                                         yaw=float(yaw), pitch=float(pitch), chibi=chibi,
                                         shape=_head_shape(a.get("shape")))
            focus_ids.append(el["id"])
            g = draw_sketch.head_guides(p, el)
            lines.append("face guides: " + "; ".join(f"{k} {coords.fmt_point(v)}" for k, v in g.items()))
            lines.extend(_fit_eye_parts(p, el, g))
            hw, hh = el["box"][2] - el["box"][0], el["box"][3] - el["box"][1]
            ew, eh = (0.3 * hw, 0.24 * hh) if chibi else (0.24 * hw, 0.17 * hh)
            lines.append(f"eyes on the eye line, centred on those points, about {ew:.0f} x {eh:.0f} "
                         f"units each (anime; the far eye narrower); nose and mouth on the centre line; "
                         f"easel line eye <eye part> draws them")
            return draw_sketch.render_ops(p, el), _el_box_px(p, el)
        label = f"sketch head yaw {yaw:g} pitch {pitch:g}"
    elif sub == "horizon":
        a = parse(rest, Spec("easel sketch horizon 450 --vp 500,450", ["y"], {"vp": "value", "vp2": "value"}, min_pos=1))
        y = coords.parse_float(a["y"], "y", 0, 1000, ctx.notes)
        vps = [list(coords.parse_point(a[k], ctx.notes)) for k in ("vp", "vp2") if a.get(k)]

        def mutate():
            old = [el for el in d["sketch"] if el["kind"] == "horizon"]
            for el in old:
                d["sketch"].remove(el)
            el = draw_sketch.new_element(p, "horizon", y=round(y, 1), vps=vps)
            focus_ids.append(el["id"])
            return draw_sketch.render_ops(p, el), (0, 0, p.W, p.H)
        label = f"sketch horizon {y:g}"
    elif sub in ("joint", "joints"):
        if len(rest) < 3:
            raise UsageError("sketch joint needs the figure (id or subject), a joint name and x,y",
                             "easel sketch joint s1 rwr 620,540")
        el = draw_sketch.find(p, rest[0])
        if el["kind"] != "figure":
            raise UsageError(f"{el['id']} is not a figure", "easel sketch list")
        toks = rest[1:]
        moves = []
        i = 0
        while i < len(toks):
            j = draw_sketch.norm_joint(toks[i])
            if i + 1 >= len(toks):
                raise UsageError(f"no x,y after {toks[i]}", f"easel sketch joint {el['id']} {toks[i]} 500,500")
            moves.append((j, coords.parse_point(toks[i + 1], ctx.notes)))
            i += 2

        def mutate():
            for j, pt in moves:
                el["joints"][j] = [round(pt[0], 1), round(pt[1], 1)]
            el["n"] = dc.next_order(d)
            lines.extend(draw_sketch.figure_report(p, el))
            focus_ids.append(el["id"])
            return draw_sketch.render_ops(p, el), _el_box_px(p, el)
        label = f"sketch joint {el['id']} " + " ".join(f"{draw_sketch.SHORT[j]} {coords.fmt_point(v)}" for j, v in moves)
    elif sub in ("move", "scale", "rotate", "delete", "remove", "rm"):
        if not rest:
            raise UsageError(f"sketch {sub} needs an element id", f"easel sketch {sub} s1" +
                             (" --by 20,0" if sub == "move" else " 1.2" if sub == "scale" else
                              " 15" if sub == "rotate" else ""))
        el = draw_sketch.find(p, rest[0])
        if sub == "move":
            a = parse(rest[1:], Spec("easel sketch move s1 --by 20,-10", [], {"by": "value", "to": "value"}))
            if a.get("by"):
                nums = [v for v, _, _ in coords._numbers(a["by"])]
                if len(nums) != 2:
                    raise UsageError("--by needs dx,dy in units", "easel sketch move s1 --by 20,-10")
                dx, dy = nums
            elif a.get("to"):
                tx, ty = coords.parse_point(a["to"], ctx.notes)
                cx, cy = draw_sketch.centre(el)
                dx, dy = tx - cx, ty - cy
            else:
                raise UsageError("say --by dx,dy or --to x,y", "easel sketch move s1 --by 20,-10")

            def mutate():
                draw_sketch.move(el, dx, dy)
                el["n"] = dc.next_order(d)
                focus_ids.append(el["id"])
                return draw_sketch.render_ops(p, el), _el_box_px(p, el)
            label = f"sketch move {el['id']} by {dx:g},{dy:g}"
        elif sub == "scale":
            f = coords.parse_float(rest[1] if len(rest) > 1 else "", "factor", 0.1, 10, ctx.notes,
                                   f"easel sketch scale {el['id']} 1.2")

            def mutate():
                draw_sketch.scale(el, f)
                el["n"] = dc.next_order(d)
                focus_ids.append(el["id"])
                return draw_sketch.render_ops(p, el), _el_box_px(p, el)
            label = f"sketch scale {el['id']} x{f:g}"
        elif sub == "rotate":
            deg = coords.parse_float(rest[1] if len(rest) > 1 else "", "degrees", -360, 360, ctx.notes,
                                     f"easel sketch rotate {el['id']} 15")

            def mutate():
                draw_sketch.rotate(el, deg)
                el["n"] = dc.next_order(d)
                focus_ids.append(el["id"])
                return draw_sketch.render_ops(p, el), _el_box_px(p, el)
            label = f"sketch rotate {el['id']} {deg:g}"
        else:
            def mutate():
                bb = _el_box_px(p, el)
                d["sketch"].remove(el)
                return [], bb
            label = f"sketch delete {el['id']}"
    else:
        raise UsageError(f"unknown sketch kind {sub!r}", USAGE["sketch"])
    return draw_step(ctx, label=label, layer="sketch", owned=_sketch_owned(p, mutate), lines=lines,
                     overlay=_sketch_overlay(p), full_view=True, note="the sketch (ids in blue, joints in orange)")


def _fit_eye_parts(p, el, g) -> List[str]:
    """Planned eye parts move onto the construction's eye guides (same size), so later stages find the eyes."""
    d = dc.dstate(p)
    eyes = sorted([n for n in d["parts"] if n.startswith("eye") and (not el.get("subject") or
                                                                   d["parts"][n]["of"] == el.get("subject"))],
                  key=lambda n: (d["parts"][n]["box"][0] + d["parts"][n]["box"][2]))
    if len(eyes) != 2:
        return []
    out = []
    w = (el["box"][2] - el["box"][0])
    for n, key in zip(eyes, ("left eye", "right eye")):
        b = d["parts"][n]["box"]
        sc = (0.3, 0.24) if el.get("chibi") else (0.24, 0.17)
        bw, bh = sc[0] * w, sc[1] * (el["box"][3] - el["box"][1])
        cx, cy = g[key]
        nb = [round(cx - bw / 2, 1), round(cy - bh / 2, 1), round(cx + bw / 2, 1), round(cy + bh / 2, 1)]
        if max(abs(nb[i] - b[i]) for i in range(4)) > 8:
            d["parts"][n]["box"] = nb
            out.append(f"plan part {n} moved onto the eye guide: box {coords.fmt_box(nb)}")
    return out


def _head_box(p, subj: str):
    d = dc.dstate(p)
    for el in d["sketch"]:
        if el["kind"] == "figure" and (not subj or el.get("subject") == subj):
            hd, ch = np.array(el["joints"]["head"]), np.array(el["joints"]["chin"])
            hpx, cpx = dc.u2px(p, [hd])[0], dc.u2px(p, [ch])[0]
            r = float(np.hypot(*(cpx - hpx)))
            top = hpx[1] - r
            x0, x1 = hpx[0] - 0.8 * r, hpx[0] + 0.8 * r
            return [round(x0 * 1000 / p.W, 1), round(top * 1000 / p.H, 1), round(x1 * 1000 / p.W, 1),
                    round(cpx[1] * 1000 / p.H, 1)]
    for n in ("head", "face"):
        if n in d["parts"] and (not subj or d["parts"][n]["of"] == subj):
            return list(d["parts"][n]["box"])
    return None


# ============================================================================================ line

LINE_SUBS = ("close", "move", "redo", "delete", "remove", "rm", "weight", "list", "show", "gaps", "trace", "eye", "tube")


def _line_owned(p, mutate):
    return (lambda: dc.dstate(p)["lines"], draw_line.render_ops, mutate)


def _line_overlay(p, ids=None, gaps: bool = True):
    def ov(img, area):
        d = dc.dstate(p)
        tags = []
        for r in d["lines"]:
            if ids is not None and r["id"] not in ids:
                continue
            path = draw_line.record_path(p, r)
            if len(path):
                q = path[len(path) // 2]
                tags.append((r["id"], (float(q[0]), float(q[1])), (30, 90, 200)))
        out = img
        if gaps and d["lines"]:
            an = draw_line.analyze(p)
            pts = [tuple(g["a"]["pt"]) for g in an["gaps"][:8]]
            out = mark_points(out, pts, area)
            for g in an["gaps"][:8]:
                tags.append((g["id"], (float(g["a"]["pt"][0]), float(g["a"]["pt"][1]) - 0.02 * p.H), (255, 0, 200)))
        return tag_image(out, tags, area, 12)
    return ov


def _line_post(p):
    def post():
        an = draw_line.analyze(p)
        return draw_line.gap_lines(p, an, 2)
    return post


def _shade_layer(p) -> str:
    """Painted pictures keep their brushwork on the paint layer, which lies over the shade layer: their light and
    shadow go on the detail layer above it."""
    return "detail" if (p.state.get("style") or "paint") in ("paint", "photo") else "shade"


def _inner_parts(p, r) -> np.ndarray:
    """The regions that sit inside region ``r`` (eyes in a face, a muzzle in a head): their box lies within r's box
    and most of their edge borders r."""
    out = np.zeros((p.H, p.W), bool)
    rb = coords.mask_bbox_px(r.mask)
    if rb is None:
        return out
    ring = cv2.dilate(r.mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    for n in p.region_names():
        if n == r.name or n in ("background", "bg", "sky"):
            continue
        m = p.region_mask(n)
        b = coords.mask_bbox_px(m)
        if b is None or b[0] < rb[0] or b[1] < rb[1] or b[2] > rb[2] or b[3] > rb[3]:
            continue
        edge = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~m
        if edge.any() and float(ring[edge].mean()) > 0.6:
            out |= m
    return out


def _head_shape(v) -> str:
    v = str(v or "oval").strip().lower()
    if v not in ("round", "oval", "sharp"):
        raise UsageError(f"unknown head shape {v!r}: round, oval or sharp", "easel sketch head girl --shape round")
    return v


def _close_points(p, g, src) -> Tuple[np.ndarray, np.ndarray]:
    """Where a gap bridge starts and ends. If the open end overshoots or curls past the other line (the bridge would
    turn back on itself and leave a hook), the end is trimmed back to where it comes closest, and a short straight
    bridge joins it there."""
    a = np.asarray(g["a"]["pt"], float)
    b = np.asarray(g["b_pt"], float)
    tid = g["b"].split()[0] if g.get("b") else None
    if g["a"].get("near") and (tid is None or g["dist"] > g["a"]["dist"] + 1):
        tid = g["a"]["near"][0]
    try:
        tpath = draw_line.record_path(p, draw_line.find(p, tid)) if tid else np.array([b])
        P = draw_line.record_path(p, src)
    except Exception:  # noqa: BLE001
        return a, b
    if len(P) < 4:
        return a, b
    rev = g["a"]["which"] == "start"
    Q = P[::-1] if rev else P
    S = max(p.W, p.H)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(Q, axis=0).T))])
    near_end = s >= s[-1] - 0.08 * S
    idx = np.nonzero(near_end)[0]
    dd = np.array([np.min(np.hypot(*(tpath - Q[i]).T)) for i in idx])
    k = int(idx[int(np.argmin(dd))])
    tangent = Q[-1] - Q[max(0, len(Q) - 4)]
    bridge = b - Q[-1]
    turns_back = float(tangent @ bridge) < 0
    if k < len(Q) - 2 and (turns_back or dd.min() < 0.6 * np.hypot(*bridge)):
        Qn = Q[:k + 1]
        j = int(np.argmin(np.hypot(*(tpath - Qn[-1]).T)))
        newp = Qn[::-1] if rev else Qn
        src["path"] = [list(map(float, q)) for q in dc.px2u(p, draw_line.rdp(newp, 0.25))]
        return Qn[-1], tpath[j]
    return a, b


def cmd_line(ctx: Ctx, args: List[str]) -> Report:
    p = ctx.project
    d = dc.dstate(p)
    if not args:
        raise UsageError("give the line's points, 0-1000: 3-8 points along the edge", USAGE["line"])
    sub = args[0].lower()
    rest = args[1:]
    if sub in ("list", "show", "gaps"):
        rep = DReport()
        if sub == "list" or sub == "show":
            for r in d["lines"][-14:]:
                rep.add("  " + draw_line.describe(p, r))
            if len(d["lines"]) > 14:
                rep.add(f"  ... {len(d['lines']) - 14} earlier lines")
        an = draw_line.analyze(p) if d["lines"] else {"gaps": [], "stray": []}
        rep.lines.extend(draw_line.gap_lines(p, an, 6) if d["lines"] else ["no lines yet"])
        if not an["gaps"] and d["lines"]:
            rep.add("no open gaps")
        if not ctx.batch:
            img = _line_overlay(p)(p.composite_u8("view"), (0, 0, p.W, p.H))
            show_one(ctx, rep, img, (0, 0, p.W, p.H), "LINES (ids blue, gaps magenta)", "line ids and open gaps",
                     f"lines-{cli._view_counter(p):04d}")
        if sub == "gaps":
            workflow_draw.refresh(p)
            p.save_state()
        rep.next = (f"easel line close {an['gaps'][0]['id']}" if an["gaps"] else cli._finish_next(ctx))
        return rep
    lines: List[str] = []
    new_ids: List[str] = []
    color_default = "line"
    if sub == "close":
        if not rest:
            raise UsageError("line close needs a gap id from  easel line gaps", "easel line close G1")
        an = draw_line.analyze(p)
        ids = [x.upper() for x in re.split(r"[,\s]+", " ".join(rest)) if x]
        if ids == ["ALL"]:
            ids = [g["id"] for g in an["gaps"] if g["dist"] < 0.025 * max(p.W, p.H)]
        gaps = [g for g in an["gaps"] if g["id"] in ids]
        if not gaps and " ".join(rest).strip().lower() == "all":
            rep = DReport()
            rep.add("no small gaps to close")
            return rep
        if not gaps:
            raise UsageError(f"no gap {', '.join(ids)}; gaps now: {', '.join(g['id'] for g in an['gaps']) or 'none'} "
                             f"(gap ids are numbered again after every close: close several in one command)",
                             "easel line close G1,G3,G4")

        def mutate():
            ops, bb = [], None
            for g in gaps:
                a = g["a"]["pt"]
                b = g["b_pt"]
                src = draw_line.find(p, g["a"]["line"])
                a, b = _close_points(p, g, src)
                path = dc.resample(np.array([a, b], float), 1.5)
                rec = draw_line.make_record(p, path, dc.px2u(p, [a, b]), weight=src.get("weight", 1.0),
                                            taper="none", color=src.get("color", "line"), closed=False,
                                            src=f"close {g['id']}")
                d["lines"].append(rec)
                new_ids.append(rec["id"])
                ops += draw_line.render_ops(p, rec)
                bb = coords.union_box(bb, _path_box(p, path))
            return ops, bb
        label = f"line close {', '.join(g['id'] for g in gaps)}"
    elif sub in ("move", "redo", "delete", "remove", "rm", "weight"):
        if not rest:
            raise UsageError(f"line {sub} needs a line id", f"easel line {sub} L3")
        targets = [draw_line.find(p, x) for x in re.split(r"[,\s]+", rest[0]) if x]
        rec = targets[0]
        if sub == "move":
            a = parse(rest[1:], Spec("easel line move L3 --by 10,0", [], {"by": "value"}))
            nums = [v for v, _, _ in coords._numbers(a.get("by") or "")]
            if len(nums) != 2:
                raise UsageError("line move needs --by dx,dy (units)", f"easel line move {rec['id']} --by 10,0")
            dx, dy = nums

            def mutate():
                bb = None
                for r in targets:
                    bb = coords.union_box(bb, _path_box(p, draw_line.record_path(p, r)))
                    r["ctrl"] = [[x + dx, y + dy] for x, y in r["ctrl"]]
                    r["path"] = [[round(x + dx, 2), round(y + dy, 2)] for x, y in r["path"]]
                    r["n"] = dc.next_order(d)
                    r["part"] = draw_line.guess_part(p, r)
                    bb = coords.union_box(bb, _path_box(p, draw_line.record_path(p, r)))
                return sum((draw_line.render_ops(p, r) for r in targets), []), bb
            label = f"line move {','.join(r['id'] for r in targets)} by {dx:g},{dy:g}"
        elif sub == "weight":
            w = coords.parse_float(rest[1] if len(rest) > 1 else "", "weight", 0.3, 6, ctx.notes,
                                   f"easel line weight {rec['id']} 1.5")

            def mutate():
                for r in targets:
                    r["weight"] = round(w, 2)
                    r["n"] = dc.next_order(d)
                return (sum((draw_line.render_ops(p, r) for r in targets), []),
                        _path_box(p, np.vstack([draw_line.record_path(p, r) for r in targets])))
            label = f"line weight {','.join(r['id'] for r in targets)} {w:g}"
        elif sub == "redo":
            a = parse(rest[1:], Spec(f'easel line redo {rec["id"]} "x,y x,y x,y"', ["pts*"],
                                     {"closed": "flag", "no-snap": "flag", "no-join": "flag", "weight": "value",
                                      "taper": "value", "bow": "value"}, min_pos=1))
            pts = _parse_pts(ctx, a["pts"], 2, f'easel line redo {rec["id"]} "x,y x,y x,y"')
            wt = _num(a, "weight", rec.get("weight", 1.0), 0.3, 6, ctx) if a.get("weight") else rec.get("weight", 1.0)
            tp = a.get("taper") or rec.get("taper", "both")
            bw = _num(a, "bow", None, -0.5, 0.5, ctx) if a.get("bow") else None

            def mutate():
                bb = _path_box(p, draw_line.record_path(p, rec))
                path, nts = draw_line.fit(p, pts, closed=bool(a.get("closed")), snap=not a.get("no-snap"),
                                          join=not a.get("no-join"), exclude=rec["id"], bow=bw)
                lines.extend(nts)
                nr = draw_line.make_record(p, path, pts, weight=wt, taper=tp,
                                           color=rec.get("color", "line"), closed=bool(a.get("closed")))
                nr["id"] = rec["id"]
                d["seq"]["L"] -= 1
                d["lines"][d["lines"].index(rec)] = nr
                new_ids.append(nr["id"])
                return draw_line.render_ops(p, nr), coords.union_box(bb, _path_box(p, path))
            label = f"line redo {rec['id']}"
        else:
            def mutate():
                bb = None
                for r in targets:
                    bb = coords.union_box(bb, _path_box(p, draw_line.record_path(p, r)))
                    d["lines"].remove(r)
                return [], bb
            label = f"line delete {','.join(r['id'] for r in targets)}"
    elif sub == "tube":
        a = parse(rest, Spec('easel line tube "300,700 200,750 160,860" --width "40,70,20"', ["pts*"],
                             {"width": "value", "weight": "value", "along": "value"}, min_pos=0))
        src = a.get("along") or a.get("pts")
        if not src:
            raise UsageError("say the axis points (or --along s3)", 'easel line tube "300,700 200,750 160,860" '
                             '--width "40,70,20"')
        if re.fullmatch(r"\s*s\d+\s*", str(src)):
            el = draw_sketch.find(p, str(src).strip())
            axis = max((poly for poly, closed, wf in draw_sketch.shapes(p, el)), key=lambda q: len(q))
        else:
            axis = dc.catmull_rom(dc.u2px(p, coords.parse_points(src, ctx.notes, 2)), spacing=1.5)
        ws = [_units_to_px(p, float(v)) for v in re.split(r"[,\s]+", str(a.get("width") or "30").strip()) if v]
        poly = dc.tube_polygon(axis, ws)
        w = _num(a, "weight", 1.2, 0.3, 6, ctx)

        def mutate():
            path = dc.resample(np.vstack([poly, poly[:1]]), 1.5)
            ctrl = dc.px2u(p, dc.rdp(path, 0.004 * max(p.W, p.H)))
            rec = draw_line.make_record(p, path, ctrl, weight=w, taper="none", color="line", closed=True,
                                        src="tube")
            d["lines"].append(rec)
            new_ids.append(rec["id"])
            return draw_line.render_ops(p, rec), _path_box(p, path)
        label = "line tube"
    elif sub == "eye":
        a = parse(rest, Spec("easel line eye eye-l --open 0.8", ["where*"], {"open": "value", "tilt": "value",
                                                                             "weight": "value", "lash": "value"},
                             min_pos=1))
        targets = []
        for w_ in re.split(r"[,\s]+(?=[a-zA-Z])", str(a["where"])):
            w_ = w_.strip()
            if not w_:
                continue
            if coords.looks_like_box(w_):
                targets.append((w_, coords.parse_box(w_, ctx.notes)))
            elif w_ in d["parts"]:
                targets.append((w_, d["parts"][w_]["box"]))
            else:
                raise UsageError(f"{w_} is not a planned part or a box", "easel line eye eye-l,eye-r")
        opn = _num(a, "open", 0.85, 0.2, 1.2, ctx)
        tilt = _num(a, "tilt", 6.0, -30, 30, ctx)
        w = _num(a, "weight", 0.9, 0.3, 4, ctx)
        lash = _num(a, "lash", 2.4, 0.5, 6, ctx)

        def mutate():
            ops, bb = [], None
            mid = np.mean([(b[0] + b[2]) / 2 for _, b in targets])
            for name, b in targets:
                outer_left = (b[0] + b[2]) / 2 < mid if len(targets) > 1 else True
                loop, lash_pts = draw_line.eye_shape(b, opn, tilt, outer_left)
                for pts_u, closed, wt, taper in ((loop, True, w, "none"), (lash_pts, False, lash, "both")):
                    path, _ = draw_line.fit(p, pts_u, closed=closed, snap=False, join=False, bow=0.0)
                    rec = draw_line.make_record(p, path, pts_u, weight=wt, taper=taper, color="line", closed=closed,
                                                src=f"eye {name}")
                    d["lines"].append(rec)
                    new_ids.append(rec["id"])
                    ops += draw_line.render_ops(p, rec)
                    bb = coords.union_box(bb, _path_box(p, path))
            return ops, bb
        label = f"line eye {', '.join(n for n, _ in targets)}"
    elif sub == "trace":
        a = parse(rest, Spec("easel line trace s3 --weight 1.2", ["id"], {"weight": "value", "color": "value",
                                                                          "taper": "value", "all": "flag",
                                                                          "outline": "flag"},
                             min_pos=1))
        el = draw_sketch.find(p, a["id"])
        if el["kind"] == "head" and a.get("outline"):
            # the whole head silhouette (skull and jaw as one closed line): animals, chibi heads, bald heads
            shp = draw_sketch.head_shapes(p, el)
            msk = np.zeros((p.H, p.W), np.uint8)
            cv2.fillPoly(msk, [np.round(shp[0][0]).astype(np.int32)], 1)
            jaw = np.vstack([shp[1][0], shp[2][0][::-1]])
            cv2.fillPoly(msk, [np.round(jaw).astype(np.int32)], 1)
            cs, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)
            polys = [(dc.resample(np.vstack([c, c[:1]]), 2.0)[:-1], True)]
        elif el["kind"] == "head" and not a.get("all"):
            # the face outline: left jaw, chin, right jaw as one line (the skull stays under the hair)
            shp = draw_sketch.head_shapes(p, el)
            jl, jr = shp[1][0], shp[2][0]
            polys = [(np.vstack([jl, jr[::-1][1:]]), False)]
        else:
            polys = [(poly, closed) for poly, closed, wf in draw_sketch.shapes(p, el) if wf >= 0.9]
        if not polys:
            raise UsageError(f"{el['id']} has nothing to trace", "easel sketch list")
        w = _num(a, "weight", 1.0, 0.3, 6, ctx)

        def mutate():
            ops, bb = [], None
            for poly, closed in polys:
                path = dc.resample(np.vstack([poly, poly[:1]]) if closed else poly, 1.5)
                ctrl = dc.px2u(p, dc.rdp(path, 0.01 * max(p.W, p.H)))
                rec = draw_line.make_record(p, path, ctrl, weight=w, taper=str(a.get("taper") or "both"),
                                            color=str(a.get("color") or "line"), closed=closed, src=f"trace {el['id']}")
                d["lines"].append(rec)
                new_ids.append(rec["id"])
                ops += draw_line.render_ops(p, rec)
                bb = coords.union_box(bb, _path_box(p, path))
            return ops, bb
        label = f"line trace {el['id']}"
    else:
        a = parse(args, Spec(USAGE["line"], ["pts*"], {"weight": "value", "taper": "value", "closed": "flag",
                                                      "no-snap": "flag", "snap": "flag", "no-join": "flag",
                                                      "join": "flag", "mirror": "value", "color": "value",
                                                      "bg": "flag", "size": "value", "bow": "value",
                                                      "behind": "value"},
                             alias={"colour": "color", "close": "closed", "width": "weight", "w": "weight"},
                             min_pos=1))
        if a.get("mirror") is None and "--mirror" in args:
            a["mirror"] = "auto"
        pts = _parse_pts(ctx, a["pts"], 2, USAGE["line"])
        w = _num(a, "weight", 1.0, 0.3, 6, ctx)
        if a.get("size") and not a.get("weight"):
            w = coords.parse_float(a["size"], "size", 0.5, 30, ctx.notes) / draw_line.base_width(p)
        taper = str(a.get("taper") or "both").lower()
        if taper not in ("both", "start", "end", "none"):
            raise UsageError("--taper is both, start, end or none", USAGE["line"])
        colour = str(a.get("color") or color_default)
        draw_plan.swatch(p, colour, f'easel line "..." --color line')     # validate early
        closed = bool(a.get("closed"))
        mirror = a.get("mirror")

        def mutate():
            bow = _num(a, "bow", None, -0.5, 0.5, ctx) if a.get("bow") else None
            path, nts = draw_line.fit(p, pts, closed=closed, snap=not a.get("no-snap"), join=not a.get("no-join"),
                                      bow=bow)
            lines.extend(nts)
            closed2 = closed or "the line is closed" in " ".join(nts)
            runs = [path]
            if a.get("behind"):
                # the parts inside the shapes in front are hidden: an ear behind the head, a staff behind a hand
                front = np.zeros((p.H, p.W), np.uint8)
                for fid in re.split(r"[,\s]+", str(a["behind"]).strip()):
                    if not fid:
                        continue
                    fr = draw_line.find(p, fid)
                    fp = draw_line.record_path(p, fr)
                    if len(fp) >= 3:
                        cv2.fillPoly(front, [np.round(fp).astype(np.int32)], 1)
                front = cv2.erode(front, np.ones((3, 3), np.uint8))
                xi = np.clip(np.round(path[:, 0]).astype(int), 0, p.W - 1)
                yi = np.clip(np.round(path[:, 1]).astype(int), 0, p.H - 1)
                vis = front[yi, xi] == 0
                runs, cur = [], []
                for q, v in zip(path, vis):
                    if v:
                        cur.append(q)
                    elif cur:
                        runs.append(np.array(cur))
                        cur = []
                if cur:
                    runs.append(np.array(cur))
                runs = [r_ for r_ in runs if len(r_) >= 3]
                if not runs:
                    raise UsageError("the whole line is behind those shapes", 'easel line "x,y ..." --behind L1')
                closed2 = False
                lines.append(f"{len(runs)} visible piece(s); the rest is behind {a['behind']}")
            ops, bb = [], None
            rec = None
            for run in runs:
                ctrl_ = pts if len(runs) == 1 and run is path else dc.px2u(p, dc.rdp(run, 0.004 * max(p.W, p.H)))
                rec = draw_line.make_record(p, run, ctrl_, weight=w, taper=taper, color=colour, closed=closed2)
                if a.get("bg"):
                    rec["bg"] = True
                d["lines"].append(rec)
                new_ids.append(rec["id"])
                ops += draw_line.render_ops(p, rec)
                bb = coords.union_box(bb, _path_box(p, run))
            if d.get("locks"):
                # hair locks stay in front of lines drawn after them
                try:
                    from . import draw_hair
                    lines.extend(draw_hair.refit(p))
                except Exception:  # noqa: BLE001
                    pass
            if mirror:
                ax = _mirror_axis(p, mirror, rec)
                mpts = [[2 * ax - x, y] for x, y in pts]
                mpath, mn = draw_line.fit(p, mpts, closed=closed, snap=not a.get("no-snap"),
                                          join=not a.get("no-join"))
                mrec = draw_line.make_record(p, mpath, mpts, weight=w, taper=taper, color=colour,
                                             closed=closed2, src=f"mirror of {rec['id']} at x={ax:g}")
                d["lines"].append(mrec)
                new_ids.append(mrec["id"])
                ops += draw_line.render_ops(p, mrec)
                bb = coords.union_box(bb, _path_box(p, mpath))
                lines.append(f"mirrored across x={ax:g}: {mrec['id']}")
            return ops, bb
        label = f"line {len(pts)} points weight {w:g}"
    rep = draw_step(ctx, label=label, layer="line", owned=_line_owned(p, mutate), lines=lines,
                    post=_line_post(p), overlay=_line_overlay(p), note="the line art (new ids in blue, gaps magenta)",
                    area_extra=_subject_box_px(p))
    if new_ids:
        rep.lines.insert(1, f"new: {', '.join(new_ids)}"
                         + (f" (part {draw_line.find(p, new_ids[0]).get('part')})"
                            if draw_line.find(p, new_ids[0]).get("part") else ""))
    return rep


def _path_box(p, path):
    a = np.asarray(path, float).reshape(-1, 2)
    if not len(a):
        return None
    pad = 3 * draw_line.base_width(p)
    return (int(max(0, a[:, 0].min() - pad)), int(max(0, a[:, 1].min() - pad)),
            int(min(p.W, a[:, 0].max() + pad + 1)), int(min(p.H, a[:, 1].max() + pad + 1)))


def _subject_box_px(p):
    d = dc.dstate(p)
    bb = None
    for s in d["subjects"].values():
        bb = coords.union_box(bb, coords.box_to_px(s["box"], p.W, p.H))
    return bb


def _mirror_axis(p, val, rec) -> float:
    s = str(val).lower().replace("x=", "").strip()
    if s not in ("auto", "", "true", "yes"):
        return coords.parse_float(s, "mirror", 0, 1000)
    d = dc.dstate(p)
    part = rec.get("part")
    if part and part in d["parts"]:
        subj = d["parts"][part]["of"]
        b = d["subjects"].get(subj, d["parts"][part])["box"]
        return (b[0] + b[2]) / 2
    if d["subjects"]:
        b = next(iter(d["subjects"].values()))["box"]
        return (b[0] + b[2]) / 2
    return 500.0


# ============================================================================================ flat

def _flat_overlay(p, labels, table, only=None):
    def ov(img, area):
        tags = []
        un = {r["n"] for r in draw_flat.unfilled(p, labels, table)}
        for r in table:
            if only is not None and r["n"] not in only:
                continue
            col = (220, 0, 120) if r["n"] in un else (30, 30, 30)
            tags.append((str(r["n"]), tuple(r["anchor"]), col))
        return tag_image(img, tags, area, 13)
    return ov


def cmd_flat(ctx: Ctx, args: List[str]) -> Report:
    p = ctx.project
    d = dc.dstate(p)
    sub = args[0].lower() if args else "show"
    gap = None
    if "--gap" in args:
        i = args.index("--gap")
        gap = coords.parse_int(args[i + 1], "gap", 0, 40, ctx.notes)
        args = args[:i] + args[i + 2:]
        sub = args[0].lower() if args else "show"
    labels, table, src = draw_flat.compute_areas(p, gap)
    if sub in ("show", "list", "areas"):
        rep = DReport()
        auto = draw_flat.auto_assign(p, labels, table)
        un = draw_flat.unfilled(p, labels, table)
        rep.add(f"{len(table)} enclosed areas in the {src} layer; {len(un)} without a flat colour")
        cells = []
        for r in table[:40]:
            guess = auto.get(r["n"])
            state = "open" if r in un else "filled"
            cells.append(f"{r['n']} {state}" + (f" ->{guess[1]}" if guess and state == "open" else ""))
        for i in range(0, len(cells), 6):
            rep.add("  " + " | ".join(cells[i:i + 6]))
        if not ctx.batch:
            img = _flat_overlay(p, labels, table)(p.composite_u8("view"), (0, 0, p.W, p.H))
            show_one(ctx, rep, img, (0, 0, p.W, p.H), "AREAS (magenta = no colour yet)", "numbered areas",
                     f"areas-{cli._view_counter(p):04d}", max_side=view.MAP_SIDE)
        rep.next = "easel flat auto" if not d["fills"] else (
            f"easel flat {','.join(str(r['n']) for r in un[:6])} --color <name>" if un else cli._finish_next(ctx))
        return rep
    assignments: List[Tuple[int, str, str]] = []      # (area, part, swatch)
    if sub == "auto":
        a = parse(args[1:], Spec("easel flat auto", [], {"redo": "flag"}))
        auto = draw_flat.auto_assign(p, labels, table)
        filled = {r["n"] for r in table} - {r["n"] for r in draw_flat.unfilled(p, labels, table)}
        for n, (part, sw) in sorted(auto.items()):
            if n in filled and not a.get("redo"):
                continue
            assignments.append((n, part, sw))
        missing = [r["n"] for r in table if r["n"] not in auto]
        label = f"flat auto: {len(assignments)} areas"
    elif sub == "refresh":
        for f in d["fills"]:
            x, y = int(np.clip(f["pt"][0], 0, p.W - 1)), int(np.clip(f["pt"][1], 0, p.H - 1))
            n = int(labels[y, x])
            if n:
                assignments.append((n, f["part"], f["swatch"]))
        missing = []
        label = "flat refresh (current palette)"
    elif sub in ("hole", "holes", "patch"):
        return _flat_hole(ctx, args[1:])
    else:
        a = parse(args, Spec("easel flat 3,5 --color skin", ["areas*"], {"color": "value", "part": "value",
                                                                         "at": "value"},
                             alias={"colour": "color", "region": "part", "name": "part"}, min_pos=0))
        if not a.get("at") and a.get("areas") and _looks_like_points(str(a["areas"]), len(table)):
            # "flat 700,700 --color torso": a point, the area under it
            ctx.notes.append(f"read {a['areas']} as a point: the area under it")
            a["at"] = a["areas"]
        if a.get("at"):
            # by place instead of number: the area under each point (no need to read numbers off the view)
            pts_at = coords.parse_points(a["at"], ctx.notes, 1)
            nums = []
            for ux, uy in pts_at:
                x_ = int(np.clip(round(ux * p.W / 1000), 0, p.W - 1))
                y_ = int(np.clip(round(uy * p.H / 1000), 0, p.H - 1))
                n_ = int(labels[y_, x_])
                if n_ <= 0:
                    raise UsageError(f"{ux:g},{uy:g} is on a line, not inside an area; move the point a little",
                                     "easel flat --at 450,700 --color scarf")
                if n_ not in nums:
                    nums.append(n_)
            a["areas"] = ",".join(map(str, nums))
        elif not a.get("areas"):
            raise UsageError("say which areas: numbers from VIEW, or --at x,y inside each", "easel flat 3,5 --color skin")
        else:
            nums = draw_flat.parse_areas(a["areas"], table, "easel flat 3,5 --color skin")
        if not a.get("color"):
            raise UsageError("say the colour: a palette name, #hex or a colour word", f"easel flat {a['areas']} "
                             "--color skin")
        sw0 = sw = str(a["color"]).strip().lower()
        if sw0 != "pick":
            draw_plan.swatch(p, sw, f"easel flat {a['areas']} --color skin")
        from .draw_plan import parts_at
        for n in nums:
            r = table[n - 1]
            if sw0 == "pick":       # the reference's colour inside this area (hand mode)
                from . import draw_ref
                sw = draw_ref.pick_area(p, labels == n)
            if a.get("part"):
                part = from_regions_clean(a["part"])
            else:
                hits = parts_at(d, dc.px2u(p, [r["anchor"]])[0])
                same = [h for h in hits if d["parts"][h].get("swatch") == sw]
                if not same:
                    # the colour names the part (a hanging scarf end below the scarf's box is still scarf)
                    same = [h for h, v in d["parts"].items() if v.get("swatch") == sw or h == sw]
                part = same[0] if same else (sw if sw in d["palette"]["swatches"] and not hits else
                                             (hits[0] if hits else re.sub(r"[^a-z0-9_\-]+", "-", sw).strip("-")
                                              or f"area{n}"))
            assignments.append((n, part, sw))
        missing = []
        label = f"flat {','.join(map(str, nums))} {sw0 if sw0 == 'pick' else sw}"
    if not assignments:
        rep = DReport()
        rep.add("nothing to fill: every area already has a colour" + (" (flat auto --redo to repaint)"
                                                                       if sub == "auto" else ""))
        rep.next = cli._finish_next(ctx)
        return rep
    parts_touched = sorted({part for _, part, _ in assignments} | set(p.state.get("regions", {})))
    files = [f"regions/{n}.png" for n in parts_touched]

    def build():
        ops = []
        for n, part, sw in assignments:
            m = labels == n
            ops.append(draw_flat.fill_op(m, draw_plan.swatch(p, sw)))
        return ops

    def post():
        for n, part, sw in assignments:
            r = table[n - 1]
            pt = [int(r["anchor"][0]), int(r["anchor"][1])]
            d["fills"] = [f for f in d["fills"] if int(labels[int(np.clip(f["pt"][1], 0, p.H - 1)),
                                                            int(np.clip(f["pt"][0], 0, p.W - 1))]) != n]
            d["fills"].append({"pt": pt, "part": part, "swatch": sw})
        regs = draw_flat.rebuild_regions(p, labels, table)
        out = []
        for name, m in regs.items():
            try:
                rn = from_regions_clean(name)
            except UsageError:
                continue
            p.save_region_mask(rn, m)
            box = coords.px_box_to_units(coords.mask_bbox_px(m), p.W, p.H)
            focus = d["parts"].get(name, {}).get("focus", 1)
            p.state["regions"][rn] = {"focus": int(focus), "box": list(box), "area": round(float(m.mean()), 4),
                                      "how": "flat areas"}
        by: Dict[str, List[int]] = {}
        for n, part, sw in assignments:
            by.setdefault(f"{part} [{sw}]", []).append(n)
        out.append("filled: " + "; ".join(f"{k} {','.join(map(str, v))}" for k, v in by.items())[:400])
        if missing:
            out.append(f"no plan part holds areas {','.join(map(str, missing[:12]))}: give them a colour:  "
                       f"easel flat {','.join(map(str, missing[:6]))} --color <name>")
        out.append(f"regions now: {', '.join(p.state['regions'])}")
        return out
    return draw_step(ctx, label=label, layer="flat", build=build, extra_files=files, post=post,
                     overlay=_flat_overlay(p, labels, table), full_view=True,
                     note="the flat colours with area numbers (magenta = still no colour)")


def _looks_like_points(text: str, n_areas: int) -> bool:
    """'700,700' or '230,640 770,640' (x,y pairs, a number beyond the area count or a space between pairs) rather
    than area numbers '3,5'."""
    toks = [t for t in re.split(r"\s+", text.strip()) if t]
    pairs = [re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", t) for t in toks]
    if not toks or not all(pairs):
        return False
    nums = [float(v) for m in pairs for v in m.groups()]
    return len(toks) > 1 or any(v > n_areas for v in nums) or any("." in t for t in toks)


def _flat_hole(ctx: Ctx, args: List[str]) -> Report:
    """easel flat hole x,y: the piece of background colour under the point (enclosed by a subject's outline or
    surrounded by its fills) takes the colour of the fill around it (or --color)."""
    p = ctx.project
    d = dc.dstate(p)
    a = parse(args, Spec("easel flat hole 230,640", ["pt*"], {"color": "value", "part": "value"},
                         alias={"colour": "color", "at": "pt"}, min_pos=1))
    pt = dc.u2px(p, [coords.parse_point(a["pt"], ctx.notes, "easel flat hole 230,640")])[0]
    m, region, colour = draw_flat.hole_mask(p, pt)
    if m is None:
        raise UsageError(f"no background colour under {a['pt']} (or the background there is the picture's own "
                         "background): point inside the gap", "easel flat hole 230,640")
    if a.get("color"):
        colour = draw_plan.swatch(p, a["color"])
        region = from_regions_clean(a["part"]) if a.get("part") else region
    if colour is None:
        raise UsageError("no fill around that gap to take the colour from: give --color", "easel flat hole 230,640 "
                         "--color torso")
    files = [f"regions/{region}.png"] if region else []
    for n in ("background", "bg"):
        if n in p.state.get("regions", {}):
            files.append(f"regions/{n}.png")

    def build():
        return [draw_flat.fill_op(m, colour)]

    def post():
        out = [f"filled the gap at {a['pt']}: {m.mean() * 100:.2f}% of the canvas in {coords.hex_colour(colour)}"
               + (f", now part of {region}" if region else "")]
        if region:
            old = p.region_mask(region) if region in p.state.get("regions", {}) else np.zeros((p.H, p.W), bool)
            nm = old | m
            p.save_region_mask(region, nm)
            info = p.state["regions"].setdefault(region, {"focus": d["parts"].get(region, {}).get("focus", 1),
                                                          "how": "flat areas"})
            info["box"] = list(coords.px_box_to_units(coords.mask_bbox_px(nm), p.W, p.H))
            info["area"] = round(float(nm.mean()), 4)
        for n in ("background", "bg"):
            if n in p.state.get("regions", {}):
                bm = p.region_mask(n) & ~m
                p.save_region_mask(n, bm)
                p.state["regions"][n]["area"] = round(float(bm.mean()), 4)
        return out
    return draw_step(ctx, label=f"flat hole {a['pt']}", layer="flat", build=build, extra_files=files, post=post,
                     area_extra=coords.mask_bbox_px(m), note="the gap filled with the colour around it")


def from_regions_clean(name: str) -> str:
    from .regions import clean_name
    return clean_name(name)


# ============================================================================================ shade family

def _targets(ctx: Ctx, spec: str, usage: str) -> List["cli.Region"]:
    p = ctx.project
    if str(spec).lower() in ("all", "every", "everything"):
        out = []
        d = dc.dstate(p)
        for n in p.region_names():
            if n in ("background", "bg", "sky") or d["parts"].get(n, {}).get("swatch") == "eye-white"                     or n.startswith("eye"):
                continue
            out.append(cli.resolve_region(p, n, ctx.notes, usage))
        if not out:
            raise UsageError("no coloured regions yet: flat makes them", "easel flat auto")
        return out
    return [region_or_part(ctx, s, usage) for s in re.split(r"[,\s]+", str(spec)) if s]


def cmd_shade(ctx: Ctx, args: List[str]) -> Report:
    p = ctx.project
    d = dc.dstate(p)
    a = parse(args, Spec(USAGE["shade"], ["region*"], {"levels": "value", "soft": "value", "amount": "value",
                                                      "color": "value", "round": "value", "light": "value"},
                         alias={"colour": "color", "level": "levels", "softness": "soft", "depth": "amount"},
                         min_pos=1))
    regs = _targets(ctx, a["region"], USAGE["shade"])
    levels = coords.parse_int(a["levels"], "levels", 1, 3, ctx.notes) if a.get("levels") else 1
    # cel shading is crisp unless the request asks for soft shading
    req = str(p.state.get("request") or "").lower()
    soft = _num(a, "soft", 0.15 if ("soft" in req or (p.state.get("style") or "") in ("paint", "photo")) else 0.0,
                0, 1, ctx)
    amount = _num(a, "amount", None, 0, 1, ctx) if a.get("amount") else None
    rnd_ = _num(a, "round", 1.0, 0.2, 3, ctx)
    light = dc.light_vec(d)
    if a.get("light"):
        deg, z, _ = dc.parse_light(a["light"])
        light = dc.light_vec({"light": {"deg": deg, "z": z}})
    clip = np.zeros((p.H, p.W), bool)
    for r in regs:
        clip |= r.mask
    info = []

    def build():
        from .ops import Fill, mask_to_rle
        ops = []
        old_sh = p.layer(_shade_layer(p), create=False) if _shade_layer(p) == "shade" else None
        for r in regs:
            if old_sh is not None and (old_sh[..., 3][r.mask] > 0.02).any():
                # shading a part again replaces its shading (after a colour change, a new light)
                ops.append(Fill(mask_rle=mask_to_rle(r.mask), color=(0, 0, 0), erase=True, feather=1.0))
            base, sh, li, nm = colour_set(p, r, a.get("color"))
            size = math.sqrt(max(1, r.mask.sum()))
            fe = 1.0 + soft * 0.06 * size
            amt = amount if amount is not None else (0.25 if nm in ("skin", "face") else 0.35)
            tones = draw_shade.form_masks(r.mask, light, levels, amt, rnd_, body=_inner_parts(p, r))
            heads = [e for e in d["sketch"] if e["kind"] == "head"]
            if nm in ("skin", "face") and heads and r.name in ("face", "head", "skin"):
                # the nose throws a small shadow on the side away from the light
                g = draw_sketch.head_guides(p, heads[-1])
                hb = heads[-1]["box"]
                nx, ny = dc.u2px(p, [g["nose"]])[0]
                hw = (hb[2] - hb[0]) * p.W / 1000
                hh = (hb[3] - hb[1]) * p.H / 1000
                ox = -float(np.sign(light[0]) or 1) * 0.025 * hw
                nm_ = np.zeros((p.H, p.W), np.uint8)
                cv2.ellipse(nm_, (int(nx + ox), int(ny - 0.01 * hh)), (max(2, int(0.022 * hw)), max(3, int(0.045 * hh))),
                            0, 0, 360, 1, -1)
                nose = nm_.astype(bool) & r.mask
                if nose.sum() > 12:
                    tones = list(tones) + [("nose", nose)]
            for name, m in tones:
                col = sh if name in ("shadow", "nose") else dc.shadow_of(sh, 0.8) if name == "deep" else li
                ops.append(draw_shade.mask_fill(m, col, 1.0 if name != "light" else 0.9,
                                                fe if name != "deep" else fe * 1.3))
            info.append(f"{r.name}: {', '.join(n for n, _ in tones) or 'too small'} in {nm}-shadow"
                        + (f" {coords.hex_colour(sh)}" if nm.startswith("#") else ""))
        return ops
    rep = draw_step(ctx, label=f"shade {', '.join(r.name for r in regs)} levels {levels} soft {soft:g}",
                    layer=_shade_layer(p), build=build, clip=clip, post=lambda: ["; ".join(info)[:300]],
                    full_view=len(regs) > 2, area_extra=None,
                    note=f"shading, light from the {dc.light_words(d['light'].get('deg', 135))}")
    return rep


def cmd_shadow(ctx: Ctx, args: List[str]) -> Report:
    p = ctx.project
    d = dc.dstate(p)
    a = parse(args, Spec(USAGE["shadow"], [], {"from": "value", "onto": "value", "offset": "value", "soft": "value",
                                              "color": "value", "opacity": "value"},
                         alias={"on": "onto", "to": "onto", "colour": "color", "src": "from"}))
    if not a.get("from") or not a.get("onto"):
        raise UsageError("shadow needs --from <region that casts> and --onto <region it falls on>", USAGE["shadow"])
    src = region_or_part(ctx, a["from"], USAGE["shadow"])
    dst = region_or_part(ctx, a["onto"], USAGE["shadow"])
    off = _units_to_px(p, _num(a, "offset", 22, 1, 200, ctx))
    soft = _num(a, "soft", 0.2, 0, 1, ctx)
    base, sh, li, nm = colour_set(p, dst, a.get("color"))
    light = dc.light_vec(d)
    m = draw_shade.cast_mask(src.mask, dst.mask, light, off, 1.0 + soft)

    def build():
        if m.sum() < 12:
            return []
        return [draw_shade.mask_fill(m, sh, _num(a, "opacity", 1.0, 0, 1, ctx), 1.0 + soft * 0.4 * off)]

    def post():
        lst = d.setdefault("cast", [])
        lst.append([src.name, dst.name])
        return [f"cast shadow {src.name} -> {dst.name}: {m.mean() * 100:.1f}% of the canvas"
                + ("; nothing fell on it: raise --offset or check the light" if m.sum() < 12 else "")]
    return draw_step(ctx, label=f"shadow {src.name} onto {dst.name} offset {off:.0f}px", layer=_shade_layer(p), build=build,
                     clip=dst.mask, post=post, area_extra=dst.box_px, note="the cast shadow")


def cmd_rim(ctx: Ctx, args: List[str]) -> Report:
    p = ctx.project
    d = dc.dstate(p)
    a = parse(args, Spec(USAGE["rim"], ["region*"], {"width": "value", "color": "value", "opacity": "value",
                                                    "back": "flag"}, alias={"colour": "color", "size": "width"},
                         min_pos=1))
    regs = _targets(ctx, a["region"], USAGE["rim"])
    wpx = _units_to_px(p, _num(a, "width", 6, 1, 60, ctx))
    light = dc.light_vec(d)
    back = bool(a.get("back")) or d["light"].get("z", 0.6) < 0
    clip = np.zeros((p.H, p.W), bool)
    for r in regs:
        clip |= r.mask

    def build():
        ops = []
        for r in regs:
            base, sh, li, nm = colour_set(p, r)
            col = draw_plan.swatch(p, a["color"]) if a.get("color") else dc.light_of(li, 1.5)
            others = np.zeros((p.H, p.W), bool)
            for n in p.region_names():
                if n not in ("background", "bg", "sky") and n != r.name:
                    others |= p.region_mask(n)
            m = draw_shade.rim_mask(r.mask, light, wpx, back, open_side=~others)
            if m.sum() >= 12:
                ops.append(draw_shade.mask_fill(m, col, _num(a, "opacity", 0.85, 0, 1, ctx), max(1.0, wpx * 0.3)))
        return ops
    return draw_step(ctx, label=f"rim {', '.join(r.name for r in regs)} width {wpx:.0f}px", layer=_shade_layer(p),
                     build=build, clip=clip, note="rim light on the edges facing the light")


def cmd_crease(ctx: Ctx, args: List[str]) -> Report:
    p = ctx.project
    a = parse(args, Spec(USAGE["crease"], ["region*"], {"width": "value", "opacity": "value", "with": "value"},
                         alias={"size": "width"}, min_pos=1))
    regs = _targets(ctx, a["region"], USAGE["crease"])
    wpx = _units_to_px(p, _num(a, "width", 14, 2, 80, ctx))
    clip = np.zeros((p.H, p.W), bool)
    for r in regs:
        clip |= r.mask

    def build():
        ops = []
        for r in regs:
            others = np.zeros((p.H, p.W), bool)
            names = [x for x in re.split(r"[,\s]+", a["with"]) if x] if a.get("with") else \
                [n for n in p.region_names() if n not in (r.name, "background", "bg", "sky")]
            for n in names:
                others |= cli.resolve_region(p, n, ctx.notes, USAGE["crease"]).mask
            m = draw_shade.crease_mask(r.mask, others & ~r.mask, wpx)
            base, sh, li, nm = colour_set(p, r)
            if m.sum() >= 12:
                ops.append(draw_shade.mask_fill(m, dc.shadow_of(sh, 0.5), _num(a, "opacity", 0.55, 0, 1, ctx),
                                                wpx * 0.8))
        return ops
    return draw_step(ctx, label=f"crease {', '.join(r.name for r in regs)}", layer=_shade_layer(p), build=build, clip=clip,
                     note="soft occlusion where the region meets its neighbours")


def cmd_blush(ctx: Ctx, args: List[str]) -> Report:
    p = ctx.project
    d = dc.dstate(p)
    a = parse(args, Spec(USAGE["blush"], ["where*"], {"size": "value", "color": "value", "opacity": "value",
                                                     "lines": "flag", "layer": "value"},
                         alias={"colour": "color"}, min_pos=1))
    where = str(a["where"])
    col = draw_plan.swatch(p, a["color"]) if a.get("color") else (
        draw_plan.swatch(p, "blush") if "blush" in d["palette"]["swatches"] else (240, 140, 140))
    op = _num(a, "opacity", 0.45, 0, 1, ctx)
    clip = None
    if coords.looks_like_point(where) or len(coords._numbers(where)) >= 4 and len(coords._numbers(where)) % 2 == 0 \
            and not coords.looks_like_box(where):
        pts = dc.u2px(p, coords.parse_points(where, ctx.notes))
        size = _units_to_px(p, _num(a, "size", 40, 4, 300, ctx))
    else:
        r = region_or_part(ctx, where, USAGE["blush"])
        clip = r.mask
        x0, y0, x1, y1 = r.box_px
        w, h = x1 - x0, y1 - y0
        head = next((el for el in d["sketch"] if el["kind"] == "head"), None)
        if head is not None:
            g = draw_sketch.head_guides(p, head)
            le, re_, mo = (dc.u2px(p, [g["left eye"]])[0], dc.u2px(p, [g["right eye"]])[0],
                           dc.u2px(p, [g["mouth"]])[0])
            yy = le[1] + 0.55 * (mo[1] - le[1])
            pts = np.array([[le[0] - 0.15 * w, yy], [re_[0] + 0.15 * w, yy]])
        else:
            pts = np.array([[x0 + 0.27 * w, y0 + 0.62 * h], [x0 + 0.73 * w, y0 + 0.62 * h]])
        size = _units_to_px(p, _num(a, "size", 0, 0, 300, ctx)) or 0.22 * w

    def build():
        ops = [draw_shade.dab(q, size, col, op, "soft", k) for k, q in enumerate(pts)]
        if a.get("lines"):
            from .ops import Stroke
            lc = dc.shadow_of(col, 0.8)
            for q in pts:
                for k in range(3):
                    x = q[0] - size * 0.25 + k * size * 0.22
                    seg = np.array([[x + size * 0.08, q[1] - size * 0.12], [x - size * 0.08, q[1] + size * 0.12]])
                    wl = max(1.0, size * 0.035)
                    if np.hypot(*(seg[1] - seg[0])) >= 4 * wl:
                        rs = dc.resample(seg, 1.0)
                        ops.append(Stroke(pts=rs.astype(np.float32), width=dc.taper_widths(rs, wl, "both"), color=lc,
                                          brush="liner"))
        return ops
    layer = p.layer_entry(a["layer"])["name"] if a.get("layer") else _shade_layer(p)
    rname = None if clip is None else str(a["where"]).strip()
    return draw_step(ctx, label=f"blush {len(pts)} dabs size {size:.0f}px", layer=layer, build=build, clip=clip,
                     post=lambda: [_detailed(p, rname)] if rname else [], note="airbrush blush")


def cmd_highlight(ctx: Ctx, args: List[str]) -> Report:
    p = ctx.project
    d = dc.dstate(p)
    a = parse(args, Spec(USAGE["highlight"], ["region*"], {"at": "value", "size": "value", "color": "value",
                                                          "opacity": "value", "soft": "flag"},
                         alias={"colour": "color"}, min_pos=1))
    r = region_or_part(ctx, a["region"], USAGE["highlight"])
    light = dc.light_vec(d)
    if a.get("at"):
        pt = dc.u2px(p, [coords.parse_point(a["at"], ctx.notes)])[0]
    else:
        pt = draw_shade.brightest_point(r.mask, light)
        if pt is None:
            raise UsageError(f"{r.name} is too small for a highlight", USAGE["highlight"])
    x0, y0, x1, y1 = r.box_px
    size = _units_to_px(p, _num(a, "size", 0, 0, 200, ctx)) or 0.12 * min(x1 - x0, y1 - y0)
    base, sh, li, nm = colour_set(p, r)
    col = draw_plan.swatch(p, a["color"]) if a.get("color") else dc.light_of(li, 1.6)

    def build():
        from .ops import Fill
        ang = math.degrees(math.atan2(light[1], light[0])) + 90
        e = dc.ellipse_pts(pt[0], pt[1], size * 0.5, size * 0.28, ang, n=40)
        return [Fill(polygons=[e.astype(np.float32)], color=col, opacity=_num(a, "opacity", 0.85, 0, 1, ctx),
                     feather=max(1.0, size * (0.25 if a.get("soft") else 0.06)))]
    return draw_step(ctx, label=f"highlight {r.name} at {dc.fmt_pt(p, pt)}", layer="detail", build=build,
                     clip=r.mask, post=lambda: [_detailed(p, r.name)], area_extra=r.box_px,
                     note="the highlight")


def _detailed(p, name) -> str:
    _mark_list(dc.dstate(p), "detailed", name)
    return ""


# ============================================================================================ detail tools

def cmd_hair(ctx: Ctx, args: List[str]) -> Report:
    p = ctx.project
    d = dc.dstate(p)
    a = parse(args, Spec(USAGE["hair"], ["region"], {"flow": "value", "flow2": "value", "flow3": "value",
                                                    "count": "value", "width": "value", "no-shine": "flag",
                                                    "color": "value"},
                         alias={"colour": "color", "n": "count", "size": "width"}, min_pos=1))
    r = region_or_part(ctx, a["region"], USAGE["hair"])
    if not a.get("flow"):
        raise UsageError("hair needs --flow \"x,y x,y x,y\": a curve the way the hair falls (root to tip); add "
                         "--flow2 for a second direction", f'easel hair {r.name} --flow "450,150 420,300 400,480"')
    curves = [dc.u2px(p, coords.parse_points(a[k], ctx.notes, 2)) for k in ("flow", "flow2", "flow3") if a.get(k)]
    base, sh, li, nm = colour_set(p, r, a.get("color"))
    size = math.sqrt(max(1, r.mask.sum()))
    count = int(_num(a, "count", max(30, min(160, size / 4)), 5, 600, ctx))
    w = _units_to_px(p, _num(a, "width", 0, 0, 20, ctx)) or max(1.2, size / 160)
    light = dc.light_vec(d)
    # the strands stay on the hair: inside its flats, near the flow curves given, and along the drawn locks
    guides = []
    lk = (d.get("locks") or {}).get(r.name)
    if lk:
        guides = [dc.u2px(p, k["path"]) for k in lk.get("locks", []) if len(k.get("path", [])) >= 2
                  and not k.get("poly")]
    near = 0.2 * size
    clip = r.mask & ~_other_regions(p, r.name)

    def post():
        return [_detailed(p, r.name), f"strands start within {_px_to_units(p, near):.0f} units of the flow curve(s)"
                + (f" and follow the {len(guides)} locks of {r.name}" if guides else "")
                + "; another part of the hair: one more  easel hair  with a flow curve there"]
    return draw_step(ctx, label=f"hair {r.name}: {count} strands along {len(curves)} flow curve(s)", layer="detail",
                     build=lambda: draw_detail.hair_ops(clip, curves, base, dc.shadow_of(sh, 0.6), li, light, count,
                                                        w, not a.get("no-shine"), seed=p.state["step"] + 1,
                                                        guides_px=guides, near=near),
                     clip=clip, post=post, area_extra=r.box_px,
                     note="strands and shine")


def _px_to_units(p, px: float) -> float:
    return float(px) * 1000.0 / max(p.W, p.H)


def _other_regions(p, name: str) -> np.ndarray:
    """The coloured areas of every other region but the background: a detail mark on ``name`` stays off them."""
    out = np.zeros((p.H, p.W), bool)
    for n in p.region_names():
        if n in (name, "background", "bg", "sky"):
            continue
        try:
            out |= p.region_mask(n)
        except Exception:  # noqa: BLE001
            continue
    return out


def cmd_eye(ctx: Ctx, args: List[str]) -> Report:
    p = ctx.project
    d = dc.dstate(p)
    a = parse(args, Spec(USAGE["eye"], ["region*"], {"color": "value", "look": "value", "box": "value"},
                         alias={"colour": "color", "iris": "color"}, min_pos=0))
    if a.get("box") or (a.get("region") and coords.looks_like_box(a["region"])):
        box = coords.parse_box(a.get("box") or a["region"], ctx.notes, USAGE["eye"])
        bp = coords.box_to_px(box, p.W, p.H)
        m = dc.poly_mask(p.H, p.W, dc.ellipse_pts((bp[0] + bp[2]) / 2, (bp[1] + bp[3]) / 2, (bp[2] - bp[0]) / 2,
                                                  (bp[3] - bp[1]) / 2))
        name = "eye box"
        comps = [m]
        bpx = bp
    else:
        if not a.get("region"):
            eyes = [n for n in p.region_names() if n.startswith("eye")]
            if not eyes:
                raise UsageError("name the eye region (or give --box)", USAGE["eye"])
            a["region"] = ",".join(eyes)
        regs = [region_or_part(ctx, s, USAGE["eye"]) for s in re.split(r"[,\s]+", a["region"]) if s]
        comps = []
        from scipy import ndimage as ndi
        for r in regs:
            lab, n = ndi.label(r.mask)
            sizes = ndi.sum(r.mask, lab, range(1, n + 1)) if n else []
            big = max(sizes) if n else 0
            comps += [lab == k + 1 for k in range(n) if sizes[k] > 0.15 * big]
        name = ", ".join(r.name for r in regs)
        bpx = None
        for r in regs:
            bpx = coords.union_box(bpx, r.box_px)
    iris = draw_plan.swatch(p, a["color"]) if a.get("color") else (
        draw_plan.swatch(p, "iris") if "iris" in d["palette"]["swatches"] else (70, 110, 190))
    look = (0.0, 0.0)
    clip = np.zeros((p.H, p.W), bool)
    for m in comps:
        clip |= m
    if a.get("look"):
        tx, ty = dc.u2px(p, [coords.parse_point(a["look"], ctx.notes)])[0]
        ys, xs = np.nonzero(clip)
        cx, cy = xs.mean(), ys.mean()
        dv = np.array([tx - cx, ty - cy])
        n = np.linalg.norm(dv)
        look = tuple(np.clip(dv / max(n, 1e-6) * min(1.0, n / (0.05 * max(p.W, p.H))), -1, 1))
    light = dc.light_vec(d)
    lid = dc.shadow_of(draw_plan.swatch(p, "eye-white") if "eye-white" in d["palette"]["swatches"]
                       else (240, 240, 250), 1.2)

    def build():
        ops = []
        for k, m in enumerate(comps):
            ops += draw_detail.eye_ops(m, iris, look, lid, light, seed=k)
        return ops
    return draw_step(ctx, label=f"eye {name}: {len(comps)} eye(s), iris {coords.hex_colour(iris)}", layer="detail",
                     build=build, clip=clip, post=lambda: [_detailed(p, n_) for n_ in name.split(", ")] + [
                         "the line art stays on top; redraw a lash line with  easel line  if it is too thin"],
                     area_extra=bpx, note="the eyes")


def cmd_fold(ctx: Ctx, args: List[str]) -> Report:
    p = ctx.project
    d = dc.dstate(p)
    a = parse(args, Spec(USAGE["fold"], ["pts*"], {"region": "value", "depth": "value", "width": "value",
                                                  "color": "value"}, alias={"colour": "color", "on": "region"},
                         min_pos=1))
    pts = dc.u2px(p, coords.parse_points(a["pts"], ctx.notes, 2, USAGE["fold"]))
    if not a.get("region"):
        # no region given: the fold belongs to the coloured area most of its points lie on, and stays in it
        votes: Dict[str, int] = {}
        for n in p.region_names():
            if n in ("background", "bg", "sky"):
                continue
            m_ = p.region_mask(n)
            k_ = sum(1 for q in pts if 0 <= int(q[1]) < p.H and 0 <= int(q[0]) < p.W and m_[int(q[1]), int(q[0])])
            if k_:
                votes[n] = k_
        if votes:
            a["region"] = max(votes, key=votes.get)
            ctx.notes.append(f"the fold stays inside {a['region']} (the area its points lie on)")
    r = region_or_part(ctx, a["region"], USAGE["fold"]) if a.get("region") else None
    if r is not None:
        base, sh, li, nm = colour_set(p, r)
    else:
        img = dc.composite_u8(p)
        q = pts.mean(0).astype(int)
        base = tuple(int(v) for v in img[min(p.H - 1, q[1]), min(p.W - 1, q[0])])
        sh = dc.shadow_of(base)
    line_rgb = draw_plan.swatch(p, a["color"]) if a.get("color") else draw_plan.swatch(p, "line") \
        if "line" in d["palette"]["swatches"] else (40, 30, 40)
    w = draw_line.base_width(p) * _num(a, "width", 0.6, 0.2, 3, ctx)
    light = dc.light_vec(d)
    return draw_step(ctx, label=f"fold {len(pts)} points" + (f" on {r.name}" if r else ""), layer="detail",
                     build=lambda: draw_detail.fold_ops(pts, w, line_rgb, sh, light, _num(a, "depth", 1.0, 0.2, 3, ctx),
                                                        base_rgb=base),
                     clip=r.mask if r is not None else None,
                     post=lambda: [_detailed(p, r.name)] if r is not None else [], area_extra=r.box_px if r else None,
                     note="the fold")


def cmd_pattern(ctx: Ctx, args: List[str]) -> Report:
    p = ctx.project
    a = parse(args, Spec(USAGE["pattern"], ["region", "kind"], {"spacing": "value", "angle": "value", "size": "value",
                                                               "color": "value", "opacity": "value", "layer": "value"},
                         alias={"colour": "color", "gap": "spacing"}, min_pos=2))
    kind = str(a["kind"]).lower()
    kinds = ("hatch", "cross", "dots", "tone", "stripes", "checks")
    if kind not in kinds:
        raise UsageError(f"unknown pattern {kind!r}; one of {', '.join(kinds)}", USAGE["pattern"])
    r = region_or_part(ctx, a["region"], USAGE["pattern"], allow_box=True)
    base, sh, li, nm = colour_set(p, r)
    col = draw_plan.swatch(p, a["color"]) if a.get("color") else sh
    sp = _units_to_px(p, _num(a, "spacing", 12, 2, 200, ctx))
    size = _units_to_px(p, _num(a, "size", 0, 0, 60, ctx)) or (max(1.0, sp * 0.12) if kind in ("hatch", "cross")
                                                              else max(3.0, sp * 0.45))
    layer = p.layer_entry(a["layer"])["name"] if a.get("layer") else "detail"
    op = _num(a, "opacity", 1.0, 0, 1, ctx)

    def build():
        ops = draw_detail.pattern_ops(r.box_px, kind, col, sp, _num(a, "angle", 45, -180, 180, ctx), size)
        for o in ops:
            o.opacity = op
        return ops
    return draw_step(ctx, label=f"pattern {kind} on {r.name}", layer=layer, build=build, clip=r.mask,
                     post=lambda: [_detailed(p, r.name)], area_extra=r.box_px, note=f"{kind} pattern")


def cmd_scatter(ctx: Ctx, args: List[str]) -> Report:
    p = ctx.project
    a = parse(args, Spec(USAGE["scatter"], ["where*"], {"kind": "value", "count": "value", "size": "value",
                                                       "color": "value", "colors": "value", "spread": "value",
                                                       "angle": "value", "jitter": "value", "opacity": "value",
                                                       "layer": "value", "region": "value"},
                         alias={"colour": "color", "colours": "colors", "n": "count", "stamp": "kind"}, min_pos=1))
    kind = str(a.get("kind") or "dot").lower()
    if kind not in draw_detail.STAMPS:
        raise UsageError(f"unknown --kind {kind!r}; stamps: {', '.join(draw_detail.STAMPS)}", USAGE["scatter"])
    where = str(a["where"])
    mask, path, clip, r = None, None, None, None
    nums = coords._numbers(where)
    if len(nums) >= 4 and len(nums) % 2 == 0 and not coords.looks_like_box(where) or \
            (len(nums) >= 4 and re.search(r"\d\s+\d", where)):
        path = dc.u2px(p, coords.parse_points(where, ctx.notes, 2))
        if a.get("region"):
            r = region_or_part(ctx, a["region"], USAGE["scatter"])
            clip = r.mask
    else:
        r = region_or_part(ctx, where, USAGE["scatter"], allow_box=True)
        mask = r.mask
        clip = r.mask
    cs = a.get("colors") or a.get("color")
    if cs:
        cols = [draw_plan.swatch(p, c.strip()) for c in re.split(r"[,;]\s*", str(cs)) if c.strip()]
    elif r is not None:
        base, sh, li, nm = colour_set(p, r)
        cols = [sh, base, li]
    else:
        cols = [(40, 40, 40)]
    size = _units_to_px(p, _num(a, "size", 10, 1, 200, ctx))
    exact = path is not None and not a.get("count")      # points and no --count: one mark on each point
    count = len(path) if exact else int(_num(a, "count", 40, 1, 3000, ctx))
    layer = p.layer_entry(a["layer"])["name"] if a.get("layer") else "detail"
    ang = _num(a, "angle", None, -360, 360, ctx) if a.get("angle") else None

    def build():
        if exact:
            pts_, sz = path, size
            if a.get("region"):
                # --region: each mark moves to the middle of that shape nearby and fits inside it with a margin
                rm = p.region_mask(str(a["region"]).strip())
                dt = cv2.distanceTransform(rm.astype(np.uint8), cv2.DIST_L2, 5)
                rad = int(max(6, 0.04 * max(p.W, p.H)))
                moved = []
                P_ = np.asarray(path, float)
                u_ = P_[-1] - P_[0] if len(P_) > 1 else np.array([1.0, 0.0])
                u_ = u_ / (np.linalg.norm(u_) + 1e-9)
                nrm_ = np.array([-u_[1], u_[0]])
                for q in P_:
                    # slide across the row of marks only, so their spacing along it stays as given
                    best, bv = q, -1.0
                    for t_ in np.linspace(-rad, rad, 2 * rad + 1):
                        c_ = q + nrm_ * t_
                        xi, yi = int(round(c_[0])), int(round(c_[1]))
                        if 0 <= xi < p.W and 0 <= yi < p.H and dt[yi, xi] > bv:
                            best, bv = c_, float(dt[yi, xi])
                    moved.append(best)
                    if bv > 0:
                        sz = min(sz, 2 * 0.62 * bv)
                pts_ = np.array(moved, float)
            return draw_detail.stamp_ops(kind, pts_, sz, cols, ang, _num(a, "opacity", 1.0, 0, 1, ctx),
                                         seed=p.state["step"] + 1)
        return draw_detail.scatter_ops(kind, mask=mask, path_px=path, count=count, size=size, colors=cols,
                                       angle=ang, jitter=_num(a, "jitter", 0.5, 0, 1, ctx),
                                       spread=_units_to_px(p, _num(a, "spread", 8, 0, 300, ctx)),
                                       opacity=_num(a, "opacity", 1.0, 0, 1, ctx), seed=p.state["step"] + 1)
    return draw_step(ctx, label=f"scatter {count} {kind}" + (" on the points" if exact else
                                                              f" in {r.name}" if r is not None and mask is not None
                                                              else " along the curve"),
                     layer=layer, build=build, clip=clip,
                     post=lambda: [_detailed(p, r.name)] if r is not None else [],
                     area_extra=r.box_px if r is not None else None, note=f"{kind} stamps")


# ============================================================================================ painterly

def cmd_paint(ctx: Ctx, args: List[str]) -> Report:
    p = ctx.project
    d = dc.dstate(p)
    a = parse(args, Spec(USAGE["paint"], ["region*"], {"dir": "value", "flow": "value", "size": "value",
                                                      "brush": "value", "density": "value", "color": "value",
                                                      "light": "value"},
                         alias={"direction": "dir", "angle": "dir", "colour": "color"}, min_pos=1))
    regs = _targets(ctx, a["region"], USAGE["paint"])
    brush = str(a.get("brush") or "flat").lower()
    from . import manual
    names = manual.brush_names(manual.engine())
    if brush not in names:
        raise UsageError(f"unknown --brush {brush!r}; brushes: {', '.join(names)}", USAGE["paint"])
    light = dc.light_vec(d)
    if a.get("light"):
        deg, z, _ = dc.parse_light(a["light"])
        light = dc.light_vec({"light": {"deg": deg, "z": z}})
    # a region paints only where it shows: nearer subjects (plan depth order) keep their own paint
    from .draw_checks import depth_order
    order = depth_order(p)

    def subject_of(n):
        s0 = draw_checks.subject_of(p, n)
        return s0 if s0 in order else None

    def visible(r):
        s0 = subject_of(r.name)
        if s0 is None:
            return r.mask
        nearer = set(order[order.index(s0) + 1:])
        m = r.mask.copy()
        for n in p.state.get("regions", {}):
            if n != r.name and subject_of(n) in nearer:
                m &= ~p.region_mask(n)
        return m
    masks = {r.name: visible(r) for r in regs}
    clip = np.zeros((p.H, p.W), bool)
    for r in regs:
        clip |= masks[r.name]
    info = []

    def build():
        ops = []
        under = dc.composite_u8(p)          # strokes pick up the paint already there (wet blending)
        for k, r in enumerate(regs):
            base, sh, li, nm = colour_set(p, r, a.get("color"))
            x0, y0, x1, y1 = r.box_px
            size = _units_to_px(p, _num(a, "size", 0, 0, 200, ctx)) or max(3.0, min(x1 - x0, y1 - y0) / 14)
            if a.get("flow"):
                field = draw_detail.flow_field([dc.u2px(p, coords.parse_points(a["flow"], ctx.notes, 2))],
                                               (p.H, p.W))
            else:
                field = draw_detail.angle_field(_num(a, "dir", 0.0, -360, 360, ctx), (p.H, p.W))
            ops += draw_detail.brush_ops(masks[r.name], field, base, sh, li, light, size, brush,
                                         _num(a, "density", 1.0, 0.2, 4, ctx), seed=p.state["step"] * 31 + k,
                                         under=under)
            info.append(f"{r.name} size {size:.0f}px")
        return ops

    def post():
        for r in regs:
            _mark_list(d, "painted", r.name)
        return ["; ".join(info)]
    return draw_step(ctx, label=f"paint {', '.join(r.name for r in regs)} ({brush})", layer="paint", build=build,
                     clip=clip, post=post, full_view=len(regs) > 1, area_extra=None,
                     note="brushwork, lighter on the lit side")


def cmd_band(ctx: Ctx, args: List[str]) -> Report:
    p = ctx.project
    d = dc.dstate(p)
    a = parse(args, Spec(USAGE["band"], ["name", "pts*"], {"above": "flag", "below": "flag", "closed": "flag",
                                                          "to": "value", "color": "value", "color2": "value",
                                                          "rough": "value", "depth": "value", "layer": "value",
                                                          "focus": "value", "along": "value", "width": "value",
                                                          "tiers": "value", "ellipse": "value", "behind": "value",
                                                          "front": "flag", "shift": "value"},
                         alias={"colour": "color", "colour2": "color2", "shape": "closed", "mass": "closed",
                                "widths": "width"},
                         min_pos=1))
    name = from_regions_clean(a["name"])
    if a.get("shift"):
        return _band_shift(ctx, name, a["shift"])
    if a.get("ellipse"):
        # a clean oval: a backdrop disc, a moon, a spotlight
        bx = coords.parse_box(a["ellipse"], ctx.notes)
        x0_, y0_, x1_, y1_ = coords.box_to_px(bx, p.W, p.H)
        pts = dc.ellipse_pts((x0_ + x1_) / 2, (y0_ + y1_) / 2, (x1_ - x0_) / 2, (y1_ - y0_) / 2)
        a["closed"] = True
        a["tiers_done"] = True
    elif a.get("tiers"):
        # a tiered silhouette (a conifer, a pagoda roof) from its axis: base point and top point, the width at the
        # base, the number of tiers
        ax = dc.u2px(p, coords.parse_points(a.get("pts") or "", ctx.notes, 2))
        n_t = coords.parse_int(a["tiers"], "tiers", 2, 12, ctx.notes)
        wb = _units_to_px(p, _num(a, "width", 160, 10, 1000, ctx) if a.get("width") else 160)
        from . import draw_scene
        # each tree its own: proportions, tier spacing, tips and notches from the band's name and the step
        pts = draw_scene.tier_outline(ax[0], ax[-1], wb, n_t, seed=draw_scene.shape_seed(name, p.state["step"]))
        a["closed"] = True
        a["width"] = None
        a["tiers_done"] = True
    elif a.get("along") or a.get("width"):
        # a tube along an axis (a sketch curve id or points) with widths along its length: a saxophone body, a
        # tail, a branch, a river, in a dozen numbers instead of an outline
        src = a.get("along") or a.get("pts")
        if not src:
            raise UsageError("say the axis: --along s3 (a sketch curve) or points", 'easel band tail --along s3 '
                             '--width "20,40,30,5" --color fur')
        if re.fullmatch(r"\s*s\d+\s*", str(src)):
            el = draw_sketch.find(p, str(src).strip())
            axis = max((poly for poly, closed, wf in draw_sketch.shapes(p, el)), key=lambda q: len(q))
        else:
            axis = dc.catmull_rom(dc.u2px(p, coords.parse_points(src, ctx.notes, 2)), spacing=1.5)
        ws = [float(v) for v in re.split(r"[,\s]+", str(a.get("width") or "30").strip()) if v]
        a["closed"] = True
        a["pts"] = None
        pts = dc.tube_polygon(axis, [_units_to_px(p, w) for w in ws])
    else:
        if not a.get("pts"):
            raise UsageError("band needs points (or --along an axis with --width, or --ellipse x0,y0,x1,y1)",
                             USAGE["band"])
        pts = dc.u2px(p, coords.parse_points(a["pts"], ctx.notes, 2, USAGE["band"]))
    ridge_note = _band_ridge_note(p, name, a, pts) if a.get("closed") and not a.get("tiers_done") else ""
    if not a.get("color"):
        raise UsageError("band needs --color (a palette name, #hex or a colour word)", USAGE["band"])
    c1 = draw_plan.swatch(p, a["color"])
    c2 = draw_plan.swatch(p, a["color2"]) if a.get("color2") else None
    rough = _num(a, "rough", 0.0, 0, 1, ctx)
    side = "above" if a.get("above") else "below"
    painterly = (p.state.get("style") or "paint") in ("paint", "photo")
    layer = p.layer_entry(a["layer"])["name"] if a.get("layer") else ("paint" if painterly else "flat")
    if a.get("closed"):
        if len(pts) < 3:
            raise UsageError("a closed shape needs 3+ points", USAGE["band"])
        poly = pts if a.get("width") or a.get("along") or a.get("tiers_done") else             dc.corner_spline(pts, closed=True, spacing=2.0)     # smooth, sharp where it turns hard
        poly = draw_detail.rough_polygon(poly, rough, max(p.W, p.H), seed=p.state["step"] + 1)
        how = "closed shape"
    else:
        to = dc.u2px(p, coords.parse_points(a["to"], ctx.notes, 2)) if a.get("to") else None
        poly = draw_detail.band_polygon(pts, p.W, p.H, side, rough, p.state["step"] + 1, to)
        how = f"band {'between two curves' if to is not None else side + ' the curve'}"
    m = dc.poly_mask(p.H, p.W, poly)
    band_clip = None
    depth_note = ""
    if not a.get("behind") and not a.get("front"):
        # depth order: a farther subject drawn now goes behind the nearer ones already painted (a sun drawn after
        # the mountains sets behind them)
        subj = draw_checks.subject_of(p, name)
        if subj is None and m.any():
            ys_, xs_ = np.nonzero(m)
            cu = (xs_.mean() * 1000 / p.W, ys_.mean() * 1000 / p.H)
            subj = draw_plan.subject_at(d, cu)
        if subj:
            front_m, front_n = draw_checks.nearer_mask(p, subj)
            hid = float((front_m & m).sum()) / max(1, m.sum())
            if front_n and 0 < hid < 0.7:
                m = m & ~front_m
                band_clip = m
                depth_note = (f"behind {', '.join(front_n)} (nearer in the depth order; draw it in front with "
                              f"--front, change the order with  easel plan order far,...,near)")
            elif front_n and hid >= 0.7:
                depth_note = (f"in front of {', '.join(front_n)}: they would hide {hid:.0%} of it, so the depth "
                              f"order is probably off:  easel plan order far,...,near")
    front_regions: List[str] = []
    m_region: Optional[np.ndarray] = None
    keep: List[str] = []       # regions that stay whole: the ones this band goes behind
    if a.get("behind"):
        # a shape behind others already painted (a window frame behind a plant): it shows only around them, and
        # around every subject nearer than it in the depth order too (a wall behind the cat is also behind the
        # window, the sill and the pots)
        front = np.zeros((p.H, p.W), bool)
        for fn in re.split(r"[,\s]+", str(a["behind"]).strip()):
            if fn:
                rr = region_or_part(ctx, fn, USAGE["band"])
                front |= rr.mask
                front_regions.append(rr.name)
        subj_b = draw_checks.subject_of(p, name)
        if subj_b is None and m.any():
            ys_, xs_ = np.nonzero(m)
            subj_b = draw_plan.subject_at(d, (xs_.mean() * 1000 / p.W, ys_.mean() * 1000 / p.H))
        for rn in _band_nearer_regions(p, subj_b):
            if rn != name and rn not in front_regions:
                front |= p.region_mask(rn)
                front_regions.append(rn)
        keep = list(front_regions)
        m = m & ~front
        if name in d["bands"] and name in p.state.get("regions", {}):
            # drawn again behind: the shape moves back, it does not shrink to the new outline (an oval over a
            # rectangle wall keeps the wall's corners)
            m_region = (p.region_mask(name) | m) & ~front
        band_clip = m
    elif band_clip is not None:
        front_regions = _band_nearer_regions(p, subj)
        keep = list(front_regions)
    # the same colour drawn earlier over the shapes in front (a sun disc painted over the mountains with  shape)
    # goes: those pixels take the front shape's own colour again
    cleanup: List[Tuple[np.ndarray, tuple]] = []
    lay_arr = p.layer(layer, create=False)
    if front_regions and lay_arr is not None:
        rgb_l = np.clip(lay_arr[..., :3], 0, 1) * 255
        d1 = np.sqrt(((rgb_l - np.array(c1, float)) ** 2).sum(-1))
        close = (d1 < 30) & (lay_arr[..., 3] > 0.5)
        for fn in front_regions:
            if fn not in d["bands"]:
                continue
            try:
                fc = draw_plan.swatch(p, d["bands"][fn]["color"])
            except UsageError:
                continue
            # only the far colour itself, not the front shape's own brushwork in a near hue (an orange cat on a
            # warm wall keeps its fur)
            d2 = np.sqrt(((rgb_l - np.array(fc, float)) ** 2).sum(-1))
            hit = close & (d1 * 2.0 < d2) & p.region_mask(fn)
            hit = cv2.morphologyEx(hit.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
            if hit.sum() >= 12:
                cleanup.append((cv2.dilate(hit.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                                & p.region_mask(fn), fc))
    if cleanup and band_clip is not None:
        band_clip = band_clip.copy()
        for cm, _ in cleanup:
            band_clip |= cm
    # bands drawn earlier that this one covers lose that part of their region; the ones it goes behind keep theirs
    later = [n for n in d["bands"] if n != name and n not in keep
             and not (a.get("behind") and n in str(a["behind"]))]
    files = [f"regions/{name}.png"] + [f"regions/{n}.png" for n in later]

    def build():
        from .ops import Fill, mask_to_rle
        kw = {}
        if c2 is not None:
            ys = poly[:, 1]
            kw = dict(color2=c2, p0=(float(poly[:, 0].mean()), float(max(0, ys.min()))),
                      p1=(float(poly[:, 0].mean()), float(min(p.H, ys.max()))))
        ops = [Fill(mask_rle=mask_to_rle(cm), color=tuple(int(v) for v in fc), feather=1.0) for cm, fc in cleanup]
        own = Fill(polygons=[poly.astype(np.float32)], color=c1, feather=1.0, **kw)
        if cleanup and band_clip is not None:
            own = Fill(mask_rle=mask_to_rle(m), color=c1, feather=1.0, **kw)
        return ops + [own]

    def post():
        prev_depths = [b.get("depth", 1) for b in d["bands"].values()]
        depth = _num(a, "depth", None, 1, 20, ctx) if a.get("depth") else (max(prev_depths) - 1 if prev_depths
                                                                            and side == "below" else 5)
        rm = m if m_region is None else m_region
        bb_ = coords.mask_bbox_px(rm)
        fill_ = float(rm.sum()) / max(1, (bb_[2] - bb_[0]) * (bb_[3] - bb_[1])) if bb_ else 0.0
        d["bands"][name] = {"color": str(a["color"]), "color2": a.get("color2"), "depth": depth, "how": how,
                            "depth_set": bool(a.get("depth")), "layer": layer, "fill": round(fill_, 3)}
        p.save_region_mask(name, rm)
        box = coords.px_box_to_units(coords.mask_bbox_px(rm), p.W, p.H)
        focus = coords.parse_int(a["focus"], "focus", 1, 3) if a.get("focus") else \
            p.state["regions"].get(name, {}).get("focus", 1)
        p.state["regions"][name] = {"focus": focus, "box": list(box), "area": round(float(rm.mean()), 4),
                                    "how": how}
        cut = []
        for n in later:
            if n in p.state["regions"]:
                om = p.region_mask(n)
                if (om & m).any():
                    nm2 = om & ~m
                    if nm2.any():
                        p.save_region_mask(n, nm2)
                        p.state["regions"][n]["area"] = round(float(nm2.mean()), 4)
                        cut.append(n)
        dn = depth_note or (f"behind {', '.join(keep)} (they keep their paint and regions)"
                            if a.get("behind") and keep else "")
        return [f"region {name}: {rm.mean() * 100:.0f}% of the canvas ({how}), depth {depth:g}"
                + (f"; it covers part of {', '.join(cut)} (their regions now end at its edge)" if cut else "")
                + (f"; drawn {dn}" if dn else "")] + ([ridge_note] if ridge_note else [])
    return draw_step(ctx, label=f"band {name} {coords.hex_colour(c1)}" + (f" -> {coords.hex_colour(c2)}" if c2 else ""),
                     layer=layer, build=build, extra_files=files, post=post, full_view=True, clip=band_clip,
                     note=f"the new {how}")


RIDGE_WORDS = ("mountain", "ridge", "hill", "range", "peak", "cliff", "dune", "skyline")


def _band_ridge_note(p, name: str, a: dict, pts: np.ndarray) -> str:
    """A mountain or ridge band whose top edge is an even zigzag or a sine wave: a line with an irregular contour
    in its box to draw instead (uneven peak spacing and heights, one dominant peak)."""
    from . import draw_scene
    subj = draw_checks.subject_of(p, name) or ""
    toks = re.split(r"[^a-z]+", (name + " " + subj).lower())
    if not any(tk.startswith(w) for tk in toks for w in RIDGE_WORDS):
        return ""
    reg = draw_scene.ridge_regularity(pts)
    if reg is None:
        return ""
    cv_gap, cv_h, n_pk = reg
    if cv_gap >= 0.15 and cv_h >= 0.12:
        return ""
    bx = [int(round(v)) for v in coords.px_box_to_units(coords.mask_bbox_px(dc.poly_mask(p.H, p.W, pts)), p.W, p.H)]
    sug = draw_scene.irregular_ridge(bx, draw_scene.shape_seed(name, p.state["step"]), peaks=max(4, min(n_pk, 6)))
    pts_s = " ".join(f"{x},{y}" for x, y in sug)
    what = "evenly spaced" if cv_gap < 0.15 else "all about the same height"
    return (f"note: the ridge of {name} is a regular zigzag ({n_pk} peaks {what}); real ridges have uneven peaks "
            f"and one that dominates. Redraw it with uneven spacing and heights, e.g.  easel band {name} "
            f"\"{pts_s}\" --closed --color {a.get('color') or name}")


def _band_nearer_regions(p, subject: Optional[str]) -> List[str]:
    """The regions of the subjects nearer than ``subject`` in the depth order (the subject, its parts and the bands
    that sit on it): what a band of ``subject`` stays behind."""
    if not subject:
        return []
    order = draw_checks.depth_order(p)
    if subject not in order:
        return []
    d = dc.dstate(p)
    regs = p.state.get("regions", {})
    out: List[str] = []
    for s in order[order.index(subject) + 1:]:
        own = [s] + [n for n, v in d["parts"].items() if v["of"] == s] + \
            [n for n in d.get("bands", {}) if n not in d["subjects"] and n not in d["parts"]
             and draw_checks.region_subject(p, n) == s]
        out += [n for n in own if n in regs and n not in out]
    return out


def _band_shift(ctx: Ctx, name: str, by: str) -> Report:
    """band <name> --shift dx,dy: move a band's shape (its colour and its region); where it was takes the colour
    around it."""
    p = ctx.project
    d = dc.dstate(p)
    if name not in d["bands"] or name not in p.state.get("regions", {}):
        raise UsageError(f"{name} is not a band; bands: {', '.join(d['bands']) or 'none'}",
                         "easel band sax --shift -100,0")
    nums = [v for v, _, _ in coords._numbers(by)]
    if len(nums) != 2:
        raise UsageError("--shift needs dx,dy in units", f"easel band {name} --shift -100,0")
    dxp, dyp = _units_to_px(p, nums[0]), _units_to_px(p, nums[1])
    painterly = (p.state.get("style") or "paint") in ("paint", "photo")
    layer = d["bands"][name].get("layer") or ("paint" if painterly else "flat")
    M = p.region_mask(name)
    Mt = cv2.warpAffine(M.astype(np.uint8), np.float32([[1, 0, dxp], [0, 1, dyp]]), (p.W, p.H),
                        flags=cv2.INTER_NEAREST) > 0
    arr = p.layer(layer)
    ring = cv2.dilate(M.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~M & (arr[..., 3] > 0.5)
    under = tuple(int(v) for v in np.median(arr[..., :3][ring], axis=0) * 255) if ring.any() else None
    inside = arr[..., :3][M & (arr[..., 3] > 0.5)]
    colour = tuple(int(v) for v in np.median(inside, axis=0) * 255) if len(inside) else \
        draw_plan.swatch(p, d["bands"][name]["color"])

    def build():
        from .ops import Fill, mask_to_rle
        ops = []
        gone = M & ~Mt
        if gone.any():
            ops.append(Fill(mask_rle=mask_to_rle(gone), color=under or (0, 0, 0), erase=under is None, feather=1.0))
        ops.append(Fill(mask_rle=mask_to_rle(Mt), color=colour, feather=1.0))
        return ops

    def post():
        p.save_region_mask(name, Mt)
        box = coords.px_box_to_units(coords.mask_bbox_px(Mt), p.W, p.H)
        p.state["regions"][name].update({"box": list(box), "area": round(float(Mt.mean()), 4)})
        return [f"band {name} moved by {nums[0]:g},{nums[1]:g}: now {coords.fmt_box(box)}; where it was takes "
                f"{coords.hex_colour(under) if under else 'nothing (transparent)'}"]
    return draw_step(ctx, label=f"band {name} shift {nums[0]:g},{nums[1]:g}", layer=layer, build=build,
                     extra_files=[f"regions/{name}.png"], post=post, full_view=True, note=f"{name} moved")


def cmd_haze(ctx: Ctx, args: List[str]) -> Report:
    p = ctx.project
    d = dc.dstate(p)
    a = parse(args, Spec(USAGE["haze"], [], {"color": "value", "amount": "value", "layer": "value"},
                         alias={"colour": "color"}))
    bands = {n: b for n, b in d["bands"].items() if n in p.state.get("regions", {})}
    if not bands:
        raise UsageError("haze works on bands (easel band ...) ordered by --depth", USAGE["band"])
    sky = next((n for n in ("sky", "background") if n in d["palette"]["swatches"]), None)
    col = draw_plan.swatch(p, a["color"]) if a.get("color") else (draw_plan.swatch(p, sky) if sky else (200, 210, 225))
    amt = _num(a, "amount", 0.5, 0, 1, ctx)
    far = {}
    if any(b.get("depth_set") for b in bands.values()):
        depths = [b.get("depth", 1) for b in bands.values()]
        dmin, dmax = min(depths), max(depths)
        far = {n: (b.get("depth", 1) - dmin) / max(dmax - dmin, 1e-6) for n, b in bands.items()}
    else:
        # no --depth given: a band whose mass sits lower in the picture is nearer (landscape order)
        cy = {}
        for n in bands:
            if n in ("sky", "background"):
                continue
            ys_ = np.nonzero(p.region_mask(n))[0]
            if len(ys_) and ys_.max() < 0.9 * p.H:      # a mass that reaches the bottom edge is foreground
                cy[n] = float(np.median(ys_))
        if len(cy) > 1:
            lo_, hi_ = min(cy.values()), max(cy.values())
            far = {n: 1 - (v - lo_) / max(hi_ - lo_, 1e-6) for n, v in cy.items()}
        elif cy:
            far = {n: 1.0 for n in cy}          # one band between the sky and the foreground: it is the far one
    layer = p.layer_entry(a["layer"])["name"] if a.get("layer") else "paint"
    info = []

    def build():
        ops = []
        for n in bands:
            if n in ("sky", "background") or n not in far:
                continue
            t = far[n]
            if t <= 0.01:
                continue
            ops.append(draw_shade.mask_fill(p.region_mask(n), col, amt * t * 0.85, 2.0))
            info.append(f"{n} {amt * t * 0.85:.2f}")
        return ops
    return draw_step(ctx, label=f"haze amount {amt:g}", layer=layer, build=build, post=lambda: ["; ".join(info)],
                     full_view=True, note="atmosphere: far bands fade toward the sky colour")


def cmd_detail(ctx: Ctx, args: List[str]) -> Report:
    p = ctx.project
    d = dc.dstate(p)
    rep = DReport()
    rep.add("detail tools (the tool lays the marks; you say where and which way):")
    rep.add('  hair <region> --flow "x,y x,y x,y" [--flow2 ...]   strands along the fall of the hair + shine')
    rep.add("  eye <region> [--color iris] [--look x,y]          iris, pupil, catch-lights, lid shadow")
    rep.add('  fold "x,y x,y x,y" --region <r>                   a cloth fold with its shadow')
    rep.add("  pattern <region> hatch|cross|dots|stripes|checks   clipped to the region")
    rep.add("  scatter <region|\"x,y ...\"> --kind leaf|petal|grass|star|dot|dash|needle|snow --count n")
    rep.add("  highlight <region>   blush <region>   rim <region>   crease <region>")
    for n in workflow_draw._ordered_parts(p.state):
        f = d["parts"][n].get("focus", 2)
        if f >= 2:
            rep.add(f"  {n} (focus {f}):  {workflow_draw.detail_command(n, f)[1]}")
    rep.next = cli._finish_next(ctx)
    return rep


# ============================================================================================ install

DRAW_ROUTED = {"plan": cmd_plan, "palette": cmd_palette, "sketch": cmd_sketch, "line": cmd_line, "ink": cmd_line,
               "flat": cmd_flat, "shade": cmd_shade, "paint": cmd_paint, "detail": cmd_detail}
DRAW_ALWAYS = {"shadow": cmd_shadow, "rim": cmd_rim, "crease": cmd_crease, "blush": cmd_blush,
               "highlight": cmd_highlight, "hair": cmd_hair, "eye": cmd_eye, "fold": cmd_fold, "pattern": cmd_pattern,
               "scatter": cmd_scatter, "band": cmd_band, "haze": cmd_haze}

_ORIG: Dict[str, Callable] = {}


def _working_canvas(ctx: Ctx) -> None:
    """A free project gets its working canvas (3x the delivered picture) before the first thing is drawn."""
    try:
        p = ctx.project
    except UsageError:
        return
    if p.ref is not None or p.state.get("work_scale"):
        return
    from . import draw_canvas
    for ln in draw_canvas.ensure(p):
        ctx.notes.append(ln)


def _noref(ctx: Ctx) -> bool:
    try:
        p = ctx.project
    except UsageError:
        return False
    return p.ref is None or p.state.get("mode") == "hand"


def _route(orig, fn):
    def run(ctx, args):
        if orig is None or _noref(ctx):
            _working_canvas(ctx)
            return fn(ctx, args)
        return orig(ctx, args)
    run.__name__ = getattr(fn, "__name__", "draw")
    run._draw = True  # type: ignore[attr-defined]
    return run


def install(commands: Optional[dict] = None) -> None:
    """Route the draw commands in ``commands`` (default ``cli.COMMANDS``) and the workflow functions for projects
    without a reference. Safe to call twice."""
    if commands is None:
        commands = cli.COMMANDS
    for name, fn in DRAW_ROUTED.items():
        cur = commands.get(name)
        if getattr(cur, "_draw", False):
            continue
        commands[name] = _route(cur, fn)
    for name, fn in DRAW_ALWAYS.items():
        if not getattr(commands.get(name), "_draw", False):
            commands[name] = _route(None, fn)
    for k, v in USAGE.items():
        cli.USAGE.setdefault(k, v)
    cur = commands.get("do")
    if cur is not None and not getattr(cur, "_plan_view", False):
        commands["do"] = _wrap_do(cur)
    for mod in _extensions():
        for name, fn in getattr(mod, "COMMANDS", {}).items():
            if not getattr(commands.get(name), "_draw", False):
                commands[name] = _route(commands.get(name), fn)
        for k, v in getattr(mod, "USAGE", {}).items():
            cli.USAGE.setdefault(k, v)
            USAGE.setdefault(k, v)
    for mod in _extensions():
        # WRAP = {name: factory(current command) -> command}: a module that adds sub-commands to an existing one
        # several modules may wrap one command (draw_ref and draw_text both wrap text): each wraps it once
        for name, wrap in getattr(mod, "WRAP", {}).items():
            cur = commands.get(name)
            done = getattr(cur, "_wrapped_by", ())
            if cur is not None and mod.__name__ not in done:
                new = wrap(cur)
                new._draw = new._wrapped = True  # type: ignore[attr-defined]
                new._wrapped_by = tuple(done) + (mod.__name__,)  # type: ignore[attr-defined]
                commands[name] = new
    _patch_workflow()


PLAN_CMDS = ("plan", "palette")


def _wrap_do(orig):
    """A batch of plan / palette commands shows the plan view (boxes, light, palette), not the empty canvas."""
    def run(ctx, args):
        rep = orig(ctx, args)
        try:
            if not _noref(ctx) or getattr(ctx, "exit_code", 0):
                return rep
            text = " ".join(args) if (any(";" in x for x in args) or len(args) == 1) else " ; ".join(args)
            cmds = cli.split_batch(text)
            heads = [c.split()[1] if c.split()[0] == "easel" and len(c.split()) > 1 else c.split()[0]
                     for c in cmds if c.split()]
            if heads and all(h in PLAN_CMDS for h in heads):
                p = ctx.project
                p.reload()
                img = draw_plan.plan_overlay(p, p.composite_u8("view"))
                rep.view = cli._save_view(ctx, view.render([view.Panel(img, "PLAN: subjects, parts, light, palette")],
                                                           (0, 0, p.W, p.H), max_side=view.MAP_SIDE),
                                          f"plan-{cli._view_counter(p):04d}")
                rep.view_note = "subject boxes (black), part boxes 'name f<focus>', light arrow, palette strip"
                if str(rep.next or "").startswith("easel plan   (look"):
                    rep.next = cli._finish_next(ctx)
        except Exception:  # noqa: BLE001 - the batch ran; a view problem must not hide its report
            pass
        return rep
    run.__name__ = "do"
    run._plan_view = True  # type: ignore[attr-defined]
    return run


EXTENSIONS = ("draw_face", "draw_hair", "draw_wear", "draw_scene", "draw_ref", "draw_text", "draw_canvas",
              "draw_surface")


def _extensions() -> list:
    """Command modules of the drawing half (each defines COMMANDS {name: fn(ctx, args)} and USAGE). A module that
    fails to import is left out, so one broken extension never stops easel."""
    import importlib
    out = []
    for name in EXTENSIONS:
        try:
            out.append(importlib.import_module(f"{__package__}.{name}"))
        except Exception:  # noqa: BLE001
            continue
    return out


_LAST = {"draw": False}


def _patch_workflow() -> None:
    if getattr(workflow, "_draw_patched", False):
        return
    o_next, o_gate, o_todo, o_lines, o_words = (workflow.next_action, workflow.check_gate, workflow.build_todo,
                                                workflow.plan_lines, workflow.gate_words)

    # every wrapper passes its arguments through untouched, so a change of signature on the workflow side
    # (a new optional parameter) never breaks a project with a reference
    def next_action(project, *args, **kw):
        _LAST["draw"] = workflow_draw.is_draw(project)
        if _LAST["draw"] and project.state.get("mode") == "hand":
            from . import draw_ref
            return draw_ref.hand_next(project, *workflow_draw.next_action(project))
        return workflow_draw.next_action(project) if _LAST["draw"] else o_next(project, *args, **kw)

    def check_gate(project, stage, *args, **kw):
        _LAST["draw"] = workflow_draw.is_draw(project)
        return workflow_draw.check_gate(project, stage) if _LAST["draw"] else o_gate(project, stage, *args, **kw)

    def build_todo(state, *args, **kw):
        _LAST["draw"] = workflow_draw.is_draw(state)
        return workflow_draw.build_todo(state) if _LAST["draw"] else o_todo(state, *args, **kw)

    def plan_lines(state, *args, **kw):
        return workflow_draw.plan_lines(state) if workflow_draw.is_draw(state) else o_lines(state, *args, **kw)

    def gate_words(stage, *args, **kw):
        if _LAST["draw"] and stage in workflow_draw.GATE_WORDS:
            return workflow_draw.GATE_WORDS[stage]
        return o_words(stage, *args, **kw)

    workflow.next_action, workflow.check_gate, workflow.build_todo = next_action, check_gate, build_todo
    workflow.plan_lines, workflow.gate_words = plan_lines, gate_words
    workflow._draw_patched = True


def main(argv: Optional[List[str]] = None) -> int:
    install()
    return cli.main(argv)


if __name__ == "__main__":
    sys.exit(main())
