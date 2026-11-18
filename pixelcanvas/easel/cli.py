"""The ``easel`` command (DESIGN.md sections 6 and 9).

Every command prints a short report (at most 25 lines) that ends with one ``NEXT:`` line; commands that change the
picture also write an observation picture and print ``VIEW: <path>``. Usage errors print what was wrong and a
corrected example and exit with 2; nothing ever prints a stack trace (details go to ``<project>/errors.log``).
"""
from __future__ import annotations

import io
import os
import re
import shlex
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import compose, coords, manual, metrics, regions, view, workflow
from .coords import UsageError
from .manual import EngineMissing
from .project import LAYER_NAMES, Project, save_png

MAX_LINES = 25


# ============================================================================================ reports

@dataclass
class Report:
    lines: List[str] = field(default_factory=list)
    view: Optional[str] = None
    view_note: str = ""
    view_ref: Optional[str] = None
    summary: str = ""        # short measured effect ("lines 30 -> 52") for batch reports
    worse: str = ""          # set when the step lowered its stage's key score
    step: Optional[int] = None
    next: str = ""
    touched: Optional[Tuple[int, int, int, int]] = None   # px box changed (for batches)
    mutated: bool = False

    def add(self, line: str) -> None:
        self.lines.append(line)

    def text(self) -> str:
        body = [ln for ln in self.lines if ln is not None]
        tail = []
        if self.view:
            tail.append(f"VIEW: {self.view}" + (f"   ({self.view_note})" if self.view_note else ""))
        if self.view_ref:
            tail.append(f"VIEW-REF: {self.view_ref}   (the reference over the same area)")
        tail.append("NEXT: " + (self.next or "easel next"))
        room = MAX_LINES - len(tail)
        if len(body) > room:
            cut = len(body) - room + 1
            body = body[: room - 1] + [f"... ({cut} more lines; see  easel status)"]
        return "\n".join(body + tail)


class Ctx:
    def __init__(self, project_arg: Optional[str] = None, batch: bool = False):
        self.project_arg = project_arg
        self._project: Optional[Project] = None
        self.notes: List[str] = []
        self.batch = batch
        self.cmdline = ""

    @property
    def project(self) -> Project:
        if self._project is None:
            self._project = Project.find(self.project_arg, notes=self.notes)
        return self._project


# ============================================================================================ argument parsing

@dataclass
class Spec:
    usage: str
    pos: List[str] = field(default_factory=list)       # positional names; a trailing '*' joins the rest
    opts: Dict[str, str] = field(default_factory=dict)  # name -> 'value' | 'flag'
    alias: Dict[str, str] = field(default_factory=dict)
    min_pos: int = 0


def parse(tokens: List[str], spec: Spec) -> Dict[str, object]:
    import difflib
    out: Dict[str, object] = {}
    posvals: List[str] = []
    i = 0
    names = list(spec.opts)
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("-") and len(t) > 1 and not re.match(r"^-\d", t):
            key, val = t.lstrip("-"), None
            if "=" in key:
                key, val = key.split("=", 1)
            key = key.lower().replace("_", "-")
            key = spec.alias.get(key, key)
            if key not in spec.opts:
                close = difflib.get_close_matches(key, names, n=1, cutoff=0.5)
                hint = f" (did you mean --{close[0]}?)" if close else ""
                opts = " ".join("--" + n for n in names) or "none"
                raise UsageError(f"unknown option --{key}{hint}; options: {opts}", spec.usage)
            if spec.opts[key] == "flag":
                if val is not None and val.lower() in ("0", "false", "no", "off"):
                    out[key] = False
                else:
                    out[key] = True
            else:
                if val is None:
                    if i + 1 >= len(tokens):
                        if key in ("color", "colors"):
                            raise UsageError(f"--{key} needs a value; in the shell an unquoted # starts a comment and "
                                             f"drops the rest of the line: quote it, --{key} \"#ffffff\", or leave "
                                             "out the #: --color ffffff", spec.usage)
                        raise UsageError(f"--{key} needs a value", spec.usage)
                    i += 1
                    val = tokens[i]
                    # values like "100, 200" split by the shell: glue following number tokens
                    while i + 1 < len(tokens) and re.fullmatch(r"[\d.,%()\[\]\s-]+", tokens[i + 1]) and \
                            re.search(r"[,(\[]\s*$", val):
                        i += 1
                        val += " " + tokens[i]
                out[key] = val
        else:
            posvals.append(t)
        i += 1
    star = spec.pos and spec.pos[-1].endswith("*")
    plain = [p.rstrip("*") for p in spec.pos]
    if star:
        n_fixed = len(plain) - 1
        for k, v in zip(plain[:n_fixed], posvals[:n_fixed]):
            out[k] = v
        if len(posvals) > n_fixed:
            out[plain[-1]] = " ".join(posvals[n_fixed:])
    else:
        if len(posvals) > len(plain):
            # coordinates split by the shell ("200, 100, 600, 500"): join into the last positional
            if plain and all(re.fullmatch(r"[\d.,%()\[\]-]+", v) for v in posvals[len(plain) - 1:]):
                posvals = posvals[: len(plain) - 1] + [" ".join(posvals[len(plain) - 1:])]
            else:
                raise UsageError(f"too many arguments: {' '.join(posvals[len(plain):])!r}", spec.usage)
        for k, v in zip(plain, posvals):
            out[k] = v
    if len([p for p in plain if p in out]) < spec.min_pos:
        missing = [p for p in plain if p not in out][0]
        raise UsageError(f"missing <{missing}>", spec.usage)
    return out


def _num(a, key, default, lo=None, hi=None, notes=None, usage=""):
    if a.get(key) is None:
        return default
    return coords.parse_float(a[key], key, lo, hi, notes, usage)


# ============================================================================================ shared helpers

class Region:
    def __init__(self, name: str, mask: np.ndarray, box_px, label: str, named: bool):
        self.name, self.mask, self.box_px, self.label, self.named = name, mask, box_px, label, named


def resolve_region(p: Project, spec: Optional[str], notes: list, usage: str = "easel ink hair") -> Region:
    H, W = p.H, p.W
    if spec is None or str(spec).strip().lower() in ("", "all", "canvas", "whole", "everything", "*"):
        return Region("all", np.ones((H, W), bool), (0, 0, W, H), "all", True)
    s = str(spec).strip()
    m = regions.MARK_RE.match(s)
    if m:
        labels = _marks_labels(p)
        k = int(m.group(1))
        mask = labels == k
        if not mask.any():
            raise UsageError(f"there is no mark {k}; marks go from 1 to {int(labels.max())}", "easel marks")
        return Region(f"m{k}", mask, coords.mask_bbox_px(mask), f"mark {k}", True)
    if coords.looks_like_box(s):
        box = coords.parse_box(s, notes, usage)
        bp = coords.box_to_px(box, W, H)
        return Region(coords.fmt_box(box), regions.box_mask(H, W, bp), bp, f"box {coords.fmt_box(box)}", False)
    names = p.region_names()
    hit = regions.match_name(s, names)
    if hit is None and p.state.get("draw"):
        # a planned part or subject (draw tools): its box
        try:
            from .draw_core import dstate
            d = dstate(p)
            for group in ("parts", "subjects"):
                e = (d.get(group) or {}).get(s.lower())
                if e and e.get("box"):
                    bp = coords.box_to_px(e["box"], W, H)
                    return Region(s.lower(), regions.box_mask(H, W, bp), bp, f"{group[:-1]} {s.lower()}", False)
            names = names + sorted((d.get("parts") or {}).keys())
        except UsageError:
            raise
        except Exception:  # noqa: BLE001
            pass
    if hit is None:
        sug = regions.suggest(s, names)
        known = ", ".join(names) if names else "none yet"
        hint = f" (did you mean {sug}?)" if sug else ""
        ex = usage.replace("hair", sug or (names[0] if names else "all"))
        raise UsageError(f"unknown region {s!r}{hint}; regions: {known}; also: all, a box x0,y0,x1,y1, m<mark>", ex)
    mask = p.region_mask(hit)
    box = coords.mask_bbox_px(mask) or (0, 0, W, H)
    return Region(hit, mask, box, hit, True)


def _marks_labels(p: Project) -> np.ndarray:
    m = p.state.get("marks")
    if not m:
        raise UsageError("no marks yet; make them first", "easel marks")
    from PIL import Image
    return np.asarray(Image.open(p.root / m["file"]), dtype=np.int32)


def _view_counter(p: Project) -> int:
    f = p.root / "views" / "counter.txt"
    try:
        n = int(f.read_text(encoding="utf-8").strip()) + 1
    except (OSError, ValueError):
        n = 1
    f.write_text(str(n), encoding="utf-8")
    return n


def _save_view(ctx: Ctx, img: np.ndarray, tag: str) -> str:
    p = ctx.project
    path = p.save_view(img, tag)
    return p.rel(path)


def _ref_view(ctx: Ctx, area) -> Optional[str]:
    """The reference over ``area`` as its own picture; the same area always gives the same file."""
    p = ctx.project
    if p.ref is None:
        return None
    x0, y0, x1, y1 = (int(v) for v in area)
    rev = p.state.get("target_rev")            # a changed target (easel recolor ...) gets views of its own
    path = p.root / "views" / (f"ref-{x0}-{y0}-{x1}-{y1}" + (f"-t{rev}" if rev else "") + ".png")
    if not path.exists():
        save_png(path, view.render([view.Panel(p.ref, "REFERENCE")], (x0, y0, x1, y1)))
    return p.rel(path)


def _show(ctx: Ctx, rep: "Report", panel: "view.Panel", area, tag: str, note: str, with_ref: bool = True,
          max_side: int = view.VIEW_SIDE) -> None:
    """Write one panel as ``VIEW:`` and, with a reference, the reference over the same area as ``VIEW-REF:``
    (two separate pictures compare better than one composite for the test model)."""
    p = ctx.project
    rep.view = _save_view(ctx, view.render([panel], area, max_side=max_side), tag)
    rep.view_note = f"{note}, area {_area_note(p, area)}"
    if with_ref:
        rep.view_ref = _ref_view(ctx, area)


def _region_map_panel(p: Project, title: str, highlight: Optional[str] = None) -> "view.Panel":
    base = p.ref if p.ref is not None else p.composite_u8("view")
    regs = [(nm, p.region_mask(nm), int(info.get("focus", 1))) for nm, info in p.state["regions"].items()]
    return view.region_map(base, regs, title, highlight)


class _ScaledComparison:
    """A comparison made at the delivered size, used with working-canvas masks and boxes."""

    def __init__(self, cmp, p: Project):
        self.c, self.p, self.s = cmp, p, p.scale

    def _m(self, mask):
        return self.p.mask_to_deliver(mask)

    def scores(self, mask=None, *a, **k):
        return self.c.scores(self._m(mask), *a, **k)

    def worst(self, mask=None, *a, **k):
        out = self.c.worst(self._m(mask), *a, **k)
        for o in out:
            if "box_px" in o:
                o["box_px"] = tuple(int(round(v * self.s)) for v in o["box_px"])
        return out

    def diagnose(self, box_px, *a, mask=None, **k):
        b = tuple(int(round(v / self.s)) for v in box_px)
        return self.c.diagnose(b, *a, mask=self._m(mask), **k)

    def __getattr__(self, name):
        return getattr(self.c, name)


def _scored(p: Project, img: np.ndarray) -> np.ndarray:
    """The picture the scores judge: as delivered, and from the detail stage on with the finishing sharpen the
    picture will be delivered with (when the session plans it and it was not turned off), so a level-1 step is
    scored on what the export gives, not on the plain area reduction that blurs its fine strokes."""
    st = p.state
    if st.get("deliver_filter") != "sharpen" and st.get("finish_sharpen") and not st.get("sharpen_off")             and st.get("stage") in ("detail", "finish") and p.ref is not None:
        try:
            from .pipeline import finish_sharpen
        except Exception:  # noqa: BLE001
            return delivered(p, img)
        return finish_sharpen(p.to_deliver(img), p.ref_deliver)
    return delivered(p, img)


def _comparison(p: Project, canvas_u8: np.ndarray):
    if p.ref is None:
        return None
    if p.scale > 1 or p.state.get("deliver_filter") == "sharpen":
        cmp = metrics.measure(p.ref_deliver, _scored(p, canvas_u8), paper=p.state.get("paper"),
                              use_lines=workflow.uses_lines(p.state))
        return _ScaledComparison(cmp, p)
    return metrics.measure(p.ref, canvas_u8, paper=p.state.get("paper"),
                           use_lines=workflow.uses_lines(p.state))


def _focus_spots(spots: List[dict], emphasis: Optional[str]) -> List[dict]:
    """Keep the spots of the kind the current stage fixes (lines in the line stage ...)."""
    if emphasis not in ("lines", "colour", "structure", "detail"):
        return spots
    return [s for s in spots if s.get("kind") == emphasis]


def _spots_lines(spots: List[dict], n: int = 3, emphasis: Optional[str] = None) -> List[str]:
    if not spots:
        if emphasis in ("lines", "colour", "structure", "detail"):
            return [f"worst spots for {emphasis}: none above the noise level (all problems:  easel look --show diff)"]
        return ["worst spots: none above the noise level"]
    out = [f"worst spots{' for ' + emphasis if emphasis in ('lines', 'colour', 'structure', 'detail') else ''}"
           " (box x0,y0,x1,y1 -> problem):"]
    for i, s in enumerate(spots[:n], 1):
        out.append(f"  {i}) {coords.fmt_box(s['box'])}  {s['problem']}")
    return out


def _panel_box(p: Project, box_px) -> Tuple[int, int, int, int]:
    return coords.pad_box_px(box_px, p.W, p.H, 0.05, min_side=max(24, int(0.04 * max(p.W, p.H))))


def _area_note(p: Project, box_px) -> str:
    return coords.fmt_box(coords.px_box_to_units(box_px, p.W, p.H))


def _region_outline(region: Region):
    if region.name == "all" or not region.named:
        return []
    return [(region.mask, "", (0, 200, 255))]


def _finish_next(ctx: Ctx, comparison=None) -> str:
    why, cmd = workflow.next_action(ctx.project, comparison)
    return f"{cmd}   ({why})"


# ============================================================================================ painting steps

LAYER_FOR = {"sketch": "sketch", "ink": "line", "flat": "flat", "shade": "shade", "paint": "paint",
             "detail": "detail"}
KEY_METRIC = {"line": "lines", "flat": "colour", "shade": "colour", "block-in": "colour", "forms": "overall",
              "detail": "overall"}


def default_layer(p: Project, tool: str) -> str:
    stage = p.state["stage"]
    lay = workflow.STAGES.get(stage, {}).get("layer") or "detail"
    if tool in ("fill", "gradient", "shape") and lay in ("sketch", "line"):
        lay = "flat"
    return lay


def _sketch_panel(p: Project) -> "view.Panel":
    faded = (p.ref.astype(np.float32) * 0.25 + 255 * 0.75)
    sk = p.layer("sketch")
    a = np.clip(sk[..., 3:4] * 1.5, 0, 1)
    ov = faded * (1 - a) + np.array([20, 70, 220], np.float32) * a
    return view.Panel(np.clip(ov, 0, 255).astype(np.uint8), "SKETCH (blue) OVER REFERENCE")


def paint_step(ctx: Ctx, *, label: str, layer: str, build: Callable[[], list], clip: Optional[np.ndarray],
               area_px, region: Optional[Region], tick: Optional[Tuple[str, str]] = None,
               pixel_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None, extra_lines: List[str] = (),
               sketch_view: bool = False, key_metric: Optional[str] = "overall",
               stage_tool: bool = False, extra_files: Tuple[str, ...] = (),
               on_apply: Optional[Callable[[list], None]] = None) -> Report:
    """Apply ops (or a pixel filter) to a layer as one undoable step and report the measured effect."""
    p = ctx.project
    rep = Report(mutated=True)
    entry_layer = p.layer_entry(layer)
    if not entry_layer.get("visible", True):
        ctx.notes.append(f"layer {layer} is hidden; show it with  easel layer {layer} --show")
    before_score = p.composite_u8("score")
    cmp_b = _comparison(p, before_score)
    mask = region.mask if region is not None else None
    s_b = cmp_b.scores(mask) if cmp_b else {}
    t0 = time.time()
    eng = None
    if pixel_fn is None:
        eng = manual.engine()
        ops = build()
        manual.check_marks(eng, ops, stage_tool=stage_tool)
    else:
        ops = []
    t_build = time.time() - t0
    step = p.begin_step([f"layers/{layer}.png", *extra_files])
    try:
        arr = p.layer(layer).copy()
        if pixel_fn is not None:
            arr = pixel_fn(arr)
            bbox = area_px
        else:
            bbox = eng.apply_ops(arr, ops, clip=None if clip is None else clip.astype(np.float32),
                                 alpha_lock=bool(entry_layer.get("alpha_lock")))
        p._write_layer_file(layer, arr)
        ops_file = p.save_ops(step, {"cmd": ctx.cmdline, "layer": layer,
                                     "ops": manual.ops_jsonable(eng, ops) if eng else [],
                                     "filter": None if pixel_fn is None else label})
        if on_apply is not None:
            on_apply(ops)
        stage_ticked = []
        if tick:
            stage_ticked = workflow.tick(p.state, tick[0], tick[1], note=f"step {step}")
            key = f"{tick[0]}:{tick[1]}"
            p.state.setdefault("attempts", {})[key] = workflow.tries(p, key) + 1
            workflow.save_tries(p, key, p.state["attempts"][key])
        after_view = p.composite_u8("view")
        after_score = p.composite_u8("score")
        cmp_a = _comparison(p, after_score)
        s_a = cmp_a.scores(mask) if cmp_a else {}
        spots = _focus_spots(cmp_a.worst(mask, n=5, emphasis=key_metric), key_metric)[:3] if cmp_a else []
        changed = bbox if bbox else None
        touched = changed or area_px
        units_box = coords.px_box_to_units(touched, p.W, p.H) if touched else None
        p.end_step(step, {"cmd": ctx.cmdline, "layer": layer, "ops": ops_file, "n_ops": len(ops),
                          "region": region.name if region else None,
                          "bbox": list(units_box) if units_box else None, "before": s_b, "after": s_a})
    except Exception:
        p.abort_step(step)
        raise
    rep.touched = touched
    rep.step = step
    what = f"{len(ops)} ops" if pixel_fn is None else "pixel filter"
    rep.add(f"step {step}  {label}  (layer {layer})  {what}  box {coords.fmt_box(units_box) if units_box else '-'}"
            + (f"  [{t_build:.1f}s]" if t_build > 3 else ""))
    if not changed and pixel_fn is None:
        rep.add("nothing changed on the layer (empty area, alpha lock, or clip outside the region)")
    if s_a:
        name = region.label if region else "area"
        if sketch_view:
            rep.add(f"{name}: the sketch layer is not scored; compare it with the reference in VIEW")
        else:
            rep.add(f"{name}:  " + metrics.fmt_delta(s_b, s_a))
            km = key_metric
            keys = tuple(dict.fromkeys(k for k in (km, "overall") if k and s_a.get(k) is not None))
            rep.summary = metrics.fmt_delta(s_b, s_a, keys=keys) if keys else ""
            if s_b.get(km) is not None and s_a.get(km) is not None and s_a[km] < s_b[km] - 2:
                rep.add(f"WARNING: {km} fell {s_b[km]} -> {s_a[km]}; undo it with  easel undo")
                rep.worse = f"{km} {s_b[km]} -> {s_a[km]}"
        for ln in extra_lines:
            rep.add(ln)
        if not sketch_view:
            rep.lines.extend(_spots_lines(spots, emphasis=key_metric))
    if stage_ticked:
        rep.add(f"todo ticked: {', '.join(str(i) for i in stage_ticked)}")
    if not ctx.batch:
        if region is not None and region.name == "all":
            area = (0, 0, p.W, p.H)
        elif region is not None and region.named:
            area = _panel_box(p, coords.union_box(region.box_px, touched))
        else:
            area = _panel_box(p, touched or (region.box_px if region else (0, 0, p.W, p.H)))
        if sketch_view and p.ref is not None:
            _show(ctx, rep, _sketch_panel(p), area, f"step-{step:04d}",
                  "your sketch in blue over the faded reference", with_ref=False)
        else:
            after_p = view.Panel(after_view, f"CANVAS AFTER STEP {step}")
            after_p.spots = [(s["box_px"], str(i)) for i, s in enumerate(spots[:3], 1)]
            _show(ctx, rep, after_p, area, f"step-{step:04d}",
                  "the canvas after this step" + (", worst spots boxed" if after_p.spots else ""))
    rep.next = _finish_next(ctx, None)
    if s_a and key_metric and s_b.get(key_metric) is not None and s_a.get(key_metric) is not None \
            and s_a[key_metric] < s_b[key_metric] - 2:
        rep.next = f"easel undo   (the step made {key_metric} worse), then try other settings"
    return rep


# ============================================================================================ commands

USAGE = {
    "new": "easel new my_picture --ref reference.png",
    "style": "easel style cel",
    "status": "easel status",
    "next": "easel next",
    "todo": 'easel todo --all',
    "stage": "easel stage done",
    "undo": "easel undo",
    "redo": "easel redo",
    "look": "easel look hair --show diff",
    "marks": "easel marks --n 40",
    "pick": "easel pick 500,400",
    "palette": "easel palette hair --k 6",
    "score": "easel score hair",
    "region": "easel region add hair --marks 3,4,9 --focus 2",
    "sketch": "easel sketch hair",
    "ink": "easel ink hair --detail 0.6",
    "flat": "easel flat hair",
    "shade": "easel shade hair --levels 3",
    "paint": "easel paint sky --size 30",
    "airbrush": "easel airbrush sky --sizes 64,32,16,8",
    "retouch": "easel retouch face",
    "dot": "easel dot all",
    "sharpen": "easel sharpen",
    "detail": "easel detail face",
    "stroke": 'easel stroke "300,400 350,380 420,390" --size 4 --color "#202020"',
    "fill": "easel fill 500,400 --color pick",
    "shape": 'easel shape ellipse 400,300,600,450 --color "#ffcc00"',
    "gradient": "easel gradient sky --from 500,0 --to 500,400 --colors pick",
    "erase": "easel erase 400,300,460,360 --layer line",
    "smooth": "easel smooth face --amount 0.5",
    "adjust": "easel adjust hair --brightness 0.1",
    "layer": "easel layer shade --mode multiply --opacity 0.8",
    "export": "easel export final.png --psd --timelapse film.gif --stages stages",
    "do": 'easel do "ink hair; ink face; look"',
    "fx": "easel fx grain --amount 0.4",
    "text": 'easel text "TITLE" --box 100,50,900,150 --color "#ffffff"',
    "fonts": "easel fonts serif",
    "help": "easel help",
}


def cmd_help(ctx: Ctx, args: List[str]) -> Report:
    rep = Report()
    rep.add("easel: a paint program. Coordinates are 0-1000 on each axis (x right, y down).")
    rep.add("after each command: read the report, open VIEW (the canvas) and, to compare, VIEW-REF (the reference)")
    rep.add("setup:   new  style  status  next  todo [--all]  stage  undo [n]  redo")
    rep.add("look:    look [region|box] [--show both|canvas|ref|diff|marks|layers] [--grid]   marks   pick   "
            "palette   score")
    rep.add("regions: region add <name> --marks 3,4 | --box x0,y0,x1,y1 | --poly \"x,y ...\" | --wand x,y  "
            "[--focus 1-3]; region rest background; region focus <name> 3; region list; region remove <name>")
    rep.add("stages:  sketch  ink  flat  shade  paint [--size N]  airbrush  detail [--level 2|1]  dot  retouch <region>   "
            "(each takes a region, default all; no options = the engine's own settings for the style)")
    rep.add("hand:    stroke  fill  shape  gradient  erase  smooth  adjust   (act on the stage's layer or --layer)")
    rep.add("finish:  sharpen (the finishing sharpen, fitted to the reference)   "
            "fx <glow|grain|blur|vignette|paper|sharpen|levels> [region] [--amount 0..1]   "
            "text \"<string>\" --box x0,y0,x1,y1 [--font --color --align --outline px,#hex --vertical --fit]   fonts")
    rep.add("other:   layer  export <file.png> [--psd] [--ora] [--timelapse film.gif|film.mp4] [--stages dir]   "
            "do \"<cmd>; <cmd>\"")
    if args:
        c = args[0].lower()
        if c in USAGE:
            rep.add(f"example: {USAGE[c]}")
        if c in ("region", "regions"):
            rep.add("region add <name> --marks 3,4 | --wand x,y | --box x0,y0,x1,y1 | --poly \"x,y ...\"  "
                    "[--plus <region|m12|12>] [--minus <region|m12|12>] [--focus 1-3] [--no-refine] [--tol 12]")
    rep.next = "easel next"
    return rep


# ---------------------------------------------------------------------------------------- new / style

def cmd_new(ctx: Ctx, args: List[str]) -> Report:
    sp = Spec(USAGE["new"], ["dir"], {"ref": "value", "size": "value", "request": "value", "n": "value",
                                      "mode": "value", "scale": "value", "fine": "value"},
              alias={"reference": "ref"}, min_pos=1)
    a = parse(args, sp)
    mode = str(a.get("mode") or "").lower()
    if mode and mode not in ("hand", "trace", "free"):
        raise UsageError(f"unknown --mode {mode!r}: hand (you draw every mark over the reference), trace (stage "
                         "commands follow the reference) or free (no reference)", "easel new pic --ref ref.jpg --mode hand")
    if mode in ("hand", "trace") and not a.get("ref"):
        raise UsageError(f"--mode {mode} draws over a reference: add --ref", f"easel new pic --ref ref.jpg --mode {mode}")
    if mode == "free" and a.get("ref"):
        raise UsageError("--mode free is for a picture without a reference: drop --ref or use --mode hand",
                         "easel new pic --mode free")
    size = None
    if a.get("size"):
        nums = [int(float(x)) for x in re.findall(r"\d+(?:\.\d+)?", str(a["size"]))]
        if len(nums) != 2:
            raise UsageError("--size needs WxH like 1024x768", "easel new my_picture --size 1024x768")
        size = (max(16, nums[0]), max(16, nums[1]))
    p = Project.create(a["dir"], a.get("ref"), size, a.get("request") or "", notes=ctx.notes)
    ctx._project = p
    # with a reference the picture decides the style (the request may ask for changes: easel change ...)
    style, why = workflow.guess_style(p.ref, (a.get("request") or "") if p.ref is None else "")
    p.state["style"] = style
    if p.ref is not None:
        try:
            from .pipeline import soft_ramp_share
            p.state["soft_ramp"] = round(float(soft_ramp_share(p.ref)), 3)
        except Exception:  # noqa: BLE001
            pass
    p.state["style_guess"] = {"style": style, "why": why}
    # the working canvas: marks are laid at scale x the delivered picture (free pictures 3x, so the finest strokes
    # survive the reduction; reference projects 1x); the 1600 px limit is on the delivered picture
    if a.get("scale"):
        scale = int(coords.parse_int(a["scale"], "scale", 1, 4, ctx.notes, "easel new pic --scale 2"))
    elif p.ref is None and style != "poster":
        from .draw_canvas import wanted_scale
        scale = wanted_scale(p.W, p.H)
    elif p.ref is not None and mode != "hand" and _streaks_available():
        scale = _auto_scale(p.W, p.H)   # trace: 4x up to a 400 px long side, 3x up to 700, else 2x
    else:
        scale = 1
    fine = str(a.get("fine") or ("streaks" if (p.ref is not None and scale > 1 and mode != "hand"
                                               and _streaks_available()) else "plan")).lower()
    if fine not in ("streaks", "plan"):
        raise UsageError(f"unknown --fine {fine!r}: streaks (thin strokes whose colour and width follow the "
                         "reference) or plan (liner passes and hairline runs)", "easel new pic --ref r.jpg --fine plan")
    p.state["fine"] = fine
    if p.ref is not None and mode != "hand" and fine == "streaks" and scale > 1 and dots_available():
        # round 9: dots after level 1 (texture points only), the finishing sharpen at delivery
        p.state["stipple"] = STIPPLE_ROUNDS
        p.state["stipple_edges"] = False
        p.state["finish_sharpen"] = True
        if contours_available():
            # round 10: thin lines as one stroke each before the detail streaks; smooth areas to a finer tolerance
            p.state["contours"] = True
            p.state["smooth_tol"] = SMOOTH_TOL
    p.state["deliver_filter"] = "area"
    p.set_work_scale(scale)
    local_line = None
    if p.ref is not None:
        try:
            from .stage_local import char_map
            sh = char_map(p.ref).shares
            local_line = ("local map of the reference: " + ", ".join(f"{k} {100 * float(sh.get(k, 0)):.0f}%"
                                                                      for k in ("flat", "ramp", "texture"))
                          + "  (crisp shapes on flat areas, airbrush on soft ramps, texture and grain only where "
                            "the reference has them)")
            p.state["local_map"] = {k: round(float(sh.get(k, 0)), 3) for k in ("flat", "ramp", "texture")}
        except Exception:  # noqa: BLE001
            local_line = None
    if mode == "hand":
        from . import draw_core, draw_ref, workflow_draw
        p.state["mode"] = "hand"
        draw_core.dstate(p)
        draw_ref._hs(p)
        workflow_draw.build_todo(p.state)
    else:
        workflow.build_todo(p.state)
    p.save_state()
    rep = Report(mutated=True)
    dw, dh = p.deliver_size()
    rep.add(f"project {p.rel(p.root)}  canvas {p.W}x{p.H}  reference {'yes' if p.ref is not None else 'none'}"
            + (f"  (working canvas {p.state['work_scale']}x the {dw}x{dh} picture you export; coordinates stay "
               "0-1000)" if p.state.get("work_scale", 1) > 1 else ""))
    if local_line:
        rep.add(local_line)
    rep.add(f"style guess: {style} ({workflow.STYLE_WORDS[style]})")
    rep.add(f"  because: {why}")
    rep.add(f"  other styles: {', '.join(s for s in workflow.STYLES if s != style)}  (change with  easel style <name>)")
    if mode == "hand":
        from . import workflow_draw
        rep.add("mode hand: you draw every mark; the reference is there to look at, snap to (--snap ref) and take "
                "colours from (pick@x,y)")
        rep.add("stages: " + " > ".join(workflow_draw.plan_of(p.state)))
    else:
        rep.add("stages: " + " > ".join(workflow.PLANS[style]))
    if p.ref is not None:
        n = int(coords.parse_float(a["n"], "n", 12, 60)) if a.get("n") else 40
        mrep = _make_marks(ctx, n)
        rep.lines.extend(mrep.lines)
        rep.view, rep.view_note, rep.view_ref = mrep.view, mrep.view_note, mrep.view_ref
    rep.next = _finish_next(ctx)
    return rep


def cmd_style(ctx: Ctx, args: List[str]) -> Report:
    a = parse(args, Spec(USAGE["style"], ["name"]))
    p = ctx.project
    rep = Report()
    if not a.get("name"):
        g = p.state.get("style_guess") or {}
        rep.add(f"style: {p.state['style']}  ({workflow.STYLE_WORDS[p.state['style']]})")
        if g:
            rep.add(f"tool's guess: {g.get('style')} because {g.get('why')}")
        rep.add("styles: " + "; ".join(workflow.STYLE_WORDS[s] for s in workflow.STYLES))
        rep.next = _finish_next(ctx)
        return rep
    name = str(a["name"]).strip().lower()
    alias = {"anime": "cel", "manga": "ink", "painting": "paint", "painterly": "paint", "photograph": "photo",
             "flat": "poster", "line": "ink"}
    name = alias.get(name, name)
    if name not in workflow.STYLES:
        import difflib
        c = difflib.get_close_matches(name, workflow.STYLES, n=1)
        raise UsageError(f"unknown style {a['name']!r}; styles: {', '.join(workflow.STYLES)}",
                         f"easel style {c[0] if c else 'cel'}")
    step = p.begin_step([])
    old = p.state["style"]
    p.state["style"] = name
    plan = workflow.plan_of(p.state)          # the style's stages (with a change stage when changes are stated)
    if p.state["stage"] not in plan:
        ctx.notes.append(f"stage {p.state['stage']} is not in the {name} plan; back to sketch")
        p.state["stage"] = "sketch"
    p.state["stages_done"] = [s for s in p.state["stages_done"] if s in plan]
    workflow.build_todo(p.state)
    p.end_step(step, {"cmd": ctx.cmdline, "style": [old, name]})
    rep.mutated = True
    rep.add(f"step {step}  style {old} -> {name}  ({workflow.STYLE_WORDS[name]})")
    rep.add("stages: " + " > ".join(f"[{s}]" if s == p.state["stage"] else s for s in plan))
    rep.next = _finish_next(ctx)
    return rep


# ---------------------------------------------------------------------------------------- status / next / todo

def cmd_status(ctx: Ctx, args: List[str]) -> Report:
    parse(args, Spec(USAGE["status"]))
    p = ctx.project
    st = p.state
    rep = Report()
    plan = workflow.plan_of(st)
    rep.add(f"project {p.rel(p.root)}  {p.W}x{p.H}  style {st['style']}  step {st['step']}"
            + ("  FINISHED" if st.get("finished") else ""))
    rep.add("stages: " + " > ".join(f"[{s}]" if s == st["stage"] else (s + " (done)" if s in st["stages_done"]
                                                                          else s) for s in plan))
    stage = st["stage"]
    items = [t for t in st["todo"] if t["stage"] == stage]
    marks = {"open": "[ ]", "done": "[x]", "skipped": "[-]"}
    line = "todo now: " + "  ".join(f"{marks[t['status']]} {t['id']} {t['text'] if t['kind'] != 'gate' else 'gate'}"
                                    for t in items)
    rep.add(line if len(line) < 400 else line[:397] + "...")
    total = [t for t in st["todo"] if t["kind"] != "gate"]
    rep.add(f"todo overall: {sum(t['status'] != 'open' for t in total)}/{len(total)} done")
    rep.add(fingerprint_line(p))
    cmp = None
    if p.ref is not None:
        cmp = _comparison(p, p.composite_u8("score"))
        s_all = cmp.scores()
        rep.add("whole picture:  " + metrics.fmt_scores(s_all))
        names = list(st["regions"].keys())
        if names:
            rep.add("regions (focus, overall, weakest):")
            rows = []
            for n in names:
                s = cmp.scores(p.region_mask(n))
                axes = {k: v for k, v in s.items() if k != "overall" and v is not None}
                weak = min(axes, key=axes.get) if axes else "-"
                rows.append(f"{n} f{st['regions'][n].get('focus', 1)} {s['overall']} ({weak} {axes.get(weak, '-')})")
            for i in range(0, len(rows), 3):
                rep.add("  " + "   |   ".join(rows[i:i + 3]))
        else:
            rep.add("regions: none yet")
    ok, unmet = workflow.check_gate(p, stage, cmp)
    if ok:
        rep.add(f"gate {stage}: passes ({workflow.gate_words(stage, p.state.get('style'))})")
    else:
        u = unmet[:3]
        rep.add(f"gate {stage}: {len(unmet)} short: " + "; ".join(
            _unmet_words(x) for x in u))
    jl = [j for j in p.journal_lines(6) if "cmd" in j or "undo" in j or "redo" in j][-5:]
    if jl:
        rep.add("last steps:")
        for j in jl:
            if "undo" in j or "redo" in j:
                rep.add(f"  {'undo' if 'undo' in j else 'redo'} of step "
                        f"{', '.join(map(str, j.get('undo') or j.get('redo') or []))}")
                continue
            r = j.get("region")
            b, a2 = j.get("before") or {}, j.get("after") or {}
            delta = f"  overall {b.get('overall', '-')} -> {a2.get('overall', '-')}" if b or a2 else ""
            rep.add(f"  {j.get('step', '')} {_short_cmd(j['cmd'])}{delta}")
    rep.next = _finish_next(ctx, cmp)
    return rep


def _short_cmd(c: str, n: int = 60) -> str:
    c = c.replace("easel ", "", 1)
    return c if len(c) <= n else c[: n - 3] + "..."


def cmd_next(ctx: Ctx, args: List[str]) -> Report:
    parse(args, Spec(USAGE["next"]))
    p = ctx.project
    why, cmd = workflow.next_action(p)
    rep = Report()
    rep.add(f"stage {p.state['stage']}: {why}")
    if "<" in cmd:
        rep.add("fill in the <...> parts; e.g.  easel region add face --marks 12,13 --focus 3")
    rep.next = cmd
    return rep


def cmd_todo(ctx: Ctx, args: List[str]) -> Report:
    p = ctx.project
    st = p.state
    rep = Report()
    sub = args[0].lower() if args else "list"
    if sub in ("--rebuild", "rebuild"):
        step = p.begin_step([])
        workflow.build_todo(st)
        p.end_step(step, {"cmd": ctx.cmdline})
        rep.mutated = True
        rep.add(f"step {step}  todo rebuilt for the style, stages and regions: {len(st['todo'])} items "
                "(done items stay done)")
        rep.next = _finish_next(ctx)
        return rep
    if sub in ("--all", "all", "plan", "--plan", "full"):
        rep.add("the whole plan, one line per stage; copy every line into your todo tool now (TodoWrite / "
                "todowrite), work through them in order, mark a line completed when  easel stage done  closes it:")
        for ln in workflow.plan_lines(st):
            rep.add(ln)
        rep.add(f"easel's own per-region items: {sum(t['status'] != 'open' for t in st['todo'])}/{len(st['todo'])} "
                "done (easel todo shows the current stage)")
        rep.next = _finish_next(ctx)
        return rep
    if sub in ("list", "ls", "show"):
        stage_filter = None
        rows = st["todo"]
        cur = st["stage"]
        marks = {"open": "[ ]", "done": "[x]", "skipped": "[-]"}
        for stg in workflow.plan_of(st):
            items = [t for t in rows if t["stage"] == stg]
            if stg != cur and all(t["status"] != "open" for t in items):
                rep.add(f"{stg}: all {len(items)} done")
                continue
            if stg != cur:
                rep.add(f"{stg}: {sum(t['status'] != 'open' for t in items)}/{len(items)} done")
                continue
            rep.add(f"{stg} (current):")
            for t in items:
                rep.add(f"  {marks[t['status']]} {t['id']}  {t['text']}" + (f"   -> {t['cmd']}" if t['status'] == 'open'
                                                                          and t['kind'] != 'gate' else "")
                        + (f"  ({t['note']})" if t.get("note") else ""))
        rep.next = _finish_next(ctx)
        return rep
    if sub == "add":
        if len(args) < 2:
            raise UsageError("todo add needs the text", 'easel todo add "darken the hair shadow"')
        step = p.begin_step([])
        st["todo"].append({"stage": st["stage"], "region": None, "kind": "user", "key": None,
                           "text": " ".join(args[1:]), "cmd": "", "status": "open", "note": ""})
        for i, t in enumerate(st["todo"], 1):
            t["id"] = i
        p.end_step(step, {"cmd": ctx.cmdline})
        rep.mutated = True
        rep.add(f"todo {len(st['todo'])} added to stage {st['stage']}: {' '.join(args[1:])}")
        rep.next = _finish_next(ctx)
        return rep
    if sub in ("done", "skip", "tick", "check"):
        if len(args) < 2 or not re.fullmatch(r"#?\d+", args[1]):
            raise UsageError(f"todo {sub} needs the item number", f'easel todo {sub} 3 "reason"')
        n = int(args[1].lstrip("#"))
        item = next((t for t in st["todo"] if t["id"] == n), None)
        if item is None:
            raise UsageError(f"there is no todo item {n} (1-{len(st['todo'])})", "easel todo")
        note = " ".join(args[2:])
        if sub == "skip" and not note:
            raise UsageError("todo skip needs a reason", f'easel todo skip {n} "the reference has nothing there"')
        if item["kind"] == "gate":
            raise UsageError("gates close with  easel stage done", "easel stage done")
        step = p.begin_step([])
        item["status"] = "skipped" if sub == "skip" else "done"
        item["note"] = note
        p.end_step(step, {"cmd": ctx.cmdline})
        rep.mutated = True
        rep.add(f"todo {n} {item['status']}: {item['text']}" + (f" ({note})" if note else ""))
        rep.next = _finish_next(ctx)
        return rep
    raise UsageError(f"unknown todo action {sub!r}; use --all, list, add, done, skip or --rebuild", USAGE["todo"])


# ---------------------------------------------------------------------------------------- stage

def cmd_stage(ctx: Ctx, args: List[str]) -> Report:
    sp = Spec(USAGE["stage"], ["action", "name"], {"because": "value"}, alias={"reason": "because"})
    a = parse(args, sp)
    p = ctx.project
    st = p.state
    rep = Report()
    action = str(a.get("action") or "show").lower()
    stage = st["stage"]
    if action in ("show", "gate", "status"):
        ok, unmet = workflow.check_gate(p, stage)
        rep.add(f"stage {stage}: {workflow.STAGES[stage]['what']}")
        rep.add(f"gate: {workflow.gate_words(stage, p.state.get('style'))}")
        if ok:
            rep.add("gate passes")
        for u in unmet[:8]:
            rep.add(f"  short: {_unmet_words(u)}"
                    f"   raise it:  {u['fix']}")
        rep.next = "easel stage done" if ok else unmet[0]["fix"]
        return rep
    if action in ("done", "close", "next", "finish"):
        ok, unmet = workflow.check_gate(p, stage)
        because = a.get("because")
        if not ok and not because:
            rep.add(f"stage {stage} cannot close yet: {len(unmet)} measurement(s) short")
            for u in unmet[:8]:
                rep.add(f"  {_unmet_words(u)}"
                        f"   raise it:  {u['fix']}")
            if len(unmet) > 8:
                rep.add(f"  ... {len(unmet) - 8} more")
            rep.add(f'to close anyway give a reason:  easel stage done --because "<why it is good enough>"')
            rep.next = unmet[0]["fix"]
            ctx.exit_code = 1  # type: ignore[attr-defined]
            return rep
        step = p.begin_step(["layers/sketch.png"])
        new = workflow.advance(st)
        if because:
            st.setdefault("overrides", []).append({"stage": stage, "because": because})
        p.end_step(step, {"cmd": ctx.cmdline, "stage": [stage, new], "because": because})
        rep.mutated = True
        if new is None:
            rep.add(f"step {step}  stage {stage} closed; the picture is finished")
        else:
            rep.add(f"step {step}  stage {stage} closed" + (" (gate overridden: " + because + ")" if because and not ok
                                                            else "") + f"; now stage {new}: "
                    f"{workflow.STAGES[new]['what']}")
            rep.add(f"gate of {new}: {workflow.gate_words(new, st.get('style'))}")
            if new in ("flat", "block-in") and "sketch" in st["stages_done"]:
                rep.add("the sketch layer is now hidden (it stays in the project)")
        plan = workflow.plan_of(st)
        k = plan.index(stage) + 1 if stage in plan else 0
        if stage == "plan":
            rep.add("your todo tool: now run  easel todo --all  and copy its lines into your todo tool")
        elif k:
            rep.add(f"your todo tool: mark line {k} ({stage}) completed"
                    + (f" and line {k + 1} ({new}) in progress" if new else ""))
        rep.next = "easel todo --all" if stage == "plan" else _finish_next(ctx)
        return rep
    if action in ("back", "reopen"):
        name = a.get("name")
        plan = workflow.plan_of(st)
        if not name:
            raise UsageError("stage back needs the stage name", f"easel stage back {plan[1]}")
        name = str(name).lower().replace("blockin", "block-in").replace("block_in", "block-in")
        if name not in plan:
            raise UsageError(f"{name!r} is not a stage of the {st['style']} plan: {', '.join(plan)}",
                             f"easel stage back {plan[1]}")
        step = p.begin_step([])
        workflow.go_back(st, name)
        p.end_step(step, {"cmd": ctx.cmdline, "stage": [stage, name]})
        rep.mutated = True
        rep.add(f"step {step}  back to stage {name}: {workflow.STAGES[name]['what']}")
        rep.next = _finish_next(ctx)
        return rep
    raise UsageError(f"unknown stage action {action!r}; use show, done, back <name>", USAGE["stage"])


# ---------------------------------------------------------------------------------------- undo / redo

def _undo_redo(ctx: Ctx, args: List[str], redo: bool) -> Report:
    a = parse(args, Spec(USAGE["redo" if redo else "undo"], ["n"], {"to": "value"}))
    p = ctx.project
    if a.get("to") and not redo:
        to = coords.parse_int(a["to"], "to", 0, 100000)
        n = int(p.state["step"]) - to
        if n < 1:
            raise UsageError(f"the project is at step {p.state['step']}; --to needs an earlier step", "easel undo --to 12")
    else:
        n = coords.parse_int(a["n"], "n", 1, 50) if a.get("n") else 1
    steps = p.redo(n) if redo else p.undo(n)
    rep = Report(mutated=bool(steps))
    word = "redo" if redo else "undo"
    if not steps:
        rep.add(f"nothing to {word}")
        rep.next = _finish_next(ctx)
        return rep
    cmds = {j.get("step"): j.get("cmd") for j in p.journal_lines(400) if "step" in j}
    rep.add(f"{word} {len(steps)} step(s): back to step {p.state['step']}" if not redo else
            f"redo {len(steps)} step(s): now at step {p.state['step']}")
    for s in steps:
        rep.add(f"{word} step {s}: {_short_cmd(cmds.get(s, '?'))}")
    rep.add(f"now at step {p.state['step']}; {p.can_undo()} step(s) can be undone, {p.can_redo()} redone")
    if not ctx.batch:
        _show(ctx, rep, view.Panel(p.composite_u8("view"), f"CANVAS AT STEP {p.state['step']}"), (0, 0, p.W, p.H),
              f"{word}-{_view_counter(p):04d}", f"the canvas after the {word}")
    rep.next = _finish_next(ctx)
    return rep


def cmd_undo(ctx, args):
    return _undo_redo(ctx, args, False)


def cmd_redo(ctx, args):
    return _undo_redo(ctx, args, True)


# ---------------------------------------------------------------------------------------- look / marks / score

SHOWS = ("both", "canvas", "ref", "diff", "marks", "layers")


def cmd_look(ctx: Ctx, args: List[str]) -> Report:
    sp = Spec(USAGE["look"], ["where*"], {"show": "value", "layer": "value", "grid": "flag"},
              alias={"mode": "show", "view": "show"})
    a = parse(args, sp)
    p = ctx.project
    rep = Report()
    show = str(a.get("show") or "both").lower()
    show = {"reference": "ref", "compare": "both", "side": "both", "error": "diff", "heat": "diff",
            "regions": "marks", "layer": "layers"}.get(show, show)
    where = a.get("where")
    if where and str(where).lower() in SHOWS and not a.get("show"):
        show, where = str(where).lower(), None
    if show not in SHOWS:
        raise UsageError(f"unknown --show {show!r}; one of {', '.join(SHOWS)}", USAGE["look"])
    if a.get("layer") and show not in ("layers",):
        show = "layers"
    region = resolve_region(p, where, ctx.notes, "easel look hair")
    area = (0, 0, p.W, p.H) if region.name == "all" else _panel_box(p, region.box_px)
    canvas = p.composite_u8("view")
    cmp = _comparison(p, p.composite_u8("score")) if p.ref is not None else None
    emph = KEY_METRIC.get(p.state["stage"])
    spots = _focus_spots(cmp.worst(region.mask if region.name != "all" else None, n=5, emphasis=emph),
                         emph)[:3] if cmp else []
    panels: List[view.Panel] = []
    if show == "layers":
        names = [a["layer"]] if a.get("layer") else [e["name"] for e in p.state["layers"]]
        for nm in names:
            e = p.layer_entry(nm)
            arr = p.layer(e["name"], create=False)
            flags = []
            if e["mode"] != "normal":
                flags.append(e["mode"])
            if e["opacity"] < 1:
                flags.append(f"{int(e['opacity'] * 100)}%")
            if not e.get("visible", True):
                flags.append("hidden")
            if e.get("empty"):
                flags.append("empty")
            panels.append(view.Panel(view.layer_on_checker(arr, p.H, p.W),
                                     e["name"].upper() + (f" ({', '.join(flags)})" if flags else "")))
    elif show == "marks":
        if p.state["regions"]:
            panels = [_region_map_panel(p, "REGIONS (tinted, name f<focus>)")]
        else:
            if not p.state.get("marks"):
                return _make_marks(ctx, 40)
            panels = [_marks_panel(p, _marks_labels(p), p.state["marks"]["table"])]
    tag = f"look-{_view_counter(p):04d}"
    spot_list = [(s["box_px"], str(i)) for i, s in enumerate(spots, 1)]
    if panels:   # layers / marks / regions: one map picture
        img = view.render(panels, area, grid=bool(a.get("grid")), max_side=view.MAP_SIDE)
        rep.view = _save_view(ctx, img, tag)
        rep.view_note = " | ".join(pn.title.split(" (")[0].lower() for pn in panels) + f", area {_area_note(p, area)}"
    elif show == "ref" and p.ref is not None:
        rep.view = _ref_view(ctx, area)
        rep.view_note = f"the reference, area {_area_note(p, area)}"
    elif show == "diff" and cmp is not None:
        dp = view.Panel(view.heat_panel(canvas, cmp.error_map()), "DIFF (bright = wrong)")
        dp.spots = spot_list
        _show(ctx, rep, dp, area, tag, "where the canvas differs from the reference, bright = wrong")
    else:
        cp = view.Panel(canvas, f"CANVAS AT STEP {p.state['step']}")
        cp.spots = spot_list
        if a.get("grid"):
            rep.view = _save_view(ctx, view.render([cp], area, grid=True), tag)
            rep.view_note = f"the canvas with a grid, area {_area_note(p, area)}"
            rep.view_ref = _ref_view(ctx, area) if show == "both" else None
        else:
            _show(ctx, rep, cp, area, tag, "the canvas" + (", worst spots boxed" if spot_list else ""),
                  with_ref=(show == "both"))
    if p.state["stage"] == "finish" and region.name == "all" and show in ("diff", "both", "canvas"):
        if workflow.tick(p.state, "finish", None, key="look"):
            p.save_state()
    if p.state["stage"] in ("detail", "finish") and region.name in p.state["regions"] and \
            show in ("diff", "both", "canvas"):
        if workflow.tick(p.state, p.state["stage"], region.name, key="closeup"):
            p.save_state()
    rep.add(f"look {region.label}  show {show}  area {_area_note(p, area)}  stage {p.state['stage']}")
    if cmp is not None and show not in ("layers", "marks"):
        s = cmp.scores(region.mask if region.name != "all" else None)
        rep.add(f"{region.label}:  " + metrics.fmt_scores(s))
        rep.lines.extend(_spots_lines(spots, emphasis=emph))
    rep.next = _finish_next(ctx, cmp)
    return rep


def _marks_panel(p: Project, labels: np.ndarray, table: list) -> "view.Panel":
    pan = view.Panel(p.ref, "MARKS (numbers for  region add --marks)")
    pan.labels = labels
    pan.label_anchors = {r["n"]: tuple(r["anchor"]) for r in table}
    return pan


def _make_marks(ctx: Ctx, n: int) -> Report:
    p = ctx.project
    ref = p.require_ref("marks")
    labels, table = regions.compute_marks(ref, n)
    for r in table:
        r["anchor"] = list(regions.label_anchor(labels == r["n"]))
    rel = f"cache/marks_{n}.png"
    save_png(p.root / rel, labels.astype(np.uint8) if labels.max() < 256 else labels.astype(np.uint16))
    p.state["marks"] = {"n": n, "file": rel, "table": table}
    workflow.tick(p.state, "plan", None, key="marks")
    p.save_state()
    (p.root / "marks.txt").write_text(
        "\n".join(f"{r['n']}\t{coords.fmt_box(r['box'])}\t{r['colour']}\t{r['hex']}\t{r['share'] * 100:.1f}%"
                  for r in table) + "\n", encoding="utf-8")
    rep = Report()
    rep.add(f"marks: the reference cut into {len(table)} numbered areas "
            "(number colour share% box; full table in marks.txt)")
    cells = [f"{r['n']} {r['colour']} {r['share'] * 100:.0f}% {coords.fmt_box(r['box'])}" for r in table]
    per = 3 if len(cells) <= 45 else 4
    for i in range(0, len(cells), per):
        rep.add("  " + "  |  ".join(cells[i:i + per]))
    if not ctx.batch:
        img = view.render([_marks_panel(p, labels, table)], (0, 0, p.W, p.H), max_side=view.MAP_SIDE)
        rep.view = _save_view(ctx, img, f"marks-{_view_counter(p):04d}")
        rep.view_note = "the reference cut into numbered areas"
        rep.view_ref = _ref_view(ctx, (0, 0, p.W, p.H))
    rep.next = _finish_next(ctx)
    return rep


def cmd_marks(ctx: Ctx, args: List[str]) -> Report:
    a = parse(args, Spec(USAGE["marks"], [], {"n": "value"}))
    n = coords.parse_int(a["n"], "n", 12, 60, ctx.notes, USAGE["marks"]) if a.get("n") else 40
    return _make_marks(ctx, n)


def cmd_score(ctx: Ctx, args: List[str]) -> Report:
    a = parse(args, Spec(USAGE["score"], ["where*"]))
    p = ctx.project
    p.require_ref("score")
    cmp = _comparison(p, p.composite_u8("score"))
    rep = Report()
    if a.get("where"):
        r = resolve_region(p, a["where"], ctx.notes, "easel score hair")
        s = cmp.scores(r.mask)
        rep.add(f"{r.label}:  " + metrics.fmt_scores(s))
        if r.named and r.name in p.state["regions"]:
            rep.add(f"coverage {workflow.coverage(p, r.mask):.0f}%")
        rep.lines.extend(_spots_lines(cmp.worst(r.mask, n=5, emphasis=KEY_METRIC.get(p.state["stage"])), 5))
    else:
        rep.add("whole picture:  " + metrics.fmt_scores(cmp.scores()))
        rows = []
        for n in p.state["regions"]:
            s = cmp.scores(p.region_mask(n))
            rows.append((s["overall"] if s["overall"] is not None else 0, n, s))
        rows.sort()
        if rows:
            rep.add("regions, worst first:")
        for _, n, s in rows[:12]:
            rep.add(f"  {n} (f{p.state['regions'][n].get('focus', 1)}):  " + metrics.fmt_scores(s))
        rep.lines.extend(_spots_lines(cmp.worst(n=5, emphasis=KEY_METRIC.get(p.state["stage"])), 5))
    rep.next = _finish_next(ctx, cmp)
    return rep


def cmd_pick(ctx: Ctx, args: List[str]) -> Report:
    a = parse(args, Spec(USAGE["pick"], ["where*"], min_pos=1))
    p = ctx.project
    rep = Report()
    canvas = p.composite_u8("score")
    w = str(a["where"])
    if coords.looks_like_point(w):
        pt = coords.parse_point(w, ctx.notes, USAGE["pick"])
        px = coords.point_to_px(pt, p.W, p.H)
        where = f"at {coords.fmt_point(pt)}"
        rc = manual.pick_colour(p.ref, px) if p.ref is not None else None
        cc = manual.pick_colour(canvas, px)
    else:
        r = resolve_region(p, w, ctx.notes, "easel pick hair")
        where = f"in {r.label} (median)"
        rc = manual.pick_colour(p.ref, mask=r.mask) if p.ref is not None else None
        cc = manual.pick_colour(canvas, mask=r.mask)
    if rc is not None:
        rep.add(f"reference {where}: {coords.hex_colour(rc)} ({coords.colour_name(rc)})")
    rep.add(f"canvas    {where}: {coords.hex_colour(cc)} ({coords.colour_name(cc)})")
    rep.next = f"use it:  --color {coords.hex_colour(rc or cc)}   or  {_finish_next(ctx)}"
    return rep


def cmd_palette(ctx: Ctx, args: List[str]) -> Report:
    a = parse(args, Spec(USAGE["palette"], ["where*"], {"k": "value"}))
    p = ctx.project
    ref = p.require_ref("palette")
    r = resolve_region(p, a.get("where"), ctx.notes, "easel palette hair")
    k = coords.parse_int(a["k"], "k", 1, 16, ctx.notes) if a.get("k") else 8
    pal = manual.palette(ref, None if r.name == "all" else r.mask, k)
    rep = Report()
    rep.add(f"palette of the reference in {r.label} (biggest share first):")
    merged: Dict[str, list] = {}
    for c, sh in pal:
        h = coords.hex_colour(c)
        if h in merged:
            merged[h][1] += sh
        else:
            merged[h] = [c, sh]
    pal = [(c, sh) for c, sh in merged.values() if sh >= 0.005]
    for i, (c, sh) in enumerate(pal, 1):
        rep.add(f"  {i}. {coords.hex_colour(c)}  {coords.colour_name(c):<16} {sh * 100:4.0f}%")
    rep.next = _finish_next(ctx)
    return rep


# ---------------------------------------------------------------------------------------- regions

def cmd_region(ctx: Ctx, args: List[str]) -> Report:
    if not args:
        args = ["list"]
    sub = args[0].lower()
    rest = args[1:]
    if sub in ("list", "ls", "show"):
        return _region_list(ctx)
    if sub in ("add", "new", "set", "make", "define"):
        return _region_add(ctx, rest)
    if sub == "rest":
        return _region_rest(ctx, rest)
    if sub in ("focus", "weight"):
        return _region_focus(ctx, rest)
    if sub in ("remove", "rm", "delete", "del"):
        return _region_remove(ctx, rest)
    # "easel region hair --marks 3,4" (add without the verb)
    if rest and rest[0].startswith("--"):
        return _region_add(ctx, args)
    raise UsageError(f"unknown region action {sub!r}; use add, rest, focus, remove, list", USAGE["region"])


def _region_list(ctx: Ctx) -> Report:
    p = ctx.project
    rep = Report()
    regs = p.state["regions"]
    if not regs:
        rep.add("no regions yet")
    for n, r in regs.items():
        rep.add(f"  {n}: focus {r.get('focus', 1)}  {r['area'] * 100:.0f}% of canvas  box {coords.fmt_box(r['box'])}"
                f"  ({r.get('how', '')})")
    if regs:
        f = workflow.plan_facts(p)
        rep.add(f"covered {f['covered']:.0f}% of the canvas")
    rep.next = _finish_next(ctx)
    return rep


def _store_region(ctx: Ctx, name: str, mask: np.ndarray, how: str, focus: Optional[int], label_cmd: str) -> Report:
    p = ctx.project
    rep = Report(mutated=True)
    if not mask.any():
        raise UsageError(f"the region {name} came out empty; try a bigger box, other marks or a higher --tol",
                         "easel region add hair --box 200,50,700,450")
    existed = name in p.state["regions"]
    step = p.begin_step([f"regions/{name}.png"])
    p.save_region_mask(name, mask)
    box = coords.px_box_to_units(coords.mask_bbox_px(mask), p.W, p.H)
    prev_focus = p.state["regions"].get(name, {}).get("focus", 1)
    p.state["regions"][name] = {"focus": int(focus if focus is not None else prev_focus), "box": list(box),
                                "area": round(float(mask.mean()), 4), "how": how}
    workflow.build_todo(p.state)
    f = workflow.plan_facts(p)
    if f["n"] >= 3:
        workflow.tick(p.state, "plan", None, key="regions")
    if f["covered"] >= 90:
        workflow.tick(p.state, "plan", None, key="rest")
    p.end_step(step, {"cmd": ctx.cmdline, "region": name})
    overlaps = []
    for other in p.region_names():
        if other == name:
            continue
        ov = (p.region_mask(other) & mask).sum() / max(1, mask.sum())
        if ov > 0.05:
            overlaps.append(f"{other} {ov:.0%}")
    rep.add(f"step {step}  region {name} {'replaced' if existed else 'added'}: {mask.mean() * 100:.1f}% of canvas, "
            f"box {coords.fmt_box(box)}, focus {p.state['regions'][name]['focus']}  ({how})")
    if overlaps:
        rep.add(f"overlaps: {', '.join(overlaps)} of it (fine; use --minus <other> for a clean cut)")
    rep.add(f"regions: {f['n']}  covered {f['covered']:.0f}% of the canvas  "
            f"focus-3: {', '.join(f['focus3']) or 'none'}")
    if not ctx.batch:
        pan = _region_map_panel(p, f"REGIONS (new: {name}, magenta outline)", highlight=name)
        img = view.render([pan], (0, 0, p.W, p.H), max_side=view.MAP_SIDE)
        rep.view = _save_view(ctx, img, f"step-{step:04d}")
        rep.view_note = (f"every region tinted in its own colour; check that {name} covers its part and nothing "
                         "else")
    rep.touched = coords.mask_bbox_px(mask)
    rep.next = _finish_next(ctx)
    return rep


def _region_add(ctx: Ctx, args: List[str]) -> Report:
    sp = Spec(USAGE["region"], ["name"], {"box": "value", "marks": "value", "poly": "value", "wand": "value",
                                          "tol": "value", "focus": "value", "plus": "value", "minus": "value",
                                          "refine": "flag", "no-refine": "flag", "snap": "flag", "no-snap": "flag",
                                          "keep-holes": "flag"},
              alias={"mark": "marks", "polygon": "poly", "lasso": "poly", "rect": "box", "add": "plus",
                     "subtract": "minus", "weight": "focus", "tolerance": "tol"}, min_pos=1)
    a = parse(args, sp)
    p = ctx.project
    name = regions.clean_name(a["name"])
    H, W = p.H, p.W
    ways = [k for k in ("box", "marks", "poly", "wand") if a.get(k)]
    if not ways and not a.get("plus"):
        raise UsageError(f"say how to make region {name}: --marks, --box, --poly or --wand",
                         f"easel region add {name} --marks 3,4,9")
    if len(ways) > 1:
        raise UsageError(f"give one of --marks / --box / --poly / --wand (got {', '.join(ways)}); "
                         "combine with --plus / --minus <region>", f"easel region add {name} --marks 3,4 --plus hat")
    mask = np.zeros((H, W), bool)
    how = ""
    if a.get("box"):
        box = coords.parse_box(a["box"], ctx.notes, f"easel region add {name} --box 200,50,700,450")
        bp = coords.box_to_px(box, W, H)
        refine = (p.ref is not None) and not a.get("no-refine") and a.get("refine", True) is not False
        if refine:
            mask = regions.refine_box(p.ref, bp, ctx.notes)
            how = f"box {coords.fmt_box(box)} refined"
        else:
            mask = regions.box_mask(H, W, bp)
            how = f"box {coords.fmt_box(box)}"
    elif a.get("marks"):
        labels = _marks_labels(p)
        nums = coords.parse_int_list(a["marks"], f"easel region add {name} --marks 3,4,9")
        top = int(labels.max())
        bad = [n for n in nums if n < 1 or n > top]
        if bad:
            raise UsageError(f"marks {', '.join(map(str, bad))} do not exist (1-{top}); see  easel marks",
                             f"easel region add {name} --marks {','.join(str(n) for n in nums if 1 <= n <= top) or '1'}")
        mask = np.isin(labels, nums)
        how = "marks " + ",".join(map(str, nums))
    elif a.get("poly"):
        pts = coords.parse_points(a["poly"], ctx.notes, 3, f'easel region add {name} --poly "100,100 400,120 300,400"')
        pts_px = coords.points_to_px(pts, W, H)
        snap = p.ref is not None and not a.get("no-snap")
        mask = regions.poly_region(p.ref, H, W, pts_px, snap=snap)
        how = f"polygon of {len(pts)} points" + (" snapped to edges" if snap else "")
    elif a.get("wand"):
        img = p.ref if p.ref is not None else p.composite_u8("score")
        pt = coords.parse_point(a["wand"], ctx.notes, f"easel region add {name} --wand 500,300 --tol 15")
        tol = _num(a, "tol", 12.0, 1, 100, ctx.notes)
        mask = regions.wand(img, coords.point_to_px(pt, W, H), tol)
        how = f"wand at {coords.fmt_point(pt)} tol {tol:g}"
        if p.ref is not None:
            mask = regions.grow_into_lines(p.ref, mask)
        if not a.get("keep-holes"):
            from scipy import ndimage as _ndi
            filled = _ndi.binary_fill_holes(mask)
            if filled.sum() > mask.sum():
                how += " (holes filled)"
            mask = filled
    for key, op in (("plus", "|"), ("minus", "-")):
        if a.get(key):
            val = str(a[key]).strip()
            items = [val] if coords.looks_like_box(val) else re.split(r"[,\s]+", val)
            for other in items:
                if not other:
                    continue
                if other.isdigit() and other not in p.state["regions"]:
                    other = f"m{other}"          # a plain number is a mark number
                r = resolve_region(p, other, ctx.notes, f"easel region add {name} --marks 3 --{key} hair")
                mask = (mask | r.mask) if op == "|" else (mask & ~r.mask)
                how += f" {key} {r.label}"
    focus = coords.parse_int(a["focus"], "focus", 1, 3, ctx.notes) if a.get("focus") else None
    return _store_region(ctx, name, mask, how.strip(), focus, ctx.cmdline)


def _region_rest(ctx: Ctx, args: List[str]) -> Report:
    a = parse(args, Spec("easel region rest background", ["name"], {"focus": "value"}))
    p = ctx.project
    name = regions.clean_name(a.get("name") or "background")
    union = np.zeros((p.H, p.W), bool)
    for n in p.region_names():
        if n != name:
            union |= p.region_mask(n)
    mask = ~union
    if not mask.any():
        raise UsageError("every pixel is already in a region; nothing is left for the rest", "easel region list")
    focus = coords.parse_int(a["focus"], "focus", 1, 3, ctx.notes) if a.get("focus") else None
    return _store_region(ctx, name, mask, "rest of the canvas", focus, ctx.cmdline)


def _region_focus(ctx: Ctx, args: List[str]) -> Report:
    a = parse(args, Spec("easel region focus face 3", ["name", "focus"], min_pos=2))
    p = ctx.project
    r = resolve_region(p, a["name"], ctx.notes, "easel region focus hair 3")
    if r.name not in p.state["regions"]:
        raise UsageError("focus applies to named regions", "easel region focus face 3")
    f = coords.parse_int(a["focus"], "focus", 1, 3, ctx.notes, "easel region focus face 3")
    step = p.begin_step([])
    p.state["regions"][r.name]["focus"] = f
    workflow.build_todo(p.state)
    p.end_step(step, {"cmd": ctx.cmdline})
    rep = Report(mutated=True)
    rep.add(f"step {step}  region {r.name} focus {f}")
    rep.next = _finish_next(ctx)
    return rep


def _region_remove(ctx: Ctx, args: List[str]) -> Report:
    a = parse(args, Spec("easel region remove hair", ["name"], min_pos=1))
    p = ctx.project
    r = resolve_region(p, a["name"], ctx.notes, "easel region remove hair")
    if r.name not in p.state["regions"]:
        raise UsageError(f"{a['name']} is not a named region", "easel region list")
    step = p.begin_step([f"regions/{r.name}.png"])
    del p.state["regions"][r.name]
    p.delete_region_file(r.name)
    workflow.build_todo(p.state)
    p.end_step(step, {"cmd": ctx.cmdline})
    rep = Report(mutated=True)
    rep.add(f"step {step}  region {r.name} removed")
    rep.next = _finish_next(ctx)
    return rep


# ---------------------------------------------------------------------------------------- stage tools

def _stage_tool(ctx: Ctx, args: List[str], tool: str) -> Report:
    opts = {"layer": "value", "seed": "value"}
    if tool == "detail":
        opts.update({"level": "value"})
    if tool == "sketch":
        opts.update({"looseness": "value"})
    elif tool == "ink":
        opts.update({"detail": "value", "weight": "value", "pen": "value", "color": "value", "method": "value"})
    elif tool == "flat":
        opts.update({"colors": "value"})
    elif tool == "shade":
        opts.update({"levels": "value", "soft": "value"})
    elif tool in ("paint", "detail"):
        opts.update({"size": "value", "brush": "value", "density": "value", "strokes": "value"})
    a = parse(args, Spec(USAGE[tool], ["region*"], opts,
                         alias={"colour": "color", "colours": "colors", "brush-size": "size", "max-strokes": "strokes",
                                "n": "colors", "tone": "levels", "softness": "soft"}))
    p = ctx.project
    if p.ref is None:
        hint = workflow.HAND_HINT.get(p.state["stage"], 'easel stroke "x,y x,y ..." --color "#rrggbb"')
        raise UsageError(f"{tool} traces a reference picture and this project has none; draw this stage with "
                         "the hand tools, then tick the todo item with  easel todo done <n>", hint.split("   (")[0])
    ref = p.ref
    region = resolve_region(p, a.get("region"), ctx.notes, USAGE[tool])
    layer = (p.layer_entry(a["layer"])["name"] if a.get("layer") else LAYER_FOR[tool])
    fn = manual.stage_function(tool)
    st = p.state
    seed = int(a["seed"]) if a.get("seed") and str(a["seed"]).isdigit() else st["step"] + 1
    mask = region.mask
    clip = mask
    n = ctx.notes
    u = USAGE[tool]
    kw: Dict[str, object] = {"seed": seed}
    size_val = None
    pp = _pipe_params(st, {"ink": "line"}.get(tool, tool))
    if tool == "sketch":
        kw["looseness"] = _num(a, "looseness", float(pp.get("looseness", 0.5)), 0, 1, n, u)
        call = lambda: fn(ref, mask, **kw)  # noqa: E731
    elif tool == "ink":
        kw["detail"] = _num(a, "detail", float(pp.get("detail", 0.5)), 0, 1, n, u)
        kw["weight"] = _num(a, "weight", 1.0, 0.3, 3, n, u)
        pen = str(a.get("pen") or "pen").lower()
        if pen not in ("pen", "pencil", "brush", "marker"):
            raise UsageError(f"unknown --pen {pen!r}; one of pen, pencil, brush", "easel ink hair --pen pencil")
        kw["pen"] = pen
        col = a.get("color")
        kw["color"] = "auto" if not col or str(col).lower() == "auto" else coords.parse_colour(col, u)
        lm = _ref_lines(p)
        extra = {"lines": lm} if lm is not None and _accepts(fn, "lines") else {}
        if _accepts(fn, "true_thin"):
            kw["true_thin"] = bool(pp.get("true_thin", True))
        method = str(a.get("method") or pp.get("method") or "strokes").lower()
        if method not in ("strokes", "masses", "auto"):
            raise UsageError(f"unknown --method {method!r}; one of strokes, masses, auto", "easel ink all --method auto")
        if _accepts(fn, "method"):
            extra["method"] = method
        call = lambda: fn(ref, mask, **kw, **extra)  # noqa: E731
    elif tool == "flat":
        c = a.get("colors")
        kw["colors"] = "auto" if not c or str(c).lower() == "auto" else coords.parse_int(c, "colors", 1, 32, n, u)
        lm = _ref_lines(p)
        if pp.get("line_mask") == "none":                 # poster: flats first, no line barrier
            kw["line_mask"] = np.zeros((p.H, p.W), bool)
        elif lm is not None and getattr(lm, "mask", None) is not None:
            kw["line_mask"] = lm.mask                       # the reference's line map, as the pipeline uses
        else:
            line = p.layer("line", create=False)
            kw["line_mask"] = None if line is None else (line[..., 3] > 0.5)
        if pp.get("ink_mask"):            # the ink already drawn (solid masses too) is line art for the flats
            line = p.layer("line", create=False)
            if line is not None:
                base_lm = kw.get("line_mask")
                kw["line_mask"] = (line[..., 3] > 0.5) if base_lm is None else (base_lm | (line[..., 3] > 0.5))
        if pp.get("skip_dark") is not None and _accepts(fn, "skip_dark"):
            kw["skip_dark"] = float(pp["skip_dark"])
        call = lambda: fn(ref, mask, **kw)  # noqa: E731
    elif tool == "shade":
        kw["levels"] = coords.parse_int(a["levels"], "levels", 2, 5, n, u) if a.get("levels") else \
            int(_pipe_params(st, "shade").get("levels", 3))
        kw["soft"] = _num(a, "soft", float(_pipe_params(st, "shade").get("soft", 0.3)), 0, 1, n, u)
        base = p.composite_u8("score", upto="shade")
        if pp.get("air") and _accepts(fn, "air"):
            kw["air"] = pp["air"]
        if p.scale > 1:
            # the shade shapes (level 4) found on the delivered picture, laid at the working size (fast_coarse)
            def call(kw=kw, base=base):
                from .ops import scale_ops
                return scale_ops(fn(p.ref_deliver, p.mask_to_deliver(mask), p.to_deliver(base), **kw), p.scale,
                                 (p.H, p.W))
        else:
            call = lambda: fn(ref, mask, base, **kw)  # noqa: E731
        flat = p.layer("flat", create=False)
        if flat is not None:                                # shadows stay on painted areas, as in the pipeline
            clip = mask.astype(np.float32) * flat[..., 3]
    else:
        stage_for_size = st["stage"] if tool == "paint" else "detail"
        if tool == "paint" and stage_for_size not in ("block-in", "forms"):
            stage_for_size = "block-in" if "block-in" not in st["stages_done"] else "forms"
        hi = 6.0 if tool == "detail" else 400.0
        eng = manual.engine()
        names = manual.brush_names(eng) + (["shapes"] if tool == "detail" else [])   # fitted small shapes
        brush_arg = str(a["brush"]).lower() if a.get("brush") else None
        if brush_arg and brush_arg not in names:
            raise UsageError(f"unknown --brush {brush_arg!r}; brushes: {', '.join(names)}",
                             f"easel {tool} sky --brush round")
        density = _num(a, "density", 1.0, 0.25, 4, n, u)
        level = None
        if a.get("size"):
            size_val = _num(a, "size", 3.0, 1, hi, n, u)      # >= 1 px: the mark rules (DESIGN.md 10.1) hold
            passes = [(size_val, brush_arg or ("round" if size_val > 3 else "liner"), None)]
        elif tool == "detail" and _level_api():
            level = _level_arg(a, n, u, region, retouch=False)
            passes = _level_pass_list(p, st, region, level, brush_arg)
            if not passes:
                cur_lv = level_of(st, region.name)
                if cur_lv == 0:
                    msg = f"{region.name} is already retouched (level 0): nothing finer to lay"
                elif cur_lv == 1:
                    msg = f"{region.name} is already at level 1; the next and last level is  easel retouch {region.name}"
                else:
                    msg = f"{region.name} is already at level {cur_lv}; go finer with  easel detail {region.name} --level 1"
                raise UsageError(msg, f"easel look {region.name}")
        else:
            passes = _default_passes(st, stage_for_size, p.W, p.H, brush_arg)
        part = None
        sized_ = [x for x in passes if x[0] is not None]
        kw.update(size=(sized_[0][0] if len(sized_) == 1 else "/".join(f"{s:g}" for s, _, _ in sized_))
                  if sized_ else None,
                  brush="/".join(dict.fromkeys(b.replace("fast:", "") for _, b, _ in sized_)) or None,
                  density=density)
        if level is not None:
            kw["level"] = level
        if a.get("strokes"):
            kw["max_strokes"] = coords.parse_int(a["strokes"], "strokes", 1, 100000, n, u)
        ground = (tool == "paint" and stage_for_size == "block-in" and not a.get("size"))
        pl, _pk, pparams = _pipe_entry(st, stage_for_size)
        if pl and not a.get("layer") and tool == "detail":
            layer = pl                     # ink style: tone and texture under the ink, on the paint layer
        if tool == "detail" and level is not None and level <= 1 and not a.get("layer"):
            layer = "detail"               # the finest levels lie on top, over the ink too (as the pipeline)
        extra = {k: pparams[k] for k in ("threshold", "structure", "tips") if k in pparams and _accepts(fn, k)}
        sized = [x for x in passes if x[1] != "retouch"]
        texture = bool(pparams.get("texture")) and not a.get("size") and bool(sized) and \
            not workflow.dots_planned(st)      # with dots after level 1 the texture pass is left to streaks and dots
        highlights = bool(pparams.get("highlights")) and not a.get("size") and bool(sized) and \
            (level is None or level_of(st, region.name) is None)
        call = _passes_call(p, layer, fn, ref, mask, passes, density, seed, mask,
                            kw.get("max_strokes"), ground, extra, texture, highlights)
        make_call = (lambda m, ps=passes, tex=texture, hil=highlights: _passes_call(  # noqa: E731
            p, layer, fn, ref, m, ps, density, seed, m, kw.get("max_strokes"), ground, extra, tex, hil))
        size_val = passes[0][0] if passes[0][0] is not None else 1.0
        if tool == "paint" and not a.get("size"):
            size_val = workflow.paint_size(stage_for_size, p.W, p.H)   # the stage the passes belong to
    stage = workflow.stage_of_command(tool, size_val, p.W, p.H, workflow.plan_of(st))
    tick = (stage, region.name) if stage and region.named else None
    label = f"{tool} {region.label}" + "".join(f" --{k} {v}" for k, v in kw.items()
                                                 if k not in ("seed", "line_mask") and v not in (None, "auto"))
    if tool == "detail" and level is not None:
        return _level_step(ctx, p, region, level, label, layer, make_call, clip, stage, tick)
    return paint_step(ctx, label=label, layer=layer, build=lambda: list(call()), clip=clip,
                      area_px=region.box_px, region=region, tick=tick, sketch_view=(tool == "sketch"),
                      key_metric=KEY_METRIC.get(stage or "", "overall"), stage_tool=True)


# ---------------------------------------------------------------------------------------- detail levels

LEVEL_WORDS = {5: "block-in", 4: "forms", 3: "edges", 2: "fine strokes", 1: "hairlines",
               0: "retouch: reference pixels copied"}


def _streaks_available() -> bool:
    try:
        from .stage_streaks import streak_ops  # noqa: F401
        from .ops import scale_ops  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _level_api() -> bool:
    return workflow.levels_available()


def level_of(st: dict, name: str) -> Optional[int]:
    """The finest detail level a region has had (None = no detail pass yet); 'all' counts for every region."""
    lv = st.get("levels") or {}
    vals = [v for v in (lv.get(name), lv.get("all")) if v is not None]
    if name == "all" and st.get("regions"):
        per = [level_of(st, r) for r in st["regions"]]
        vals = [max(v for v in per if v is not None)] if all(v is not None for v in per) else vals
    return min(vals) if vals else None


def _level_arg(a, n, u, region, retouch: bool) -> int:
    if retouch:
        return 0
    lv = int(coords.parse_int(a["level"], "level", 1, 2, n, u)) if a.get("level") else 2
    return lv


def _level_pass_list(p: Project, st: dict, region, level: int, brush_arg=None) -> list:
    """The pipeline's passes of the detail stage down to ``level``, without the ones the region already had."""
    from .pipeline import level_passes, pass_level
    _l, _k, params = _pipe_entry(st, "detail")
    if not params:
        params = {"sizes": (2.5, 1.5, 1.0)}
    cur = level_of(st, region.name)
    if _level_pending(st, region.name, level):
        cur = st.get("level_pending_from", {}).get(f"{region.name}:{level}", cur)
    out = []
    sized = [x for x in _default_passes(st, "detail", p.W, p.H, brush_arg)]
    for size, brush, ea in sized:
        lv = pass_level(size, "liner" if brush == "streak" else brush.replace("fast:", ""))
        if lv < level or (cur is not None and lv >= cur):
            continue
        out.append((size, brush, ea))
    for size, brush, lv in level_passes(params, p.H, p.W, max_level=level):
        if brush != "retouch" or (cur is not None and lv >= cur):
            continue
        if lv == 1 and st.get("fine") == "streaks":
            out.append((1.0, "streak", None))           # level 1: hairline streaks (pipeline max_level=1)
        else:
            out.append((None, "retouch", lv))
    return out


def region_error(p: Project, mask: np.ndarray, canvas: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """(mean Delta E, share of pixels visibly off: Delta E > 5) of the canvas against the reference in ``mask``."""
    from .stage_levels import residual_de
    canvas = p.composite_u8("score") if canvas is None else canvas
    de = residual_de(p.ref_deliver, delivered(p, canvas))       # at the delivered size
    m = np.asarray(p.mask_to_deliver(mask), bool)
    if not m.any():
        return 0.0, 0.0
    v = de[m]
    return float(v.mean()), float((v > 5.0).mean())


def error_words(e: Tuple[float, float]) -> str:
    return f"off by {e[0]:.1f} Delta E on average, {100 * e[1]:.0f}% of it visibly off"


def detail_rank(p: Project) -> List[Tuple[str, int, Tuple[float, float]]]:
    """Regions in the order to refine: focus first, then the largest remaining difference."""
    st = p.state
    canvas = p.composite_u8("score")
    rows = []
    for name, r in st.get("regions", {}).items():
        e = region_error(p, p.region_mask(name), canvas)
        rows.append((name, int(r.get("focus", 1)), e))
    rows.sort(key=lambda t: (-t[1], -(t[2][0] * max(t[2][1], 0.01))))
    return rows


def retouch_mask(p: Project) -> np.ndarray:
    f = p.root / "retouched.png"
    if f.exists():
        try:
            from PIL import Image as _I
            m = np.asarray(_I.open(f).convert("L")) > 0
            if m.shape == (p.H, p.W):
                return m
        except OSError:
            pass
    return np.zeros((p.H, p.W), bool)


def fingerprint_line(p: Project) -> str:
    """Levels per region, share of the canvas retouched (level 0), working scale: what the picture is made of."""
    st = p.state
    lv = st.get("levels") or {}
    parts = []
    for name in st.get("regions", {}) or []:
        v = level_of(st, name)
        parts.append(f"{name} {'-' if v is None else v}")
    if not parts and lv:
        parts = [f"{k} {v}" for k, v in lv.items()]
    share = float(st.get("retouched", 0.0) or 0.0)
    dots = st.get("dotted")
    filt = st.get("deliver_filter") or "area"
    return (f"levels: {', '.join(parts) or 'none yet'}; retouched {100 * share:.1f}% of the canvas; "
            + ("contours; " if st.get("contours") else "")
            + (f"dotted {100 * float(dots):.1f}% of the canvas; " if dots is not None else "")
            + f"drawn at {int(st.get('work_scale', 1) or 1)}x; fine {st.get('fine') or 'plan'}; delivered "
            + ("area + sharpen" if filt == "sharpen" else filt))


STIPPLE_ROUNDS = 2          # dot rounds (each sees the dots of the one before)
SMOOTH_TOL = 0.8            # Delta E: the first detail streak pass's tolerance where the reference is smooth


def contours_available() -> bool:
    try:
        import inspect
        from .stage_streaks import contour_ops, streak_ops  # noqa: F401
        return "smooth_tol" in inspect.signature(streak_ops).parameters
    except Exception:  # noqa: BLE001
        return False


def _contour_part(p: Project, mask: np.ndarray, layer: str, eng, info: dict) -> list:
    """The contour pass of `detail all`: each thin line of the reference still off as ONE stroke along its centre
    line (stage_streaks.contour_ops), applied to the working layer; records the lines drawn and the error on them."""
    from .stage_streaks import contour_ops
    from .stage_levels import residual_de
    s = p.scale
    canvas = p.composite_u8("score")
    cst = {}
    ops = list(contour_ops(p.ref, mask, canvas, max_width=3.0 * s, min_len=3.0 * s, stats=cst))
    info["lines"] = int(cst.get("lines", len(ops)) or 0)
    if not ops:
        return ops
    orig = p.layer(layer)
    entry = p.layer_entry(layer)
    was_empty = entry.get("empty")
    cur = orig.copy()
    eng.apply_ops(cur, ops, clip=mask.astype(np.float32))
    p._layers[layer] = cur
    entry["empty"] = False
    after_c = p.composite_u8("score")
    p._layers[layer] = orig            # the step's loop lays the ops itself
    entry["empty"] = was_empty
    foot = np.abs(after_c.astype(np.int16) - canvas.astype(np.int16)).max(2) > 0
    fd = np.asarray(p.mask_to_deliver(foot), bool)
    if fd.any():
        info["before"] = float(residual_de(p.ref_deliver, delivered(p, canvas))[fd].mean())
        info["after"] = float(residual_de(p.ref_deliver, delivered(p, after_c))[fd].mean())
    return ops


def _auto_scale(W: int, H: int) -> int:
    try:
        from .pipeline import auto_work_scale
        return int(auto_work_scale(W, H))
    except Exception:  # noqa: BLE001
        return 2


def dots_available() -> bool:
    try:
        from .stage_streaks import stipple_ops  # noqa: F401
        from .pipeline import finish_sharpen  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def delivered(p: Project, img: np.ndarray) -> np.ndarray:
    """The picture as export delivers it: reduced to the delivered size (area filter), then the finishing sharpen
    once  easel sharpen  has set it."""
    out = p.to_deliver(img)
    if p.state.get("deliver_filter") == "sharpen" and p.ref is not None:
        from .pipeline import finish_sharpen
        out = finish_sharpen(out, p.ref_deliver)
    return out


def _de_pair(ref_d: np.ndarray, img_d: np.ndarray) -> Tuple[float, float]:
    from .stage_levels import residual_de
    v = residual_de(ref_d, img_d).ravel()
    return float(v.mean()), float((v > 5.0).mean())


def _most_changed_box(p: Project, a: np.ndarray, b: np.ndarray, frac: float = 0.2) -> Tuple[int, int, int, int]:
    """The working-px box (a side of about ``frac`` of the picture) where pictures a and b differ most."""
    import cv2
    d = np.abs(a[..., :3].astype(np.int16) - b[..., :3].astype(np.int16)).sum(2).astype(np.float32)
    if d.shape[:2] != (p.H, p.W):
        d = cv2.resize(d, (p.W, p.H), interpolation=cv2.INTER_NEAREST)
    sx = max(16, min(p.W, int(frac * max(p.W, p.H))))
    sy = max(16, min(p.H, int(frac * max(p.W, p.H))))
    s = cv2.boxFilter(d, -1, (sx, sy), normalize=False)
    y, x = np.unravel_index(int(np.argmax(s)), s.shape)
    x0 = int(np.clip(x - sx // 2, 0, max(0, p.W - sx)))
    y0 = int(np.clip(y - sy // 2, 0, max(0, p.H - sy)))
    return (x0, y0, min(p.W, x0 + sx), min(p.H, y0 + sy))


def cmd_dot(ctx: Ctx, args: List[str]) -> Report:
    """Stippling, after level 1: soft airbrush dots of 1.5 delivered px on the isolated points a stroke cannot carry
    (a glint, a star, a speck of sand or foliage), each in the colour that brings its delivered pixel to the
    reference; points on lines and edges are left out (--edges dots them too)."""
    a = parse(args, Spec(USAGE["dot"], ["region*"], {"rounds": "value", "edges": "flag", "layer": "value"},
                         alias={"round": "rounds"}))
    p = ctx.project
    p.require_ref("dot")
    if not dots_available():
        raise UsageError("this engine has no stippling pass", "easel detail all --level 1")
    if p.scale < 2:
        raise UsageError("dots are laid on a working canvas larger than the picture; this one is drawn at 1x",
                         "easel detail all --level 1")
    st = p.state
    region = resolve_region(p, a.get("region"), ctx.notes, USAGE["dot"])
    rounds = int(coords.parse_int(a["rounds"], "rounds", 1, 4, ctx.notes, USAGE["dot"])) if a.get("rounds") \
        else int(st.get("stipple") or STIPPLE_ROUNDS)
    edges = bool(a.get("edges")) or bool(st.get("stipple_edges", False))
    layer = p.layer_entry(a["layer"])["name"] if a.get("layer") else "detail"     # on top, as the pipeline
    s = p.scale
    bands = _bands(region.mask)
    parts = [(r, b) for r in range(rounds) for b in bands]
    key = region.name
    pend = (st.get("dot_pending") or {}).get(key) or {}
    start, dots0 = int(pend.get("next", 0)), int(pend.get("dots", 0))
    if start >= len(parts):
        start, dots0 = 0, 0
    whole = region.name == "all" and len(bands) == 1
    before_view = p.composite_u8("view")
    before_score = p.composite_u8("score")
    before = region_error(p, region.mask)
    prog = {"next": start, "dots": 0, "seconds": 0.0}
    from .stage_streaks import stipple_ops
    eng = manual.engine()

    def build():
        """Round by round, band by band from where the last call stopped, each part seeing the dots before it."""
        t0 = time.time()
        had = layer in p._layers
        orig = p._layers.get(layer)
        entry = p.layer_entry(layer)
        was_empty = entry.get("empty")
        out = []
        prev = 0.0
        try:
            for k in range(start, len(parts)):
                _r, (y0, y1) = parts[k]
                if whole:
                    m, md = None, None
                else:
                    m = np.zeros_like(region.mask)
                    m[y0:y1] = region.mask[y0:y1]
                    md = np.asarray(p.mask_to_deliver(m), bool)
                cd = p.to_deliver(p.composite_u8("score"))
                ops = list(stipple_ops(p.ref_deliver, cd, s, md, edges=edges))
                if ops:
                    cur = p.layer(layer).copy()
                    eng.apply_ops(cur, ops, clip=None if m is None else m.astype(np.float32))
                    p._layers[layer] = cur
                    entry["empty"] = False
                out += ops
                prog["dots"] += int(sum(len(o.pts) for o in ops))
                prog["next"] = k + 1
                el = time.time() - t0
                part, prev = el - prev, el
                if k + 1 < len(parts) and el + part > LEVEL_BUDGET:
                    break
        finally:
            entry["empty"] = was_empty
            if had:
                p._layers[layer] = orig
            else:
                p._layers.pop(layer, None)
        prog["seconds"] = time.time() - t0
        return out

    dw, dh = p.deliver_size()

    def on_apply(ops):
        dp = st.setdefault("dot_pending", {})
        total = dots0 + prog["dots"]
        if prog["next"] < len(parts):
            dp[key] = {"next": prog["next"], "dots": total}
            return
        dp.pop(key, None)
        dd = st.setdefault("dots", {})
        dd[key] = int(dd.get(key, 0)) + total
        st["dotted"] = round(sum(dd.values()) / float(dw * dh), 5)
        workflow.tick(st, "detail", None, key="dot")

    label = f"dot {region.label}" + (f" --rounds {rounds}" if a.get("rounds") else "") + \
        (" --edges" if a.get("edges") else "")
    rep = paint_step(ctx, label=label, layer=layer, build=build, clip=None if whole else region.mask,
                     area_px=region.box_px, region=region, key_metric=KEY_METRIC.get("detail", "overall"),
                     stage_tool=True, on_apply=on_apply)
    if not getattr(rep, "mutated", False):
        return rep
    if prog["next"] < len(parts):
        cmdline = ctx.cmdline.replace("easel ", "", 1) if ctx.cmdline else label
        rep.lines.insert(1, f"{region.name}: {prog['next']} of {len(parts)} parts dotted in {prog['seconds']:.0f} s; "
                            f"the same command continues where this one stopped")
        rep.next = f"easel {cmdline}   (continue: {len(parts) - prog['next']} part(s) of {region.name} left)"
        return rep
    total = dots0 + prog["dots"]
    after = region_error(p, region.mask)
    rep.lines.insert(1, f"dotted {total} texture points in {region.name}: {100 * total / float(dw * dh):.1f}% of the "
                        f"delivered pixels (soft dots of 1.5 px, each in the colour that brings its pixel to the "
                        f"reference" + ("" if edges else "; points on lines and edges left out") + ")")
    rep.lines.insert(2, f"{region.name}: {error_words(before)} -> {error_words(after)}")
    after_score = p.composite_u8("score")
    bd, ad = p.to_deliver(before_score), p.to_deliver(after_score)
    ch = np.abs(ad.astype(np.int16) - bd.astype(np.int16)).max(2) > 2
    if ch.any():
        from .stage_levels import residual_de
        rd = p.ref_deliver
        rep.lines.insert(3, f"on the {int(ch.sum())} pixels the dots changed: off by "
                            f"{float(residual_de(rd, bd)[ch].mean()):.1f} -> {float(residual_de(rd, ad)[ch].mean()):.1f} "
                            "Delta E")
    if not ctx.batch:
        after_view = p.composite_u8("view")
        box = _most_changed_box(p, before_view, after_view)
        _show(ctx, rep, view.Panel(after_view, f"CANVAS AFTER STEP {rep.step}: THE DOTS"), box,
              f"step-{rep.step:04d}-dots", "where the dots changed the canvas most")
        rep.next = "open VIEW and VIEW-REF (the dots up close), then: " + (rep.next or "easel next")
    return rep


def cmd_sharpen(ctx: Ctx, args: List[str]) -> Report:
    """The finishing sharpen of the delivered picture: an unsharp mask on the luminance (radius 0.8 px), its amount
    fitted to the reference per 32 px tile; scores and export use it from now on (--off takes it away)."""
    a = parse(args, Spec(USAGE["sharpen"], [], {"off": "flag"}))
    p = ctx.project
    p.require_ref("sharpen")
    try:
        from .pipeline import finish_sharpen
    except Exception:  # noqa: BLE001
        raise UsageError("this engine has no finishing sharpen", "easel fx sharpen --amount 0.3")
    st = p.state
    base = p.to_deliver(p.composite_u8("score"))
    sharp = finish_sharpen(base, p.ref_deliver)
    was = st.get("deliver_filter") == "sharpen"
    want = not a.get("off")
    old, new = (sharp if was else base), (sharp if want else base)
    e_b, e_a = _de_pair(p.ref_deliver, old), _de_pair(p.ref_deliver, new)
    diff = np.abs(sharp.astype(np.int16) - base.astype(np.int16)).max(2)
    changed = float((diff > 2).mean())
    step = p.begin_step([])
    try:
        st["deliver_filter"] = "sharpen" if want else "area"
        st["sharpen_off"] = not want
        stale = bool(st.get("exported")) and was != want
        if stale:
            st["exported"] = None
        ticked = workflow.tick(st, "finish", None, key="sharpen") if want else []
        p.end_step(step, {"cmd": ctx.cmdline, "deliver_filter": st["deliver_filter"]})
    except Exception:
        p.abort_step(step)
        raise
    rep = Report(mutated=True)
    rep.step = step
    rep.add(f"step {step}  sharpen{' --off' if not want else ''}: the delivered picture "
            + ("gets the finishing sharpen (an unsharp mask on the luminance, its amount fitted to the reference "
               "per 32 px tile)" if want else "goes back to the plain area reduction"))
    rep.add(f"the sharpen changes {100 * changed:.1f}% of the delivered pixels by more than 2 levels; whole picture: "
            f"{error_words(e_b)} -> {error_words(e_a)}")
    if (diff > 2).any():
        from .stage_levels import residual_de
        m_ = diff > 2
        rep.add(f"on those pixels: off by {float(residual_de(p.ref_deliver, old)[m_].mean()):.1f} -> "
                f"{float(residual_de(p.ref_deliver, new)[m_].mean()):.1f} Delta E")
    rep.add("look shows the working canvas; the scores, export and the VIEW below show the delivered picture")
    if stale:
        rep.add("the picture was exported before this step: export it again")
    if ticked:
        rep.add(f"todo ticked: {', '.join(str(i) for i in ticked)}")
    if not ctx.batch:
        import cv2
        big = cv2.resize(new, (p.W, p.H), interpolation=cv2.INTER_NEAREST)
        box = _most_changed_box(p, base, sharp)
        _show(ctx, rep, view.Panel(big, "DELIVERED PICTURE" + (", SHARPENED" if want else "")), box,
              f"step-{step:04d}-sharpen", "where the sharpen changes the delivered picture most")
    rep.next = _finish_next(ctx, None)
    if not ctx.batch:
        rep.next = "open VIEW and VIEW-REF (the sharpen up close), then: " + (rep.next or "easel next")
    return rep


LEVEL_BUDGET = 40.0         # s: one detail / retouch call stops after this and the same command continues
BAND_PX = 1_200_000         # working pixels per band of a region (streak and retouch passes run band by band)


def _bands(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Row ranges that cut a region into bands of about BAND_PX of its pixels."""
    rows = np.asarray(mask, bool).sum(1)
    tot = int(rows.sum())
    if tot <= BAND_PX:
        ys = np.flatnonzero(rows)
        return [(int(ys[0]), int(ys[-1]) + 1)] if len(ys) else [(0, mask.shape[0])]
    out, y0, acc = [], None, 0
    for y, n in enumerate(rows):
        if n and y0 is None:
            y0 = y
        acc += int(n)
        if y0 is not None and acc >= BAND_PX:
            out.append((y0, y + 1))
            y0, acc = None, 0
    if y0 is not None:
        out.append((y0, mask.shape[0]))
    return out


def _level_step(ctx: Ctx, p: Project, region, level: int, label: str, layer: str, make_call, clip, stage, tick):
    st = p.state
    before = region_error(p, region.mask) if p.ref is not None else None
    retouched = {"px": None}
    key = f"{region.name}:{level}"
    pend = st.get("level_pending") or {}
    start = int(pend.get(key, 0)) if isinstance(pend, dict) else 0
    passes = getattr(make_call, "__defaults__", None)
    passes = list(passes[0]) if passes else []
    tex = bool(make_call.__defaults__[1]) if passes else False
    hil = bool(make_call.__defaults__[2]) if passes else False
    whole = [x for x in passes if x[1] not in ("streak", "retouch")]
    fine_ps = [x for x in passes if x[1] in ("streak", "retouch")]
    # the parts of one level step, in the pipeline's order: the whole-region passes (shapes, strokes), the streak /
    # retouch passes band by band (each band cropped: its cost follows its size), the texture pass
    bands = []
    if whole or hil:
        bands.append(("whole", None))
    if level == 2 and any(x[1] == "streak" for x in fine_ps) and st.get("contours") and contours_available():
        bands.append(("contours", None))
    for y0, y1 in (_bands(region.mask) if fine_ps else []):
        bands.append(("band", (y0, y1)))
    if tex:
        bands.append(("texture", None))
    if not bands:
        bands = [("whole", None)]
    progress = {"next": start, "seconds": 0.0}

    def call():
        """Part by part from where the last call stopped, each part seeing the ones before it."""
        t0 = time.time()
        had = layer in p._layers
        orig = p._layers.get(layer)
        entry = p.layer_entry(layer)
        was_empty = entry.get("empty")
        out = []
        try:
            for b in range(start, len(bands)):
                kind_b, rows = bands[b]
                if kind_b == "band":
                    m = np.zeros_like(region.mask)
                    m[rows[0]:rows[1]] = region.mask[rows[0]:rows[1]]
                    c = make_call(m, fine_ps, False, False)
                elif kind_b == "texture":
                    m = region.mask
                    c = make_call(m, [], True, False)
                elif kind_b == "contours":
                    m = region.mask
                    info = progress.setdefault("contours", {})
                    c = (lambda m_=m, info_=info: _contour_part(p, m_, layer, manual.engine(), info_))
                else:
                    m = region.mask
                    c = make_call(m, whole, False, hil)
                ops = list(c())
                if ops:
                    cur = p.layer(layer).copy()
                    manual.engine().apply_ops(cur, ops, clip=m.astype(np.float32))
                    p._layers[layer] = cur
                    entry["empty"] = False
                out += ops
                progress["next"] = b + 1
                el = time.time() - t0
                part = el - progress.get("prev_el", 0.0)
                progress["prev_el"] = el
                # stop when the next part (about as long as this one) would pass the budget
                if b + 1 < len(bands) and el + part > LEVEL_BUDGET:
                    break
        finally:
            entry["empty"] = was_empty
            if had:
                p._layers[layer] = orig
            else:
                p._layers.pop(layer, None)
        progress["seconds"] = time.time() - t0
        return out

    def on_apply(ops):
        pl = st.setdefault("level_pending", {})
        pf = st.setdefault("level_pending_from", {})
        if progress["next"] < len(bands):
            pl[key] = progress["next"]          # not finished: the same command continues from here
            pf.setdefault(key, level_of(st, region.name))
            return
        pl.pop(key, None)
        pf.pop(key, None)
        if level == 2 and region.name == "all" and st.get("stage") == "detail":
            workflow.tick(st, "detail", "all")
        lv = st.setdefault("levels", {})
        if region.name == "all":
            lv["all"] = level if lv.get("all") is None else min(lv["all"], level)
        else:
            cur = level_of(st, region.name)
            lv[region.name] = level if cur is None else min(cur, level)
        zero = [o for o in ops if type(o).__name__ == "Retouch" and int(getattr(o, "level", 0) or 0) == 0
                and level == 0]
        if level == 0:
            m = retouch_mask(p)
            for o in ops:
                if type(o).__name__ != "Retouch":
                    continue
                ys = np.asarray(o.y).astype(int)
                xs = np.asarray(o.x).astype(int)
                if getattr(o, "n", None) is not None and getattr(o, "axis", None) is not None:
                    n_ = np.asarray(o.n).astype(int)
                    ax = np.asarray(o.axis).astype(int)
                    for k in range(int(n_.max()) if len(n_) else 0):
                        sel = n_ > k
                        yy = ys[sel] + np.where(ax[sel] == 1, k, 0)
                        xx = xs[sel] + np.where(ax[sel] == 0, k, 0)
                        ok = (yy < p.H) & (xx < p.W)
                        m[yy[ok], xx[ok]] = True
                else:
                    m[ys, xs] = True
            from .project import save_png
            save_png(p.root / "retouched.png", (m * 255).astype(np.uint8))
            st["retouched"] = round(float(m.mean()), 5)
            retouched["px"] = int(m.sum())
        del zero
        if region.name == "all" and level == 2 and stage == "detail" and p.ref is not None:
            st["detail_rank"] = [r for r, _f, _e in detail_rank(p)]
            workflow.build_todo(st)
        if level <= 1 and region.name != "all":
            workflow.tick(st, "detail", region.name, key="refine")
        if level == 0 and region.name != "all":
            workflow.tick(st, "detail", region.name, key="retouch")

    rep = paint_step(ctx, label=label, layer=layer, build=lambda: list(call()), clip=clip,
                     area_px=region.box_px, region=region,
                     tick=tick if (level == 2 and progress["next"] >= len(bands)) else None,
                     key_metric=KEY_METRIC.get(stage or "", "overall"), stage_tool=True,
                     extra_files=("retouched.png",) if level == 0 else (), on_apply=on_apply)
    if not getattr(rep, "mutated", False):
        return rep
    ci = progress.get("contours")
    if ci is not None:
        rep.lines.insert(1, f"contours: {ci.get('lines', 0)} thin lines of the reference drawn as one stroke each"
                         + (f"; error on those lines {ci['before']:.1f} -> {ci['after']:.1f} Delta E"
                            if "before" in ci else ""))
    if progress["next"] < len(bands):
        cmdline = ctx.cmdline.replace("easel ", "", 1) if ctx.cmdline else label
        rep.lines.insert(1, f"{region.name}: {progress['next']} of {len(bands)} parts done in {progress['seconds']:.0f} s; "
                            f"the same command continues where this one stopped")
        rep.next = f"easel {cmdline}   (continue: {len(bands) - progress['next']} part(s) of {region.name} left)"
        return rep
    lines = []
    if before is not None:
        after = region_error(p, region.mask)
        lines.append(f"{region.name} at level {level} ({LEVEL_WORDS[level]}): {error_words(before)} -> "
                     f"{error_words(after)}")
    if level == 0:
        lines.append(f"retouch copies the reference's pixels there (1 px dabs in its colours); retouched now: "
                     f"{100 * float(st.get('retouched', 0)):.1f}% of the canvas, shown in the picture's record")
    if region.name == "all" and level == 2 and p.ref is not None:
        rows = detail_rank(p)
        lines.append("refine next, in this order (focus first, then the largest difference):")
        for name, f, e in rows[:8]:
            lines.append(f"  {name} (focus {f}): {error_words(e)}  ->  easel detail {name} --level 1")
    for k, ln in enumerate(lines):
        rep.lines.insert(1 + k, ln)
    return rep


def _level_pending(st: dict, region_name: str, level: int) -> bool:
    return f"{region_name}:{level}" in (st.get("level_pending") or {})


def cmd_retouch(ctx: Ctx, args: List[str]) -> Report:
    """Level 0: 1 px dabs in the reference's colour on the pixels of a region still off (hairlines first)."""
    a = parse(args, Spec(USAGE["retouch"], ["region*"], {"because": "value", "layer": "value"}))
    p = ctx.project
    p.require_ref("retouch")
    region = resolve_region(p, a.get("region"), ctx.notes, USAGE["retouch"])
    if region.name == "all" and not a.get("because"):
        raise UsageError("retouch works on a region you name (the places that still differ); the whole picture "
                         "needs a reason", 'easel retouch all --because "the whole picture is a photo study"')
    if not _level_api():
        raise UsageError("this engine has no retouching pass", "easel detail face --level 1")
    st = p.state
    layer = p.layer_entry(a["layer"])["name"] if a.get("layer") else "detail"     # on top, as the pipeline
    passes = _level_pass_list(p, st, region, 0)
    passes = [x for x in passes if x[1] == "retouch"] or [(None, "retouch", 0)]
    fn = lambda *a_, **k_: []  # noqa: E731
    make_call = (lambda m, ps=passes, tex=False, hil=False: _passes_call(  # noqa: E731
        p, layer, fn, p.ref, m, ps, 1.0, p.state["step"] + 1, m, None, False, None, tex, hil))
    label = f"retouch {region.label}" + (f" --because {a['because']!r}" if a.get("because") else "")
    return _level_step(ctx, p, region, 0, label, layer, make_call, region.mask, "detail", None)


_LINES_CACHE: Dict[str, object] = {}


def _accepts(fn, name: str) -> bool:
    import inspect
    try:
        ps = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in ps or any(v.kind == v.VAR_KEYWORD for v in ps.values())


def _ref_lines(p: Project):
    """The reference's line map over the whole picture (what the scripted pipeline traces), once per process."""
    key = str(p.root)
    if key not in _LINES_CACHE:
        try:
            from .trace import line_map
            _LINES_CACHE.clear()
            _LINES_CACHE[key] = line_map(p.ref, mask=np.ones((p.H, p.W), bool))
        except Exception:  # noqa: BLE001  (engine without line_map: the stage computes its own)
            _LINES_CACHE[key] = None
    return _LINES_CACHE[key]


def _pipe_entry(st: dict, stage: str):
    """(layer, kind, params) of the pipeline's stage for this style, or (None, None, {})."""
    try:
        from .pipeline import STAGES as PIPE
        for name, layer, kind, params in PIPE.get(st.get("style") or "paint", []):
            if name == stage:
                return layer, kind, dict(params)
    except Exception:  # noqa: BLE001
        pass
    return None, None, {}


def _pipe_params(st: dict, stage: str) -> dict:
    """The scripted pipeline's parameters for this style and stage (the engine's ceiling settings)."""
    try:
        from .pipeline import STAGES as PIPE
        for name, _layer, _kind, params in PIPE.get(st.get("style") or "paint", []):
            if name == stage:
                return dict(params)
    except Exception:  # noqa: BLE001
        pass
    return {}


def _default_passes(st: dict, stage: str, W: int, H: int, brush_arg: Optional[str]):
    """[(size px, brush, edge_aware)] for paint / detail without --size: the pipeline's passes for the stage."""
    params = _pipe_params(st, stage)
    sizes = params.get("sizes")
    if not sizes:
        sizes = (2.5, 1.5, 1.0) if stage == "detail" else (workflow.paint_size(stage, W, H),)
    s_w = max(1, int(st.get("work_scale", 1) or 1))
    out = []
    for sz in sizes:
        if isinstance(sz, str) and sz.upper().startswith("L/"):
            size = max(W, H) / float(sz[2:])
        else:
            size = float(sz)
            if s_w > 1 and size > 3:      # the same marks on the delivered picture (pipeline.stage_passes)
                size *= s_w
        br = brush_arg or (params.get("brush", "round") if size > 3 or params.get("brush") == "soft" else "liner")
        out.append(_pass_kind(st, stage, round(size, 2), br, params.get("edge_aware")))
    if stage == "detail" and st.get("smooth_tol"):
        k = next((i for i, x in enumerate(out) if x[1] == "streak"), None)
        if k is not None:           # the first (widest) streak pass: the smooth areas' finer tolerance
            out[k] = (out[k][0], "streak", "smooth")
    return out


def _streaks_cropped(ref, mask, canvas, width: float, smooth_tol=None) -> list:
    """streak_ops on the region's box (plus the reach of a stroke) instead of the whole canvas: the same strokes
    (those whose middle lies in the region), at a cost that follows the region's size."""
    from .stage_streaks import STREAK_HALF, streak_ops
    from .ops import Streak
    kw = {"smooth_tol": smooth_tol} if smooth_tol else {}
    H, W = ref.shape[:2]
    if mask is None or np.asarray(mask).all():
        return list(streak_ops(ref, mask, canvas, width=width, **kw))
    m = np.asarray(mask, bool)
    ys, xs = np.nonzero(m)
    if not len(ys):
        return []
    pad = 2 * int(STREAK_HALF) + int(np.ceil(width)) + 4
    x0, y0 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
    x1, y1 = min(W, int(xs.max()) + 1 + pad), min(H, int(ys.max()) + 1 + pad)
    if (x1 - x0) * (y1 - y0) > 0.85 * W * H:
        return list(streak_ops(ref, m, canvas, width=width, **kw))
    sub = list(streak_ops(np.ascontiguousarray(ref[y0:y1, x0:x1]), m[y0:y1, x0:x1],
                          np.ascontiguousarray(canvas[y0:y1, x0:x1]), width=width, **kw))
    out = []
    for o in sub:
        if isinstance(o, Streak):
            o = Streak(pts=o.pts + np.array([x0, y0], np.float32), n=o.n, colors=o.colors, width=o.width, own=True)
        out.append(o)
    return out


def _pass_kind(st: dict, stage: str, size: float, br: str, edge_aware):
    """One pass as the pipeline runs it on this project's canvas: on a working canvas the forms / block-in passes
    (level 4-5) are found on the delivered picture and laid scaled ("fast:" brush); with fine="streaks" the liner
    passes and the detail stage's small brush pass are streaks."""
    s_w = max(1, int(st.get("work_scale", 1) or 1))
    try:
        from .pipeline import STREAK_MAX, pass_level
    except Exception:  # noqa: BLE001
        return (size, br, edge_aware)
    lv = pass_level(size, br)
    if st.get("fine") == "streaks":
        small_detail = stage == "detail" and br not in ("shapes", "soft") and size <= STREAK_MAX * s_w
        if br == "liner" or small_detail:
            return (size if br == "liner" else float(STREAK_MAX), "streak", None)
    if s_w > 1 and lv >= 4:
        return (size, "fast:" + br, edge_aware)
    return (size, br, edge_aware)


def _passes_call(p: Project, layer: str, fn, ref, mask, passes, density, seed, clip, max_strokes, ground,
                 extra=None, texture=False, highlights=False):
    """Several paint passes in one step: each pass sees the canvas left by the one before (as the pipeline runs
    them); the ops of all passes are returned in order and applied together by paint_step."""
    def call():
        eng = manual.engine()
        entry = p.layer_entry(layer)
        had = layer in p._layers
        orig = p._layers.get(layer)
        was_empty = entry.get("empty")
        cur = p.layer(layer).copy()
        clipf = None if clip is None else np.asarray(clip, np.float32)
        out = []
        try:
            if ground:
                from .stage_paint import ground_ops
                g = list(ground_ops(ref, mask))
                eng.apply_ops(cur, g, clip=clipf)
                out += g
            try:
                from .stage_paint import highlight_ops
            except ImportError:
                highlight_ops = None
            if highlights and highlight_ops is not None:   # thin light marks first (edge lights, glints)
                p._layers[layer] = cur
                entry["empty"] = False
                canvas = p.composite_u8("score")
                h = list(highlight_ops(ref, mask, canvas, seed=seed * 100 + 98))
                eng.apply_ops(cur, h, clip=clipf)
                out += h
            try:
                from .pipeline import MAX_CALLS
            except Exception:  # noqa: BLE001
                MAX_CALLS = 1
            for i, (size, brush, edge_aware) in enumerate(passes):
                if brush == "retouch":          # level 1 hairlines / level 0 retouching dabs on the residual
                    from .stage_levels import retouch_ops
                    p._layers[layer] = cur
                    entry["empty"] = False
                    canvas = p.composite_u8("score")
                    ops = list(retouch_ops(ref, mask, canvas, level=int(edge_aware)))
                    eng.apply_ops(cur, ops, clip=clipf)
                    out += ops
                    continue
                if brush == "streak":           # thin strokes whose colour and width follow the reference
                    p._layers[layer] = cur
                    entry["empty"] = False
                    canvas = p.composite_u8("score")
                    ops = _streaks_cropped(ref, mask, canvas, float(size),
                                           p.state.get("smooth_tol") if edge_aware == "smooth" else None)
                    eng.apply_ops(cur, ops, clip=clipf)
                    out += ops
                    continue
                fast = brush.startswith("fast:")
                brush = brush[5:] if fast else brush
                first_n = 1
                for k in range(MAX_CALLS):
                    p._layers[layer] = cur
                    entry["empty"] = False
                    canvas = p.composite_u8("score")
                    kw = dict(size=size, brush=brush, density=density, seed=seed * 100 + i + 31 * k)
                    if edge_aware is not None and _accepts(fn, "edge_aware"):
                        kw["edge_aware"] = edge_aware
                    if max_strokes:
                        kw["max_strokes"] = max_strokes
                    kw.update(extra or {})
                    pst = {}
                    if _accepts(fn, "stats"):
                        kw["stats"] = pst
                    if fast:      # found on the delivered picture, laid at the working size (pipeline fast_coarse)
                        from .ops import scale_ops
                        s_w = p.scale
                        kw["size"] = size / s_w
                        md = p.mask_to_deliver(mask)
                        ops = list(scale_ops(fn(p.ref_deliver, md, p.to_deliver(canvas), **kw), s_w, (p.H, p.W)))
                    else:
                        ops = list(fn(ref, mask, canvas, **kw))
                    eng.apply_ops(cur, ops, clip=clipf)
                    out += ops
                    if k == 0:
                        first_n = len(ops)
                    if not pst.get("out_of_time") or len(ops) < 0.08 * max(first_n, 1):
                        break
            if texture:                      # the fine streaks of the texture as directed pen marks
                from .stage_paint import texture_ops
                p._layers[layer] = cur
                entry["empty"] = False
                canvas = p.composite_u8("score")
                ops = list(texture_ops(ref, mask, canvas, seed=seed * 100 + 99))
                eng.apply_ops(cur, ops, clip=clipf)
                out += ops
        finally:
            entry["empty"] = was_empty
            if had:
                p._layers[layer] = orig
            else:
                p._layers.pop(layer, None)
        return out
    return call


def cmd_airbrush(ctx: Ctx, args: List[str]) -> Report:
    """Soft airbrush dabs (one Spray op per region) toward the reference, clear of its hard edges."""
    a = parse(args, Spec(USAGE["airbrush"], ["region*"], {"sizes": "value", "layer": "value"}))
    p = ctx.project
    ref = p.require_ref("airbrush")
    region = resolve_region(p, a.get("region"), ctx.notes, USAGE["airbrush"])
    _pl, _pk, pparams = _pipe_entry(p.state, "forms")
    sizes = tuple(int(x) for x in re.findall(r"\d+", str(a["sizes"]))) if a.get("sizes") else \
        tuple(v * p.scale for v in (pparams.get("sizes") or (64, 32, 16, 8)))
    if not sizes:
        raise UsageError("--sizes needs numbers like 64,32,16,8", USAGE["airbrush"])
    layer = p.layer_entry(a["layer"])["name"] if a.get("layer") else "paint"
    from .airbrush import airbrush_ops, step_edges
    canvas = p.composite_u8("score")
    seed = p.state["step"] + 1
    build = lambda: list(airbrush_ops(ref, canvas, region.mask.astype(np.int32), sizes=sizes, seed=seed,  # noqa: E731
                                      edges=step_edges(ref, sigma=2.0, thr=float(pparams.get("edge", 4.0)))))
    stage = "forms" if "forms" in workflow.plan_of(p.state) else None
    tick = (stage, region.name) if stage and region.named else None
    return paint_step(ctx, label=f"airbrush {region.label} --sizes {','.join(map(str, sizes))}", layer=layer,
                      build=build, clip=region.mask, area_px=region.box_px, region=region, tick=tick,
                      key_metric=KEY_METRIC.get(stage or "", "overall"), stage_tool=True)


def cmd_sketch(ctx, args):
    return _stage_tool(ctx, args, "sketch")


def cmd_ink(ctx, args):
    return _stage_tool(ctx, args, "ink")


def cmd_flat(ctx, args):
    return _stage_tool(ctx, args, "flat")


def cmd_shade(ctx, args):
    return _stage_tool(ctx, args, "shade")


def cmd_paint(ctx, args):
    return _stage_tool(ctx, args, "paint")


def cmd_detail(ctx, args):
    return _stage_tool(ctx, args, "detail")


# ---------------------------------------------------------------------------------------- hand tools

def _layer_arg(p: Project, a, tool: str) -> str:
    return p.layer_entry(a["layer"])["name"] if a.get("layer") else default_layer(p, tool)


def _colour_arg(p: Project, a, default="pick"):
    c = a.get("color") or default
    if str(c).lower() in ("pick", "ref", "auto", "reference"):
        if p.ref is None:
            raise UsageError("--color pick needs a reference; give a colour", "--color c04030")
        return "pick"
    try:
        return coords.parse_colour(c)
    except UsageError:
        if not p.state.get("draw"):
            raise
        from .draw_plan import swatch       # a palette name (hair, skin-shadow) or pick@x,y, as the draw tools take
        return swatch(p, c)


def _hand_pad(p: Project) -> float:
    """Hand tools are scored over the mark plus this much of its neighbourhood (px)."""
    return max(12.0, 0.02 * max(p.W, p.H))


def _default_size(p: Project) -> float:
    return max(2.0, round(max(p.W, p.H) / 300.0, 1))


def cmd_stroke(ctx: Ctx, args: List[str]) -> Report:
    sp = Spec(USAGE["stroke"], ["path*"], {"brush": "value", "size": "value", "color": "value", "opacity": "value",
                                          "snap": "flag", "closed": "flag", "layer": "value", "smooth": "value"},
              alias={"colour": "color", "width": "size", "close": "closed"}, min_pos=1)
    a = parse(args, sp)
    p = ctx.project
    pts = coords.parse_points(a["path"], ctx.notes, 1, USAGE["stroke"])
    px = coords.points_to_px(pts, p.W, p.H)
    snap = bool(a.get("snap"))
    if snap and p.ref is None:
        ctx.notes.append("--snap needs a reference; drawn as given")
        snap = False
    path = manual.smooth_path(px, closed=bool(a.get("closed")), snap_ref=p.ref if snap else None,
                              smoothing=_num(a, "smooth", 0.5, 0, 1, ctx.notes))
    size = _num(a, "size", _default_size(p), 0.5, 300, ctx.notes, USAGE["stroke"])
    col = _colour_arg(p, a, default="#202020")
    colour = manual.footprint_colour(p.ref, path, size) if col == "pick" else col
    brush = str(a.get("brush") or "pen").lower()
    opacity = _num(a, "opacity", 1.0, 0, 1, ctx.notes)
    layer = _layer_arg(p, a, "stroke")
    seed = p.state["step"] + 1

    def build():
        eng = manual.engine()
        if brush not in manual.brush_names(eng):
            raise UsageError(f"unknown --brush {brush!r}; brushes: {', '.join(manual.brush_names(eng))}",
                             "easel stroke \"100,100 200,150\" --brush pen")
        return [manual.stroke_op(eng, path, size, colour, opacity, brush, seed)]
    pad = size + _hand_pad(p)
    x0, y0 = path.min(0) - pad
    x1, y1 = path.max(0) + pad
    area = (int(max(0, x0)), int(max(0, y0)), int(min(p.W, x1 + 1)), int(min(p.H, y1 + 1)))
    reg = Region("stroke", regions.box_mask(p.H, p.W, area), area, "around the stroke", False)
    label = f"stroke {len(pts)} points" + (" snapped" if snap else "") + f" size {size:g}"
    return paint_step(ctx, label=label, layer=layer, build=build, clip=None, area_px=area, region=reg)


def cmd_fill(ctx: Ctx, args: List[str]) -> Report:
    sp = Spec(USAGE["fill"], ["point*"], {"color": "value", "tol": "value", "bounds": "value", "layer": "value",
                                         "gap": "value", "opacity": "value"},
              alias={"colour": "color", "tolerance": "tol", "bound": "bounds"}, min_pos=1)
    a = parse(args, sp)
    p = ctx.project
    pt = coords.parse_point(a["point"], ctx.notes, USAGE["fill"])
    px = coords.point_to_px(pt, p.W, p.H)
    line = p.layer("line", create=False)
    bounds = str(a.get("bounds") or ("line" if line is not None else ("ref" if p.ref is not None else "canvas"))).lower()
    if bounds not in ("line", "ref", "canvas"):
        raise UsageError(f"unknown --bounds {bounds!r}; one of line, ref, canvas", "easel fill 500,400 --bounds ref")
    if bounds == "line" and line is None:
        ctx.notes.append("the line layer is empty; filled by reference colour instead (--bounds ref)")
        bounds = "ref" if p.ref is not None else "canvas"
    if bounds == "ref" and p.ref is None:
        bounds = "canvas"
    tol = _num(a, "tol", 12.0, 1, 100, ctx.notes)
    gap = coords.parse_int(a["gap"], "gap", 0, 20, ctx.notes) if a.get("gap") else 3
    mask = manual.bucket_mask(px, bounds, None if line is None else line[..., 3], p.ref, p.composite_u8("score"),
                              tol, gap)
    col = _colour_arg(p, a)
    colour = manual.pick_colour(p.ref, mask=mask) if col == "pick" else col
    layer = _layer_arg(p, a, "fill")
    opacity = _num(a, "opacity", 1.0, 0, 1, ctx.notes)
    box = coords.mask_bbox_px(mask)
    if bounds == "line" and mask.mean() > 0.4:
        ctx.notes.append(f"the fill covered {mask.mean():.0%} of the canvas: the line art does not close this "
                         "area; if that is wrong, undo and use --bounds ref")
    reg = Region("fill", mask, box, "filled area", False)
    label = f"fill at {coords.fmt_point(pt)} bounds {bounds}: {mask.mean() * 100:.1f}% of canvas, " \
            f"{coords.hex_colour(colour)} ({coords.colour_name(colour)})"
    return paint_step(ctx, label=label, layer=layer,
                      build=lambda: [manual.fill_for_mask(manual.engine(), mask, colour, opacity)],
                      clip=mask, area_px=box, region=reg)


def cmd_shape(ctx: Ctx, args: List[str]) -> Report:
    sp = Spec(USAGE["shape"], ["kind", "coords*"], {"color": "value", "outline": "value", "fill": "flag",
                                                   "layer": "value", "opacity": "value", "brush": "value"},
              alias={"colour": "color"}, min_pos=2)
    a = parse(args, sp)
    p = ctx.project
    kind = str(a["kind"]).lower()
    kind = {"circle": "ellipse", "oval": "ellipse", "rectangle": "rect", "box": "rect", "polygon": "poly"}.get(kind, kind)
    kind = {"tri": "triangle"}.get(kind, kind)
    if kind not in ("ellipse", "rect", "triangle", "line", "poly"):
        raise UsageError(f"unknown shape {a['kind']!r}; one of ellipse, rect, triangle, line, poly", USAGE["shape"])
    col = _colour_arg(p, a)
    if kind in ("ellipse", "rect", "triangle"):
        box = coords.parse_box(a["coords"], ctx.notes, f"easel shape {kind} 400,300,600,450")
        bp = [coords.unit_to_px(box[0], p.W), coords.unit_to_px(box[1], p.H), coords.unit_to_px(box[2], p.W),
              coords.unit_to_px(box[3], p.H)]
        if kind == "triangle":          # apex at the top centre of the box, base along its bottom
            pts = np.array([[(bp[0] + bp[2]) / 2.0, bp[1]], [bp[2], bp[3]], [bp[0], bp[3]]], np.float32)
        else:
            pts = manual.ellipse_points(bp) if kind == "ellipse" else manual.rect_points(bp)
    else:
        upts = coords.parse_points(a["coords"], ctx.notes, 2 if kind == "line" else 3,
                                   f'easel shape {kind} "100,100 300,120 200,300"')
        pts = coords.points_to_px(upts, p.W, p.H)
    outline = _num(a, "outline", 0.0, 0, 200, ctx.notes) if a.get("outline") else 0.0
    do_fill = bool(a.get("fill")) or (not outline and kind != "line")
    if kind == "line" and not outline:
        outline = _default_size(p)
    from .regions import polygon_mask
    shape_mask = polygon_mask(p.H, p.W, pts) if kind != "line" else np.zeros((p.H, p.W), bool)
    colour = manual.pick_colour(p.ref, mask=shape_mask) if (col == "pick" and shape_mask.any()) else \
        (manual.pick_colour(p.ref, pts[0]) if col == "pick" else col)
    opacity = _num(a, "opacity", 1.0, 0, 1, ctx.notes)
    brush = str(a.get("brush") or "pen").lower()
    layer = _layer_arg(p, a, "stroke" if (outline and not do_fill) else "shape")   # an outline is a line

    def build():
        eng = manual.engine()
        ops = []
        if do_fill and shape_mask.any():
            ops.append(eng.Fill(polygons=[pts.astype(np.float32)], mask_rle=None, color=tuple(colour),
                                opacity=opacity))
        if outline:
            path = np.vstack([pts, pts[:1]]) if kind != "line" else pts
            ops.append(manual.stroke_op(eng, manual.resample(path, 2.0), outline, colour, opacity, brush,
                                        p.state["step"] + 1))
        return ops
    x0, y0 = pts.min(0) - outline - _hand_pad(p)
    x1, y1 = pts.max(0) + outline + _hand_pad(p)
    area = (int(max(0, x0)), int(max(0, y0)), int(min(p.W, x1)), int(min(p.H, y1)))
    reg = Region("shape", regions.box_mask(p.H, p.W, area), area, f"{kind} area", False)
    return paint_step(ctx, label=f"shape {kind} {coords.hex_colour(colour)} ({coords.colour_name(colour)})"
                      + (" filled" if do_fill else "") + (f" outline {outline:g}" if outline else ""),
                      layer=layer, build=build, clip=None, area_px=area, region=reg)


def cmd_gradient(ctx: Ctx, args: List[str]) -> Report:
    sp = Spec(USAGE["gradient"], ["region*"], {"from": "value", "to": "value", "colors": "value", "layer": "value",
                                              "opacity": "value"},
              alias={"colours": "colors", "color": "colors", "start": "from", "end": "to"})
    a = parse(args, sp)
    p = ctx.project
    region = resolve_region(p, a.get("region"), ctx.notes, USAGE["gradient"])
    if not a.get("from") or not a.get("to"):
        raise UsageError("gradient needs --from x,y and --to x,y", USAGE["gradient"])
    f = coords.parse_point(a["from"], ctx.notes, USAGE["gradient"])
    t = coords.parse_point(a["to"], ctx.notes, USAGE["gradient"])
    fp, tp = coords.point_to_px(f, p.W, p.H), coords.point_to_px(t, p.W, p.H)
    cs = str(a.get("colors") or "pick")
    if cs.lower() in ("pick", "ref", "auto"):
        ref = p.require_ref("--colors pick")
        c1, c2 = manual.pick_colour(ref, fp), manual.pick_colour(ref, tp)
    else:
        parts = [x for x in re.split(r"[,;\s]+(?=#|[a-zA-Z])", cs.strip()) if x]
        if len(parts) != 2:
            raise UsageError("--colors needs two colours like \"#a0c0ff,#203060\" (or pick)", USAGE["gradient"])
        c1, c2 = coords.parse_colour(parts[0].strip(", ")), coords.parse_colour(parts[1].strip(", "))
    layer = _layer_arg(p, a, "gradient")
    opacity = _num(a, "opacity", 1.0, 0, 1, ctx.notes)
    return paint_step(ctx, label=f"gradient {region.label} {coords.hex_colour(c1)} -> {coords.hex_colour(c2)}",
                      layer=layer, build=lambda: [manual.fill_for_mask(manual.engine(), region.mask, c1, opacity,
                                                                       color2=c2, p0=fp, p1=tp)],
                      clip=region.mask, area_px=region.box_px, region=region)


def cmd_erase(ctx: Ctx, args: List[str]) -> Report:
    sp = Spec(USAGE["erase"], ["target*"], {"size": "value", "layer": "value"}, alias={"width": "size"}, min_pos=1)
    a = parse(args, sp)
    p = ctx.project
    layer = _layer_arg(p, a, "erase")
    t = str(a["target"])
    nums = len(re.findall(r"[-+]?\d*\.?\d+", t))
    if nums >= 4 and nums % 2 == 0 and nums != 4 or (nums == 2):
        pts = coords.parse_points(t, ctx.notes, 1, USAGE["erase"])
        path = manual.resample(coords.points_to_px(pts, p.W, p.H), 2.0)
        size = _num(a, "size", _default_size(p) * 3, 0.5, 400, ctx.notes)
        x0, y0 = path.min(0) - size - _hand_pad(p)
        x1, y1 = path.max(0) + size + _hand_pad(p)
        area = (int(max(0, x0)), int(max(0, y0)), int(min(p.W, x1 + 1)), int(min(p.H, y1 + 1)))
        reg = Region("erase", regions.box_mask(p.H, p.W, area), area, "around the erased path", False)
        return paint_step(ctx, label=f"erase path on {layer}", layer=layer,
                          build=lambda: [manual.stroke_op(manual.engine(), path, size, (0, 0, 0), 1.0, "round",
                                                          0, erase=True)],
                          clip=None, area_px=area, region=reg)
    region = resolve_region(p, t, ctx.notes, USAGE["erase"])
    return paint_step(ctx, label=f"erase {region.label} on {layer}", layer=layer,
                      build=lambda: [manual.fill_for_mask(manual.engine(), region.mask, (0, 0, 0), 1.0, erase=True)],
                      clip=region.mask, area_px=region.box_px, region=region)


def _blend_params(st: dict) -> Optional[dict]:
    """The pipeline's blend for the current stage: its "blend" stage up to the shade stage, its finish blend from the
    detail stage on. The layers are the ones the blend melts under the detail (flat / shade, paint): never the
    detail layer, which holds the level-1 strokes, the contours and the dots by then (the pipeline blends before it
    paints them). ``name`` is the pipeline stage the blend belongs to."""
    try:
        from .pipeline import STAGES as PIPE
    except Exception:  # noqa: BLE001
        return None
    want = "finish" if st.get("stage") in ("detail", "finish") else "blend"
    plan = PIPE.get(st.get("style") or "paint", [])
    hit = [(n, p) for n, _l, k, p in plan if k == "blend" and n == want] or         [(n, p) for n, _l, k, p in plan if k == "blend"]
    if not hit:
        return None
    name, params = hit[0]
    out = dict(params)
    out["layers"] = tuple(ln for ln in params.get("layers", ()) if ln != "detail")
    out["name"] = name
    return out


def cmd_smooth(ctx: Ctx, args: List[str]) -> Report:
    a = parse(args, Spec(USAGE["smooth"], ["region*"], {"amount": "value", "layer": "value"}))
    p = ctx.project
    region = resolve_region(p, a.get("region"), ctx.notes, USAGE["smooth"])
    try:
        from .stage_shade import blend_ops
    except Exception:  # noqa: BLE001
        blend_ops = None
    if blend_ops is not None and p.ref is not None:
        bp = _blend_params(p.state) or {"layers": (default_layer(p, "smooth"),), "radius": 6.0, "tol": 25.0}
        bare = not a.get("layer") and a.get("amount") is None
        done = (p.state.get("blended") or {}).get(f"{bp.get('name', '-')}:{region.name}")
        if bare and done:
            # the same blend again: it melts what is already melted, the scores do not move; say so, go on
            rep = Report()
            rep.add(f"smooth {region.label}: this blend was done at step {done} (layers "
                    f"{', '.join(bp.get('layers', ())) or '-'}); a second one changes nothing - no step made")
            rep.next = _finish_next(ctx)
            return rep
        layers = [p.layer_entry(a["layer"])["name"]] if a.get("layer") else             [ln for ln in bp.get("layers", ()) if p.layer(ln, create=False) is not None]
        if not layers:
            raise UsageError("nothing painted to blend yet; name the layer", f"easel smooth {region.name} --layer flat")
        amount = _num(a, "amount", 1.0, 0, 1, ctx.notes)
        ops = list(blend_ops(p.ref, region.mask, radius=float(bp.get("radius", 3.0)), tol=float(bp.get("tol", 10.0)),
                             strength=float(bp.get("strength", 1.0)) * amount))
        if not ops:
            rep = Report()
            rep.add(f"smooth {region.label}: the reference has no smooth area there to blend; nothing changed")
            rep.next = _finish_next(ctx)
            return rep
        reps = []
        for k, ln in enumerate(layers):
            sub_ctx = ctx if k == len(layers) - 1 else Ctx(ctx.project_arg, batch=True)
            sub_ctx._project, sub_ctx.cmdline = p, ctx.cmdline
            reps.append(paint_step(sub_ctx, label=f"smooth {region.label} (blend, layer {ln})", layer=ln,
                                   build=lambda: list(ops), clip=region.mask, area_px=region.box_px, region=region,
                                   stage_tool=True))
        rep = reps[-1]
        if bare:
            p.state.setdefault("blended", {})[f"{bp.get('name', '-')}:{region.name}"] = rep.step
        ticked = workflow.tick(p.state, p.state["stage"], None, key="blend")
        if ticked or bare:
            p.save_state()
        if ticked:
            rep.add(f"todo ticked: {', '.join(str(i) for i in ticked)}")
        if not rep.next.startswith("easel undo"):
            rep.next = _finish_next(ctx)            # after the tick: the blend is not offered again
        if len(reps) > 1:
            rep.lines.insert(0, f"blended {len(reps)} layers ({', '.join(layers)}) as {len(reps)} steps; "
                                f"undo them with  easel undo {len(reps)}")
        return rep
    layer = _layer_arg(p, a, "smooth")
    amount = _num(a, "amount", 0.5, 0, 1, ctx.notes)
    if p.layer(layer, create=False) is None:
        raise UsageError(f"layer {layer} is empty; name the layer to smooth", f"easel smooth {region.name} --layer paint")
    return paint_step(ctx, label=f"smooth {region.label} amount {amount:g}", layer=layer, build=lambda: [],
                      clip=None, area_px=region.box_px, region=region,
                      pixel_fn=lambda arr: manual.smooth_layer(arr, region.mask, amount))


def cmd_adjust(ctx: Ctx, args: List[str]) -> Report:
    a = parse(args, Spec(USAGE["adjust"], ["region*"], {"brightness": "value", "contrast": "value",
                                                       "saturation": "value", "hue": "value", "layer": "value"},
                         alias={"lightness": "brightness", "sat": "saturation"}))
    p = ctx.project
    region = resolve_region(p, a.get("region"), ctx.notes, USAGE["adjust"])
    layer = _layer_arg(p, a, "adjust")
    vals = {k: _num(a, k, 0.0, -1, 1, ctx.notes) for k in ("brightness", "contrast", "saturation")}
    vals["hue"] = _num(a, "hue", 0.0, -180, 180, ctx.notes)
    if not any(vals.values()):
        raise UsageError("adjust needs at least one of --brightness --contrast --saturation (-1..1) --hue (degrees)",
                         USAGE["adjust"])
    if p.layer(layer, create=False) is None:
        raise UsageError(f"layer {layer} is empty; name the layer to adjust", f"easel adjust {region.name} --layer flat "
                         "--brightness 0.1")
    desc = " ".join(f"{k} {v:+g}" for k, v in vals.items() if v)
    return paint_step(ctx, label=f"adjust {region.label} {desc}", layer=layer, build=lambda: [], clip=None,
                      area_px=region.box_px, region=region,
                      pixel_fn=lambda arr: manual.adjust_layer(arr, region.mask, **vals))


# ---------------------------------------------------------------------------------------- layer / export / later

def cmd_layer(ctx: Ctx, args: List[str]) -> Report:
    sp = Spec(USAGE["layer"], ["name"], {"mode": "value", "opacity": "value", "hide": "flag", "show": "flag",
                                        "clip": "flag", "no-clip": "flag", "lock": "flag", "unlock": "flag"},
              alias={"blend": "mode", "visible": "show", "hidden": "hide", "alpha-lock": "lock"})
    a = parse(args, sp)
    p = ctx.project
    rep = Report()
    if not a.get("name") or str(a["name"]).lower() in ("list", "ls"):
        for e in p.state["layers"]:
            rep.add(f"  {e['name']:<7} {e['mode']:<8} opacity {e['opacity']:.2f}"
                    f"{'' if e.get('visible', True) else '  hidden'}{'  clip' if e.get('clip') else ''}"
                    f"{'  alpha-lock' if e.get('alpha_lock') else ''}{'  empty' if e.get('empty') else ''}")
        rep.next = "easel look --show layers"
        return rep
    e = p.layer_entry(a["name"])
    step = p.begin_step([])
    e = p.layer_entry(a["name"])
    changes = []
    if a.get("mode"):
        m = str(a["mode"]).lower()
        if m not in compose.MODES:
            p.abort_step(step)
            raise UsageError(f"unknown mode {m!r}; one of {', '.join(compose.MODES)}", USAGE["layer"])
        e["mode"] = m
        changes.append(f"mode {m}")
    if a.get("opacity"):
        e["opacity"] = coords.parse_float(a["opacity"], "opacity", 0, 1, ctx.notes)
        changes.append(f"opacity {e['opacity']:.2f}")
    for flag, key, val in (("hide", "visible", False), ("show", "visible", True), ("clip", "clip", True),
                           ("no-clip", "clip", False), ("lock", "alpha_lock", True), ("unlock", "alpha_lock", False)):
        if a.get(flag):
            e[key] = val
            if e["name"] == "sketch" and key == "visible":
                e["force_visible"] = val
            changes.append(flag)
    if not changes:
        p.abort_step(step)
        raise UsageError("say what to change: --mode --opacity --hide --show --clip --no-clip --lock --unlock",
                         USAGE["layer"])
    p.end_step(step, {"cmd": ctx.cmdline})
    rep.mutated = True
    rep.add(f"step {step}  layer {e['name']}: {', '.join(changes)}")
    if not ctx.batch:
        _show(ctx, rep, view.Panel(p.composite_u8("view"), f"CANVAS AFTER STEP {step}"), (0, 0, p.W, p.H),
              f"step-{step:04d}", "the canvas")
    rep.next = _finish_next(ctx)
    return rep


def cmd_export(ctx: Ctx, args: List[str]) -> Report:
    from . import export
    return export.cmd_export(ctx, args)


def cmd_fx(ctx: Ctx, args: List[str]) -> Report:
    from . import fx
    rep = fx.cmd_fx(ctx, args)
    if args and str(args[0]).lower() == "grain" and rep.mutated and ctx.project.state.get("stage") == "finish":
        p = ctx.project
        if workflow.tick(p.state, "finish", None, key="grain"):
            p.save_state()
    return rep


def cmd_text(ctx: Ctx, args: List[str]) -> Report:
    from . import text
    return text.cmd_text(ctx, args)


def cmd_fonts(ctx: Ctx, args: List[str]) -> Report:
    from . import text
    return text.cmd_fonts(ctx, args)


# ---------------------------------------------------------------------------------------- do (batch)

def split_batch(text: str) -> List[str]:
    out, cur, q = [], [], None
    for ch in text:
        if q:
            cur.append(ch)
            if ch == q:
                q = None
        elif ch in "\"'":
            q = ch
            cur.append(ch)
        elif ch in ";\n":
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur).strip())
    return [c for c in out if c]


def cmd_do(ctx: Ctx, args: List[str]) -> Report:
    if not args:
        raise UsageError("do needs commands separated by ;", USAGE["do"])
    if any(";" in x for x in args) or len(args) == 1:
        text = " ".join(args)
    elif all(" " in x for x in args):
        text = " ; ".join(args)          # easel do "ink hair" "ink face"
    else:
        text = " ".join(args)            # one unquoted command
    cmds = split_batch(text)
    p = ctx.project
    start_step = p.state["step"]
    rep = Report()
    touched = None
    last_next = ""
    worse: List[Tuple[int, str, str]] = []
    painted: List[Tuple[int, str]] = []
    for i, c in enumerate(cmds, 1):
        try:
            toks = shlex.split(c, posix=True)
        except ValueError as e:
            raise UsageError(f"command {i} {c!r}: {e}", USAGE["do"])
        if toks and toks[0] in ("easel", "python"):
            toks = toks[1:] if toks[0] == "easel" else toks[3:]
        if not toks:
            continue
        if toks[0] == "do":
            raise UsageError("do cannot contain do", USAGE["do"])
        sub = Ctx(ctx.project_arg, batch=True)
        sub._project = p
        sub.cmdline = "easel " + " ".join(toks)
        try:
            r = dispatch(sub, toks)
        except (UsageError, EngineMissing, manual.MarkRuleError) as e:
            rep.add(f"[{i}] {c}")
            rep.add(f"    FAILED: {getattr(e, 'message', str(e))}")
            ex = getattr(e, "example", "")
            if ex:
                rep.add(f"    EXAMPLE: {ex}")
            rest = cmds[i:]
            if rest:
                rep.add(f"    not executed: {'; '.join(rest)}")
            ctx.exit_code = 2 if isinstance(e, UsageError) else 1  # type: ignore[attr-defined]
            break
        except Exception as e:  # noqa: BLE001
            log = _log_error(sub, toks)
            rep.add(f"[{i}] {c}")
            rep.add(f"    FAILED (internal): {type(e).__name__}: {str(e).splitlines()[0][:120] if str(e) else ''}"
                    + (f" (details in {log})" if log else ""))
            if cmds[i:]:
                rep.add(f"    not executed: {'; '.join(cmds[i:])}")
            ctx.exit_code = 1  # type: ignore[attr-defined]
            p.reload()
            break
        ctx.notes.extend(sub.notes)
        rep.mutated |= r.mutated
        body = [ln for ln in r.lines if not ln.startswith("  ") and not ln.startswith("worst spots")]
        head = body[0].split("  (layer")[0] if body else c
        rep.add(f"[{i}] {head}" + (f":  {r.summary}" if r.summary else "") + ("   WORSE" if r.worse else ""))
        keep = [ln for ln in r.lines[1:] if ln.startswith(("FIX", "CHECK", "WARNING"))]
        if i == len(cmds):          # the last command's own report in full: its numbers are what comes next
            keep = [ln for ln in r.lines[1:] if not ln.startswith("worst spots") and
                    not re.match(r"\s+\d\) \d", ln)]
        for ln in keep[:12]:
            rep.add("    " + ln.strip())
        if r.worse and r.step is not None:
            worse.append((r.step, c, r.worse))
        if r.step is not None:
            painted.append((r.step, c))
        touched = coords.union_box(touched, r.touched)
        last_next = r.next
        p.reload()
    sketching = p.state["stage"] == "sketch" and p.ref is not None and any(
        str(c).strip().startswith(("sketch", "easel sketch")) for c in cmds)
    if p.state["step"] != start_step or touched:
        area = _panel_box(p, touched) if touched else (0, 0, p.W, p.H)
        after = p.composite_u8("view")
        cmp = _comparison(p, p.composite_u8("score"))
        emph = KEY_METRIC.get(p.state["stage"])
        spots = []
        if cmp and p.state["stage"] not in ("plan", "sketch"):
            m = regions.box_mask(p.H, p.W, area) if touched else None
            spots = _focus_spots(cmp.worst(m, n=5, emphasis=emph), emph)[:3]
            rep.lines.extend(_spots_lines(spots, emphasis=emph))
        regioning = p.state["regions"] and all(str(c).strip().split()[0] in ("region", "regions") or
                                               str(c).strip().startswith("easel region") for c in cmds)
        if regioning:
            img = view.render([_region_map_panel(p, "REGIONS (tinted, name f<focus>)")], (0, 0, p.W, p.H),
                              max_side=view.MAP_SIDE)
            rep.view = _save_view(ctx, img, f"step-{p.state['step']:04d}")
            rep.view_note = "every region tinted in its own colour; check each covers its part and nothing else"
        elif sketching:
            _show(ctx, rep, _sketch_panel(p), area, f"step-{p.state['step']:04d}",
                  "your sketch in blue over the faded reference", with_ref=False)
        else:
            ap = view.Panel(after, f"CANVAS AFTER STEP {p.state['step']}")
            ap.spots = [(s["box_px"], str(k)) for k, s in enumerate(spots, 1)]
            _show(ctx, rep, ap, area, f"step-{p.state['step']:04d}",
                  f"the canvas after steps {start_step + 1}-{p.state['step']}")
    rep.next = last_next or _finish_next(ctx)
    if worse:
        first = min(s for s, _, _ in worse)
        n_undo = max(1, int(p.state["step"]) - first + 1)     # steps of one batch are consecutive
        good_after = [c for s, c in painted if s > first and all(s != w for w, _, _ in worse)]
        rep.add(f"WARNING: {len(worse)} step(s) made the picture worse: "
                + "; ".join(f"{c} ({w})" for _, c, w in worse))
        rep.next = f"easel undo {n_undo}   (removes the last {n_undo} steps, back to before step {first})" + \
            (f"; then run the good ones again:  easel do \"{'; '.join(good_after)}\"" if good_after else "") + \
            "; then try other settings or hand tools there"
    if getattr(ctx, "exit_code", 0):
        rep.next = "fix the failed command and run it and the ones after it again"
    return rep


# ============================================================================================ dispatch / main

COMMANDS: Dict[str, Callable] = {
    "new": cmd_new, "style": cmd_style, "status": cmd_status, "next": cmd_next, "todo": cmd_todo,
    "stage": cmd_stage, "undo": cmd_undo, "redo": cmd_redo, "look": cmd_look, "view": cmd_look,
    "marks": cmd_marks, "pick": cmd_pick, "palette": cmd_palette, "score": cmd_score, "region": cmd_region,
    "regions": lambda c, a: cmd_region(c, a or ["list"]),
    "sketch": cmd_sketch, "ink": cmd_ink, "line": cmd_ink, "flat": cmd_flat, "shade": cmd_shade,
    "paint": cmd_paint, "detail": cmd_detail, "airbrush": cmd_airbrush, "retouch": cmd_retouch, "dot": cmd_dot, "stipple": cmd_dot,
    "sharpen": cmd_sharpen, "stroke": cmd_stroke, "fill": cmd_fill, "shape": cmd_shape,
    "gradient": cmd_gradient, "erase": cmd_erase, "smooth": cmd_smooth, "adjust": cmd_adjust,
    "layer": cmd_layer, "layers": lambda c, a: cmd_layer(c, a or ["list"]), "export": cmd_export, "do": cmd_do,
    "fx": cmd_fx, "text": cmd_text, "fonts": cmd_fonts, "help": cmd_help,
}

_DRAW_INSTALLED = False


def _unmet_words(u: dict) -> str:
    """One unmet gate item: its own sentence when it has one (the draw gates' holes), else metric < need."""
    if u.get("text"):
        return str(u["text"])
    return f"{u['region'] + ' ' if u.get('region') else ''}{u['metric']} {u['value']} < {u['need']}"


def _install_draw() -> None:
    """Drawing without a reference picture (ORIGINAL's modules): routes commands for such projects only.  When
    cli_draw is imported first it is still half-built while it imports this module; it is installed on the first
    command then."""
    global _DRAW_INSTALLED
    if _DRAW_INSTALLED:
        return
    try:
        from . import cli_draw as _cd
        if not hasattr(_cd, "install"):
            return
        _cd.install(COMMANDS)
    except Exception as _e:  # noqa: BLE001
        sys.stderr.write(f"easel: the no-reference commands failed to load ({type(_e).__name__}: {_e})\n")
    _DRAW_INSTALLED = True


_install_draw()


def _install_change() -> None:
    """A redraw that differs from its reference (cli_change): change, recolor, region refine, look --show changes."""
    try:
        from . import cli_change as _cc
        _cc.install(COMMANDS)
    except Exception as _e:  # noqa: BLE001
        sys.stderr.write(f"easel: the change commands failed to load ({type(_e).__name__}: {_e})\n")


_install_change()


def _log_seconds(ctx, seconds: float, code: int) -> None:
    """Seconds per command, kept beside the project (outside the undo history), for the session record."""
    p = getattr(ctx, "_project", None)
    if p is None:
        return
    try:
        import json as _json
        with open(p.root / "timing.jsonl", "a", encoding="utf-8") as f:
            f.write(_json.dumps({"cmd": ctx.cmdline, "s": round(float(seconds), 2), "code": int(code)}) + chr(10))
    except Exception:  # noqa: BLE001
        pass


def _wrap_mode() -> None:
    """`easel mode hand` right after `easel new --ref`: hand mode draws on the 1x canvas (the 2x stroke canvas is
    trace mode's); the fresh project is returned to 1x and its marks made again before the switch."""
    orig = COMMANDS.get("mode")
    if orig is None or getattr(orig, "_easel_wrapped", False):
        return

    def mode(ctx, args):
        try:
            p = ctx.project
            fresh = all(e.get("empty", True) for e in p.state.get("layers", []) if e["name"] != "paper")
            if args and str(args[0]).lower() == "hand" and p.scale > 1 and fresh and p.ref is not None:
                p.set_work_scale(1)
                p.state["fine"] = "plan"
                p.save_state()
                if p.state.get("marks"):
                    _make_marks(ctx, int(p.state["marks"].get("n", 40)))
        except UsageError:
            raise
        except Exception:  # noqa: BLE001
            pass
        return orig(ctx, args)
    mode._easel_wrapped = True
    COMMANDS["mode"] = mode


_wrap_mode()


def dispatch(ctx: Ctx, toks: List[str]) -> Report:
    import difflib
    name = toks[0].lower().lstrip("-")
    fn = COMMANDS.get(name)
    if fn is None:
        close = difflib.get_close_matches(name, list(COMMANDS), n=1, cutoff=0.5)
        raise UsageError(f"unknown command {toks[0]!r}" + (f" (did you mean {close[0]}?)" if close else "") +
                         "; see  easel help", USAGE.get(close[0], "easel next") if close else "easel next")
    return fn(ctx, toks[1:])


def _setup_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            if not stream.isatty():
                stream.reconfigure(encoding="utf-8", errors="replace")
            else:
                stream.reconfigure(errors="replace")
        except (AttributeError, ValueError, io.UnsupportedOperation):
            pass


def _pop_project(argv: List[str]) -> Tuple[Optional[str], List[str]]:
    out, proj, i = [], None, 0
    while i < len(argv):
        t = argv[i]
        if t in ("--project", "-p") and i + 1 < len(argv):
            proj = argv[i + 1]
            i += 2
            continue
        if t.startswith("--project="):
            proj = t.split("=", 1)[1]
            i += 1
            continue
        out.append(t)
        i += 1
    return proj, out


def run(argv: List[str]) -> Tuple[int, str]:
    """Run one command line; returns (exit code, report text). Never raises."""
    _install_draw()
    _install_change()
    _wrap_mode()
    proj, argv = _pop_project(list(argv))
    ctx = Ctx(proj)
    ctx.cmdline = "easel " + " ".join(shlex.quote(a) if (" " in a or not a) else a for a in argv)
    ctx.exit_code = 0  # type: ignore[attr-defined]
    if not argv:
        argv = ["next"] if _has_project(proj) else ["help"]
    if len(argv) > 1 and any(t in ("--help", "-h", "-?", "help") for t in argv[1:]) and argv[0] != "text":
        argv = ["help", argv[0]]
    try:
        t_cmd = time.time()
        rep = dispatch(ctx, argv)
        code = getattr(ctx, "exit_code", 0)
        _log_seconds(ctx, time.time() - t_cmd, code)
        lines = rep.text().split("\n")
        if ctx.notes:
            notes = "note: " + "; ".join(dict.fromkeys(ctx.notes))
            lines.insert(1 if lines else 0, notes if len(notes) < 300 else notes[:297] + "...")
        return code, "\n".join(lines)
    except UsageError as e:
        example = _full_example(e.example, argv)
        out = [f"ERROR: {e.message}"]
        if ctx.notes:
            out.append("note: " + "; ".join(dict.fromkeys(ctx.notes)))
        out.append(f"EXAMPLE: {example or 'easel help'}")
        out.append(f"NEXT: {example or 'easel next'}")
        return 2, "\n".join(out)
    except EngineMissing as e:
        return 1, f"ERROR: {e}\nNEXT: use the hand tools meanwhile (easel stroke / fill / gradient), or  easel status"
    except manual.MarkRuleError as e:
        return 1, (f"ERROR: {e}; nothing was painted\n"
                   "NEXT: try other settings (a bigger --size), or draw it by hand:  easel stroke / fill")
    except KeyboardInterrupt:
        return 130, "ERROR: interrupted\nNEXT: easel status"
    except Exception as e:  # noqa: BLE001
        log = _log_error(ctx, argv)
        msg = str(e).splitlines()[0] if str(e) else ""
        return 1, (f"ERROR: internal problem in '{argv[0]}': {type(e).__name__}: {msg[:160]}"
                   + (f" (details in {log})" if log else "")
                   + "\nNEXT: easel status   (the project was left as it was before this command)")


def _full_example(example: str, argv: List[str]) -> str:
    """Turn an option-only example (``--color #c04030``) into the user's own command with that option fixed."""
    if not example or example.startswith("easel") or not example.startswith("--") or not argv:
        return example
    parts = example.split()
    opt, val = parts[0], parts[1:]
    out, i, done = [], 0, False
    while i < len(argv):
        t = argv[i]
        if t.split("=", 1)[0] == opt:
            out.extend([opt, *val])
            done = True
            i += 1 if "=" in t else 2
            continue
        out.append(t)
        i += 1
    if not done:
        out.extend([opt, *val])
    return "easel " + " ".join(shlex.quote(x) if (" " in x or not x) else x for x in out)


def _has_project(proj) -> bool:
    try:
        Project.find(proj)
        return True
    except Exception:  # noqa: BLE001
        return False


def _log_error(ctx: Ctx, argv) -> str:
    try:
        if ctx._project is None:
            try:
                ctx._project = Project.find(ctx.project_arg)
            except Exception:  # noqa: BLE001
                pass
        root = ctx._project.root if ctx._project is not None else Path(os.getcwd())
        path = root / "errors.log"
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(argv)}\n{traceback.format_exc()}\n")
        return Path(os.path.relpath(path)).as_posix() if not str(path).startswith("\\\\") else str(path)
    except Exception:  # noqa: BLE001
        return ""


def main(argv: Optional[List[str]] = None) -> int:
    _setup_stdout()
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] in ("--version", "-V", "version"):
        try:
            from importlib.metadata import version
            v = version("pixelcanvas")
        except Exception:  # noqa: BLE001
            v = "dev"
        print(f"easel (pixelcanvas) {v}")
        return 0
    code, text = run(args)
    print(text)
    sys.stdout.flush()
    return code
