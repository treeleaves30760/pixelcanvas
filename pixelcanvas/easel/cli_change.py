"""Commands for a redraw that differs from its reference: ``change`` (state the request's changes), ``recolor``,
``region refine`` and ``look --show changes``. Hooked into the command table by ``install``; a project without
stated changes never reaches them."""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np

from . import change as chg
from . import change_ops as co
from . import coords, view, workflow
from .coords import UsageError

USAGE = {
    "change": 'easel change add recolor hair --to "#5fbf6a"',
    "recolor": 'easel recolor hair --to "#5fbf6a"',
    "refine": "easel region refine hair --by hue",
}
KIND_EXAMPLES = {"recolor": 'easel change add recolor hair --to "#5fbf6a"'}


def _cli():
    from . import cli
    return cli


# ----------------------------------------------------------------------------------------------- shared

def _region_name(p, name: Optional[str], example: str) -> str:
    cli = _cli()
    if not name:
        raise UsageError("say which region", example)
    r = cli.resolve_region(p, name, [], example)
    if r.name not in p.state["regions"]:
        raise UsageError(f"{name} is not a named region; changes work on the plan's regions: "
                         f"{', '.join(p.region_names()) or 'none yet'}", example)
    return r.name


def _changes(st: dict) -> dict:
    c = st.get("changes")
    if not c:
        c = {"decided": False, "none": False, "items": []}
        st["changes"] = c
    c.setdefault("items", [])
    return c


def _renumber(c: dict) -> None:
    for i, it in enumerate(c["items"], 1):
        it["id"] = i


def _recolor_spec():
    cli = _cli()
    return cli.Spec(USAGE["recolor"], ["region"], {"to": "value", "chroma": "value", "light": "value",
                                                     "band": "value"},
                    alias={"color": "to", "colour": "to"}, min_pos=1)


def _recolor_params(p, a: dict, notes: list) -> dict:
    cli = _cli()
    if not a.get("to"):
        raise UsageError("recolor needs the new colour: --to \"#rrggbb\"", USAGE["recolor"])
    rgb = coords.parse_colour(a["to"], USAGE["recolor"])
    return {"to": coords.hex_colour(rgb), "rgb": list(rgb),
            "chroma": cli._num(a, "chroma", 1.0, 0.2, 3.0, notes, USAGE["recolor"]),
            "light": cli._num(a, "light", 0.0, 0.0, 1.0, notes, USAGE["recolor"]),
            "band": cli._num(a, "band", None, 10.0, 90.0, notes, USAGE["recolor"])}


def _cmdline(kind: str, region: str, par: dict) -> str:
    if kind == "recolor":
        extra = "".join(f" --{k} {par[k]:g}" for k, d in (("chroma", 1.0), ("light", 0.0), ("band", None))
                        if par.get(k) is not None and (d is None or abs(float(par[k]) - d) > 1e-9))
        return f'easel recolor {region} --to "{par["to"]}"{extra}'
    return f"easel {kind} {region}"


def _find(c: dict, kind: str, region: Optional[str]) -> Optional[dict]:
    for it in c["items"]:
        if it["kind"] == kind and it.get("region") == region:
            return it
    return None


def _target_step(ctx, extra_files: List[str] = ()):
    """Begin one undoable step that may rewrite the target (and, late in the work, the painted layers)."""
    p = ctx.project
    files = list(chg.TARGET_FILES) + list(extra_files)
    if chg.late(p.state):
        files += chg.late_files(p)
    return p.begin_step(files)


def _apply_target(ctx, rep) -> List[List[str]]:
    """Rebuild the target from the stated changes; handle a change made after tracing began."""
    p = ctx.project
    old, new = chg.rebuild(p)
    changed = float((co.delta_e(old, new) > 3.0).mean()) if old.shape == new.shape else 1.0
    stale_new = chg.mark_stale(p, old, new)
    workflow.build_todo(p.state)
    rep.add(f"target: {100 * changed:.1f}% of the picture changed; the stages trace it and every score now "
            "measures against it (the original stays as orig.png)")
    if stale_new:
        by_stage: Dict[str, List[str]] = {}
        for s, r in stale_new:
            by_stage.setdefault(s, []).append(r)
        rep.add("made after tracing began: the changed pixels were erased from the painted layers; redraw "
                + "; ".join(f"{s}: {', '.join(rs)}" for s, rs in by_stage.items()))
    return stale_new


def _change_view(ctx, rep, its: List[dict], step: int, tag: str) -> None:
    """ORIGINAL | TARGET over the changed regions, stray patches boxed, regions outlined."""
    cli = _cli()
    p = ctx.project
    if ctx.batch:
        return
    orig = chg.orig_work(p)
    tgt = np.asarray(p.ref)
    s = p.scale
    union = np.zeros((p.H, p.W), bool)
    spots = []
    outlines = []
    for it in its:
        if it.get("region") and it["region"] in p.state["regions"]:
            m = p.region_mask(it["region"])
            union |= m
            outlines.append((m, it["region"], (0, 200, 255)))
        for k, b in enumerate((it.get("check") or {}).get("stray_boxes") or [], 1):
            x0, y0, x1, y1 = (int(v * s) for v in b)
            spots.append(((max(0, x0 - 3), max(0, y0 - 3), min(p.W, x1 + 3), min(p.H, y1 + 3)),
                          f"{it['id']}.{k}"))
    box = coords.mask_bbox_px(union) or (0, 0, p.W, p.H)
    area = cli._panel_box(p, box)
    a = view.Panel(orig, "ORIGINAL")
    b = view.Panel(tgt, "TARGET (what the stages trace)")
    for pan in (a, b):
        pan.outlines = list(outlines)
        pan.spots = list(spots[:9])
    img = view.render([a, b], area)
    rep.view = cli._save_view(ctx, img, f"{tag}-{step:04d}")
    rep.view_note = ("the original and the target side by side, changed regions outlined"
                     + (", stray patches boxed (they keep the old colour)" if spots else ""))


# ----------------------------------------------------------------------------------------------- change

def cmd_change(ctx, args: List[str]):
    cli = _cli()
    p = ctx.project
    st = p.state
    rep = cli.Report()
    sub = args[0].lower() if args else "list"
    rest = args[1:]
    if p.ref is None:
        raise UsageError("changes edit a reference picture and this project has none", "easel new pic --ref ref.jpg")
    if sub in ("list", "ls", "show"):
        c = st.get("changes") or {}
        if c.get("none"):
            rep.add("no changes: the request asks for a plain redraw")
        elif not c.get("items"):
            rep.add("no changes stated yet")
        for it in c.get("items") or []:
            ok = bool(it.get("check")) and not chg.item_unmet(it)
            rep.add(f"  [{'x' if ok else ' '}] {it['id']}  {describe_cmd(it)}")
            if it.get("check"):
                for ln in chg.check_lines(it):
                    rep.add("      " + ln)
        rep.next = cli._finish_next(ctx)
        return rep
    if sub == "none":
        c = _changes(st)
        if c["items"]:
            raise UsageError(f"{len(c['items'])} change(s) are stated; drop them first", "easel change drop 1")
        # not an undo step: the step numbers (which seed the brushes) stay those of a plain redraw
        c["decided"], c["none"] = True, True
        p.save_state()
        p.journal({"cmd": ctx.cmdline, "changes": "none"})
        rep.mutated = True
        rep.add("no changes: the picture is a plain redraw of the reference")
        rep.next = cli._finish_next(ctx)
        return rep
    if sub == "add":
        if not rest:
            raise UsageError(f"change add needs the kind and its arguments; kinds: {', '.join(chg.KINDS)}",
                             USAGE["change"])
        kind = rest[0].lower().replace("recolour", "recolor")
        if kind not in chg.KINDS:
            raise UsageError(f"unknown change kind {rest[0]!r}; kinds: {', '.join(chg.KINDS)}", USAGE["change"])
        if not p.state["regions"]:
            raise UsageError("define the plan's regions first: a change works on a region",
                             "easel region add hair --marks 3,4,9")
        a = cli.parse(rest[1:], _recolor_spec())
        region = _region_name(p, a.get("region"), KIND_EXAMPLES[kind])
        par = _recolor_params(p, a, ctx.notes)
        c = _changes(st)
        step = p.begin_step([])
        it = _find(c, kind, region)
        cmd = _cmdline(kind, region, par)
        if it is None:
            it = {"id": len(c["items"]) + 1, "kind": kind, "region": region, "cmd": cmd, "params": None,
                  "check": None, "to": par["to"]}
            c["items"].append(it)
            word = "added"
        else:
            it["cmd"], it["to"] = cmd, par["to"]
            word = "replaced"
        c["decided"], c["none"] = True, False
        workflow.build_todo(st)
        p.end_step(step, {"cmd": ctx.cmdline})
        rep.mutated = True
        rep.add(f"step {step}  change {it['id']} {word}: {describe_cmd(it)}")
        rep.add("stages: " + " > ".join(workflow.plan_of(st)) + "   (the change stage makes it on the target the "
                "next stages trace)")
        rep.add(f"changes stated: {len(c['items'])}; add one per change the request asks for")
        rep.next = cli._finish_next(ctx)
        return rep
    if sub in ("drop", "remove", "rm", "delete"):
        if not rest or not re.fullmatch(r"#?\d+", rest[0]):
            raise UsageError("change drop needs the change number", "easel change drop 2")
        n = int(rest[0].lstrip("#"))
        c = _changes(st)
        it = next((x for x in c["items"] if x["id"] == n), None)
        if it is None:
            raise UsageError(f"there is no change {n}", "easel change list")
        step = _target_step(ctx)
        try:
            c["items"].remove(it)
            _renumber(c)
            if (p.root / "orig.png").exists():
                _apply_target(ctx, rep)
            workflow.build_todo(st)
            p.end_step(step, {"cmd": ctx.cmdline})
        except Exception:
            p.abort_step(step)
            raise
        rep.mutated = True
        rep.lines.insert(0, f"step {step}  change {n} dropped: {describe_cmd(it)}")
        rep.next = cli._finish_next(ctx)
        return rep
    raise UsageError(f"unknown change action {sub!r}; use add, none, drop, list", USAGE["change"])


def describe_cmd(it: dict) -> str:
    return it["cmd"].replace("easel ", "", 1)


# ----------------------------------------------------------------------------------------------- recolor

def cmd_recolor(ctx, args: List[str]):
    cli = _cli()
    p = ctx.project
    if p.ref is None:
        raise UsageError("recolor changes a reference picture and this project has none", USAGE["recolor"])
    a = cli.parse(args, _recolor_spec())
    region = _region_name(p, a.get("region"), USAGE["recolor"])
    par = _recolor_params(p, a, ctx.notes)
    st = p.state
    c = _changes(st)
    rep = cli.Report(mutated=True)
    step = _target_step(ctx)
    try:
        it = _find(c, "recolor", region)
        added = it is None
        if added:
            it = {"id": len(c["items"]) + 1, "kind": "recolor", "region": region, "check": None}
            c["items"].append(it)
        it["params"], it["to"] = par, par["to"]
        it["cmd"] = _cmdline("recolor", region, par)
        c["decided"], c["none"] = True, False
        before = dict(it.get("check") or {})
        _apply_target(ctx, rep)
        p.end_step(step, {"cmd": ctx.cmdline, "change": it["id"], "check": {k: v for k, v in it["check"].items()
                                                                          if k != "stray_boxes"}})
    except Exception:
        p.abort_step(step)
        raise
    head = f"step {step}  recolor {region} -> {par['to']}  (change {it['id']}" + (", added" if added else "") + ")"
    lines = [head] + chg.check_lines(it)
    if before.get("stray") is not None and before.get("stray") != it["check"]["stray"]:
        lines.append(f"stray patches {before['stray']} -> {it['check']['stray']}")
    rep.lines = lines + rep.lines
    um = chg.item_unmet(it)
    if not um:
        rep.add(f"change {it['id']} done: hue reached, shading kept, no stray patches")
    _change_view(ctx, rep, [it], step, "change")
    rep.next = um[0]["fix"] if um else cli._finish_next(ctx)
    return rep


# ----------------------------------------------------------------------------------------------- region refine

def region_refine(ctx, args: List[str]):
    cli = _cli()
    p = ctx.project
    a = cli.parse(args, cli.Spec(USAGE["refine"], ["name"], {"by": "value", "band": "value", "reach": "value"},
                                 min_pos=1))
    name = _region_name(p, a.get("name"), USAGE["refine"])
    by = str(a.get("by") or "hue").lower()
    if by not in ("hue", "lines", "colour", "color", "line"):
        raise UsageError(f"unknown --by {by!r}: hue (take the region's own colour that touches it) or lines (snap to "
                         "the areas its line art encloses)", USAGE["refine"])
    by = "hue" if by in ("colour", "color") else ("lines" if by == "line" else by)
    import cv2
    # measured and refined at the delivered size, where the change checks measure; the added / removed pixels are
    # then laid on the working-size region
    img_d = chg.orig_d(p) if (p.root / "orig.png").exists() else np.asarray(p.ref_deliver)
    img = chg.working(p, img_d)
    old = p.region_mask(name).copy()
    old_d = chg.region_d(p, name)
    reach_d, min_px = chg.sizes(p)
    band = cli._num(a, "band", None, 10.0, 90.0, ctx.notes, USAGE["refine"])
    reach = int(cli._num(a, "reach", reach_d, 1, 400, ctx.notes, USAGE["refine"]))
    focus = chg.protected(p, name, img_d, band)
    if by == "hue":
        new_d, info = co.refine_by_hue(img_d, old_d, band=band, reach=reach, others=focus)
    else:
        new_d, info = co.refine_by_lines(img_d, old_d)
    up = (lambda m: cv2.resize(m.astype(np.uint8), (p.W, p.H), interpolation=cv2.INTER_NEAREST) > 0) \
        if p.scale > 1 else (lambda m: m)
    new = (old & ~up(old_d & ~new_d)) | up(new_d & ~old_d)
    s_info = co.colour_info(img_d, old_d, band)
    dom = {"grey": s_info["grey"]}
    sb = co.stray_patches(img_d, old_d, s_info, reach, min_px, focus)
    sa = co.stray_patches(img_d, new_d, s_info, reach, min_px, focus)
    sa["boxes"] = [tuple(int(v * p.scale) for v in b) for b in sa["boxes"]]
    uses = [it for it in (p.state.get("changes") or {}).get("items") or [] if it.get("region") == name
            and it.get("params")]
    rep = cli.Report(mutated=True)
    files = [f"regions/{name}.png"]
    step = _target_step(ctx, files) if uses else p.begin_step(files)
    try:
        p.save_region_mask(name, new)
        e = p.state["regions"][name]
        e["area"] = round(float(new.mean()), 4)
        e["box"] = list(coords.px_box_to_units(coords.mask_bbox_px(new) or (0, 0, p.W, p.H), p.W, p.H))
        e["how"] = (e.get("how") or "") + f" refined by {by}"
        if uses:
            _apply_target(ctx, rep)
        workflow.build_todo(p.state)
        p.end_step(step, {"cmd": ctx.cmdline, "region": name})
    except Exception:
        p.abort_step(step)
        raise
    lines = [f"step {step}  region {name} refined by {by}: {100 * old.mean():.1f}% -> {100 * new.mean():.1f}% of the "
             f"canvas (+{info['added']} px, -{info['removed']} px)"]
    if not dom["grey"]:
        lines.append(f"patches of {name}'s colour touching it but outside it: {sb['n']} -> {sa['n']}")
    for it in uses:
        lines += [f"change {it['id']} made again on the new region:"] + ["  " + ln for ln in chg.check_lines(it)]
    rep.lines = lines + rep.lines
    if not ctx.batch:
        pan = view.Panel(img, f"REGION {name}: before (cyan) and after (magenta)")
        pan.outlines = [(old, "before", (0, 200, 255)), (new, "after", (255, 0, 200))]
        pan.spots = [((x0, y0, x1, y1), str(k)) for k, (x0, y0, x1, y1) in enumerate(sa["boxes"][:6], 1)]
        area = cli._panel_box(p, coords.mask_bbox_px(new | old) or (0, 0, p.W, p.H))
        rep.view = cli._save_view(ctx, view.render([pan], area), f"refine-{step:04d}")
        rep.view_note = "the region's outline before and after over the original"
    um = [u for it in uses for u in chg.item_unmet(it)]
    rep.next = um[0]["fix"] if um else cli._finish_next(ctx)
    return rep


# ----------------------------------------------------------------------------------------------- look --show changes

def look_changes(ctx, args: List[str]):
    cli = _cli()
    p = ctx.project
    its = [it for it in (p.state.get("changes") or {}).get("items") or []]
    rep = cli.Report()
    if not its:
        rep.add("no changes stated")
        rep.next = cli._finish_next(ctx)
        return rep
    for it in its:
        rep.add(f"change {it['id']}: {describe_cmd(it)}")
        for ln in chg.check_lines(it):
            rep.add("  " + ln)
    _change_view(ctx, rep, [it for it in its if it.get("check")] or its, int(p.state.get("step", 0)), "changes")
    rep.next = cli._finish_next(ctx)
    return rep


# ----------------------------------------------------------------------------------------------- install

def install(commands: Dict) -> None:
    if commands.get("change") is cmd_change:
        return
    commands["change"] = cmd_change
    commands["changes"] = lambda c, a: cmd_change(c, a or ["list"])
    commands["recolor"] = cmd_recolor
    commands["recolour"] = cmd_recolor
    region = commands.get("region")
    look = commands.get("look")

    def region_cmd(ctx, args, _orig=region):
        if args and str(args[0]).lower() in ("refine", "snap"):
            return region_refine(ctx, args[1:])
        return _orig(ctx, args)

    def look_cmd(ctx, args, _orig=look):
        toks = [str(t).lower() for t in args]
        if "changes" in toks or "change" in toks or "--show=changes" in toks:
            return look_changes(ctx, args)
        return _orig(ctx, args)

    commands["region"] = region_cmd
    commands["look"] = look_cmd
    commands["view"] = look_cmd
    try:
        cli = _cli()
        for k, v in USAGE.items():
            if k != "refine":
                cli.USAGE.setdefault(k, v)
    except Exception:  # noqa: BLE001
        pass
