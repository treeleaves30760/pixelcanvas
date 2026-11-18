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
    "canvas": 'easel canvas --size 900x1200 --place 60,190,940,930 --color "#f3e9d8"',
    "graphic": 'easel graphic disc 300,280,820,700 --color "#e8644a"',
    "palette map": 'easel palette map --to "#1d2a44,#f3e9d8" --accent "jacket=#e8644a"',
    "cutout": "easel cutout figure --regions hair,face,coat",
    "background": "easel background --from lake.easel",
    "light": 'easel light --tint "#ffb070" --amount 0.2 --rim left',
    "shadow (setting)": "easel shadow figure --dir left",
}
KIND_EXAMPLES = {
    "recolor": 'easel change add recolor hair --to "#5fbf6a"',
    "palette": 'easel change add palette map --to "#1d2a44,#f3e9d8" --accent "jacket=#e8644a"',
    "canvas": 'easel change add canvas --size 900x1200 --place 60,190,940,930 --color "#f3e9d8"',
    "graphic": 'easel change add graphic disc 300,280,820,700 --color "#e8644a"',
    "text": 'easel change add text "SLOW SUMMER" --box 70,40,930,175 --font display --weight black',
    "shape": 'easel change add shape rect 22,18,978,982 --outline 4 --color "#1d2a44" --layer text',
    "cutout": "easel change add cutout figure --regions hair,face,coat",
    "background": "easel change add background",
    "light": "easel change add light",
    "shadow": "easel change add shadow figure",
}


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
    edge = []
    for it in its:
        for k, b in enumerate((it.get("check") or {}).get("edge_boxes") or [], 1):
            x0, y0, x1, y1 = (int(v * s) for v in b)
            edge.append(((x0, y0, min(p.W, x1), min(p.H, y1)), f"edge {k}"))
        if edge:
            break
    box = coords.mask_bbox_px(union) or (0, 0, p.W, p.H)
    area = cli._panel_box(p, box) if not edge else (0, 0, p.W, p.H)
    a = view.Panel(orig, "ORIGINAL")
    b = view.Panel(tgt, "TARGET (what the stages trace)")
    for pan in (a, b):
        pan.outlines = list(outlines)
        pan.spots = list(spots[:9])
    b.spots += edge
    img = view.render([a, b], area)
    if edge:
        import cv2
        tiles = []
        for (x0, y0, x1, y1), lab in edge:
            crop = tgt[y0:y1, x0:x1]
            if crop.size:
                tiles.append(view.Panel(cv2.resize(crop, (300, 300), interpolation=cv2.INTER_CUBIC),
                                        f"{lab} (about 3x)"))
        if tiles:
            close = view.render(tiles, (0, 0, 300, 300))
            w = max(img.shape[1], close.shape[1])
            pad = lambda m: np.pad(m, ((0, 0), (0, w - m.shape[1]), (0, 0)), constant_values=255)  # noqa: E731
            img = np.concatenate([pad(img), pad(close)], axis=0)
    rep.view = cli._save_view(ctx, img, f"{tag}-{step:04d}")
    rep.view_note = ("the original and the target side by side, changed regions outlined"
                     + (", stray patches boxed (they keep the old colour)" if spots else "")
                     + ("; below: the three weakest places of the cut-out's edge, close up" if edge else ""))


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
            raise UsageError("define the plan's regions first: changes work on the plan's regions",
                             "easel region add hair --marks 3,4,9")
        key, region, par, cmd = parse_kind(ctx, kind, rest[1:])
        c = _changes(st)
        step = p.begin_step([])
        it = next((x for x in c["items"] if x["kind"] == kind and x.get("key", x.get("region")) == key), None)
        if it is None:
            it = {"id": len(c["items"]) + 1, "kind": kind, "key": key, "region": region, "cmd": cmd,
                  "params": par if chg.KINDS[kind]["stage"] == "finish" else None, "check": None,
                  "to": (par or {}).get("to")}
            c["items"].append(it)
            word = "added"
        else:
            it["cmd"], it["to"] = cmd, (par or {}).get("to")
            if chg.KINDS[kind]["stage"] == "finish":
                it["params"] = par
            word = "replaced"
        c["decided"], c["none"] = True, False
        workflow.build_todo(st)
        p.end_step(step, {"cmd": ctx.cmdline})
        rep.mutated = True
        where = "in the finish stage, over the traced picture" if chg.KINDS[kind]["stage"] == "finish" else \
            "in the change stage, on the target the next stages trace"
        rep.add(f"step {step}  change {it['id']} {word}: {describe_cmd(it)}  (made {where})")
        rep.add("stages: " + " > ".join(workflow.plan_of(st)))
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
    uses = [it for it in (p.state.get("changes") or {}).get("items") or [] if it.get("params") and
            (it.get("region") == name or name in [r for r, _c in (it["params"].get("accents") or [])])]
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
    if info.get("kept"):
        lines.append(f"the line art does not close around {name} (gaps join it to the areas beside it), so it stays "
                     f"as it was; to move its edge draw it:  easel region add {name} --poly \"x,y x,y ...\"")
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


# ----------------------------------------------------------------------------------------------- kinds: parsing

def _join(toks: List[str]) -> str:
    import shlex
    return "easel " + " ".join(shlex.quote(t) if (" " in t or not t or "#" in t or "=" in t) else t for t in toks)


def _palette_spec():
    cli = _cli()
    return cli.Spec(USAGE["palette map"], [], {"to": "value", "accent": "value", "gamma": "value"},
                    alias={"colors": "to", "colours": "to", "accents": "accent"})


def _palette_params(p, a: dict, notes: list) -> dict:
    cli = _cli()
    u = USAGE["palette map"]
    if not a.get("to"):
        raise UsageError("palette map needs its colours, dark to light: --to \"#1d2a44,#f3e9d8\"", u)
    parts = [x for x in re.split(r"[,;\s]+", str(a["to"]).strip()) if x]
    if len(parts) < 2:
        raise UsageError("palette map needs at least two colours, dark to light", u)
    rgbs = [list(coords.parse_colour(x, u)) for x in parts]
    accents = []
    if a.get("accent"):
        for part in re.split(r"[,;]+", str(a["accent"])):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise UsageError(f"--accent takes region=colour pairs, got {part!r}", u)
            r, col = part.split("=", 1)
            accents.append([_region_name(p, r.strip(), u), list(coords.parse_colour(col.strip(), u))])
    return {"rgbs": rgbs, "to": ",".join(coords.hex_colour(c) for c in rgbs), "accents": accents,
            "gamma": cli._num(a, "gamma", 1.1, 0.4, 3.0, notes, u)}


def _canvas_spec():
    cli = _cli()
    return cli.Spec(USAGE["canvas"], [], {"size": "value", "place": "value", "color": "value"},
                    alias={"colour": "color", "box": "place"})


def _canvas_params(p, a: dict, notes: list) -> dict:
    u = USAGE["canvas"]
    if not a.get("size") or not a.get("place"):
        raise UsageError("canvas needs the new size and where the reference goes in it: --size WxH --place "
                         "x0,y0,x1,y1 (0-1000 of the new canvas)", u)
    nums = [int(float(x)) for x in re.findall(r"\d+(?:\.\d+)?", str(a["size"]))]
    if len(nums) != 2:
        raise UsageError("--size needs WxH like 900x1200", u)
    box = coords.parse_box(a["place"], notes, u)
    col = coords.parse_colour(a["color"], u) if a.get("color") else tuple(p.state.get("paper") or (255, 255, 255))
    return {"size": nums, "place": list(box), "rgb": list(col)}


def _graphic_spec():
    cli = _cli()
    return cli.Spec(USAGE["graphic"], ["shape", "box"], {"color": "value", "width": "value", "front": "flag"},
                    alias={"colour": "color"}, min_pos=2)


def _graphic_params(p, a: dict, notes: list) -> dict:
    cli = _cli()
    u = USAGE["graphic"]
    shape = str(a["shape"]).lower()
    shape = {"circle": "disc", "oval": "disc", "ellipse": "disc", "rectangle": "rect", "box": "rect",
             "band": "rect", "outline": "frame"}.get(shape, shape)
    if shape not in ("disc", "rect", "ring", "frame"):
        raise UsageError(f"unknown graphic {a['shape']!r}; one of disc, rect, ring, frame", u)
    box = coords.parse_box(a["box"], notes, u)
    if not a.get("color"):
        raise UsageError("graphic needs its colour: --color \"#rrggbb\"", u)
    rgb = coords.parse_colour(a["color"], u)
    width = cli._num(a, "width", 6.0 if shape in ("ring", "frame") else 0.0, 0, 200, notes, u)
    return {"shape": shape, "box": list(box), "rgb": list(rgb), "width": width, "behind": not a.get("front")}


TEXT_OPTS = {"box": "value", "font": "value", "color": "value", "weight": "value", "align": "value",
             "outline": "value", "vertical": "flag", "fit": "flag", "size": "value"}
SHAPE_OPTS = {"color": "value", "outline": "value", "fill": "flag", "layer": "value", "opacity": "value",
              "brush": "value"}


def parse_kind(ctx, kind: str, toks: List[str]):
    """(key, region, params, command line) of a change stated with  easel change add <kind> ..."""
    cli = _cli()
    p = ctx.project
    if kind == "recolor":
        a = cli.parse(toks, _recolor_spec())
        region = _region_name(p, a.get("region"), KIND_EXAMPLES[kind])
        par = _recolor_params(p, a, ctx.notes)
        return region, region, par, _cmdline(kind, region, par)
    if kind == "palette":
        if not toks or toks[0].lower() != "map":
            raise UsageError("a palette change is a map:  change add palette map --to ...", KIND_EXAMPLES[kind])
        _palette_params(p, cli.parse(toks[1:], _palette_spec()), ctx.notes)
        return "palette", None, None, _join(["palette", *toks])
    if kind == "canvas":
        _canvas_params(p, cli.parse(toks, _canvas_spec()), ctx.notes)
        return "canvas", None, None, _join(["canvas", *toks])
    if kind == "graphic":
        _graphic_params(p, cli.parse(toks, _graphic_spec()), ctx.notes)
        cmd = _join(["graphic", *toks])
        return cmd, None, None, cmd
    if kind == "cutout":
        par = _cutout_params(p, cli.parse(toks, _cutout_spec()))
        return par["subject"], None, None, _cutout_cmd(par)
    if kind == "background":
        if not toks:
            return "background", None, None, chg.BG_FORMS
        a = cli.parse(toks, _background_spec())
        _background_params(p, a, ctx.notes)
        return "background", None, None, _join(["background", *toks])
    if kind == "light":
        _light_params(p, cli.parse(toks, _light_spec()), ctx.notes)
        return "light", None, None, _join(["light", *toks]) if toks else "easel light --rim auto"
    if kind == "shadow":
        _shadow_params(p, cli.parse(toks, _shadow_spec()), ctx.notes)
        return "shadow", None, None, _join(["shadow", *toks]) if toks else "easel shadow figure"
    if kind == "text":
        a = cli.parse(toks, cli.Spec(KIND_EXAMPLES[kind], ["string*"], TEXT_OPTS, alias={"colour": "color"}))
        if not a.get("string"):
            raise UsageError("a text change needs its words", KIND_EXAMPLES[kind])
        return str(a["string"]), None, {"text": str(a["string"])}, _join(["text", *toks])
    if kind == "shape":
        cli.parse(toks, cli.Spec(KIND_EXAMPLES[kind], ["kind", "coords*"], SHAPE_OPTS, alias={"colour": "color"},
                                 min_pos=2))
        cmd = _join(["shape", *toks])
        return cmd, None, {"tokens": list(toks)}, cmd
    raise UsageError(f"unknown change kind {kind!r}", USAGE["change"])


def _item_for(st: dict, kind: str, key, cmd: str, region=None) -> tuple:
    c = _changes(st)
    it = next((x for x in c["items"] if x["kind"] == kind and x.get("key", x.get("region")) == key), None)
    added = it is None
    if added:
        it = {"id": len(c["items"]) + 1, "kind": kind, "key": key, "region": region, "check": None}
        c["items"].append(it)
    it["cmd"] = cmd
    c["decided"], c["none"] = True, False
    return it, added


def _target_change(ctx, kind: str, key, cmd: str, par: dict, head: str, region=None, extra_files=(), pre=None):
    """Make (or remake) one target change: rebuild the target, report its check, show it."""
    cli = _cli()
    p = ctx.project
    rep = cli.Report(mutated=True)
    step = _target_step(ctx, list(extra_files))
    try:
        it, added = _item_for(p.state, kind, key, cmd, region)
        it["params"] = par
        if pre is not None:
            pre()
        _apply_target(ctx, rep)
        p.end_step(step, {"cmd": ctx.cmdline, "change": it["id"]})
    except Exception:
        p.abort_step(step)
        raise
    rep.lines = [f"step {step}  {head}  (change {it['id']}" + (", added" if added else "") + ")"] + \
        chg.check_lines(it) + rep.lines
    um = chg.item_unmet(it, p.state)
    if not um:
        rep.add(f"change {it['id']} done")
    _change_view(ctx, rep, [it], step, "change")
    rep.next = um[0]["fix"] if um else cli._finish_next(ctx)
    return rep


# ----------------------------------------------------------------------------------------------- palette map

def cmd_palette_map(ctx, args: List[str]):
    cli = _cli()
    p = ctx.project
    if p.ref is None:
        raise UsageError("palette map changes a reference picture and this project has none", USAGE["palette map"])
    par = _palette_params(p, cli.parse(args, _palette_spec()), ctx.notes)
    cmd = _join(["palette", "map", *args])
    return _target_change(ctx, "palette", "palette", cmd, par,
                          f"palette map {par['to']}" + "".join(f" + {r} {coords.hex_colour(c)}"
                                                               for r, c in par["accents"]))


# ----------------------------------------------------------------------------------------------- graphic

def cmd_graphic(ctx, args: List[str]):
    cli = _cli()
    p = ctx.project
    if p.ref is None:
        raise UsageError("graphic lays a shape on a reference picture's ground; this project has none (draw it "
                         "with  easel shape)", USAGE["graphic"])
    par = _graphic_params(p, cli.parse(args, _graphic_spec()), ctx.notes)
    cmd = _join(["graphic", *args])
    return _target_change(ctx, "graphic", cmd, cmd, par,
                          f"graphic {par['shape']} {coords.hex_colour(par['rgb'])}"
                          + (" behind the figure" if par["behind"] else " in front"))


# ----------------------------------------------------------------------------------------------- canvas

def cmd_canvas(ctx, args: List[str]):
    """The reference placed (scaled to fit, centred) in a new canvas of another size; margins in one colour. Made
    once, before tracing: the regions move with it, the marks are made again."""
    import cv2
    from .project import fit_size, save_png
    cli = _cli()
    p = ctx.project
    st = p.state
    u = USAGE["canvas"]
    if p.ref is None:
        raise UsageError("canvas places a reference picture; this project has none (easel new --size)", u)
    painted = [e["name"] for e in st["layers"] if e["name"] != "paper" and not e.get("empty", True)]
    if st.get("stage") not in ("plan", "change") or painted:
        raise UsageError("canvas changes the picture's size: make it before tracing (stage plan or change, nothing "
                         "painted yet)", "easel stage back change")
    if any(it["kind"] == "canvas" and it.get("params") for it in (st.get("changes") or {}).get("items") or []):
        raise UsageError("the canvas is already set; easel undo takes it back", "easel undo")
    par = _canvas_params(p, cli.parse(args, _canvas_spec()), ctx.notes)
    s = p.scale
    dw, dh = p.deliver_size()
    nw, nh = fit_size(int(par["size"][0]), int(par["size"][1]))
    bx = coords.box_to_px(par["place"], nw, nh)
    sc = min((bx[2] - bx[0]) / dw, (bx[3] - bx[1]) / dh)
    pw, ph = max(1, int(round(dw * sc))), max(1, int(round(dh * sc)))
    ox = int(round((bx[0] + bx[2]) / 2 - pw / 2))
    oy = int(round((bx[1] + bx[3]) / 2 - ph / 2))
    files = list(chg.TARGET_FILES) + [f"layers/{e['name']}.png" for e in st["layers"]] + \
        [f"regions/{n}.png" for n in p.region_names()]
    rep = cli.Report(mutated=True)
    step = p.begin_step(files)
    try:
        src = chg.orig_d(p)
        interp = cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC
        placed = cv2.resize(src, (pw, ph), interpolation=interp)
        new = np.empty((nh, nw, 3), np.uint8)
        new[:] = np.asarray(par["rgb"], np.uint8)
        x0, y0 = max(0, ox), max(0, oy)
        x1, y1 = min(nw, ox + pw), min(nh, oy + ph)
        new[y0:y1, x0:x1] = placed[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        save_png(p.root / "orig.png", new)
        W2, H2 = nw * s, nh * s
        inside = np.zeros((H2, W2), bool)
        inside[y0 * s:y1 * s, x0 * s:x1 * s] = True
        for n in p.region_names():
            m = p.region_mask(n)
            mm = cv2.resize(m.astype(np.uint8), (pw * s, ph * s), interpolation=cv2.INTER_NEAREST) > 0
            out = np.zeros((H2, W2), bool)
            out[y0 * s:y1 * s, x0 * s:x1 * s] = mm[(y0 - oy) * s:(y1 - oy) * s, (x0 - ox) * s:(x1 - ox) * s]
            if chg.is_rest(st["regions"][n]):
                out |= ~inside                                  # the new margins belong to the rest region
            p.save_region_mask(n, out)
            st["regions"][n]["area"] = round(float(out.mean()), 4)
            st["regions"][n]["box"] = list(coords.px_box_to_units(coords.mask_bbox_px(out) or (0, 0, W2, H2), W2, H2))
        st["width"], st["height"] = W2, H2
        st["deliver"] = [nw, nh] if s > 1 else None
        if s == 1:
            st.pop("deliver", None)
        st["marks"] = None
        paper = np.zeros((H2, W2, 4), np.float32)
        paper[..., :3] = np.asarray(st.get("paper") or (255, 255, 255), np.float32) / 255.0
        paper[..., 3] = 1.0
        p._layers.clear()
        for e in st["layers"]:
            p._write_layer_file(e["name"], paper if e["name"] == "paper" else None)
        it, added = _item_for(st, "canvas", "canvas", _join(["canvas", *args]))
        it["params"] = par
        it["check"] = {"size": f"{nw}x{nh}", "box": coords.fmt_box(coords.px_box_to_units((ox, oy, ox + pw, oy + ph),
                                                                                         nw, nh)),
                       "scale": round(sc, 3)}
        _apply_target(ctx, rep)
        p.end_step(step, {"cmd": ctx.cmdline, "change": it["id"]})
    except Exception:
        p.abort_step(step)
        raise
    rep.lines = [f"step {step}  canvas {nw}x{nh} (working {W2}x{H2}): the reference at scale {sc:.2f}, placed at "
                 f"{it['check']['box']}  (change {it['id']})",
                 f"margins {coords.hex_colour(par['rgb'])}; the regions moved with the picture (the rest region "
                 "takes the margins); marks are made again with  easel marks"] + rep.lines
    _change_view(ctx, rep, [it], step, "canvas")
    rep.next = cli._finish_next(ctx)
    return rep


# ----------------------------------------------------------------------------------------------- the figure in a new setting

def _names(p, val, example: str) -> List[str]:
    out: List[str] = []
    for part in re.split(r"[,;\s]+", str(val or "").strip()):
        if part:
            out.append(_region_name(p, part, example))
    return list(dict.fromkeys(out))


def _cutout_spec():
    cli = _cli()
    return cli.Spec(USAGE["cutout"], ["subject"], {"regions": "value", "plus": "value", "minus": "value",
                                                   "ground": "value", "open-corners": "flag"},
                    alias={"region": "regions", "parts": "regions", "add": "plus", "subtract": "minus"})


def _cutout_params(p, a: dict) -> dict:
    u = USAGE["cutout"]
    if not a.get("regions"):
        raise UsageError("cutout needs the figure's regions: --regions hair,face,coat (the plan's regions that make "
                         "the figure; the rest of the canvas is its ground)", u)
    ground = str(a.get("ground") or "auto").lower()
    if ground not in ("auto", "white", "painted"):
        raise UsageError(f"--ground is auto, white or painted, not {ground!r}", u)
    return {"subject": regions_clean(a.get("subject") or "figure"), "regions": _names(p, a["regions"], u),
            "plus": _names(p, a.get("plus"), u), "minus": _names(p, a.get("minus"), u), "ground": ground,
            "corners": not a.get("open-corners")}


def regions_clean(name: str) -> str:
    from .regions import clean_name
    return clean_name(str(name))


def _cutout_cmd(par: dict) -> str:
    cmd = f"easel cutout {par['subject']} --regions {','.join(par['regions'])}"
    for k in ("plus", "minus"):
        if par.get(k):
            cmd += f" --{k} {','.join(par[k])}"
    if par.get("ground", "auto") != "auto":
        cmd += f" --ground {par['ground']}"
    if not par.get("corners", True):
        cmd += " --open-corners"
    return cmd


def cmd_cutout(ctx, args: List[str]):
    cli = _cli()
    p = ctx.project
    if p.ref is None:
        raise UsageError("cutout cuts a figure out of a reference picture; this project has none", USAGE["cutout"])
    par = _cutout_params(p, cli.parse(args, _cutout_spec()))
    st = p.state
    old = next((x for x in (st.get("changes") or {}).get("items") or [] if x["kind"] == "cutout"), None)
    key = old.get("key") if old else par["subject"]
    rep = _target_change(ctx, "cutout", key, _cutout_cmd(par), par,
                         f"cutout {par['subject']} from {', '.join(par['regions'])}")
    return rep


def _background_spec():
    cli = _cli()
    return cli.Spec(USAGE["background"], [], {"from": "value", "gradient": "value", "angle": "value",
                                              "color": "value"},
                    alias={"colour": "color", "file": "from", "project": "from", "picture": "from"})


def load_background(path: str):
    """The picture behind the figure: a project folder's finished picture (its export, else its layers at the
    delivered size) or a picture file."""
    from pathlib import Path
    from .project import Project, load_image_rgb
    u = USAGE["background"]
    f = Path(path)
    if not f.exists() and Path(path + ".easel").exists():
        f = Path(path + ".easel")
    if not f.exists():
        raise UsageError(f"no project or picture at {path}", u)
    if f.is_dir():
        if not (f / "state.json").exists():
            raise UsageError(f"{path} is a folder but not an easel project (no state.json)", u)
        q = Project(f)
        ex = q.state.get("exported")
        if ex and Path(ex).exists() and Path(ex).suffix.lower() == ".png":
            return load_image_rgb(Path(ex)), "project"
        return np.asarray(_cli().delivered(q, q.composite_u8("export"))), "project"
    return load_image_rgb(f), "file"


def _background_params(p, a: dict, notes: list):
    cli = _cli()
    u = USAGE["background"]
    forms = [k for k in ("from", "gradient", "color") if a.get(k)]
    if len(forms) != 1:
        raise UsageError("background takes one of --from <project folder or picture file>, --gradient \"#a,#b\", "
                         "--color \"#hex\"", u)
    if a.get("from"):
        img, kind = load_background(str(a["from"]))
        return {"source": kind, "from": str(a["from"])}, img
    if a.get("gradient"):
        parts = [x for x in re.split(r"[,;\s]+", str(a["gradient"]).strip()) if x]
        if len(parts) != 2:
            raise UsageError("--gradient takes two colours: \"#top,#bottom\"", u)
        return {"source": "gradient", "rgbs": [list(coords.parse_colour(x, u)) for x in parts],
                "angle": cli._num(a, "angle", 90.0, -360.0, 360.0, notes, u)}, None
    return {"source": "color", "rgbs": [list(coords.parse_colour(a["color"], u))]}, None


def cmd_background(ctx, args: List[str]):
    from .project import save_png
    cli = _cli()
    p = ctx.project
    if p.ref is None:
        raise UsageError("background goes behind a figure cut out of a reference picture; this project has none",
                         USAGE["background"])
    par, img = _background_params(p, cli.parse(args, _background_spec()), ctx.notes)
    if img is not None:
        dw, dh = p.deliver_size()
        k = min(1.0, 2.0 * max(dw, dh) / max(img.shape[:2]))
        if k < 1.0:
            import cv2
            img = cv2.resize(img, (int(img.shape[1] * k), int(img.shape[0] * k)), interpolation=cv2.INTER_AREA)
    cmd = _join(["background", *args])
    # the picture is kept in the project, so the target can be rebuilt without the source
    return _target_change(ctx, "background", "background", cmd, par,
                          f"background ({par['source']})",
                          pre=(lambda: save_png(p.root / "background_src.png", img)) if img is not None else None)


def _light_spec():
    cli = _cli()
    return cli.Spec(USAGE["light"], ["amount_pos"], {"tint": "value", "amount": "value", "rim": "value",
                                                  "rim-color": "value", "rim-strength": "value"},
                    alias={"color": "tint", "colour": "tint", "rim-colour": "rim-color", "key": "tint"})


def _light_params(p, a: dict, notes: list) -> dict:
    cli = _cli()
    u = USAGE["light"]
    if a.get("amount_pos") is not None and a.get("amount") is None:
        a["amount"] = a["amount_pos"]
    tint = str(a.get("tint") or "auto")
    rgb = None if tint.lower() == "auto" else list(coords.parse_colour(tint, u))
    rim = str(a.get("rim") or "auto").lower()
    if rim not in ("auto", "left", "right", "top", "none", "off"):
        raise UsageError(f"--rim is left, right, top, none or auto (the background's lighter side), not {rim!r}", u)
    return {"tint": coords.hex_colour(rgb) if rgb else "auto", "rgb": rgb,
            "amount": cli._num(a, "amount", 0.2, 0.0, 0.8, notes, u), "rim": rim,
            "rim_rgb": list(coords.parse_colour(a["rim-color"], u)) if a.get("rim-color") else None,
            "rim_strength": cli._num(a, "rim-strength", 0.45, 0.0, 1.0, notes, u)}


def cmd_light(ctx, args: List[str]):
    cli = _cli()
    p = ctx.project
    if p.ref is None:
        raise UsageError("light lights a figure cut out of a reference picture; this project has none (in a free "
                         "picture:  easel plan light 160)", USAGE["light"])
    par = _light_params(p, cli.parse(args, _light_spec()), ctx.notes)
    return _target_change(ctx, "light", "light", _join(["light", *args]), par,
                          f"light: tint {par['tint']} {par['amount']:.2f}, rim {par['rim']}")


def _shadow_spec():
    cli = _cli()
    return cli.Spec(USAGE["shadow (setting)"], ["subject"], {"ground": "value", "dir": "value",
                                                              "strength": "value"},
                    alias={"direction": "dir", "y": "ground"})


def _shadow_params(p, a: dict, notes: list) -> dict:
    cli = _cli()
    u = USAGE["shadow (setting)"]
    d = str(a.get("dir") or "auto").lower()
    if d not in ("auto", "left", "right", "back", "front"):
        raise UsageError(f"--dir is left, right, back, front or auto (away from the light), not {d!r}", u)
    g = a.get("ground")
    return {"subject": regions_clean(a.get("subject") or "figure"), "dir": d,
            "ground": None if g is None else cli._num(a, "ground", None, 0.0, 1000.0, notes, u),
            "strength": cli._num(a, "strength", 0.55, 0.05, 0.9, notes, u)}


def cmd_scene_shadow(ctx, args: List[str]):
    cli = _cli()
    p = ctx.project
    par = _shadow_params(p, cli.parse(args, _shadow_spec()), ctx.notes)
    return _target_change(ctx, "shadow", "shadow", _join(["shadow", *args]), par,
                          f"shadow under {par['subject']}, falling {par['dir']}")


def _scene_shadow_here(ctx, args: List[str]) -> bool:
    """easel shadow is the free-mode shading tool; on a reference project with a cut-out (or with --ground / --dir
    and no --from) it is the contact shadow."""
    try:
        p = ctx.project
    except Exception:  # noqa: BLE001
        return False
    if p.ref is None:
        return False
    toks = [str(t).lower() for t in args]
    if "--from" in toks or "--onto" in toks:
        return False
    its = (p.state.get("changes") or {}).get("items") or []
    return any(x["kind"] in ("cutout", "shadow") for x in its) or "--ground" in toks or "--dir" in toks


# ----------------------------------------------------------------------------------------------- finishing changes

def _tick_finish(ctx) -> None:
    p = getattr(ctx, "_project", None)
    if p is None or not (p.state.get("changes") or {}).get("items"):
        return
    try:
        workflow.build_todo(p.state)
        p.save_state()
    except Exception:  # noqa: BLE001
        pass


# ----------------------------------------------------------------------------------------------- install

def _install_shadow(commands: Dict) -> None:
    """easel shadow: the free-mode tool, and the contact shadow on a reference project with a cut-out."""
    cur = commands.get("shadow")
    if getattr(cur, "_scene", False):
        return

    def shadow_cmd(ctx, args, _orig=cur):
        if _orig is None or _scene_shadow_here(ctx, args):
            return cmd_scene_shadow(ctx, args)
        return _orig(ctx, args)
    for attr in ("_draw", "_wrapped", "_wrapped_by"):
        if hasattr(cur, attr):
            setattr(shadow_cmd, attr, getattr(cur, attr))
    shadow_cmd._scene = True  # type: ignore[attr-defined]
    commands["shadow"] = shadow_cmd


def install(commands: Dict) -> None:
    _install_shadow(commands)
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
    commands["canvas"] = cmd_canvas
    commands["graphic"] = cmd_graphic
    commands["cutout"] = cmd_cutout
    commands["background"] = cmd_background
    commands["light"] = cmd_light
    palette = commands.get("palette")
    text = commands.get("text")
    shape = commands.get("shape")

    def palette_cmd(ctx, args, _orig=palette):
        if args and str(args[0]).lower() == "map":
            return cmd_palette_map(ctx, args[1:])
        return _orig(ctx, args)

    def text_cmd(ctx, args, _orig=text):
        rep = _orig(ctx, args)
        _tick_finish(ctx)
        return rep

    def shape_cmd(ctx, args, _orig=shape):
        import shlex
        p = None
        try:
            p = ctx.project
        except Exception:  # noqa: BLE001
            pass
        if p is not None:
            mine = ["shape", *[str(a) for a in args]]
            for it in (p.state.get("changes") or {}).get("items") or []:
                if it["kind"] == "shape" and shlex.split(it["cmd"])[1:] == mine:
                    made = p.state.setdefault("finish_made", [])   # saved with the shape's own undo step
                    if it["cmd"] not in made:
                        made.append(it["cmd"])
        rep = _orig(ctx, args)
        _tick_finish(ctx)
        return rep

    commands["palette"] = palette_cmd
    commands["text"] = text_cmd
    commands["shape"] = shape_cmd
    try:
        cli = _cli()
        for k, v in USAGE.items():
            if k != "refine":
                cli.USAGE.setdefault(k, v)
    except Exception:  # noqa: BLE001
        pass
