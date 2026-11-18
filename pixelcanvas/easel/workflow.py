"""Styles, stage plans, the tool-generated todo list, gates computed from the scores, ``next`` (DESIGN.md section 7)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import coords, metrics

STYLES = ("cel", "ink", "paint", "photo", "poster")
PLANS = {
    "cel": ["plan", "sketch", "line", "flat", "shade", "detail", "finish"],
    "ink": ["plan", "sketch", "line", "flat", "detail", "finish"],
    "paint": ["plan", "sketch", "block-in", "forms", "detail", "finish"],
    "photo": ["plan", "sketch", "block-in", "forms", "detail", "finish"],
    "poster": ["plan", "sketch", "flat", "line", "detail", "finish"],
}


def _plans_from_pipeline() -> None:
    """The stage list of each style follows the scripted pipeline (its blend passes are todo items of the shade
    and finish stages, not stages of their own), so a style's plan never drops a stage the engine runs."""
    try:
        from .pipeline import STAGES as PIPE
    except Exception:  # noqa: BLE001
        return
    for style, stages in PIPE.items():
        names = [n for n, _l, _k, _p in stages if n != "blend"]
        if "finish" not in names:
            names.append("finish")
        PLANS[style] = ["plan"] + names


_plans_from_pipeline()
STYLE_WORDS = {
    "cel": "anime / cel: clean lines, flat or softly graded colour",
    "ink": "ink: monochrome line drawing, manga, screentone",
    "paint": "paint: brushwork without outlines (gouache, oil, digital painting)",
    "photo": "photo: photographs and photoreal renders (soft brushes, more passes)",
    "poster": "poster: flat graphic shapes and lettering",
}
# stage -> command, layer it paints on, what it does
STAGES = {
    "plan": {"cmd": None, "layer": None, "what": "look at the reference, choose the style, define regions"},
    "sketch": {"cmd": "sketch", "layer": "sketch", "what": "loose pencil construction lines per region"},
    "line": {"cmd": "ink", "layer": "line", "what": "line art per region"},
    "flat": {"cmd": "flat", "layer": "flat", "what": "flat base colours under the lines"},
    "shade": {"cmd": "shade", "layer": "shade", "what": "shadow and light shapes over the flats"},
    "block-in": {"cmd": "paint", "layer": "paint", "what": "big brush strokes: the main colour masses"},
    "forms": {"cmd": "paint", "layer": "paint", "what": "medium strokes that turn the masses into forms"},
    "detail": {"cmd": "detail", "layer": "detail", "what": "fine brush on the worst spots, focus regions first"},
    "finish": {"cmd": None, "layer": "detail",
               "what": "compare the whole picture, fix what remains, finishing touches (fx, text), export"},
    "change": {"cmd": None, "layer": None,
               "what": "make the request's changes on the target (the picture the next stages trace)"},
}
# command -> stage it belongs to (for ticking todo items); paint is resolved by --size
CMD_STAGE = {"sketch": "sketch", "ink": "line", "flat": "flat", "shade": "shade", "detail": "detail",
             "airbrush": "forms"}

GATES = {
    "line": {"lines": 50},
    "flat": {"coverage": 98, "colour": 70},
    "block-in": {"coverage": 95, "colour": 65},
    "shade": {"colour": 75, "structure": 40},
    "forms": {"colour": 75, "structure": 40},
    "detail": {"overall": 70, "overall_focus3": 75},
}
# A photograph's grain is finer than any brush: the scripted pipeline itself ends the forms stage near structure
# 12-17 and the detail stage near overall 70 on a grainy landscape photograph, so these gates are set where the engine's best
# result passes (a gate the engine cannot meet only makes the model chase it).
GATES_BY_STYLE = {
    "photo": {"forms": {"colour": 75, "structure": 10}, "detail": {"overall": 60, "overall_focus3": 65}},
}
MAX_TRIES = 3


def gate_for(stage: str, style: Optional[str] = None) -> dict:
    return GATES_BY_STYLE.get(style or "", {}).get(stage) or GATES.get(stage, {})


def uses_lines(state: dict) -> bool:
    """Painterly styles have no line art: their dark thin marks are texture, so the lines axis is left out."""
    return (state.get("style") or "paint") not in ("paint", "photo")


def paint_size(stage: str, W: int, H: int) -> int:
    long_side = max(W, H)
    if stage == "block-in":
        return max(8, int(round(long_side / 30)))
    if stage == "forms":
        return max(4, int(round(long_side / 75)))
    return 3


def stage_of_command(cmd: str, size: Optional[float], W: int, H: int, plan: List[str]) -> Optional[str]:
    if cmd == "paint":
        if "block-in" not in plan:
            return None
        if size is None:
            return "block-in"
        mid = (paint_size("block-in", W, H) + paint_size("forms", W, H)) / 2
        return "block-in" if size >= mid else "forms"
    return CMD_STAGE.get(cmd)


# ----------------------------------------------------------------------------------------------- style guess

def guess_style(ref: Optional[np.ndarray], request: str = "") -> Tuple[str, str]:
    """(style, one-line reason) from line density, colour count and texture (or words in the request)."""
    req = (request or "").lower()
    for words, st in ((("manga", "ink drawing", "line drawing", "monochrome"), "ink"),
                      (("anime", "cel", "cartoon", "comic"), "cel"),
                      (("poster", "logo", "flat design", "graphic"), "poster"),
                      (("photo", "photoreal", "realistic"), "photo"),
                      (("painting", "oil", "gouache", "watercolor", "watercolour", "landscape"), "paint")):
        if any(w in req for w in words):
            return st, f"the request says {next(w for w in words if w in req)!r}"
    if ref is None:
        return "paint", "no reference; painterly default"
    small = ref
    H, W = ref.shape[:2]
    if max(H, W) > 800:
        s = 800 / max(H, W)
        small = cv2.resize(ref, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    sat = float(hsv[..., 1].mean()) / 255.0
    lab = cv2.cvtColor(small.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    L = lab[..., 0]
    lines = metrics.line_map(L, max(small.shape[:2]))
    line_d = float(lines.mean())
    gx = cv2.Sobel(lab, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lab, cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt((gx ** 2 + gy ** 2).sum(2))
    flat = float((g < 4.0).mean())                 # share of near-flat pixels
    hp = L - cv2.GaussianBlur(L, (0, 0), 1.5)
    texture = float(np.sqrt((hp[~lines] ** 2).mean())) if (~lines).any() else 0.0
    q = (small // 32).reshape(-1, 3)
    keys, counts = np.unique(q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2], return_counts=True)
    counts = np.sort(counts)[::-1]
    ncol = int(np.searchsorted(np.cumsum(counts), 0.9 * counts.sum()) + 1)
    facts = f"lines {line_d:.1%}, flat areas {flat:.0%}, texture {texture:.1f}, colours {ncol}, saturation {sat:.2f}"
    try:        # the engine's own guess (it also tells photographs from paintings); same style as its pipeline
        from .pipeline import guess_style as _engine_guess
        g = _engine_guess(ref)
        if g in STYLES:
            words = {"ink": "monochrome with lines", "cel": "clear dark lines and smooth colour",
                     "poster": "flat shapes, few colours", "photo": "fine smooth grain of a photograph",
                     "paint": "soft edges and brush texture"}
            return g, f"{words[g]}; " + facts
    except Exception:  # noqa: BLE001
        pass
    if sat < 0.08 and line_d > 0.02 and flat > 0.3:
        return "ink", "monochrome with lines; " + facts
    if line_d > 0.025 and texture < 4.5 and flat > 0.45:
        return "cel", "clear dark lines and smooth colour; " + facts
    if line_d < 0.02 and flat > 0.6 and ncol <= 12:
        return "poster", "flat shapes, few colours; " + facts
    return "paint", "soft edges and brush texture; " + facts


# ----------------------------------------------------------------------------------------------- todo

def _ordered_regions(state: dict, stage: str) -> List[str]:
    regs = state["regions"]
    names = list(regs.keys())
    if stage in ("flat", "block-in", "shade", "forms"):
        return sorted(names, key=lambda n: (-regs[n].get("area", 0), n))
    return sorted(names, key=lambda n: (-regs[n].get("focus", 1), -regs[n].get("area", 0), n))


def _pipe_has(state: dict, stage: str, kind: str) -> bool:
    try:
        from .pipeline import STAGES as PIPE
    except Exception:  # noqa: BLE001
        return False
    return any(n == stage and k == kind for n, _l, k, _p in PIPE.get(state.get("style") or "paint", []))


def levels_available() -> bool:
    """The engine has detail levels 1 / 0 (stage_levels.retouch_ops, ops.Retouch, pipeline.level_passes)."""
    try:
        from .ops import Retouch  # noqa: F401
        from .pipeline import level_passes  # noqa: F401
        from .stage_levels import retouch_ops  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _focus3(state: dict) -> List[str]:
    return [r for r in _ordered_regions(state, "detail") if state["regions"][r].get("focus", 1) >= 3]


def task_command(state: dict, stage: str, region: str) -> str:
    W, H = state["width"], state["height"]
    if stage == "sketch":
        return f"easel sketch {region}"
    if stage == "line":
        return f"easel ink {region}"
    if stage == "flat":
        return f"easel flat {region}"
    if stage == "shade":
        return f"easel shade {region}"
    if stage == "forms" and state.get("style") == "photo":
        return f"easel airbrush {region}"
    if stage in ("block-in", "forms"):
        return f"easel paint {region}" if region == "all" else f"easel paint {region} --size {paint_size(stage, W, H)}"
    if stage == "detail":
        return f"easel detail {region}"
    return "easel status"


def dots_planned(state: dict) -> bool:
    """Dots (stippling) after level 1: on a working canvas with streaks, when the project was set up for them."""
    return bool(state.get("stipple")) and state.get("fine") == "streaks" and \
        int(state.get("work_scale", 1) or 1) > 1 and state.get("mode") != "hand"


def _open_key(state: dict, stage: str, key: str) -> bool:
    return any(t["stage"] == stage and t.get("key") == key and t["status"] == "open"
               for t in state.get("todo", []))


def build_todo(state: dict) -> None:
    """(Re)generate the todo list for the style and regions, keeping the status of existing items."""
    plan = plan_of(state)
    old = {(t["stage"], t.get("region"), t["kind"], t.get("key")): t for t in state.get("todo", [])}
    user = [t for t in state.get("todo", []) if t["kind"] == "user"]
    items: List[dict] = []

    def add(stage, kind, text, cmd, region=None, key=None):
        prev = old.get((stage, region, kind, key))
        items.append({"stage": stage, "region": region, "kind": kind, "key": key, "text": text, "cmd": cmd,
                      "status": prev["status"] if prev else "open", "note": prev.get("note", "") if prev else ""})

    for stage in plan:
        if stage == "plan":
            if state.get("reference") is not None or "reference" not in state:
                add("plan", "task", "look at the numbered marks of the reference", "easel marks", key="marks")
            add("plan", "task", "define 4-12 regions (focus 3 for face / eyes / hands)",
                "easel region add <name> --marks <numbers> --focus 3", key="regions")
            add("plan", "task", "cover the rest of the canvas", "easel region rest background", key="rest")
        elif stage == "change":
            from .change import todo_items
            for e in todo_items(state):
                add("change", "task", e["text"], e["cmd"], region=e["region"], key=e["key"])
                items[-1]["status"] = e["status"]
        elif stage == "finish":
            if (state.get("changes") or {}).get("items"):
                from .change import todo_items
                for e in todo_items(state, "finish"):           # lettering, a frame: over the traced picture
                    add("finish", "task", e["text"], e["cmd"], region=e["region"], key=e["key"])
                    items[-1]["status"] = e["status"]
            add("finish", "task", "compare the whole picture with the reference", "easel look", key="look")
            if _pipe_has(state, "finish", "grain"):
                add("finish", "task", "photo grain, measured from the reference (as the scripted pipeline)",
                    "easel fx grain", key="grain")
            # the blend is offered once, where the pipeline blends the paint under the detail: in the shade stage
            # when the style has one (cel, poster), else here (paint, photo: the paint layer, never the detail)
            if _pipe_has(state, "finish", "blend") and not _pipe_has(state, "blend", "blend")                     and state.get("soft_ramp", 0) >= 0.15:
                add("finish", "task", "blend patchy paint (soft-shaded reference)", "easel smooth all", key="blend")
            for r in _focus3(state):
                add("finish", "task", f"last close look at {r}: open VIEW and VIEW-REF, fix what differs by hand",
                    f"easel look {r}", region=r, key="closeup")
            if state.get("finish_sharpen"):
                add("finish", "task", "the finishing sharpen of the delivered picture, fitted to the reference",
                    "easel sharpen", key="sharpen")
            add("finish", "task", "export the picture (PNG; --psd --ora for layers, --timelapse film.gif for the film)",
                "easel export final.png", key="export")
        elif stage == "sketch":
            for r in _ordered_regions(state, stage):
                add(stage, "task", f"sketch {r}", task_command(state, stage, r), region=r)
        else:
            # the stage tool on the whole picture is the engine's best result (its scripted ceiling); regions
            # are then checked by the gate, and only the ones it lists get region fixes
            add(stage, "task", f"{STAGES[stage]['cmd']} all  (the whole picture; then fix the regions the gate "
                "lists)", task_command(state, stage, "all"), region="all")
            if stage == "shade" and _pipe_has(state, "blend", "blend") and state.get("soft_ramp", 0) >= 0.15:
                add(stage, "task", "blend the steps between tone shapes (soft-shaded reference)",
                    "easel smooth all", region="all", key="blend")
            if stage == "detail":
                regs = state["regions"]
                rank = [r for r in state.get("detail_rank") or [] if r in regs]
                order = rank + [r for r in _ordered_regions(state, "detail") if r not in rank]
                lv_ok = levels_available()
                for r in order:
                    f = regs[r].get("focus", 1)
                    if lv_ok:
                        add(stage, "task", f"refine {r} (focus {f}): easel detail {r} --level 1",
                            f"easel detail {r} --level 1", region=r, key="refine")
                    if f >= 3:
                        add(stage, "task", f"look close at {r} beside the reference and fix what differs by hand",
                            f"easel look {r}", region=r, key="closeup")
                if dots_planned(state):
                    add(stage, "task", "dot the texture points (after level 1): easel dot all", "easel dot all",
                        key="dot")
        add(stage, "gate", f"gate {stage}: {gate_words(stage, state.get('style'))}", f"easel stage done", key="gate")
    for t in user:
        items.append(t)
    # stable ids: position in the list
    for i, t in enumerate(items, 1):
        t["id"] = i
    state["todo"] = items


STAGE_LOOKS = {
    "plan": "regions sit on their parts; the face / eyes / hands are focus 3",
    "sketch": "blue construction lines follow every silhouette and main division",
    "line": "clean outlines on every edge the reference draws, nothing doubled",
    "flat": "every area filled with its base colour, no white gaps",
    "shade": "shadow shapes where the reference has them, same side, same softness",
    "block-in": "every area covered with big strokes of the right colour masses",
    "forms": "masses turned into forms: light and shadow sides, main edges",
    "detail": "coarse pass on all, then fine detail region by region (focus 3 first): eyes, mouth, hands crisp",
    "finish": "the whole picture matches the reference; exported",
    "change": "the target shows every change; nothing of the old colours left at the edges",
}


def plan_lines(state: dict) -> List[str]:
    """The whole plan as one line per stage, for the agent's own todo tool (TodoWrite / todowrite)."""
    out = []
    plan = plan_of(state)
    for k, stage in enumerate(plan, 1):
        items = [t for t in state["todo"] if t["stage"] == stage]
        tasks = [t for t in items if t["kind"] in ("task", "user")]
        done = all(t["status"] != "open" for t in items)
        cur = stage == state["stage"]
        if stage == "plan":
            regs = state["regions"]
            what = ("regions " + ", ".join(f"{n} f{r.get('focus', 1)}" for n, r in regs.items())) if regs else \
                "marks, 4-12 regions, focus 3 on what the viewer looks at first, region rest background"
        elif stage == "finish":
            close = [t["region"] for t in tasks if t.get("key") == "closeup"]
            what = "easel look; fix what differs by hand" + \
                (f"; last close look at {', '.join(close)}" if close else "") + "; easel export final.png"
        else:
            regs = [t["region"] for t in tasks if t.get("key") != "closeup" and t.get("region")]
            close = [t["region"] for t in tasks if t.get("key") == "closeup"]
            what = "; ".join(t["cmd"] for t in tasks if t.get("key") != "closeup") or f"easel {STAGES[stage]['cmd']}"
            if stage != "sketch":
                what += "; then fix only the regions the gate lists"
            if close:
                what += "; close look + hand fixes: " + ", ".join(close)
        mark = "done" if done or stage in state.get("stages_done", []) else ("in progress" if cur else "open")
        out.append(f"{k}. [{mark}] {stage}: {what} -> gate: {gate_words(stage, state.get('style'))}; looks right: {STAGE_LOOKS[stage]}")
    return out


def gate_words(stage: str, style: Optional[str] = None) -> str:
    if stage == "plan":
        return ">= 3 regions, >= 90% covered, one focus-3 region"
    if stage == "sketch":
        return "every region sketched"
    if stage == "finish":
        return "a last close look at every focus-3 region, picture exported"
    if stage == "change":
        from .change import GATE_WORDS
        return GATE_WORDS
    g = gate_for(stage, style)
    parts = []
    for k, v in g.items():
        if k == "overall_focus3":
            parts.append(f"focus-3 regions overall >= {v}")
        elif k == "coverage":
            parts.append(f"coverage >= {v}%")
        else:
            parts.append(f"{k} >= {v}")
    return ", ".join(parts) + " per region"


def tick(state: dict, stage: str, region: Optional[str], key: Optional[str] = None, note: str = "") -> List[int]:
    """Mark matching open task items done. region 'all' ticks every task of the stage."""
    if state.get("stale"):
        from .change import clear_stale
        clear_stale(state, stage, region)
    done = []
    for t in state.get("todo", []):
        if t["kind"] != "task" or t["stage"] != stage or t["status"] != "open":
            continue
        if key is not None:
            if t.get("key") == key and (region is None or t.get("region") == region):
                t["status"] = "done"
                t["note"] = note
                done.append(t["id"])
            continue
        if t.get("key") == "closeup":          # ticked only by looking at the region
            continue
        if t.get("key") in ("fine", "refine", "retouch", "dot", "sharpen", "blend"):   # ticked only by their own command
            continue
        if region == "all" or t.get("region") == region:
            t["status"] = "done"
            t["note"] = note
            done.append(t["id"])
    return done


def open_tasks(state: dict, stage: str) -> List[dict]:
    return [t for t in state.get("todo", []) if t["stage"] == stage and t["kind"] in ("task", "user")
            and t["status"] == "open"]


def plan_of(state: dict) -> List[str]:
    plan = PLANS[state.get("style") or "paint"]
    if (state.get("changes") or {}).get("items"):
        return plan[:1] + ["change"] + plan[1:]          # stated changes: their own stage right after the plan
    return plan


# ----------------------------------------------------------------------------------------------- gates

def coverage(project, mask: np.ndarray) -> float:
    """Share (0-100) of the region covered by paint (any layer except paper, sketch, fx, text)."""
    a = None
    for e in project.state["layers"]:
        if e["name"] in ("paper", "sketch", "fx", "text") or e.get("empty") or not e.get("visible", True):
            continue
        arr = project.layer(e["name"], create=False)
        if arr is None:
            continue
        a = arr[..., 3] if a is None else np.maximum(a, arr[..., 3])
    if a is None:
        return 0.0
    n = int(mask.sum())
    return 100.0 * float((a[mask] > 0.5).sum()) / max(1, n)


def plan_facts(project) -> dict:
    st = project.state
    regs = st["regions"]
    H, W = project.H, project.W
    union = np.zeros((H, W), bool)
    for n in regs:
        union |= project.region_mask(n)
    return {"n": len(regs), "covered": 100.0 * float(union.mean()),
            "focus3": [n for n, r in regs.items() if r.get("focus", 1) >= 3]}


def check_gate(project, stage: str, comparison=None) -> Tuple[bool, List[dict]]:
    """(passes, unmet items). Each unmet item: {region, metric, value, need, fix}."""
    st = project.state
    unmet: List[dict] = []
    if stage == "plan":
        f = plan_facts(project)
        if f["n"] < 3:
            unmet.append({"region": None, "metric": "regions", "value": f["n"], "need": 3,
                          "fix": "easel region add <name> --marks <numbers>   (numbers from  easel marks)"})
        if f["covered"] < 90:
            unmet.append({"region": None, "metric": "covered %", "value": round(f["covered"]), "need": 90,
                          "fix": "easel region rest background"})
        if not f["focus3"] and f["n"] >= 1 and not st.get("no_focus"):
            first = max(st["regions"], key=lambda n: st["regions"][n].get("focus", 1))
            unmet.append({"region": None, "metric": "focus-3 regions", "value": 0, "need": 1,
                          "fix": f"easel region focus {first} 3   (the place the viewer looks first)"})
        if st.get("request") and st.get("has_ref"):
            from .change import decision_unmet
            unmet += decision_unmet(st)
        return not unmet, unmet
    if stage == "change":
        from .change import unmet_all
        unmet = unmet_all(st)
        return not unmet, unmet
    if stage == "sketch":
        for t in st["todo"]:
            if t["stage"] == "sketch" and t["kind"] == "task" and t["status"] == "open":
                unmet.append({"region": t["region"], "metric": "sketched", "value": "no", "need": "yes",
                              "fix": t["cmd"]})
        return not unmet, unmet
    if stage == "finish":
        for t in st["todo"]:
            if t["stage"] == "finish" and t.get("key") == "closeup" and t["status"] == "open":
                unmet.append({"region": t["region"], "metric": "close look", "value": "no", "need": "yes",
                              "fix": t["cmd"]})
        if _open_key(st, "finish", "sharpen"):
            unmet.append({"region": None, "metric": "sharpened", "value": "no", "need": "yes",
                          "fix": "easel sharpen", "text": "the finishing sharpen is not applied yet"})
        if (st.get("changes") or {}).get("items"):
            from .change import unmet_all
            unmet = unmet_all(st, "finish") + unmet
        if not st.get("exported"):
            unmet.append({"region": None, "metric": "exported", "value": "no", "need": "yes",
                          "fix": "easel export final.png"})
        return not unmet, unmet
    if stage == "detail" and project.ref is not None and levels_available():
        # every region refined to level 1 (the picture's default end state)
        lv = st.get("levels") or {}
        for name, r in st["regions"].items():
            if r.get("area", 1.0) < 0.01:
                continue
            cur = min([v for v in (lv.get(name), lv.get("all")) if v is not None], default=None)
            if cur is None or cur > 1:
                unmet.append({"region": name, "metric": "level", "value": "-" if cur is None else cur, "need": 1,
                              "fix": f"easel detail {name} --level 1",
                              "text": f"{name} (focus {r.get('focus', 1)}) is not refined to level 1 yet"})
    if stage == "detail" and project.ref is not None and _open_key(st, "detail", "dot"):
        unmet.append({"region": None, "metric": "dotted", "value": "no", "need": "yes", "fix": "easel dot all",
                      "text": "the texture points are not dotted yet"})
    if stage == "detail" and project.ref is not None and levels_available():
        return not unmet, unmet          # the level is the detail gate; no score minimum beyond it
    gate = gate_for(stage, st.get("style"))
    if not gate:
        return not unmet, unmet
    if project.ref is None:
        return not unmet, unmet
    if comparison is None:
        from .cli import _comparison
        comparison = _comparison(project, project.composite_u8("score"))
    for name in st["regions"] or ["all"]:
        mask = project.region_mask(name) if name != "all" else np.ones((project.H, project.W), bool)
        s = comparison.scores(mask)
        focus = st["regions"].get(name, {}).get("focus", 1)
        tries = tries_of(project, f"{stage}:{name}")
        if name != "all" and st["regions"].get(name, {}).get("area", 1.0) < 0.01:
            continue          # under 1% of the canvas: too small for these measurements to mean anything
        for metric, need in gate.items():
            if metric == "overall_focus3":
                if focus < 3:
                    continue
                metric_key = "overall"
            else:
                metric_key = metric
            if metric == "overall" and focus >= 3 and "overall_focus3" in gate:
                continue
            if metric_key == "coverage":
                v = coverage(project, mask)
            else:
                v = s.get(metric_key)
                if v is None:
                    continue
            if v < need:
                unmet.append({"region": name, "metric": metric_key, "value": int(round(v)), "need": need,
                              "fix": fix_command(st, stage, name, metric_key, tries)})
    return not unmet, unmet


def _tries_file(project):
    return project.root / "tries.json"


def tries_of(project, key: str) -> int:
    """How often a stage tool ran on (stage, region): kept outside the undo history."""
    import json
    try:
        saved = json.loads(_tries_file(project).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    return max(int(saved.get(key, 0)), int(project.state.get("attempts", {}).get(key, 0)))


tries = tries_of


def save_tries(project, key: str, n: int) -> None:
    import json
    f = _tries_file(project)
    try:
        saved = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    saved[key] = max(int(saved.get(key, 0)), int(n))
    f.write_text(json.dumps(saved), encoding="utf-8")


def fix_command(st: dict, stage: str, region: str, metric: str, tries: int) -> str:
    """The command that should raise ``metric`` for ``region``; it escalates with the number of tries so a
    repeated failure never gets the same advice twice."""
    W, H = st["width"], st["height"]
    big, mid = paint_size("block-in", W, H), paint_size("forms", W, H)
    if tries >= MAX_TRIES:
        return (f"easel look {region} --show diff   then fix by hand (easel stroke / fill), or  "
                f"easel stage done --because \"...\"")
    t = min(tries, 2)
    if metric == "lines":
        return f"easel ink {region} --detail {[0.7, 0.85, 1.0][t]}" + (" --weight 0.8" if t == 2 else "")
    if metric == "coverage":
        if stage == "flat":
            return [f"easel flat {region}", f"easel flat {region} --colors 12", f"easel fill <x,y> --color pick"][t]
        return f"easel paint {region} --size {[big, big, mid][t]}" + (" --density 1.5" if t else "")
    if metric == "colour":
        if stage == "flat":
            return f"easel flat {region} --colors {[6, 10, 14][t]}"
        if stage == "shade":
            return f"easel shade {region} --levels {[3, 4, 5][t]}"
        if stage == "block-in":
            return f"easel paint {region} --size {[big, big, mid][t]}" + (" --density 1.5" if t == 1 else "")
        return [f"easel paint {region} --size {mid}", f"easel paint {region} --size {max(3, mid // 2)}",
                f"easel detail {region}"][t]
    if metric == "structure":
        if stage == "shade":
            return f"easel shade {region} --levels {[4, 4, 5][t]} --soft {[0.2, 0.1, 0.0][t]}"
        return [f"easel paint {region} --size {mid}", f"easel paint {region} --size {max(3, mid // 2)} --density 1.5",
                f"easel detail {region} --size 3"][t]
    if metric == "overall":
        return [f"easel detail {region}", f"easel detail {region} --size 2", f"easel detail {region} --size 2 "
                "--density 1.5"][t]
    return f"easel score {region}"


# ----------------------------------------------------------------------------------------------- stage moves

def stage_index(state: dict, stage: str) -> int:
    plan = plan_of(state)
    return plan.index(stage) if stage in plan else -1


def advance(state: dict) -> Optional[str]:
    plan = plan_of(state)
    cur = state["stage"]
    if cur not in state["stages_done"]:
        state["stages_done"].append(cur)
    for t in state["todo"]:               # closing a stage settles its items, so status never shows stale ones
        if t["stage"] == cur and t["kind"] in ("task", "user") and t["status"] == "open":
            t["status"] = "skipped"
            t["note"] = t.get("note") or "left open when the stage was closed"
    i = plan.index(cur) if cur in plan else 0
    if i + 1 < len(plan):
        state["stage"] = plan[i + 1]
        for t in state["todo"]:
            if t["stage"] == cur and t["kind"] == "gate":
                t["status"] = "done"
        return state["stage"]
    for t in state["todo"]:
        if t["stage"] == cur and t["kind"] == "gate":
            t["status"] = "done"
    state["finished"] = True
    return None


def go_back(state: dict, stage: str) -> None:
    plan = plan_of(state)
    state["stage"] = stage
    i = plan.index(stage)
    state["stages_done"] = [s for s in state["stages_done"] if s in plan and plan.index(s) < i]
    state["finished"] = False
    for t in state["todo"]:
        if t["kind"] == "gate" and t["stage"] in plan and plan.index(t["stage"]) >= i:
            t["status"] = "open"
        elif t["stage"] in plan and plan.index(t["stage"]) >= i and \
                t.get("note") == "left open when the stage was closed":
            t["status"], t["note"] = "open", ""


# ----------------------------------------------------------------------------------------------- next

def next_action(project, comparison=None) -> Tuple[str, str]:
    """(why, command): the single next thing to do."""
    st = project.state
    if st.get("finished"):
        return ("the picture is finished and exported" + (f" to {st['exported']}" if st.get("exported") else ""),
                "easel look --show both")
    stage = st["stage"]
    if project.ref is None and stage != "finish":
        return _next_without_reference(project)
    if st.get("changes") and stage != "plan":
        from .change import next_step
        nc = next_step(project, "top")
        if nc:
            return nc
    if stage == "plan":
        if not st.get("marks") and project.ref is not None:
            return "see the reference cut into numbered areas", "easel marks"
        f = plan_facts(project)
        if f["n"] < 3:
            big = _biggest_marks(project)
            hint = f" e.g. marks {big}" if big else ""
            return (f"name the main parts ({f['n']} regions so far, need 3+); pick mark numbers from the marks "
                    f"view{hint}", "easel region add <name> --marks <n,n,n> [--focus 3]")
        if f["covered"] < 90:
            return f"regions cover {f['covered']:.0f}% of the canvas, need 90%", "easel region rest background"
        if not f["focus3"] and not st.get("no_focus"):
            first = max(st["regions"], key=lambda n: (st["regions"][n].get("focus", 1), -st["regions"][n]["area"]))
            return "no region has focus 3 (where the viewer looks first)", f"easel region focus {first} 3"
        if st.get("request") and st.get("has_ref"):
            from .change import next_step
            nc = next_step(project, "plan")
            if nc:
                return nc
        return "plan gate passes; check the regions over the reference, then close the plan", "easel stage done"
    tasks = open_tasks(st, stage)
    if stage == "finish":
        if tasks:
            if tasks[0].get("key") in ("look", "export", "closeup"):
                return (tasks[0]["text"] + "; finishing touches if the picture wants them: easel fx "
                        "glow|grain|vignette|paper|levels, easel text \"...\" --box x0,y0,x1,y1"), tasks[0]["cmd"]
            return tasks[0]["text"], tasks[0]["cmd"]
        ok, unmet = check_gate(project, stage)
        return ("finish gate passes", "easel stage done") if ok else (unmet[0]["metric"], unmet[0]["fix"])
    if tasks:
        t = tasks[0]
        return f"{STAGES[stage]['what']} (todo {t['id']}: {t['text']})", t["cmd"]
    ok, unmet = check_gate(project, stage, comparison)
    if ok:
        return f"every {stage} task done and the gate passes", "easel stage done"
    u = min(unmet, key=lambda u: (u["value"] - u["need"]) if isinstance(u["value"], (int, float)) else 0)
    if u.get("text"):
        return (f"gate {stage}: {u['text']}", u["fix"])
    return (f"gate {stage}: {u['region']} {u['metric']} {u['value']} < {u['need']}", u["fix"])


HAND_HINT = {
    "sketch": 'easel shape ellipse x0,y0,x1,y1 --outline 3 --color "#607080" --layer sketch   (or  easel stroke '
              '"x,y x,y ..." --size 3 --color "#607080" --layer sketch)',
    "line": 'easel stroke "x,y x,y x,y ..." --size 4 --color "#202020"   (outlines; --closed for a loop)',
    "flat": 'easel fill x,y --color "#rrggbb"   (inside the line art; or  easel shape poly "x,y ..." --color "#rrggbb")',
    "shade": 'easel shape poly "x,y x,y x,y ..." --color "#rrggbb" --opacity 0.5 --layer shade',
    "block-in": 'easel shape ellipse|poly ... --color "#rrggbb"   or  easel gradient <region> --from x,y --to x,y '
                '--colors "#a,#b"',
    "forms": 'easel stroke "x,y x,y ..." --brush round --size 12 --color "#rrggbb" --opacity 0.7',
    "detail": 'easel stroke "x,y x,y ..." --size 2 --color "#rrggbb"',
}


def _next_without_reference(project) -> Tuple[str, str]:
    """Without a reference the stage tools have nothing to trace: the model draws every stage with the hand
    tools and ticks the todo items itself."""
    st = project.state
    stage = st["stage"]
    if stage == "plan":
        f = plan_facts(project)
        if f["n"] < 3:
            return (f"no reference: name the main parts with boxes ({f['n']} regions so far, need 3+)",
                    "easel region add <name> --box x0,y0,x1,y1 [--focus 3]")
        if f["covered"] < 90:
            return f"regions cover {f['covered']:.0f}% of the canvas, need 90%", "easel region rest background"
        if not f["focus3"] and not st.get("no_focus"):
            first = max(st["regions"], key=lambda n: (st["regions"][n].get("focus", 1), -st["regions"][n]["area"]))
            return "no region has focus 3 (where the viewer looks first)", f"easel region focus {first} 3"
        return "plan gate passes", "easel stage done"
    tasks = open_tasks(st, stage)
    if stage == "finish":
        if tasks:
            return tasks[0]["text"], tasks[0]["cmd"]
        return "finish gate passes", "easel stage done"
    if tasks:
        t = tasks[0]
        hint = HAND_HINT.get(stage, "easel stroke ...")
        return (f"no reference: draw {t.get('region') or 'it'} by hand with a command like this one, look, then  "
                f"easel todo done {t['id']}", hint)
    return f"every {stage} task done", "easel stage done"


def _biggest_marks(project) -> str:
    """The three biggest marks that no region covers yet."""
    st = project.state
    m = st.get("marks")
    if not m:
        return ""
    from PIL import Image
    labels = np.asarray(Image.open(project.root / m["file"]), dtype=np.int32)
    used = np.zeros(labels.shape, bool)
    for n in st["regions"]:
        used |= project.region_mask(n)
    free = []
    for r in sorted(m["table"], key=lambda r: -r["share"]):
        sel = labels == r["n"]
        if sel.any() and used[sel].mean() < 0.5:
            free.append(str(r["n"]))
        if len(free) == 3:
            break
    return ",".join(free)
