"""Faces, markings and line style for pictures without a reference.

``easel face <head id>``   brows, eyes (outline + lash line), nose mark and mouth laid from the head construction
                           in one step, each an editable line; ``--expression`` sets their shapes.
``easel mark <region>``    split an area: a marking (muzzle, belly, tail tip, socks, a patch) or the side of a curve
                           gets its own colour and region.
``easel linestyle``        silhouette lines heavier, inner lines thinner; ``--color auto`` turns each line into a
                           darker shade of the colour it borders (a softer look than near-black).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import cv2
import numpy as np

from . import coords
from .coords import UsageError
from . import draw_core as dc

USAGE = {
    "face": "easel face s2 --expression smile",
    "mark": "easel mark head --kind muzzle --color cream",
    "linestyle": "easel linestyle --weight auto --color auto",
}

EXPRESSIONS = ("calm", "smile", "grin", "open", "sad", "angry", "surprised", "sleepy")


# ============================================================================================ face

def face_parts(el: dict, guides: Dict[str, tuple], expression: str, eye_boxes: Optional[list] = None,
               kind: str = "human") -> List[dict]:
    """Line specs (units) for a face on a head element: [{name, pts, closed, weight, taper}]; ``kind`` "animal"
    lays a muzzle with the nose at its tip instead of a human nose, mouth and brows."""
    x0, y0, x1, y1 = el["box"]
    w, h = x1 - x0, y1 - y0
    chibi = bool(el.get("chibi"))
    yaw = math.radians(float(el.get("yaw", 0)))
    ew, eh = (0.3 * w, 0.24 * h) if chibi else (0.24 * w, 0.17 * h)
    out: List[dict] = []
    centres = [guides["left eye"], guides["right eye"]]
    mid = (centres[0][0] + centres[1][0]) / 2
    from .draw_line import eye_shape
    boxes = eye_boxes or []
    for k, (cx, cy) in enumerate(centres):
        far = (k == 0 and yaw > 0.1) or (k == 1 and yaw < -0.1)
        sw = ew * (0.85 if far else 1.0)
        b = boxes[k] if k < len(boxes) and boxes[k] else [cx - sw / 2, cy - eh / 2, cx + sw / 2, cy + eh / 2]
        bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        bw, bh = b[2] - b[0], b[3] - b[1]
        outer_left = bcx < mid
        s = -1 if outer_left else 1                       # inner -> outer direction
        side = "l" if k == 0 else "r"
        if expression in ("sleepy",) or (expression == "grin"):
            # closed happy / sleepy eyes: one arc
            bend = -1 if expression == "grin" else 1
            arc = [[bcx - bw / 2, bcy], [bcx, bcy + bend * 0.35 * bh], [bcx + bw / 2, bcy]]
            out.append({"name": f"eye-{side}", "pts": arc, "closed": False, "weight": 2.0, "taper": "both"})
        else:
            opn = {"surprised": 1.1, "sad": 0.8, "angry": 0.7}.get(expression, 0.95)
            tilt = {"angry": 14.0, "sad": -6.0}.get(expression, 6.0)
            loop, lash = eye_shape(b, opn, tilt, outer_left)
            out.append({"name": f"eye-{side}", "pts": loop, "closed": True, "weight": 0.9, "taper": "none"})
            out.append({"name": f"lash-{side}", "pts": lash, "closed": False, "weight": 2.6, "taper": "both"})
        # brow: an arc above the eye; its inner end moves with the expression
        top = b[1] - (0.42 if not chibi else 0.3) * bh
        inner_dy = {"angry": 0.35, "sad": -0.35, "surprised": -0.3}.get(expression, 0.0) * bh
        outer_dy = {"angry": -0.15, "sad": 0.15, "surprised": -0.3}.get(expression, 0.05) * bh
        inner = [bcx - s * 0.45 * bw, top + inner_dy]
        outer = [bcx + s * 0.55 * bw, top + outer_dy + 0.1 * bh]
        midb = [bcx + s * 0.08 * bw, top - 0.18 * bh + (inner_dy + outer_dy) / 2]
        out.append({"name": f"brow-{side}", "pts": [inner, midb, outer], "closed": False, "weight": 1.3,
                    "taper": "both"})
    nx, ny = guides["nose"]
    if kind == "animal":
        # an animal face: no brows; a muzzle round a nose at its tip, the mouth under the nose (a small "w")
        out = [s for s in out if not s["name"].startswith("brow-")]
        mx, my = guides["mouth"]
        mw = (0.2 if not chibi else 0.17) * w
        top = ny - 0.05 * h
        out.append({"name": "muzzle", "pts": [[nx - mw, top], [nx - 0.85 * mw, my + 0.035 * h], [nx, my + 0.075 * h],
                                              [nx + 0.85 * mw, my + 0.035 * h], [nx + mw, top]],
                    "closed": False, "weight": 1.1, "taper": "both"})
        nw = 0.045 * w
        out.append({"name": "nose", "pts": [[nx - nw, ny - 0.018 * h], [nx + nw, ny - 0.018 * h],
                                            [nx, ny + 0.022 * h]], "closed": True, "weight": 1.0, "taper": "none"})
        out.append({"name": "mouth", "pts": [[nx - 0.06 * w, my - 0.012 * h], [nx - 0.03 * w, my + 0.006 * h],
                                             [nx, ny + 0.03 * h], [nx + 0.03 * w, my + 0.006 * h],
                                             [nx + 0.06 * w, my - 0.012 * h]],
                    "closed": False, "weight": 1.0, "taper": "both"})
        return out
    ss = 1 if yaw >= 0 else -1                            # the nose's shadow side is away from the turn
    out.append({"name": "nose", "pts": [[nx + ss * 0.012 * w, ny - 0.035 * h], [nx + ss * 0.02 * w, ny],
                                        [nx - ss * 0.005 * w, ny + 0.008 * h]],
                "closed": False, "weight": 0.8, "taper": "both"})
    mx, my = guides["mouth"]
    mw = (0.13 if not chibi else 0.1) * w
    if expression in ("open", "surprised", "grin"):
        oh = {"surprised": 0.07, "open": 0.045, "grin": 0.04}[expression] * h
        top_y = my - (0.01 * h if expression != "surprised" else 0.02 * h)
        if expression == "surprised":
            mw *= 0.5
        loop = [[mx - mw / 2, top_y], [mx, top_y + 0.004 * h], [mx + mw / 2, top_y],
                [mx + mw * 0.3, top_y + oh * 0.8], [mx, top_y + oh], [mx - mw * 0.3, top_y + oh * 0.8]]
        out.append({"name": "mouth", "pts": loop, "closed": True, "weight": 1.0, "taper": "none"})
    else:
        bend = {"smile": 0.025, "calm": 0.008, "sad": -0.02, "angry": -0.01, "sleepy": 0.006}.get(expression, 0.01)
        mw *= {"smile": 1.15, "sad": 0.9, "angry": 0.8}.get(expression, 0.85)
        pts = [[mx - mw / 2, my - bend * h * 0.6], [mx, my + bend * h * 0.4], [mx + mw / 2, my - bend * h * 0.6]]
        out.append({"name": "mouth", "pts": pts, "closed": False, "weight": 1.0, "taper": "both"})
    return out


def face_neck(p, el: dict):
    """The neck under a face whose head sits on a mannequin (``easel wear`` draws the same neck when it comes
    first): (ops, touched px box, report lines)."""
    from . import draw_wear
    subj = el.get("subject")
    figs = [e for e in dc.dstate(p)["sketch"] if e["kind"] == "figure" and (e.get("subject") == subj or not subj)]
    if not figs or (not subj and len(figs) != 1):
        return [], None, []
    fig = figs[0]
    subj = str(fig.get("subject") or fig["id"])
    body = draw_wear.Body(p, fig, el)
    J = body.J
    if min(J["lshoulder"][1], J["rshoulder"][1]) <= J["chin"][1]:
        return [], None, []
    return draw_wear.add_neck(p, body, subj)


def cmd_face(ctx, args):
    from . import cli_draw as C
    from . import draw_line, draw_sketch
    from .cli import Spec, parse
    p = ctx.project
    d = dc.dstate(p)
    a = parse(args, Spec(USAGE["face"], ["id"], {"expression": "value", "weight": "value", "keep-eyes": "flag",
                                                 "kind": "value"},
                         alias={"expr": "expression", "mood": "expression", "type": "kind"}, min_pos=0))
    heads = [e for e in d["sketch"] if e["kind"] == "head"]
    if a.get("id"):
        el = draw_sketch.find(p, a["id"])
        if el["kind"] != "head":
            raise UsageError(f"{el['id']} is a {el['kind']}, not a head construction", "easel sketch list")
    elif heads:
        el = heads[-1]
    else:
        raise UsageError("no head construction yet", "easel sketch head <subject>")
    expr = str(a.get("expression") or "calm").lower()
    if expr not in EXPRESSIONS:
        raise UsageError(f"unknown expression {expr!r}: {', '.join(EXPRESSIONS)}", USAGE["face"])
    wscale = C._num(a, "weight", 1.0, 0.3, 3, ctx) if a.get("weight") else 1.0
    guides = draw_sketch.head_guides(p, el)
    eye_parts = sorted([n for n in d["parts"] if n.startswith("eye")], key=lambda n: d["parts"][n]["box"][0])
    boxes = [d["parts"][n]["box"] for n in eye_parts[:2]] if len(eye_parts) >= 2 else None
    from .draw_checks import HEAD_PARTS, is_animal
    kind =str(a.get("kind") or ("animal" if is_animal(p, el.get("subject")) else "human")).lower()
    if kind not in ("human", "animal"):
        raise UsageError(f"unknown --kind {kind!r}: human or animal", "easel face s2 --kind animal")
    specs = face_parts(el, guides, expr, boxes, kind=kind)
    if kind == "animal":
        ctx.notes.append("an animal face: muzzle with the nose at its tip, no brows (--kind human for a human face)")
    have_eyes = any(r.get("src", "").startswith("eye ") or r.get("src", "").startswith("face eye")
                    for r in d["lines"])
    if a.get("keep-eyes") or have_eyes:
        specs = [s for s in specs if not (s["name"].startswith("eye-") or s["name"].startswith("lash-"))]
    new_ids: List[str] = []
    lines: List[str] = []
    extra: List[str] = []
    # each feature belongs to its own part: an eye and its lash to that eye, the brows, nose and mouth to the head
    # (on a small face the plan boxes overlap, and the nearest box would hand a brow or the nose to an eye)
    subj = el.get("subject")
    head_part = next((n for n in HEAD_PARTS if n in d["parts"] and (not subj or d["parts"][n].get("of") == subj)),
                     None)
    side_eye = {"l": eye_parts[0], "r": eye_parts[1]} if len(eye_parts) >= 2 else {}

    def part_of(name: str, guess):
        kind, _, side = name.partition("-")
        if kind in ("eye", "lash"):
            return side_eye.get(side, guess)
        return head_part or (None if str(guess or "").startswith("eye") else guess)

    def mutate():
        ops, bb = [], None
        for sp in specs:
            path, _ = draw_line.fit(p, sp["pts"], closed=sp["closed"], snap=False, join=False, bow=0.0)
            rec = draw_line.make_record(p, path, sp["pts"], weight=sp["weight"] * wscale, taper=sp["taper"],
                                        color="line", closed=sp["closed"], src=f"face {sp['name']}")
            rec["part"] = part_of(sp["name"], rec.get("part"))
            d["lines"].append(rec)
            new_ids.append(rec["id"])
            lines.append(f"{rec['id']} {sp['name']}")
            ops += draw_line.render_ops(p, rec)
            bb = coords.union_box(bb, C._path_box(p, path))
        # an animal's head runs into its body through fur: no human neck lines
        nops, nbb, nlines = face_neck(p, el) if kind != "animal" else ([], None, [])
        extra.extend(nlines)
        return ops + nops, coords.union_box(bb, nbb)
    owned = C._line_owned(p, mutate)
    rep = C.draw_step(ctx, label=f"face {el['id']} {expr}", layer="line", owned=owned,
                      post=lambda: ["lines: " + ", ".join(lines)] + extra +
                                   ["each is an ordinary line: easel line move / redo / delete <id> to change it"],
                      note="the face lines (brows, eyes, nose, mouth)")
    return rep


# ============================================================================================ marks

MARKS = ("muzzle", "belly", "tip", "socks", "patch", "chest", "mask")


def mark_mask(p, region: np.ndarray, kind: str, subject_box_px=None, at=None) -> np.ndarray:
    m = np.asarray(region, bool)
    ys, xs = np.nonzero(m)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    cx, cy = xs.mean(), ys.mean()
    H, W = m.shape
    out = np.zeros(m.shape, np.uint8)
    if kind == "muzzle":
        cv2.ellipse(out, (int(cx), int(y0 + 0.79 * bh)), (int(0.25 * bw), int(0.17 * bh)), 0, 0, 360, 1, -1)
    elif kind in ("belly", "chest"):
        cyy = cy + (0.08 if kind == "belly" else -0.12) * bh
        cv2.ellipse(out, (int(cx), int(cyy)), (int(0.27 * bw), int(0.36 * bh)), 0, 0, 360, 1, -1)
    elif kind == "mask":
        cv2.ellipse(out, (int(cx), int(y0 + 0.45 * bh)), (int(0.45 * bw), int(0.14 * bh)), 0, 0, 360, 1, -1)
    elif kind == "patch":
        px, py = at if at is not None else (cx, cy)
        r = 0.18 * math.sqrt(m.sum())
        cv2.ellipse(out, (int(px), int(py)), (int(r * 1.2), int(r)), 20, 0, 360, 1, -1)
    elif kind == "tip":
        if subject_box_px is not None:
            sx, sy = (subject_box_px[0] + subject_box_px[2]) / 2, (subject_box_px[1] + subject_box_px[3]) / 2
        else:
            sx, sy = W / 2, H / 2
        dist = np.hypot(xs - sx, ys - sy)
        far = np.argmax(dist)
        fx, fy = xs[far], ys[far]
        r = 0.32 * max(bw, bh)
        cv2.circle(out, (int(fx), int(fy)), int(r), 1, -1)
    elif kind == "socks":
        out[int(y1 - 0.24 * bh):y1 + 1, :] = 1
    o = out.astype(bool)
    # a soft organic edge: blur and threshold
    sig = max(1.0, 0.02 * math.sqrt(m.sum()))
    o = cv2.GaussianBlur(o.astype(np.float32), (0, 0), sig) > 0.5
    return o & m


def split_by_curve(region: np.ndarray, curve_px: np.ndarray, at=None) -> np.ndarray:
    """The part of ``region`` on one side of an open curve (extended to the region's edges): the side holding
    ``at``, else the smaller side."""
    m = np.asarray(region, bool)
    H, W = m.shape
    c = np.asarray(curve_px, float)
    big = 2.0 * max(H, W)
    d0 = c[0] - c[1]
    d1 = c[-1] - c[-2]
    c = np.vstack([c[0] + d0 / (np.linalg.norm(d0) + 1e-9) * big, c, c[-1] + d1 / (np.linalg.norm(d1) + 1e-9) * big])
    bar = np.zeros((H, W), np.uint8)
    cv2.polylines(bar, [np.round(c).astype(np.int32)], False, 1, 2)
    rest = m & (bar == 0)
    from scipy import ndimage as ndi
    lab, n = ndi.label(rest)
    if n < 2:
        raise UsageError("the curve does not cut the area in two: draw it from edge to edge of the area",
                         'easel mark head "300,400 450,430 600,400" --color cream')
    sizes = ndi.sum(rest, lab, range(1, n + 1))
    if at is not None:
        k = int(lab[int(np.clip(at[1], 0, H - 1)), int(np.clip(at[0], 0, W - 1))])
        if k == 0:
            k = int(np.argmin(sizes)) + 1
    else:
        order = np.argsort(sizes)[::-1]
        k = int(order[1]) + 1                             # the smaller of the two big pieces
    side = lab == k
    return cv2.dilate(side.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & m


def apply_marks(p, parts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Keep marking regions after the flat regions are rebuilt: each mark is cut out of its parent."""
    from .ops import rle_to_mask
    d = dc.dstate(p)
    for name, mk in d.get("marks", {}).items():
        par = mk.get("parent")
        if par not in parts:
            continue
        try:
            mm = rle_to_mask(mk["rle"]).astype(bool)
        except Exception:  # noqa: BLE001
            continue
        if mm.shape != parts[par].shape:
            continue
        mm &= parts[par]
        parts[name] = mm
        parts[par] = parts[par] & ~mm
    return parts


def cmd_mark(ctx, args):
    from . import cli_draw as C
    from . import draw_plan
    from .cli import Spec, parse
    from .ops import Fill, mask_to_rle
    p = ctx.project
    d = dc.dstate(p)
    a = parse(args, Spec(USAGE["mark"], ["where", "pts*"], {"kind": "value", "color": "value", "at": "value",
                                                           "name": "value"},
                         alias={"colour": "color", "side": "at"}, min_pos=1))
    rname = str(a["where"]).strip()
    if rname not in p.state.get("regions", {}):
        raise UsageError(f"{rname} has no coloured area yet; mark it after  easel flat auto", "easel flat auto")
    M = p.region_mask(rname)
    at = None
    if a.get("at"):
        at = dc.u2px(p, coords.parse_points(a["at"], ctx.notes, 1))[0]
    if a.get("pts"):
        curve = dc.u2px(p, coords.parse_points(a["pts"], ctx.notes, 2))
        mm = split_by_curve(M, dc.catmull_rom(curve, spacing=2.0), at)
        kind = "curve"
    else:
        kind = str(a.get("kind") or "").lower()
        if kind not in MARKS:
            raise UsageError(f"say --kind ({', '.join(MARKS)}) or give a curve that cuts the area", USAGE["mark"])
        sb = None
        part = d["parts"].get(rname)
        if part and part.get("of") in d["subjects"]:
            b = d["subjects"][part["of"]]["box"]
            sb = coords.box_to_px(b, p.W, p.H)
        mm = mark_mask(p, M, kind, sb, at)
    if mm.sum() < 20:
        raise UsageError("the marking came out empty: pick another kind or draw a curve", USAGE["mark"])
    colour = a.get("color") or ("cream" if "cream" in d["palette"].get("swatches", {}) else "white")
    rgb = draw_plan.swatch(p, colour)
    name = C.from_regions_clean(a.get("name") or f"{rname}-{kind if kind != 'curve' else 'mark'}")
    sw_name = str(colour).strip().lower()

    def build():
        return [Fill(mask_rle=mask_to_rle(mm), color=tuple(int(v) for v in rgb), feather=1.0)]

    def post():
        d.setdefault("marks", {})[name] = {"parent": rname, "kind": kind, "color": sw_name,
                                           "rle": mask_to_rle(mm)}
        p.save_region_mask(name, mm)
        p.save_region_mask(rname, M & ~mm)
        box = coords.px_box_to_units(coords.mask_bbox_px(mm), p.W, p.H)
        p.state["regions"][name] = {"focus": p.state["regions"][rname].get("focus", 2), "box": list(box),
                                    "area": round(float(mm.mean()), 4), "how": f"mark on {rname}"}
        if rname in d["parts"]:
            d["parts"].setdefault(name, {"of": d["parts"][rname].get("of"), "box": list(box),
                                         "focus": d["parts"][rname].get("focus", 2), "swatch": sw_name})
        return [f"region {name}: {kind} of {rname} in {colour}; shade and detail it by that name"]
    return C.draw_step(ctx, label=f"mark {rname} {kind}", layer="flat", build=build, clip=M,
                       extra_files=[f"regions/{name}.png", f"regions/{rname}.png"], post=post,
                       note=f"the {kind} marking")


# ============================================================================================ line style

def outer_mask(p) -> np.ndarray:
    """Pixels outside every closed outline (flood fill from the picture's edge through the line layer)."""
    ln = p.layer("line", create=False)
    H, W = p.H, p.W
    if ln is None:
        return np.ones((H, W), bool)
    bar = (ln[..., 3] > 0.3).astype(np.uint8)
    bar = cv2.dilate(bar, np.ones((3, 3), np.uint8))
    free = (1 - bar).astype(np.uint8)
    ff = np.zeros((H + 2, W + 2), np.uint8)
    out = np.zeros((H, W), bool)
    for (x, y) in ((0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1), (W // 2, 0), (W // 2, H - 1), (0, H // 2),
                   (W - 1, H // 2)):
        if free[y, x] and not out[y, x]:
            img = free.copy()
            cv2.floodFill(img, ff, (x, y), 2)
            out |= img == 2
            ff[:] = 0
    return out


def cmd_linestyle(ctx, args):
    from . import cli_draw as C
    from . import draw_line
    from .cli import Spec, parse
    p = ctx.project
    d = dc.dstate(p)
    a = parse(args, Spec(USAGE["linestyle"], [], {"weight": "value", "color": "value", "outer": "value",
                                                  "inner": "value"}, alias={"colour": "color"}))
    if not d["lines"]:
        raise UsageError("no lines yet", 'easel line "x,y x,y x,y"')
    wmode = str(a.get("weight") or "auto").lower()
    cmode = str(a.get("color") or "keep").lower()
    k_out = C._num(a, "outer", 1.5, 0.5, 4, ctx) if a.get("outer") else 1.5
    k_in = C._num(a, "inner", 0.75, 0.2, 2, ctx) if a.get("inner") else 0.75
    outside = outer_mask(p) if wmode == "auto" else None
    flat = p.layer("flat", create=False)
    info = {"outer": 0, "inner": 0, "colour": 0}

    def mutate():
        bb = None
        near = cv2.dilate(outside.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) if outside is not None \
            else None
        for r in d["lines"]:
            P = draw_line.record_path(p, r)
            if len(P) < 2:
                continue
            xi = np.clip(np.round(P[:, 0]).astype(int), 0, p.W - 1)
            yi = np.clip(np.round(P[:, 1]).astype(int), 0, p.H - 1)
            base = r.get("base_weight", r.get("weight", 1.0))
            r["base_weight"] = base
            if near is not None:
                share = float(near[yi, xi].mean())
                if share > 0.5:
                    r["weight"] = round(base * k_out, 2)
                    info["outer"] += 1
                else:
                    r["weight"] = round(base * k_in, 2) if not str(r.get("src", "")).startswith("face lash") \
                        else base
                    info["inner"] += 1
            if cmode == "auto" and flat is not None:
                # the darker colour the line borders, darkened twice: a coloured line
                t = np.gradient(P, axis=0)
                t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
                nrm = np.stack([-t[:, 1], t[:, 0]], 1)
                off = 4.0 + r["weight"] * 2
                cols = []
                for sgn in (1, -1):
                    q = P + sgn * nrm * off
                    qx = np.clip(np.round(q[:, 0]).astype(int), 0, p.W - 1)
                    qy = np.clip(np.round(q[:, 1]).astype(int), 0, p.H - 1)
                    al = flat[qy, qx, 3] > 0.5
                    if al.any():
                        cols.append(np.median(flat[qy, qx, :3][al], axis=0) * 255)
                if cols:
                    c = min(cols, key=lambda v: dc.lum(v))
                    lc = dc.shadow_of(dc.shadow_of(tuple(float(x) for x in c)))
                    while dc.lum(lc) > 32:          # dark enough to read as a line against both sides
                        lc = dc.shadow_of(lc)
                    r["color"] = coords.hex_colour(tuple(int(x) for x in lc))
                    info["colour"] += 1
            elif cmode in ("line", "dark", "black"):
                r["color"] = "line"
            bb = coords.union_box(bb, C._path_box(p, P))
        ops = []
        for r in d["lines"]:
            ops += draw_line.render_ops(p, r)
        return ops, bb
    owned = C._line_owned(p, mutate)
    return C.draw_step(ctx, label=f"linestyle weight {wmode} color {cmode}", layer="line", owned=owned,
                       post=lambda: [f"{info['outer']} silhouette lines x{k_out:g}, {info['inner']} inner lines "
                                     f"x{k_in:g}" + (f", {info['colour']} lines coloured from their fills"
                                                     if info["colour"] else "")],
                       full_view=True, note="line weights and colours")


COMMANDS = {"face": cmd_face, "mark": cmd_mark, "linestyle": cmd_linestyle}
