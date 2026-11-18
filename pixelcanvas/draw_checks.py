"""Checks without a reference: the tool cannot compare with a target, so it measures the drawing's own state.

``holes(p, stage)`` returns the measured holes of a stage as dicts ``{metric, value, need, fix, text, region}``;
every hole blocks that stage's gate (``workflow_draw.check_gate``) and its ``text`` is the ``FIX:`` line the
commands print, so what the report shows is exactly what the gate measures. A hole's text is one plain sentence
with coordinates and the command that fixes it.

Holes: face features missing on a head construction (mouth, nose, brows; a muzzle on an animal), eyes of different
size or level, small line gaps (in the line stage and every later stage), lines outside every subject, sketch
proportions, unfilled areas, a planned part with no area of its own (and where its area leaks), lock areas coloured
as something else than hair, parts of one subject that should touch but have background between them, background
colour enclosed inside a subject, a focus-3 part with less detail than the rest, a subject whose marks do not fill or
centre in its planned box, and a farther subject's colour drawn over a nearer one.

``question(p, stage)`` asks one yes/no question about the VIEW, only where the eye alone can judge.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import coords
from .draw_core import composite_u8 as dc_composite, dstate, lum, px2u, u2px

FIGURE_WORDS = ("girl", "boy", "man", "woman", "person", "people", "character", "child", "kid", "lady", "knight",
                "mage", "witch", "wizard", "warrior", "princess", "prince", "hero", "student", "idol", "figure",
                "portrait", "self")
ANIMAL_WORDS = ("fox", "cat", "dog", "wolf", "bear", "rabbit", "bunny", "mouse", "rat", "deer", "horse", "lion",
                "tiger", "panda", "raccoon", "squirrel", "otter", "kitten", "puppy", "pig", "cow", "sheep", "goat",
                "monkey", "hamster", "ferret", "badger", "kitsune", "animal", "creature", "beast")
HEAD_PARTS = ("head", "face")


# things that are rectangles in the picture: a box-shaped band is the thing itself, not a placeholder
BOX_THINGS = {"wall", "walls", "window", "windows", "sill", "windowsill", "door", "floor", "ceiling", "table",
              "shelf", "board", "sign", "frame", "panel", "box", "book", "books", "rug", "carpet", "curtain",
              "screen", "building", "tower", "bench", "bed", "counter", "desk"}


def _hole(metric: str, text: str, fix: str, value="", need="", region: Optional[str] = None) -> dict:
    return {"metric": metric, "text": text, "fix": fix, "value": value, "need": need, "region": region}


def _u(p, q) -> str:
    return f"{int(round(q[0] * 1000 / p.W))},{int(round(q[1] * 1000 / p.H))}"


def _units(p, px: float) -> float:
    return px * 1000.0 / max(p.W, p.H)


def _part_mask(p, name: str) -> Optional[np.ndarray]:
    if name in p.state.get("regions", {}):
        try:
            return p.region_mask(name)
        except Exception:  # noqa: BLE001
            return None
    d = dstate(p)
    if name in d["parts"]:
        b = d["parts"][name]["box"]
        x0, y0, x1, y1 = coords.box_to_px(b, p.W, p.H)
        m = np.zeros((p.H, p.W), bool)
        m[y0:y1, x0:x1] = True
        return m
    return None


def _edge_density(img: np.ndarray, mask: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    e = cv2.Canny(g, 40, 110) > 0
    return float(e[mask].mean()) if mask.any() else 0.0


def _centre_in(el, box) -> bool:
    from .draw_sketch import centre
    cx, cy = centre(el)
    return box[0] - 20 <= cx <= box[2] + 20 and box[1] - 20 <= cy <= box[3] + 20


def _line_alpha(p) -> np.ndarray:
    ln = p.layer("line", create=False)
    return ln[..., 3] if ln is not None else np.zeros((p.H, p.W), np.float32)


def _regions(p) -> List[str]:
    return [n for n in p.state.get("regions", {}) if n not in ("bg",)]


def _rmask(p, n: str) -> np.ndarray:
    try:
        return p.region_mask(n)
    except Exception:  # noqa: BLE001
        return np.zeros((p.H, p.W), bool)


def is_animal(p, subject: Optional[str]) -> bool:
    """An animal subject (a fox mage is still a fox: its face needs a muzzle, not a human nose and mouth)."""
    from .workflow_draw import is_animal_state
    dstate(p)
    return is_animal_state(p.state, subject or "")


def is_figure(p, subject: str) -> bool:
    d = dstate(p)
    if any(w in subject.lower() for w in FIGURE_WORDS + ANIMAL_WORDS):
        return True
    return any(v["of"] == subject and any(w in n for w in ("head", "face", "eye", "arm", "leg", "hand", "torso"))
               for n, v in d["parts"].items())


# ============================================================================================ face features

def head_elements(p) -> List[dict]:
    """Head constructions whose subject has a planned head or face part (the faces the picture must show)."""
    d = dstate(p)
    out = []
    for el in d["sketch"]:
        if el["kind"] != "head":
            continue
        subj = el.get("subject")
        if any(n in HEAD_PARTS for n, v in d["parts"].items() if not subj or v["of"] == subj):
            out.append(el)
    done = {el.get("subject") for el in out}
    for n, v in d["parts"].items():
        if n in HEAD_PARTS and v["of"] not in done and not any(el.get("subject") is None for el in out):
            # no construction: the guides come from the part's box
            out.append({"id": None, "kind": "head", "box": list(v["box"]), "subject": v["of"], "yaw": 0, "pitch": 0})
            done.add(v["of"])
    return out


def face_missing(p, el) -> List[Tuple[str, Tuple[float, float]]]:
    """[(feature, (x, y) units)] the head construction ``el`` still lacks: mouth, nose, brow-l, brow-r (an animal:
    muzzle, nose, mouth). A feature counts when a short line lies at its guide (any line tool)."""
    from .draw_sketch import head_guides
    d = dstate(p)
    g = head_guides(p, el)
    x0, y0, x1, y1 = el["box"]
    w, h = x1 - x0, y1 - y0
    animal = is_animal(p, el.get("subject"))
    pts = []
    for r in d["lines"]:
        src = str(r.get("src", ""))
        if src.startswith(("locks ", "trace ", "close ", "eye ", "face eye", "face lash")) or r.get("bg") or \
                str(r.get("part") or "").startswith("eye"):
            continue
        P = np.asarray(r["path"], float).reshape(-1, 2)
        if len(P) < 2:
            continue
        span = max(P[:, 0].max() - P[:, 0].min(), P[:, 1].max() - P[:, 1].min())
        if span > 0.75 * w:             # an outline, not a feature
            continue
        pts.append((src, P, span))

    def near(c, rx, ry, max_span=None) -> bool:
        for src, P, span in pts:
            if max_span is not None and span > max_span:
                continue
            dx, dy = (P[:, 0] - c[0]) / rx, (P[:, 1] - c[1]) / ry
            if (dx * dx + dy * dy <= 1.0).any():
                return True
        return False

    def has(name) -> bool:
        return any(src.startswith(f"face {name}") for src, _, _ in pts)

    out = []
    mouth, nose = g["mouth"], g["nose"]
    if not (has("mouth") or near(mouth, 0.16 * w, 0.07 * h, 0.5 * w)):
        out.append(("mouth", mouth))
    if not (has("nose") or near(nose, 0.09 * w, 0.06 * h, 0.3 * w)):
        out.append(("nose", nose))
    if animal:
        muz = any(m.get("kind") == "muzzle" for m in d.get("marks", {}).values())
        if not (has("muzzle") or muz):
            out.append(("muzzle", ((mouth[0]), (nose[1] + mouth[1]) / 2)))
        return out
    for key, nm in (("left eye", "brow-l"), ("right eye", "brow-r")):
        ex, ey = g[key]
        c = (ex, ey - 0.16 * h)
        if not (has(nm) or near(c, 0.13 * w, 0.06 * h, 0.4 * w)):
            out.append((nm, c))
    return out


def face_holes(p) -> List[dict]:
    out = []
    for el in head_elements(p):
        miss = face_missing(p, el)
        if not miss:
            continue
        animal = is_animal(p, el.get("subject"))
        names = ", ".join(f"{n} at {coords.fmt_point(c)}" for n, c in miss)
        if el["id"] is None:
            cmd = f"pixelcanvas sketch head {el.get('subject') or ''}".rstrip() + "   (then  pixelcanvas face  lays them)"
        else:
            cmd = f"pixelcanvas face {el['id']}" + (" --kind animal" if animal else "")
        out.append(_hole("face features", f"the face of {el.get('subject') or el['id']} has no {names}: "
                         f"{cmd}   (or draw each with  pixelcanvas line on those points)", cmd,
                         ", ".join(n for n, _ in miss), "all", el.get("subject")))
    return out


# ============================================================================================ lines

def small_gaps(p) -> List[dict]:
    from . import draw_line
    d = dstate(p)
    if not d.get("lines"):
        return []
    an = draw_line.analyze(p)
    return [g for g in an["gaps"] if g["dist"] < 0.025 * max(p.W, p.H)]


def gap_holes(p) -> List[dict]:
    gaps = small_gaps(p)
    if not gaps:
        return []
    ids = ",".join(g["id"] for g in gaps[:6])
    where = "; ".join(f"{g['id']} at {_u(p, g['a']['pt'])} ({g['a']['line']} {g['a']['which']})" for g in gaps[:3])
    return [_hole("small gaps", f"{len(gaps)} small line gap(s) a fill leaks through: {where}:  pixelcanvas line close {ids}",
                  f"pixelcanvas line close {ids}", len(gaps), 0)]


def eye_holes(p) -> List[dict]:
    """Two eyes of one face drawn at different sizes or levels."""
    from . import draw_line
    d = dstate(p)
    eyes = [n for n in d["parts"] if n.startswith("eye")]
    boxes = {}
    for n in eyes:
        recs = [r for r in d["lines"] if r.get("part") == n and not str(r.get("src", "")).startswith("locks ")]
        if not recs:
            continue
        P = np.vstack([np.asarray(r["path"], float).reshape(-1, 2) for r in recs])
        boxes[n] = (P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max(), [r["id"] for r in recs])
    out = []
    by = {}
    for n in boxes:
        by.setdefault(d["parts"][n]["of"], []).append(n)
    for subj, ns in by.items():
        if len(ns) != 2:
            continue
        a, b = sorted(ns, key=lambda n: boxes[n][0])
        A, B = boxes[a], boxes[b]
        ha, hb = A[3] - A[1], B[3] - B[1]
        if min(ha, hb) <= 0:
            continue
        cya, cyb = (A[1] + A[3]) / 2, (B[1] + B[3]) / 2
        if abs(cya - cyb) > 0.3 * max(ha, hb):
            lo, hi = (a, b) if cya > cyb else (b, a)
            dy = round(abs(cya - cyb))
            ids = ",".join(boxes[lo][4][:4])
            out.append(_hole("eye level", f"{lo} sits {dy} units lower than {hi}:  pixelcanvas line move {ids} --by 0,-{dy}",
                             f"pixelcanvas line move {ids} --by 0,-{dy}", f"{dy} units", "same level", lo))
        elif min(ha, hb) / max(ha, hb) < 0.75:
            small = a if ha < hb else b
            out.append(_hole("eye size", f"{small} is {1 - min(ha, hb) / max(ha, hb):.0%} smaller than the other eye: "
                             f"pixelcanvas line eye {small} --open 0.9   (after  pixelcanvas line delete "
                             f"{','.join(boxes[small][4][:4])})", f"pixelcanvas line delete {','.join(boxes[small][4][:4])}",
                             "smaller", "same size", small))
    return out


def stray_line_holes(p) -> List[dict]:
    d = dstate(p)
    if not d["subjects"]:
        return []
    for r in d["lines"]:
        if r.get("bg") or r.get("src") or r.get("wear"):
            continue            # lines the tools lay (clothes, locks, faces) follow the figure, not the boxes
        pts = np.asarray(r["path"], float).reshape(-1, 2)
        inside = np.zeros(len(pts), bool)
        for sb in d["subjects"].values():
            b = sb["box"]
            inside |= (pts[:, 0] >= b[0] - 15) & (pts[:, 0] <= b[2] + 15) & (pts[:, 1] >= b[1] - 15) & \
                (pts[:, 1] <= b[3] + 15)
        if inside.mean() < 0.5:
            return [_hole("line outside", f"line {r['id']} lies mostly outside every subject box:  pixelcanvas line delete "
                          f"{r['id']}   (or plan the thing it draws:  pixelcanvas plan subject <name> --box ...)",
                          f"pixelcanvas line delete {r['id']}", r["id"], "inside a subject")]
    return []


def sketch_holes(p) -> List[dict]:
    from . import draw_sketch
    d = dstate(p)
    out = []
    for el in d["sketch"]:
        if el["kind"] == "figure":
            if is_animal(p, el.get("subject")):
                continue        # a mannequin under an animal is a guide: human limb proportions do not apply
            for ln in draw_sketch.figure_report(p, el):
                if "move" in ln:
                    out.append(_hole("proportion", f"{el['id']}: {ln}   (pixelcanvas sketch joint {el['id']} <joint> x,y)",
                                     f"pixelcanvas sketch joint {el['id']} <joint> x,y", ln.split(";")[0], "usual"))
    return out


# ============================================================================================ flat areas

def unfilled_holes(p, stage: str) -> List[dict]:
    from . import draw_flat
    d = dstate(p)
    if not d["lines"]:
        return []
    labels, table, _ = draw_flat.compute_areas(p)
    un = [r for r in draw_flat.unfilled(p, labels, table) if r["share"] > 0.0004]
    if not un or (stage != "flat" and sum(r["share"] for r in un) <= 0.01):
        return []
    nums = ",".join(str(r["n"]) for r in un[:8])
    return [_hole("unfilled areas", f"areas {nums} have no flat colour ({sum(r['share'] for r in un) * 100:.1f}% of "
                  f"the canvas; e.g. area {un[0]['n']} at {_u(p, un[0]['anchor'])}):  pixelcanvas flat {nums} --color "
                  f"<name>", f"pixelcanvas flat {nums} --color <name>", len(un), 0)]


def _box_px(p, b):
    return coords.box_to_px(b, p.W, p.H)


def hood_up(p, subject: Optional[str]) -> bool:
    """The subject wears a hood up (``pixelcanvas wear <subject> hood``): it closes round the face and hides the neck."""
    for r in dstate(p)["lines"]:
        w = r.get("wear") or {}
        if w.get("piece") == "hood" and w.get("of") == subject and not str(w.get("item", "")).startswith("hood roll"):
            return True
    return False


def part_area_holes(p) -> List[dict]:
    """A planned part with no coloured area of its own: name it, the area that holds its box, and where that area
    runs out of the box (the leak) into a neighbour or the background."""
    from . import draw_flat
    from .workflow_draw import is_text
    d = dstate(p)
    regs = p.state.get("regions", {})
    if not d.get("fills") and not regs:
        return []
    skipped = {t.get("region") for t in p.state.get("todo", []) if t.get("status") == "skipped"
               and t.get("key") == "flat-part"}
    labels, table, _ = draw_flat.compute_areas(p) if d["lines"] else (None, [], "")
    owner = {}
    for f in d["fills"]:
        if labels is not None:
            x, y = int(np.clip(f["pt"][0], 0, p.W - 1)), int(np.clip(f["pt"][1], 0, p.H - 1))
            owner[int(labels[y, x])] = f["part"]
    out = []
    for n, prt in d["parts"].items():
        if is_text(n) or n in regs or n in skipped:
            continue
        if any(r.get("parent") == n for r in d.get("marks", {}).values()):
            continue
        if n.startswith("neck") and hood_up(p, prt.get("of")):
            continue            # the hood closes round the face: the neck is under it
        x0, y0, x1, y1 = _box_px(p, prt["box"])
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        sw = prt.get("swatch") or n
        if labels is None:
            out.append(_hole("part without area", f"{n} has no coloured area of its own: draw its outline in "
                             f"{coords.fmt_box(prt['box'])} (pixelcanvas line \"x,y ...\" --closed), then  pixelcanvas flat --at "
                             f"<x,y inside> --color {sw} --part {n}; or drop the part:  pixelcanvas plan remove {n}",
                             f"pixelcanvas plan remove {n}", "none", "own area", n))
            continue
        sub = labels[y0:y1, x0:x1]
        vals, cnt = np.unique(sub[sub > 0], return_counts=True)
        if not len(vals):
            continue
        A = int(vals[int(np.argmax(cnt))])
        am = labels == A
        inside = float(am[y0:y1, x0:x1].sum()) / max(1, am.sum())
        holder = owner.get(A, "no colour")
        ys, xs = np.nonzero(am[y0:y1, x0:x1])
        cx, cy = int(np.median(xs)) + x0, int(np.median(ys)) + y0
        if inside >= 0.7:
            # the area is the part's own shape, only coloured as another part
            cmd = f"pixelcanvas flat --at {_u(p, (cx, cy))} --color {sw} --part {n}"
            out.append(_hole("part without area", f"{n} has no area of its own: area {A} inside its box is coloured "
                             f"as {holder}:  {cmd}", cmd, "none", "own area", n))
            continue
        # where the area leaves the box: on each side of the box, the middle of the crossing
        leaks = []
        band = max(2, int(0.004 * max(p.W, p.H)))
        for side, sl, axis in (("left", (slice(y0, y1), slice(x0, x0 + band)), 0),
                               ("right", (slice(y0, y1), slice(x1 - band, x1)), 0),
                               ("top", (slice(y0, y0 + band), slice(x0, x1)), 1),
                               ("bottom", (slice(y1 - band, y1), slice(x0, x1)), 1)):
            m = am[sl]
            if m.sum() < band * 4:
                continue
            yy, xx = np.nonzero(m)
            q = (int(np.median(xx)) + sl[1].start, int(np.median(yy)) + sl[0].start)
            edge = (side == "bottom" and y1 >= p.H - 2) or (side == "top" and y0 <= 1) or \
                (side == "left" and x0 <= 1) or (side == "right" and x1 >= p.W - 2)
            leaks.append(f"{_u(p, q)} ({'the canvas edge, close it with a line along the edge' if edge else side})")
        where = "; ".join(leaks[:3]) or f"around {_u(p, (cx, cy))}"
        bx = prt["box"]
        cmd = (f'pixelcanvas line "{bx[0]:.0f},{bx[1]:.0f} {bx[0]:.0f},{bx[3]:.0f} {bx[2]:.0f},{bx[3]:.0f} '
               f'{bx[2]:.0f},{bx[1]:.0f}"   (its outline: change the points to its real edge)')
        out.append(_hole("part without area", f"{n} has no area of its own: its box {coords.fmt_box(bx)} is filled "
                         f"by area {A} ({holder}), which runs out of it at {where}; draw {n}'s outline so it closes "
                         f"there, then  pixelcanvas flat --at <x,y inside> --color {sw} --part {n}   (or drop the part:  "
                         f"pixelcanvas plan remove {n})", cmd, "none", "own area", n))
    return out


def lock_holes(p) -> List[dict]:
    """Areas inside the front locks of a hair part that flat gave another colour (a fringe lock filled with skin)."""
    from . import draw_flat
    d = dstate(p)
    if not d.get("locks") or not d.get("fills") or not d["lines"]:
        return []
    labels, table, _ = draw_flat.compute_areas(p)
    out = []
    owner = {}
    for f in d["fills"]:
        x, y = int(np.clip(f["pt"][0], 0, p.W - 1)), int(np.clip(f["pt"][1], 0, p.H - 1))
        owner[int(labels[y, x])] = f["part"]
    for part in d["locks"]:
        if part not in d["parts"]:
            continue
        hm = draw_flat.lock_mask(p, part, front_only=True)
        if hm is None:
            continue
        bad = []
        for r in table:
            if owner.get(r["n"]) in (part, None):
                continue
            m = labels == r["n"]
            if r["share"] > 0.0003 and float(hm[m].mean()) >= 0.6:
                bad.append(r)
        if bad:
            nums = ",".join(str(r["n"]) for r in bad[:8])
            sw = d["parts"][part].get("swatch") or part
            cmd = f"pixelcanvas flat {nums} --color {sw} --part {part}"
            out.append(_hole("lock colour", f"areas {nums} lie inside the locks of {part} but are coloured "
                             f"{owner.get(bad[0]['n'])} (e.g. at {_u(p, bad[0]['anchor'])}):  {cmd}", cmd,
                             len(bad), 0, part))
    return out


# ============================================================================================ connection

BODY_WORDS = ("torso", "body", "neck", "chest", "robe", "coat", "dress", "shirt", "jacket", "sweater", "kimono",
              "uniform", "cloak", "cape", "hoodie", "blouse", "top", "armor", "armour", "tunic", "clothes", "shoulder")


def _bg_mask(p) -> np.ndarray:
    """Pixels that show the background: the background region, or no flat colour at all."""
    regs = p.state.get("regions", {})
    fl = p.layer("flat", create=False)
    m = np.zeros((p.H, p.W), bool)
    for n in ("background", "bg", "sky"):
        if n in regs:
            m |= _rmask(p, n)
    if fl is not None:
        m |= fl[..., 3] < 0.5
    pa = p.layer("paint", create=False)
    if pa is not None:
        m &= ~(pa[..., 3] > 0.5) | m
    return m


def connection_holes(p) -> List[dict]:
    """Parts of one subject whose plan boxes overlap but whose coloured areas have background between them (a head
    floating over its collar). Fix: a neck (two lines and a fill) or moving the part."""
    d = dstate(p)
    regs = p.state.get("regions", {})
    if not regs:
        return []
    from .workflow_draw import is_scenery
    bg = _bg_mask(p)
    out = []
    S = max(p.W, p.H)
    seen = set()
    for a, pa in d["parts"].items():
        if a not in regs or is_scenery(p.state, a):
            continue            # sky, sea and cliffs are bands of the scene: they meet by the horizon
        for b, pb in d["parts"].items():
            if b == a or b not in regs or pa["of"] != pb["of"] or (b, a) in seen or is_scenery(p.state, b):
                continue
            seen.add((a, b))
            A, B = pa["box"], pb["box"]
            iw, ih = min(A[2], B[2]) - max(A[0], B[0]), min(A[3], B[3]) - max(A[1], B[1])
            small = min((A[2] - A[0]) * (A[3] - A[1]), (B[2] - B[0]) * (B[3] - B[1]))
            head_pair = (a in HEAD_PARTS or b in HEAD_PARTS) and any(w in (b if a in HEAD_PARTS else a)
                                                                    for w in BODY_WORDS)
            if head_pair:
                if iw <= 0 or ih < -30:
                    continue    # a head and a body planned apart (or side by side)
            elif iw <= 0 or ih <= 0 or iw * ih < 0.1 * small:
                continue        # the plan does not make them touch (a staff beside the head)
            ma, mb = _rmask(p, a), _rmask(p, b)
            if not ma.any() or not mb.any():
                continue
            k = max(3, int(0.006 * S))
            if (cv2.dilate(ma.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)).astype(bool) & mb).any():
                continue
            dt, idx = cv2.distanceTransformWithLabels((~mb).astype(np.uint8), cv2.DIST_L2, 5,
                                                      labelType=cv2.DIST_LABEL_PIXEL)
            ys, xs = np.nonzero(ma)
            j = int(np.argmin(dt[ys, xs]))
            qa = (int(xs[j]), int(ys[j]))
            dist = float(dt[ys[j], xs[j]])
            # the nearest pixel of b: search a small window
            yb, xb = np.nonzero(mb)
            jb = int(np.argmin((xb - qa[0]) ** 2 + (yb - qa[1]) ** 2))
            qb = (int(xb[jb]), int(yb[jb]))
            if dist < 0.012 * S:
                continue
            n = max(2, int(dist))
            seg = np.stack([np.linspace(qa[0], qb[0], n), np.linspace(qa[1], qb[1], n)], 1).astype(int)
            on_bg = float(bg[np.clip(seg[:, 1], 0, p.H - 1), np.clip(seg[:, 0], 0, p.W - 1)].mean())
            if on_bg < 0.5:
                continue        # something of the picture lies between them (hair, a scarf): they connect
            gu = round(_units(p, dist))
            if head_pair:
                h, body = (a, b) if a in HEAD_PARTS else (b, a)
                qh, qbody = (qa, qb) if h == a else (qb, qa)
                hm, bm = (ma, mb) if h == a else (mb, ma)
                # the chin: the middle of the head's lowest rows; the collar: the body straight below it
                hy, hx = np.nonzero(hm)
                low = hy >= hy.max() - max(2, int(0.01 * S))
                cx_, cy_ = int(np.median(hx[low])), int(hy.max())
                col = bm[cy_:, cx_]
                if col.any():
                    qh, qbody = (cx_, cy_), (cx_, cy_ + int(np.argmax(col)))
                    gu = round(_units(p, qbody[1] - qh[1]))
                hb = coords.mask_bbox_px(_rmask(p, h))
                hw = (hb[2] - hb[0]) if hb else 0.2 * p.W
                xl, xr = qh[0] - 0.12 * hw, qh[0] + 0.12 * hw
                y_top, y_bot = qh[1] - 0.02 * p.H, qbody[1] + 0.02 * p.H
                l1 = f'pixelcanvas line "{_u(p, (xl, y_top))} {_u(p, (xl - 0.01 * hw, y_bot))}"'
                l2 = f'pixelcanvas line "{_u(p, (xr, y_top))} {_u(p, (xr + 0.01 * hw, y_bot))}"'
                sw = "skin" if "skin" in d["palette"]["swatches"] else (d["parts"][h].get("swatch") or h)
                mid = _u(p, ((qh[0] + qbody[0]) / 2, (qh[1] + qbody[1]) / 2))
                out.append(_hole("parts apart", f"{h} and {body} do not touch: background between {_u(p, qh)} (chin) "
                                 f"and {_u(p, qbody)} ({body}), {gu} units; draw a neck:  {l1}; {l2}; then  pixelcanvas "
                                 f"flat --at {mid} --color {sw} --part neck   (or move the head down {gu} units)",
                                 l1, f"{gu} units", "touching", h))
            else:
                out.append(_hole("parts apart", f"{a} and {b} do not touch: background between {_u(p, qa)} and "
                                 f"{_u(p, qb)}, {gu} units; extend one of them across:  pixelcanvas line \"{_u(p, qa)} "
                                 f"{_u(p, qb)}\" along the edges that should meet, then flat the closed area",
                                 f'pixelcanvas line "{_u(p, qa)} {_u(p, qb)}"', f"{gu} units", "touching", a))
    return out


NECK_STOP = BODY_WORDS + ("scarf", "collar", "neck")


def neck_holes(p) -> List[dict]:
    """A person's head over clothes with hair or background straight under the chin: no neck. Fix: two neck lines
    from the jaw down to the clothes and a skin fill between them."""
    d = dstate(p)
    regs = p.state.get("regions", {})
    S = max(p.W, p.H)
    out = []
    for h in HEAD_PARTS:
        if h not in regs or h not in d["parts"]:
            continue
        subj = d["parts"][h]["of"]
        if is_animal(p, subj) or hood_up(p, subj):
            continue            # fur runs into the body; a hood closes round the face and hides the neck
        body = [n for n, v in d["parts"].items() if v["of"] == subj and n in regs
                and any(w in n for w in NECK_STOP) and n != h]
        if not body:
            continue
        hm = _rmask(p, h)
        hy, hx = np.nonzero(hm)
        if not len(hy):
            continue
        low = hy >= hy.max() - max(2, int(0.01 * S))
        cx, cy = int(np.median(hx[low])), int(hy.max())
        bm = np.zeros((p.H, p.W), bool)
        for n in body:
            bm |= _rmask(p, n)
        col = bm[cy + 1:, cx]
        if not col.any():
            continue
        gap = int(np.argmax(col))
        if gap < 0.015 * S or gap > 0.25 * S:
            continue
        skin = np.zeros((p.H, p.W), bool)
        for n, v in d["parts"].items():
            if n in regs and v.get("swatch") == "skin" and n != h:
                skin |= _rmask(p, n)
        for n in regs:
            if n.startswith("neck"):
                skin |= _rmask(p, n)
        seg = skin[cy + 1:cy + 1 + gap, cx]
        if seg.mean() > 0.5:
            continue
        hb = coords.mask_bbox_px(hm)
        hw = hb[2] - hb[0]
        # the jaw where the neck starts: a little above the chin, about a fifth of the face width to each side
        y_top = cy - 0.12 * (hb[3] - hb[1])
        y_bot = cy + gap + 0.01 * p.H
        xl, xr = cx - 0.17 * hw, cx + 0.17 * hw
        l1 = f'pixelcanvas line "{_u(p, (xl, y_top))} {_u(p, (xl + 0.01 * hw, y_bot))}"'
        l2 = f'pixelcanvas line "{_u(p, (xr, y_top))} {_u(p, (xr - 0.01 * hw, y_bot))}"'
        mid = _u(p, (cx, cy + gap / 2))
        what = "hair" if any(n in regs and _rmask(p, n)[cy + 1 + gap // 2, cx] for n in d["parts"]
                             if "hair" in n) else "background"
        out.append(_hole("no neck", f"{subj} has no neck: {what} shows under the chin from {_u(p, (cx, cy))} down "
                         f"to {body[0]} at {_u(p, (cx, cy + gap))}; draw it:  {l1}; {l2}; then  pixelcanvas flat --at "
                         f"{mid} --color skin --part neck", l1, f"{round(_units(p, gap))} units of {what}", "a neck",
                         h))
    return out


def background_holes(p) -> List[dict]:
    """Background colour enclosed inside a subject: a sliver between a fill and its outline, or a piece of the
    background that the subject's fills surround."""
    d = dstate(p)
    sw = d["palette"]["swatches"]
    key = next((k for k in ("background", "sky") if k in sw), None)
    if key is None or not d["subjects"]:
        return []
    from .draw_plan import swatch
    bgc = np.array(swatch(p, key), float)
    img = dc_composite(p, "fills").astype(np.float32)
    S = max(p.W, p.H)
    near = np.sqrt(((img - bgc) ** 2).sum(-1)) < 14.0
    line = _line_alpha(p) > 0.3
    # a band is painted whole, whatever its colours (a night sea's own gradient runs through the sky's blue): its
    # pixels are that thing; background shows through only outside the bands
    owned = np.zeros((p.H, p.W), bool)
    for n in d.get("bands", {}):
        if n not in ("background", "bg", "sky", key) and n in p.state.get("regions", {}):
            owned |= _rmask(p, n)
    cand = near & ~line & ~owned
    n, lab, stats, cent = cv2.connectedComponentsWithStats(cand.astype(np.uint8), 8)
    if n <= 2:
        return []
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    fills = ~near & ~line
    out = []
    min_area = 0.0004 * p.W * p.H
    for k in range(1, n):
        if k == main or stats[k, cv2.CC_STAT_AREA] < min_area:
            continue
        cu = (cent[k][0] * 1000 / p.W, cent[k][1] * 1000 / p.H)
        subj = [s for s, v in d["subjects"].items() if v["box"][0] <= cu[0] <= v["box"][2]
                and v["box"][1] <= cu[1] <= v["box"][3]]
        if not subj:
            continue
        x, y, w, h = stats[k, :4]
        comp = lab[y:y + h, x:x + w] == k
        pad = 4
        ring_full = np.zeros((h + 2 * pad, w + 2 * pad), bool)
        ring_full[pad:pad + h, pad:pad + w] = comp
        dil = cv2.dilate(ring_full.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ~ring_full
        ys, xs = np.nonzero(dil)
        ys, xs = ys + y - pad, xs + x - pad
        ok = (ys >= 0) & (ys < p.H) & (xs >= 0) & (xs < p.W)
        ys, xs = ys[ok], xs[ok]
        if not len(ys):
            continue
        fill_share = float(fills[ys, xs].mean())
        thick = float(cv2.distanceTransform(np.pad(comp, 1).astype(np.uint8), cv2.DIST_L2, 3).max())
        thin = thick < 0.012 * S
        if fill_share < 0.25 and not thin:
            continue        # a gap the outline frames on every side (between an arm and the body): background
        yy, xx = np.nonzero(comp)
        j = int(np.argmax(cv2.distanceTransform(np.pad(comp, 1).astype(np.uint8), cv2.DIST_L2, 3)[1:-1, 1:-1]
                          [yy, xx]))
        q = (int(xx[j] + x), int(yy[j] + y))
        out.append(_hole("background inside", f"background colour inside {subj[0]} at {_u(p, q)} "
                         f"({stats[k, cv2.CC_STAT_AREA] * 100 / (p.W * p.H):.1f}% of the canvas, "
                         f"{'a sliver between a fill and its outline' if thin else 'surrounded by its fills'}):  "
                         f"pixelcanvas flat hole {_u(p, q)}", f"pixelcanvas flat hole {_u(p, q)}",
                         _u(p, q), "none", subj[0]))
        if len(out) >= 3:
            break
    return out


# ============================================================================================ detail

def focus_density(p) -> Tuple[Dict[str, Tuple[float, int]], float]:
    from .workflow_draw import is_text
    d = dstate(p)
    img = dc_composite(p)
    if any(e["name"] == "surface" and not e.get("empty", True) for e in p.state["layers"]):
        # the surface pass puts texture on every area: the focus is measured on the drawing without it
        from . import compose
        img = compose.to_uint8(compose.composite(p.state["layers"], p.layers_dict(), p.H, p.W,
                                                 background=p.background(), skip=["sketch", "surface"]))
    dens, share = {}, {}
    for name, prt in d["parts"].items():
        if is_text(name):
            continue
        m = _part_mask(p, name)
        if m is not None and m.sum() > 100:
            dens[name] = (_edge_density(img, m), prt.get("focus", 2))
            share[name] = float(m.mean())
    # small parts (a staff, the feet) are all outline and read as dense: the yardstick is the bigger parts
    # (a part with no coloured area of its own is measured over its plan box, which is not the part: left out)
    regs = p.state.get("regions", {})
    rest = [v for n, (v, f) in dens.items() if f < 3 and share[n] > 0.015 and n in regs] or \
        [v for n, (v, f) in dens.items() if f < 3 and n in regs] or [v for n, (v, f) in dens.items() if f < 3]
    return dens, float(np.mean(rest)) if rest else 0.0


DETAIL_RATIO = 0.8


def detail_holes(p) -> List[dict]:
    dens, mean_rest = focus_density(p)
    out = []
    if not mean_rest:
        return out
    surf = dstate(p).get("surface", {})
    heads = {el.get("subject"): el for el in head_elements(p)}
    for n, (v, f) in dens.items():
        if n in HEAD_PARTS:
            el = heads.get(dstate(p)["parts"].get(n, {}).get("of")) or (list(heads.values())[-1] if heads else None)
            if el is not None and not face_missing(p, el):
                continue    # a face (skin or fur) is sparse in a professional picture too; its features are checked
                #             on their own (face_holes), its eyes are parts of their own
        if surf.get(n, {}).get("material") in ("skin", "glass") and not (n in HEAD_PARTS and head_elements(p)
                                                                          and face_missing(p, head_elements(p)[-1])):
            continue        # a face is sparse in a professional picture too; its features are checked on their own
        if f >= 3 and v < DETAIL_RATIO * mean_rest:
            from .workflow_draw import detail_command
            cmd = detail_command(n, f)[1]
            if n in HEAD_PARTS:
                heads = head_elements(p)
                if heads and face_missing(p, heads[-1]):
                    cmd = f"pixelcanvas face {heads[-1]['id']}"
                else:
                    cmd = f"pixelcanvas blush {n} --lines"
            out.append(_hole("focus detail", f"focus part {n} has less detail than the other parts ({v:.3f} vs "
                             f"{mean_rest:.3f} edge density, need {DETAIL_RATIO * mean_rest:.3f}): add detail there "
                             f"first:  {cmd}", cmd, f"{v:.3f}", f"{DETAIL_RATIO * mean_rest:.3f}", n))
    return out


# ============================================================================================ subjects in their boxes

PLACEHOLDERS: Dict[int, set] = {}       # id(project) -> {(subject, band)} found by the last subject_marks


def subject_marks(p) -> Dict[str, np.ndarray]:
    """What each subject has on the canvas: the coloured areas of the subject and its parts (and the regions whose
    middle lies in its box), plus the detail layer inside its box. A plain rectangle band of a thing that is not
    ground (a 'pines' band that is a box) is a placeholder, not the thing, and does not count."""
    from .workflow_draw import is_ground
    d = dstate(p)
    regs = p.state.get("regions", {})
    subs = d["subjects"]
    out = {s: np.zeros((p.H, p.W), bool) for s in subs}
    PLACEHOLDERS[id(p)] = set()
    own = {}
    for n in regs:
        if n in ("background", "bg"):
            continue
        if n in subs:
            own[n] = n
        elif n in d["parts"]:
            own[n] = d["parts"][n]["of"]
        elif any(n.startswith(s + "-") or n.startswith(s + "_") for s in subs):
            own[n] = max((s for s in subs if n.startswith(s)), key=len)     # pines-1, pines-2 ...
        else:
            b = regs[n].get("box")
            if not b:
                continue
            c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            # a thing belongs to the smallest box that holds it, a non-ground subject before a ground band
            hits = sorted((is_ground(s), (v["box"][2] - v["box"][0]) * (v["box"][3] - v["box"][1]), s)
                          for s, v in subs.items()
                          if v["box"][0] <= c[0] <= v["box"][2] and v["box"][1] <= c[1] <= v["box"][3])
            if hits:
                own[n] = hits[0][2]
    for n, s in own.items():
        if s not in out:
            continue
        m = _rmask(p, n)
        if not m.any():
            continue
        if not is_ground(s):
            bb = coords.mask_bbox_px(m)
            # how box-like the band was when drawn (later shapes cut into its region)
            fill = d.get("bands", {}).get(n, {}).get("fill", m.sum() / max(1, (bb[2] - bb[0]) * (bb[3] - bb[1])))
            sb = _box_px(p, subs[s]["box"])
            boxy = set(re.split(r"[-_\s]+", s.lower())) & BOX_THINGS
            if fill > 0.92 and (bb[2] - bb[0]) > 0.8 * (sb[2] - sb[0]) and not boxy:
                PLACEHOLDERS.setdefault(id(p), set()).add((s, n))
                continue        # a box-shaped band: a placeholder field
        out[s] |= m
    det = p.layer("detail", create=False)
    if det is not None:
        a = det[..., 3] > 0.5
        for s, v in subs.items():
            x0, y0, x1, y1 = _box_px(p, v["box"])
            out[s][y0:y1, x0:x1] |= a[y0:y1, x0:x1]
    return out


def fill_holes(p) -> List[dict]:
    """A subject whose marks fill too little of its planned box, sit in one end of it, or sit off its centre."""
    from .workflow_draw import is_ground, is_text
    d = dstate(p)
    out = []
    marks = subject_marks(p)
    for s, v in d["subjects"].items():
        if is_ground(s) or is_text(s):
            continue
        parts = [n for n, q in d["parts"].items() if q["of"] == s]
        if parts and all(is_text(n) for n in parts):
            continue
        x0, y0, x1, y1 = _box_px(p, v["box"])
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        m = marks[s][y0:y1, x0:x1]
        cover = float(m.mean())
        bx = v["box"]
        sw = s if s in d["palette"]["swatches"] else next((q.get("swatch") for q in d["parts"].values()
                                                          if q["of"] == s and q.get("swatch")), s)
        holders = [n for (s2, n) in PLACEHOLDERS.get(id(p), set()) if s2 == s]
        if cover < 0.003:
            if holders:
                cmd = f'pixelcanvas band {s}-1 "x,y x,y x,y ..." --closed --color {sw}-shadow'
                out.append(_hole("subject fill", f"{s}: planned box {coords.fmt_box(bx)} holds only the plain "
                                 f"box-shaped band {holders[0]}, no {s} drawn: draw their shapes over it, one closed "
                                 f"shape each (e.g.  {cmd}; for trees: pixelcanvas band {s}-1 \"x,ybottom x,ytop\" --tiers 5 "
                                 f"--width 160 --color {sw}-shadow)", cmd, "0%", ">= 20%", s))
            continue            # nothing of it drawn yet: the todo items say what to draw
        rows = m.mean(1) > 0.03
        cols = m.mean(0) > 0.03
        ry = np.flatnonzero(rows)
        rx = np.flatnonzero(cols)
        ys, xs = np.nonzero(m)
        cx_u = (xs.mean() + x0) * 1000 / p.W
        cy_u = (ys.mean() + y0) * 1000 / p.H
        bw, bh = bx[2] - bx[0], bx[3] - bx[1]
        ext_h = (ry[-1] - ry[0] + 1) / (y1 - y0) if len(ry) else 0.0
        ext_w = (rx[-1] - rx[0] + 1) / (x1 - x0) if len(rx) else 0.0
        head = f"{s}: planned box {coords.fmt_box(bx)}, marks cover {cover:.0%} of it"
        draw_cmd = f'pixelcanvas band {s}-2 "x,y x,y x,y ..." --closed --color {sw}'
        fix = (f'draw {s} across its box (bigger shapes or more of them:  {draw_cmd}), or plan the box you '
               f'mean:  pixelcanvas plan subject {s} --box x0,y0,x1,y1')
        if cover < 0.2 or ext_h < 0.5 or ext_w < 0.5:
            if ext_h < 0.5 and len(ry):
                top = (ry[0] + y0) * 1000 / p.H
                bot = (ry[-1] + y0) * 1000 / p.H
                where = (f", all in the top {bot - bx[1]:.0f} units" if top - bx[1] < 0.15 * bh else
                         f", all in the bottom {bx[3] - top:.0f} units" if bx[3] - bot < 0.15 * bh else
                         f", only y {top:.0f}-{bot:.0f}")
            elif ext_w < 0.5 and len(rx):
                where = f", only x {(rx[0] + x0) * 1000 / p.W:.0f}-{(rx[-1] + x0) * 1000 / p.W:.0f}"
            else:
                where = ""
            out.append(_hole("subject fill", f"{head}{where}: {fix}", draw_cmd,
                             f"{cover:.0%}", ">= 20% and >= half the box each way", s))
            continue
        dx, dy = cx_u - (bx[0] + bx[2]) / 2, cy_u - (bx[1] + bx[3]) / 2
        if abs(dx) > 0.12 * bw:
            side = "right" if dx > 0 else "left"
            band = _main_band(p, s, marks[s])
            if band:
                cmd = f"pixelcanvas band {band} --shift {-dx:.0f},0"
                how = f"move it:  {cmd}"
            else:
                mb = coords.px_box_to_units(coords.mask_bbox_px(marks[s]), p.W, p.H)
                lo, hi = ((bx[0], mb[0]) if dx > 0 else (mb[2], bx[2]))
                cmd = (f'pixelcanvas band {s}-2 "x,y x,y x,y ..." --closed --color {sw}   (more {s} on the '
                       f'{"left" if dx > 0 else "right"}, x {lo:.0f}-{hi:.0f})')
                how = (f"draw more on the {'left' if dx > 0 else 'right'} side (x {lo:.0f}-{hi:.0f}):  {cmd}; or "
                       f"plan the box where it is:  pixelcanvas plan subject {s} --box {coords.fmt_box(mb)}")
            out.append(_hole("subject centre", f"{head}, but its mass sits {abs(dx):.0f} units {side} of the box "
                             f"centre x={(bx[0] + bx[2]) / 2:.0f}: {how}", cmd, f"{abs(dx):.0f} units",
                             f"<= {0.12 * bw:.0f}", s))
    return out


def _main_band(p, s: str, m: np.ndarray) -> Optional[str]:
    """The band region that makes most of subject ``s``'s marks (what  band --shift  can move)."""
    d = dstate(p)
    best, val = None, 0
    for n in d.get("bands", {}):
        if n not in p.state.get("regions", {}):
            continue
        v = int((_rmask(p, n) & m).sum())
        if v > val:
            best, val = n, v
    return best if best and val > 0.5 * m.sum() else None


# ============================================================================================ depth

def depth_order(p) -> List[str]:
    """Subjects from far to near: the order the model gave with  plan order, else the order they were planned."""
    from .workflow_draw import is_ground
    d = dstate(p)
    order: List[str] = []
    for s in d.get("order", []):
        s = d["parts"][s]["of"] if s in d["parts"] else s
        if s in d["subjects"] and s not in order:
            order.append(s)
    if order:
        return order + [s for s in d["subjects"] if s not in order]
    # no order given: sky and background at the back, then the subjects in the order they were planned (a painter
    # plans and paints back to front)
    subs = list(d["subjects"])
    back = [s for s in subs if any(w in s for w in ("sky", "background", "cloud")) and is_ground(s)]
    return back + [s for s in subs if s not in back]


def subject_of(p, name: str) -> Optional[str]:
    """The subject a region or part name belongs to: the subject itself, a part's subject, or the subject the name
    starts with (pines-l1, pines_2 and pines-shadow belong to pines; the longest subject name wins)."""
    d = dstate(p)
    if name in d["subjects"]:
        return name
    if name in d["parts"]:
        return d["parts"][name]["of"]
    pre = [s for s in d["subjects"] if name.startswith(s + "-") or name.startswith(s + "_")]
    return max(pre, key=len) if pre else None


def region_subject(p, name: str) -> Optional[str]:
    """``subject_of``, else the subject whose box holds the region's middle (a band named for what it is)."""
    return subject_of(p, name) or _band_subject(p, name)


def nearer_mask(p, subject: str) -> Tuple[np.ndarray, List[str]]:
    """The coloured areas of the subjects nearer than ``subject`` in the depth order (what a mark of ``subject``
    drawn now has to stay behind), and their names."""
    order = depth_order(p)
    if subject not in order:
        return np.zeros((p.H, p.W), bool), []
    near = order[order.index(subject) + 1:]
    regs = p.state.get("regions", {})
    d = dstate(p)
    m = np.zeros((p.H, p.W), bool)
    names = []
    for s in near:
        own = [s] + [n for n, v in d["parts"].items() if v["of"] == s] + \
            [n for n in d.get("bands", {}) if n not in d["subjects"] and n not in d["parts"]
             and region_subject(p, n) == s]
        hit = False
        for n in own:
            if n in regs:
                m |= _rmask(p, n)
                hit = True
        if hit:
            names.append(s)
    return m, names


def _band_subject(p, name: str) -> Optional[str]:
    d = dstate(p)
    b = p.state.get("regions", {}).get(name, {}).get("box")
    if not b:
        return None
    c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    hits = sorted(((v["box"][2] - v["box"][0]) * (v["box"][3] - v["box"][1]), s) for s, v in d["subjects"].items()
                  if v["box"][0] <= c[0] <= v["box"][2] and v["box"][1] <= c[1] <= v["box"][3])
    return hits[0][1] if hits else None


WATER_WORDS = ("water", "lake", "sea", "river", "ocean", "pond", "bay", "harbour", "harbor")


def _is_water(name: str) -> bool:
    return any(t in WATER_WORDS for t in re.split(r"[-_\s]+", name.lower()))


def depth_holes(p) -> List[dict]:
    """A farther subject (or a part of it) whose colour is drawn over a nearer subject's area: the sun in front of
    the mountains it should set behind."""
    from .draw_plan import swatch
    d = dstate(p)
    order = depth_order(p)
    if len(order) < 2:
        return []
    marks = None
    img = dc_composite(p).astype(np.float32)
    out = []
    regs = p.state.get("regions", {})

    def bands_of(s):
        # bands named after the subject (pines-l1, pines-shadow) are the subject's areas too
        return [n for n in d.get("bands", {}) if n not in d["subjects"] and n not in d["parts"]
                and subject_of(p, n) == s]
    for i, far in enumerate(order[:-1]):
        names = [n for n, v in d["parts"].items() if v["of"] == far] + [far]
        for n in names:
            sw = d["parts"].get(n, {}).get("swatch") if n in d["parts"] else (n if n in d["palette"]["swatches"]
                                                                            else None)
            if not sw or sw in ("background", "line"):
                continue
            try:
                c = np.array(swatch(p, sw), float)
            except Exception:  # noqa: BLE001
                continue
            for near in order[i + 1:]:
                if marks is None:
                    marks = subject_marks(p)
                nm = marks.get(near)
                if nm is None or not nm.any():
                    continue
                own = [q for q in [near] + [k for k, v in d["parts"].items() if v["of"] == near] + bands_of(near)
                       if q in regs]
                if not own:
                    continue
                try:
                    oc = np.array(swatch(p, d["parts"].get(own[0], {}).get("swatch") or own[0]), float)
                except Exception:  # noqa: BLE001
                    oc = None
                if oc is not None and float(np.sqrt(((oc - c) ** 2).sum())) < 45.0:
                    continue    # two greens a few steps apart: the colours cannot tell the two things apart
                area = np.zeros((p.H, p.W), bool)
                for q in own:
                    area |= _rmask(p, q)
                # a wet brush picks up the colour next to it along the edge of an area: a spill lies inside it
                area = cv2.erode(area.astype(np.uint8), np.ones((3, 3), np.uint8),
                                 iterations=max(2, int(0.005 * max(p.W, p.H)))).astype(bool)
                close = np.sqrt(((img - c) ** 2).sum(-1)) < 22.0
                if oc is not None:
                    close &= np.sqrt(((img - oc) ** 2).sum(-1)) > 40.0
                hit = close & area
                if _is_water(near) and n in regs:
                    # water mirrors what stands above it: the far thing's colour in its own columns is its
                    # reflection (the sun's path down the lake), not the far thing drawn in front
                    fb = coords.mask_bbox_px(_rmask(p, n))
                    if fb:
                        w_ = fb[2] - fb[0]
                        pad = max(0.5 * w_, 0.1 * p.W)
                        cols = np.zeros(p.W, bool)
                        cols[int(max(0, fb[0] - pad)):int(min(p.W, fb[2] + pad))] = True
                        hit &= ~cols[None, :]
                bx0 = d["parts"][n]["box"] if n in d["parts"] else d["subjects"][far]["box"]
                backdrop = (bx0[2] - bx0[0]) * (bx0[3] - bx0[1]) >= 0.8e6
                # a backdrop shares hues with what stands on it (an orange cat on a warm wall): only a real
                # spill counts there
                # a real spill is a patch, not scattered pixels or a thin line: shadow strands of a surface, or the
                # soft edge where two colours of the near thing blend through the far colour (a blue pot shading
                # into orange passes through the sill's brown), match the far colour without being it
                k = max(5, int(0.004 * max(p.W, p.H))) | 1
                hit = cv2.morphologyEx(hit.astype(np.uint8), cv2.MORPH_OPEN,
                                       np.ones((k, k), np.uint8)).astype(bool)
                if hit.sum() < (0.01 if backdrop else 0.0008) * p.W * p.H:
                    continue
                ys, xs = np.nonzero(hit)
                q = (int(np.median(xs)), int(np.median(ys)))
                bx = d["parts"][n]["box"] if n in d["parts"] else d["subjects"][far]["box"]
                full_bx = (bx[2] - bx[0]) * (bx[3] - bx[1]) >= 0.8e6
                if full_bx:     # a backdrop (a wall, the sky): a rectangle behind, not an oval over everything
                    bx = [int(round(v)) for v in bx]
                    cmd = (f"pixelcanvas band {n} \"{bx[0]},{bx[1]} {bx[2]},{bx[1]} {bx[2]},{bx[3]} {bx[0]},{bx[3]}\" "
                           f"--closed --color {sw} --behind {near}")
                else:
                    cmd = f"pixelcanvas band {n} --ellipse {coords.fmt_box(bx)} --color {sw} --behind {near}"
                out.append(_hole("depth", f"{n} ({far}, farther) is drawn over {near} (nearer) at {_u(p, q)}, "
                                 f"{hit.mean() * 100:.1f}% of the canvas: draw it behind (this also clears its colour "
                                 f"from {near}):  {cmd}   "
                                 f"(a shape that is not an oval: pixelcanvas band {n} \"x,y ...\" --closed --color {sw} "
                                 f"--behind {near})", cmd, f"{hit.mean() * 100:.1f}%", "0", n))
                break
    return out[:2]


# ============================================================================================ per stage

_HOLES: Dict[tuple, List[dict]] = {}


def holes(p, stage: str) -> List[dict]:
    """Every measured hole of ``stage`` (each blocks the stage's gate). Computed once per state of the picture (the
    FIX lines and the gate of one command ask for the same list)."""
    import json as _json
    try:
        key = (id(p), stage, int(p.state.get("step", 0)), id(dc_composite(p)),
               hash(_json.dumps(p.state.get("draw", {}), sort_keys=True, default=str)),
               hash(_json.dumps(p.state.get("regions", {}), sort_keys=True, default=str)))
    except Exception:  # noqa: BLE001
        key = None
    if key is not None and key in _HOLES:
        return [dict(h) for h in _HOLES[key]]
    out = _holes(p, stage)
    if key is not None:
        if len(_HOLES) > 16:
            _HOLES.clear()
        _HOLES[key] = [dict(h) for h in out]
    return out


def _holes(p, stage: str) -> List[dict]:
    d = dstate(p)
    style = p.state.get("style") or "paint"
    out: List[dict] = []
    if stage in ("plan",):
        return out
    try:
        if stage == "sketch":
            out += sketch_holes(p)
            return out
        has_lines = bool(d["lines"])
        if stage == "line":
            out += face_holes(p)
            out += eye_holes(p)
            out += stray_line_holes(p)
            return out
        if has_lines and stage in ("flat", "shade", "detail", "finish"):
            out += gap_holes(p)
        if stage in ("flat", "shade", "detail", "finish") and has_lines and d.get("fills"):
            out += unfilled_holes(p, stage)
        if stage == "flat":
            out += lock_holes(p)
            out += part_area_holes(p)
            out += connection_holes(p)
            out += neck_holes(p)
            out += background_holes(p)
            if style == "poster":
                out += fill_holes(p)
        if stage in ("block-in", "forms", "flat", "detail", "finish"):
            out += depth_holes(p)
        if stage in ("detail", "finish"):
            out += detail_holes(p)
            out += fill_holes(p)
            if dstate(p).get("surface"):
                from .draw_surface import gate_holes
                out += gate_holes(p, stage)
        if stage == "detail" and (style in ("paint", "photo") or not has_lines):
            out += face_holes(p)
        if stage == "finish":
            out += connection_holes(p)
            out += neck_holes(p)
            out += background_holes(p)
    except Exception as e:  # noqa: BLE001 - a check must never break a command
        out.append(_hole("check failed", f"(a check failed: {type(e).__name__}: {e})", "pixelcanvas next"))
        out[-1]["soft"] = True
    seen, uniq = set(), []
    for h in out:
        if h["text"] not in seen:
            seen.add(h["text"])
            uniq.append(h)
    return uniq


def findings(p, stage: str, limit: int = 3) -> List[str]:
    """The FIX lines of a stage: the holes its gate measures (the first ``limit``)."""
    return [h["text"] for h in holes(p, stage)][:limit]


# ============================================================================================ question

def question(p, stage: str, subject: Optional[str] = None) -> str:
    """One yes/no question about the VIEW where only the eye can judge (plan, sketch, block-in, forms, finish);
    the rest is measured and gated. Empty when there is nothing to ask."""
    d = dstate(p)
    parts = d["parts"]
    if stage == "plan":
        if not parts:
            return "Does each subject box hold the whole subject, with some room around it? If not: plan subject " \
                   "<name> --box ... again with a bigger box."
        return "Does each part box sit on the right place of its subject (head at the top, feet at the bottom)? " \
               "If not: plan part <name> --box ... again."
    if stage == "sketch":
        figs = [el for el in d["sketch"] if el["kind"] == "figure"]
        posed = [el for el in figs if el.get("rig") and el.get("rig_joints") == el.get("joints")]
        if posed:
            s = posed[0].get("subject")
            return (f"Does {s}'s pose read as what the request says (lean, which foot is down, where the hands go)? "
                    f"If not: pixelcanvas pose {s} --set \"<joint>: <angle> <degrees>\" (--show joints names them).")
        if figs:
            return (f"Does the mannequin {figs[0]['id']} stand the way the request needs (arms, legs, lean)? If not: "
                    f"pixelcanvas sketch joint {figs[0]['id']} <joint> x,y for each joint that is off.")
        return "Does the sketch show every subject's big shape at the right size? If not: pixelcanvas sketch move <id> " \
               "--by dx,dy or pixelcanvas sketch scale <id> 1.2."
    if stage in ("block-in",):
        return "Do the big shapes read from far away (squint at VIEW: sky, land, subject separate)? If not: " \
               "change the value of one band with pixelcanvas band <name> ... --color <darker>."
    if stage == "forms":
        return "Is every mass lighter on the side the light comes from? If not: pixelcanvas paint <name> --dir <angle> " \
               "again with --light <side>."
    if stage == "finish":
        return "Does the whole picture show what the request asks for, with nothing missing? If not: name the " \
               "missing thing and draw it."
    return ""
