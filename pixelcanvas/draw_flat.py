"""Flat colour without a reference: the enclosed areas of the line art, numbered; the model gives colours by area
number and palette name (``flat 3,5 --color skin``), or lets the plan decide (``flat auto``: each area takes the
colour of the smallest planned part whose box holds it, and the drawn clothes and hands take the areas inside
them). Areas become regions named after their part.

Areas are computed from the line layer (from the sketch layer while there is no line art yet), with gaps up to
``gap`` px closed, and then grown under the lines so a fill always reaches the line's centre: no white seams.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import coords
from .coords import UsageError
from .draw_core import composite_u8 as dc_composite, dilate, dstate, lum, px2u


def default_gap(p) -> int:
    return max(2, int(round(0.006 * max(p.W, p.H))))


def barrier_source(p) -> Tuple[Optional[np.ndarray], str]:
    line = p.layer("line", create=False)
    if line is not None and (line[..., 3] > 0.3).sum() > 0.0005 * p.W * p.H:
        return line[..., 3], "line"
    sk = p.layer("sketch", create=False)
    if sk is not None:
        return sk[..., 3], "sketch"
    return None, "none"


def compute_areas(p, gap: Optional[int] = None) -> Tuple[np.ndarray, List[dict], str]:
    """(labels [H,W] int32 0 = none, table, source). Cached by the barrier's content."""
    alpha, src = barrier_source(p)
    gap = default_gap(p) if gap is None else int(gap)
    H, W = p.H, p.W
    key = hashlib.sha1((alpha > 0.3).tobytes() if alpha is not None else b"none").hexdigest()[:16] + f"-{gap}-{src}-v2"
    cdir = p.root / "cache"
    cdir.mkdir(exist_ok=True)
    cf = cdir / f"areas-{key}.npz"
    if cf.exists():
        z = np.load(cf, allow_pickle=False)
        return z["labels"], json.loads(str(z["table"])), src
    if alpha is None:
        labels = np.ones((H, W), np.int32)
    else:
        bar = alpha > 0.3
        closed = dilate(bar, gap) if gap > 0 else bar
        free = ~closed
        lab, n = ndi.label(free)
        sizes = ndi.sum(free, lab, range(1, n + 1)) if n else np.zeros(0)
        min_area = max(12, int(0.0002 * H * W))
        keep = np.flatnonzero(sizes >= min_area) + 1
        good = np.isin(lab, keep)
        # thin closed shapes (a staff, a strap, a whisker band) vanish when gaps are closed: keep them as areas
        lab2, n2 = ndi.label(~bar)
        if n2:
            sizes2 = ndi.sum(~bar, lab2, range(1, n2 + 1))
            covered = ndi.sum(good, lab2, range(1, n2 + 1))
            nxt = int(lab.max()) + 1
            for k in np.flatnonzero((sizes2 >= min_area) & (covered < 0.05 * sizes2)) + 1:
                m2 = lab2 == k
                lab[m2] = nxt
                good |= m2
                nxt += 1
        # every pixel (lines, closed gaps, specks) goes to the nearest kept area
        if good.any():
            _, (iy, ix) = ndi.distance_transform_edt(~good, return_indices=True)
            lab = lab[iy, ix]
        else:
            lab = np.ones((H, W), np.int32)
        labels = lab.astype(np.int32)
    table = _table(p, labels)
    # renumber in reading order (top to bottom in bands, left to right)
    order = sorted(table, key=lambda r: (int(r["anchor"][1] // max(1, 0.06 * H)), r["anchor"][0]))
    remap = np.zeros(int(labels.max()) + 1, np.int32)
    for i, r in enumerate(order, 1):
        remap[r["old"]] = i
        r["n"] = i
    labels = remap[labels]
    table = sorted(order, key=lambda r: r["n"])
    for r in table:
        r.pop("old", None)
    np.savez_compressed(cf, labels=labels, table=json.dumps(table))
    return labels, table, src


def _table(p, labels) -> List[dict]:
    from .regions import label_anchor
    out = []
    H, W = labels.shape
    objs = ndi.find_objects(labels)
    for k, sl in enumerate(objs, 1):
        if sl is None:
            continue
        sub = labels[sl] == k
        if not sub.any():
            continue
        ax, ay = label_anchor(sub)
        y0, x0 = sl[0].start, sl[1].start
        full_box = (x0, y0, sl[1].stop, sl[0].stop)
        border = x0 == 0 or y0 == 0 or sl[1].stop == W or sl[0].stop == H
        out.append({"old": k, "anchor": [int(x0 + ax), int(y0 + ay)], "box": list(full_box),
                    "area": int(sub.sum()), "share": round(float(sub.sum()) / (H * W), 4), "border": bool(border)})
    return out


def area_mask(labels, nums) -> np.ndarray:
    return np.isin(labels, list(nums))


def parse_areas(text, table: List[dict], example: str) -> List[int]:
    nums = coords.parse_int_list(text, example)
    top = len(table)
    bad = [n for n in nums if n < 1 or n > top]
    if bad:
        raise UsageError(f"areas {', '.join(map(str, bad))} do not exist (1-{top}); see  pixelcanvas flat", example)
    return nums


# ============================================================================================ assignment

def auto_assign(p, labels, table) -> Dict[int, Tuple[str, str]]:
    """area -> (part, swatch): the plan boxes first (``box_assign``), then the drawn clothes and hands decide the
    areas round them (``draw_wear.wear_corrections``): a coat front inside a scarf's box is coat, a hanging scarf
    end is scarf, a hand is skin."""
    out = box_assign(p, labels, table)
    if any(r.get("wear") for r in dstate(p)["lines"]):
        from .draw_wear import wear_corrections
        out.update(wear_corrections(p, labels, table, out))
    out.update(outline_corrections(p, labels, table, out))
    out.update(eye_corrections(p, labels, table, out))
    out.update(lock_corrections(p, labels, table, out))
    if dstate(p).get("drawn2d"):
        from .draw_figure2d import drawn_corrections
        out.update(drawn_corrections(p, labels, table))
    from .scene3d import flat_filter        # a scene under the figures: its areas are not painted over
    return flat_filter(p, labels, out)


def eye_corrections(p, labels, table, assigned) -> Dict[int, Tuple[str, str]]:
    """A small area inside the closed outline of an eye part is that eye, whatever box or garment holds it: on a
    small figure the eye's box is smaller than the area its outline makes, and a hood round the face would take it."""
    d = dstate(p)
    from .draw_core import poly_mask, u2px
    inside: List[Tuple[str, np.ndarray]] = []
    for r in d["lines"]:
        part = str(r.get("part") or "")
        if r.get("wear") or not r.get("closed") or not part.startswith("eye") or part not in d["parts"]:
            continue
        P = u2px(p, r["path"])
        if len(P) >= 3:
            inside.append((part, poly_mask(p.H, p.W, P)))
    out: Dict[int, Tuple[str, str]] = {}
    if not inside:
        return out
    # the area that fills the inside of the eye's outline (off the lines) is the eye
    alpha, _ = barrier_source(p)
    off = ~(alpha > 0.3) if alpha is not None else np.ones((p.H, p.W), bool)
    shares = {r["n"]: r["share"] for r in table}
    for part, q in inside:
        lab = labels[q & off]
        lab = lab[lab > 0]
        if not len(lab):
            continue
        n = int(np.bincount(lab).argmax())
        if (lab == n).mean() < 0.5 or shares.get(n, 1.0) > 0.01:
            continue
        cur = assigned.get(n)
        if cur is None or not str(cur[0]).startswith(("eye", "iris", "pupil")):
            out[n] = (part, d["parts"][part].get("swatch") or part)
    return out


def outline_owner(p, labels, table) -> Dict[int, Tuple[str, float]]:
    """area -> (part, share): the planned part whose clothes lines (pixelcanvas wear) make most of the area's outline,
    when they make at least half of it. Hair locks and hand-drawn lines are left out."""
    d = dstate(p)
    from .draw_core import u2px
    recs = [r for r in d["lines"] if r.get("wear") and r.get("part") in d["parts"]]
    if not recs:
        return {}
    own = np.zeros((p.H, p.W), np.int32)
    names = [None]
    from .draw_line import base_width
    for r in recs:
        P = u2px(p, r["path"])
        if len(P) < 2:
            continue
        names.append(r["part"])
        th = max(2, int(round(base_width(p) * float(r.get("weight", 1.0)) + 3)))
        cv2.polylines(own, [np.round(P).astype(np.int32)], bool(r.get("closed")), len(names) - 1, th)
    ln = p.layer("line", create=False)
    line = ln[..., 3] > 0.3 if ln is not None else np.zeros((p.H, p.W), bool)
    out = {}
    k = np.ones((7, 7), np.uint8)
    for r in table:
        if r["share"] > 0.15:
            continue
        m = labels == r["n"]
        ring = (cv2.dilate(m.astype(np.uint8), k) > 0) & ~m & line
        if ring.sum() < 20:
            continue
        ids = own[ring]
        if not (ids > 0).any():
            continue
        cnt = np.bincount(ids, minlength=len(names))
        by: Dict[str, int] = {}
        for i in np.flatnonzero(cnt[1:]) + 1:
            by[names[i]] = by.get(names[i], 0) + int(cnt[i])
        part, v = max(by.items(), key=lambda t: t[1])
        share = v / float(ring.sum())
        if share >= 0.5:
            out[r["n"]] = (part, share)
    return out


def outline_corrections(p, labels, table, assigned) -> Dict[int, Tuple[str, str]]:
    """An area the clothes lines of one part outline is that part (the back end of a scarf below the hair's box is
    scarf, not hair)."""
    d = dstate(p)
    out = {}
    shares = {r["n"]: r["share"] for r in table}
    for n, (part, share) in outline_owner(p, labels, table).items():
        cur = assigned.get(n)
        if shares.get(n, 0) < 0.003:
            continue            # buttons and knots: too small to tell by their outline
        if cur is None or (cur[0] != part and cur[0] not in ("background",) and share >= 0.75):
            out[n] = (part, d["parts"][part].get("swatch") or part)
    return out


def lock_mask(p, part: str, front_only: bool = False) -> Optional[np.ndarray]:
    """The hair the locks of ``part`` cover (union of the lock shapes from the  pixelcanvas locks  record); with
    ``front_only`` only the locks in front of the face (fringe and side locks)."""
    rec = (dstate(p).get("locks") or {}).get(part)
    if not rec or not rec.get("locks"):
        return None
    from .draw_core import poly_mask, u2px
    from .draw_hair import FRONT, lock_shape
    S = float(max(p.W, p.H))
    m = np.zeros((p.H, p.W), bool)
    for k in rec["locks"]:
        if front_only and k.get("layer") not in FRONT:
            continue
        if k.get("poly"):
            if not front_only:
                m |= poly_mask(p.H, p.W, u2px(p, k["poly"]))
            continue
        path = u2px(p, k["path"])
        w = np.asarray(k.get("width") or [0.0], float) * S / 1000.0
        if len(path) < 2 or len(w) != len(path) or w.max() <= 0:
            continue
        poly = lock_shape(path, w)[2]
        m |= poly_mask(p.H, p.W, poly)
    return m if m.any() else None


def lock_corrections(p, labels, table, assigned) -> Dict[int, Tuple[str, str]]:
    """Areas that lie inside a hair part's locks are hair, whatever box holds them: a fringe lock closed inside
    the head's outline is hair, not skin (the forehead between the locks stays skin)."""
    d = dstate(p)
    out: Dict[int, Tuple[str, str]] = {}
    for part in (d.get("locks") or {}):
        if part not in d["parts"]:
            continue
        sw = d["parts"][part].get("swatch") or part
        front = lock_mask(p, part, front_only=True)
        if front is None:
            continue
        for r in table:
            cur = assigned.get(r["n"])
            if cur is not None and cur[0] == part:
                continue
            m = labels == r["n"]
            if r["share"] > 0.2:
                continue
            # only the locks in front of the face: the back hair's shapes run behind the body and the clothes
            if float(front[m].mean()) >= 0.6:
                out[r["n"]] = (part, sw)
    return out


def box_assign(p, labels, table) -> Dict[int, Tuple[str, str]]:
    """area -> (part, swatch) from the plan: the smallest part box holding the area's anchor; the area that
    touches the border outside every subject -> background."""
    from .draw_plan import parts_at, subject_at
    d = dstate(p)
    out = {}
    for r in table:
        u = px2u(p, [r["anchor"]])[0]
        hits = parts_at(d, u)
        bx_ = px2u(p, [r["box"][:2], r["box"][2:]]).reshape(-1)
        edges = sum([bx_[0] <= 2, bx_[1] <= 2, bx_[2] >= 998, bx_[3] >= 998])
        if r["border"] and edges >= 3:
            # an area that runs round the picture is the background, whatever box its centre falls in
            out[r["n"]] = ("background", "background")
            continue
        if hits:
            # prefer the smallest part box that holds most of the area, over one that only holds its centre
            bx = px2u(p, [r["box"][:2], r["box"][2:]]).reshape(-1)
            barea = max(1.0, (bx[2] - bx[0]) * (bx[3] - bx[1]))

            def share(n):
                b = d["parts"][n]["box"]
                iw = max(0.0, min(b[2], bx[2]) - max(b[0], bx[0]))
                ih = max(0.0, min(b[3], bx[3]) - max(b[1], bx[1]))
                return iw * ih / barea
            full = [n for n in hits if share(n) >= 0.7]
            part = full[0] if full else max(hits, key=share)
            from .workflow_draw import is_text
            if is_text(part):
                # lettering is set with pixelcanvas text on top: its box keeps the colour around it
                out[r["n"]] = ("background", "background")
                continue
            out[r["n"]] = (part, d["parts"][part]["swatch"])
        elif subject_at(d, u) is None or r["border"] and r["share"] > 0.15:
            out[r["n"]] = ("background", "background")
    return out


def filled_share(p, labels, n) -> float:
    fl = p.layer("flat", create=False)
    if fl is None:
        return 0.0
    m = labels == n
    return float((fl[..., 3][m] > 0.5).mean()) if m.any() else 0.0


def unfilled(p, labels, table) -> List[dict]:
    fl = p.layer("flat", create=False)
    a = fl[..., 3] if fl is not None else np.zeros(labels.shape, np.float32)
    out = []
    for r in table:
        m = labels == r["n"]
        if m.any() and float((a[m] > 0.5).mean()) < 0.6:
            out.append(r)
    return out


def split_pieces(p, labels, table, min_share: float = 0.0004) -> List[dict]:
    """Areas that show an old fill but hold no fill point: a later line split them off a filled area, and they
    need their own colour again."""
    fl = p.layer("flat", create=False)
    d = dstate(p)
    if fl is None or not d.get("fills"):
        return []
    own = {int(labels[int(np.clip(f["pt"][1], 0, p.H - 1)), int(np.clip(f["pt"][0], 0, p.W - 1))])
           for f in d["fills"]}
    a = fl[..., 3]
    return [r for r in table if r["n"] not in own and r["share"] >= min_share
            and float((a[labels == r["n"]] > 0.5).mean()) >= 0.6]


def assigned_at(p, pt_px) -> Optional[dict]:
    """The latest flat assignment whose point lies in the same area as pt."""
    d = dstate(p)
    for f in reversed(d["fills"]):
        if f["pt"] == list(pt_px):
            return f
    return None


def rebuild_regions(p, labels, table) -> Dict[str, np.ndarray]:
    """Region per part = union of the areas assigned to it (by the points stored in draw.fills)."""
    d = dstate(p)
    parts: Dict[str, np.ndarray] = {}
    owner: Dict[int, str] = {}
    for f in d["fills"]:
        x, y = int(np.clip(f["pt"][0], 0, p.W - 1)), int(np.clip(f["pt"][1], 0, p.H - 1))
        n = int(labels[y, x])
        if n:
            owner[n] = f["part"]
    for n, part in owner.items():
        m = labels == n
        parts[part] = parts.get(part, np.zeros(labels.shape, bool)) | m
    try:
        from .draw_face import apply_marks          # markings (muzzle, tail tip) stay cut out of their parent
        parts = apply_marks(p, parts)
    except Exception:  # noqa: BLE001
        pass
    return parts


def fill_op(mask: np.ndarray, rgb) -> object:
    from .ops import Fill, mask_to_rle
    # one pixel of overlap with the neighbours, so two areas of one colour meet without a seam
    m = cv2.dilate(np.asarray(mask, np.uint8), np.ones((3, 3), np.uint8)) > 0
    return Fill(mask_rle=mask_to_rle(m), color=tuple(int(c) for c in rgb), feather=1.0)


def hole_mask(p, pt_px) -> Tuple[Optional[np.ndarray], Optional[str], Optional[tuple]]:
    """(mask, neighbour region, neighbour colour) of the background-coloured piece under ``pt_px``: pixels of the
    background swatch's colour joined to the point, bounded by lines and other colours. None when the point is on
    the picture's main background (the largest such piece)."""
    from .draw_plan import swatch
    d = dstate(p)
    sw = d["palette"]["swatches"]
    key = next((k for k in ("background", "sky") if k in sw), None)
    if key is None:
        return None, None, None
    bgc = np.array(swatch(p, key), float)
    img = dc_composite(p, "fills").astype(np.float32)
    near = np.sqrt(((img - bgc) ** 2).sum(-1)) < 14.0
    ln = p.layer("line", create=False)
    line = ln[..., 3] > 0.3 if ln is not None else np.zeros((p.H, p.W), bool)
    cand = (near & ~line).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    if n <= 1:
        return None, None, None
    x, y = int(np.clip(pt_px[0], 0, p.W - 1)), int(np.clip(pt_px[1], 0, p.H - 1))
    k = int(lab[y, x])
    if k == 0:
        r = max(3, int(0.012 * max(p.W, p.H)))
        win = lab[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
        ks = win[win > 0]
        if not len(ks):
            return None, None, None
        k = int(np.bincount(ks).argmax())
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if k == main:
        return None, None, None
    m = lab == k
    ring = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~m & ~near & ~line
    fl = p.layer("flat", create=False)
    colour = None
    if ring.any():
        src = fl[..., :3][ring & (fl[..., 3] > 0.5)] if fl is not None else np.zeros((0, 3))
        if len(src) == 0:
            src = img[ring] / 255.0
        q = np.round(np.clip(src, 0, 1) * 255).astype(np.int32)
        keys, cnt = np.unique(q[:, 0] * 65536 + q[:, 1] * 256 + q[:, 2], return_counts=True)
        kk = int(keys[int(np.argmax(cnt))])
        colour = (kk >> 16, (kk >> 8) & 255, kk & 255)
    best, val = None, 0
    for name in p.state.get("regions", {}):
        if name in ("background", "bg", "sky"):
            continue
        try:
            v = int((p.region_mask(name) & ring).sum())
        except Exception:  # noqa: BLE001
            continue
        if v > val:
            best, val = name, v
    grown = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~(line & ~m)
    return grown | m, best, colour


# ============================================================================================ checks

def close_values(p, labels, table) -> List[str]:
    """Neighbouring areas of different colours whose values are too close to read apart in grey."""
    fl = p.layer("flat", create=False)
    if fl is None:
        return []
    d = dstate(p)
    rgb = (np.clip(fl[..., :3], 0, 1) * 255).astype(np.uint8)
    cols = {}
    for r in table:
        m = labels == r["n"]
        if (fl[..., 3][m] > 0.5).mean() < 0.6:
            continue
        cols[r["n"]] = tuple(int(v) for v in np.median(rgb[m], axis=0))
    out = []
    seen = set()
    k = np.ones((3, 3), np.uint8)
    names = {}
    for f in d["fills"]:
        x, y = int(np.clip(f["pt"][0], 0, p.W - 1)), int(np.clip(f["pt"][1], 0, p.H - 1))
        names[int(labels[y, x])] = f.get("swatch") or "?"
    for n in cols:
        m = labels == n
        ring = (cv2.dilate(m.astype(np.uint8), k, iterations=3) > 0) & ~m
        for o in np.unique(labels[ring]):
            o = int(o)
            if o == n or o not in cols or (min(n, o), max(n, o)) in seen:
                continue
            seen.add((min(n, o), max(n, o)))
            if cols[n] == cols[o] or names.get(n) == names.get(o):
                continue
            dl = abs(lum(cols[n]) - lum(cols[o]))
            if dl < 7:
                out.append((dl, n, o))
    out.sort()
    lines = []
    for dl, n, o in out[:2]:
        a, b = names.get(n, str(n)), names.get(o, str(o))
        lines.append(f"areas {n} ({a}) and {o} ({b}) touch and have almost the same value (L {dl:.0f} apart): "
                     f"they read as one shape in grey; make one darker:  pixelcanvas palette set {b} <darker colour>")
    return lines
