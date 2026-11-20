"""``pixelcanvas draw``: the 2D drawing over the 3D guide, the way an anime illustrator draws over a pose model.

The 3D scene (``pose``, ``dress``, ``hair3d``, ``block``) says where every limb goes and what is in front; ``draw``
draws each part again in 2D with anime proportions (anime2d) as editable line records (few control points, smooth,
tapered, heavier on the silhouette), lays the face on the head's face frame, and makes every part an area the
colour tools know. ``flat auto`` fills the areas, ``shade <who>`` lays cel shadows as clean shapes.

  pixelcanvas draw [<who>|all|scene] [--part hair|face|head|torso|arm-r|arm-l|hand-r|hand-l|leg-r|leg-l|foot-r|foot-l|
                   clothes|props]
  pixelcanvas shade <who>|all            (on a drawn figure: two-tone cel shadows, hair in three tones, a highlight
                                          band on the hair, cast shadows from the hair, chin, arms and skirt)
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import anime2d as A2
from . import coords
from . import draw_core as dc
from . import draw_pose as DP
from . import face3d as FC
from . import figure3d as F3
from . import shape2d as S2
from .coords import UsageError

USAGE = {"draw": "pixelcanvas draw girl   (or: draw all | draw scene | draw girl --part hair)"}
GROUPS = ("hair", "face", "head", "torso", "arm-r", "arm-l", "hand-r", "hand-l", "leg-r", "leg-l", "foot-r", "foot-l",
          "clothes", "props")
LAB_MAX = 1600
NPZ = "draw/draw2d.npz"
W_SIL, W_IN, W_HAIR, W_DET, W_BLOCK = 1.35, 0.7, 0.6, 0.42, 0.55     # the outer silhouette about 2x the inner lines


# ============================================================================================ the scene in 2D

def scene_key(p, subj: str = "@scene") -> str:
    """What a drawing of ``subj`` was made from: its posed figure (with what it holds) and the camera; for the
    blocks ("@scene") the blocks and the camera."""
    d = dc.dstate(p)
    recs = []
    for el in d["sketch"]:
        if el["kind"] == "horizon" and el.get("camera"):
            recs.append(el)
        elif subj == "@scene" and el["kind"] == "block":
            recs.append(el)
        elif el["kind"] == "prop" and el.get("subject") == subj:
            recs.append(el)
        elif DP.is_rigged(el) and el.get("subject") == subj:
            recs.append(el)
    recs = [{k: v for k, v in el.items() if k not in ("n", "face2d")} for el in recs]
    return hashlib.sha1(json.dumps([p.W, p.H, recs], sort_keys=True, default=str).encode()).hexdigest()[:16]


def _clip_rect(P: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """A 2D polygon clipped to a rectangle (Sutherland-Hodgman)."""
    def clip(poly, inside, cross):
        out = []
        n = len(poly)
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            ia, ib = inside(a), inside(b)
            if ia:
                out.append(a)
            if ia != ib:
                out.append(cross(a, b))
        return out
    poly = [np.asarray(q, float) for q in P]
    for k, lim, lo in ((0, x0, True), (0, x1, False), (1, y0, True), (1, y1, False)):
        if not poly:
            break

        def inside(q, k=k, lim=lim, lo=lo):
            return q[k] >= lim if lo else q[k] <= lim

        def cross(a, b, k=k, lim=lim):
            t = (lim - a[k]) / (b[k] - a[k]) if b[k] != a[k] else 0.0
            return a + t * (b - a)
        poly = clip(poly, inside, cross)
    return np.asarray(poly) if poly else np.zeros((0, 2))


def block_parts(p, cam) -> List[dict]:
    from .render3d import corners as _corners  # noqa: F401
    out = []
    for el in DP.blocks_rec(p):
        if el.get("pos") is None:
            continue
        for k, pr in enumerate(DP._block(el).prims(el["id"])):
            if pr["t"] == "box":
                A = np.asarray(pr["A"], float)
                c = np.asarray(pr["c"], float)
                for ax in range(3):
                    for sg in (1.0, -1.0):
                        n = sg * A[:, ax]
                        if np.linalg.norm(n) < 1e-6:
                            continue
                        fc = c + n
                        nu = n / np.linalg.norm(n)
                        if nu @ (cam.C - fc) <= 0:
                            continue
                        o1, o2 = [A[:, j] for j in range(3) if j != ax]
                        quad = np.array([fc + o1 + o2, fc - o1 + o2, fc - o1 - o2, fc + o1 - o2])
                        out.append({"name": f"{el['name']}-{k}-{ax}{'+' if sg > 0 else '-'}", "kind": "plane",
                                    "mat": el["name"], "subj": "@" + el["name"], "group": "scene", "joints": {},
                                    "rad": 1.0, "details": [], "bias": 0.0, "line": True, "pts": quad,
                                    "normal": nu, "block": el["name"]})
            else:
                from .anime2d import _prim_samples
                P_, N_ = _prim_samples(pr, 300)
                out.append({"name": f"{el['name']}-{k}", "kind": "blob", "mat": el["name"], "subj": "@" + el["name"],
                            "group": "scene", "joints": {}, "rad": 0.3, "details": [], "bias": 0.0, "line": True,
                            "pts": P_, "nrm": N_, "hull": True, "block": el["name"]})
    return out


def figure_parts(p, el) -> List[dict]:
    from .draw_render import look_of
    body, J, F = DP._fig_state(el)
    held = [pr for pr in DP.props_rec(p) if pr.get("subject") == el["subject"]]
    cam = DP.camera_of(p)
    return A2.figure_parts(el, body, J, F, look_of(el), el["subject"], held, eye=cam.C, cam=cam)


def build(p) -> dict:
    """Every figure and block as 2D parts: outlines (canvas px) and the z-buffer of them (label res)."""
    cam = DP.camera_of(p)
    S = float(max(p.W, p.H))
    s = min(1.0, LAB_MAX / S)
    Wl, Hl = max(1, int(round(p.W * s))), max(1, int(round(p.H * s)))
    s = Wl / float(p.W)
    R = S2.Raster(Wl, Hl, s)
    parts: List[dict] = []
    for el in DP.figures(p):
        parts += figure_parts(p, el)
    parts += block_parts(p, cam)
    outs: List[Optional[dict]] = []
    decals = []
    for k, pt in enumerate(parts):
        o = None
        try:
            o = _outline(p, cam, pt, R)
        except Exception:  # noqa: BLE001 - a part that cannot be drawn is left out, the rest draws
            o = None
        outs.append(o)
        if o is None:
            continue
        b = pt.get("bias", 0.0)
        if pt.get("decal"):
            decals.append(k)
        elif o.get("hit") is not None:
            R.add_hits(k, o["hit"], b)
        elif pt["kind"] == "tube":
            R.add_tube(k, o, b)
        elif pt["kind"] == "plane":
            R.add_plane(k, o["poly"], cam, pt["pts"][0], pt["normal"], b)
        else:
            R.add_blob(k, o["poly"], o["spx"], o["sz"], o["snrm"], b)
    for k in decals:
        pt = parts[k]
        over = [j for j, q in enumerate(parts) if q["subj"] == pt["subj"] and q["name"] in pt["decal"]]
        R.add_decal(k, outs[k]["poly"], over)
    # joints on the canvas, the radius of each part in px
    for pt, o in zip(parts, outs):
        if o is None:
            continue
        js = {}
        for n, P in pt["joints"].items():
            if P is not None:
                q, z = cam.project(np.asarray(P, float)[None])
                js[n] = q[0]
        o["j2d"] = js
        c3 = _centre3(pt)
        q, z = cam.project(c3[None])
        o["rpx"] = float(pt["rad"]) * cam.f / max(float(z[0]), 0.05)
        o["zc"] = float(z[0])
    return {"parts": parts, "outs": outs, "R": R, "cam": cam, "s": s}


def _centre3(pt: dict) -> np.ndarray:
    if pt["kind"] == "tube":
        return np.mean([c for c, _ in pt["rings"]], 0)
    if pt.get("flat3") is not None:
        return np.asarray(pt["flat3"], float).mean(0)
    if pt.get("canopy") is not None:
        return np.asarray(pt["canopy"]["c"], float)
    return np.asarray(pt["pts"], float).mean(0)


def _outline(p, cam, pt: dict, R) -> Optional[dict]:
    W, H = p.W, p.H
    m = 0.5 * max(W, H)
    if pt["kind"] == "tube":
        o = S2.tube_outline(cam, pt)
        if o is None:
            return None
        o["poly"] = _clip_rect(o["poly"], -m, -m, W + m, H + m)
        if len(o["poly"]) < 3:
            return None
        if pt.get("prims"):
            P = o["poly"]
            pad = 0.25 * (P.max(0) - P.min(0)).max() + 4
            o["hit"] = R.hits(cam, pt["prims"], (P[:, 0].min() - pad, P[:, 1].min() - pad, P[:, 0].max() + pad,
                                                  P[:, 1].max() + pad))
            if o["hit"] is None:
                return None
        return o
    if pt.get("prims"):
        q, zz = cam.project(pt["pts"])
        if (zz <= 0.05).any():
            return None
        pad = 0.15 * (q.max(0) - q.min(0)).max() + 4
        hit = R.hits(cam, pt["prims"], (q[:, 0].min() - pad, q[:, 1].min() - pad, q[:, 0].max() + pad,
                                        q[:, 1].max() + pad))
        if hit is None:
            return None
        poly = S2.mask_outline(hit[0], hit[1], R.s)
        if poly is None:
            return None
        return {"poly": poly, "hit": hit}
    if pt["kind"] == "plane":
        P3 = S2.clip_near(cam, pt["pts"])
        if len(P3) < 3:
            return None
        q, _ = cam.project(P3)
        q = _clip_rect(q, -m, -m, W + m, H + m)
        return {"poly": q} if len(q) >= 3 else None
    if pt.get("flat3") is not None:
        q, zz = cam.project(pt["flat3"])
        if (zz <= 0.05).any():
            return None
        n3 = np.asarray(pt["normal"], float)
        ns = S2.unit(np.array([n3 @ cam.right, -(n3 @ cam.up), -(n3 @ cam.fwd)]))
        c2 = q.mean(0)
        spx = np.vstack([q, c2[None], 0.5 * (q + c2)])
        sz = np.concatenate([zz, [zz.mean()], zz])
        return {"poly": q, "spx": spx, "sz": sz, "snrm": np.repeat(ns[None], len(spx), 0)}
    if pt.get("canopy") is not None:
        return _canopy_outline(cam, pt)
    px, z, nrm = S2.front_samples(cam, pt["pts"], pt["nrm"], slack=0.0 if pt.get("splat") else 0.15)
    if len(px) < 3:
        return None
    if pt.get("poly3") is not None:
        q, zz = cam.project(pt["poly3"])
        if (zz <= 0.05).any():
            return None
        poly = q
    elif pt.get("splat"):
        poly = _splat_outline(px, R.s, cam.f / max(float(np.median(z)), 0.05) * float(pt["rad"]))
        if poly is None:
            return None
    else:
        poly = S2._hull(px)
    return {"poly": poly, "spx": px, "sz": z, "snrm": nrm}


def _canopy_outline(cam, pt: dict) -> Optional[dict]:
    """The canopy seen from the camera: the hull of its top and rib tips with the rim scalloped between the tips,
    the ribs that face the viewer as lines."""
    cn = pt["canopy"]
    tips, apex, c, up = cn["tips"], cn["apex"], cn["c"], cn["up"]
    qt, zt = cam.project(tips)
    qa, za = cam.project(apex[None])
    if (zt <= 0.05).any() or za[0] <= 0.05:
        return None
    allp = np.vstack([qt, qa])
    hull = cv2.convexHull(allp.astype(np.float32), returnPoints=False).ravel().tolist()
    order = hull[::-1] if cv2.contourArea(allp[hull].astype(np.float32), oriented=True) < 0 else hull
    poly = []
    n = len(tips)
    for i, j in zip(order, order[1:] + order[:1]):
        poly.append(allp[i])
        if i < n and j < n and (abs(i - j) == 1 or abs(i - j) == n - 1):
            m = 0.5 * (allp[i] + allp[j])
            poly.append(m + (qa[0] - m) * 0.08)              # the rim sags between two ribs
    poly = np.asarray(poly)
    P = S2._catmull(np.vstack([poly, poly[:2]]), 10 * len(poly))[:-10] if len(poly) > 3 else poly
    toward = (tips - c) @ S2.unit(cam.C - c)
    ribs = [np.vstack([qa[0], qt[i]]) for i in range(n) if toward[i] > -0.05 * np.linalg.norm(tips[0] - c)]
    n3 = S2.unit(up)
    ns = S2.unit(np.array([n3 @ cam.right, -(n3 @ cam.up), -(n3 @ cam.fwd)]))
    spx = np.vstack([qt, qa])
    sz = np.concatenate([zt, za])
    return {"poly": P, "spx": spx, "sz": sz, "snrm": np.repeat(ns[None], len(spx), 0), "details_px": ribs}


def _splat_outline(px: np.ndarray, s: float, size_px: float) -> Optional[np.ndarray]:
    """The smooth outline of the area a cloud of samples covers (a hair cap seen from the front: the forehead
    stays out)."""
    P = px * s
    x0, y0 = P.min(0) - 8
    x1, y1 = P.max(0) + 8
    w, h = int(x1 - x0) + 1, int(y1 - y0) + 1
    if w * h > 6e6 or w < 2 or h < 2:
        return None
    m = np.zeros((h, w), np.uint8)
    xi = np.round(P[:, 0] - x0).astype(int)
    yi = np.round(P[:, 1] - y0).astype(int)
    m[yi, xi] = 1
    # samples on a sphere of radius size_px: their spacing on the screen
    gap = max(2, int(round(size_px * s * 0.16)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))
    m = cv2.dilate(m, ker)
    m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap + 1, gap + 1)))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)
    c = S2.smooth_closed(c, max(1.5, gap * 0.8))
    return (c + [x0, y0]) / s


# ============================================================================================ strokes

def _bend(a: dict, b: dict, shared) -> float:
    """Degrees two limb segments bend at their shared joint (0 = straight on)."""
    for n in shared:
        J = a["joints"].get(n)
        if J is None:
            continue
        oa = [v for m, v in a["joints"].items() if m != n and v is not None]
        ob = [v for m, v in b["joints"].items() if m != n and v is not None]
        if not oa or not ob:
            continue
        u1, u2 = S2.unit(np.asarray(oa[0]) - J), S2.unit(np.asarray(ob[0]) - J)
        return 180.0 - math.degrees(math.acos(float(np.clip(u1 @ u2, -1, 1))))
    return 0.0


def _keep_fn(sc: dict, k: int, R):
    parts, outs = sc["parts"], sc["outs"]
    pk, ok = parts[k], outs[k]

    def keep(q: int, P, z_in: float, z_out: float) -> int:
        if q < 0:
            return 1
        pq, oq = parts[q], outs[q]
        if pq["subj"] != pk["subj"]:
            return 1
        if pk["kind"] == "plane" or pq["kind"] == "plane":
            return 2
        hair = pk["mat"] == "hair" and pq["mat"] == "hair"
        if pq["mat"] == pk["mat"]:
            shared = set(pk["joints"]) & set(pq["joints"])
            if shared:
                pos = [o["j2d"][n] for o in (ok, oq) for n in shared if n in o.get("j2d", {})]
                if not pos:
                    pos = list(ok.get("j2d", {}).values()) + list(oq.get("j2d", {}).values())
                r = max(ok["rpx"], oq["rpx"]) if hair else min(ok["rpx"], oq["rpx"])
                near = any(float(np.hypot(*(np.asarray(P) - q2))) < 1.9 * r for q2 in pos) if pos else True
                # a joint bent hard (a raised knee, a folded arm): the near segment overlaps the far one and its
                # end is drawn (the knee over the shin)
                if near and not hair and _bend(pk, pq, shared) > 70.0 and z_in < z_out:
                    return 2
                # where two parts of one material meet at their joint and lie at one depth: one continuous shape
                if near and abs(z_out - z_in) < 0.7 * min(pk["rad"], pq["rad"]):
                    return 0
        if hair and abs(z_out - z_in) < 1.2 * max(pk["rad"], pq["rad"]):
            return 0            # locks side by side in one bundle: one mass (strand lines come from the details)
        return 3 if hair else 2
    return keep


def strokes(p, sc: dict, subj_keep, groups: Optional[set]) -> List[dict]:
    """Stroke specs (canvas px) for the parts of the subjects in ``subj_keep`` (and groups)."""
    R = sc["R"]
    S = float(max(p.W, p.H))
    step = max(1.0, 0.75 / R.s)
    min_len = 0.004 * S
    out = []
    for k, (pt, o) in enumerate(zip(sc["parts"], sc["outs"])):
        if o is None or not pt.get("line", True) or pt["subj"] not in subj_keep:
            continue
        if groups and pt["group"] not in groups:
            continue
        block = pt["kind"] == "plane" or pt.get("block")
        for st in S2.outline_strokes(o["poly"], k, R, _keep_fn(sc, k, R), step, min_len * 1.5,
                                     sigma=0.6 if block else 2.5, bridge=int(round(0.006 * S / step))):
            c = st["cls"]
            if block:
                w = W_BLOCK
            elif c == 1:
                w = W_SIL
            elif c == 3 or pt["mat"] == "hair":
                w = W_HAIR
            else:
                w = W_IN
            out.append({"pts": st["pts"], "closed": st["closed"], "w": w, "part": pt, "k": k})
        cam = sc["cam"]
        for q in o.get("details_px", []):
            dense = S2.resample(q, step)
            for run in S2.detail_strokes(dense, k, R, step, min_len * 0.7):
                out.append({"pts": run, "closed": False, "w": W_DET, "part": pt, "k": k, "detail": True})
        for det in pt.get("details", []):
            P3 = np.asarray(det["pts"], float)
            if det.get("nrm") is not None:
                ok_ = (np.asarray(det["nrm"]) * S2.unit(cam.C - P3)).sum(1) > 0.05
                if ok_.sum() < 2:
                    continue
                P3 = P3[ok_]
            q, z = cam.project(P3)
            if (z <= 0.05).any():
                continue
            dense = S2._catmull(q, max(8, len(q) * 6))
            for run in S2.detail_strokes(dense, k, R, step, min_len * 0.7):
                out.append({"pts": run, "closed": False, "w": det.get("w", W_DET), "part": pt, "k": k,
                            "detail": True})
    return out


# ============================================================================================ the face

def face_specs(p, sc: dict, el: dict) -> Tuple[list, List[dict]]:
    """Face fills (Fill ops for the paint layer) and line specs on the 2D head of figure ``el``."""
    from .draw_render import HAIR, _col, look_of
    from .ops import Fill, mask_to_rle
    lk = look_of(el)
    fc = lk.get("face") or {"expression": "neutral"}
    body, J, F = DP._fig_state(el)
    rig = F3.Rig(body, J, F)
    cam = sc["cam"]
    R = sc["R"]
    heads = [k for k, pt in enumerate(sc["parts"]) if pt["subj"] == el["subject"] and pt["name"] == "head"]
    if not heads or sc["outs"][heads[0]] is None:
        return [], []
    hk = heads[0]
    iris = _col(p, fc.get("eyes"), "#4a78c0")
    lash = dc.shadow_of(_col(p, (lk.get("hair") or {}).get("color"), HAIR), 1.6)
    lash = tuple(int(v * 0.55) for v in lash)
    specs = FC.project(rig, cam, FC.features(fc.get("expression", "neutral"), iris, lash))
    # where the head shows (not under the bangs): full-res mask of the head's label
    lab = R.lab == hk
    head_m = cv2.resize(lab.astype(np.uint8), (p.W, p.H), interpolation=cv2.INTER_NEAREST) > 0
    fills, lines = [], []
    clips: Dict[str, np.ndarray] = {}
    for sp in specs:
        P = sp["px"]
        vis = sp["vis"] & ((R.at(P) == hk) | bool(sp.get("float")))
        sp["vis2"] = vis
        if sp.get("fill") is None:
            continue
        if vis.mean() < (0.25 if sp["kind"] == "tear" else 0.5):
            continue
        m = dc.poly_mask(p.H, p.W, P)
        if not sp.get("float"):
            m &= head_m
        if sp.get("clip"):
            cm = clips.get(sp["clip"])
            if cm is None:
                continue
            m &= cm
        clips[sp["name"]] = m
        if m.sum() < 3:
            continue
        col = tuple(int(c) for c in sp["fill"])
        if sp["kind"] == "iris":
            # two flat tones (a darker top under the lid), not a gradient
            y0, y1 = float(P[:, 1].min()), float(P[:, 1].max())
            fills.append(Fill(mask_rle=mask_to_rle(m), color=col, feather=0.6))
            top = m.copy()
            top[int(y0 + 0.45 * (y1 - y0)):] = False
            fills.append(Fill(mask_rle=mask_to_rle(top), color=dc.shadow_of(col, 1.3), feather=0.6))
        elif sp["kind"] == "blush":
            fills.append(Fill(mask_rle=mask_to_rle(m), color=col, feather=0.003 * max(p.W, p.H),
                              opacity=float(sp.get("alpha", 0.4))))
        else:
            fills.append(Fill(mask_rle=mask_to_rle(m), color=col, feather=0.6,
                              opacity=0.85 if sp["kind"] == "tear" else 1.0))
    for sp in specs:
        if sp["weight"] <= 0:
            continue
        P, vis = sp["px"], sp["vis2"]
        for run in S2.runs(vis, False):
            if len(run) >= 2:
                lines.append({"name": sp["name"], "px": P[run], "closed": bool(sp["closed"] and vis.all()),
                              "weight": sp["weight"], "taper": sp["taper"], "rgb": sp.get("rgb")})
    return fills, lines


# ============================================================================================ regions, colours

def region_of(p, pt: dict) -> str:
    from .draw_render import region_name
    if pt.get("block"):
        return pt["block"]
    if str(pt["mat"]).startswith("prop:"):
        return pt["mat"].split(":", 1)[1]
    return region_name(p, pt["subj"], pt["mat"])


def colour_of(p, pt: dict, cache: dict) -> Tuple[int, int, int]:
    from .draw_render import _col, block_colour, material_colours
    if pt.get("block"):
        el = next((b for b in DP.blocks_rec(p) if b["name"] == pt["block"]), None)
        return block_colour(p, el) if el else (200, 192, 180)
    if str(pt["mat"]).startswith("prop:"):
        nm = pt["mat"].split(":", 1)[1]
        pe = next((x for x in DP.props_rec(p) if x.get("name") == nm), {})
        return _col(p, pe.get("color"), {"sword": "#c8ccd4", "umbrella": "#5a7ab8"}.get(pe.get("type"), "#8a6a50"))
    subj = pt["subj"]
    if subj not in cache:
        el = next(e for e in DP.figures(p) if e["subject"] == subj)
        cache[subj] = material_colours(p, el)
    return cache[subj].get(pt["mat"], cache[subj].get("trim", (128, 128, 128)) if pt["mat"] == "button"
                           else (128, 128, 128))


def _table(p, sc: dict) -> List[dict]:
    cache: dict = {}
    out = []
    for pt, o in zip(sc["parts"], sc["outs"]):
        out.append({"name": pt["name"], "subj": pt["subj"], "mat": pt["mat"], "group": pt["group"],
                    "region": region_of(p, pt), "rgb": list(colour_of(p, pt, cache)), "drawn": o is not None,
                    "block": pt.get("block")})
    return out


def save_maps(p, sc: dict, table: List[dict]) -> None:
    R = sc["R"]
    n = np.clip(np.round(R.nrm * 127), -127, 127).astype(np.int8)
    np.savez_compressed(p.root / NPZ, lab=R.lab, z=np.where(np.isfinite(R.z), R.z, 1e4).astype(np.float32), nrm=n,
                        s=np.array([R.s]), table=json.dumps(table))


def load_maps(p) -> Optional[dict]:
    f = p.root / NPZ
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=False)
    return {"lab": z["lab"], "z": z["z"], "nrm": z["nrm"].astype(np.float32) / 127.0, "s": float(z["s"][0]),
            "table": json.loads(str(z["table"]))}


# ============================================================================================ the command

def _subjects(p, who: str) -> Tuple[List[str], bool]:
    figs = [el["subject"] for el in DP.figures(p)]
    w = (who or "all").lower()
    if w in ("all", "everything", ""):
        return figs, True
    if w in ("scene", "blocks", "room", "background"):
        return [], True
    if w not in figs:
        from .draw_render import figure_of
        figure_of(p, w, USAGE["draw"])
    return [w], False


def drawn(p) -> dict:
    return dc.dstate(p).get("drawn2d") or {}


def is_drawn_region(p, name: str) -> bool:
    info = p.state.get("regions", {}).get(name, {})
    return str(info.get("how", "")).startswith("2D over the 3D guide")


def stale_subjects(p) -> List[str]:
    return [s for s, v in drawn(p).items() if v.get("key") != scene_key(p, s)]


def do_draw(ctx, who: str = "all", groups: Optional[set] = None, label: Optional[str] = None):
    import time
    from . import cli_draw, draw_line, workflow_draw, brush
    from .draw_render import _restore, _snapshot, keeps_clean  # noqa: F401
    from .ops import ops_to_jsonable
    p = ctx.project
    d = dc.dstate(p)
    if not DP.figures(p) and not DP.blocks_rec(p):
        raise UsageError("nothing posed in 3D to draw over: pose a figure or add a block first",
                         "pixelcanvas pose girl --preset stand")
    subjects, with_blocks = _subjects(p, who)
    t0 = time.time()
    sc = build(p)
    t1 = time.time()
    keep = set(subjects) | ({"@" + b["name"] for b in DP.blocks_rec(p)} if with_blocks else set())
    sts = strokes(p, sc, keep, groups)
    want_face = groups is None or "face" in groups
    face_fills: list = []
    face_lines: List[Tuple[str, dict]] = []
    for el in DP.figures(p):
        # every drawn figure keeps its face fills on the paint layer (the layer is made again each time)
        if el["subject"] in subjects or el["subject"] in drawn(p):
            ff, fl = face_specs(p, sc, el)
            face_fills += ff
            if el["subject"] in subjects and want_face:
                face_lines += [(el["subject"], x) for x in fl]
    table = _table(p, sc)
    t2 = time.time()
    rep = cli_draw.DReport(mutated=True)
    before = dc.composite_u8(p, "view")
    lab_up = cv2.resize(sc["R"].lab, (p.W, p.H), interpolation=cv2.INTER_NEAREST)
    regions: Dict[str, np.ndarray] = {}
    for k, t in enumerate(table):
        if t["subj"] not in keep:
            continue
        regions.setdefault(t["region"], np.zeros((p.H, p.W), bool))
    by_region: Dict[str, List[int]] = {}
    for k, t in enumerate(table):
        if t["region"] in regions:
            by_region.setdefault(t["region"], []).append(k)
    for rn, ks in by_region.items():
        regions[rn] = np.isin(lab_up, ks)
    files = dc.owned_files("line") + ["layers/paint.png", "draw/r3d/paint2d.npz", NPZ]
    files += [f"regions/{n}.png" for n in set(regions) | set(p.state.get("regions", {}))]
    step = p.begin_step(list(dict.fromkeys(files)))
    ids: Dict[str, List[str]] = {}
    try:
        note = dc.owned_fold(p, "line", d["lines"], draw_line.render_ops)

        def gone(r):
            if r.get("d2d") not in keep:
                return False
            if groups is None:
                return True
            return r.get("d2d_group") in groups
        d["lines"] = [r for r in d["lines"] if not gone(r)]
        S = float(max(p.W, p.H))
        eps = 0.0016 * S
        over = 0.0035 * S
        wscale = _line_scales(p)
        for st in sts:
            pt = st["part"]
            pts = st["pts"]
            if not st["closed"] and len(pts) >= 4:
                # a small overlap at each end: the stroke runs on a little into the line it meets
                for end, nb in ((0, 3), (-1, -4)):
                    d_ = S2.unit(pts[end] - pts[nb])
                    pts = np.vstack([pts[:1] + d_ * over, pts]) if end == 0 else np.vstack([pts, pts[-1:] + d_ * over])
            ctrl = S2.simplify(pts, eps, closed=st["closed"])
            if len(ctrl) < 2:
                continue
            if st["closed"] and len(ctrl) < 4:
                continue
            ctrl_u = dc.px2u(p, ctrl)
            path, _ = draw_line.fit(p, ctrl_u, closed=st["closed"], snap=False, join=False, bow=0.0)
            if len(path) < 2 or dc.path_length(path) < 3:
                continue
            rec = draw_line.make_record(p, path, ctrl_u, weight=st["w"] * wscale.get(pt["subj"], 1.0),
                                        taper="none" if st["closed"] else "both",
                                        color="line", closed=st["closed"], src=f"draw {pt['subj']} {pt['name']}")
            rec["d2d"] = pt["subj"]
            rec["d2d_group"] = pt["group"]
            rec["d2d_part"] = pt["name"]
            rec["part"] = region_of(p, pt)
            rec["even"] = True                  # one confident weight along the stroke, tapered at its ends
            d["lines"].append(rec)
            ids.setdefault(f"{pt['subj'].lstrip('@')} {pt['group']}", []).append(rec["id"])
        for subj, fl in face_lines:
            P = fl["px"]
            if len(P) < 2 or dc.path_length(P) < 3:
                continue
            rec = draw_line.make_record(p, P, dc.px2u(p, P[[0, -1]]), weight=fl["weight"] * wscale.get(subj, 1.0),
                                        taper=fl["taper"],
                                        color="line", closed=fl["closed"], src=f"face {fl['name']}")
            rec["d2d"], rec["d2d_group"], rec["d2d_part"] = subj, "face", fl["name"]
            rec["even"] = True
            if fl.get("rgb"):
                rec["auto_rgb"] = [int(v) for v in fl["rgb"]]
                rec["rgb_fixed"] = True
            d["lines"].append(rec)
            ids.setdefault(f"{subj} face", []).append(rec["id"])
        dc.owned_write(p, "line", d["lines"], draw_line.render_ops)
        # the face fills: on the paint layer, made again every time
        arr = p.layer("paint").copy()
        arr = _restore(p, "paint2d", arr)
        a0 = arr.copy()
        if face_fills:
            brush.apply_ops(arr, face_fills)
        _snapshot(p, "paint2d", a0, arr)
        p._write_layer_file("paint", arr)
        # areas: one region per material of each figure, one per block
        from .regions import clean_name
        for rn, m in regions.items():
            rn2 = clean_name(rn)
            if not m.any():
                continue
            p.save_region_mask(rn2, m)
            info = p.state["regions"].setdefault(rn2, {"focus": 1})
            info.update({"box": list(coords.px_box_to_units(coords.mask_bbox_px(m), p.W, p.H)),
                         "area": round(float(m.mean()), 4), "how": "2D over the 3D guide"})
        _register(p, table, regions)
        save_maps(p, sc, table)
        dd = d.setdefault("drawn2d", {})
        for s_ in list(subjects) + (["@scene"] if with_blocks else []):
            ent = dd.setdefault(s_, {})
            ent["key"] = scene_key(p, s_)
            ent["groups"] = sorted(set(ent.get("groups", [])) | (set(groups) if groups else set(GROUPS)))
        ticked = workflow_draw.refresh(p)
        p.end_step(step, {"cmd": ctx.cmdline, "layer": "line,paint", "ops": None, "n_ops": len(sts), "bbox": None})
    except Exception:
        p.abort_step(step)
        raise
    t3 = time.time()
    rep.step = step
    n_lines = sum(len(v) for v in ids.values())
    rep.add(f"step {step}  {label or 'draw ' + (who or 'all')}: {n_lines} line records over the 3D guide "
            f"({t3 - t0:.1f} s: shapes {t1 - t0:.1f}, strokes {t2 - t1:.1f}, layers {t3 - t2:.1f})")
    for key_, v in sorted(ids.items()):
        rep.add(f"  {key_}: {len(v)} strokes {v[0]}..{v[-1]}")
    rep.add(_gap_line(p, sc, subjects))
    rep.add("areas: " + ", ".join(sorted(regions))[:400])
    rep.add("change one stroke:  pixelcanvas line redo <id> \"x,y x,y ...\" | line weight <id> 1.2 | line delete <id>;  "
            "redraw one part:  pixelcanvas draw <who> --part hair")
    if note:
        rep.add("note: " + note)
    if ticked:
        rep.add(f"todo ticked: {', '.join(map(str, ticked))}")
    if not ctx.batch:
        cli_draw.show_before_after(ctx, rep, before, dc.composite_u8(p, "view"), (0, 0, p.W, p.H), step,
                                   "the 2D drawing over the 3D guide")
    cli_draw._tail(ctx, rep)
    rep.summary = f"draw {n_lines} strokes"
    return rep


def _line_scales(p) -> Dict[str, float]:
    """subject -> line weight factor from the figure's size on the canvas: a figure whose head is a tenth of the
    canvas draws at 1; a far, small figure draws thinner lines, a close-up a little heavier."""
    out: Dict[str, float] = {}
    S = float(max(p.W, p.H))
    for el in DP.figures(p):
        try:
            hj = dc.u2px(p, [el["j2d"]["head-top"], el["j2d"]["chin"]])
            r = float(np.linalg.norm(hj[0] - hj[1])) / S
        except Exception:  # noqa: BLE001
            continue
        out[el["subject"]] = float(np.clip(math.sqrt(max(r, 1e-4) / 0.10), 0.55, 1.2))
    return out


def _gap_line(p, sc: dict, subjects: List[str]) -> str:
    """How far the 2D figure departs from the 3D guide's silhouette (the anime shape is slimmer)."""
    gm = p.root / "draw" / "guide.npz"
    if not gm.exists() or not subjects:
        return "the 3D guide: pixelcanvas render draws it (faded under the drawing in the views)"
    try:
        z = np.load(gm, allow_pickle=False)
        fig = z["fig"]
        R = sc["R"]
        fig_l = cv2.resize(fig.astype(np.uint8), (R.W, R.H), interpolation=cv2.INTER_NEAREST) > 0
        mine = np.isin(R.lab, [k for k, pt in enumerate(sc["parts"]) if pt["subj"] in subjects])
        a3, a2 = float(fig_l.sum()), float(mine.sum())
        if a3 <= 0:
            return ""
        inter = float((fig_l & mine).sum())
        return (f"against the 3D guide: the 2D figures cover {a2 / a3 * 100:.0f}% of its area ({inter / max(a2, 1) * 100:.0f}% "
                f"of the drawing lies inside it): slimmer limbs and trunk, smaller hands and feet, locks for hair")
    except Exception:  # noqa: BLE001
        return ""


def _register(p, table: List[dict], regions: Dict[str, np.ndarray]) -> None:
    """Each drawn material is a plan part with its colour as a swatch (blocks: a swatch)."""
    d = dc.dstate(p)
    from .regions import clean_name
    seen = set()
    for t in table:
        rn = clean_name(t["region"])
        if rn in seen or rn not in p.state.get("regions", {}) or t["region"] not in regions:
            continue
        seen.add(rn)
        d["palette"]["swatches"][rn] = dc.to_hex(tuple(t["rgb"]))
        if t.get("block") or t["subj"].startswith("@"):
            continue
        m = regions[t["region"]]
        if rn not in d["parts"]:
            box = list(coords.px_box_to_units(coords.mask_bbox_px(m), p.W, p.H))
            focus = 2 if t["mat"] in ("hair",) + tuple(F3.GARMENTS) else 1
            d["parts"][rn] = {"of": t["subj"], "box": box, "focus": focus, "swatch": rn, "d2d": True}
        if d["parts"][rn].get("d2d") or not d["parts"][rn].get("swatch"):
            d["parts"][rn]["swatch"] = rn


def cmd_draw(ctx, args):
    from .cli import Spec, parse
    p = ctx.project
    a = parse(args, Spec(USAGE["draw"], ["who"], {"part": "value"}, alias={"parts": "part", "only": "part"},
                         min_pos=0))
    who = str(a.get("who") or "all").lower()
    groups = None
    if a.get("part"):
        groups = set()
        for w in str(a["part"]).replace(",", " ").split():
            w = w.lower()
            w = {"arms": "arm", "legs": "leg", "hands": "hand", "feet": "foot", "clothing": "clothes",
                 "body": "torso"}.get(w, w)
            if w in ("arm", "leg", "hand", "foot"):
                groups |= {f"{w}-r", f"{w}-l"}
            elif w in GROUPS:
                groups.add(w)
            else:
                raise UsageError(f"--part is one of {', '.join(GROUPS)}", USAGE["draw"])
        if who in ("all", "scene"):
            raise UsageError("--part redraws one figure's part: name the figure", "pixelcanvas draw girl --part hair")
    return do_draw(ctx, who, groups, label=f"draw {who}" + (f" --part {a['part']}" if a.get("part") else ""))


# ============================================================================================ flat: areas from the drawing

def drawn_corrections(p, labels: np.ndarray, table: List[dict]) -> Dict[int, Tuple[str, str]]:
    """area -> (part, swatch) from the 2D drawing's parts: the region most of the area lies in."""
    mp = load_maps(p)
    if mp is None or not drawn(p):
        return {}
    lab = mp["lab"]
    tb = mp["table"]
    Hl, Wl = lab.shape
    small = cv2.resize(labels.astype(np.int32), (Wl, Hl), interpolation=cv2.INTER_NEAREST)
    regs = sorted({t["region"] for t in tb})
    ridx = {r: i for i, r in enumerate(regs)}
    part_reg = np.array([ridx[t["region"]] for t in tb] + [len(regs)], np.int64)
    rl = part_reg[np.where(lab >= 0, lab, len(tb))]
    nA = int(labels.max()) + 1
    K = len(regs) + 1
    hist = np.bincount((small.astype(np.int64) * K + rl).ravel(), minlength=nA * K).reshape(nA, K)
    from .regions import clean_name
    out = {}
    for r in table:
        n = r["n"]
        if n >= nA:
            continue
        hrow = hist[n]
        tot = hrow.sum()
        if tot == 0:
            continue
        j = int(np.argmax(hrow[:-1])) if len(regs) else -1
        if j < 0 or hrow[j] < 0.5 * tot:
            continue
        rn = clean_name(regs[j])
        out[n] = (rn, rn)
    return out


# ============================================================================================ cel shadows

def do_cel_shade(ctx, who: str = "all"):
    from . import brush, cli_draw, workflow_draw
    from .draw_render import _block_shadow, _restore, _snapshot
    from .ops import Fill, mask_to_rle
    p = ctx.project
    d = dc.dstate(p)
    mp = load_maps(p)
    if mp is None:
        raise UsageError("nothing drawn over the 3D guide yet", "pixelcanvas draw all")
    subjects, with_blocks = _subjects(p, who)
    lab, z, nrm, s = mp["lab"], mp["z"], mp["nrm"], mp["s"]
    tb = mp["table"]
    Hl, Wl = lab.shape
    lv = dc.light_vec(d)
    lam = (nrm * lv[None, None, :].astype(np.float32)).sum(-1)
    S = float(max(p.W, p.H))
    ops = []
    figs = {el["subject"]: el for el in DP.figures(p)}
    keep = set(subjects) | ({"@" + b["name"] for b in DP.blocks_rec(p)} if with_blocks else set())
    # screen-space cast shadows: a pixel is shaded when, toward the light, a nearer part of the same figure is
    # in front (under bangs, a chin, an arm, a skirt)
    subj_of = np.array([t["subj"] for t in tb] + [""], object)
    lab_s = np.where(lab >= 0, lab, len(tb))
    L2 = np.array([lv[0], lv[1]], float)
    L2 = L2 / max(np.linalg.norm(L2), 1e-6)
    cast = np.zeros_like(lab, bool)          # long reach: an arm on the body, a skirt on the legs
    near = np.zeros_like(lab, bool)          # short reach: the bangs on the forehead, the chin on the neck
    for subj in subjects:
        el = figs.get(subj)
        if el is None:
            continue
        body, J, F = DP._fig_state(el)
        hj = dc.u2px(p, [el["j2d"]["head-top"], el["j2d"]["chin"]])
        hpx = float(np.linalg.norm(hj[0] - hj[1])) * s
        mine = subj_of[lab_s] == subj
        zm = np.where(mine, z, 1e4).astype(np.float32)
        for dist, tgt in ((0.05, near), (0.1, cast), (0.18, cast)):
            dx, dy = L2 * hpx * dist
            M = np.float32([[1, 0, -dx], [0, 1, -dy]])
            zs = cv2.warpAffine(zm, M, (Wl, Hl), flags=cv2.INTER_NEAREST, borderValue=1e4)
            tgt |= mine & (zs < z - body.h * 0.1)
    # hair is shaded as one mass, not lock by lock: its normals smoothed over the whole hair of each figure
    lam_hair = lam.copy()
    for subj in subjects:
        el = figs.get(subj)
        if el is None:
            continue
        hm = np.isin(lab, [k for k, t in enumerate(tb) if t["subj"] == subj and t["mat"] == "hair"])
        if not hm.any():
            continue
        hj = dc.u2px(p, [el["j2d"]["head-top"], el["j2d"]["chin"]])
        sig = max(1.0, float(np.linalg.norm(hj[0] - hj[1])) * s * 0.12)
        w = cv2.GaussianBlur(hm.astype(np.float32), (0, 0), sig)
        ns = np.stack([cv2.GaussianBlur((nrm[..., c] * hm).astype(np.float32), (0, 0), sig) for c in range(3)], -1)
        ns = ns / np.maximum(w[..., None], 1e-4)
        ns /= np.maximum(np.linalg.norm(ns, axis=-1, keepdims=True), 1e-4)
        lh = (ns * lv[None, None, :].astype(np.float32)).sum(-1)
        lam_hair[hm] = lh[hm]
    # one light, simple shapes: the lighting is smoothed inside each part before it is cut, so the terminator
    # runs as one calm curve along a limb or the torso (no wobble from the tube's facets, no patches)
    # the cut uses a raking version of the plan light (an illustrator's light comes from the side, so every form
    # gets a readable shadow side; a light from the front alone leaves the limbs flat)
    lv_c = np.array([lv[0], lv[1], min(float(lv[2]), 0.3)], np.float64)
    lv_c /= max(1e-6, float(np.linalg.norm(lv_c)))
    lam_c = (nrm * lv_c[None, None, :].astype(np.float32)).sum(-1)
    lam_s = _part_smoothed(lam_c, lab, [k for k, t in enumerate(tb) if t["subj"] in keep and t["drawn"]
                                      and not t.get("block") and t["mat"] != "hair"], max(1.0, 0.012 * max(Hl, Wl)))
    far_parts = _far_limbs(lab, z, tb, subjects)
    shade_parts: Dict[str, Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]] = {}
    for k, t in enumerate(tb):
        if t["subj"] not in keep or not t["drawn"]:
            continue
        rn = t["region"]
        m = lab == k
        if not m.any():
            continue
        if k in far_parts and t["mat"] != "hair" and t["group"] != "head":
            if rn not in shade_parts:
                shade_parts[rn] = (np.zeros_like(m), np.zeros_like(m), tuple(t["rgb"]))
            shade_parts[rn][0][...] |= m            # the far leg: in shadow as one shape
            continue
        if rn not in shade_parts:
            shade_parts[rn] = (np.zeros_like(m), np.zeros_like(m), tuple(t["rgb"]))
        sm, dm, _ = shade_parts[rn]
        if t.get("block"):
            if float(np.median(lam[m])) < 0.12:
                sm |= m
            continue
        thr = 0.05                               # one terminator for skin and cloth alike: one consistent light
        if t["mat"] == "hair":
            sm |= m & ((lam_hair < thr) | near)     # the hair: one mass, short cast shadows (bangs on the cap)
            dm |= m & (lam_hair < thr - 0.45)
        elif str(t["mat"]).startswith("prop:"):
            # a held thing (an umbrella, a sword): one tone per part, its ribs and edges do not band it
            if float(np.median(lam_s[m])) < thr:
                sm |= m
        elif t["group"].startswith("hand"):
            sm |= m & ((lam_s < -0.25) | cast | near)   # a hand's normals are flat: no half-shaded fist
        elif t["group"] == "head":
            sm |= m & ((lam < -0.35) | near)        # the face stays lit: its far side, the shadow under bangs and chin
        else:
            sm |= m & ((lam_s < thr) | cast | near)
    files = ["layers/shade.png", "draw/r3d/shade2d.npz"]
    step = p.begin_step(files)
    try:
        r_px = 0.005 * S
        min_a = (0.016 * S) ** 2
        for rn, (sm, dm, rgb) in shade_parts.items():
            for mm, depth in ((sm, 1.0), (dm, 1.6)):
                if not mm.any():
                    continue
                big = cv2.resize(mm.astype(np.uint8), (p.W, p.H), interpolation=cv2.INTER_LINEAR) > 0
                from .regions import clean_name
                rc = clean_name(rn)
                reg = p.region_mask(rc) if rc in p.state.get("regions", {}) else big
                big = S2.clean_shape(big, r_px, min_a) & reg
                if big.sum() < 6:
                    continue
                col = _block_shadow(rgb) if rn in {t["region"] for t in tb if t.get("block")} else \
                    dc.shadow_of(rgb, depth)
                ops.append(Fill(mask_rle=mask_to_rle(big), color=tuple(int(c) for c in col), feather=0.6))
        ops += _hair_bands(p, mp, subjects, lv)
        ops += _figure_shadows(p, mp, subjects)
        arr = p.layer("shade").copy()
        arr = _restore(p, "shade2d", arr)
        a0 = arr.copy()
        if ops:
            brush.apply_ops(arr, ops)
        _snapshot(p, "shade2d", a0, arr)
        p._write_layer_file("shade", arr)
        sd = d.setdefault("shaded2d", {})
        for s_ in list(subjects) + (["@scene"] if with_blocks else []):
            sd[s_] = scene_key(p, s_)
        ticked = workflow_draw.refresh(p)
        p.end_step(step, {"cmd": ctx.cmdline, "layer": "shade", "ops": None, "n_ops": len(ops), "bbox": None})
    except Exception:
        p.abort_step(step)
        raise
    rep = cli_draw.DReport(mutated=True)
    rep.step = step
    rep.add(f"step {step}  shade {who}: {len(ops)} cel shapes (hard edges, two tones, a third on hair; a highlight band "
            f"on the hair; shadows cast by the hair, chin, arms and skirt; the light from "
            f"{dc.light_words(d['light'].get('deg', 135))})")
    if ticked:
        rep.add(f"todo ticked: {', '.join(map(str, ticked))}")
    cli_draw._tail(ctx, rep)
    rep.summary = f"shade {len(ops)} shapes"
    return rep


def _part_smoothed(val: np.ndarray, lab: np.ndarray, ks: List[int], sig: float) -> np.ndarray:
    """``val`` blurred inside each part ``k`` of ``ks`` on its own (no value crosses a part's contour)."""
    out = val.copy()
    pad = int(3 * sig) + 2
    H, W = lab.shape
    for k in ks:
        m = lab == k
        if m.sum() < 4:
            continue
        ys, xs = np.nonzero(m)
        y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + 1 + pad)
        x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + 1 + pad)
        mm = m[y0:y1, x0:x1].astype(np.float32)
        num = cv2.GaussianBlur(val[y0:y1, x0:x1].astype(np.float32) * mm, (0, 0), sig)
        den = cv2.GaussianBlur(mm, (0, 0), sig)
        sub = out[y0:y1, x0:x1]
        sel = mm > 0
        sub[sel] = (num / np.maximum(den, 1e-4))[sel]
    return out


def _far_limbs(lab: np.ndarray, z: np.ndarray, tb: List[dict], subjects: List[str]) -> set:
    """Part ids of each figure's far leg (and foot): the leg clearly behind the other one is shaded as a whole, as
    an illustrator separates the two legs."""
    out: set = set()
    for subj in subjects:
        zs = {}
        ids = {}
        for side in "rl":
            ks = [k for k, t in enumerate(tb) if t["subj"] == subj and t["group"] in (f"leg-{side}", f"foot-{side}")
                  and not t.get("block")]
            m = np.isin(lab, ks)
            if m.sum() < 20:
                break
            zs[side] = float(np.median(z[m]))
            ids[side] = ks
        if len(zs) < 2:
            continue
        hm = np.isin(lab, [k for k, t in enumerate(tb) if t["subj"] == subj and t["group"] == "head"])
        if not hm.any():
            continue
        ys, xs = np.nonzero(hm)
        head = float(max(np.ptp(ys), np.ptp(xs), 1))
        far, near_ = ("r", "l") if zs["r"] > zs["l"] else ("l", "r")
        # depth in scene units; a clear step only (legs side by side stay lit alike)
        if zs[far] - zs[near_] > 0.12:
            out |= set(ids[far])
    return out


def _hair_bands(p, mp: dict, subjects: List[str], lv) -> list:
    """A highlight on the crown on the lit side (the anime 'angel ring'): a few short pieces with pointed lower
    edges, on the hair cap only (never across the bangs, where a band reads as a headband); spiky hair gets none."""
    from .draw_render import look_of
    from .ops import Fill, mask_to_rle
    out = []
    cam = DP.camera_of(p)
    lab, tb, s = mp["lab"], mp["table"], mp["s"]
    for el in DP.figures(p):
        hair = look_of(el).get("hair") or {}
        if el["subject"] not in subjects or not hair or str(hair.get("style") or "") == "spiky":
            continue
        body, J, F = DP._fig_state(el)
        rig = F3.Rig(body, J, F)
        h = rig.h
        Hd = rig.Hd
        c = rig.cr_c
        y0 = 0.3 * h
        R0 = 0.45 * h
        # an arc on the lit side of the crown, its lower edge in points
        side = 1.0 if lv[0] > 0 else -1.0
        th = np.linspace(-0.5, 0.5, 40) * math.pi + side * 0.35
        teeth = np.abs(np.sin(th * 9.0))
        gaps = np.abs(np.sin(th * 3.0 + 0.4)) < 0.28           # breaks: pieces of shine, not a ring
        up = [c + Hd @ np.array([R0 * math.sin(t), y0, R0 * math.cos(t)]) for t in th]
        dn = [c + Hd @ np.array([R0 * 0.99 * math.sin(t), y0 - (0.05 + 0.06 * tt) * h, R0 * 0.99 * math.cos(t)])
              for t, tt in zip(th, teeth)]
        P3 = np.array(up + dn[::-1])
        nrm = (P3 - c)
        vis = (nrm * (cam.C - P3)).sum(1) > 0
        if vis.mean() < 0.3:
            continue
        q, zz = cam.project(P3)
        hc, _ = cam.project(c[None])
        if float(np.median(q[:, 1])) > float(hc[0, 1]):
            continue                    # seen from below, the crown is out of sight: no band on the forehead
        band = dc.poly_mask(p.H, p.W, q)
        hair_ids = [k for k, t in enumerate(tb) if t["subj"] == el["subject"] and t["mat"] == "hair"]
        hm = cv2.resize(np.isin(lab, hair_ids).astype(np.uint8), (p.W, p.H), interpolation=cv2.INTER_NEAREST) > 0
        # only where the band faces the camera: the head's front half in the picture
        band &= hm
        cap_ids = [k for k in hair_ids if tb[k]["name"] == "hair-cap"]
        if cap_ids:
            band &= cv2.resize(np.isin(lab, cap_ids).astype(np.uint8), (p.W, p.H),
                               interpolation=cv2.INTER_NEAREST) > 0
        bu = band.astype(np.uint8)
        for i in np.nonzero(gaps)[0]:
            qa, qb = q[i], q[len(th) * 2 - 1 - i]
            cv2.line(bu, tuple(int(v) for v in np.round(qa)), tuple(int(v) for v in np.round(qb)), 0,
                     max(2, int(0.006 * max(p.W, p.H))))
        band = bu > 0
        if band.sum() < 10:
            continue
        hairc = tuple(next(t["rgb"] for t in tb if t["subj"] == el["subject"] and t["mat"] == "hair"))
        out.append(Fill(mask_rle=mask_to_rle(band), color=dc.light_of(hairc, 1.2), feather=0.6))
    return out


def _ground_shadow_colour(rgb) -> Tuple[int, int, int]:
    """A contact shadow on the ground: the ground's own hue, clearly darker and a little cooler (never grey)."""
    L, C, h = dc.rgb_to_lch(tuple(int(v) for v in rgb))
    from .draw_scene import _toward
    return dc.lch_to_rgb(max(6.0, L - 16.0 - 0.08 * L), C * 1.1 + 4.0, _toward(h, 285.0, 0.25) if C > 4 else 285.0)


def _figure_shadows(p, mp: dict, subjects: List[str]) -> list:
    """One compact contact shadow on the ground under each figure: a dark core ellipse round the feet with a soft
    edge, about as wide as the shoulders, a little off toward the side away from the light. A figure off the
    ground (a jump) gets a detached shadow on the ground below it, smaller and lighter the higher it is. With
    blocks the shadow lies on them; with no blocks it lies on the ground behind the figure (never over a figure)."""
    from .draw_render import _block_shadow
    from .ops import Fill, mask_to_rle
    from .render3d import light_world
    lab, tb = mp["lab"], mp["table"]
    block_ids = [k for k, t in enumerate(tb) if t.get("block")]
    from .scene3d import floor_mask
    sfl = floor_mask(p)                     # the floor of a scene kit (pixelcanvas scene) takes shadows too
    cam = DP.camera_of(p)
    L = light_world(cam, dc.light_vec(dc.dstate(p)))
    off = np.array([-L[0], 0.0, -L[2]])
    off = off / max(1e-6, float(np.linalg.norm(off))) if np.linalg.norm(off) > 1e-3 else np.zeros(3)
    up = cv2.resize(lab, (p.W, p.H), interpolation=cv2.INTER_NEAREST)
    figure_px = (up >= 0) & ~np.isin(up, block_ids)
    blocks = np.isin(up, block_ids) if block_ids else None
    if sfl is not None:
        # a scene kit's floor is the ground (with any blocks on it)
        blocks = (sfl[0] if blocks is None else (blocks | sfl[0])) & ~figure_px
    try:
        flat = p.layer("flat", create=False)
    except Exception:  # noqa: BLE001
        flat = None
    out = []
    for el in DP.figures(p):
        if el["subject"] not in subjects:
            continue
        J = {k: np.asarray(v, float) for k, v in el["j3d"].items()}
        feet = [J[f"heel-{s_}"] for s_ in "rl"] + [J[f"foot-{s_}"] for s_ in "rl"]
        lift = max(0.0, min(f[1] for f in feet))
        if block_ids and lift > 0.15:
            continue                    # sitting or standing on a block: no shadow on the floor far below
        if lift > 4.0:
            continue
        fc = np.mean([[f[0], 0.0, f[2]] for f in feet], 0)
        shoulders = float(np.linalg.norm((J["shoulder-r"] - J["shoulder-l"])[[0, 2]]))
        shoulders = max(shoulders, 0.12 * float(J["head-top"][1] - min(f[1] for f in feet)))
        sc_ = 1.0 / (1.0 + 2.0 * lift)
        spread = float(np.ptp([f[0] for f in feet])) * (0.4 if lift > 0.05 else 1.0)
        rx = max(0.6 * shoulders, 0.5 * spread + 0.08) * sc_
        rz = (0.26 * shoulders if lift > 0.05 else max(0.26 * shoulders, 0.5 * float(np.ptp([f[2] for f in feet]))
                                                              + 0.05)) * sc_
        if lift > 0.05:
            rx = 0.6 * shoulders * sc_
        # compact: a stride or a kneel never spreads it past the shoulders' width
        rx = min(rx, 1.0 * shoulders * sc_)
        rz = min(rz, 0.45 * shoulders * sc_)
        if lift > 0.05:
            fc = np.array([J["pelvis"][0] if "pelvis" in J else fc[0], 0.0, J["pelvis"][2] if "pelvis" in J
                           else fc[2]]) * 0.5 + fc * 0.5
        c = fc + off * 0.12 * rx
        ang = np.linspace(0, 2 * math.pi, 48, endpoint=False)
        ring = np.stack([c[0] + rx * np.cos(ang), np.zeros_like(ang), c[2] + rz * np.sin(ang)], 1)
        q, z = cam.project(ring)
        if (z <= 0.1).any():
            continue
        m = np.zeros((p.H, p.W), np.uint8)
        cv2.fillPoly(m, [np.round(q).astype(np.int32)], 1)
        ell = m > 0
        ground = blocks if blocks is not None else ~figure_px
        sh = ell & ground
        if sh.sum() < 10:
            continue
        cx, cy = np.round(q.mean(0)).astype(int)
        if ground is blocks:
            k0 = up[int(np.clip(cy, 0, p.H - 1)), int(np.clip(cx, 0, p.W - 1))]
            if k0 >= 0 and tb[k0].get("block"):
                base = tuple(tb[k0]["rgb"])
            elif sfl is not None:
                base = tuple(int(v) for v in sfl[1](cx, cy))
            else:
                base = tuple(tb[block_ids[0]]["rgb"])
            col = _ground_shadow_colour(base)
        else:
            if flat is not None and (flat[..., 3][sh] > 0.5).any():
                base = tuple(int(v) for v in np.median(flat[..., :3][sh & (flat[..., 3] > 0.5)], 0) * 255)
            else:
                base = (225, 220, 205)
            col = _ground_shadow_colour(base)
        ry_px = max(2.0, float(np.ptp(q[:, 1])) / 2.0)
        op = float(np.clip(1.0 - 0.45 * lift, 0.45, 1.0))
        # one calm flat shape, a little soft at its rim (the anime contact shadow)
        out.append(Fill(mask_rle=mask_to_rle(sh), color=col, feather=max(1.2, 0.12 * ry_px), opacity=0.6 * op))
    return out


def flat_drawn(ctx, rep) -> None:
    """After ``flat auto``: every part drawn over the guide takes its flat colour on exactly its shape (the areas
    the line art encloses can run into each other where a joint has no line)."""
    from . import brush
    from .draw_render import _restore, _snapshot
    from .ops import Fill, mask_to_rle
    from .regions import clean_name
    p = ctx.project
    mp = load_maps(p)
    if mp is None:
        return
    lab, tb = mp["lab"], mp["table"]
    keep = set(drawn(p))
    keep |= {"@" + b["name"] for b in DP.blocks_rec(p)} if "@scene" in keep else set()
    up = cv2.resize(lab, (p.W, p.H), interpolation=cv2.INTER_NEAREST)
    by_reg: Dict[str, List[int]] = {}
    for k, t in enumerate(tb):
        if t["subj"] in keep and t["drawn"]:
            by_reg.setdefault(t["region"], []).append(k)
    ops = []
    masks = {}
    # blocks first, then the figures (a figure's colours win on its own shapes)
    for rn, ks in sorted(by_reg.items(), key=lambda kv: 0 if tb[kv[1][0]].get("block") else 1):
        m = np.isin(up, ks)
        if m.sum() < 4:
            continue
        rgb = tuple(tb[ks[0]]["rgb"])
        try:
            from .draw_plan import swatch
            rgb = swatch(p, clean_name(rn))
        except Exception:  # noqa: BLE001
            pass
        ops.append(Fill(mask_rle=mask_to_rle(cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0),
                        color=tuple(int(c) for c in rgb), feather=0.8))
        masks[clean_name(rn)] = m
    if not ops:
        return
    files = ["layers/flat.png", "draw/r3d/flat2d.npz"] + [f"regions/{n}.png" for n in set(masks) |
                                                           set(p.state.get("regions", {}))]
    step = p.begin_step(list(dict.fromkeys(files)))
    try:
        arr = p.layer("flat").copy()
        arr = _restore(p, "flat2d", arr)
        a0 = arr.copy()
        brush.apply_ops(arr, ops)
        _snapshot(p, "flat2d", a0, arr)
        p._write_layer_file("flat", arr)
        allm = np.zeros((p.H, p.W), bool)
        for rn, m in masks.items():
            allm |= m
        for rn, m in masks.items():
            p.save_region_mask(rn, m)
            info = p.state["regions"].setdefault(rn, {"focus": 1})
            info.update({"box": list(coords.px_box_to_units(coords.mask_bbox_px(m), p.W, p.H)),
                         "area": round(float(m.mean()), 4), "how": "2D over the 3D guide"})
        # areas outside the drawing keep theirs, minus the drawn shapes
        for rn in list(p.state.get("regions", {})):
            if rn in masks:
                continue
            try:
                om = p.region_mask(rn)
            except Exception:  # noqa: BLE001
                continue
            if (om & allm).any():
                nm = om & ~allm
                p.save_region_mask(rn, nm)
                p.state["regions"][rn]["area"] = round(float(nm.mean()), 4)
        from . import workflow_draw
        workflow_draw.refresh(p)
        p.end_step(step, {"cmd": ctx.cmdline + " (the drawn shapes)", "layer": "flat", "ops": None,
                          "n_ops": len(ops), "bbox": None})
    except Exception:
        p.abort_step(step)
        raise
    rep.add(f"step {step}: {len(ops)} drawn parts took their flat colour on exactly their shapes "
            f"({', '.join(sorted(masks))[:300]})")


def wrap_flat(orig):
    def run(ctx, args):
        rep = orig(ctx, args)
        sub = str(args[0]).lower() if args else ""
        if sub in ("auto", "refresh") and drawn(ctx.project) and getattr(rep, "mutated", True):
            flat_drawn(ctx, rep)
        return rep
    return run


def wrap_shade(orig):
    """``shade <who>|all`` on figures drawn over the 3D guide: cel shapes; anything else: the shade tool."""
    def run(ctx, args):
        p = ctx.project
        first = str(args[0]).lower() if args else ""
        dd = drawn(p)
        figs = [el["subject"] for el in DP.figures(p)]
        if dd and (first in ("all", "scene") or (first in figs and first in dd)):
            return do_cel_shade(ctx, first)
        return orig(ctx, args)
    return run


COMMANDS = {"draw": cmd_draw}
WRAP = {"shade": wrap_shade, "flat": wrap_flat}
