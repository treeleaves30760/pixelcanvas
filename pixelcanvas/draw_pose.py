"""3D layout without a reference: ``pixelcanvas camera``, ``pose``, ``block``, ``prop`` and ``form``.

The camera, the posed figures, the blocks and the props are sketch records, so the sketch layer is still the render
of its records and undo works as for any sketch element:

- the camera is the ``horizon`` record with a ``camera`` field (eye height, fov, tilt): it draws the horizon and a
  light ground grid;
- a posed figure is an ordinary ``figure`` record (subject, pose, heads and the 14 joints in 0-1000 units, l = picture
  left, exactly what ``sketch figure`` writes, so wear / hand / face / hair / checks / todo work on it) plus ``rig``
  (what was asked: preset, joint edits, place, size, facing, targets), ``j3d`` (joints in metres), ``j2d`` (every
  joint projected) and ``frames`` (part orientations);
- a ``block`` record holds a box, stairs, cylinder, plane or railing standing on the ground; a ``prop`` record a
  thing held in a hand.

Figures, blocks and props are drawn together through one z-buffer (a figure behind a wall is hidden); a figure's
lines hidden behind its own parts are dashed. Every change re-solves the scene from the records (a moved block
moves the figure sitting on it; a new camera re-places everything at the same picture points). The part-id, depth and
normal maps are saved as ``draw/scene3d.npz``; ``pixelcanvas form`` saves the light and shadow of the forms as
``draw/form3d.npz`` and the shade stage uses it for areas on a posed figure.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import coords
from . import draw_core as dc
from . import pose3d as P3
from . import render3d as R3
from .coords import UsageError

RENDER_MAX = 1600            # longest side of the z-buffer (contours are scaled up to the canvas)
FORM_MAX = 800
KINDS_3D = ("block", "prop")

USAGE = {
    "camera": "pixelcanvas camera --horizon 700 --eye-height 0.9 --fov 50",
    "pose": "pixelcanvas pose girl --preset run --at 520,900 --yaw 30 [--lean 15 --twist 20 --bend 10 --look camera|left|right|up|down|back|at:x,y --energy 0..1 --mirror --arm-r raised|forward|back|down|bent|out|hip|behind|crossed|face|chin|hair --leg-l forward|back|bent|raised|kick|out|tucked --hand-r fist|open|relaxed|point|grip|peace|cup --hand-l-to hip|chest|x,y|<block> --foot-r-to x,y --on <block> --check]",
    "block": "pixelcanvas block wall box --at 300,820 --size 4,0.3,1.0",
    "prop": "pixelcanvas prop sword --kind sword --hand r",
    "form": "pixelcanvas form girl",
}


# ============================================================================================ records

def camera_record(p) -> Optional[dict]:
    for el in dc.dstate(p)["sketch"]:
        if el["kind"] == "horizon" and el.get("camera"):
            return el
    return None


def camera_of(p, W: Optional[int] = None, H: Optional[int] = None) -> P3.Camera:
    el = camera_record(p)
    W = W or p.W
    H = H or p.H
    if el is None:
        return P3.Camera(W, H)
    c = el["camera"]
    return P3.Camera(W, H, float(el["y"]), float(c.get("eye", 1.5)), float(c.get("fov", 50)),
                     float(c.get("tilt", 0)))


def is_rigged(el: dict) -> bool:
    """A figure posed in 3D whose 2D joints were not moved by hand since."""
    return el.get("kind") == "figure" and bool(el.get("rig")) and el.get("rig_joints") == el.get("joints")


def figures(p) -> List[dict]:
    return [el for el in dc.dstate(p)["sketch"] if is_rigged(el)]


def blocks_rec(p) -> List[dict]:
    return [el for el in dc.dstate(p)["sketch"] if el["kind"] == "block"]


def props_rec(p) -> List[dict]:
    return [el for el in dc.dstate(p)["sketch"] if el["kind"] == "prop"]


def _block(el: dict) -> P3.Block:
    return P3.Block(el["name"], el["type"], el["pos"], el["size"], el.get("rot", 0.0), el.get("steps", 6))


def _fig_state(el: dict):
    body = P3.Body(float(el["rig"].get("height") or 1.6), float(el["rig"].get("heads") or 6.5))
    J = {k: np.asarray(v, float) for k, v in el["j3d"].items()}
    F = {k: np.asarray(v, float).reshape(3, 3) for k, v in el["frames"].items()}
    return body, J, F


def solve_block(el: dict, cam: P3.Camera) -> None:
    at = el.get("at")
    lift = float(el.get("lift", 0.0))
    g = cam.ground_point(at[0], at[1], 0.0) if at else None
    if g is None:
        g = cam.ground_point(500, cam.horizon + 0.55 * (1000 - cam.horizon), 0.0)
        if g is None:
            g = np.array([0.0, 0.0, 6.0])
    off = el.get("offset") or [0.0, 0.0]
    el["pos"] = [round(float(g[0] + off[0]), 4), round(lift, 4), round(float(g[2] + off[1]), 4)]


def solve_scene(p) -> Tuple[Dict[str, dict], List[str]]:
    """Place every block, figure and prop again from its record (the camera may have moved, a block a figure sits
    on may have changed). Returns the solved figures by subject and notes."""
    d = dc.dstate(p)
    cam = camera_of(p)
    notes: List[str] = []
    blocks: Dict[str, P3.Block] = {}
    for el in sorted(blocks_rec(p), key=lambda e: e.get("n", 0)):
        solve_block(el, cam)
        blocks[el["name"]] = _block(el)
    solved: Dict[str, dict] = {}
    for el in sorted([e for e in d["sketch"] if e["kind"] == "figure" and e.get("rig")], key=lambda e: e.get("n", 0)):
        if el.get("rig_joints") != el.get("joints"):
            for k in ("rig", "rig_joints", "j3d", "j2d", "frames", "head_view"):
                el.pop(k, None)
            notes.append(f"{el['id']} ({el.get('subject')}): its joints were moved by hand, so it is a flat "
                         f"mannequin now (pixelcanvas pose {el.get('subject')} poses it in 3D again)")
            continue
        res = P3.solve_figure(el["rig"], cam, blocks, solved)
        write_figure(el, res, cam)
        solved[el["subject"]] = res
        notes.extend(_follow_head(p, el))
    return solved, notes


def _follow_head(p, fig: dict) -> List[str]:
    """Head constructions of a posed subject move and turn with its pose (lines already traced from them stay)."""
    hd, ch = np.array(fig["joints"]["head"], float), np.array(fig["joints"]["chin"], float)
    hpx, cpx = dc.u2px(p, [hd])[0], dc.u2px(p, [ch])[0]
    r = float(np.hypot(*(cpx - hpx)))
    box = [round((hpx[0] - 0.8 * r) * 1000 / p.W, 1), round((hpx[1] - r) * 1000 / p.H, 1),
           round((hpx[0] + 0.8 * r) * 1000 / p.W, 1), round(cpx[1] * 1000 / p.H, 1)]
    yaw = float(np.clip(fig["head_view"][0], -80, 80))
    pitch = float(np.clip(fig["head_view"][1], -60, 60))
    out = []
    for el in dc.dstate(p)["sketch"]:
        if el["kind"] != "head" or el.get("subject") != fig["subject"]:
            continue
        if max(abs(a - b) for a, b in zip(el["box"], box)) > 0.5 or abs(el.get("yaw", 0) - yaw) > 0.5:
            el["box"], el["yaw"], el["pitch"] = box, round(yaw, 1), round(pitch, 1)
            out.append(f"head construction {el['id']} moved with the pose (yaw {yaw:.0f}); lines traced from it "
                       f"before stay where they were: redo them with pixelcanvas line trace {el['id']}")
    return out


def write_figure(el: dict, res: dict, cam: P3.Camera) -> None:
    J, F = res["J"], res["F"]
    el["j3d"] = {k: [round(float(x), 4) for x in v] for k, v in J.items()}
    el["frames"] = {k: [round(float(x), 5) for x in np.asarray(v).reshape(-1)] for k, v in F.items()}
    U = cam.project_units(np.array(list(J.values())))
    el["j2d"] = {k: [round(float(u[0]), 1), round(float(u[1]), 1)] for k, u in zip(J, U)}
    joints, left_is_r = P3.compat_joints(cam, J)
    el["joints"] = joints
    el["rig_joints"] = json.loads(json.dumps(joints))
    el["picture_left"] = "r" if left_is_r else "l"
    hy, hp = P3.view_angles(cam, F["head"], J["head"])
    el["head_view"] = [round(hy, 1), round(hp, 1)]
    # the face frame on the 3D head, projected: where the eyes, brows, nose, mouth and chin are in the picture (the
    # face and eye tools place features on it; -r / -l = the figure's own right / left)
    from .figure3d import Rig, face_frame
    ff = face_frame(Rig(res["body"], J, F))
    U = cam.project_units(np.array(list(ff.values())))
    el["face2d"] = {k: [round(float(u[0]), 1), round(float(u[1]), 1)] for k, u in zip(ff, U)}
    el["pose"] = res["preset"]
    el["heads"] = float(res["body"].heads)


def scene_prims(p) -> List[dict]:
    d = dc.dstate(p)
    heads = {e.get("subject") for e in d["sketch"] if e["kind"] == "head"}
    out: List[dict] = []
    figs = {el["subject"]: el for el in figures(p)}
    for el in figs.values():
        body, J, F = _fig_state(el)
        for pr in P3.body_prims(body, J, F):
            pr["owner"] = el["id"]
            pr["hidden_lines"] = True
            pr["noline"] = pr["part"] == "head" and el["subject"] in heads
            out.append(pr)
    for el in blocks_rec(p):
        if el.get("pos") is not None:
            out += _block(el).prims(el["id"])
    for el in props_rec(p):
        fig = figs.get(el.get("subject"))
        if fig is None:
            continue
        body, J, F = _fig_state(fig)
        for pr in P3.prop_prims(el["type"], el["hand"], el.get("length"), body, J, F, el["id"]):
            pr["hidden_lines"] = True
            out.append(pr)
    return out


# ============================================================================================ the rendered scene

_CACHE: Dict[str, dict] = {}


def _key(p) -> str:
    d = dc.dstate(p)
    recs = []
    for el in d["sketch"]:
        if el["kind"] in KINDS_3D or is_rigged(el) or (el["kind"] == "horizon" and el.get("camera")):
            recs.append({k: v for k, v in el.items() if k != "n"})
        elif el["kind"] == "head":
            recs.append({"head": el.get("subject")})
    return hashlib.sha1(json.dumps([p.W, p.H, recs], sort_keys=True, default=str).encode()).hexdigest()


def render_scale(p, cap: int = RENDER_MAX) -> float:
    return min(1.0, cap / float(max(p.W, p.H)))


def get_scene(p) -> dict:
    """The z-buffer of every figure, block and prop, and their construction lines (cached per state)."""
    key = _key(p)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    s = render_scale(p)
    Wr, Hr = max(1, int(round(p.W * s))), max(1, int(round(p.H * s)))
    s = Wr / float(p.W)
    cam = camera_of(p, Wr, Hr)
    prims = scene_prims(p)
    R = R3.render(cam, prims)
    skip = {(pr["owner"], pr["part"]) for pr in prims if pr.get("noline")}
    lines = R3.contour_lines(R, prims, s, skip, dash_px=0.012 * max(p.W, p.H))
    sc = {"key": key, "R": R, "prims": prims, "scale": s, "lines": lines, "cam": cam}
    if len(_CACHE) > 3:
        _CACHE.clear()
    _CACHE[key] = sc
    return sc


def save_maps(p) -> Optional[str]:
    """draw/scene3d.npz: part id, face, depth (m), normal (screen x right, y down, z toward the viewer) per pixel of
    the z-buffer, and the parts table (JSON)."""
    if not (figures(p) or blocks_rec(p)):
        return None
    sc = get_scene(p)
    R = sc["R"]
    parts = [{"owner": pr.get("owner"), "part": pr["part"], "group": pr.get("group")} for pr in sc["prims"]]
    owners = {el["id"]: (el.get("subject") or el.get("name")) for el in dc.dstate(p)["sketch"]}
    path = p.root / "draw" / "scene3d.npz"
    path.parent.mkdir(exist_ok=True)
    dep = np.where(np.isfinite(R["depth"]), R["depth"], 0).astype(np.float32)
    np.savez_compressed(path, id=R["id"].astype(np.int16), face=R["face"], depth=dep,
                        normal=R["normal"].astype(np.float16), scale=np.float32(sc["scale"]),
                        parts=json.dumps(parts), owners=json.dumps(owners), key=sc["key"])
    return p.rel(path)


# ============================================================================================ sketch shapes

def element_shapes(p, el) -> Optional[List[Tuple[np.ndarray, bool, float]]]:
    """Polylines (canvas px, closed, weight) of a 3D record; None = draw it the 2D way."""
    k = el["kind"]
    if k == "horizon":
        return camera_shapes(p, el)
    if k == "figure" and not is_rigged(el):
        return None
    return list(get_scene(p)["lines"].get(el["id"], []))


def camera_shapes(p, el) -> List[Tuple[np.ndarray, bool, float]]:
    cam = camera_of(p)
    y = cam.horizon_px()
    out = [(np.array([[0, y], [p.W - 0.01, y]]), False, 0.9)]
    if y >= p.H - 2:
        return out
    zs = [1, 2, 3, 4, 5, 6, 8, 10, 13, 17, 22, 30]
    sc = get_scene(p) if (figures(p) or blocks_rec(p)) else None

    def clip(P):
        P, z = cam.project(P)
        ok = (z > 0.2) & (P[:, 0] > -0.05 * p.W) & (P[:, 0] < 1.05 * p.W) & (P[:, 1] > y) & (P[:, 1] < 1.02 * p.H)
        if sc is not None:              # the grid lies on the ground: blocks and figures stand in front of it
            s = sc["scale"]
            D = sc["R"]["depth"]
            qx = np.clip((P[:, 0] * s).astype(int), 0, D.shape[1] - 1)
            qy = np.clip((P[:, 1] * s).astype(int), 0, D.shape[0] - 1)
            ok &= z <= D[qy, qx] + 0.05
        runs, cur = [], []
        for q, k in zip(P, ok):
            if k:
                cur.append(q)
            elif cur:
                runs.append(np.array(cur))
                cur = []
        if cur:
            runs.append(np.array(cur))
        return [r for r in runs if len(r) >= 2]
    for z in zs:
        for Q in clip(np.array([[x, 0.0, z] for x in np.linspace(-30, 30, 241)])):
            out.append((Q, False, 0.3))
    for x in range(-12, 13, 2):
        for Q in clip(np.array([[x, 0.0, z] for z in np.geomspace(0.3, 40, 160)])):
            out.append((Q, False, 0.3))
    return out


def label_points(p, el) -> List[Tuple[str, Tuple[float, float]]]:
    if el["kind"] == "block":
        sc = get_scene(p)
        b = owner_box(p, sc, el["id"])
        if b is None:
            return []
        return [(el["name"], ((b[0] + b[2]) / 2, b[1] + 0.02 * p.H))]
    if el["kind"] == "prop":
        sc = get_scene(p)
        b = owner_box(p, sc, el["id"])
        return [] if b is None else [(el["name"], (b[2], b[1]))]
    return []


def owner_box(p, sc, owner: str, visible: bool = True) -> Optional[Tuple[float, float, float, float]]:
    R, prims, s = sc["R"], sc["prims"], sc["scale"]
    ks = [k for k, pr in enumerate(prims) if pr.get("owner") == owner]
    if not ks:
        return None
    if visible:
        m = np.isin(R["id"], ks)
        if not m.any():
            return None
        ys, xs = np.nonzero(m)
        return (xs.min() / s, ys.min() / s, (xs.max() + 1) / s, (ys.max() + 1) / s)
    bs = [R["own"][k][0] for k in ks if R["own"][k] is not None]
    if not bs:
        return None
    a = np.array(bs, float) / s
    return (a[:, 0].min(), a[:, 1].min(), a[:, 2].max(), a[:, 3].max())


def visibility(sc, owner: str) -> Dict[str, float]:
    """Visible share of each limb group of a figure (arm-r, leg-l, head, torso)."""
    R, prims = sc["R"], sc["prims"]
    tot: Dict[str, int] = {}
    vis: Dict[str, int] = {}
    for k, pr in enumerate(prims):
        if pr.get("owner") != owner or R["own"][k] is None:
            continue
        (x0, y0, x1, y1), hm = R["own"][k]
        g = pr.get("group", "")
        tot[g] = tot.get(g, 0) + int(hm.sum())
        vis[g] = vis.get(g, 0) + int((R["id"][y0:y1, x0:x1] == k).sum())
    return {g: vis.get(g, 0) / max(1, n) for g, n in tot.items()}


# ============================================================================================ form shading

_FORM: Dict[str, dict] = {}


def form_data(p) -> Optional[dict]:
    """Lit / core shadow / cast shadow per pixel (canvas size) under the plan light, or None without 3D figures."""
    if not figures(p):
        return None
    lv = dc.light_vec(dc.dstate(p))
    key = _key(p) + json.dumps([round(float(v), 4) for v in lv])
    if key in _FORM:
        return _FORM[key]
    s = render_scale(p, FORM_MAX)
    Wr, Hr = max(1, int(round(p.W * s))), max(1, int(round(p.H * s)))
    cam = camera_of(p, Wr, Hr)
    prims = scene_prims(p)
    M = R3.form_maps(cam, prims, lv)
    import cv2
    body = np.zeros(M["id"].shape, bool)
    owner_of = np.array([pr.get("owner", "") for pr in prims] + [""], object)
    fig_ids = {el["id"] for el in figures(p)}
    idm = M["id"]
    body[idm >= 0] = np.isin(owner_of[idm[idm >= 0]], list(fig_ids))
    up = {"cls": cv2.resize(M["cls"], (p.W, p.H), interpolation=cv2.INTER_NEAREST),
          "lam": cv2.resize(M["lam"], (p.W, p.H), interpolation=cv2.INTER_LINEAR),
          "body": cv2.resize(body.astype(np.uint8), (p.W, p.H), interpolation=cv2.INTER_NEAREST) > 0,
          "small": M, "scale": s, "body_small": body}
    if len(_FORM) > 2:
        _FORM.clear()
    _FORM[key] = up
    return up


def save_form(p, fd: dict) -> str:
    path = p.root / "draw" / "form3d.npz"
    path.parent.mkdir(exist_ok=True)
    M = fd["small"]
    np.savez_compressed(path, cls=M["cls"], lam=M["lam"].astype(np.float16), ground=M["ground"],
                        body=fd["body_small"], scale=np.float32(fd["scale"]),
                        legend=json.dumps({"0": "nothing", "1": "lit", "2": "core shadow", "3": "cast shadow"}))
    return p.rel(path)


def form_tones(p, mask: np.ndarray, levels: int = 1, amount: float = 0.35) -> Optional[list]:
    """Shade-stage tones for an area that lies on a posed figure, from the figure's 3D form: ('shadow', m) where the
    form turns away from the light or a part / block shadows it, ('deep', m) the core of the turned-away side,
    ('light', m) the parts facing the light. None when the area is not mostly on a posed figure."""
    m = np.asarray(mask, bool)
    if m.sum() < 30:
        return None
    fd = form_data(p)
    if fd is None:
        return None
    on = fd["body"] & m
    if on.sum() < 0.5 * m.sum():
        return None
    from scipy import ndimage as ndi
    cls, lam = fd["cls"], fd["lam"]
    # clothes reach past the mannequin: their pixels take the nearest body pixel's light
    if (m & ~fd["body"]).any():
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sub_body = fd["body"][y0:y1, x0:x1] & m[y0:y1, x0:x1]
        if sub_body.any():
            _, (iy, ix) = ndi.distance_transform_edt(~sub_body, return_indices=True)
            c2 = cls[y0:y1, x0:x1][iy, ix]
            l2 = lam[y0:y1, x0:x1][iy, ix]
            cls = cls.copy()
            lam = lam.copy()
            cls[y0:y1, x0:x1] = np.where(m[y0:y1, x0:x1], c2, cls[y0:y1, x0:x1])
            lam[y0:y1, x0:x1] = np.where(m[y0:y1, x0:x1], l2, lam[y0:y1, x0:x1])
    thr = 0.05 + 0.5 * (float(amount) - 0.35)
    sig = max(1.0, 0.004 * max(p.W, p.H))
    min_px = max(12, int(0.003 * m.sum()))
    sh = m & ((lam <= thr) | (cls == 3))
    sh = dc.clean_mask(dc.smooth_mask(sh, sig) & m, min_px)
    out = []
    if sh.sum() >= 12:
        out.append(("shadow", sh))
    if levels >= 2:
        dp = dc.clean_mask(dc.smooth_mask(m & (lam <= thr - 0.35), sig) & m, min_px)
        if dp.sum() >= 12:
            out.append(("deep", dp))
    if levels >= 3:
        hl = dc.clean_mask(dc.smooth_mask(m & (lam >= 0.75) & (cls == 1), sig) & m, min_px)
        if hl.sum() >= 12:
            out.append(("light", hl))
    return out


# ============================================================================================ commands

def _ctx_parts():
    from . import cli, cli_draw
    return cli, cli_draw


def _person_table(cam: P3.Camera, height: float = 1.6) -> str:
    rows = cam.person_table(height)
    if not rows:
        return "the horizon is at the bottom: the ground is not in the picture"
    return f"a {height:g} m person standing at " + "; ".join(f"y {y:.0f} is {h:.0f} units tall ({dist:.1f} m away)"
                                                            for y, dist, h in rows)


def _camera_words(cam: P3.Camera) -> str:
    if cam.eye < 0.9:
        a = "a low camera (looking up at standing people: heads above the horizon)"
    elif cam.eye > 2.5:
        a = "a high camera (looking down: people's heads below the horizon)"
    else:
        a = "eye level (a standing adult's eyes on the horizon)"
    if abs(cam.tilt) > 2:
        a += f", tilted {'up' if cam.tilt > 0 else 'down'} {abs(cam.tilt):g} (verticals converge)"
    return a


def _overlay(p, subj_joints: Optional[str] = None):
    from .cli_draw import tag_image
    from . import draw_sketch

    def ov(img, area):
        d = dc.dstate(p)
        tags = []
        for el in d["sketch"]:
            if el["kind"] in ("block", "prop"):
                for txt, xy in label_points(p, el):
                    tags.append((txt, xy, (40, 110, 60)))
            elif is_rigged(el):
                j2 = el["j2d"]
                if subj_joints and el["subject"] == subj_joints:
                    names = [k for k in j2 if k not in ("neck-top", "head-top", "chest", "heel-r", "heel-l")]
                else:
                    names = ["hand-r", "hand-l", "foot-r", "foot-l"]
                for n in names:
                    q = dc.u2px(p, [j2[n]])[0]
                    tags.append((n, (float(q[0]), float(q[1])), (200, 70, 0)))
                top = dc.u2px(p, [j2["head-top"]])[0]
                tags.append((f"{el['subject']} {el['id']}", (float(top[0]), float(top[1]) - 0.03 * p.H), (40, 60, 150)))
            elif el["kind"] in ("figure", "head", "ellipse", "box", "line", "curve"):
                for txt, xy in draw_sketch.label_points(p, el):
                    tags.append((txt, xy, (40, 60, 150)))
        return tag_image(img, tags, area, 12)
    return ov


def _step(ctx, label, mutate, lines, show=None, note="the 3D layout on the sketch layer"):
    from . import cli_draw, draw_sketch
    p = ctx.project
    rep = cli_draw.draw_step(ctx, label=label, layer="sketch",
                             owned=(lambda: dc.dstate(p)["sketch"], draw_sketch.render_ops, mutate), lines=lines,
                             overlay=_overlay(p, show), full_view=True, note=note)
    try:
        save_maps(p)
    except Exception:  # noqa: BLE001 - the maps are derived data; the step itself is done
        pass
    return rep


def _all_ops(p, kinds=None) -> list:
    from . import draw_sketch
    ops = []
    for el in dc.dstate(p)["sketch"]:
        if (el["kind"] in KINDS_3D or is_rigged(el) or el["kind"] == "horizon") and \
                (kinds is None or el["kind"] in kinds):
            ops += draw_sketch.render_ops(p, el)
    return ops


def _ensure_camera(p, subj: Optional[str], height: float) -> List[str]:
    """The default camera (eye level) the first time a figure is posed; with a plan box for the subject, the
    horizon goes where a standing adult's eyes are in that box."""
    if camera_record(p) is not None:
        return []
    from . import draw_sketch
    d = dc.dstate(p)
    old = [el for el in d["sketch"] if el["kind"] == "horizon"]
    y = float(old[0]["y"]) if old else 450.0
    why = "eye level, horizon 450"
    box = d["subjects"].get(subj, {}).get("box") if subj else None
    if box is not None and not old:
        bh = box[3] - box[1]
        y = float(np.clip(box[3] - 0.985 * bh * 1.5 / max(height, 0.5), 5, 990))
        why = f"eye level: horizon {y:.0f}, where a standing adult's eyes are in {subj}'s plan box"
    for el in old:
        d["sketch"].remove(el)
    draw_sketch.new_element(p, "horizon", y=round(y, 1), vps=[], camera={"eye": 1.5, "fov": 50.0, "tilt": 0.0,
                                                                         "auto": True})
    return [f"camera: none was set, so the default one ({why}, eye height 1.5 m); change it with "
            f"pixelcanvas camera --horizon <y> --eye-height <m>"]


def cmd_camera(ctx, args):
    from .cli import Report, Spec, parse
    cli, C = _ctx_parts()
    p = ctx.project
    a = parse(args, Spec(USAGE["camera"], [], {"horizon": "value", "eye-height": "value", "fov": "value",
                                               "tilt": "value", "show": "flag"},
                         alias={"eye": "eye-height", "height": "eye-height", "y": "horizon", "angle": "tilt",
                                "pitch": "tilt", "lens": "fov"}))
    cur = camera_record(p)
    if cur is not None and not any(a.get(k) is not None for k in ("horizon", "eye-height", "fov", "tilt")):
        cam = camera_of(p)
        rep = Report()
        rep.add(f"camera: horizon {cam.horizon:g}, eye height {cam.eye:g} m, fov {cam.fov:g}, tilt {cam.tilt:g}: "
                f"{_camera_words(cam)}")
        rep.add(_person_table(cam))
        if not ctx.batch:
            img = _overlay(p)(p.composite_u8("view"), (0, 0, p.W, p.H))
            C.show_one(ctx, rep, img, (0, 0, p.W, p.H), "CAMERA", "the sketch with the horizon and ground grid",
                       f"camera-{cli._view_counter(p):04d}")
        rep.next = cli._finish_next(ctx)
        return rep
    old = camera_of(p)
    hz = coords.parse_float(a["horizon"], "horizon", -500, 1500, ctx.notes, USAGE["camera"]) if a.get("horizon") \
        else old.horizon
    eye = coords.parse_float(a["eye-height"], "eye-height", 0.05, 200, ctx.notes, USAGE["camera"]) \
        if a.get("eye-height") else old.eye
    fov = coords.parse_float(a["fov"], "fov", 10, 120, ctx.notes, USAGE["camera"]) if a.get("fov") else old.fov
    tilt = coords.parse_float(a["tilt"], "tilt", -60, 60, ctx.notes, USAGE["camera"]) if a.get("tilt") is not None \
        else old.tilt
    lines: List[str] = []
    d = dc.dstate(p)

    def mutate():
        from . import draw_sketch
        for el in [e for e in d["sketch"] if e["kind"] == "horizon"]:
            d["sketch"].remove(el)
        draw_sketch.new_element(p, "horizon", y=round(float(hz), 1), vps=[],
                                camera={"eye": float(eye), "fov": float(fov), "tilt": float(tilt), "auto": False})
        solved, notes = solve_scene(p)
        cam = camera_of(p)
        lines.append(f"camera: horizon {hz:g}, eye height {eye:g} m, fov {fov:g}, tilt {tilt:g}: {_camera_words(cam)}")
        lines.append(_person_table(cam))
        moved = list(solved) + [el["name"] for el in blocks_rec(p)]
        if moved:
            lines.append(f"re-placed at the same picture points: {', '.join(moved)}")
        lines.extend(notes)
        return _all_ops(p), (0, 0, p.W, p.H)
    return _step(ctx, f"camera horizon {hz:g} eye {eye:g} m", mutate, lines)


def _target(ctx, v: str):
    s = str(v).strip()
    if coords.looks_like_box(s) or (s[:1].isdigit() or s[:1] == "-") and "," in s:
        return [float(x) for x in coords.parse_point(s, ctx.notes, USAGE["pose"])]
    return s.lower()


def cmd_pose(ctx, args):
    from .cli import Report, Spec, parse
    cli, C = _ctx_parts()
    p = ctx.project
    d = dc.dstate(p)
    if args and args[0].lower() in ("list", "presets", "help"):
        rep = Report()
        word = " ".join(args[1:]).strip().lower()
        if word:
            hits = [(k, v) for k, v in P3.PRESETS.items() if word in k or word in v["desc"].lower()]
            rep.add(f"{len(hits)} presets with {word!r}:" if hits else f"no preset mentions {word!r}")
            for k, v in hits:
                rep.add(f"  {k}: {v['desc']}")
        else:
            rep.add(f"{len(P3.PRESETS)} presets (pixelcanvas pose <subject> --preset <name>; pose list <word> "
                    f"describes them):")
            for group, names in P3.preset_groups():
                rep.add(f"  {group}: {', '.join(names)}")
            rep.add("  modifiers: --lean --twist --bend <deg>, --look camera|left|right|up|down|back|at:x,y, "
                    "--energy 0..1, --mirror, --arm-r/-l raised|forward|back|down|bent|out|hip|behind|crossed|face|"
                    "chin|hair, --leg-r/-l forward|back|bent|raised|kick|out|tucked|relaxed, --hand-r <shape>, "
                    "--hand-r-to <anchor|x,y|block>, --on <block>, --check")
        rep.next = cli._finish_next(ctx)
        return rep
    a = parse(args, Spec(USAGE["pose"], ["subject"],
                         {"preset": "value", "at": "value", "height": "value", "heads": "value", "chibi": "flag",
                          "yaw": "value", "set": "value", "mirror": "flag", "on": "value", "hand-r": "value",
                          "hand-l": "value", "foot-r": "value", "foot-l": "value", "show": "value", "lift": "value",
                          "reset": "flag", "hand-shape-r": "value", "hand-shape-l": "value",
                          "lean": "value", "twist": "value", "bend": "value", "look": "value", "energy": "value",
                          "arm-r": "value", "arm-l": "value", "leg-r": "value", "leg-l": "value", "check": "flag",
                          "no-gesture": "flag", "no-balance": "flag"},
                         alias={"pose": "preset", "facing": "yaw", "face": "yaw", "turn": "yaw", "edit": "set",
                                "joints": "set", "joint": "set", "sit-on": "on", "stand-on": "on", "size": "height",
                                "right-hand": "hand-r", "left-hand": "hand-l", "right-foot": "foot-r",
                                "left-foot": "foot-l", "hr": "hand-r", "hl": "hand-l", "flip": "mirror",
                                "hand-r-shape": "hand-shape-r", "hand-l-shape": "hand-shape-l",
                                "hand-r-to": "hand-r", "hand-l-to": "hand-l", "foot-r-to": "foot-r",
                                "foot-l-to": "foot-l", "right-arm": "arm-r", "left-arm": "arm-l",
                                "right-leg": "leg-r", "left-leg": "leg-l", "look-at": "look", "gaze": "look",
                                "exaggerate": "energy"},
                         min_pos=1))
    subj = str(a["subject"]).lower()
    if subj in P3.PRESETS:
        raise UsageError(f"{subj!r} is a preset: name the subject first", f"pixelcanvas pose girl --preset {subj}")
    old = next((el for el in d["sketch"] if el["kind"] == "figure" and el.get("subject") == subj), None)
    spec = dict(old["rig"]) if old is not None and old.get("rig") and not a.get("reset") else {}
    req = str(p.state.get("request") or "").lower()
    if a.get("preset"):
        spec["preset"] = P3.preset_key(a["preset"])
        if not a.get("set"):
            spec["edits"] = {}
    spec.setdefault("preset", "stand")
    if a.get("set"):
        spec["edits"] = P3.merge_edits(spec.get("edits") or {}, P3.parse_edits(a["set"]))
    mods = {} if a.get("preset") else dict(spec.get("mods") or {})
    for k in ("lean", "twist", "bend"):
        if a.get(k) is not None:
            mods[k] = coords.parse_float(a[k], k, -90, 90, ctx.notes, USAGE["pose"])
    if a.get("energy") is not None:
        mods["energy"] = coords.parse_float(a["energy"], "energy", 0, 1, ctx.notes, USAGE["pose"])
    if a.get("look"):
        lk = P3.parse_look(a["look"])
        if lk == "ahead":
            mods.pop("look", None)
        else:
            mods["look"] = lk
    for k in ("arm-r", "arm-l"):
        if a.get(k):
            mods[k] = P3.arm_word(a[k])
    for k in ("leg-r", "leg-l"):
        if a.get(k):
            mods[k] = P3.leg_word(a[k])
    spec["mods"] = mods
    if a.get("no-gesture"):
        spec["gesture"] = False
    elif a.get("preset"):
        spec.pop("gesture", None)
    if a.get("no-balance"):
        spec["balance"] = False
    elif a.get("preset"):
        spec.pop("balance", None)
    if a.get("chibi"):
        spec["heads"] = 2.5
    if a.get("heads"):
        spec["heads"] = coords.parse_float(a["heads"], "heads", 2.0, 9.0, ctx.notes, USAGE["pose"])
    spec.setdefault("heads", 2.5 if "chibi" in req else 6.5)
    if a.get("height"):
        spec["height"] = coords.parse_float(str(a["height"]).lower().replace("m", ""), "height", 0.3, 4.0, ctx.notes,
                                            USAGE["pose"])
    spec.setdefault("height", 1.6 if float(spec["heads"]) >= 4.2 else 1.1)
    if a.get("yaw") is not None:
        spec["yaw"] = coords.parse_float(a["yaw"], "yaw", -360, 360, ctx.notes, USAGE["pose"])
    elif a.get("preset"):
        spec["yaw"] = None
    if a.get("mirror"):
        spec["mirror"] = True
    elif a.get("preset"):
        spec["mirror"] = False
    if a.get("lift") is not None:
        spec["lift"] = coords.parse_float(a["lift"], "lift", 0, 20, ctx.notes, USAGE["pose"])
    if a.get("on"):
        on = str(a["on"]).lower()
        spec["on"] = None if on in ("ground", "floor", "none") else on
    targets = dict(spec.get("targets") or {})
    from .figure3d import HAND_SHAPES
    for s in "rl":                          # --hand-r fist: a hand shape; --hand-r hip / x,y / <block>: a target
        v = str(a.get(f"hand-{s}") or "").lower()
        if v in HAND_SHAPES and not a.get(f"hand-shape-{s}"):
            a[f"hand-shape-{s}"] = v
            a[f"hand-{s}"] = None
    for k in ("hand-r", "hand-l", "foot-r", "foot-l"):
        if a.get(k):
            v = str(a[k]).lower()
            if v in ("none", "free", "off"):
                targets.pop(k, None)
            else:
                targets[k] = _target(ctx, a[k])
    spec["targets"] = targets
    hands = dict(spec.get("hands") or {})
    if a.get("preset"):
        hands = dict(P3.PRESETS[spec["preset"]].get("hands") or {})
        if spec.get("mirror"):
            hands = {("l" if s == "r" else "r"): v for s, v in hands.items()}
    for s in "rl":
        v = a.get(f"hand-shape-{s}")
        if v:
            v = str(v).lower()
            if v in ("auto", "default", "preset"):
                hands.pop(s, None)
            elif v not in HAND_SHAPES:
                raise UsageError(f"--hand-shape-{s} is one of {', '.join(HAND_SHAPES)}",
                                 f"pixelcanvas pose {subj} --hand-shape-{s} fist")
            else:
                hands[s] = v
    spec["hands"] = hands
    if a.get("at"):
        spec["at"] = [float(v) for v in coords.parse_point(a["at"], ctx.notes, USAGE["pose"])]
    elif not spec.get("at") and not spec.get("on"):
        box = d["subjects"].get(subj, {}).get("box")
        spec["at"] = [round((box[0] + box[2]) / 2, 1), round(box[3] - 0.015 * (box[3] - box[1]), 1)] if box \
            else [500.0, 950.0]
    show = str(a.get("show") or "").lower()
    changes = [k for k, v in a.items() if v not in (None, False, "") and k not in ("subject", "show", "check")]
    if a.get("check") and not changes and old is not None and old.get("rig"):
        # a check, not a change: the pose's problems in words; no picture, no step
        from . import cli_draw
        solved, _ = solve_scene(p)
        res = solved.get(subj)
        rep = cli_draw.DReport()
        probs = res["check"] if res else ["not posed"]
        rep.add(f"check {subj} ({old['rig'].get('preset')}): " + ("ok, no problems" if not probs else
                                                                 f"{len(probs)} problem(s)"))
        for line in probs:
            rep.add("  " + line)
        rep.add(f"balance: {res['balance']}" if res else "")
        rep.next = cli._finish_next(ctx)
        return rep
    if show and old is not None and old.get("j2d") and not changes:
        # a look, not a change: the posed figure with its joints named; no step, no history, the state untouched
        from . import cli_draw
        rep = cli_draw.DReport()
        rep.add(f"{subj}: {old['rig'].get('preset')} (joints named; -r / -l = the figure's own right / left); "
                f"nothing changed")
        if not ctx.batch:
            img = _overlay(p, subj if show.startswith("joint") else None)(dc.composite_u8(p, "view"),
                                                                          (0, 0, p.W, p.H))
            cli_draw.show_one(ctx, rep, img, (0, 0, p.W, p.H), f"POSE: {subj}", "the posed figure, joints named",
                              f"pose-{cli._view_counter(p):04d}")
        rep.next = cli._finish_next(ctx)
        return rep
    lines: List[str] = []
    if subj not in d["subjects"]:
        ctx.notes.append(f"{subj} is not a planned subject (pixelcanvas plan subject {subj} --box ... gives it a box)")

    def mutate():
        from . import draw_sketch
        lines.extend(_ensure_camera(p, subj, float(spec["height"])))
        for el in [e for e in d["sketch"] if e["kind"] == "figure" and e.get("subject") == subj]:
            if el is not old:
                d["sketch"].remove(el)
        if old is not None:
            el = old
            el["n"] = dc.next_order(d)
        else:
            el = draw_sketch.new_element(p, "figure", subject=subj, pose=spec["preset"], heads=float(spec["heads"]),
                                         joints={})
        el["rig"] = json.loads(json.dumps(spec))
        el["joints"] = el.get("rig_joints") or {}
        el["rig_joints"] = el["joints"]
        solved, notes = solve_scene(p)
        res = solved.get(subj)
        if res is None:
            raise UsageError(f"{subj} could not be posed", USAGE["pose"])
        lines.extend(_pose_report(p, el, res))
        lines.extend(notes)
        lines.extend(res["notes"])
        return _all_ops(p, ("figure", "prop")), _fig_box_px(p, el)
    return _step(ctx, f"pose {subj} {spec['preset']}", mutate, lines, show=subj if show.startswith("joint") else None,
                 note="the posed figure on the sketch layer (solid: seen; dashed: behind the figure's own parts; "
                      "-r / -l = the figure's own right / left)")


def _fig_box_px(p, el):
    sc = get_scene(p)
    b = owner_box(p, sc, el["id"], visible=False)
    if b is None:
        return (0, 0, p.W, p.H)
    return tuple(int(v) for v in (max(0, b[0]), max(0, b[1]), min(p.W, b[2]), min(p.H, b[3])))


def _pose_report(p, el, res) -> List[str]:
    cam = camera_of(p)
    sc = get_scene(p)
    rig = el["rig"]
    J, F, body = res["J"], res["F"], res["body"]
    fy, _ = P3.view_angles(cam, F["chest"], J["chest"])
    dist = float(cam.to_cam([J["pelvis"]])[0, 2])
    out = [f"{el['subject']} ({el['id']}): {res['preset']} ({P3.PRESETS[res['preset']]['desc']}), "
           f"{body.heads:g} heads, {body.Ht:g} m, {dist:.1f} m from the camera, {P3.facing_words(fy)}"]
    vb = owner_box(p, sc, el["id"])
    if vb is not None:
        ub = coords.px_box_to_units(tuple(int(v) for v in vb), p.W, p.H)
        hh = abs(el["j2d"]["head-top"][1] - el["j2d"]["chin"][1])
        out.append(f"box {coords.fmt_box(ub)} (units); head {hh:.0f} units tall; picture-left limbs are the figure's "
                   f"{'right' if el.get('picture_left') == 'r' else 'left'} ones (lsh, lwr ... in the 2D joints)")
    cts = res["contacts"]
    out.append("touching: " + (", ".join(f"{n} on the {s}" for n, s, _ in cts) if cts else "nothing") +
               f"; balance: {res['balance']}" + ("; check: ok" if not res.get("check") else ""))
    for line in res.get("check") or []:
        out.append(f"check: {line}")
    mods = rig.get("mods") or {}
    if mods:
        out.append("modifiers: " + ", ".join(f"--{k} {v[0]:g},{v[1]:g}" if isinstance(v, list) else
                                            f"--{k} {v:g}" if isinstance(v, float) else f"--{k} {v}"
                                            for k, v in mods.items()))
    if res["support"] == "ground" and P3.PRESETS[res["preset"]]["rest"] == "seat" and \
            not any(n == "seat" for n, _, _ in cts):
        out.append(f"the seat is in the air: give her something to sit on (pixelcanvas block bench box --at x,y, then "
                   f"pixelcanvas pose {el['subject']} --on bench)")
    vis = visibility(sc, el["id"])
    hid = [f"{g} {'hidden' if v < 0.3 else 'partly hidden'} ({v:.0%} seen)" for g, v in sorted(vis.items())
           if v < 0.7 and g not in ("torso",)]
    if hid:
        out.append("hidden: " + "; ".join(hid))
    if res["clamps"]:
        out.append("clamped to the body's limits: " + "; ".join(res["clamps"][:4]))
    out.extend(res["ik"])
    hy, hp = el["head_view"]
    out.append(f"head seen at yaw {hy:.0f}, pitch {hp:.0f}: pixelcanvas sketch head {el['subject']} takes them "
               f"(the face is drawn on it)")
    if el["subject"]:
        out.append(f'change: pixelcanvas pose {el["subject"]} --lean 15 --twist 20 --look camera --arm-l hip '
                   '--energy 0.8 | --set "arm-r: raise 90; knee-l: 60" (degrees; -r / -l = her own right / left; '
                   '--show joints names them)')
    return out


def cmd_block(ctx, args):
    from .cli import Report, Spec, parse
    cli, C = _ctx_parts()
    p = ctx.project
    d = dc.dstate(p)
    if not args:
        raise UsageError("say the block's name and kind (box, stairs, cylinder, plane, railing)", USAGE["block"])
    if args[0].lower() in ("list", "show"):
        rep = Report()
        for el in blocks_rec(p):
            rep.add(f"  {el['id']} {el['name']}: {el['type']} {'x'.join(f'{v:g}' for v in el['size'])} m at "
                    f"{coords.fmt_point(el['at']) if el.get('at') else '-'}, top {_block(el).top_centre()[1]:.2f} m")
        if not blocks_rec(p):
            rep.add("no blocks yet")
        rep.next = cli._finish_next(ctx)
        return rep
    if args[0].lower() in ("delete", "remove", "rm"):
        if len(args) < 2:
            raise UsageError("say which block", "pixelcanvas block delete wall")
        name = args[1].lower()
        el = next((e for e in blocks_rec(p) if e["name"] == name), None)
        if el is None:
            raise UsageError(f"no block {name!r}; blocks: {', '.join(e['name'] for e in blocks_rec(p)) or 'none'}",
                             "pixelcanvas block list")
        lines: List[str] = []

        def mutate_del():
            d["sketch"].remove(el)
            solved, notes = solve_scene(p)
            lines.extend(notes)
            return _all_ops(p), (0, 0, p.W, p.H)
        return _step(ctx, f"block delete {name}", mutate_del, lines)
    a = parse(args, Spec(USAGE["block"], ["name", "kind"],
                         {"at": "value", "size": "value", "rot": "value", "steps": "value", "lift": "value",
                          "offset": "value", "color": "value"},
                         alias={"rotate": "rot", "turn": "rot", "angle": "rot", "yaw": "rot", "n": "steps", "colour": "color",
                                "height": "lift", "pos": "at", "move": "offset"}, min_pos=1))
    name = str(a["name"]).lower()
    old = next((e for e in blocks_rec(p) if e["name"] == name), None)
    word = str(a.get("kind") or "").lower()
    kind = P3.block_kind(word) if word else (old["type"] if old else P3.block_kind(name))
    if kind is None:
        raise UsageError(f"unknown block kind {word or name!r}; kinds: {', '.join(P3.BLOCK_KINDS)} (or a word like "
                         f"wall, bench, building, step, stairs, pole, lamp, tree, floor, water, railing, fence)",
                         f"pixelcanvas block {name} box --at 500,800 --size 1.5,0.45,0.45")
    dflt = P3.BLOCK_WORDS.get(word, P3.BLOCK_WORDS.get(name, P3.BLOCK_WORDS[kind if kind != "cylinder" else
                                                                            "cylinder"]))[1]
    if a.get("size"):
        nums = [v for v, _, _ in coords._numbers(a["size"])]
        if len(nums) not in (1, 2, 3):
            raise UsageError("--size is width,depth,height in metres", USAGE["block"])
        size = list(dflt)
        if len(nums) == 1:
            size = [nums[0], nums[0], nums[0]] if kind != "cylinder" else [nums[0], nums[0], dflt[2]]
        elif len(nums) == 2:
            size = [nums[0], nums[0], nums[1]] if kind == "cylinder" else [nums[0], dflt[1], nums[1]]
        else:
            size = nums
    else:
        size = list(old["size"]) if old and old["type"] == kind else list(dflt)
    rec = {"name": name, "type": kind, "size": [float(v) for v in size],
           "at": [float(v) for v in coords.parse_point(a["at"], ctx.notes, USAGE["block"])] if a.get("at")
           else (old.get("at") if old else None),
           "rot": float(coords.parse_float(a["rot"], "rot", -360, 360, ctx.notes)) if a.get("rot") is not None
           else (old.get("rot", 0.0) if old else 0.0),
           "steps": int(coords.parse_int(a["steps"], "steps", 1, 40, ctx.notes)) if a.get("steps")
           else (old.get("steps", 6) if old else 6),
           "lift": float(coords.parse_float(a["lift"], "lift", 0, 100, ctx.notes)) if a.get("lift") is not None
           else (old.get("lift", 0.0) if old else 0.0),
           "offset": [v for v, _, _ in coords._numbers(a["offset"])][:2] if a.get("offset")
           else (old.get("offset") if old else None),
           "color": str(a["color"]) if a.get("color") else (old.get("color") if old else None)}
    if a.get("color"):
        from .draw_plan import swatch
        swatch(p, a["color"], USAGE["block"])
    cam0 = camera_of(p)
    if rec["at"] is not None and cam0.ground_point(rec["at"][0], rec["at"][1]) is None:
        raise UsageError(f"--at {coords.fmt_point(rec['at'])} is not on the ground: it must be below the horizon "
                         f"(y > {cam0.horizon:.0f})", f"pixelcanvas block {name} {kind} --at 500,"
                                                      f"{min(990, cam0.horizon + 300):.0f}")
    lines: List[str] = []

    def mutate():
        from . import draw_sketch
        if old is not None:
            old.update(rec)
            old["n"] = dc.next_order(d)
            el = old
        else:
            el = draw_sketch.new_element(p, "block", **rec)
        solved, notes = solve_scene(p)
        cam = camera_of(p)
        b = _block(el)
        dist = float(cam.to_cam([b.pos])[0, 2])
        lines.append(f"block {name} ({el['id']}): {kind} {' x '.join(f'{v:g}' for v in el['size'])} m (width, depth, "
                     f"height), front at {dist:.1f} m from the camera, top {b.top_centre()[1]:.2f} m" +
                     (f", {el['steps']} steps" if kind == "stairs" else "") +
                     (f", turned {el['rot']:g}" if el.get("rot") else ""))
        sc = get_scene(p)
        vb = owner_box(p, sc, el["id"])
        if vb is not None:
            lines.append(f"box {coords.fmt_box(coords.px_box_to_units(tuple(int(v) for v in vb), p.W, p.H))} (units)")
        users = [f["subject"] for f in figures(p) if f["rig"].get("on") == name or name in
                 (f["rig"].get("targets") or {}).values()]
        if users:
            lines.append(f"re-posed on it: {', '.join(users)}")
        elif kind in ("box", "stairs"):
            lines.append(f"a figure uses it with pixelcanvas pose <subject> --preset "
                         f"{'sit-stairs' if kind == 'stairs' else 'sit-ledge'} --on {name}   (or --hand-r {name})")
        lines.extend(notes)
        return _all_ops(p), (0, 0, p.W, p.H)
    return _step(ctx, f"block {name} {kind}", mutate, lines)


def cmd_prop(ctx, args):
    from .cli import Spec, parse
    p = ctx.project
    d = dc.dstate(p)
    if args and args[0].lower() in ("delete", "remove", "rm"):
        name = args[1].lower() if len(args) > 1 else ""
        el = next((e for e in props_rec(p) if e["name"] == name), None)
        if el is None:
            raise UsageError(f"no prop {name!r}; props: {', '.join(e['name'] for e in props_rec(p)) or 'none'}",
                             "pixelcanvas prop delete sword")

        def mutate_del():
            d["sketch"].remove(el)
            return _all_ops(p), (0, 0, p.W, p.H)
        return _step(ctx, f"prop delete {name}", mutate_del, [])
    a = parse(args, Spec(USAGE["prop"], ["name"], {"kind": "value", "hand": "value", "of": "value",
                                                   "length": "value"},
                         alias={"type": "kind", "subject": "of", "in": "hand", "size": "length"}, min_pos=1))
    name = str(a["name"]).lower()
    kind = str(a.get("kind") or name).lower()
    if kind not in P3.PROP_KINDS:
        raise UsageError(f"unknown prop kind {kind!r}; kinds: {', '.join(P3.PROP_KINDS)}",
                         f"pixelcanvas prop {name} --kind stick --hand r")
    hand = str(a.get("hand") or ("both" if kind in ("ball",) else "r")).lower()
    hand = {"right": "r", "left": "l", "two": "both", "2": "both"}.get(hand, hand)
    if hand not in ("r", "l", "both"):
        raise UsageError("--hand is r, l or both (the figure's own right / left hand)", USAGE["prop"])
    figs = figures(p)
    subj = str(a["of"]).lower() if a.get("of") else (figs[0]["subject"] if len(figs) == 1 else None)
    if subj is None or not any(f["subject"] == subj for f in figs):
        names = ", ".join(f["subject"] for f in figs) or "none posed yet"
        raise UsageError(f"say whose hand holds it with --of <subject> (posed figures: {names})",
                         f"pixelcanvas prop {name} --kind {kind} --hand r --of girl")
    length = coords.parse_float(a["length"], "length", 0.02, 5, ctx.notes) if a.get("length") else None
    lines: List[str] = []

    def mutate():
        from . import draw_sketch
        old = next((e for e in props_rec(p) if e["name"] == name), None)
        rec = {"name": name, "type": kind, "subject": subj, "hand": hand, "length": length}
        if old is not None:
            old.update(rec)
            old["n"] = dc.next_order(d)
            el = old
        else:
            el = draw_sketch.new_element(p, "prop", **rec)
        lines.append(f"prop {name} ({el['id']}): {kind}, {length or P3.PROP_KINDS[kind]:g} m, in {subj}'s "
                     f"{'hands' if hand == 'both' else ('right' if hand == 'r' else 'left') + ' hand'}; it moves "
                     f"with the hand when the pose changes")
        return _all_ops(p, ("prop",)), (0, 0, p.W, p.H)
    return _step(ctx, f"prop {name} {kind}", mutate, lines)


def cmd_form(ctx, args):
    from .cli import Report, Spec, parse
    from . import view
    cli, C = _ctx_parts()
    p = ctx.project
    a = parse(args, Spec(USAGE["form"], ["subject"], {}))
    figs = figures(p)
    if not figs:
        raise UsageError("no posed figure: form shading comes from the 3D figures", "pixelcanvas pose girl --preset stand")
    subj = str(a.get("subject") or "").lower()
    if subj and not any(f["subject"] == subj for f in figs):
        raise UsageError(f"{subj!r} is not posed; posed: {', '.join(f['subject'] for f in figs)}", USAGE["form"])
    fd = form_data(p)
    path = save_form(p, fd)
    d = dc.dstate(p)
    rep = Report()
    rep.add(f"light from the {dc.light_words(d['light'].get('deg', 135))} (plan light); 3D form shading saved: {path}")
    small = fd["small"]
    owner_of = [pr.get("owner") for pr in scene_prims(p)]
    for f in figs:
        if subj and f["subject"] != subj:
            continue
        ks = [k for k, o in enumerate(owner_of) if o == f["id"]]
        m = np.isin(small["id"], ks)
        if not m.any():
            continue
        c = small["cls"][m]
        rep.add(f"{f['subject']}: lit {np.mean(c == 1):.0%}, core shadow {np.mean(c == 2):.0%}, cast shadow "
                f"{np.mean(c == 3):.0%}")
    g = small["ground"]
    if g.any():
        rep.add(f"ground in cast shadow: {np.mean(small['cls'][g] == 3):.1%} of the ground in the picture")
    rep.add("the shade stage uses it: pixelcanvas shade <part> on an area of a posed figure follows these forms")
    if not ctx.batch:
        import cv2
        base = p.composite_u8("view").astype(np.float32)
        tone = np.array([0, 245, 150, 90], np.float32)[fd["cls"]]
        any_ = fd["cls"] > 0
        img = base.copy()
        img[any_] = 0.35 * base[any_] + 0.65 * tone[any_, None]
        C.show_one(ctx, rep, img.clip(0, 255).astype(np.uint8), (0, 0, p.W, p.H), "FORM (light, core, cast)",
                   "light grey lit, mid grey core shadow, dark grey cast shadow", f"form-{cli._view_counter(p):04d}")
    rep.next = cli._finish_next(ctx)
    return rep


COMMANDS = {"camera": cmd_camera, "pose": cmd_pose, "block": cmd_block, "prop": cmd_prop, "form": cmd_form}
