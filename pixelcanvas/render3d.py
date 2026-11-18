"""Rendering 3D layout primitives through the camera: a ray-cast z-buffer (part id, face, depth, normal per pixel),
construction lines from the id edges (hidden lines of a figure's own parts as dashes), and form shading from the
normals with cast shadows from every part and block. Pure functions on numpy arrays.

A primitive is ``{"t": "ell" | "box" | "cyl", "c": centre (3,), "A": 3x3 (columns = the semi-axes in world
space), "owner": record id, "part": name, "group": name}``: the unit sphere, cube [-1, 1]^3 or cylinder
(x^2 + z^2 <= 1, |y| <= 1) mapped by x = c + A q.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .pose3d import Camera

EPS = 1e-4
_CUBE = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)


def intersect(pr: dict, O, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First hit t (> EPS, inf = miss), world normal (unit) and face index of rays O + t D with a primitive."""
    A = np.asarray(pr["A"], float)
    Ai = np.linalg.inv(A)
    o = (np.asarray(O, float) - pr["c"]) @ Ai.T
    d = np.asarray(D, float) @ Ai.T
    if o.ndim == 1:
        o = np.broadcast_to(o, d.shape)
    n = len(d)
    t = np.full(n, np.inf)
    nl = np.zeros((n, 3))
    face = np.zeros(n, np.int8)
    kind = pr["t"]
    if kind == "ell":
        a = (d * d).sum(1)
        b = 2 * (o * d).sum(1)
        c = (o * o).sum(1) - 1
        disc = b * b - 4 * a * c
        ok = disc >= 0
        sq = np.sqrt(np.where(ok, disc, 0))
        t0 = (-b - sq) / (2 * a)
        t1 = (-b + sq) / (2 * a)
        tt = np.where(t0 > EPS, t0, np.where(t1 > EPS, t1, np.inf))
        t = np.where(ok, tt, np.inf)
        hit = np.isfinite(t)
        nl[hit] = o[hit] + t[hit, None] * d[hit]
    elif kind == "box":
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / np.where(np.abs(d) < 1e-12, 1e-12, d)
            t1 = (-1 - o) * inv
            t2 = (1 - o) * inv
        tn = np.minimum(t1, t2)
        tf = np.maximum(t1, t2)
        tmin = tn.max(1)
        tmax = tf.min(1)
        ok = tmax >= np.maximum(tmin, EPS)
        enter = tmin > EPS
        t = np.where(ok, np.where(enter, tmin, tmax), np.inf)
        ax = np.where(enter, tn.argmax(1), tf.argmin(1))
        idx = np.arange(n)
        sgn = np.where(enter, -np.sign(d[idx, ax]), np.sign(d[idx, ax]))
        nl[idx, ax] = sgn
        face = (ax * 2 + (sgn > 0)).astype(np.int8)
        hit = np.isfinite(t)
        nl[~hit] = 0
    else:  # cylinder
        a = d[:, 0] ** 2 + d[:, 2] ** 2
        b = 2 * (o[:, 0] * d[:, 0] + o[:, 2] * d[:, 2])
        c = o[:, 0] ** 2 + o[:, 2] ** 2 - 1
        disc = b * b - 4 * a * c
        ok = (disc >= 0) & (a > 1e-12)
        sq = np.sqrt(np.where(ok, disc, 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            ts = []
            for tc in ((-b - sq) / (2 * a), (-b + sq) / (2 * a)):
                y = o[:, 1] + tc * d[:, 1]
                ts.append(np.where(ok & (tc > EPS) & (np.abs(y) <= 1), tc, np.inf))
            side = np.minimum(ts[0], ts[1])
            caps = []
            for yy in (1.0, -1.0):
                tc = (yy - o[:, 1]) / np.where(np.abs(d[:, 1]) < 1e-12, 1e-12, d[:, 1])
                x = o[:, 0] + tc * d[:, 0]
                z = o[:, 2] + tc * d[:, 2]
                caps.append(np.where((tc > EPS) & (x * x + z * z <= 1), tc, np.inf))
        allt = np.stack([side, caps[0], caps[1]], 1)
        which = allt.argmin(1)
        t = allt[np.arange(n), which]
        face = which.astype(np.int8)
        hit = np.isfinite(t)
        p = o + np.where(hit, t, 0)[:, None] * d
        nl[:, 0] = np.where(which == 0, p[:, 0], 0)
        nl[:, 2] = np.where(which == 0, p[:, 2], 0)
        nl[:, 1] = np.where(which == 1, 1.0, np.where(which == 2, -1.0, 0.0))
        nl[~hit] = 0
    nw = nl @ Ai
    nn = np.linalg.norm(nw, axis=1, keepdims=True)
    nw = np.where(nn > 1e-12, nw / np.maximum(nn, 1e-12), 0)
    return t, nw, face


def bound_radius(pr: dict) -> float:
    A = np.asarray(pr["A"], float)
    cols = np.linalg.norm(A, axis=0)
    return float(cols.max() if pr["t"] == "ell" else math.sqrt((cols ** 2).sum()))


def _bbox(cam: Camera, pr: dict) -> Optional[Tuple[int, int, int, int]]:
    P = pr["c"] + _CUBE @ np.asarray(pr["A"], float).T
    q = cam.to_cam(P)
    front = q[:, 2] > 0.05
    if not front.any():
        return None
    if not front.all():
        x0, y0, x1, y1 = 0, 0, cam.W, cam.H
    else:
        px, _ = cam.project(P)
        x0, y0 = int(math.floor(px[:, 0].min())) - 1, int(math.floor(px[:, 1].min())) - 1
        x1, y1 = int(math.ceil(px[:, 0].max())) + 2, int(math.ceil(px[:, 1].max())) + 2
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(cam.W, x1), min(cam.H, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def render(cam: Camera, prims: List[dict]) -> dict:
    """Z-buffer over the camera's picture: ``id`` (index into prims, -1 = nothing), ``face``, ``depth`` (camera z,
    inf = nothing), ``normal`` (screen space: x right, y down, z toward the viewer) and ``own`` (per primitive: its
    pixel box and its full silhouette there, hidden parts included)."""
    W, H = cam.W, cam.H
    ids = np.full((H, W), -1, np.int32)
    face = np.zeros((H, W), np.int8)
    depth = np.full((H, W), np.inf, np.float32)
    normal = np.zeros((H, W, 3), np.float32)
    own: List[Optional[tuple]] = []
    for k, pr in enumerate(prims):
        bb = _bbox(cam, pr)
        if bb is None:
            own.append(None)
            continue
        x0, y0, x1, y1 = bb
        ys, xs = np.mgrid[y0:y1, x0:x1]
        D = cam.rays(xs.ravel() + 0.5, ys.ravel() + 0.5)
        t, n, f = intersect(pr, cam.C, D)
        hit = np.isfinite(t) & (t > 0.05)
        h, w = y1 - y0, x1 - x0
        hm = hit.reshape(h, w)
        own.append((bb, hm))
        if not hm.any():
            continue
        tt = t.reshape(h, w).astype(np.float32)
        sub = depth[y0:y1, x0:x1]
        better = hm & (tt < sub)
        sub[better] = tt[better]
        ids[y0:y1, x0:x1][better] = k
        face[y0:y1, x0:x1][better] = f.reshape(h, w)[better]
        nr = n.reshape(h, w, 3)
        scr = np.stack([nr @ cam.right, -(nr @ cam.up), -(nr @ cam.fwd)], -1).astype(np.float32)
        normal[y0:y1, x0:x1][better] = scr[better]
    return {"id": ids, "face": face, "depth": depth, "normal": normal, "own": own, "W": W, "H": H}


# ============================================================================================ lines

def _smooth_closed(P: np.ndarray, sigma: float) -> np.ndarray:
    n = len(P)
    if n < 6 or sigma < 0.3:
        return P
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.vstack([P[-r:], P, P[:r]])
    out = np.stack([np.convolve(pad[:, i], k, mode="valid") for i in range(2)], 1)
    return out[:n]


def _thin(P: np.ndarray, step: float) -> np.ndarray:
    if len(P) < 3:
        return P
    seg = np.hypot(*np.diff(P, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    keep = np.unique(np.searchsorted(cum, np.arange(0, cum[-1], step)))
    return P[keep]


def _dashes(P: np.ndarray, dash: float) -> List[np.ndarray]:
    if len(P) < 2:
        return []
    seg = np.hypot(*np.diff(P, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    out = []
    s = 0.0
    while s < cum[-1]:
        e = min(s + dash, cum[-1])
        m = (cum >= s) & (cum <= e)
        if m.sum() >= 2:
            out.append(P[m])
        s += 2 * dash
    return out


def contour_lines(R: dict, prims: List[dict], scale: float, skip=frozenset(),
                  dash_px: float = 12.0) -> Dict[str, List[Tuple[np.ndarray, bool, float]]]:
    """Construction lines per owner, in canvas pixels (render pixels / scale): the outline of every visible face
    of every primitive (weight 1), and for figure parts (``hidden_lines``) the outline hidden behind the same
    figure's other parts as dashes (weight 0.3). Primitives whose part is in ``skip`` (a head with its own head
    construction) block the view but draw no line."""
    ids, face = R["id"], R["face"]
    out: Dict[str, List[Tuple[np.ndarray, bool, float]]] = {}
    sig = 1.2
    k3 = np.ones((3, 3), np.uint8)
    for k, pr in enumerate(prims):
        rec = R["own"][k]
        if rec is None:
            continue
        (x0, y0, x1, y1), hm = rec
        owner = pr.get("owner", "")
        lst = out.setdefault(owner, [])
        sub = ids[y0:y1, x0:x1]
        vis = sub == k
        if (owner, pr["part"]) in skip:
            continue
        if vis.any():
            sf = face[y0:y1, x0:x1]
            for f in np.unique(sf[vis]):
                m = (vis & (sf == f)).astype(np.uint8)
                if m.sum() < 6:
                    continue
                cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for c in cs:
                    if len(c) < 8:
                        continue
                    P = c[:, 0, :].astype(float) + [x0 + 0.5, y0 + 0.5]
                    P = _smooth_closed(P, sig) if pr["t"] == "ell" else P
                    P = _thin(np.vstack([P, P[:1]]), 1.5) / scale
                    if pr["t"] != "ell":
                        P = cv2.approxPolyDP(P.astype(np.float32).reshape(-1, 1, 2), 0.8 / scale,
                                             True).reshape(-1, 2).astype(float)
                        P = np.vstack([P, P[:1]])
                    lst.append((P, False, 1.0))
        if not pr.get("hidden_lines") or not hm.any():
            continue
        visd = cv2.dilate(vis.astype(np.uint8), k3, iterations=2) > 0
        cs, _ = cv2.findContours(hm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            if len(c) < 12:
                continue
            q = c[:, 0, :]
            occ = sub[q[:, 1], q[:, 0]]
            same = np.array([o >= 0 and prims[o].get("owner") == owner for o in occ])
            hidden = (~visd[q[:, 1], q[:, 0]]) & same
            if not hidden.any():
                continue
            if hidden.all():
                runs = [np.arange(len(q))]
            else:
                start = int(np.argmin(hidden))           # roll so the contour starts on a visible point
                order = np.roll(np.arange(len(q)), -start)
                hs = hidden[order]
                runs, cur = [], []
                for i, hflag in zip(order, hs):
                    if hflag:
                        cur.append(i)
                    elif cur:
                        runs.append(np.array(cur))
                        cur = []
                if cur:
                    runs.append(np.array(cur))
            for run in runs:
                if len(run) < 6:
                    continue
                P = (q[run].astype(float) + [x0 + 0.5, y0 + 0.5]) / scale
                for dsh in _dashes(_thin(P, 1.5), dash_px):
                    if len(dsh) >= 2:
                        lst.append((dsh, False, 0.3))
    return out


# ============================================================================================ form shading

def light_world(cam: Camera, light_screen) -> np.ndarray:
    """The plan light (screen space: x right, y down, z toward the viewer) as a world direction toward the light."""
    l = np.asarray(light_screen, float)
    v = l[0] * cam.right + (-l[1]) * cam.up + (-l[2]) * cam.fwd
    return v / (np.linalg.norm(v) + 1e-12)


def form_maps(cam: Camera, prims: List[dict], light_screen, R: Optional[dict] = None, thr: float = 0.0,
              ground: bool = True) -> dict:
    """Per pixel: ``lam`` = n . l (-1..1), ``cls`` 0 nothing / 1 lit / 2 core (the form turns away from the light)
    / 3 cast (a part or block stands between it and the light), ``ground`` = the ground plane was hit there. The
    render ``R`` is reused when given (same camera size)."""
    if R is None:
        R = render(cam, prims)
    H, W = R["id"].shape
    Lw = light_world(cam, light_screen)
    ys, xs = np.mgrid[0:H, 0:W]
    D = cam.rays(xs + 0.5, ys + 0.5)
    hit = R["id"] >= 0
    t = np.where(hit, R["depth"], np.inf)
    ns = R["normal"]
    nw = ns[..., 0:1] * cam.right + (-ns[..., 1:2]) * cam.up + (-ns[..., 2:3]) * cam.fwd
    gmask = np.zeros((H, W), bool)
    if ground:
        with np.errstate(divide="ignore", invalid="ignore"):
            tg = (0.0 - cam.eye) / D[..., 1]
        gmask = (~hit) & (tg > 0.05) & np.isfinite(tg) & (tg < 400)
        t = np.where(gmask, tg, t)
        nw[gmask] = [0.0, 1.0, 0.0]
    any_ = hit | gmask
    lam = np.zeros((H, W), np.float32)
    lam[any_] = (nw[any_] @ Lw).astype(np.float32)
    P = cam.C + t[any_][:, None] * D[any_]
    N = nw[any_]
    O = P + N * 0.01
    shadow = np.zeros(len(O), bool)
    for pr in prims:
        c = pr["c"]
        rad = bound_radius(pr)
        oc = O - c
        b = oc @ Lw
        cc = (oc * oc).sum(1) - rad * rad
        cand = (b * b - cc >= 0) & ((b < 0) | (cc < 0)) & ~shadow
        if not cand.any():
            continue
        idx = np.nonzero(cand)[0]
        tt, _, _ = intersect(pr, O[idx], np.broadcast_to(Lw, (len(idx), 3)))
        shadow[idx[np.isfinite(tt) & (tt > 1e-3)]] = True
    cls = np.zeros((H, W), np.uint8)
    lam_a = lam[any_]
    v = np.where(lam_a <= thr, 2, np.where(shadow, 3, 1)).astype(np.uint8)
    cls[any_] = v
    return {"lam": lam, "cls": cls, "ground": gmask, "id": R["id"]}
