"""Anime faces on a 3D head: brows, eyes (lash line, lower lid, iris, pupil, catch-lights), nose, mouth, tears and
blush laid out on the face (head-frame offsets in units of the head height), lifted onto the head's surface and
projected through the camera, so they turn, tilt and foreshorten with the head. Pure functions.

``features(rig, expression)`` -> specs ``{name, pts (N,2 head-frame), closed, weight, taper, fill, kind}``;
``project(rig, cam, specs, depth)`` -> the same with ``px`` (canvas pixels) and the hidden points dropped (bangs over
an eye hide it; brows show through the hair, as anime draws them).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

EXPRESSIONS = ("neutral", "smile", "open", "grin", "angry", "sad", "crying", "surprised", "panicked", "sleepy")
EXPR_ALIAS = {"calm": "neutral", "happy": "smile", "laugh": "grin", "cry": "crying", "tears": "crying",
              "wail": "crying", "shock": "surprised", "shocked": "surprised", "scared": "panicked", "panic": "panicked",
              "mad": "angry", "tired": "sleepy", "default": "neutral"}

EYE_C = (0.165, -0.06)
EYE_W, EYE_H = 0.2, 0.15


def expression_key(word: str) -> Optional[str]:
    w = str(word or "neutral").strip().lower()
    w = EXPR_ALIAS.get(w, w)
    return w if w in EXPRESSIONS else None


def _ell(cx, cy, rx, ry, n=28, a0=0.0, a1=2 * math.pi):
    t = np.linspace(a0, a1, n)
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1)


def _curve(pts, n=24):
    """A smooth curve through a few points (quadratic B-spline-ish via Chaikin)."""
    P = np.asarray(pts, float)
    for _ in range(3):
        if len(P) < 3:
            break
        Q = [P[0]]
        for a, b in zip(P[:-1], P[1:]):
            Q += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        Q.append(P[-1])
        P = np.array(Q)
    return P


def features(expression: str, iris_rgb=(70, 110, 190), lash_rgb=(40, 28, 34)) -> List[dict]:
    e = expression_key(expression) or "neutral"
    out: List[dict] = []
    W, Hh = EYE_W, EYE_H
    opn = {"smile": 0.82, "sad": 0.78, "angry": 0.72, "surprised": 1.15, "panicked": 1.2, "sleepy": 0.42}.get(e, 1.0)
    for s, sg in (("r", 1.0), ("l", -1.0)):
        cx, cy = sg * EYE_C[0], EYE_C[1]

        def P(u, v):          # u: toward the outer corner, v: up (eye units)
            return [cx + sg * u * W, cy + v * Hh]
        # brows
        bi = {"angry": -0.32, "sad": 0.3, "crying": 0.38, "surprised": 0.35, "panicked": 0.42}.get(e, 0.0)
        bo = {"angry": 0.18, "sad": -0.15, "crying": -0.18, "surprised": 0.32, "panicked": 0.05}.get(e, 0.04)
        inner_u = -0.42 if e not in ("crying", "angry") else -0.5
        brow = _curve([P(inner_u, 1.02 + bi), P(0.05, 1.18 + 0.5 * (bi + bo)), P(0.55, 0.98 + bo)])
        out.append({"name": f"brow-{s}", "pts": brow, "closed": False, "weight": 1.2, "taper": "both", "kind": "brow",
                    "rgb": lash_rgb})
        if e in ("grin",):
            arc = _curve([P(-0.48, -0.05), P(0.0, 0.32), P(0.48, -0.05)])
            out.append({"name": f"eye-{s}", "pts": arc, "closed": False, "weight": 2.4, "taper": "both",
                        "kind": "lash", "rgb": lash_rgb})
            continue
        if e == "crying":
            # squeezed shut: a ">" for her right eye, "<" for her left, the point toward the nose
            vee = np.array([P(0.48, 0.32), P(-0.38, -0.02), P(0.48, -0.3)])
            out.append({"name": f"eye-{s}", "pts": _curve(vee[:2], 6).tolist() + _curve(vee[1:], 6)[1:].tolist(),
                        "closed": False, "weight": 2.4, "taper": "both", "kind": "lash", "rgb": lash_rgb})
            # tears: a stream from the eye down the cheek, and drops thrown off sideways
            tear = np.vstack([_curve([P(-0.05, -0.35), P(0.08, -0.9), P(0.0, -1.7), P(0.12, -2.4)]),
                              _curve([P(0.42, -2.4), P(0.36, -1.7), P(0.42, -0.9), P(0.3, -0.35)])])
            out.append({"name": f"tear-{s}", "pts": tear, "closed": True, "weight": 0.7, "taper": "none",
                        "kind": "tear", "fill": (190, 228, 255)})
            for k, (u, v, r) in enumerate(((1.05, 0.2, 0.09), (1.4, -0.25, 0.07))):
                out.append({"name": f"drop-{s}{k}", "pts": _ell(*P(u, v), r * W, r * W * 1.3, 16), "closed": True,
                            "weight": 0.6, "taper": "none", "kind": "tear", "fill": (190, 228, 255)})
            continue
        if e == "sleepy" or opn < 0.5:
            out.append({"name": f"eye-{s}", "pts": _curve([P(-0.5, -0.05), P(0.0, -0.2), P(0.5, -0.02)]),
                        "closed": False, "weight": 2.4, "taper": "both", "kind": "lash", "rgb": lash_rgb})
            out.append({"name": f"lid-{s}", "pts": _curve([P(-0.35, -0.38), P(0.3, -0.4)]), "closed": False,
                        "weight": 0.6, "taper": "both", "kind": "lid", "rgb": lash_rgb})
            continue
        top = 0.5 * opn
        tilt_in = {"angry": -0.18, "sad": 0.12}.get(e, 0.0)
        upper = _curve([P(-0.5, -0.05 + tilt_in), P(-0.2, top), P(0.25, top * 0.95), P(0.52, 0.12)])
        lower = _curve([P(0.48, 0.0), P(0.3, -0.45 * opn), P(-0.15, -0.5 * opn), P(-0.5, -0.12)])
        loop = np.vstack([upper, lower])
        out.append({"name": f"white-{s}", "pts": loop, "closed": True, "weight": 0.0, "taper": "none",
                    "kind": "white", "fill": (252, 252, 255)})
        ir = 0.3 if e not in ("panicked", "surprised") else 0.22
        icx, icy = P(0.0, -0.02)
        out.append({"name": f"iris-{s}", "pts": _ell(icx, icy, ir * W, 0.48 * Hh * opn, 32), "closed": True,
                    "weight": 0.0, "taper": "none", "kind": "iris", "fill": tuple(int(v) for v in iris_rgb),
                    "clip": f"white-{s}"})
        pr = 0.45 if e != "panicked" else 0.3
        out.append({"name": f"pupil-{s}", "pts": _ell(icx, icy - 0.04 * Hh, ir * W * pr, 0.48 * Hh * opn * pr, 24),
                    "closed": True, "weight": 0.0, "taper": "none", "kind": "pupil", "clip": f"white-{s}",
                    "fill": tuple(int(v * 0.35) for v in iris_rgb)})
        for k, (u, v, r) in enumerate(((-0.12, 0.18, 0.11), (0.12, -0.25, 0.05))):
            hx, hy = P(u, v)
            out.append({"name": f"shine-{s}{k}", "pts": _ell(hx, hy, r * W, r * W, 16), "closed": True,
                        "weight": 0.0, "taper": "none", "kind": "shine", "fill": (255, 255, 255), "clip": f"white-{s}"})
        out.append({"name": f"lash-{s}", "pts": upper, "closed": False, "weight": 2.6, "taper": "both",
                    "kind": "lash", "rgb": lash_rgb})
        out.append({"name": f"lid-{s}", "pts": _curve([P(-0.1, -0.5 * opn), P(0.35, -0.42 * opn)]), "closed": False,
                    "weight": 0.6, "taper": "both", "kind": "lid", "rgb": lash_rgb})
    # nose: a small mark
    out.append({"name": "nose", "pts": np.array([[0.008, -0.2], [0.0, -0.225]]), "closed": False, "weight": 0.8,
                "taper": "both", "kind": "nose"})
    mx, my = 0.0, -0.33
    mouth_fill = (150, 50, 62)
    if e == "neutral":
        out.append({"name": "mouth", "pts": _curve([[-0.05, my], [0.0, my - 0.006], [0.05, my]]), "closed": False,
                    "weight": 1.0, "taper": "both", "kind": "mouth"})
    elif e in ("smile", "sleepy"):
        out.append({"name": "mouth", "pts": _curve([[-0.065, my + 0.018], [0.0, my - 0.015], [0.065, my + 0.018]]),
                    "closed": False, "weight": 1.0, "taper": "both", "kind": "mouth"})
    elif e == "sad":
        out.append({"name": "mouth", "pts": _curve([[-0.05, my - 0.012], [0.0, my + 0.008], [0.05, my - 0.012]]),
                    "closed": False, "weight": 1.0, "taper": "both", "kind": "mouth"})
    else:
        if e == "open":
            loop = _ell(mx, my - 0.01, 0.045, 0.03, 24)
        elif e == "surprised":
            loop = _ell(mx, my - 0.01, 0.03, 0.04, 24)
        elif e == "grin":
            loop = np.vstack([_curve([[-0.085, my + 0.01], [0.085, my + 0.01]], 6),
                              _curve([[0.085, my + 0.01], [0.04, my - 0.07], [-0.04, my - 0.07], [-0.085, my + 0.01]])])
        elif e == "angry":
            loop = np.vstack([_curve([[-0.06, my - 0.005], [0.0, my + 0.008], [0.06, my - 0.005]]),
                              _curve([[0.06, my - 0.005], [0.03, my - 0.035], [-0.03, my - 0.035], [-0.06, my - 0.005]])])
        elif e == "panicked":
            xs = np.linspace(-0.07, 0.07, 9)
            top = np.stack([xs, my + 0.01 + 0.012 * (np.arange(9) % 2)], 1)
            loop = np.vstack([top, _curve([[0.07, my + 0.01], [0.0, my - 0.07], [-0.07, my + 0.01]])])
        else:   # crying: a wide wailing mouth
            loop = np.vstack([_curve([[-0.1, my + 0.02], [0.0, my + 0.005], [0.1, my + 0.02]]),
                              _curve([[0.1, my + 0.02], [0.06, my - 0.1], [-0.06, my - 0.1], [-0.1, my + 0.02]])])
        out.append({"name": "mouth", "pts": loop, "closed": True, "weight": 1.0, "taper": "none", "kind": "mouth",
                    "fill": mouth_fill})
        if e in ("grin", "crying", "panicked"):
            lo = float(loop[:, 1].min())
            out.append({"name": "tongue", "pts": _ell(0.0, lo + 0.022, 0.05 if e != "panicked" else 0.035, 0.025, 20),
                        "closed": True, "weight": 0.0, "taper": "none", "kind": "tongue", "fill": (226, 120, 132),
                        "clip": "mouth"})
    if e in ("crying", "smile", "grin", "panicked", "surprised"):
        for s, sg in (("r", 1.0), ("l", -1.0)):
            out.append({"name": f"blush-{s}", "pts": _ell(sg * 0.2, -0.22, 0.075, 0.035, 24), "closed": True,
                        "weight": 0.0, "taper": "none", "kind": "blush", "fill": (240, 140, 150), "alpha": 0.4})
    if e == "panicked":
        out.append({"name": "sweat", "pts": np.vstack([_curve([[0.36, 0.25], [0.33, 0.1], [0.36, 0.05], [0.39, 0.1],
                                                               [0.36, 0.25]])]),
                    "closed": True, "weight": 0.7, "taper": "none", "kind": "tear", "fill": (190, 228, 255)})
    return out


def project(rig, cam, specs: List[dict], depth: Optional[np.ndarray] = None, depth_scale: float = 1.0,
            tol: float = 0.012) -> List[dict]:
    """Specs with ``px`` (canvas px of the camera) and ``vis`` (per point: not hidden by something in front of the
    face, per the z-buffer ``depth`` at ``depth_scale`` render px per camera px)."""
    from .figure3d import face_surface
    out = []
    for sp in specs:
        P = np.asarray(sp["pts"], float)
        Wp = face_surface(rig, P)
        px, z = cam.project(Wp)
        vis = np.ones(len(P), bool)
        if depth is not None and sp["kind"] not in ("brow",):
            H, W = depth.shape
            xi = np.clip((px[:, 0] * depth_scale).astype(int), 0, W - 1)
            yi = np.clip((px[:, 1] * depth_scale).astype(int), 0, H - 1)
            vis = z <= depth[yi, xi] + tol
        # the far side of a turned head: a feature on a surface facing away is hidden
        nrm = Wp - rig.cr_c
        toward = cam.C - Wp
        vis &= (nrm * toward).sum(1) > -0.002
        q = dict(sp)
        q["px"], q["vis"] = px, vis
        out.append(q)
    return out
