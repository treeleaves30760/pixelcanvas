"""Anime faces on a 3D head: brows, eyes (lash line, lower lash, crease, iris bands, pupil, catch-lights), nose,
mouth, tears and blush laid out on the face (head-frame offsets in units of the head height), lifted onto the head's
surface and projected through the camera, so they turn, tilt and foreshorten with the head. Pure functions.

``features(expression, iris, lash, gaze)`` -> specs ``{name, pts (N,2 head-frame), closed, weight, taper, fill,
kind}``; ``project(rig, cam, specs, depth)`` -> the same with ``px`` (canvas pixels) and the hidden points dropped
(bangs over an eye hide it; brows show through the hair, as anime draws them).
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

EXPRESSIONS = ("neutral", "smile", "open", "grin", "laugh", "shout", "angry", "sad", "crying", "teary", "surprised",
               "panicked", "smug", "embarrassed", "sleepy", "closed", "shut", "wink")
EXPR_ALIAS = {"calm": "neutral", "happy": "smile", "cry": "crying", "tears": "crying", "wail": "crying",
              "shock": "surprised", "shocked": "surprised", "wide": "surprised", "scared": "panicked",
              "panic": "panicked", "mad": "angry", "tired": "sleepy", "half": "sleepy", "half-closed": "sleepy",
              "default": "neutral", "laughing": "laugh", "yell": "shout", "shouting": "shout", "scream": "shout",
              "furious": "shout", "blush": "embarrassed", "shy": "embarrassed", "flustered": "embarrassed",
              "smirk": "smug", "teary-eyed": "teary", "tearful": "teary", "content": "closed", "eyes-closed": "closed",
              "squeeze": "shut", "><": "shut", "pained": "shut", "winking": "wink"}
GAZES = {"center": (0.0, 0.0), "centre": (0.0, 0.0), "viewer": (0.0, 0.0), "front": (0.0, 0.0),
         "left": (0.28, 0.0), "right": (-0.28, 0.0), "up": (0.0, 0.2), "down": (0.0, -0.18),
         "up-left": (0.24, 0.16), "up-right": (-0.24, 0.16), "down-left": (0.24, -0.15), "down-right": (-0.24, -0.15)}

EYE_C = (0.165, -0.065)
EYE_W, EYE_H = 0.215, 0.175


def expression_key(word: str) -> Optional[str]:
    w = str(word or "neutral").strip().lower()
    w = EXPR_ALIAS.get(w, w)
    return w if w in EXPRESSIONS else None


def gaze_key(word) -> Optional[tuple]:
    """``left`` / ``right`` (the picture's sides when the face looks at you), ``up``, ``down``, ``up-left`` ...,
    ``center``; or ``x,y`` in -1..1 (x toward the picture's left, y up). None when it does not parse."""
    w = str(word or "center").strip().lower()
    if w in GAZES:
        return GAZES[w]
    try:
        x, y = (float(v) for v in w.replace(" ", ",").split(",") if v)
    except ValueError:
        return None
    return (max(-1.0, min(1.0, x)) * 0.3, max(-1.0, min(1.0, y)) * 0.22)


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


def _tint(rgb, k):
    """k > 0 toward white, k < 0 toward black."""
    c = np.asarray(rgb, float)
    c = c + (255 - c) * k if k > 0 else c * (1 + k)
    return tuple(int(round(v)) for v in np.clip(c, 0, 255))


# per expression: (eye style, opening); brows (inner, outer lift in eye heights)
_EYE = {"neutral": ("open", 1.0), "smile": ("open", 0.88), "open": ("open", 1.0), "grin": ("happy", 0),
        "laugh": ("happy", 0), "shout": ("open", 0.92), "angry": ("open", 0.8), "sad": ("open", 0.86),
        "crying": ("shut", 0), "teary": ("open", 0.94), "surprised": ("open", 1.16), "panicked": ("open", 1.18),
        "smug": ("open", 0.62), "embarrassed": ("open", 0.9), "sleepy": ("open", 0.52), "closed": ("closed", 0),
        "shut": ("shut", 0), "wink": ("open", 0.94)}
_BROW = {"angry": (-0.34, 0.2), "shout": (-0.36, 0.22), "sad": (0.3, -0.16), "crying": (0.38, -0.18),
         "teary": (0.32, -0.14), "surprised": (0.38, 0.34), "panicked": (0.42, 0.06), "smug": (-0.08, 0.12),
         "embarrassed": (0.26, -0.06), "shut": (-0.2, 0.12), "laugh": (0.12, 0.1), "grin": (0.06, 0.06),
         "sleepy": (-0.06, -0.06)}
# the upper lid's drop over the iris (eye heights): sleepy and smug eyes are half covered
_LID = {"sleepy": 0.42, "smug": 0.3}


def _open_eye(out, e, s, sg, P, cx, cy, iris_rgb, lash_rgb, gx, gy):
    W, Hh = EYE_W, EYE_H
    _, opn = _EYE.get(e, ("open", 1.0))
    tilt_in = {"angry": -0.2, "shout": -0.2, "sad": 0.14, "teary": 0.12, "embarrassed": 0.06}.get(e, 0.0)
    top = 0.5 * opn
    out_y = 0.14 - (0.08 if e in ("sad", "teary") else 0.0) + (0.06 if e in ("angry", "shout") else 0.0)
    upper = _curve([P(-0.52, -0.02 + tilt_in), P(-0.25, top * 0.94), P(0.15, top), P(0.5, out_y + 0.04),
                    P(0.56, out_y - 0.02)])
    lo = 0.5 if opn >= 1 else 0.47
    bot = [P(0.52, out_y - 0.14), P(0.3, -0.84 * lo), P(-0.12, -lo), P(-0.48, -0.16)]
    lower = _curve(bot)
    out.append({"name": f"white-{s}", "pts": np.vstack([upper, lower]), "closed": True, "weight": 0.0,
                "taper": "none", "kind": "white", "fill": (252, 252, 255)})
    # the upper lash's shadow on the white
    sx, sy = P(0.0, top + 0.1)
    out.append({"name": f"lashshade-{s}", "pts": _ell(sx, sy, 0.62 * W, 0.34 * Hh, 28), "closed": True,
                "weight": 0.0, "taper": "none", "kind": "white", "fill": (198, 204, 228), "clip": f"white-{s}"})
    small = e in ("panicked", "surprised")
    ir = 0.31 if not small else 0.22
    iry = 0.58 if not small else 0.42
    icx = cx + gx * W                         # gaze x is in the picture frame, the same for both eyes
    icy = cy + (-0.06 + gy) * Hh
    out.append({"name": f"iris-{s}", "pts": _ell(icx, icy, ir * W, iry * Hh, 32), "closed": True,
                "weight": 0.0, "taper": "none", "kind": "iris", "fill": iris_rgb, "clip": f"white-{s}"})
    # a darker upper band under the lid
    band = np.vstack([_ell(icx, icy, ir * W, iry * Hh, 18, 0.0, math.pi),
                      _curve([[icx - ir * W, icy + 0.02 * Hh], [icx, icy - 0.12 * Hh],
                              [icx + ir * W, icy + 0.02 * Hh]], 10)])
    out.append({"name": f"irisdark-{s}", "pts": band, "closed": True, "weight": 0.0, "taper": "none",
                "kind": "iris", "fill": _tint(iris_rgb, -0.45), "clip": f"iris-{s}"})
    # a lighter lower crescent
    cres = np.vstack([_ell(icx, icy, 0.82 * ir * W, 0.86 * iry * Hh, 16, math.pi * 1.1, math.pi * 1.9),
                      _ell(icx, icy - 0.28 * iry * Hh, 0.6 * ir * W, 0.38 * iry * Hh, 16, math.pi * 1.95,
                           math.pi * 1.05)])
    out.append({"name": f"irislight-{s}", "pts": cres, "closed": True, "weight": 0.0, "taper": "none",
                "kind": "iris", "fill": _tint(iris_rgb, 0.5), "clip": f"iris-{s}"})
    pr = 0.42 if e != "panicked" else 0.26
    out.append({"name": f"pupil-{s}", "pts": _ell(icx, icy + 0.03 * Hh, ir * W * pr, iry * Hh * pr * 1.05, 24),
                "closed": True, "weight": 0.0, "taper": "none", "kind": "pupil", "clip": f"iris-{s}",
                "fill": _tint(iris_rgb, -0.75)})
    # catch-lights: a big one high on the light's side (the picture's left), a small one low on the other side
    for k, (dx, dy, r) in enumerate(((0.36, 0.3, 0.13), (-0.42, -0.45, 0.06))):
        hx, hy = icx + dx * ir * W, icy + dy * iry * Hh
        out.append({"name": f"shine-{s}{k}", "pts": _ell(hx, hy, r * W, r * W * 1.15, 16), "closed": True,
                    "weight": 0.0, "taper": "none", "kind": "shine", "fill": (255, 255, 255), "clip": f"white-{s}"})
    drop = _LID.get(e, 0.0)
    if drop:
        # a heavy upper lid: skin over the top of the eye, its edge is the lash line
        lid_top = _curve([P(-0.6, 0.15), P(-0.2, top + 0.3), P(0.3, top + 0.3), P(0.66, out_y + 0.2)])
        lid_low = _curve([P(0.56, out_y - 0.02), P(0.15, top - drop), P(-0.25, top * 0.94 - drop * 0.9),
                          P(-0.52, -0.02 + tilt_in)])
        out.append({"name": f"eyelid-{s}", "pts": np.vstack([lid_top, lid_low]), "closed": True, "weight": 0.0,
                    "taper": "none", "kind": "skin", "fill": None, "clip": f"white-{s}"})
        upper = _curve([P(-0.52, -0.02 + tilt_in), P(-0.25, top * 0.94 - drop * 0.9), P(0.15, top - drop),
                        P(0.5, out_y + 0.02), P(0.56, out_y - 0.02)])
    if e == "teary":
        # a tear pooled on the lower lid: a wet line along it and a third catch-light
        out.append({"name": f"tearpool-{s}", "pts": _curve([P(-0.3, -0.6), P(0.1, -0.68), P(0.42, -0.48)]),
                    "closed": False, "weight": 0.9, "taper": "both", "kind": "tear", "rgb": (110, 165, 220)})
        out.append({"name": f"shine-{s}2", "pts": _ell(icx - 0.1 * ir * W, icy - 0.5 * iry * Hh, 0.1 * W, 0.04 * W,
                                                      12),
                    "closed": True, "weight": 0.0, "taper": "none", "kind": "shine", "fill": (235, 245, 255),
                    "clip": f"white-{s}"})
    # the upper lash line: heavy, thin at the inner corner, a flick past the outer corner
    lash = np.vstack([upper, _curve([P(0.56, out_y - 0.02), P(0.66, out_y + 0.1)], 4)[1:]])
    out.append({"name": f"lash-{s}", "pts": lash, "closed": False, "weight": 2.4, "taper": "both", "kind": "lash",
                "rgb": lash_rgb, "wprof": [[0, 0.35], [0.3, 0.85], [0.72, 1.3], [1, 0.6]]})
    # the lower lash: a short thin hint at the outer side
    out.append({"name": f"lower-{s}", "pts": _curve([P(0.5, out_y - 0.17), P(0.28, -0.84 * lo - 0.02),
                                                      P(0.0, -lo - 0.02)]),
                "closed": False, "weight": 0.8, "taper": "both", "kind": "lid", "rgb": lash_rgb})
    # the lid crease above
    out.append({"name": f"lid-{s}", "pts": _curve([P(-0.26, top + 0.2), P(0.15, top + 0.3), P(0.5, out_y + 0.25)]),
                "closed": False, "weight": 0.55, "taper": "both", "kind": "lid", "rgb": lash_rgb})


def features(expression: str, iris_rgb=(70, 110, 190), lash_rgb=(40, 28, 34), gaze=(0.0, 0.0)) -> List[dict]:
    """Head-frame specs of the face. Each open eye: lid crease, white with the lash's shadow, iris (a darker upper
    band under the lid, a lighter lower crescent), pupil, two catch-lights, the upper lash line (heavy, tapered, a
    flick at the outer corner) and a lower lash hint. ``gaze`` moves the irises (head units, x toward the picture's
    left when she faces you, y up)."""
    e = expression_key(expression) or "neutral"
    out: List[dict] = []
    W, Hh = EYE_W, EYE_H
    gx, gy = (float(gaze[0]), float(gaze[1])) if gaze is not None else (0.0, 0.0)
    if e == "embarrassed" and gx == 0 and gy == 0:
        gx, gy = 0.2, -0.12                                # her eyes slide away
    iris_rgb = tuple(int(v) for v in iris_rgb)
    for s, sg in (("r", 1.0), ("l", -1.0)):
        cx, cy = sg * EYE_C[0], EYE_C[1]

        def P(u, v, cx=cx, cy=cy, sg=sg):          # u: toward the outer corner, v: up (eye units)
            return [cx + sg * u * W, cy + v * Hh]
        st = _EYE.get(e, ("open", 1.0))[0]
        if e == "wink" and s == "l":
            st = "happy"
        # brows: a tapered stroke, fuller at the inner end
        bi, bo = _BROW.get(e, (0.0, 0.04))
        inner_u = -0.42 if e not in ("crying", "angry", "shout") else -0.5
        brow = _curve([P(inner_u, 1.0 + bi), P(0.05, 1.16 + 0.5 * (bi + bo)), P(0.58, 0.96 + bo)])
        out.append({"name": f"brow-{s}", "pts": brow, "closed": False, "weight": 1.15, "taper": "both",
                    "kind": "brow", "rgb": lash_rgb, "wprof": [[0, 0.75], [0.3, 1.15], [1, 0.55]]})
        if st == "happy":
            # a closed laughing eye: an arch
            arc = _curve([P(-0.5, -0.12), P(-0.05, 0.36), P(0.5, -0.06)])
            out.append({"name": f"eye-{s}", "pts": arc, "closed": False, "weight": 2.1, "taper": "both",
                        "kind": "lash", "rgb": lash_rgb, "wprof": [[0, 0.5], [0.5, 1.1], [1, 0.8]]})
            continue
        if st == "closed":
            # a closed calm eye: the lash curve bowed down, with a short flick
            arc = _curve([P(-0.5, 0.02), P(0.0, -0.22), P(0.52, -0.02)])
            out.append({"name": f"eye-{s}", "pts": arc, "closed": False, "weight": 2.1, "taper": "both",
                        "kind": "lash", "rgb": lash_rgb, "wprof": [[0, 0.5], [0.6, 1.1], [1, 0.9]]})
            out.append({"name": f"flick-{s}", "pts": _curve([P(0.48, -0.04), P(0.64, -0.18)], 4), "closed": False,
                        "weight": 1.1, "taper": "end", "kind": "lash", "rgb": lash_rgb})
            continue
        if st == "shut":
            # squeezed shut: a ">" for her right eye, "<" for her left, the point toward the nose
            vee = np.array([P(0.48, 0.34), P(-0.38, -0.02), P(0.48, -0.3)])
            out.append({"name": f"eye-{s}", "pts": _curve(vee[:2], 6).tolist() + _curve(vee[1:], 6)[1:].tolist(),
                        "closed": False, "weight": 2.2, "taper": "both", "kind": "lash", "rgb": lash_rgb})
            if e == "crying":
                # tears: a stream from the eye down the cheek, and drops thrown off sideways
                tear = np.vstack([_curve([P(0.0, -0.3), P(0.1, -0.75), P(0.04, -1.25), P(0.14, -1.7)]),
                                  _curve([P(0.36, -1.7), P(0.3, -1.25), P(0.36, -0.75), P(0.26, -0.3)])])
                out.append({"name": f"tear-{s}", "pts": tear, "closed": True, "weight": 0.7, "taper": "none",
                            "kind": "tear", "fill": (190, 228, 255), "rgb": (84, 140, 196)})
                for k, (x, y, r) in enumerate(((0.55, -0.02, 0.028), (0.68, -0.14, 0.02))):
                    cx_, cy_ = sg * x, y
                    drop = np.vstack([_ell(cx_, cy_ - 0.2 * r, r, r, 14, math.pi, 2 * math.pi),
                                      [[cx_, cy_ + 1.6 * r]]])
                    out.append({"name": f"drop-{s}{k}", "pts": drop, "closed": True, "weight": 0.6, "taper": "none",
                                "kind": "tear", "fill": (190, 228, 255), "float": True, "rgb": (84, 140, 196)})
            continue
        _open_eye(out, e, s, sg, P, cx, cy, iris_rgb, lash_rgb, gx, gy)
    # nose: a small mark on the shadow side
    out.append({"name": "nose", "pts": np.array([[0.014, -0.17], [0.018, -0.2], [0.002, -0.214]]), "closed": False,
                "weight": 0.85, "taper": "both", "kind": "nose"})
    _mouth(out, e)
    if e in ("crying", "smile", "grin", "laugh", "panicked", "surprised", "embarrassed", "teary", "wink", "shut"):
        strong = e in ("embarrassed", "crying")
        for s, sg in (("r", 1.0), ("l", -1.0)):
            out.append({"name": f"blush-{s}", "pts": _ell(sg * 0.2, -0.215, 0.085 if strong else 0.075,
                                                          0.038 if strong else 0.032, 24),
                        "closed": True, "weight": 0.0, "taper": "none", "kind": "blush", "fill": (240, 130, 140),
                        "alpha": 0.55 if strong else 0.38})
            if e == "embarrassed":
                for k in range(3):
                    x0 = sg * (0.15 + 0.04 * k)
                    out.append({"name": f"hatch-{s}{k}", "pts": np.array([[x0 + 0.012, -0.196],
                                                                          [x0 - 0.012, -0.234]]),
                                "closed": False, "weight": 0.7, "taper": "both", "kind": "hatch",
                                "rgb": (206, 92, 108)})
    if e == "panicked":
        out.append({"name": "sweat", "pts": _curve([[0.5, 0.3], [0.47, 0.15], [0.5, 0.1], [0.53, 0.15], [0.5, 0.3]]),
                    "closed": True, "weight": 0.7, "taper": "none", "kind": "tear", "fill": (190, 228, 255),
                    "float": True})
    return out


def _mouth(out: List[dict], e: str) -> None:
    mx, my = 0.0, -0.33
    mouth_fill = (150, 50, 62)

    def line(pts, w=1.0):
        out.append({"name": "mouth", "pts": _curve(pts), "closed": False, "weight": w, "taper": "both",
                    "kind": "mouth"})
    if e in ("neutral", "sleepy", "closed"):
        line([[-0.045, my], [0.0, my - 0.006], [0.045, my]])
        return
    if e in ("smile", "wink"):
        line([[-0.065, my + 0.016], [0.0, my - 0.016], [0.065, my + 0.016]])
        return
    if e == "teary":
        line([[-0.05, my - 0.004], [-0.02, my + 0.006], [0.02, my - 0.004], [0.05, my + 0.004]], 0.9)
        return
    if e == "sad":
        line([[-0.05, my - 0.014], [0.0, my + 0.008], [0.05, my - 0.014]])
        return
    if e == "smug":
        # a one-sided smirk: up at the picture's right corner
        line([[-0.06, my + 0.002], [-0.01, my - 0.008], [0.03, my - 0.002], [0.065, my + 0.022]])
        return
    if e == "embarrassed":
        xs = np.linspace(-0.045, 0.045, 7)
        line(np.stack([xs, my + 0.006 * np.cos(np.arange(7) * math.pi)], 1), 0.9)
        return
    if e == "open":
        loop = np.vstack([_curve([[-0.055, my + 0.008], [0.0, my - 0.002], [0.055, my + 0.008]], 8),
                          _curve([[0.055, my + 0.008], [0.03, my - 0.045], [-0.03, my - 0.045],
                                  [-0.055, my + 0.008]])])
    elif e == "surprised":
        loop = _ell(mx, my - 0.012, 0.028, 0.04, 24)
    elif e in ("grin", "laugh"):
        wd, dp = (0.085, 0.07) if e == "grin" else (0.095, 0.1)
        loop = np.vstack([_curve([[-wd, my + 0.012], [0.0, my + 0.002], [wd, my + 0.012]], 8),
                          _curve([[wd, my + 0.012], [0.045, my - dp], [-0.045, my - dp], [-wd, my + 0.012]])])
    elif e == "angry":
        loop = np.vstack([_curve([[-0.06, my - 0.005], [0.0, my + 0.008], [0.06, my - 0.005]]),
                          _curve([[0.06, my - 0.005], [0.03, my - 0.035], [-0.03, my - 0.035], [-0.06, my - 0.005]])])
    elif e == "shout":
        loop = np.vstack([_curve([[-0.08, my + 0.006], [0.0, my + 0.02], [0.08, my + 0.006]], 8),
                          _curve([[0.08, my + 0.006], [0.06, my - 0.1], [-0.06, my - 0.1], [-0.08, my + 0.006]])])
    elif e == "panicked":
        xs = np.linspace(-0.07, 0.07, 9)
        top = np.stack([xs, my + 0.01 + 0.012 * (np.arange(9) % 2)], 1)
        loop = np.vstack([top, _curve([[0.07, my + 0.01], [0.0, my - 0.07], [-0.07, my + 0.01]])])
    else:   # crying: a wide wailing mouth
        loop = np.vstack([_curve([[-0.1, my + 0.02], [0.0, my + 0.005], [0.1, my + 0.02]]),
                          _curve([[0.1, my + 0.02], [0.06, my - 0.1], [-0.06, my - 0.1], [-0.1, my + 0.02]])])
    out.append({"name": "mouth", "pts": loop, "closed": True, "weight": 1.0, "taper": "none", "kind": "mouth",
                "fill": mouth_fill})
    if e in ("shout", "laugh"):
        ty = float(loop[:, 1].max())
        out.append({"name": "teeth", "pts": np.array([[-0.09, ty + 0.01], [0.09, ty + 0.01], [0.09, ty - 0.018],
                                                      [-0.09, ty - 0.018]]),
                    "closed": True, "weight": 0.0, "taper": "none", "kind": "teeth", "fill": (250, 250, 252),
                    "clip": "mouth"})
    if e in ("grin", "laugh", "crying", "panicked", "shout", "open"):
        lo_ = float(loop[:, 1].min())
        out.append({"name": "tongue", "pts": _ell(0.0, lo_ + 0.008, 0.05 if e != "panicked" else 0.035,
                                                  0.03 if e != "open" else 0.02, 20),
                    "closed": True, "weight": 0.0, "taper": "none", "kind": "tongue", "fill": (226, 120, 132),
                    "clip": "mouth"})


def project(rig, cam, specs: List[dict], depth: Optional[np.ndarray] = None, depth_scale: float = 1.0,
            tol: float = 0.012) -> List[dict]:
    """Specs with ``px`` (canvas px of the camera) and ``vis`` (per point: not hidden by something in front of the
    face, per the z-buffer ``depth`` at ``depth_scale`` render px per camera px)."""
    from .figure3d import face_surface
    out = []
    for sp in specs:
        P = np.asarray(sp["pts"], float)
        if sp.get("float"):
            # in the air beside the head, in its front plane
            Wp = rig.J["head"] + (P[:, 0:1] * rig.h) * rig.Hd[:, 0] + (P[:, 1:2] * rig.h) * rig.Hd[:, 1] + \
                0.3 * rig.h * rig.Hd[:, 2]
            px, z = cam.project(Wp)
            q = dict(sp)
            q["px"], q["vis"] = px, np.ones(len(P), bool)
            out.append(q)
            continue
        Wp = face_surface(rig, P)
        missed = face_surface.missed.copy()
        px, z = cam.project(Wp)
        vis = ~missed
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
