# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import numpy as np

from ..color import F32
from ..solid import (Material)
from .tables import EXPRESSIONS
from .shapes import _Decal, _arc_pts, _bez, _e2, _mix, _seg2, _shade, _stroke2


# ===================================================================================================== face

_MOUTH_COL, _TONGUE_COL, _TEETH_COL = "#5c2420", "#e66f78", "#fbf6ee"


def _face_layout(build_cfg, R):
    """Eye / mouth / nose placement in head-local metres for a build (eyes set low and wide on toy builds)."""
    e = build_cfg.get("eye", 1.0)
    heads = build_cfg["heads"]
    t = float(np.clip((heads - 3.0) / 3.0, 0, 1))          # 0 = toy proportions, 1 = heroic
    return dict(ex=(0.37 - 0.06 * t) * R, ey=(-0.11 + 0.1 * t) * R, erx=(0.125 - 0.03 * t) * R * e,
                ery=(0.155 - 0.06 * t) * R * e, my=(-0.5 - 0.06 * t) * R, ny=(-0.3 + 0.02 * t) * R,
                mw=(1.0 - 0.1 * t))


def _face_mats(skin, eye_color, brow_color, style):
    gl = dict(roughness=0.22, specular=0.8, shadow=False)
    bead = style in ("bead", "dot")
    ns = dict(shadow=False)
    return {
        "sclera": Material(color="#fbf8f2", roughness=0.3, specular=0.5, subsurface=0.2, **ns),
        "iris": Material(color=("#2a1c16" if bead else eye_color), **gl),
        "pupil": Material(color="#140d0b", **gl),
        "shine": Material(color="#ffffff", roughness=0.4, specular=0.2, emissive=0.35, shadow=False),
        "lid": Material(color=_shade(skin, 0.93), roughness=0.7, specular=0.2, subsurface=0.4, **ns),
        "lash": Material(color="#2a1a16", roughness=0.5, specular=0.3, **ns),
        "brow": Material(color=brow_color or "#3a2a22", roughness=0.75, specular=0.2, **ns),
        "mouth": Material(color=_MOUTH_COL, roughness=0.5, specular=0.3, **ns),
        "tongue": Material(color=_TONGUE_COL, roughness=0.45, specular=0.4, subsurface=0.3, **ns),
        "teeth": Material(color=_TEETH_COL, roughness=0.3, specular=0.5, **ns),
    }


def _union2(fs):
    def f(x, y):
        d = fs[0](x, y)
        for g in fs[1:]:
            d = np.minimum(d, g(x, y))
        return d
    return f


def _face(head, lay, expr, skin, eye_color, brow_color, style="round", gaze=(0.0, 0.0), freckles=False,
          blush=True, detail="normal", mouth=None):
    """Painted face decals for a head: returns {part name: (head-local shape, Material)}."""
    R = head.R
    ex, ey, erx, ery, my = lay["ex"], lay["ey"], lay["erx"], lay["ery"], lay["my"]
    E = dict(EXPRESSIONS.get(expr, EXPRESSIONS["neutral"])) if isinstance(expr, str) else dict(expr)
    gx, gy = gaze
    if "gaze" in E and tuple(gaze) == (0.0, 0.0):
        gx, gy = E["gaze"]
    zmin = 0.15 * R
    out = {}
    mode = E.get("eyes", "open")
    modes = {1: mode, -1: mode}
    if mode == "wink":
        modes = {1: "open", -1: "happy"}
    if mode in ("half", "wide"):
        modes = {1: "open", -1: "open"}
    if style == "dot":
        erx, ery = erx * 0.6, ery * 0.62
    wide = 1.12 if mode == "wide" else 1.0
    lid = E.get("lid")
    low = E.get("low")
    bead = style in ("bead", "dot")
    mats = _face_mats(skin, eye_color, brow_color, style)

    def ell(s, k=1.0):
        return lambda x, y: _e2(x, y, s * ex, ey, erx * wide * k, ery * wide * k)

    def lid_line(s, x, y):
        h, tilt = lid
        y0 = ey + ery * wide * (2 * h - 1)
        k = math.tan(math.radians(tilt))
        return (y - (F32(y0) + F32(k * s) * (x - F32(s * ex)))) * F32(1 / math.sqrt(1 + k * k))

    def opening(s):
        base = ell(s)
        if lid is None and low is None:
            return base

        def f(x, y):
            d = base(x, y)
            if lid is not None:
                d = np.maximum(d, lid_line(s, x, y))
            if low is not None:
                d = np.maximum(d, F32(ey - ery * (1 - 2 * (1 - low)) * 0.0 - ery * low) - y)
            return d
        return f

    open_sides = [s for s in (1, -1) if modes[s] == "open"]
    shut_sides = [s for s in (1, -1) if modes[s] in ("happy", "closed")]
    bulge = (0.07 if bead else 0.045) * R

    def eye_relief(x, y):
        d = np.minimum(ell(1)(x, y), ell(-1)(x, y))
        x = np.clip(-d / F32(ery * 0.9), 0, 1)
        return F32(0.006 * R) + F32(bulge) * x * (2 - x)

    def box_for(sides, padx, pady, cy=ey):
        xs = [s * ex for s in sides]
        return ((min(xs) - padx, cy - pady), (max(xs) + padx, cy + pady))

    if open_sides:
        D_eye = _union2([opening(s) for s in open_sides])
        ir = erx * wide * (0.98 if bead else (0.6 if mode == "wide" else 0.7))
        icx = {s: s * ex + gx * erx * (0.04 if bead else 0.34) for s in open_sides}
        icy = ey + gy * ery * (0.04 if bead else 0.32) - (0.0 if bead else 0.05 * ery)
        bx = box_for(open_sides, erx * 1.3, ery * 1.3)
        if not bead:
            out["eyes"] = (_Decal(head, D_eye, bx, eye_relief, zmin), mats["sclera"])
        ird = _union2([(lambda s: lambda x, y: _e2(x, y, icx[s], icy, ir, ir * (1.12 if bead else 1.08)))(s)
                       for s in open_sides])
        out["irises"] = (_Decal(head, lambda x, y: np.maximum(ird(x, y), D_eye(x, y)), bx,
                                lambda x, y: eye_relief(x, y) + F32(0.004 * R), zmin), mats["iris"])
        pr = ir * (0.6 if bead else 0.5)
        pdd = _union2([(lambda s: lambda x, y: _e2(x, y, icx[s] + gx * ir * 0.12, icy + gy * ir * 0.12 - 0.05 * ir,
                                                  pr, pr * 1.08))(s) for s in open_sides])
        out["pupils"] = (_Decal(head, lambda x, y: np.maximum(pdd(x, y), D_eye(x, y)), bx,
                                lambda x, y: eye_relief(x, y) + F32(0.008 * R), zmin), mats["pupil"])
        hr = ir * (0.34 if bead else 0.32)
        hl = _union2([(lambda s: lambda x, y: np.minimum(
            _e2(x, y, icx[s] - 0.34 * ir, icy + 0.38 * ir, hr, hr * 1.1),
            _e2(x, y, icx[s] + 0.32 * ir, icy - 0.36 * ir, hr * 0.42, hr * 0.42)))(s) for s in open_sides])
        out["eye shine"] = (_Decal(head, lambda x, y: np.maximum(hl(x, y), D_eye(x, y)), bx,
                                   lambda x, y: eye_relief(x, y) + F32(0.012 * R), zmin), mats["shine"])
        if lid is not None:
            def lid_d(x, y, _s=tuple(open_sides)):
                d = None
                for s in _s:
                    di = np.maximum(_e2(x, y, s * ex, ey + 0.04 * ery, erx * wide * 1.12, ery * wide * 1.16),
                                    -lid_line(s, x, y))
                    d = di if d is None else np.minimum(d, di)
                return d

            def lash_d(x, y, _s=tuple(open_sides)):
                d = None
                for s in _s:
                    di = np.maximum(np.abs(lid_line(s, x, y)) - F32(0.028 * R),
                                    _e2(x, y, s * ex, ey, erx * wide * 1.16, ery * wide * 1.1))
                    d = di if d is None else np.minimum(d, di)
                return d
            lb = box_for(open_sides, erx * 1.4, ery * 1.5)
            out["lids"] = (_Decal(head, lid_d, lb, lambda x, y: eye_relief(x, y) + F32(0.016 * R), zmin),
                           mats["lid"])
            out["lashes"] = (_Decal(head, lash_d, lb, lambda x, y: eye_relief(x, y) + F32(0.02 * R), zmin),
                             mats["lash"])
    if shut_sides:
        strokes = []
        for s in shut_sides:
            if modes[s] == "happy":
                strokes.append(_arc_pts(s * ex, ey - ery * 0.5, erx * 1.05, 22, 158, 9, ery * 1.0))
            else:
                strokes.append(_arc_pts(s * ex, ey + ery * 0.35, erx * 1.0, 202, 338, 9, ery * 0.72))
        w = 0.034 * R * (1.25 if bead else 1.0)
        dfn = (lambda st: lambda x, y: _union2([(lambda p: lambda a, b: _stroke2(a, b, p, w))(p) for p in st])(x, y))(
            strokes)
        out["closed eyes"] = (_Decal(head, dfn, box_for(shut_sides, erx * 1.4, ery * 1.5), F32(0.012 * R), zmin),
                              mats["lash"])
    # ---- brows
    if brow_color is not None:
        braise, btilt, barch = E.get("brows", (0, 0, 0.25))
        by = ey + ery * wide + 0.15 * R + braise * R
        bw = erx * 1.05
        bpts = []
        for s in (1, -1):
            inner = (s * (ex - bw), by + btilt * R)
            outer = (s * (ex + bw), by - 0.02 * R)
            mid = (s * ex, by + 0.06 * R * barch + btilt * R * 0.4)
            bpts.append(_bez(inner, mid, outer, 7))
        ws = list(np.linspace(0.034, 0.022, 7) * R)
        D_br = (lambda bp: lambda x, y: np.minimum(_stroke2(x, y, bp[0], ws), _stroke2(x, y, bp[1], ws)))(bpts)
        out["brows"] = (_Decal(head, D_br, ((-ex - bw - 0.06 * R, by - 0.16 * R), (ex + bw + 0.06 * R, by + 0.16 * R)),
                               F32(0.022 * R), zmin), mats["brow"])
    # ---- mouth
    for name, (dfn, relief, bx2) in _mouth(mouth or E.get("mouth", "small_smile"), my, R, lay.get("mw", 1.0)).items():
        out[name] = (_Decal(head, dfn, bx2, F32(relief), zmin), mats[name])
    # ---- blush and freckles
    bl = E.get("blush", 0.6) * (1.0 if blush is True else float(blush or 0))
    if bl > 0.05:
        bcx, bcy = ex + 0.06 * R, ey - ery - 0.1 * R
        k = 0.8 + 0.2 * min(bl, 1.6)
        brx, bry = 0.1 * R * k, 0.06 * R * k
        D_bl = (lambda x, y: np.minimum(_e2(x, y, bcx, bcy, brx, bry), _e2(x, y, -bcx, bcy, brx, bry)))
        out["blush"] = (_Decal(head, D_bl, ((-bcx - brx * 1.2, bcy - bry * 1.3), (bcx + brx * 1.2, bcy + bry * 1.3)),
                               F32(0.0012 * R), zmin),
                        Material(color=_mix(skin, "#ff6a6a", min(0.16 + 0.13 * bl, 0.5)), roughness=0.7,
                                 specular=0.15, subsurface=0.3, shadow=False))
    if freckles:
        # a tidy sprinkle across each cheek: an arc of round dots, a little jitter in size, none touching
        pts, rad = [], []
        rng = np.random.default_rng(7)
        n_f = 5 if detail != "low" else 3
        for s in (1, -1):
            for i in range(n_f):
                t = (i / (n_f - 1) - 0.5) if n_f > 1 else 0.0
                pts.append((s * (ex + 0.01 * R + t * 0.2 * R), ey - ery - 0.075 * R - 0.05 * R * (1 - 4 * t * t)
                            + (0.022 * R if i % 2 else 0.0)))
                rad.append(0.015 * R * rng.uniform(0.8, 1.15))
        P = np.array(pts)
        fr = max(rad)

        def D_fr(x, y, P=P):
            d = None
            for (px, py), rr_ in zip(P, rad):
                di = np.sqrt((x - F32(px)) ** 2 + (y - F32(py)) ** 2) - F32(rr_)
                d = di if d is None else np.minimum(d, di)
            return d
        out["freckles"] = (_Decal(head, D_fr, ((P[:, 0].min() - fr * 2, P[:, 1].min() - fr * 2),
                                              (P[:, 0].max() + fr * 2, P[:, 1].max() + fr * 2)), F32(0.0035 * R), zmin),
                           Material(color=_mix(_shade(skin, 0.72), "#a0522d", 0.25), roughness=0.7, specular=0.1,
                                    shadow=False))
    return out


def _mouth(kind, my, R, mw=1.0):
    """2D mouth shapes: {part: (d2 function, relief, 2D box)}."""
    w = 0.03 * R
    rel = 0.006 * R
    box = ((-0.32 * R, my - 0.3 * R), (0.32 * R, my + 0.22 * R))
    out = {}
    if kind in ("smile", "small_smile"):
        big = kind == "smile"
        r = (0.17 if big else 0.115) * R * mw
        pts = _arc_pts(0, my + r * 0.75, r, 222 if big else 236, 318 if big else 304, 9)
        ww = w * (1.0 if big else 0.85)
        out["mouth"] = (lambda x, y: _stroke2(x, y, pts, ww), rel, box)
    elif kind == "frown":
        r = 0.14 * R * mw
        pts = _arc_pts(0, my - r * 0.95, r, 50, 130, 9)
        out["mouth"] = (lambda x, y: _stroke2(x, y, pts, w), rel, box)
    elif kind == "flat":
        pts = [(-0.085 * R * mw, my + 0.004 * R), (0.0, my - 0.004 * R), (0.085 * R * mw, my + 0.002 * R)]
        out["mouth"] = (lambda x, y: _stroke2(x, y, pts, w * 0.9), rel, box)
    elif kind == "wavy":
        xs = np.linspace(-0.09, 0.09, 9) * R * mw
        pts = [(float(x), my + 0.012 * R * math.sin(i * 1.6)) for i, x in enumerate(xs)]
        out["mouth"] = (lambda x, y: _stroke2(x, y, pts, w * 0.8), rel, box)
    elif kind == "smirk":
        pts = _bez((-0.1 * R, my + 0.005 * R), (0.03 * R, my - 0.05 * R), (0.13 * R, my + 0.04 * R), 9)
        out["mouth"] = (lambda x, y: _stroke2(x, y, pts, w * 0.9), rel, box)
    elif kind in ("o", "small_o"):
        rx, ry = ((0.07, 0.09) if kind == "o" else (0.045, 0.052))
        out["mouth"] = (lambda x, y: _e2(x, y, 0, my - 0.02 * R, rx * R, ry * R), rel, box)
        if kind == "o":
            out["tongue"] = (lambda x, y: np.maximum(_e2(x, y, 0, my - 0.085 * R, 0.05 * R, 0.035 * R),
                                                     _e2(x, y, 0, my - 0.02 * R, rx * R, ry * R)), rel * 1.6, box)
    elif kind == "tongue":
        r = 0.16 * R * mw
        pts = _arc_pts(0, my + r * 0.75, r, 225, 315, 9)
        out["mouth"] = (lambda x, y: _stroke2(x, y, pts, w), rel, box)
        out["tongue"] = (lambda x, y: np.maximum(_e2(x, y, 0.03 * R, my - 0.07 * R, 0.065 * R, 0.08 * R),
                                                 y - F32(my - 0.012 * R)), rel * 2.2, box)
    elif kind in ("open_smile", "laugh", "smirk_open"):
        big = kind == "laugh"
        rx, ry = (0.19, 0.16) if big else ((0.12, 0.085) if kind == "smirk_open" else (0.15, 0.11))
        rx *= R * mw
        ry *= R
        top = my + (0.035 if big else 0.02) * R
        cx = 0.02 * R if kind == "smirk_open" else 0.0

        def D_m(x, y):
            return np.maximum(_e2(x, y, cx, top, rx, ry), y - F32(top))
        out["mouth"] = (D_m, rel, box)
        out["tongue"] = (lambda x, y: np.maximum(_e2(x, y, cx, top - ry * 0.95, rx * 0.62, ry * 0.5), D_m(x, y)),
                         rel * 1.6, box)
        if big:
            out["teeth"] = (lambda x, y: np.maximum(D_m(x, y), F32(top - 0.05 * R) - y), rel * 1.6, box)
    elif kind == "grit":
        a, b = (-0.1 * R * mw, my - 0.01 * R), (0.1 * R * mw, my - 0.01 * R)
        out["mouth"] = (lambda x, y: _seg2(x, y, a, b, 0.05 * R), rel, box)
        out["teeth"] = (lambda x, y: np.maximum(_seg2(x, y, a, b, 0.032 * R),
                                                F32(0.007 * R) - np.abs(y - F32(my - 0.01 * R))), rel * 1.6, box)
    else:
        return _mouth("small_smile", my, R, mw)
    return out
