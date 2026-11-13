# -*- coding: utf-8 -*-
"""Reference: read the style of a reference picture as numbers, and measure how far a draft is from it.

``probe(ref)`` returns a style sheet: the palette with shares and roles (background, skin, line, accent), the
line colour and weight, the shading structure of each large colour region (flat, hard tone bands with the
shadow tone as a lightness / hue shift of the base, or a soft gradient), highlight bands and specular spots,
value and saturation range, edge softness and grain. ``compare(ref, draft)`` lists the measurable gaps, largest
first, with one match figure 0-100. ``tools/match.py`` prints both; ``render.py --ref`` appends the gap list.

    from pixelcanvas import reference
    sheet = reference.probe("ref.png")
    print(reference.sheet_text(sheet))
    print("\\n".join(reference.gap_lines(reference.compare("ref.png", "draft.png"))))
"""
from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image, ImageFilter

from . import filters as _f
from .color import F32
from ._args import strict as _strict

__all__ = ["probe", "compare", "sheet_text", "gap_lines", "add_to_report", "load"]

WORK = 480                                                    # working height (px) of the measurements
_RES = getattr(Image, "Resampling", Image)


# ----------------------------------------------------------------------------------------------- loading

def load(img, height=WORK):
    """Any picture (path, PIL image, Canvas, float or uint8 array) -> float32 sRGB (h, w, 3) 0-1, at most
    ``height`` px high."""
    if isinstance(img, (str, os.PathLike)):
        img = Image.open(img).convert("RGB")
    if hasattr(img, "buf"):
        img = np.clip(np.asarray(img.buf, F32)[..., :3], 0, 1)
    if isinstance(img, np.ndarray):
        a = np.asarray(img, F32)
        if a.max() > 1.5:
            a = a / 255.0
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, -1)
        img = Image.fromarray((np.clip(a[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8))
    img = img.convert("RGB")
    if img.height > height:
        img = img.resize((max(1, round(img.width * height / img.height)), height), _RES.LANCZOS)
    return np.asarray(img, F32) / 255.0


def _lab(rgb):
    from .palette import srgb_to_oklab
    return srgb_to_oklab(np.clip(rgb, 0, 1)).astype(F32)


def _hex(lab):
    from .palette import oklab_to_srgb, to_hex
    return to_hex(oklab_to_srgb(np.asarray(lab, np.float64)))


def _hue(lab):
    return math.degrees(math.atan2(float(lab[2]), float(lab[1]))) % 360


def _chroma(lab):
    return math.hypot(float(lab[1]), float(lab[2]))


def _dhue(a, b):
    return ((b - a + 180.0) % 360.0) - 180.0


def _kmeans(pts, k, seed=0, iters=12):
    """k-means++ in OKLab (the palette.from_image recipe) on (n, 3) points -> centres (k, 3)."""
    rng = np.random.default_rng(seed)
    pts = pts.astype(np.float64)
    k = min(int(k), len(pts))
    cent = [pts[rng.integers(len(pts))]]
    for _ in range(1, k):
        d = np.min([((pts - c) ** 2).sum(1) for c in cent], 0)
        cent.append(pts[rng.choice(len(pts), p=d / d.sum())] if d.sum() > 0 else pts[rng.integers(len(pts))])
    cent = np.array(cent)
    for _ in range(iters):
        lab = np.argmin(((pts[:, None] - cent[None]) ** 2).sum(-1), 1)
        for j in range(k):
            if np.any(lab == j):
                cent[j] = pts[lab == j].mean(0)
    return cent


def _merge(cent, pts, tol=0.035):
    """Merge centres closer than ``tol`` (OKLab) -> centres sorted by lightness."""
    cent = list(cent)
    changed = True
    while changed and len(cent) > 1:
        changed = False
        C = np.array(cent)
        d = np.sqrt(((C[:, None] - C[None]) ** 2).sum(-1)) + np.eye(len(C)) * 9
        i, j = np.unravel_index(np.argmin(d), d.shape)
        if d[i, j] < tol:
            lab = np.argmin(((pts[:, None] - C[None]) ** 2).sum(-1), 1)
            sel = (lab == i) | (lab == j)
            new = pts[sel].mean(0) if sel.any() else (C[i] + C[j]) / 2
            cent = [c for n, c in enumerate(cent) if n not in (i, j)] + [new]
            changed = True
    C = np.array(cent)
    return C[np.argsort(C[:, 0])]


def _minmax(L, size, fn):
    im = Image.fromarray((np.clip(L, 0, 1) * 255 + 0.5).astype(np.uint8))
    return np.asarray(im.filter(fn(size)), F32) / 255.0


def _same_family(a, b, pale=True):
    """Two OKLab colours that can be tones of one material (same hue, or a pale colour and its cool shadow)."""
    dl = abs(float(a[0] - b[0]))
    ca, cb = _chroma(a), _chroma(b)
    lo_, hi_ = (a, b) if a[0] < b[0] else (b, a)
    if pale and _chroma(lo_) < 0.09 and _chroma(hi_) < 0.035 and 200 < _hue(lo_) < 335 and 0.03 < dl < 0.3:
        return True                                              # a pale colour and its cool tinted shadow
    if ca > 0.025 and cb > 0.025:                                # same hue, similar colour strength
        return abs(_dhue(_hue(a), _hue(b))) < 28 and dl < 0.4 and max(ca, cb) <= 2.2 * min(ca, cb) + 0.03
    if max(ca, cb) < 0.03:
        return dl < 0.32
    lo_, hi_ = (a, b) if a[0] < b[0] else (b, a)
    return _chroma(lo_) < 0.09 and _chroma(hi_) < 0.035 and 200 < _hue(lo_) < 335 and 0.03 < dl < 0.3


def _tone_steps(lab, line, k=3):
    """Crisp colour edges with flat sides, sampled across x and y (non-maximum suppressed) -> dict(light, dark:
    (n, 3) OKLab of the lighter / darker side; step: a tone step (one material, lightness differs 0.035-0.32, no
    drawn line between); line: a drawn line between the sides). The rest are boundaries between things."""
    h, w = lab.shape[:2]
    pad = k + 3
    P = np.pad(lab, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    near = _minmax(line.astype(F32), 2 * k + 1, ImageFilter.MaxFilter) > 0.5
    out_l, out_d, out_line = [], [], []
    for ax in (1, 0):
        def at(off):
            if ax == 1:
                return P[pad:pad + h, pad + off:pad + off + w]
            return P[pad + off:pad + off + h, pad:pad + w]

        def dist(a, b):
            return np.sqrt(((a - b) ** 2).sum(-1))
        A, A2, B, B2 = at(-k), at(-k - 2), at(k), at(k + 2)
        c1, cm, cp = dist(at(1), at(-1)), dist(at(0), at(-2)), dist(at(2), at(0))
        dAB = dist(A, B)
        ok = (dAB > 0.035) & (c1 >= cm) & (c1 > cp) & (c1 > 0.4 * dAB) & (dist(A, A2) < 0.02) & (dist(B, B2) < 0.02)
        a, b = A[ok], B[ok]
        swap = (a[:, 0] < b[:, 0])[:, None]
        out_l.append(np.where(swap, b, a))
        out_d.append(np.where(swap, a, b))
        out_line.append(near[ok])
    light, dark, on_line = np.concatenate(out_l), np.concatenate(out_d), np.concatenate(out_line)
    if len(light) > 60000:                                       # plenty: a fixed subsample keeps it fast
        keep = np.random.default_rng(0).choice(len(light), 60000, replace=False)
        light, dark, on_line = light[keep], dark[keep], on_line[keep]
    dl = light[:, 0] - dark[:, 0]
    ca, cb = np.hypot(light[:, 1], light[:, 2]), np.hypot(dark[:, 1], dark[:, 2])
    hu = np.degrees(np.arctan2(light[:, 2], light[:, 1]))
    hd = np.degrees(np.arctan2(dark[:, 2], dark[:, 1]))
    dh = np.abs(((hd - hu + 180.0) % 360.0) - 180.0)
    hd360 = hd % 360.0
    pale = (cb < 0.09) & (ca < 0.035) & (hd360 > 200) & (hd360 < 335)
    fam = pale | np.where((ca > 0.025) & (cb > 0.025), (dh < 28) & (np.maximum(ca, cb) <= 2.2 * np.minimum(ca, cb) + 0.03),
                   np.where(np.maximum(ca, cb) < 0.03, True,
                            (cb < 0.09) & (ca < 0.035) & (hd360 > 200) & (hd360 < 335)))
    step = fam & (dl > 0.06) & (dl < 0.32) & ~on_line
    return {"light": light, "dark": dark, "step": step, "line": on_line}

# ----------------------------------------------------------------------------------------------- probe

def probe(img, colors=8, seed=0):
    """Measure a picture's style -> dict (JSON-ready): palette, roles, lines, regions, highlights, values.
    ``colors``: palette size printed (the region analysis uses its own finer clustering)."""
    rgb = load(img)
    h, w = rgb.shape[:2]
    lab = _lab(rgb)
    L = lab[..., 0]
    C = np.hypot(lab[..., 1], lab[..., 2])

    # lines: thin strokes darker than their surroundings (black top-hat of the lightness)
    r = max(2, int(round(0.008 * h)))
    close = _minmax(_minmax(L, 2 * r + 1, ImageFilter.MaxFilter), 2 * r + 1, ImageFilter.MinFilter)
    top = close - L
    line = top > 0.12
    core = top > 0.22
    line_share = float(line.mean())
    lines = {"present": False, "share": round(line_share, 4)}
    if line_share > 0.004 and core.sum() > 20:
        area = float(line.sum())
        per = float(np.count_nonzero(line[:, 1:] != line[:, :-1]) + np.count_nonzero(line[1:] != line[:-1]))
        width_px = max(2.0 * area / max(per, 1.0), 0.8)
        lc = np.median(lab[core], 0)
        lines = {"present": True, "share": round(line_share, 4), "color": _hex(lc),
                 "width": round(width_px / h, 4), "coloured": bool(_chroma(lc) > 0.025 and lc[0] > 0.12),
                 "lab": [round(float(v), 4) for v in lc]}

    # mid-scale steps: flat (no change), soft (a gradient), step (a hard edge in reach)
    Lb = _f.gaussian_blur(L, 1.0)
    d = max(2, int(round(0.03 * h)))
    P = np.pad(Lb, d, mode="edge")
    D = np.maximum(np.abs(P[d:-d, 2 * d:] - P[d:-d, :-2 * d]), np.abs(P[2 * d:, d:-d] - P[:-2 * d, d:-d]))
    soft = (D >= 0.01) & (D < 0.05)
    flat = D < 0.01

    near_line = _minmax(line.astype(F32), 2 * d + 1, ImageFilter.MaxFilter) > 0.5

    # palette with shares (non-line pixels)
    keep = ~line
    step = max(1, int(math.sqrt(keep.sum() / 6000.0)))
    pts = lab[::step, ::step][keep[::step, ::step]].reshape(-1, 3)
    if len(pts) < 16:
        pts = lab.reshape(-1, 3)
    pal = _merge(_kmeans(pts, colors, seed), pts, 0.02)
    lbl = np.argmin(((lab[..., None, :] - pal[None, None]) ** 2).sum(-1), -1)
    share = np.bincount(lbl[keep], minlength=len(pal)) / max(float(keep.sum()), 1.0)
    order = np.argsort(-share)
    palette = [{"hex": _hex(pal[i]), "share": round(float(share[i]), 4)} for i in order if share[i] >= 0.005]

    # regions: a finer clustering grouped into families of adjacent tones of one hue, then joined across the
    # crisp tone-step edges that run between them (a cel band is a step, a gradient is not)
    fine = _merge(_kmeans(pts, 22, seed + 1), pts, 0.02)
    K = len(fine)
    fl = np.argmin(((lab[..., None, :] - fine[None, None]) ** 2).sum(-1), -1)
    fl = np.where(keep, fl, K)
    fs = np.bincount(fl[keep], minlength=K) / max(float(keep.sum()), 1.0)
    g = np.hypot(*np.gradient(Lb))
    stable = np.where(keep & (g < 0.02), fl, K)                 # anti-aliased edge pixels do not link tones
    o = 3
    pairs = np.concatenate([(stable[:, o:] * (K + 1) + stable[:, :-o]).ravel(),
                            (stable[o:] * (K + 1) + stable[:-o]).ravel()])
    adj = np.bincount(pairs, minlength=(K + 1) ** 2).reshape(K + 1, K + 1)[:K, :K]
    adj = adj + adj.T
    csoft = np.array([float(soft[fl == i].mean()) if fs[i] > 0 else 0.0 for i in range(K)]) > 0.35
    parent = list(range(K))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i in range(K):
        for j in range(i + 1, K):
            if csoft[i] != csoft[j] or min(fs[i], fs[j]) < 0.003 or \
                    adj[i, j] < max(6, 0.02 * math.sqrt(min(fs[i], fs[j]) * h * w)):
                continue
            if _same_family(fine[i], fine[j]) and abs(float(fine[i][0] - fine[j][0])) > 0.015:
                parent[root(i)] = root(j)
    st = _tone_steps(lab, line)
    n_e = len(st["light"])
    li = np.argmin(((st["light"][:, None] - fine[None]) ** 2).sum(-1), 1) if n_e else np.zeros(0, int)
    di = np.argmin(((st["dark"][:, None] - fine[None]) ** 2).sum(-1), 1) if n_e else np.zeros(0, int)
    stp = st["step"]
    cnt = np.zeros((K, K))
    np.add.at(cnt, (li[stp], di[stp]), 1)
    cnt = cnt + cnt.T
    for i in range(K):                                          # tones joined by a run of step edges: one material
        for j in range(i + 1, K):
            if csoft[i] != csoft[j] and min(_chroma(fine[i]), _chroma(fine[j])) < 0.03:
                continue                                        # a pale sky against white cloth: two things
            if cnt[i, j] >= max(8.0, 0.15 * math.sqrt(min(fs[i], fs[j]) * h * w)) and min(fs[i], fs[j]) >= 0.002                     and _same_family(fine[i], fine[j], pale=not (csoft[i] or csoft[j])):
                parent[root(i)] = root(j)
    fam = {}
    for i in range(K):
        fam.setdefault(root(i), []).append(i)
    regions = []
    for members in fam.values():
        tot = float(sum(fs[i] for i in members))
        if tot < 0.03:
            continue
        sel = np.isin(fl, members)
        soft_share = float(soft[sel].mean()) if sel.any() else 0.0
        base = max(members, key=lambda i: fs[i])
        inside = stp & np.isin(li, members) & np.isin(di, members) & ~csoft[di]   # a band is a flat tone
        eff = fs.copy()                                         # a graded base counts as one tone
        soft_m = [i for i in members if csoft[i]]
        eff[soft_m] = sum(fs[i] for i in soft_m)
        big = eff[li] >= eff[di]                                # the larger tone is the base, the smaller a band
        sh_e, hl_e = inside & big, inside & ~big
        need = max(8.0, 0.2 * math.sqrt(tot * h * w))
        n_sh, n_hl = int(sh_e.sum()), int(hl_e.sum())
        banded = n_sh >= need

        def support(idx):
            u, c = np.unique(idx, return_counts=True)
            return [int(x) for x, n in zip(u, c) if n >= 0.2 * len(idx)]
        darks = support(di[sh_e]) if n_sh else []
        lights = [i for i in support(li[sh_e]) if i not in darks] if n_sh else []
        if banded:
            kind = "bands on a graded base" if len(lights) >= 2 or (soft_share > 0.35 and csoft[base]) else "bands"
        elif soft_share > 0.35 or len([i for i in members if fs[i] >= 0.1 * tot]) >= 2:
            kind = "gradient"
        else:
            kind = "flat"
        tones = 1 + min(2, max(len(darks), 1)) if banded else 1
        base_set = lights or [base]
        reg = {"base": _hex(fine[base]), "share": round(tot, 4), "structure": kind, "tones": tones,
               "soft": round(soft_share, 3), "steps": n_sh + n_hl,
               "colors": [_hex(fine[i]) for i in sorted(members, key=lambda i: -float(fine[i][0]))]}
        if banded:
            first = sh_e & np.isin(li, base_set)                # base -> first shadow band
            first = first if first.sum() >= 5 else sh_e
            a, b = st["light"][first], st["dark"][first]
            dl = float(np.median(b[:, 0] - a[:, 0]))
            ok = (np.hypot(a[:, 1], a[:, 2]) > 0.04) & (np.hypot(b[:, 1], b[:, 2]) > 0.03)
            dh = float(np.median([_dhue(_hue(x), _hue(y)) for x, y in zip(a[ok][:400], b[ok][:400])])) \
                if ok.sum() >= 5 else 0.0
            reg["shift"] = [round(dl, 3), round(dh, 1)]
            reg["shadow"] = _hex(np.median(b, 0))
            if kind == "bands on a graded base":
                ms = sorted(base_set, key=lambda i: float(fine[i][0]))
                reg["ramp"] = [_hex(fine[ms[-1]]), _hex(fine[ms[0]])]
        if n_hl >= max(6.0, 0.3 * need):
            a, b = st["light"][hl_e], st["dark"][hl_e]
            reg["highlight"] = _hex(np.median(a, 0))
            reg["lift"] = round(float(np.median(a[:, 0] - b[:, 0])), 3)
        if kind == "gradient":
            ms = sorted(members, key=lambda i: float(fine[i][0]))
            reg["ramp"] = [_hex(fine[ms[-1]]), _hex(fine[ms[0]])]
        if 0.66 < fine[base][0] < 0.95 and 0.02 < _chroma(fine[base]) < 0.13 and 20 < _hue(fine[base]) < 95:
            reg["role"] = "skin"
        regions.append(reg)
    regions.sort(key=lambda r_: -r_["share"])

    # roles
    border = np.concatenate([lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]])
    roles = {"background": _hex(pal[np.bincount(border, minlength=len(pal)).argmax()])}
    for i in order:
        a = pal[i]
        if share[i] >= 0.01 and 0.66 < a[0] < 0.94 and 0.02 < _chroma(a) < 0.13 and 20 < _hue(a) < 95:
            roles["skin"] = _hex(a)
            break
    if lines["present"]:
        roles["line"] = lines["color"]
    vivid = [i for i in order if share[i] >= 0.01 and _hex(pal[i]) != roles["background"]]
    if vivid:
        roles["accent"] = _hex(pal[max(vivid, key=lambda i: _chroma(pal[i]))])

    # highlights: light bands inside regions, small near-white spots
    white = (L > 0.9) & (C < 0.05)
    rad = max(2, int(0.02 * h))
    dens = _f.gaussian_blur(white.astype(F32), rad)
    spots = white & (dens < 0.35)
    spot_share = float(spots.mean())
    bands_hl = [r_["base"] for r_ in regions if "highlight" in r_]

    # values, saturation, edges, grain
    lo, hi = (float(v) for v in np.percentile(L[keep] if keep.any() else L, [2, 98]))
    cmed, c90 = (float(v) for v in np.percentile(C[keep] if keep.any() else C, [50, 90]))
    near = _minmax(line.astype(F32), 9, ImageFilter.MaxFilter) > 0.5          # edges away from drawn lines
    Lp = np.pad(L, ((0, 0), (3, 3)), mode="edge")
    g1 = np.maximum(np.abs(Lp[:, 4:-2] - Lp[:, 3:-3]), np.abs(Lp[:, 3:-3] - Lp[:, 2:-4]))
    g6 = np.abs(Lp[:, 6:] - Lp[:, :-6])
    sel = (~near) & (g6 > 0.06)
    sharp = float(np.percentile(g1[sel], 95) / np.percentile(g6[sel], 95)) if sel.sum() > 30 else 0.7
    edge_soft = float(np.clip((0.5 - sharp) / 0.4, 0, 1))
    hp = np.abs(L - _f.gaussian_blur(L, 1.2))
    grain = float(np.median(hp[flat & keep])) if (flat & keep).any() else 0.0

    tot_share = sum(r_["share"] for r_ in regions) or 1.0
    by = {k: round(sum(r_["share"] for r_ in regions if r_["structure"].split(" ")[0] == k) / tot_share, 3)
          for k in ("flat", "bands", "gradient")}
    banded = [r_ for r_ in regions if "shift" in r_]
    shift = None
    if banded:
        wts = np.array([r_["share"] for r_ in banded])
        shift = [round(float(np.average([r_["shift"][0] for r_ in banded], weights=wts)), 3),
                 round(float(np.average([r_["shift"][1] for r_ in banded], weights=wts)), 1)]
    lifts = [r_["lift"] for r_ in regions if "lift" in r_]
    return {"size": [int(rgb.shape[1]), int(rgb.shape[0])], "palette": palette, "roles": roles, "lines": lines,
            "regions": regions[:6], "structure": by, "shift": shift,
            "lift": round(float(np.median(lifts)), 3) if lifts else None,
            "highlights": {"bands": bands_hl, "spots": spot_share > 0.0003, "spot_share": round(spot_share, 5)},
            "values": {"low": round(lo, 3), "high": round(hi, 3), "contrast": round(hi - lo, 3),
                       "mean": round(float(L.mean()), 3)},
            "saturation": {"median": round(cmed, 3), "p90": round(c90, 3)},
            "edges": {"softness": round(edge_soft, 3)}, "grain": round(grain, 4)}


def _word(v, words):
    for lim, wd in words:
        if v < lim:
            return wd
    return words[-1][1]


def sheet_text(p, name="reference"):
    """The probe as a short style sheet with ready-to-paste lines (list of str joined by newlines)."""
    out = [f"STYLE SHEET {name} ({p['size'][0]}x{p['size'][1]} measured)"]
    out.append("palette: " + "  ".join(f"{c['hex']} {round(c['share'] * 100)}%" for c in p["palette"]))
    out.append("roles: " + ", ".join(f"{k} {v}" for k, v in p["roles"].items()))
    ln = p["lines"]
    if ln["present"]:
        out.append(f"lines: {ln['color']} ({'coloured' if ln['coloured'] else 'near black'}), width {ln['width']} "
                   f"of the height (~{max(1, round(ln['width'] * 1080))} px at 1080)")
    else:
        out.append("lines: no outlines (shapes meet colour to colour)")
    out.append("shading by region (largest first):")
    for r_ in p["regions"]:
        s = f"  {r_['base']} {round(r_['share'] * 100)}%{' (skin)' if r_.get('role') else ''}: "
        if r_["structure"] == "gradient":
            s += f"soft gradient {r_['ramp'][0]} -> {r_['ramp'][1]}"
        elif r_["structure"].startswith("bands"):
            s += f"hard bands, {r_['tones']} tones"
            if r_["structure"] != "bands":
                s += " on a graded base" + (f" {r_['ramp'][0]} -> {r_['ramp'][1]}"
                                            if r_["ramp"][0] != r_["ramp"][1] else "")
            if "shift" in r_:
                s += f"; shadow {r_['shadow']} = L {r_['shift'][0]:+.2f}, hue {r_['shift'][1]:+.0f} deg"
        else:
            s += "flat"
        if "highlight" in r_:
            s += f"; highlight band {r_['highlight']} (L {r_['lift']:+.2f})"
        out.append(s)
    st = p["structure"]
    out.append(f"structure: {round(st['bands'] * 100)}% banded, {round(st['gradient'] * 100)}% gradient, "
               f"{round(st['flat'] * 100)}% flat")
    hl = p["highlights"]
    out.append(f"highlights: bands {'on ' + ', '.join(hl['bands']) if hl['bands'] else 'none'}; specular spots "
               f"{'yes' if hl['spots'] else 'none'}")
    v, sa = p["values"], p["saturation"]
    out.append(f"values: L {v['low']:.2f}-{v['high']:.2f} (contrast {v['contrast']:.2f}, mean {v['mean']:.2f}); "
               f"saturation median {sa['median']:.3f}, p90 {sa['p90']:.3f}; edges "
               f"{_word(p['edges']['softness'], [(0.25, 'crisp'), (0.6, 'soft'), (9, 'blurred')])} "
               f"({p['edges']['softness']:.2f}); grain "
               f"{_word(p['grain'], [(0.004, 'none'), (0.012, 'light'), (9, 'heavy')])} ({p['grain']:.4f})")
    out.append("PASTE (scene docstring / constants):")
    out.append("  PALETTE = [" + ", ".join(f'"{c["hex"]}"' for c in p["palette"]) + "]")
    if ln["present"]:
        out.append(f'  LINE, LINE_WIDTH = "{ln["color"]}", {ln["width"]}      '
                   "# sk.cel(..., line=LINE, line_width=LINE_WIDTH)")
    else:
        out.append("  LINE = None                          # no outlines: leave line= off")
    if p["shift"]:
        nb = 2 if any(r_["tones"] >= 3 for r_ in p["regions"] if "shift" in r_) else 1
        out.append(f"  SHIFT, BANDS = ({p['shift'][0]}, {p['shift'][1]}), {nb}     "
                   "# sk.cel(mask, base, bands=BANDS, shift=SHIFT)")
        per = [r_ for r_ in p["regions"] if "shift" in r_]
        if len(per) > 1:
            out.append("  SHIFTS = {" + ", ".join(f'"{r_["base"]}": ({r_["shift"][0]}, {r_["shift"][1]})'
                                                for r_ in per) + "}   # per material: sk.cel(m, c, shift=SHIFTS[c])")
    else:
        out.append("  SHIFT, BANDS = None, 0                # flat or graded fills: sk.paint / sk.ramp")
    if p["lift"] is not None:
        out.append(f"  LIFT = {p['lift']}                        # highlight = cel_tone(base, LIFT, 0)")
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------- compare

def _pal_lab(p):
    from .palette import srgb_to_oklab, rgb
    return np.array([srgb_to_oklab(np.asarray(rgb(c["hex"])[:3], F32)) for c in p["palette"]], np.float64), \
        np.array([c["share"] for c in p["palette"]], np.float64)


def compare(ref, draft):
    """Gaps between a draft and a reference -> dict(match 0-100, gaps [largest first], ref, draft probes).
    ``ref`` / ``draft``: pictures or probe dicts."""
    pr = ref if isinstance(ref, dict) else probe(ref)
    pd = draft if isinstance(draft, dict) else probe(draft)
    gaps = []

    def add(key, gap, weight, msg, fix):
        gaps.append({"key": key, "gap": round(float(min(max(gap, 0.0), 1.0)), 3), "weight": weight,
                     "message": msg, "fix": fix})

    A, wa = _pal_lab(pr)
    B, wb = _pal_lab(pd)
    dab = np.sqrt(((A[:, None] - B[None]) ** 2).sum(-1))
    d1, d2 = float((dab.min(1) * wa).sum() / wa.sum()), float((dab.min(0) * wb).sum() / wb.sum())
    dE = (d1 + d2) / 2
    worst = np.argsort(-(dab.min(1) * wa))[:3]
    miss = [f"{pr['palette'][i]['hex']} (draft nearest {pd['palette'][int(dab[i].argmin())]['hex']})"
            for i in worst if dab[i].min() > 0.04]
    add("palette", dE / 0.12, 3.0, f"palette distance {dE:.3f} (OKLab)",
        ("use the reference colours: " + ", ".join(miss)) if miss else "palette matches")

    rb, db = pr["structure"]["bands"], pd["structure"]["bands"]
    rt = max([r_["tones"] for r_ in pr["regions"] if "shift" in r_] or [1])
    if rb > db:
        add("bands", (rb - db) / max(rb, 0.25), 3.0,
            f"flat where the reference has {rt if rt <= 2 else '2-3'} tones: banded area {round(db * 100)}% vs "
            f"{round(rb * 100)}%",
            "paint skin, hair and cloth with sk.cel(mask, base, bands="
            f"{2 if rt >= 3 else 1}, shift={tuple(pr['shift']) if pr['shift'] else (-0.12, None)})")
    else:
        add("bands", (db - rb) / max(db, 0.25) * 0.6, 3.0,
            f"banded area {round(db * 100)}% vs the reference's {round(rb * 100)}%",
            "fewer shadow bands: flat fills where the reference is flat" if db - rb > 0.1 else "tone bands match")

    rg, dg = pr["structure"]["gradient"], pd["structure"]["gradient"]
    add("gradient", abs(rg - dg) / max(rg, dg, 0.25), 1.5,
        f"gradient area {round(dg * 100)}% vs the reference's {round(rg * 100)}%",
        ("grade the sky / hair with sk.ramp(mask, colors)" if rg > dg else "fewer gradients: cel bands or flat fills")
        if abs(rg - dg) > 0.1 else "gradients match")

    lr, ld = pr["lines"], pd["lines"]
    if lr["present"] and not ld["present"]:
        add("lines", 1.0, 2.0, f"no outlines; the reference draws {lr['color']} lines {lr['width']} of the height",
            f"line=\"{lr['color']}\", line_width={lr['width']} on sk.cel / sk.paint; sk.ink for inner lines")
    elif ld["present"] and not lr["present"]:
        add("lines", 0.8, 2.0, "outlines drawn; the reference has none", "drop line= (shapes meet colour to colour)")
    elif lr["present"]:
        dc = float(np.sqrt(((np.array(lr["lab"]) - np.array(ld["lab"])) ** 2).sum()))
        ratio = ld["width"] / max(lr["width"], 1e-4)
        g = 0.5 * min(dc / 0.15, 1) + 0.5 * min(abs(math.log(max(ratio, 1e-3))) / math.log(3), 1)
        add("lines", g, 2.0, f"line {ld['color']} width {ld['width']} vs {lr['color']} width {lr['width']}",
            f"line=\"{lr['color']}\", line_width={lr['width']}" if g > 0.15 else "lines match")
    else:
        add("lines", 0.0, 2.0, "no outlines in either", "lines match")

    vr, vd = pr["values"], pd["values"]
    add("contrast", abs(vr["contrast"] - vd["contrast"]) / 0.4, 1.0,
        f"value range {vd['low']:.2f}-{vd['high']:.2f} vs {vr['low']:.2f}-{vr['high']:.2f}",
        ("more contrast: darker darks / lighter lights" if vd["contrast"] < vr["contrast"] else "less contrast")
        if abs(vr["contrast"] - vd["contrast"]) > 0.05 else "contrast matches")
    add("key", abs(vr["mean"] - vd["mean"]) / 0.3, 1.0, f"mean lightness {vd['mean']:.2f} vs {vr['mean']:.2f}",
        ("lighter overall" if vd["mean"] < vr["mean"] else "darker overall")
        if abs(vr["mean"] - vd["mean"]) > 0.04 else "key matches")
    sr, sd = pr["saturation"]["median"], pd["saturation"]["median"]
    add("saturation", abs(sr - sd) / 0.08, 1.0, f"median saturation {sd:.3f} vs {sr:.3f}",
        ("more saturated colours" if sd < sr else "less saturated colours") if abs(sr - sd) > 0.01
        else "saturation matches")
    hr, hd = pr["highlights"], pd["highlights"]
    hg = 0.0
    if hr["bands"] and not hd["bands"]:
        hg += 0.5
    if hr["spots"] and not hd["spots"]:
        hg += 0.5
    add("highlights", hg, 1.0, f"highlight bands {len(hd['bands'])} vs {len(hr['bands'])}, spots "
        f"{'yes' if hd['spots'] else 'no'} vs {'yes' if hr['spots'] else 'no'}",
        "add sk.sheen on hair, sk.cel(..., highlight=0.8), eye highlights (sk.eye)" if hg else "highlights match")
    tw = sum(g["weight"] for g in gaps)
    match = 100.0 * (1.0 - sum(g["gap"] * g["weight"] for g in gaps) / tw)
    gaps.sort(key=lambda g: -g["gap"] * g["weight"])
    return {"match": round(match, 1), "gaps": gaps, "ref": pr, "draft": pd}


def gap_lines(c, limit=8):
    """The comparison as FIX-style lines, largest gap first (list of str)."""
    out = [f"match {c['match']:.0f}/100 against the reference"]
    n = 0
    for g in c["gaps"]:
        if g["gap"] < 0.08 or n >= limit or g["fix"].endswith("matches"):
            continue
        n += 1
        out.append(f"  {n}. [{g['key']} gap {g['gap']:.2f}] {g['message']} - {g['fix']}")
    if n == 0:
        out.append("  nothing measurable to fix: compare the two pictures by eye once")
    return out


def add_to_report(report, ref, picture, scene=None):
    """render.py hook: insert a REFERENCE block (the gap list) into the --check text just before its FIX block.
    ``ref``: a path (relative paths are also tried next to ``scene``); None leaves the report unchanged."""
    if not ref or report is None:
        return report
    path = ref
    if not os.path.exists(path) and scene:
        alt = os.path.join(os.path.dirname(os.path.abspath(scene)), ref)
        path = alt if os.path.exists(alt) else path
    if not os.path.exists(path):
        block = [f"REFERENCE {ref}: file not found"]
    else:
        block = [f"REFERENCE {path}"] + ["  " + t for t in gap_lines(compare(path, picture))]
    lines = report.split("\n")
    at = next((i for i in range(len(lines) - 1, -1, -1) if lines[i].startswith("FIX")), len(lines))
    if at > 0 and lines[at - 1].startswith("picture: "):
        at -= 1
    return "\n".join(lines[:at] + block + lines[at:])


for _n in ("probe", "compare", "sheet_text", "gap_lines", "add_to_report", "load"):
    globals()[_n] = _strict(globals()[_n])
