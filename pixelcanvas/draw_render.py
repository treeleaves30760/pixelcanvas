"""The 3D figure carried through line, flat and shade: ``pixelcanvas dress``, ``hair3d``, ``render`` and the 3D face.

- ``dress <who> <piece>[,<piece>...] [--color] [--length] [--fit] [--sleeve] [--style]`` puts garments on a posed
  figure: shells round its bones that bend with every joint (``figure3d.GARMENTS``).
- ``hair3d <who> --style <long|...> [--bangs] [--color] [--wind dx,dy] [--bows color]``: a cap on the skull and
  tapered locks that hang, trail behind the motion of the pose and stay outside the head and shoulders.
- ``render [<who>|all] [--line 2.2] [--bands 2|3] [--light from-plan|<words>]``: the toon render of every figure,
  block and prop: line art on the line layer (as its base; line records such as the face stay on top), flat material
  colours on the flat layer, cel shading, cast shadows and the hair ring on the shade layer. Each material is a
  region and a palette swatch, so ``flat N --color``, ``shade``, ``surface``, ``detail``, ``polish`` and ``export``
  work on the result. Bands painted behind (sky, far land) stay; marks made after a render on the ``touch`` layer
  and line records (face, hand-drawn lines) are kept by the next render.
- ``face <who> --expression <...>`` on a posed figure: brows, eyes, nose, mouth, tears and blush on the 3D head
  (``face3d``), redrawn by every render, so they turn with the head.

After a render, ``pose``, ``dress``, ``hair3d``, ``camera``, ``block`` and ``prop`` render again on their own (one more
undo step), so a re-pose never leaves stale line art.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import coords
from . import draw_core as dc
from . import draw_pose as DP
from . import face3d as FC
from . import figure3d as F3
from . import pose3d as P3
from . import toon3d as T3
from .coords import UsageError

USAGE = {
    "dress": "pixelcanvas dress girl blouse,skirt,thigh-highs,shoes --color navy",
    "hair3d": "pixelcanvas hair3d girl --style twintails --bangs blunt --color pink --bows black",
    "render": "pixelcanvas render all --line 2.2 --bands 2",
}
RENDER_MAX = 1700
R3D = "draw/r3d"
DEFAULT_LINE = 2.2

SKIN = "#fde4d4"
HAIR = "#3a2c30"
BLOCK_COLOURS = {"wall": "#e6dccb", "floor": "#b99a78", "ground": "#a8a090", "lamp": "#f2eee0", "bench": "#9a6e4c",
                 "stairs": "#cfc6b8", "step": "#cfc6b8", "railing": "#6e7480", "fence": "#8a8f98", "door": "#8a6448",
                 "window": "#bcd4e6", "ceiling": "#efe9dc", "pillar": "#ddd5c6", "column": "#ddd5c6",
                 "tree": "#6a5040", "building": "#d8d0c4", "desk": "#a07850", "table": "#a07850",
                 "locker": "#8aa0b4", "board": "#3a5a48", "box": "#c8b8a0", "crate": "#a88660", "plane": "#b8b0a0"}


# ============================================================================================ records

def figure_of(p, subj: str, usage: str) -> dict:
    s = str(subj or "").strip().lower()
    figs = DP.figures(p)
    for el in figs:
        if el.get("subject") == s:
            return el
    names = ", ".join(f["subject"] for f in figs) or "none posed yet"
    raise UsageError(f"{s!r} is not posed in 3D (posed: {names}); pose it first", f"pixelcanvas pose {s or 'girl'} "
                     "--preset stand")


def look_of(el: dict) -> dict:
    lk = el.setdefault("look", {})
    lk.setdefault("garments", {})
    return lk


def render_state(p) -> dict:
    return dc.dstate(p).get("render") or {}


def _scene_key(p, line: float, bands: int, light) -> str:
    d = dc.dstate(p)
    recs = []
    for el in d["sketch"]:
        if el["kind"] in ("block", "prop") or DP.is_rigged(el) or (el["kind"] == "horizon" and el.get("camera")):
            recs.append({k: v for k, v in el.items() if k not in ("n",)})
    return hashlib.sha1(json.dumps([p.W, p.H, recs, round(float(line), 3), int(bands), [round(float(v), 4) for v in
                                                                                       light]],
                                   sort_keys=True, default=str).encode()).hexdigest()


def render_mask(p) -> np.ndarray:
    """The areas the 3D render painted (figures, garments, hair, faces and blocks); empty when nothing is rendered."""
    m = np.zeros((p.H, p.W), bool)
    if not dc.dstate(p).get("render"):
        return m
    for n, info in p.state.get("regions", {}).items():
        if str(info.get("how", "")).startswith("3D render"):
            try:
                m |= p.region_mask(n)
            except Exception:  # noqa: BLE001
                continue
    return m


def is_render_region(p, name: str) -> bool:
    return bool(dc.dstate(p).get("render")) and         str(p.state.get("regions", {}).get(name, {}).get("how", "")).startswith("3D render")


def keeps_clean(p) -> bool:
    """A rendered cel or soft picture keeps its rendered areas as the render made them: no surface texture,
    no smoothing or softening by polish."""
    from .draw_surface import style_mode
    return bool(dc.dstate(p).get("render")) and style_mode(p) != "paint"


def is_stale(p) -> bool:
    rs = render_state(p)
    if not rs:
        return False
    return rs.get("key") != _scene_key(p, rs.get("line", DEFAULT_LINE), rs.get("bands", 2), _light(p, rs))


def _light(p, rs: Optional[dict] = None) -> np.ndarray:
    rs = rs or {}
    if rs.get("light_words"):
        deg, z, _ = dc.parse_light(rs["light_words"])
        return dc.light_vec({"light": {"deg": deg, "z": z}})
    return dc.light_vec(dc.dstate(p))


# ============================================================================================ colours and regions

def _col(p, spec, default: str) -> Tuple[int, int, int]:
    from .draw_plan import swatch
    try:
        return swatch(p, spec or default)
    except UsageError:
        return dc.from_hex(default)


def material_colours(p, el: dict) -> Dict[str, Tuple[int, int, int]]:
    lk = look_of(el)
    d = dc.dstate(p)
    pal = d["palette"]["swatches"]
    out = {"skin": _col(p, lk.get("skin") or ("skin" if "skin" in pal else None), SKIN)}
    hair = lk.get("hair") or {}
    out["hair"] = _col(p, hair.get("color"), HAIR)
    out["hair-tie"] = _col(p, hair.get("bows") or "#2a2a30", "#2a2a30")
    out["bow"] = _col(p, hair.get("bows") or "#c8303c", "#c8303c")
    out["collar"] = (248, 248, 252)
    out["button"] = _col(p, None, F3.TRIM_COLOUR)
    trim = None
    for g, o in lk["garments"].items():
        out[g] = _col(p, (o or {}).get("color"), F3.GARMENT_COLOURS.get(g, "#808080"))
        if (o or {}).get("trim"):
            trim = (o or {})["trim"]
    out["trim"] = _col(p, trim, F3.TRIM_COLOUR)
    out["button"] = out["trim"]
    return out


def block_colour(p, el: dict) -> Tuple[int, int, int]:
    d = dc.dstate(p)
    if el.get("color"):
        return _col(p, el["color"], "#c8c0b4")
    if el["name"] in d["palette"]["swatches"]:
        return _col(p, el["name"], "#c8c0b4")
    for w, c in BLOCK_COLOURS.items():
        if w in el["name"]:
            return dc.from_hex(c)
    return dc.from_hex(BLOCK_COLOURS.get(el["type"], "#c8c0b4"))


SKIN_PARTS = ("skin",)
PART_WORDS = {"skin": ("skin",), "hair": ("hair",), "blouse": ("blouse", "shirt", "top"), "shirt": ("shirt", "top"),
              "blazer": ("blazer", "jacket"), "jacket": ("jacket",), "coat": ("coat",), "hoodie": ("hoodie",),
              "skirt": ("skirt",), "dress": ("dress",), "pants": ("pants", "trousers"), "shorts": ("shorts",),
              "socks": ("socks",), "thigh-highs": ("thigh", "stockings"), "shoes": ("shoes", "loafers"),
              "boots": ("boots",), "scarf": ("scarf",), "tie": ("tie",), "ribbon": ("ribbon",)}


def region_name(p, subj: str, mat: str) -> str:
    """The plan part a material of ``subj`` is (its one part whose name says it), else '<subj>-<material>'."""
    d = dc.dstate(p)
    words = PART_WORDS.get(mat)
    if words:
        hits = [n for n, v in d["parts"].items() if v.get("of") == subj and
                any(w in n.replace("_", "-").split("-") or n == w for w in words)]
        if len(hits) == 1:
            return hits[0]
    from .regions import clean_name
    return clean_name(f"{subj}-{mat}")


# ============================================================================================ the scene

def scene(p, cam: P3.Camera) -> dict:
    """Primitives of every posed figure (dressed), block and prop with material keys '<owner>:<material>'."""
    prims: List[dict] = []
    colours: Dict[str, Tuple[int, int, int]] = {}
    regions: Dict[str, str] = {}
    figs = DP.figures(p)
    rings = {}
    heads = {}
    props = DP.props_rec(p)
    for el in figs:
        body, J, F = DP._fig_state(el)
        own = el["id"]
        held = [pr for pr in props if pr.get("subject") == el["subject"]]
        ps = F3.dressed_prims(body, J, F, el["rig"], look_of(el), own, held)
        cols = material_colours(p, el)
        for pr in ps:
            m = pr["mat"]
            key = f"{own}:{m}"
            pr["mat"] = key
            if pr.get("trim_mat"):
                pr["trim_mat"] = f"{own}:{pr['trim_mat']}"
            colours.setdefault(key, cols.get(m, (128, 128, 128)))
            regions.setdefault(key, region_name(p, el["subject"], m))
        for pr in ps:
            if pr.get("trim_mat"):
                tm = pr["trim_mat"]
                colours.setdefault(tm, cols.get(tm.split(":", 1)[1], cols["trim"]))
                regions.setdefault(tm, region_name(p, el["subject"], tm.split(":", 1)[1]))
        prims += ps
        rig = F3.Rig(body, J, F)
        rings[own] = (rig.cr_c, rig.Hd, rig.h)
        heads[own] = rig
        for pe in held:
            for pr in P3.prop_prims(pe["type"], pe["hand"], pe.get("length"), body, J, F, pe["id"]):
                if pr["part"].endswith("canopy"):
                    pr = _canopy(pr)
                part = pr["part"]
                pr["piece"] = part
                pr["mat"] = f"{pe['id']}:{part}"
                pr["seam"] = "mat"
                colours.setdefault(pr["mat"], _col(p, pe.get("color"), {"sword": "#c8ccd4", "umbrella": "#5a7ab8"}
                                                   .get(pe["type"], "#8a6a50")))
                regions.setdefault(pr["mat"], f"{pe['name']}")
                prims.append(pr)
    for el in DP.blocks_rec(p):
        if el.get("pos") is None:
            continue
        rgb = block_colour(p, el)
        for pr in DP._block(el).prims(el["id"]):
            pr["piece"] = pr["part"]
            pr["mat"] = f"{el['id']}:{el['name']}"
            pr["seam"] = "mat"
            pr["caster"] = False
            prims.append(pr)
        colours[f"{el['id']}:{el['name']}"] = rgb
        regions[f"{el['id']}:{el['name']}"] = el["name"]
    for k in list(colours):
        colours[k + "/in"] = dc.shadow_of(colours[k], 1.3)
        regions[k + "/in"] = regions[k]
    return {"prims": prims, "colours": colours, "regions": regions, "rings": rings, "heads": heads,
            "figures": [el["id"] for el in figs], "blocks": [el["id"] for el in DP.blocks_rec(p)]}


def _canopy(pr: dict) -> dict:
    """An umbrella's canopy as cloth: eight panels on ribs, a scalloped hem, open underneath."""
    from .render3d import shell
    A = np.asarray(pr["A"], float)
    up = A[:, 1] / max(np.linalg.norm(A[:, 1]), 1e-9)
    R = np.linalg.norm(A[:, 0])
    top = pr["c"] + up * 0.16
    r0 = 0.03
    x = A[:, 0] / R
    z = np.cross(x, up)
    Ash = np.stack([x * r0, up * 0.26, z * r0], 1)
    out = shell(top, Ash, flare=(R / r0 - 1.0), pleats=8, hem_amp=0.06)
    for k in ("part", "group", "owner"):
        out[k] = pr.get(k)
    return out


def toon_scene(p, cap: int = RENDER_MAX, bands: int = 2, light=None, shadows: bool = True):
    s = min(1.0, cap / float(max(p.W, p.H)))
    Wr, Hr = max(1, int(round(p.W * s))), max(1, int(round(p.H * s)))
    s = Wr / float(p.W)
    cam = DP.camera_of(p, Wr, Hr)
    sc = scene(p, cam)
    lv = light if light is not None else dc.light_vec(dc.dstate(p))
    T = T3.toon(cam, sc["prims"], lv, bands=bands, figure_owners=sc["figures"], rings=sc["rings"], shadows=shadows)
    return T, sc, cam, s


# ============================================================================================ painting

def _up(a: np.ndarray, W: int, H: int, interp=cv2.INTER_NEAREST) -> np.ndarray:
    return cv2.resize(a, (W, H), interpolation=interp)


def render_layers(p, T: dict, sc: dict, cam, s: float, line_px: float, bands: int) -> dict:
    """Ops for the flat, shade and line layers, the regions and the per-material colours."""
    from .ops import Fill, Stroke, mask_to_rle
    W, H = p.W, p.H
    mats = T["mats"]
    colours = sc["colours"]
    regions = sc["regions"]
    mat_up = _up(T["mat"].astype(np.int16), W, H)
    lam_up = _up(T["lam"], W, H, cv2.INTER_LINEAR)
    cast_up = _up(T["cast"].astype(np.float32), W, H, cv2.INTER_LINEAR)
    sig = max(0.8, 0.0012 * max(W, H))
    cast_up = cv2.GaussianBlur(cast_up, (0, 0), sig) > 0.5
    hl_up = cv2.GaussianBlur(_up(T["hl"].astype(np.float32), W, H, cv2.INTER_LINEAR), (0, 0), sig) > 0.5
    flat_ops, shade_ops = [], []
    region_masks: Dict[str, np.ndarray] = {}
    used = {}
    skin_keys = {i for i, m in enumerate(mats) if m.split(":")[-1].split("/")[0] == "skin"}
    for k, m in enumerate(mats):
        sel = mat_up == k
        if sel.sum() < 4:
            continue
        rgb = colours.get(m, (128, 128, 128))
        used[m] = rgb
        flat_ops.append(_fill(sel, rgb))
        rn = regions.get(m, m.replace(":", "-"))
        region_masks[rn] = region_masks.get(rn, np.zeros((H, W), bool)) | sel
        if m.endswith("/in"):
            continue
        thr = -0.25 if k in skin_keys else 0.08
        sh = sel & ((lam_up <= thr) | cast_up)
        sh = dc.clean_mask(sh, max(6, int(2e-6 * W * H)))
        block = m.split(":")[0] not in sc["figures"] and m.split(":")[0] in sc.get("blocks", ())
        if sh.sum() >= 6:
            shade_ops.append(_fill(sh, _block_shadow(rgb) if block else dc.shadow_of(rgb)))
        if bands >= 3:
            dp = sel & (lam_up <= thr - 0.45)
            dp = dc.clean_mask(dp, max(6, int(2e-6 * W * H)))
            if dp.sum() >= 6:
                shade_ops.append(_fill(dp, dc.shadow_of(rgb, 1.8)))
        if m.split(":")[-1] == "hair":
            hm = sel & hl_up & ~sh
            if hm.sum() >= 6:
                shade_ops.append(_fill(hm, dc.light_of(rgb, 1.6)))
    # contact shadows on the ground under figures that stand on it (with no floor block to catch the shadow)
    shade_ops += _ground_shadows(p, T, sc)
    # line art
    d = T["R"]["depth"]
    fig_px = np.isin(T["own"], [T["owners"].index(o) for o in sc["figures"] if o in T["owners"]])
    dref = float(np.median(d[fig_px])) if fig_px.any() else float(np.median(d[np.isfinite(d)])) if \
        np.isfinite(d).any() else 3.0
    seg_mat = _segment_materials(T)

    def mat_rgb(si):
        mk = seg_mat.get(si)
        return colours.get(mats[mk], (60, 60, 60)) if mk is not None else (60, 60, 60)
    strokes = T3.lines(T["R"], sc["prims"], T["seg"], T["table"], s, line_px, mat_rgb, dref)
    line_ops = [Stroke(pts=st["pts"], width=st["width"], color=st["rgb"], brush="liner") for st in strokes]
    return {"flat": flat_ops, "shade": shade_ops, "line": line_ops, "regions": region_masks, "colours": used,
            "n_strokes": len(strokes), "dref": dref, "aliases": _alias_regions(p, T, sc, W, H)}


def _alias_regions(p, T: dict, sc: dict, W: int, H: int) -> Dict[str, np.ndarray]:
    """A planned face / head part of a 3D figure: the head's skin (head, ears, neck) as its own region."""
    from .draw_checks import HEAD_PARTS
    d = dc.dstate(p)
    keys: Dict[tuple, int] = {}
    for k, pr in enumerate(sc["prims"]):
        keys.setdefault((pr.get("owner", ""), pr.get("piece") or pr.get("part") or f"p{k}"), len(keys))
    names = list(keys)
    seg_piece = np.array([t[0] for t in T["table"]] + [-1], np.int64)
    piece_px = seg_piece[T["seg"]]
    out: Dict[str, np.ndarray] = {}
    for el in DP.figures(p):
        parts = [n for n, v in d["parts"].items() if v.get("of") == el["subject"] and n in HEAD_PARTS
                 and not v.get("r3d")]
        if not parts:
            continue
        idx = [i for i, (o, pc) in enumerate(names) if o == el["id"] and pc in ("head", "ear-r", "ear-l", "neck")]
        m = _up(np.isin(piece_px, idx).astype(np.uint8), W, H) > 0
        if m.sum() > 20:
            for n in parts:
                out[n] = m
    return out


def _block_shadow(rgb) -> Tuple[int, int, int]:
    """The shadow side of a wall or a floor: the same hue, a step darker (no violet cast on a lit room)."""
    L, C, h = dc.rgb_to_lch(rgb)
    return dc.lch_to_rgb(max(5.0, L - 9.0), C * 1.12 + 1.0, h)


def _fill(mask: np.ndarray, rgb, alpha: float = 1.0):
    from .ops import Fill, mask_to_rle
    return Fill(mask_rle=mask_to_rle(mask), color=tuple(int(c) for c in rgb), feather=0.7, opacity=alpha) \
        if alpha < 1.0 else Fill(mask_rle=mask_to_rle(mask), color=tuple(int(c) for c in rgb), feather=0.7)


def _segment_materials(T: dict) -> Dict[int, int]:
    seg, mat = T["seg"], T["mat"]
    m = seg >= 0
    if not m.any():
        return {}
    s = seg[m].astype(np.int64)
    mm = mat[m].astype(np.int64)
    # the material that covers most of each segment
    key = s * 4096 + np.maximum(mm, 0)
    u, c = np.unique(key, return_counts=True)
    out: Dict[int, Tuple[int, int]] = {}
    for k, n in zip(u, c):
        si, mk = int(k // 4096), int(k % 4096)
        if si not in out or n > out[si][1]:
            out[si] = (mk, int(n))
    return {k: v[0] for k, v in out.items()}


def _ground_shadows(p, T, sc) -> list:
    """A soft shadow on the ground under each figure whose feet are on it (blocks catch their own cast shadow)."""
    from .ops import Fill
    out = []
    has_floor = any(el["type"] == "plane" or "floor" in el["name"] for el in DP.blocks_rec(p))
    if has_floor:
        return out
    cam = DP.camera_of(p)
    for el in DP.figures(p):
        J = {k: np.asarray(v, float) for k, v in el["j3d"].items()}
        feet = [J[f"heel-{s}"] for s in "rl"] + [J[f"foot-{s}"] for s in "rl"]
        low = min(f[1] for f in feet)
        if low > 0.12:
            continue
        c = np.mean([[f[0], 0.0, f[2]] for f in feet], 0)
        ang = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        rx = 0.2 + 0.5 * float(np.ptp([f[0] for f in feet]))
        rz = 0.1 + 0.5 * float(np.ptp([f[2] for f in feet]))
        ring = np.stack([c[0] + rx * np.cos(ang), np.zeros_like(ang), c[2] + rz * np.sin(ang)], 1)
        px, z = cam.project(ring)
        if (z <= 0.1).any():
            continue
        out.append(Fill(polygons=[px.astype(np.float32)], color=(60, 56, 80), feather=0.02 * max(p.W, p.H) / 3,
                        opacity=0.2))
    return out


# ============================================================================================ the face

def face_ops(p, T: dict, sc: dict, cam_r, s: float) -> Tuple[list, List[dict]]:
    """Flat-layer fills (eye whites, irises, pupils, catch-lights, mouth, tears, blush) and line specs (canvas px)
    for every figure with a face set."""
    from .ops import Fill
    from .draw_core import poly_mask
    fill_ops: list = []
    line_specs: List[dict] = []
    cam = DP.camera_of(p)
    depth = T["R"]["depth"]
    for el in DP.figures(p):
        fc = look_of(el).get("face")
        if not fc:
            continue
        rig = sc["heads"].get(el["id"])
        if rig is None:
            continue
        iris = _col(p, fc.get("eyes"), "#4a78c0")
        lash = dc.shadow_of(_col(p, (look_of(el).get("hair") or {}).get("color"), HAIR), 1.6)
        lash = tuple(int(v * 0.55) for v in lash)
        specs = FC.project(rig, cam, FC.features(fc.get("expression", "neutral"), iris, lash), depth, s)
        clips: Dict[str, np.ndarray] = {}
        for sp in specs:
            if sp.get("fill") is None:
                continue
            P = sp["px"]
            if sp["vis"].mean() < (0.25 if sp["kind"] == "tear" else 0.5):
                continue
            m = poly_mask(p.H, p.W, P)
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
                y0, y1 = float(P[:, 1].min()), float(P[:, 1].max())
                cx = float(P[:, 0].mean())
                fill_ops.append(Fill(polygons=[P.astype(np.float32)], color=dc.shadow_of(col, 1.2), color2=dc.light_of(col, 1.2),
                                     p0=(cx, y0), p1=(cx, y1), feather=0.6) if not sp.get("clip") else
                                _grad_mask(m, dc.shadow_of(col, 1.2), dc.light_of(col, 1.2), (cx, y0), (cx, y1)))
            elif sp["kind"] == "blush":
                fill_ops.append(Fill(polygons=[P.astype(np.float32)], color=col, feather=0.004 * max(p.W, p.H),
                                     opacity=float(sp.get("alpha", 0.4))))
            else:
                from .ops import mask_to_rle
                fill_ops.append(Fill(mask_rle=mask_to_rle(m), color=col, feather=0.6,
                                     opacity=0.85 if sp["kind"] == "tear" else 1.0))
        for sp in specs:
            if sp["weight"] <= 0:
                continue
            P, vis = sp["px"], sp["vis"]
            for run in T3._runs(vis, False):
                if len(run) >= 2:
                    line_specs.append({"subject": el["subject"], "name": sp["name"], "px": P[run],
                                       "closed": bool(sp["closed"] and vis.all()), "weight": sp["weight"],
                                       "taper": sp["taper"], "rgb": sp.get("rgb")})
    return fill_ops, line_specs


def _grad_mask(m, c1, c2, p0, p1):
    from .ops import Fill, mask_to_rle
    return Fill(mask_rle=mask_to_rle(m), color=tuple(int(v) for v in c1), color2=tuple(int(v) for v in c2), p0=p0,
                p1=p1, feather=0.6)


# ============================================================================================ layer bookkeeping

def _load_npz(p, name):
    f = p.root / R3D / f"{name}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return {k: z[k] for k in z.files}


def _restore(p, kind: str, arr: np.ndarray) -> np.ndarray:
    """The layer as it was before the last render painted it (the render's pixels put back to what was under)."""
    z = _load_npz(p, kind)
    if z is None:
        return arr
    x0, y0, x1, y1 = (int(v) for v in z["box"])
    if arr.shape[:2] != tuple(int(v) for v in z["shape"]):
        return arr
    cov = np.unpackbits(z["cov"])[: (y1 - y0) * (x1 - x0)].reshape(y1 - y0, x1 - x0).astype(bool)
    under = z["under"].astype(np.float32) / 255.0
    sub = arr[y0:y1, x0:x1]
    sub[cov] = under[cov]
    return arr


def _snapshot(p, kind: str, before: np.ndarray, after: np.ndarray) -> None:
    cov = np.abs(after - before).max(axis=2) > (0.5 / 255.0)
    (p.root / R3D).mkdir(parents=True, exist_ok=True)
    f = p.root / R3D / f"{kind}.npz"
    if not cov.any():
        if f.exists():
            f.unlink()
        return
    ys, xs = np.nonzero(cov)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    c = cov[y0:y1, x0:x1]
    under = (np.clip(before[y0:y1, x0:x1], 0, 1) * 255 + 0.5).astype(np.uint8)
    under[~c] = 0
    np.savez_compressed(f, box=np.array([x0, y0, x1, y1]), shape=np.array(before.shape[:2]),
                        cov=np.packbits(c.ravel()), under=under)


def band_files(p, layer: str) -> List[str]:
    """Files a band on ``layer`` changes besides its own when a render is on the layer (its under snapshot)."""
    if layer != "flat" or not (p.root / R3D / "flat.npz").exists():
        return []
    return [f"{R3D}/flat.npz"]


def band_behind_render(p, layer: str, pre: np.ndarray, ops: list) -> None:
    """After a band was painted on the flat layer: inside the render's pixels the layer shows the render again (the
    figures and blocks stand in front of the background), and the band is painted, whole, into what lies under the
    render, so it shows where a figure moves away."""
    from . import brush
    z = _load_npz(p, "flat")
    if z is None or layer != "flat":
        return
    x0, y0, x1, y1 = (int(v) for v in z["box"])
    if tuple(int(v) for v in z["shape"]) != (p.H, p.W):
        return
    cov = np.unpackbits(z["cov"])[: (y1 - y0) * (x1 - x0)].reshape(y1 - y0, x1 - x0).astype(bool)
    cur = p.layer("flat").copy()
    sub = cur[y0:y1, x0:x1]
    sub[cov] = pre[y0:y1, x0:x1][cov]
    p._write_layer_file("flat", cur)
    tmp = np.zeros((p.H, p.W, 4), np.float32)
    under = z["under"].astype(np.float32) / 255.0
    tmp[y0:y1, x0:x1][cov] = under[cov]
    brush.apply_ops(tmp, ops)
    new = (np.clip(tmp[y0:y1, x0:x1], 0, 1) * 255 + 0.5).astype(np.uint8)
    new[~cov] = 0
    np.savez_compressed(p.root / R3D / "flat.npz", box=z["box"], shape=z["shape"], cov=z["cov"], under=new)


def _touch_layer(p) -> None:
    names = [e["name"] for e in p.state["layers"]]
    if "touch" in names:
        return
    at = names.index("line") + 1 if "line" in names else len(names)
    p.state["layers"].insert(at, {"name": "touch", "mode": "normal", "opacity": 1.0, "visible": True, "clip": False,
                                  "alpha_lock": False, "empty": True})
    p._write_layer_file("touch", None)


def _files(p) -> List[str]:
    out = ["layers/flat.png", "layers/shade.png", "layers/line.png", "draw/line_base.png", "draw/line_erase.png",
           f"{R3D}/flat.npz", f"{R3D}/shade.npz", f"{R3D}/line.npz", "draw/scene3d.npz"]
    out += [f"regions/{n}.png" for n in p.state.get("regions", {})]
    return out


# ============================================================================================ render

def do_render(ctx, line: Optional[float] = None, bands: Optional[int] = None, light_words: Optional[str] = None,
              who: Optional[str] = None, label: Optional[str] = None):
    """One undoable step: the toon render of the whole 3D scene onto the line, flat and shade layers."""
    from . import brush, cli, cli_draw, draw_line, workflow_draw
    from .ops import ops_to_jsonable
    p = ctx.project
    d = dc.dstate(p)
    if not DP.figures(p) and not DP.blocks_rec(p):
        raise UsageError("nothing in 3D to render: pose a figure or add a block first",
                         "pixelcanvas pose girl --preset stand")
    rs = render_state(p)
    line = float(line if line is not None else rs.get("line", DEFAULT_LINE))
    bands = int(bands if bands is not None else rs.get("bands", 2))
    if light_words is None:
        light_words = rs.get("light_words")
    if light_words in ("from-plan", "plan", ""):
        light_words = None
    lv = _light(p, {"light_words": light_words})
    t0 = time.time()
    rep = cli_draw.DReport(mutated=True)
    before = dc.composite_u8(p, "view")
    _touch_layer(p)
    (p.root / R3D).mkdir(parents=True, exist_ok=True)
    regions_before = set(p.state.get("regions", {}))
    T, sc, cam_r, s = toon_scene(p, bands=bands, light=lv)
    t1 = time.time()
    line_px = dc.dpx(p, line)
    L = render_layers(p, T, sc, cam_r, s, line_px, bands)
    f_ops, f_lines = face_ops(p, T, sc, cam_r, s)
    t2 = time.time()
    new_regions = set(L["regions"]) | set(L["aliases"])
    files = _files(p) + [f"regions/{n}.png" for n in new_regions if n not in regions_before]
    step = p.begin_step(list(dict.fromkeys(files)))
    try:
        # the line layer: marks made on it by other commands since its last write go to its base first
        note = dc.owned_fold(p, "line", d["lines"], draw_line.render_ops)
        base_f = p.root / "draw" / "line_base.png"
        base = dc._load_rgba(base_f)
        base = np.zeros((p.H, p.W, 4), np.float32) if base is None else base
        base = _restore(p, "line", base)
        b0 = base.copy()
        brush.apply_ops(base, L["line"])
        _snapshot(p, "line", b0, base)
        dc._save_rgba(base_f, base)
        # face lines: line records owned by the render (made again every time)
        d["lines"] = [r for r in d["lines"] if not r.get("face3d")]
        for fl in f_lines:
            P = fl["px"]
            if len(P) < 2 or dc.path_length(P) < 3:
                continue
            rec = draw_line.make_record(p, P, dc.px2u(p, P[[0, -1]]), weight=fl["weight"] * line / DEFAULT_LINE,
                                        taper=fl["taper"], color="line", closed=fl["closed"],
                                        src=f"face {fl['name']}")
            rec["face3d"] = fl["subject"]
            rec["even"] = True
            if fl.get("rgb"):
                rec["auto_rgb"] = [int(v) for v in fl["rgb"]]
                rec["rgb_fixed"] = True         # a tear keeps its blue: line recolouring leaves it
            d["lines"].append(rec)
        dc.owned_write(p, "line", d["lines"], draw_line.render_ops)
        # flat and shade
        for kind, ops in (("flat", L["flat"] + f_ops), ("shade", L["shade"])):
            arr = p.layer(kind).copy()
            arr = _restore(p, kind, arr)
            a0 = arr.copy()
            if ops:
                brush.apply_ops(arr, ops)
            _snapshot(p, kind, a0, arr)
            p._write_layer_file(kind, arr)
        # regions, parts and swatches: each material is a coloured area the later tools work on
        fills = [f for f in d["fills"] if not f.get("r3d")]
        for rn, m in L["regions"].items():
            from .regions import clean_name
            rn2 = clean_name(rn)
            p.save_region_mask(rn2, m)
            box = coords.px_box_to_units(coords.mask_bbox_px(m), p.W, p.H)
            info = p.state["regions"].setdefault(rn2, {"focus": 1})
            info.update({"box": list(box), "area": round(float(m.mean()), 4), "how": "3D render"})
            ys, xs = np.nonzero(m[:: 4, :: 4])
            if len(xs):
                i = len(xs) // 2
                fills.append({"pt": [int(xs[i] * 4), int(ys[i] * 4)], "part": rn2, "swatch": rn2, "r3d": True})
        d["fills"] = fills
        for rn, m in L["aliases"].items():
            p.save_region_mask(rn, m)
            info = p.state["regions"].setdefault(rn, {"focus": 3})
            info.update({"box": list(coords.px_box_to_units(coords.mask_bbox_px(m), p.W, p.H)),
                         "area": round(float(m.mean()), 4), "how": "3D render (the head's skin)"})
        _register_parts(p, sc, L)
        d["render"] = {"key": _scene_key(p, line, bands, lv), "line": line, "bands": bands,
                       "light_words": light_words, "regions": sorted(L["regions"]),
                       "subjects": [el["subject"] for el in DP.figures(p)]}
        all_ops = L["line"] + L["flat"] + f_ops + L["shade"]
        ops_file = p.save_ops(step, {"cmd": ctx.cmdline, "layer": "line,flat,shade", "ops": ops_to_jsonable(all_ops)})
        ticked = workflow_draw.refresh(p)
        p.end_step(step, {"cmd": ctx.cmdline, "layer": "line,flat,shade", "ops": ops_file, "n_ops": len(all_ops),
                          "bbox": None})
    except Exception:
        p.abort_step(step)
        raise
    t3 = time.time()
    rep.step = step
    figs = DP.figures(p)
    rep.add(f"step {step}  {label or 'render'}  (layers line, flat, shade)  {len(all_ops)} marks: "
            f"{L['n_strokes']} lines, {len(L['flat'])} flat colours, {len(L['shade'])} shade bands, "
            f"{len(f_ops)} face fills  ({t3 - t0:.1f} s: 3D render {t1 - t0:.1f}, lines and fills "
            f"{t2 - t1:.1f}, layers {t3 - t2:.1f})")
    for el in figs:
        lk = look_of(el)
        g = ", ".join(lk["garments"]) or "no clothes (pixelcanvas dress)"
        hr = (lk.get("hair") or {}).get("style") or "no hair (pixelcanvas hair3d)"
        fc = (lk.get("face") or {}).get("expression") or "no face yet (pixelcanvas face " + el["subject"] + ")"
        rep.add(f"{el['subject']}: {el['rig'].get('preset')}; {g}; hair {hr}; face {fc}")
    rep.add("regions: " + ", ".join(sorted(L["regions"]))[:420])
    rep.add(f"line {line:g} px (delivered), {bands} bands, light from "
            f"{light_words or dc.light_words(d['light'].get('deg', 135))}")
    rep.add("re-pose, dress, hair3d, camera, block and prop render again by themselves; marks on the touch layer "
            "(--layer touch) and line records (face) stay through a render")
    if note:
        rep.add("note: " + note)
    if ticked:
        rep.add(f"todo ticked: {', '.join(map(str, ticked))}")
    if not ctx.batch:
        cli_draw.show_before_after(ctx, rep, before, dc.composite_u8(p, "view"), (0, 0, p.W, p.H), step,
                                   "the toon render: line, flat and shade")
    cli_draw._tail(ctx, rep)
    rep.summary = f"render {len(all_ops)} marks"
    return rep


def _register_parts(p, sc: dict, L: dict) -> None:
    """Each rendered material becomes a plan part (focus 2 for hair and clothes) with its colour as a swatch, so the
    shade, detail and polish tools know what it is and its colour."""
    d = dc.dstate(p)
    owners = {el["id"]: el["subject"] for el in DP.figures(p)}
    for key, rn in sc["regions"].items():
        if key.endswith("/in") or rn not in L["regions"]:
            continue
        own, _, mat = key.partition(":")
        rgb = L["colours"].get(key)
        if rgb is None:
            continue
        subj = owners.get(own)
        if subj is None:
            d["palette"]["swatches"][rn] = dc.to_hex(rgb)
            continue
        if rn not in d["parts"]:
            m = L["regions"][rn]
            box = list(coords.px_box_to_units(coords.mask_bbox_px(m), p.W, p.H))
            focus = 2 if subj and mat in ("hair",) + tuple(F3.GARMENTS) else 1
            d["parts"][rn] = {"of": subj or rn, "box": box, "focus": focus, "swatch": rn, "r3d": True}
            if subj is None and rn not in d["subjects"]:
                pass
        if d["parts"][rn].get("r3d") or not d["parts"][rn].get("swatch"):
            d["parts"][rn]["swatch"] = rn
        d["palette"]["swatches"][rn] = dc.to_hex(rgb)


# ============================================================================================ commands

def _preview(ctx, rep, title: str) -> None:
    """A quick toon preview (small, no shadows) of the 3D scene as the command's view."""
    from . import brush, cli, cli_draw
    from .ops import Stroke
    p = ctx.project
    if ctx.batch:
        return
    try:
        T, sc, cam, s = toon_scene(p, cap=700, shadows=False)
        H, W = T["mat"].shape
        img = p.composite_u8("view")
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
        for k, m in enumerate(T["mats"]):
            sel = T["mat"] == k
            rgb = np.array(sc["colours"].get(m, (128, 128, 128)), np.float32)
            img[sel & (T["tone"] == 1)] = rgb
            img[sel & (T["tone"] >= 2)] = np.array(dc.shadow_of(tuple(int(v) for v in rgb)), np.float32)
        seg_mat = _segment_materials(T)
        d = T["R"]["depth"]
        dref = float(np.median(d[np.isfinite(d)])) if np.isfinite(d).any() else 3.0
        st = T3.lines(T["R"], sc["prims"], T["seg"], T["table"], 1.0, 1.4,
                      lambda si: sc["colours"].get(T["mats"][seg_mat.get(si, 0)], (60, 60, 60)), dref)
        lay = np.zeros((H, W, 4), np.float32)
        brush.apply_ops(lay, [Stroke(pts=x["pts"], width=x["width"], color=x["rgb"], brush="liner") for x in st])
        a = lay[..., 3:4]
        img = img * (1 - a) + lay[..., :3] * 255 * a
        big = cv2.resize(np.clip(img, 0, 255).astype(np.uint8), (p.W, p.H), interpolation=cv2.INTER_LINEAR)
        cli_draw.show_one(ctx, rep, big, (0, 0, p.W, p.H), title, "a quick preview of the 3D figures (pixelcanvas "
                          "render draws them on the layers)", f"r3d-{cli._view_counter(p):04d}")
    except Exception:  # noqa: BLE001 - a preview never fails the command
        pass


def _after_change(ctx, rep, what: str):
    """After a change to the 3D scene: render again when a render exists, else say how."""
    p = ctx.project
    if render_state(p) and is_stale(p):
        r2 = do_render(ctx, label=f"render again ({what})")
        for ln in r2.lines:
            rep.add(ln)
        rep.view, rep.view_note = r2.view, r2.view_note
        if getattr(r2, "view_before", None):
            rep.view_before = r2.view_before
        rep.add("rendered again: two undo steps go back to before the change")
        rep.next = r2.next
        return rep
    return rep


def cmd_dress(ctx, args):
    from .cli import Report, Spec, parse
    from . import cli, cli_draw
    p = ctx.project
    if args and args[0].lower() in ("list", "pieces", "help"):
        rep = Report()
        rep.add(f"{len(F3.GARMENTS)} pieces (pixelcanvas dress <who> <piece>[,<piece>] [--color] [--length] [--fit] "
                f"[--sleeve] [--style]):")
        for k, v in F3.GARMENTS.items():
            rep.add(f"  {k}: {v}")
        rep.next = cli._finish_next(ctx)
        return rep
    a = parse(args, Spec(USAGE["dress"], ["subject", "pieces*"],
                         {"color": "value", "length": "value", "fit": "value", "sleeve": "value", "style": "value",
                          "trim": "value", "remove": "flag", "skin": "value"},
                         alias={"colour": "color", "sleeves": "sleeve", "off": "remove", "undress": "remove",
                                "type": "style"}, min_pos=1))
    el = figure_of(p, a["subject"], USAGE["dress"])
    lk = look_of(el)
    pieces = []
    for w in str(a.get("pieces") or "").replace(",", " ").split():
        g = F3.garment_key(w)
        if g is None:
            raise UsageError(f"unknown piece {w!r}; pieces: {', '.join(F3.GARMENTS)}", USAGE["dress"])
        pieces.append(g)
    if not pieces and not a.get("skin"):
        raise UsageError("say which piece: " + ", ".join(F3.GARMENTS), USAGE["dress"])
    for key, allowed in (("length", ("short", "knee", "long")), ("fit", ("tight", "loose")),
                         ("sleeve", ("none", "short", "long")),
                         ("style", ("pleated", "flared", "pencil", "open", "closed"))):
        if a.get(key) and str(a[key]).lower() not in allowed:
            raise UsageError(f"--{key} is one of {', '.join(allowed)}", USAGE["dress"])
    for key in ("color", "trim", "skin"):
        if a.get(key):
            from .draw_plan import swatch
            swatch(p, a[key], USAGE["dress"])
    lines: List[str] = []

    def mutate():
        if a.get("skin"):
            lk["skin"] = str(a["skin"])
            lines.append(f"{el['subject']}'s skin: {a['skin']}")
        for g in pieces:
            if a.get("remove"):
                lk["garments"].pop(g, None)
                lines.append(f"{el['subject']}: {g} taken off")
                continue
            o = dict(lk["garments"].get(g) or {})
            for key in ("color", "length", "fit", "sleeve", "style", "trim"):
                if a.get(key):
                    o[key] = str(a[key]).lower() if key not in ("color", "trim") else str(a[key])
            lk["garments"][g] = o
            lines.append(f"{el['subject']} wears {g}" + (f" ({', '.join(f'{k} {v}' for k, v in o.items())})" if o
                                                         else ""))
        lines.append(f"{el['subject']} now wears: {', '.join(lk['garments']) or 'nothing'}")
        lines.append("the clothes bend with every joint; a re-pose moves them")
        return lines
    rep, step = cli_draw.state_step(ctx, f"dress {el['subject']} {','.join(pieces)}", mutate)
    if render_state(p):
        return _finish(ctx, _after_change(ctx, rep, "dress"))
    _preview(ctx, rep, f"DRESSED: {el['subject']}")
    rep.next = cli._finish_next(ctx)
    return rep


def _finish(ctx, rep):
    from . import cli
    if not rep.next:
        rep.next = cli._finish_next(ctx)
    return rep


def cmd_hair3d(ctx, args):
    from .cli import Spec, parse
    from . import cli, cli_draw
    p = ctx.project
    a = parse(args, Spec(USAGE["hair3d"], ["subject"],
                         {"style": "value", "bangs": "value", "color": "value", "wind": "value", "bows": "value",
                          "ahoge": "value", "remove": "flag"},
                         alias={"colour": "color", "cut": "style", "fringe": "bangs", "bow": "bows"}, min_pos=1))
    el = figure_of(p, a["subject"], USAGE["hair3d"])
    lk = look_of(el)
    hair = dict(lk.get("hair") or {})
    if a.get("style"):
        st = str(a["style"]).lower().replace("twin-tails", "twintails").replace("pigtails", "twintails")
        if st not in F3.HAIR_STYLES:
            raise UsageError(f"--style is one of {', '.join(F3.HAIR_STYLES)}", USAGE["hair3d"])
        hair["style"] = st
    hair.setdefault("style", "medium")
    if a.get("bangs"):
        b = str(a["bangs"]).lower()
        if b not in F3.BANGS:
            raise UsageError(f"--bangs is one of {', '.join(F3.BANGS)}", USAGE["hair3d"])
        hair["bangs"] = b
    from .draw_plan import swatch
    for key in ("color", "bows"):
        if a.get(key):
            swatch(p, a[key], USAGE["hair3d"])
            hair[key] = str(a[key])
    if a.get("wind"):
        w = str(a["wind"]).lower()
        if w in ("auto", "none", "off"):
            hair.pop("wind", None)
        else:
            nums = [v for v, _, _ in coords._numbers(w)]
            if len(nums) != 2:
                raise UsageError("--wind dx,dy (picture directions, about -1..1) or auto", USAGE["hair3d"])
            hair["wind"] = [float(nums[0]), float(nums[1])]
    if a.get("ahoge"):
        hair["ahoge"] = str(a["ahoge"]).lower() not in ("no", "off", "false", "0", "none")

    def mutate():
        if a.get("remove"):
            lk.pop("hair", None)
            return [f"{el['subject']}: hair removed"]
        lk["hair"] = hair
        out = [f"{el['subject']}'s hair: {hair['style']}, bangs {hair.get('bangs', 'default')}, "
               f"{hair.get('color', 'dark brown')}" + (f", bows {hair['bows']}" if hair.get("bows") else "")]
        mv, fl = F3.motion_of(el["rig"], DP._fig_state(el)[2])
        out.append("the locks hang with gravity" + (", trail behind the motion" if np.linalg.norm(mv) > 0.2 else "")
                   + (", float (she is lifted)" if fl else "") + (", blown by --wind" if hair.get("wind") else ""))
        return out
    rep, step = cli_draw.state_step(ctx, f"hair3d {el['subject']} {hair['style']}", mutate)
    if render_state(p):
        return _finish(ctx, _after_change(ctx, rep, "hair3d"))
    _preview(ctx, rep, f"HAIR: {el['subject']}")
    rep.next = cli._finish_next(ctx)
    return rep


def cmd_render(ctx, args):
    from .cli import Spec, parse
    p = ctx.project
    a = parse(args, Spec(USAGE["render"], ["who"], {"line": "value", "bands": "value", "light": "value"},
                         alias={"weight": "line", "levels": "bands"}, min_pos=0))
    who = str(a.get("who") or "all").lower()
    if who not in ("all", "") and not any(f["subject"] == who for f in DP.figures(p)):
        figure_of(p, who, USAGE["render"])
    line = coords.parse_float(a["line"], "line", 0.3, 20, ctx.notes, USAGE["render"]) if a.get("line") else None
    bands = coords.parse_int(a["bands"], "bands", 2, 3, ctx.notes) if a.get("bands") else None
    lw = str(a["light"]).lower() if a.get("light") else None
    if lw and lw not in ("from-plan", "plan"):
        dc.parse_light(lw)
    return do_render(ctx, line, bands, lw, who, label=f"render {who}")


def wrap_face(orig):
    """``face <who>`` on a posed figure: the 3D face (rendered with the figure); anything else: the 2D face."""
    def run(ctx, args):
        from .cli import Spec, parse
        from . import cli, cli_draw
        p = ctx.project
        first = str(args[0]).lower() if args else ""
        figs = {f["subject"]: f for f in DP.figures(p)}
        target = None
        if first in figs:
            target = figs[first]
        elif not args or first.startswith("--"):
            if len(figs) == 1:
                target = next(iter(figs.values()))
        else:
            from . import draw_sketch
            try:
                h = draw_sketch.find(p, first)
                if h.get("subject") in figs:
                    target = figs[h["subject"]]
            except UsageError:
                target = None
        if target is None:
            return orig(ctx, args)
        rest = args[1:] if first in figs or (first and not first.startswith("--")) else args
        a = parse(rest, Spec("pixelcanvas face girl --expression smile", [], {"expression": "value", "eyes": "value",
                                                                             "kind": "value", "weight": "value",
                                                                             "keep-eyes": "flag"},
                             alias={"expr": "expression", "mood": "expression", "iris": "eyes", "eye": "eyes"}))
        e = FC.expression_key(a.get("expression") or (look_of(target).get("face") or {}).get("expression") or
                              "neutral")
        if e is None:
            raise UsageError(f"unknown expression {a.get('expression')!r}: {', '.join(FC.EXPRESSIONS)}",
                             "pixelcanvas face girl --expression crying")
        fc = dict(look_of(target).get("face") or {})
        fc["expression"] = e
        if a.get("eyes"):
            from .draw_plan import swatch
            swatch(p, a["eyes"], "pixelcanvas face girl --eyes blue")
            fc["eyes"] = str(a["eyes"])

        def mutate():
            look_of(target)["face"] = fc
            return [f"{target['subject']}'s face: {e}" + (f", eyes {fc['eyes']}" if fc.get("eyes") else "") +
                    " (on the 3D head: it turns with the head and is drawn again by every render)"]
        rep, step = cli_draw.state_step(ctx, f"face {target['subject']} {e}", mutate)
        if render_state(p):
            rep = _after_change(ctx, rep, "face")
        else:
            rep.add("the face is drawn with the figure: pixelcanvas render")
            rep.next = "pixelcanvas render"
        return _finish(ctx, rep)
    return run


def wrap_change(orig):
    """pose / camera / block / prop: after the change, render again when a render exists."""
    def run(ctx, args):
        rep = orig(ctx, args)
        try:
            p = ctx.project
            if getattr(rep, "mutated", False) and render_state(p) and is_stale(p):
                return _after_change(ctx, rep, (ctx.cmdline or "change").split()[0] if ctx.cmdline else "change")
        except UsageError:
            raise
        return rep
    return run


COMMANDS = {"dress": cmd_dress, "hair3d": cmd_hair3d, "render": cmd_render}
WRAP = {"face": wrap_face, "pose": wrap_change, "camera": wrap_change, "block": wrap_change, "prop": wrap_change}
