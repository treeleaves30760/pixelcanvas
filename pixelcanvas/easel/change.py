"""Changes: a redraw that differs from its reference ("redraw it, but with green hair").

The model states each change (``easel change add ...``) and makes it with a change command (``easel recolor ...``).
A change edits the *target*: the picture the traced stages copy and every score measures against. The original
reference is kept as ``orig.png`` (delivered size); the target is rebuilt from it by replaying the applied changes in
order, and written where the stages read their reference (``ref.png``, and ``ref_deliver.png`` on a working canvas).
A project without stated changes never creates ``orig.png`` and works exactly as before.

state["changes"] = {"decided": bool, "none": bool, "items": [{"id", "kind", "region", "cmd", "params", "check"}]}
state["stale"]   = [[stage, region], ...]  regions to redraw because the target changed after their stage painted
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from . import change_ops as co
from .coords import UsageError

# kind -> where it runs, whether it edits the target, one line for the todo
KINDS: Dict[str, Dict] = {
    "canvas": {"stage": "change", "target": False, "phase": 0,
               "words": "a new canvas size with the reference placed in it"},
    "recolor": {"stage": "change", "target": True, "phase": 1,
                "words": "recolour a region with its shading kept"},
    "palette": {"stage": "change", "target": True, "phase": 2,
                "words": "map the picture to a few colours by value, with accent regions"},
    "graphic": {"stage": "change", "target": True, "phase": 3,
                "words": "a flat shape (disc, rectangle, band) on the ground, behind the figure"},
    "cutout": {"stage": "change", "target": True, "phase": 4,
               "words": "cut the figure out of its ground (it goes over a new background)"},
    "background": {"stage": "change", "target": True, "phase": 5,
                   "words": "a new background behind the cut-out figure"},
    "light": {"stage": "change", "target": True, "phase": 6,
              "words": "light the figure by its new setting (key colour, rim light)"},
    "shadow": {"stage": "change", "target": True, "phase": 7,
               "words": "a contact shadow on the ground under the figure"},
    "text": {"stage": "finish", "target": False, "phase": 9,
             "words": "lettering over the finished picture"},
    "shape": {"stage": "finish", "target": False, "phase": 9,
              "words": "a crisp shape over the finished picture (a frame)"},
}
VALUE_ORDER_MIN = 0.90  # correlation of luminance before / after a palette map (outside accents)
ACCENT_EDGE_MIN = 0.85  # share of an accent region's edge that lies on line art
TARGET_FILES = ["ref.png", "ref_deliver.png", "orig.png", "background_src.png", "cutout.png"]
LOST_MAX = 0.10         # share of the figure's named regions the cut-out may leave out
HALO_MAX = 0.08         # share of the ring inside the silhouette still of the old ground's colour
BG_FORMS = ('easel background --from <free project folder or picture file>   |   '
            'easel background --gradient "#a,#b" --angle 90   |   easel background --color "#hex"')
STAGE_LAYERS = ("line", "flat", "shade", "paint", "detail")
HUE_TOL = 15.0          # degrees between the asked hue and the hue reached
SMALL_REGION = 0.02     # regions under this share of the canvas never give pixels to another region's refine
SHADING_MAX = 2.5       # mean drift (L units) of each pixel's lightness offset from the region's median


# ----------------------------------------------------------------------------------------------- state

def cstate(state: dict) -> dict:
    return state.get("changes") or {}


def items(state: dict) -> List[dict]:
    return list(cstate(state).get("items") or [])


def active(state: dict) -> bool:
    return bool(items(state))


def needs_decision(state: dict) -> bool:
    """A project made from a reference with a request: the model must state the request's changes (or none)."""
    return bool(state.get("has_ref")) and bool(str(state.get("request") or "").strip()) and \
        state.get("mode") != "hand" and not cstate(state).get("decided")


def describe(it: dict) -> str:
    p = it.get("params") or {}
    if it["kind"] == "recolor":
        return f"recolor {it['region']} to {p.get('to') or it.get('to') or '?'}"
    return it["cmd"].replace("easel ", "", 1)


# ----------------------------------------------------------------------------------------------- the target

def orig_d(p) -> np.ndarray:
    """The original reference at the delivered size (saved as orig.png the first time a change is made)."""
    from .project import load_image_rgb, save_png
    f = p.root / "orig.png"
    if f.exists():
        return load_image_rgb(f)
    img = np.asarray(p.ref_deliver).copy()
    save_png(f, img)
    return img


def orig_work(p) -> np.ndarray:
    """The original reference at the working size (what ref.png held before any change)."""
    if not (p.root / "orig.png").exists():
        return np.asarray(p.ref)
    return working(p, orig_d(p))


def working(p, img_d: np.ndarray) -> np.ndarray:
    import cv2
    if p.scale == 1 and img_d.shape[1] == p.W and img_d.shape[0] == p.H:
        return img_d
    big = cv2.resize(img_d, (p.W, p.H), interpolation=cv2.INTER_CUBIC)     # as Project.set_work_scale
    return np.clip(big, 0, 255).astype(np.uint8)


def region_d(p, name: str) -> np.ndarray:
    """A region at the delivered size; a delivered pixel a quarter covered belongs to it (thin strands survive)."""
    import cv2
    m = p.region_mask(name)
    if p.scale == 1:
        return m
    dw, dh = p.deliver_size()
    return cv2.resize(m.astype(np.float32), (dw, dh), interpolation=cv2.INTER_AREA) >= 0.25


def is_rest(entry: dict) -> bool:
    return str(entry.get("how") or "").startswith("rest of the canvas")


def small_others(p, name: str) -> np.ndarray:
    """The small parts (under SMALL_REGION of the canvas) other than ``name``, at the delivered size: a pixel in an
    eye stays the eye's even when the hair's region covers it too."""
    dw, dh = p.deliver_size()
    out = np.zeros((dh, dw), bool)
    for r, e in p.state["regions"].items():
        if r != name and not is_rest(e) and float(e.get("area", 1.0)) < SMALL_REGION:
            out |= region_d(p, r)
    return out


def protected(p, name: str, img_d: np.ndarray, band: Optional[float] = None) -> np.ndarray:
    """Pixels (delivered size) that region ``name`` may not take by its colour: those in another named region (not
    the rest of the canvas) that are of that region's own colour, or not close to this region's main hue. A hair
    strand lying across the face moves to the hair; the face keeps its blush and its marks."""
    L, C, h = co.lch(co.to_lab(img_d))
    mine = co.colour_info(img_d, region_d(p, name), band)
    out = np.zeros(img_d.shape[:2], bool)
    for r, e in p.state["regions"].items():
        if r == name or is_rest(e):
            continue
        m = region_d(p, r)
        if mine["grey"] or float(e.get("area", 1.0)) < SMALL_REGION:
            out |= m              # a small part (an eye, a mouth, a badge) keeps every pixel it has
            continue
        info = co.colour_info(img_d, m)
        d_mine = np.abs(co.angdiff(h, mine["from_h"]))
        # that region's own colour: nearer its main hue than this region's (a navy top keeps its navy; a purple hair
        # shadow the marks put in the top goes to the hair)
        own = (C <= 8.0) if info["grey"] else (np.abs(co.angdiff(h, info["from_h"])) <= d_mine)
        close = d_mine <= 0.57 * mine["band"]
        out |= m & (own | ~close)
    return out


def held_islands(p, name: str, img_d: np.ndarray, mask: np.ndarray, band: Optional[float] = None) -> np.ndarray:
    """Islands of this region's colour that a small region's rough outline holds while this region surrounds them (a
    sliver of hair inside the outline of a heart beside it): they go with the recolour. A small part bordered by its
    own pixels (a pupil inside its iris) stays the small part's."""
    from scipy import ndimage as ndi
    out = np.zeros(mask.shape, bool)
    small = small_others(p, name)
    if not small.any() or not mask.any():
        return out
    info = co.colour_info(img_d, mask, band)
    if info["grey"]:
        return out
    mine = co.band_mask(img_d, info["from_h"], info["band"], 6.0, info["L_lo"], info["L_hi"])
    parts = [region_d(p, r) for r, e in p.state["regions"].items()
             if r != name and not is_rest(e) and float(e.get("area", 1.0)) < SMALL_REGION]
    lab, n = ndi.label(small & mine & ~mask)
    for i in range(1, n + 1):
        c = lab == i
        ring = ndi.binary_dilation(c, iterations=2) & ~c
        if not ring.any() or (ring & mask).sum() < 0.6 * ring.sum():
            continue
        # a stray piece of the small part's outline, not the part itself (a badge laid on the hair keeps its pixels)
        if all(c.sum() < 0.5 * m.sum() for m in parts if (m & c).any()):
            out |= c
    return out


def sizes(p) -> Tuple[int, int]:
    dw, dh = p.deliver_size()
    reach = max(3, int(round(0.02 * max(dw, dh))))
    min_px = max(12, int(round(0.00002 * dw * dh)))
    return reach, min_px


def apply_item(p, img_d: np.ndarray, it: dict, orig: Optional[np.ndarray] = None,
               taken: Optional[np.ndarray] = None, scene: Optional[dict] = None
               ) -> Tuple[np.ndarray, Optional[dict], np.ndarray]:
    """One change on the delivered-size picture: (new picture, its measured check, the pixels it edited). A pixel
    belongs to one change: ``taken`` (edited by an earlier change) is left alone. Stray patches are measured on the
    original (``orig``), as  easel region refine  takes them."""
    par = it.get("params") or {}
    if it["kind"] == "recolor":
        base0 = img_d if orig is None else orig
        # a pixel the region shares with another one stays that region's when it is of its colour (a face under a
        # rough hair outline); small parts keep all theirs
        mask = region_d(p, it["region"]) & ~protected(p, it["region"], base0, par.get("band"))
        mask |= held_islands(p, it["region"], base0, mask, par.get("band"))
        if taken is not None:
            mask = mask & ~taken
        rgb = tuple(par["rgb"])
        band = par.get("band")
        out, info = co.recolor(img_d, mask, rgb, chroma=par.get("chroma", 1.0), light=par.get("light", 0.0),
                               band=band)
        reach, min_px = sizes(p)
        c = co.recolor_checks(img_d, out, mask, info, reach, min_px, with_stray=False)
        base = img_d if orig is None else orig
        full = region_d(p, it["region"])
        stray = co.stray_patches(base, full, co.colour_info(base, full, band), reach, min_px,
                                 protected(p, it["region"], base, band))
        check = {"from_h": round(info["from_h"], 1), "to_h": round(info["to_h"], 1),
                 "hue_after": round(c["hue_after"], 1), "hue_off": round(c["hue_off"], 1),
                 "shading": round(c["shading"], 2), "stray": stray["n"], "stray_px": stray["px"],
                 "stray_boxes": [list(map(int, b)) for b in stray["boxes"][:6]], "islands": c["islands"],
                 "from_L": round(info["from_L"], 1), "to_L": round(info["to_L"], 1),
                 "L_after": round(c["L_after"], 1), "from_C": round(info["from_C"], 1),
                 "to_C": round(info["to_C"], 1), "edited": round(c["edited_share"], 4), "grey": info["grey"]}
        return out, check, info["edited"]
    if it["kind"] == "palette":
        acc = [(region_d(p, r), tuple(c)) for r, c in par.get("accents", []) if r in p.state["regions"]]
        g = ground_mask(p, img_d)
        light_at = None
        if g.any():
            light_at = float(np.median((img_d[g].astype(np.float32) / 255.0) @ np.array([0.299, 0.587, 0.114])))
        out, info = co.palette_map(img_d, [tuple(c) for c in par["rgbs"]], acc, gamma=par.get("gamma", 1.1),
                                   light_at=light_at)
        line = co.line_barrier(orig if orig is not None else img_d)
        edges = {}
        for r, _c in par.get("accents", []):
            if r in p.state["regions"]:
                edges[r] = round(co.edge_on_lines(region_d(p, r), line), 3)
        refined = [r for r, _c in par.get("accents", []) if r in p.state["regions"] and
                   "refined by lines" in str(p.state["regions"][r].get("how") or "")]
        check = {"value_order": round(info["value_order"], 3), "colours": info["colours"], "accent_edge": edges,
                 "lines_refined": refined,
                 "accent_share": {r: round(float(region_d(p, r).mean()), 4) for r, _c in par.get("accents", [])
                                  if r in p.state["regions"]}}
        return out, check, np.zeros(img_d.shape[:2], bool)
    if it["kind"] == "graphic":
        ground = ground_mask(p, img_d)
        dw, dh = img_d.shape[1], img_d.shape[0]
        bp = [par["box"][0] * dw / 1000.0, par["box"][1] * dh / 1000.0, par["box"][2] * dw / 1000.0,
              par["box"][3] * dh / 1000.0]
        out, info = co.graphic(img_d, par["shape"], bp, tuple(par["rgb"]),
                               ground if par.get("behind", True) else None, width=par.get("width", 0.0))
        check = {"drawn_px": info["drawn"], "box_share": round(info["box_share"], 3),
                 "behind_px": info["held_back"]}
        return out, check, np.zeros(img_d.shape[:2], bool)
    if it["kind"] in ("cutout", "background", "light", "shadow"):
        return apply_scene(p, img_d, it, scene if scene is not None else {})
    raise UsageError(f"unknown change kind {it['kind']!r}", "easel change list")


def _named(p, names) -> np.ndarray:
    dw, dh = p.deliver_size()
    out = np.zeros((dh, dw), bool)
    for r in names or []:
        if r in p.state["regions"]:
            out |= region_d(p, r)
    return out


def _rest_d(p) -> np.ndarray:
    dw, dh = p.deliver_size()
    out = np.zeros((dh, dw), bool)
    for r, e in p.state["regions"].items():
        if is_rest(e):
            out |= region_d(p, r)
    return out


def background_picture(p, par: dict, W: int, H: int) -> Tuple[np.ndarray, float]:
    from . import change_scene as cs
    from .project import load_image_rgb
    if par["source"] in ("project", "file"):
        f = p.root / "background_src.png"
        if not f.exists():
            raise UsageError("the background picture is missing from the project; set it again",
                             "easel background --from lake.easel")
        return cs.cover(load_image_rgb(f), W, H)
    if par["source"] == "gradient":
        return cs.gradient(W, H, par["rgbs"][0], par["rgbs"][1], par.get("angle", 90.0)), 0.0
    out = np.empty((H, W, 3), np.uint8)
    out[:] = np.asarray(par["rgbs"][0], np.uint8)
    return out, 0.0


def apply_scene(p, img_d: np.ndarray, it: dict, scene: dict) -> Tuple[np.ndarray, Optional[dict], np.ndarray]:
    """The setting changes, in order: the cut-out (the figure's alpha; the picture unchanged), the background behind
    it, the light on the figure, the contact shadow on the ground. ``scene`` carries fig / alpha / bg between them."""
    from . import change_scene as cs
    from .project import save_png
    par = it.get("params") or {}
    H, W = img_d.shape[:2]
    none = np.zeros((H, W), bool)
    if it["kind"] == "cutout":
        hint = _named(p, par.get("regions"))
        plus, minus = _named(p, par.get("plus")), _named(p, par.get("minus"))
        rest = _rest_d(p) & ~hint
        white = None
        gl = cs.ground_colour(img_d, rest)
        mode = par.get("ground", "auto")
        if mode == "white" or (mode == "auto" and gl[0] > 88 and float(np.hypot(gl[1], gl[2])) < 8):
            white = ground_mask(p, img_d) & ~plus
        alpha, info = cs.cutout(img_d, hint, plus, minus, white, corners=par.get("corners", True))
        fig, fi = cs.figure_colours(img_d, alpha, gl, hint)
        spots = cs.edge_spots(img_d, alpha, info["raw"], 3)
        per = {r: round(float((region_d(p, r) & ~info["solid"]).sum() / max(1, region_d(p, r).sum())), 3)
               for r in par.get("regions") or [] if r in p.state["regions"]}
        scene.update({"alpha": alpha, "fig": fig, "orig": img_d, "ground_lab": gl, "subject": par.get("subject"),
                      "edge_boxes": [list(map(int, s["box"])) for s in spots]})
        save_png(p.root / "cutout.png", np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8))
        check = {"mode": info["mode"], "share": round(info["share"], 4), "lost": round(info["lost"], 4),
                 "lost_by": per, "pockets": info["pockets"], "smooth_px": info["smooth_px"],
                 "ring_ground": fi["ring_ground"],
                 "spots": [{"box": list(map(int, s["box"])), "contrast": s["contrast"], "moved": s["moved"],
                            "at": _units(s["box"], W, H)} for s in spots],
                 "edge_boxes": [list(map(int, s["box"])) for s in spots]}
        return img_d, check, none
    if "alpha" not in scene:
        return img_d, {"no_cutout": True}, none
    alpha, fig = scene["alpha"], scene["fig"]
    if it["kind"] == "background":
        bg, crop = background_picture(p, par, W, H)
        scene["bg"] = bg
        out = cs.composite(bg, fig, alpha)
        h = cs.halo(out, bg, alpha, scene["ground_lab"])
        side, by = cs.bright_side(bg, alpha)
        check = {"source": par["source"], "from": par.get("from"), "crop": round(crop, 3),
                 "shows": round(float(1 - alpha.mean()), 3), "halo": h["halo"], "bright_side": side,
                 "bright_by": round(by, 1), "edge_boxes": [list(s) for s in (scene.get("edge_boxes") or [])],
                 "edge_at": [_units(s, W, H) for s in (scene.get("edge_boxes") or [])]}
        return out, check, alpha > 0.002
    bg = scene.get("bg")
    if it["kind"] == "light":
        if par.get("rgb"):
            key = np.asarray(par["rgb"], np.float32)
        elif bg is not None:
            key = cs.key_colour(bg)
        else:
            return img_d, {"no_key": True}, none
        rim = par.get("rim", "auto")
        side, by = cs.bright_side(bg, alpha) if bg is not None else (None, 0.0)
        if rim == "auto":
            rim = side
        rim = None if rim in (None, "none", "off") else rim
        rim_rgb = par.get("rim_rgb") or list(np.clip(np.asarray(key) * 1.05 + 25, 0, 255))
        cool = None
        if bg is not None:
            L = co.to_lab(bg)[..., 0]
            cool = bg[L <= np.quantile(L, 0.33)].reshape(-1, 3).mean(axis=0)
        lit = cs.light(fig, alpha, key, float(par.get("amount", 0.2)), rim, rim_rgb,
                       float(par.get("rim_strength", 0.45)), cool)
        m = alpha > 0.5
        lab_k = co.to_lab(np.asarray([[np.clip(key, 0, 255)]], np.uint8))[0, 0]

        def dist(img):
            mean = co.to_lab(img)[m].mean(axis=0)
            return float(np.hypot(mean[1] - lab_k[1], mean[2] - lab_k[2]))
        before, after = dist(fig), dist(lit)
        scene["fig"] = lit
        scene["rim"] = rim
        out = cs.composite(bg, lit, alpha) if bg is not None else cs.composite(img_d, lit, alpha)
        check = {"key": co_hex(np.clip(key, 0, 255)), "amount": float(par.get("amount", 0.2)),
                 "dist_before": round(before, 1), "dist_after": round(after, 1), "rim": rim,
                 "bright_side": side, "bright_by": round(by, 1)}
        return out, check, alpha > 0.002
    if it["kind"] == "shadow":
        c = cs.contact(alpha)
        if c is None:
            return img_d, {"contact": False}, none
        gy = int(round(par["ground"] * H / 1000.0)) if par.get("ground") is not None else c["y"]
        light_side = scene.get("rim") or cs.shading_side(fig, alpha)[0]
        d = par.get("dir", "auto")
        if d == "auto":
            d = {"left": "right", "right": "left"}.get(light_side, "back")
        sh, info = cs.contact_shadow(alpha, gy, d, float(par.get("strength", 0.55)))
        base = bg if bg is not None else img_d
        dark = np.clip(base.astype(np.float32) * (1 - sh[..., None]), 0, 255).astype(np.uint8)
        if bg is not None:
            scene["bg"] = dark
        out = cs.composite(dark, fig, alpha)
        L0, L1 = co.to_lab(base)[..., 0], co.to_lab(out)[..., 0]
        strip = np.zeros((H, W), bool)
        strip[min(H - 1, gy):min(H, gy + max(2, int(0.01 * H))), c["x0"]:c["x1"] + 1] = True
        strip &= alpha < 0.5
        drop = float((L0[strip] - L1[strip]).mean()) if strip.any() else 0.0
        away = {"left": "right", "right": "left"}.get(light_side)
        check = {"contact": True, "ground": round(gy * 1000.0 / H), "dir": d, "light_side": light_side,
                 "drop_L": round(drop, 1), "conflict": bool(away and d in ("left", "right") and d != away)}
        return out, check, (sh > 0.02) & (alpha < 0.5)
    raise UsageError(f"unknown change kind {it['kind']!r}", "easel change list")


def ground_mask(p, img_d: np.ndarray) -> np.ndarray:
    """The ground of the picture at the delivered size: pixels close to the rest region's main colour that
    reach the canvas edge without crossing line art (a flat shape laid 'behind' covers only these: the white of a
    shirt enclosed by its outline stays the figure's)."""
    from scipy import ndimage as ndi
    out = np.zeros(img_d.shape[:2], bool)
    lab = co.to_lab(img_d)
    for r, e in p.state["regions"].items():
        if is_rest(e):
            m = region_d(p, r)
            if m.any():
                med = np.median(lab[m], axis=0)
                d = np.linalg.norm(lab - med, axis=-1)
                out |= d < 12.0           # any pixel of the ground colour: the plan's polygons are rough
    if not out.any():
        L, C, _h = co.lch(lab)
        out = (L > 92) & (C < 6)
    barrier = ndi.binary_dilation(co.line_barrier(img_d), iterations=2)
    cand = out & ~barrier
    lbl, n = ndi.label(cand)
    if n:
        edge = np.unique(np.concatenate([lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]]))
        reach = np.isin(lbl, edge[edge > 0])
        out = ndi.binary_dilation(reach, iterations=2) & out          # back over the thickened line edge
    return out


def rebuild(p) -> Tuple[np.ndarray, np.ndarray]:
    """Replay every applied change on the original; write the target; return (old target, new target), both at the
    delivered size. Each item's check is measured on the way."""
    from .project import save_png
    old = np.asarray(p.ref_deliver).copy()
    base = orig_d(p)
    img = base
    taken = np.zeros(base.shape[:2], bool)
    scene: dict = {}
    order = sorted(cstate(p.state).get("items") or [], key=lambda x: KINDS.get(x["kind"], {}).get("phase", 5))
    for it in order:
        if not it.get("params") or not KINDS.get(it["kind"], {}).get("target"):
            continue
        img, check, edited = apply_item(p, img, it, base, taken, scene)
        taken |= edited
        it["check"] = check
    if "alpha" not in scene and (p.root / "cutout.png").exists():
        (p.root / "cutout.png").unlink()
    if p.scale > 1:
        save_png(p.root / "ref_deliver.png", img)
        save_png(p.root / "ref.png", working(p, img))
    else:
        save_png(p.root / "ref.png", img)
    p._ref = None
    p._ref_d = None
    p.state["target_rev"] = int(p.state.get("step", 0))
    try:                    # the stages trace by style: a change can turn what the target reads as (a palette map)
        from .workflow import guess_style
        g, why = guess_style(img, "")
        cstate(p.state)["target_style"] = [g, why]
    except Exception:  # noqa: BLE001
        pass
    return old, img


def cmd_for(it: dict, **override) -> str:
    """The change's command line, with some parameters changed."""
    par = dict(it.get("params") or {}, **override)
    if it["kind"] == "recolor":
        to = par.get("to") or it.get("to")
        extra = "".join(f" --{k} {float(par[k]):g}" for k, d in (("chroma", 1.0), ("light", 0.0), ("band", None))
                        if par.get(k) is not None and (d is None or abs(float(par[k]) - d) > 1e-9))
        return f'easel recolor {it["region"]} --to "{to}"{extra}'
    return it["cmd"]


def item_unmet(it: dict, state: Optional[dict] = None) -> List[dict]:
    """What keeps a change from done: each with a sentence and the command that fixes it."""
    c = it.get("check")
    n = it["id"]
    if KINDS.get(it["kind"], {}).get("stage") == "finish":
        if state is not None and finish_done(state, it):
            return []
        return [{"region": None, "metric": f"change {n}", "value": "not made", "need": "made",
                 "text": f"change {n} ({describe(it)}) is not made yet", "fix": it["cmd"]}]
    if not it.get("params") or not c:
        return [{"region": it.get("region"), "metric": f"change {n}", "value": "not made", "need": "made",
                 "text": f"change {n} ({describe(it)}) is not made yet", "fix": it["cmd"]}]
    out = []
    if it["kind"] == "palette":
        rgbs = (it.get("params") or {}).get("rgbs") or []
        lums = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in rgbs]
        if c["value_order"] < VALUE_ORDER_MIN and lums != sorted(lums):
            order = [x for _l, x in sorted(zip(lums, [co_hex(c_) for c_ in rgbs]))]
            out.append({"region": None, "metric": "value order", "value": c["value_order"], "need": VALUE_ORDER_MIN,
                        "text": f"change {n}: the palette map kept the light-dark order only {c['value_order']:.2f} "
                                f"(need {VALUE_ORDER_MIN}): the colours go dark to light",
                        "fix": it["cmd"].replace((it.get("params") or {}).get("to", ""), ",".join(order))})
        for r, e in (c.get("accent_edge") or {}).items():
            if e < ACCENT_EDGE_MIN and r not in (c.get("lines_refined") or []):
                out.append({"region": r, "metric": "accent edge on lines", "value": e, "need": ACCENT_EDGE_MIN,
                            "text": f"change {n}: only {100 * e:.0f}% of accent region {r}'s edge follows the line "
                                    f"art (need {100 * ACCENT_EDGE_MIN:.0f}%): the accent spills past its outlines",
                            "fix": f"easel region refine {r} --by lines"})
        return out
    if it["kind"] in ("cutout", "background", "light", "shadow"):
        return scene_unmet(it, c, state)
    if it["kind"] == "graphic":
        if c["drawn_px"] == 0:
            out.append({"region": None, "metric": "drawn", "value": 0, "need": 1,
                        "text": f"change {n}: the shape covers no ground pixel (the figure fills its box)",
                        "fix": 'easel graphic disc <x0,y0,x1,y1> --color "#rrggbb"   (another place)'})
        return out
    if it["kind"] == "recolor":
        r = it["region"]
        if c["stray"] > 0:
            out.append({"region": r, "metric": "stray patches", "value": c["stray"], "need": 0,
                        "text": f"change {n}: {c['stray']} patch(es) of {r}'s old colour touch {r} but lie outside it "
                                f"({c['stray_px']} px): they keep the old colour",
                        "fix": f"easel region refine {r} --by hue"})
        if c["hue_off"] > HUE_TOL:
            out.append({"region": r, "metric": "hue off", "value": c["hue_off"], "need": HUE_TOL,
                        "text": f"change {n}: {r} reached hue {co.hue360(c['hue_after'])} deg, asked "
                                f"{co.hue360(c['to_h'])} (off by {c['hue_off']:.0f}: the asked colour is out of reach at this chroma)",
                        "fix": cmd_for(it, chroma=round(0.8 * float((it.get("params") or {}).get("chroma", 1.0)), 2))})
        if c["shading"] > SHADING_MAX:
            out.append({"region": r, "metric": "shading drift", "value": c["shading"], "need": SHADING_MAX,
                        "text": f"change {n}: {r}'s light and shadow drifted {c['shading']:.1f} L (need <= "
                                f"{SHADING_MAX}): the asked colour is out of reach for some pixels",
                        "fix": cmd_for(it, chroma=round(0.8 * float((it.get("params") or {}).get("chroma", 1.0)), 2))})
    return out


def _units(b, W: int, H: int) -> str:
    """A delivered-pixel box in 0-1000 units."""
    return ",".join(str(int(round(v))) for v in (b[0] * 1000.0 / W, b[1] * 1000.0 / H, b[2] * 1000.0 / W,
                                                  b[3] * 1000.0 / H))


def scene_unmet(it: dict, c: dict, state: Optional[dict]) -> List[dict]:
    n = it["id"]
    out: List[dict] = []
    par = it.get("params") or {}
    its = items(state) if state is not None else []
    made = lambda k: any(x["kind"] == k and x.get("params") for x in its)  # noqa: E731
    if c.get("no_cutout"):
        return [{"region": None, "metric": "cut-out", "value": "none", "need": "made",
                 "text": f"change {n} ({describe(it)}) needs the figure cut out first",
                 "fix": "easel cutout figure --regions <the figure's regions, comma separated>"}]
    if it["kind"] == "cutout":
        if c["lost"] > LOST_MAX and c.get("lost_by"):
            worst = max(c["lost_by"], key=lambda r: c["lost_by"][r])
            out.append({"region": worst, "metric": "figure lost", "value": c["lost"], "need": LOST_MAX,
                        "text": f"change {n}: the cut left out {100 * c['lost']:.0f}% of the figure's regions (most of "
                                f"{worst}: {100 * c['lost_by'][worst]:.0f}%)",
                        "fix": f"{it['cmd']} --plus {worst}"})
        if state is not None and not any(x["kind"] == "background" for x in its):
            out.append({"region": None, "metric": "background", "value": "none", "need": "made",
                        "text": f"change {n}: the figure is cut out; it needs its new background",
                        "fix": BG_FORMS})
        return out
    if it["kind"] == "background":
        if c.get("halo", 0) > HALO_MAX:
            at = (c.get("edge_at") or ["0,0,1000,1000"])[0]
            out.append({"region": None, "metric": "old ground at the edge", "value": c["halo"], "need": HALO_MAX,
                        "text": f"change {n}: {100 * c['halo']:.0f}% of the figure's edge still shows the old ground; "
                                f"the worst place is boxed in VIEW (spot 1): take it out of the figure",
                        "fix": f"easel region add notfigure --box {at}   then  "
                               f"easel cutout ... --minus notfigure"})
        return out
    if it["kind"] == "light":
        if c.get("no_key"):
            return [{"region": None, "metric": "key colour", "value": "none", "need": "set",
                     "text": f"change {n}: the light takes its colour from the background; there is none yet, so "
                             "give it:  --tint \"#hex\"", "fix": 'easel light --tint "#ffb070"'}]
        if c.get("rim") and c.get("bright_side") in ("left", "right") and c.get("bright_by", 0) > 0.1 and \
                c["rim"] in ("left", "right") and c["rim"] != c["bright_side"]:
            out.append({"region": None, "metric": "rim side", "value": c["rim"], "need": c["bright_side"],
                        "text": f"change {n}: the rim light is on the {c['rim']} but the background's light sits on "
                                f"the {c['bright_side']}: the light comes from there",
                        "fix": f"easel light --rim {c['bright_side']}" +
                               (f" --tint \"{par['tint']}\"" if par.get("rgb") else "")})
        return out
    if it["kind"] == "shadow":
        if not c.get("contact"):
            return [{"region": None, "metric": "contact", "value": "none", "need": "ground",
                     "text": f"change {n}: the figure runs off the bottom of the picture, so it stands on no ground "
                             "inside it: a contact shadow has no place",
                     "fix": f"easel change drop {n}"}]
        if c.get("conflict"):
            away = {"left": "right", "right": "left"}[c["light_side"]]
            out.append({"region": None, "metric": "shadow direction", "value": c["dir"], "need": away,
                        "text": f"change {n}: the shadow falls {c['dir']} toward the light ({c['light_side']}); "
                                f"shadows fall away from it", "fix": f"easel shadow {par.get('subject') or 'figure'} --dir {away}"})
        return out
    return out


def unmet_all(state: dict, stage: str = "change") -> List[dict]:
    out = []
    ts = cstate(state).get("target_style")
    if stage == "change" and ts and ts[0] != state.get("style") and not state.get("style_kept"):
        out.append({"region": None, "metric": "style", "value": state.get("style"), "need": ts[0],
                    "text": f"the target now reads as {ts[0]} ({str(ts[1]).split(';')[0]}), the project traces it as "
                            f"{state.get('style')}",
                    "fix": f"easel style {ts[0]}"})
    for it in items(state):
        if KINDS.get(it["kind"], {}).get("stage", "change") == stage:
            out += item_unmet(it, state)
    return out


def co_hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb)


def finish_done(state: dict, it: dict) -> bool:
    """A finishing change is made when the picture holds it: the text item with that string, or the shape step
    with that command, still applied (not undone)."""
    par = it.get("params") or {}
    if it["kind"] == "text":
        return any(str(t.get("text")) == str(par.get("text")) for t in state.get("texts") or [])
    if it["kind"] == "shape":
        return str(it.get("cmd")) in (state.get("finish_made") or [])
    return False


def check_lines(it: dict) -> List[str]:
    """The measured result of one change, in words."""
    c = it.get("check")
    if KINDS.get(it["kind"], {}).get("stage") == "finish":
        return [f"change {it['id']}: {describe(it)} (made in the finish stage, over the traced picture)"]
    if not c:
        return [f"change {it['id']}: {describe(it)}: not made yet"]
    if it["kind"] == "palette":
        acc = ", ".join(f"{r} {100 * v:.1f}% of the picture, edge on lines {100 * c['accent_edge'].get(r, 0):.0f}%"
                        for r, v in (c.get("accent_share") or {}).items())
        return [f"palette: {c['colours']} colours; light-dark order kept {c['value_order']:.2f} (need >= "
                f"{VALUE_ORDER_MIN})" + (f"; accents: {acc}" if acc else "")]
    if it["kind"] == "graphic":
        return [f"{describe(it)}: {c['drawn_px']} px drawn on the ground ({100 * c['box_share']:.0f}% of its box); "
                f"{c['behind_px']} px held back behind the figure"]
    if it["kind"] == "canvas":
        return [f"canvas {c.get('size')}: the reference placed at {c.get('box')} (scale {c.get('scale')})"]
    if c.get("no_cutout"):
        return [f"change {it['id']}: {describe(it)}: waits for the cut-out"]
    if it["kind"] == "cutout":
        pk = "; ".join(f"the {q['corner']} corner closed off by the figure taken as figure ({100 * q['share']:.1f}% of "
                       "the picture)" for q in c.get("pockets") or [])
        lines = [f"cut-out ({'white' if c['mode'] == 'white' else 'painted'} ground): the figure is "
                 f"{100 * c['share']:.1f}% of the picture; {100 * c['lost']:.1f}% of its regions left out (need <= "
                 f"{100 * LOST_MAX:.0f}%)" + (f"; {pk}" if pk else ""),
                 f"edge smoothed over {c['smooth_px']:.0f} px, feathered 1 px; old ground in the edge ring "
                 f"{100 * c['ring_ground']:.1f}% (repainted from the figure next to it)"]
        for k, s in enumerate(c.get("spots") or [], 1):
            lines.append(f"edge spot {k} at {s.get('at')}: contrast {s['contrast']:.0f} dE"
                         f"{' (the cut guesses here)' if s['contrast'] < 12 else ''}, smoothing moved "
                         f"{100 * s['moved']:.0f}%")
        return lines
    if it["kind"] == "background":
        src = {"project": f"the project {c.get('from')}", "file": f"the picture {c.get('from')}",
               "gradient": "a gradient", "color": "one colour"}.get(c["source"], c["source"])
        return [f"background from {src}" + (f" (cropped {100 * c['crop']:.0f}% to fit)" if c.get("crop") else "")
                + f"; it shows on {100 * c['shows']:.0f}% of the picture; its light sits on the {c['bright_side']}",
                f"old ground left at the figure's edge: {100 * c['halo']:.1f}% of the edge ring (need <= "
                f"{100 * HALO_MAX:.0f}%)"]
    if it["kind"] == "light":
        if c.get("no_key"):
            return [f"change {it['id']}: light: no key colour (no background, no --tint)"]
        return [f"light: figure tinted toward {c['key']} by {c['amount']:.2f}: colour distance to the key "
                f"{c['dist_before']:.0f} -> {c['dist_after']:.0f}; rim light "
                + (f"on the {c['rim']}" if c.get("rim") else "off")
                + (f" (the background's light sits on the {c['bright_side']})" if c.get("bright_side") else "")]
    if it["kind"] == "shadow":
        if not c.get("contact"):
            return [f"shadow: no contact with the ground (the figure runs off the bottom)"]
        return [f"shadow on the ground at y {c['ground']}, falling {c['dir']}; the ground under the contact is "
                f"{c['drop_L']:.0f} L darker; light from the {c['light_side']}"
                + (" (the shadow falls toward it)" if c.get("conflict") else "")]
    if it["kind"] == "recolor":
        r = it["region"]
        light = (f"lightness kept (median {c['from_L']:.0f} -> {c['L_after']:.0f}; asked colour {c['to_L']:.0f}: "
                 f"--light 1 moves it there)" if abs(c["L_after"] - c["to_L"]) > 8 else
                 f"lightness {c['from_L']:.0f} -> {c['L_after']:.0f} (asked {c['to_L']:.0f})")
        lines = [f"{r}: hue {co.hue360(c['from_h'])} -> {co.hue360(c['hue_after'])} deg (asked "
                 f"{co.hue360(c['to_h'])}), chroma {c['from_C']:.0f} -> asked {c['to_C']:.0f}; {light}",
                 f"shading kept: drift {c['shading']:.1f} L (need <= {SHADING_MAX}); {100 * c['edited']:.1f}% of the picture "
                 f"recoloured; islands {c['islands']}",
                 (f"stray patches: {c['stray']} ({c['stray_px']} px of the old colour touch {r} but lie outside it; "
                  f"boxed in VIEW)" if c["stray"] else "stray patches: 0")]
        return lines
    return [describe(it)]


# ----------------------------------------------------------------------------------------------- late changes

def late(state: dict) -> bool:
    return state.get("stage") not in ("plan", "change")


def late_files(p) -> List[str]:
    return [f"layers/{n}.png" for n in STAGE_LAYERS if (p.root / "layers" / f"{n}.png").exists()]


def mark_stale(p, old_d: np.ndarray, new_d: np.ndarray) -> List[List[str]]:
    """After a change made once tracing has started: erase the changed pixels from the layers the done stages
    painted, and list (stage, region) pairs to redraw. Returns the new pairs."""
    import cv2
    from . import workflow
    st = p.state
    if not late(st):
        return []
    ch = co.delta_e(old_d, new_d) > 3.0
    if not ch.any():
        return []
    if p.scale > 1:
        ch = cv2.resize(ch.astype(np.uint8), (p.W, p.H), interpolation=cv2.INTER_NEAREST) > 0
    plan = workflow.plan_of(st)
    stale = [list(x) for x in st.get("stale") or []]
    new = []
    erased = set()
    for stage in plan:
        if stage in ("plan", "change", "sketch", "finish"):
            continue
        if not (stage in st.get("stages_done", []) or stage == st["stage"]):
            continue
        layer = workflow.STAGES.get(stage, {}).get("layer")
        if not layer:
            continue
        arr = p.layer(layer, create=False)
        if arr is None:
            continue
        hit = ch & (arr[..., 3] > 0)
        if not hit.any():
            continue
        redraw = np.zeros_like(ch)
        for r in p.region_names():
            m = p.region_mask(r)
            if (m & hit).sum() >= max(50, 0.002 * m.sum()):
                redraw |= m
                if [stage, r] not in stale:
                    stale.append([stage, r])
                    new.append([stage, r])
        if layer not in erased and redraw.any():
            # erased only inside the regions that are redrawn: no other region is left with a hole
            arr = arr.copy()
            arr[ch & redraw, 3] = 0.0
            p._write_layer_file(layer, arr)
            erased.add(layer)
    order = {s: i for i, s in enumerate(plan)}
    stale.sort(key=lambda x: (order.get(x[0], 99), x[1]))
    st["stale"] = stale
    det = [r for s, r in stale if s == "detail"]
    if det:
        lv = st.setdefault("levels", {})
        if "all" in lv:
            v = lv.pop("all")
            for r in p.region_names():
                lv[r] = v if lv.get(r) is None else min(lv[r], v)
        for r in det:
            lv.pop(r, None)
    if st.get("exported") or st.get("finished"):
        st["exported"] = None
        st["finished"] = False
        for t in st.get("todo", []):
            if t["stage"] == "finish" and t.get("key") in ("sharpen", "export") and t["status"] != "open":
                t["status"], t["note"] = "open", "the target changed"
    if "change" not in st.get("stages_done", []):
        st.setdefault("stages_done", []).insert(1, "change")
    return new


def stale_command(state: dict, stage: str, region: str) -> str:
    from . import workflow
    if stage == "detail":
        return f"easel detail {region} --level 1" if workflow.levels_available() else f"easel detail {region}"
    return workflow.task_command(state, stage, region)


def clear_stale(state: dict, stage: str, region: Optional[str]) -> None:
    if not state.get("stale"):
        return
    state["stale"] = [x for x in state["stale"] if not (x[0] == stage and (region == "all" or x[1] == region))]


# ----------------------------------------------------------------------------------------------- workflow hooks

def todo_items(state: dict, stage: str = "change") -> List[dict]:
    out = []
    for it in items(state):
        if KINDS.get(it["kind"], {}).get("stage") != stage:
            continue
        done = finish_done(state, it) if stage == "finish" else (bool(it.get("check")) and not item_unmet(it))
        out.append({"text": f"change {it['id']}: {describe(it)}", "cmd": it["cmd"], "region": it.get("region"),
                    "key": f"change{it['id']}", "status": "done" if done else "open"})
    return out


def decision_unmet(state: dict) -> List[dict]:
    if not needs_decision(state):
        return []
    return [{"region": None, "metric": "changes stated", "value": "no", "need": "yes",
             "text": "the request's changes to the reference are not stated yet",
             "fix": 'easel change add recolor <region> --to "#rrggbb"   (one per change; or  easel change none)'}]


def next_step(project, phase: str) -> Optional[Tuple[str, str]]:
    """(why, command) when a change needs doing now, else None (the usual flow goes on)."""
    st = project.state
    if phase == "plan":
        if needs_decision(st):
            req = str(st.get("request") or "").strip()
            req = req if len(req) <= 90 else req[:87] + "..."
            return (f'the request ("{req}") may ask for a picture that differs from the reference: state each change '
                    'before tracing, one  easel change add <kind> <region> ...  per change (kinds: '
                    f'{", ".join(KINDS)}), or  easel change none  when it asks for a plain redraw',
                    'easel change add recolor <region> --to "#rrggbb"')
        return None
    if st["stage"] == "plan" or not active(st):
        return None
    for it in items(st):
        kstage = KINDS.get(it["kind"], {}).get("stage")
        if kstage == "finish":
            if st["stage"] == "finish" and not finish_done(st, it):
                return f"change {it['id']} ({describe(it)}) goes over the traced picture", it["cmd"]
            continue
        um = item_unmet(it, st)
        if um:
            return um[0]["text"], um[0]["fix"]
    stale = st.get("stale") or []
    if stale and st["stage"] != "change":
        s, r = stale[0]
        return (f"the target changed after {s} painted {r}: redraw it ({len(stale)} region-stage(s) left)",
                stale_command(st, s, r))
    return None


GATE_WORDS = "every stated change made and its checks met (hue reached, shading kept, no stray patches)"
STAGE_WHAT = "make the request's changes on the target (the picture the next stages trace)"
STAGE_LOOK = "the target shows every change; nothing of the old colours left at the edges"
