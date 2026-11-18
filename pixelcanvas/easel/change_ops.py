"""Pure picture operations for changes to the target (the picture the traced stages copy): recolour a region with
its shading kept, measure what a recolour left behind, and snap a rough region to its colour or to its line art.

Images are uint8 RGB arrays (H x W x 3); masks are bool arrays of the same size. Nothing here touches the disk.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi


# ----------------------------------------------------------------------------------------------- colour helpers

def to_lab(img_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_u8.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def lab_to_u8(lab: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def lch(lab: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b = lab[..., 1], lab[..., 2]
    return lab[..., 0], np.hypot(a, b), np.degrees(np.arctan2(b, a))


def hex_lch(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    lab = to_lab(np.asarray(rgb, np.uint8).reshape(1, 1, 3))[0, 0]
    return float(lab[0]), float(np.hypot(lab[1], lab[2])), float(np.degrees(np.arctan2(lab[2], lab[1])))


def angdiff(a, b):
    """Signed hue difference a - b in degrees, in (-180, 180]."""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def circ_mean(h: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    r = np.radians(h)
    w = np.ones_like(r) if w is None else w
    return float(np.degrees(np.arctan2((np.sin(r) * w).sum(), (np.cos(r) * w).sum())))


def hue360(h: float) -> int:
    return int(round(h % 360.0))


def delta_e(a_u8: np.ndarray, b_u8: np.ndarray) -> np.ndarray:
    return np.linalg.norm(to_lab(a_u8) - to_lab(b_u8), axis=-1)


# ----------------------------------------------------------------------------------------------- the region's colour

def dominant(img_u8: np.ndarray, mask: np.ndarray, min_c: float = 6.0) -> Dict[str, float]:
    """The region's main colour: the peak of its chroma-weighted hue histogram, the hue cluster around it (its
    half-width ``band``: hair with pink lights and purple shadows is one cluster, a red ribbon in it another), and
    the median chroma and lightness of the cluster's pixels."""
    L, C, h = lch(to_lab(img_u8))
    sel = mask & (C > min_c)
    if sel.sum() < 20:
        sel_all = mask
        return {"h": 0.0, "C": float(np.median(C[sel_all])) if sel_all.any() else 0.0, "band": 35.0,
                "L": float(np.median(L[sel_all])) if sel_all.any() else 50.0, "grey": True, "n": int(sel_all.sum())}
    hist, edges = np.histogram(h[sel], bins=72, range=(-180.0, 180.0), weights=C[sel])
    hist = hist + np.roll(hist, 1) + np.roll(hist, -1)
    ip = int(np.argmax(hist))
    peak = float(edges[ip] + 2.5)
    floor = 0.08 * hist[ip]
    lo = hi = 0
    while lo < 30 and hist[(ip - lo - 1) % 72] > floor:
        lo += 1
    while hi < 30 and hist[(ip + hi + 1) % 72] > floor:
        hi += 1
    near = sel & (np.abs(angdiff(h, peak)) <= 5.0 * max(lo, hi) + 2.5)
    h0 = circ_mean(h[near], C[near]) if near.sum() > 10 else peak
    band = float(np.clip(5.0 * max(lo, hi) + 7.5 + abs(angdiff(h0, peak)), 25.0, 75.0))
    clus = sel & (np.abs(angdiff(h, h0)) <= band)
    sel = clus if clus.sum() > 10 else sel
    Ls = L[sel]
    return {"h": h0, "C": float(np.median(C[sel])), "L": float(np.median(Ls)), "grey": False, "n": int(sel.sum()),
            "band": band, "L_lo": float(np.percentile(Ls, 2)) - 5.0, "L_hi": float(np.percentile(Ls, 98)) + 5.0}


def band_mask(img_u8: np.ndarray, h0: float, band: float, min_c: float, L_lo: float = -1.0,
              L_hi: float = 101.0) -> np.ndarray:
    """Pixels of the hue h0 (within ``band`` degrees), chroma above ``min_c``, lightness in [L_lo, L_hi]."""
    L, C, h = lch(to_lab(img_u8))
    return (np.abs(angdiff(h, h0)) <= band) & (C > min_c) & (L >= L_lo) & (L <= L_hi)


# ----------------------------------------------------------------------------------------------- recolour

def recolor(img_u8: np.ndarray, mask: np.ndarray, to_rgb, chroma: float = 1.0, light: float = 0.0,
            band: Optional[float] = None, feather: float = 1.0) -> Tuple[np.ndarray, Dict]:
    """Move the region's main colour to ``to_rgb`` and keep every pixel's place around it: hue turned by the same
    angle, chroma scaled so the median reaches the asked chroma (stronger pixels compressed toward it), lightness kept
    (``light`` 0..1 moves the region's median lightness that far toward the asked colour). Pixels of another hue
    cluster in the region (a ribbon, a badge) stay. Partial weights (the region's edge, the cluster's edge) turn the
    hue part of the way, so no in-between colour of the two hues appears."""
    lab = to_lab(img_u8)
    L, C, h = lch(lab)
    Lt, Ct, ht = hex_lch(to_rgb)
    dom = dominant(img_u8, mask)
    band = float(dom.get("band", 35.0) if band is None else band)
    f = ndi.gaussian_filter(mask.astype(np.float32), feather) if feather > 0 else mask.astype(np.float32)
    f = np.minimum(f, mask.astype(np.float32) + f * 0.5)       # the feather may soften the edge, not spill far out
    if dom["grey"]:
        # a grey region has no hue to turn: lay the asked hue over it at the asked chroma, lightness kept
        w = f
        # the asked colour by value: full at its own lightness, fading toward white and black (highlights stay white,
        # lines stay dark, nothing leaves the gamut)
        fade = np.where(L >= Lt, (100.0 - L) / max(100.0 - Lt, 1.0), L / max(Lt, 1.0))
        k = Ct * chroma * np.clip(fade, 0.0, 1.0)
        a_n = lab[..., 1] * (1 - w) + w * k * np.cos(np.radians(ht))
        b_n = lab[..., 2] * (1 - w) + w * k * np.sin(np.radians(ht))
    else:
        d = np.abs(angdiff(h, dom["h"]))
        wband = np.clip((band + 10.0 - d) / 10.0, 0.0, 1.0)
        wband = np.where(C <= 4.0, 1.0, wband)                 # near-grey pixels (lines, highlights) follow along
        w = f * wband
        C0 = max(dom["C"], 1.0)
        k = float(np.clip(Ct / C0, 0.25, 4.0)) * chroma
        # stronger than the median: compressed (square root) toward the asked chroma, the order kept
        # weaker than the median: the factor fades toward 1, so near-grey pixels (lashes, lines) stay near grey
        kp = np.where(C > C0, k * np.sqrt(C0 / np.maximum(C, 1e-3)), 1.0 + (k - 1.0) * C / C0)
        # the full recolour of every pixel (weight 1), then a straight blend in Lab by the weight: a partial pixel
        # (region edge, cluster edge) passes near grey, never through a third hue
        hn = h + angdiff(ht, dom["h"])
        Cn = C * kp
        a_r, b_r = Cn * np.cos(np.radians(hn)), Cn * np.sin(np.radians(hn))
        # weak pixels (an edge pixel half hair, half skin; a grey line) move by the main colour's shift times their
        # share, instead of turning half way round the hue circle (which paints a pale complementary fringe)
        da = Ct * chroma * np.cos(np.radians(ht)) - C0 * np.cos(np.radians(dom["h"]))
        db = Ct * chroma * np.sin(np.radians(ht)) - C0 * np.sin(np.radians(dom["h"]))
        share = np.clip(C / C0, 0.0, 1.0)
        a_s = lab[..., 1] + share * da
        b_s = lab[..., 2] + share * db
        rot = np.clip((C / C0 - 0.5) / 0.5, 0.0, 1.0)
        a_f, b_f = rot * a_r + (1.0 - rot) * a_s, rot * b_r + (1.0 - rot) * b_s
        a_n = lab[..., 1] + w * (a_f - lab[..., 1])
        b_n = lab[..., 2] + w * (b_f - lab[..., 2])
    Ln = np.clip(L + w * light * (Lt - dom["L"]), 0.0, 100.0)
    new = lab_to_u8(np.stack([Ln, a_n, b_n], -1))
    out = np.where((w > 0.02)[..., None], new, img_u8)
    info = {"from_h": dom["h"], "from_C": dom["C"], "from_L": dom["L"], "grey": dom["grey"],
            "L_lo": dom.get("L_lo", -1.0), "L_hi": dom.get("L_hi", 101.0),
            "to_h": ht, "to_C": Ct, "to_L": Lt, "band": band, "edited": (w > 0.5)}
    return out, info


def colour_info(img_u8: np.ndarray, region: np.ndarray, band: Optional[float] = None) -> Dict:
    """The region's colour as the stray / refine measures use it (``band`` None: the region's own hue cluster)."""
    dom = dominant(img_u8, region)
    return {"from_h": dom["h"], "from_C": dom["C"], "band": float(dom.get("band", 35.0) if band is None else band),
            "grey": dom["grey"], "L_lo": dom.get("L_lo", -1.0), "L_hi": dom.get("L_hi", 101.0)}


def recolor_checks(before_u8: np.ndarray, after_u8: np.ndarray, region: np.ndarray, info: Dict,
                   reach: int, min_px: int, exclude: Optional[np.ndarray] = None, with_stray: bool = True) -> Dict:
    """What the recolour did, measured: the hue reached against the asked hue, shading kept (how far each edited
    pixel's lightness offset from the region's median drifted, L units), stray patches (pixels of the old colour that touch the region
    but lie outside it, so they keep the old colour), islands (small edited specks apart from the rest)."""
    ed = info["edited"]
    La, Ca, ha = lch(to_lab(after_u8))
    Lb, Cb, hb = lch(to_lab(before_u8))
    sel = ed & (Ca > 6.0)
    if not info.get("grey"):
        sel &= (Cb > 6.0) & (np.abs(angdiff(hb, info["from_h"])) <= info["band"])   # the pixels of the main hue
    hue_after = circ_mean(ha[sel], Ca[sel]) if sel.sum() > 20 else info["to_h"]
    # shading kept: every pixel's lightness offset from the region's median, before and after (L units)
    if ed.sum() > 20:
        dl = La[ed] - Lb[ed]
        shading = float(np.abs(dl - np.median(dl)).mean())
    else:
        shading = 0.0
    stray = stray_patches(before_u8, region, info, reach, min_px, exclude) if with_stray else None
    closed = ndi.binary_closing(ed, iterations=2)
    lab_, n = ndi.label(closed)
    islands = 0
    if n > 1:
        sizes = ndi.sum(np.ones_like(lab_), lab_, index=np.arange(1, n + 1))
        islands = int((sizes < max(min_px, 0.0005 * ed.sum())).sum())
    return {"hue_after": hue_after, "hue_off": float(abs(angdiff(hue_after, info["to_h"]))),
            "shading": shading, "stray": stray, "islands": islands, "edited_share": float(ed.mean()),
            "L_after": float(np.median(La[ed])) if ed.any() else 0.0}


def stray_patches(img_u8: np.ndarray, region: np.ndarray, info: Dict, reach: int, min_px: int,
                  exclude: Optional[np.ndarray] = None) -> Dict:
    """Patches of the region's own colour (hue band, chroma, lightness range) just outside it: touching it, within
    ``reach`` px, not in ``exclude`` (the other focus-3 regions: a face keeps its own pinks)."""
    if info.get("grey"):
        return {"n": 0, "px": 0, "boxes": [], "mask": np.zeros_like(region)}
    cand = band_mask(img_u8, info["from_h"], info["band"], max(8.0, 0.35 * info["from_C"]),
                     info.get("L_lo", -1.0), info.get("L_hi", 101.0)) & ~region
    if exclude is not None:
        cand &= ~exclude
    near = ndi.binary_dilation(region, iterations=max(1, reach))
    cand &= near
    lab_, n = ndi.label(cand)
    if n == 0:
        return {"n": 0, "px": 0, "boxes": [], "mask": np.zeros_like(region)}
    touch = ndi.binary_dilation(region, iterations=1)
    ids = np.arange(1, n + 1)
    sizes = ndi.sum(np.ones_like(lab_), lab_, index=ids)
    touching = ndi.maximum(touch.astype(np.uint8), lab_, index=ids) > 0
    keep = ids[(sizes >= min_px) & touching]
    m = np.isin(lab_, keep)
    boxes = []
    objs = ndi.find_objects(lab_)
    order = sorted(keep, key=lambda i: -sizes[i - 1])
    for i in order:
        sl = objs[i - 1]
        boxes.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop))
    return {"n": int(len(keep)), "px": int(m.sum()), "boxes": boxes, "mask": m}


# ----------------------------------------------------------------------------------------------- region refine

def refine_by_hue(img_u8: np.ndarray, region: np.ndarray, band: Optional[float] = None, reach: int = 20,
                  others: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
    """Grow a region over the pixels of its own colour that touch it (within ``reach`` px); pixels of other colours
    at its edge that the region holds only by roughness are left in it (they keep their colour under a recolour).
    ``others``: pixels the region must not take (the other named regions)."""
    info = colour_info(img_u8, region, band)
    out = region.copy()
    for _ in range(64):                      # a strand longer than the reach is taken in a few rounds
        add = stray_patches(img_u8, out, info, reach, 1, others)["mask"]
        if not add.any():
            break
        out |= add
    # holes of the same colour inside the region (between strands) belong to it as well
    filled = ndi.binary_fill_holes(out)
    if not info["grey"]:
        hole_ok = band_mask(img_u8, info["from_h"], info["band"], max(8.0, 0.35 * info["from_C"]), info["L_lo"],
                            info["L_hi"])
        if others is not None:
            hole_ok &= ~others
        out = out | (filled & ~out & hole_ok)
    return out, {"added": int((out & ~region).sum()), "removed": 0, "hue": info["from_h"]}


def line_barrier(img_u8: np.ndarray) -> np.ndarray:
    """Line art of a picture: pixels clearly darker than their neighbourhood, or very dark."""
    lum = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    local = cv2.GaussianBlur(lum, (0, 0), 3.0)
    dark = max(0.22, 0.5 * float(np.median(lum)))     # an ink drawing on white: its grey hatching closes areas too
    return (lum < local - 0.10) | (lum < dark)


def refine_by_lines(img_u8: np.ndarray, region: np.ndarray, share: float = 0.5,
                    min_area: int = 30) -> Tuple[np.ndarray, Dict]:
    """Snap a region to the areas its line art encloses: every area between lines that lies mostly inside the
    region joins it, every area mostly outside leaves it; the line pixels between kept areas stay in."""
    barrier = ndi.binary_dilation(line_barrier(img_u8), iterations=1)     # one px closes the small gaps in the lines
    lab_, n = ndi.label(~barrier)
    if n == 0:
        return region.copy(), {"added": 0, "removed": 0}
    ids = np.arange(1, n + 1)
    tot = ndi.sum(np.ones_like(lab_), lab_, index=ids)
    inn = ndi.sum(region.astype(np.float32), lab_, index=ids)
    keep = ids[(inn / np.maximum(tot, 1) >= share) & (tot >= min_area)]
    kept = np.isin(lab_, keep)
    small = np.isin(lab_, ids[tot < min_area]) & region            # tiny areas keep their old membership
    out = kept | small
    out = out | (ndi.binary_closing(out, iterations=3) & barrier)    # the lines between kept areas
    if out.sum() < 0.5 * region.sum():
        # the line art does not close around the region (gaps join it to the areas beside it): left as it was
        return region.copy(), {"added": 0, "removed": 0, "kept": True}
    return out, {"added": int((out & ~region).sum()), "removed": int((region & ~out).sum())}


# ----------------------------------------------------------------------------------------------- palette map

def palette_map(img_u8: np.ndarray, colours, accents=(), gamma: float = 1.1, accent_mix: float = 0.65,
                light_at: Optional[float] = None) -> Tuple[np.ndarray, Dict]:
    """Map the picture to a few colours by value: ``colours`` dark to light, each pixel takes the colour at its
    lightness (piecewise between them). ``accents``: (mask, colour) pairs; inside the mask the light end becomes the
    accent colour (``accent_mix`` of it over the plain map), so lines stay dark and the accent sits on the lights."""
    rgb = img_u8.astype(np.float32) / 255.0
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    lo, hi = np.percentile(lum, 0.5), np.percentile(lum, 99.5)
    if light_at is not None:                       # the ground and anything lighter: the light end
        hi = min(hi, float(light_at))
    t = np.clip((lum - lo) / max(hi - lo, 1e-3), 0.0, 1.0) ** gamma
    cols = np.asarray(colours, np.float32) / 255.0
    n = len(cols)
    pos = t * (n - 1)
    i0 = np.clip(np.floor(pos).astype(int), 0, n - 2)
    f = (pos - i0)[..., None]
    out = cols[i0] * (1.0 - f) + cols[i0 + 1] * f
    share = {}
    for k, (m, c) in enumerate(accents):
        ac = np.asarray(c, np.float32) / 255.0
        # the region's own value range: a dark garment still carries the accent on its lighter parts
        tl = t
        if m.any():
            r0, r1 = np.percentile(lum[m], 5), np.percentile(lum[m], 95)
            if r1 - r0 < 0.15:                      # a flat area: its own colour is the light end, lines the dark
                r1 = float(np.median(lum[m]))
                r0 = r1 - 0.15
            tl = np.clip((lum - r0) / (r1 - r0), 0.0, 1.0) ** gamma
        acc = cols[0] * (1.0 - tl[..., None]) + ac * tl[..., None]
        mm = m[..., None]
        out = np.where(mm, out * (1.0 - accent_mix) + acc * accent_mix, out)
        share[k] = float(m.mean())
    out_u8 = (np.clip(out, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    lum2 = out @ np.array([0.299, 0.587, 0.114], np.float32)
    plain = np.ones(lum.shape, bool)                  # the accents change value on purpose: measured outside them
    for m, _c in accents:
        plain &= ~m
    step = max(1, int(plain.sum() // 200000))
    a, b = lum[plain][::step], lum2[plain][::step]
    order = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-6 and b.std() > 1e-6 else 1.0
    return out_u8, {"value_order": order, "colours": n + len(accents), "accent_share_list": list(share.values()),
                    "accent_share": {}}


def edge_on_lines(region: np.ndarray, line: np.ndarray, near: int = 2) -> float:
    """Share of a region's edge pixels that lie on (within ``near`` px of) the line art."""
    edge = region & ~ndi.binary_erosion(region, iterations=1)
    if not edge.any():
        return 1.0
    close = ndi.binary_dilation(line, iterations=near)
    return float((edge & close).sum() / edge.sum())


# ----------------------------------------------------------------------------------------------- graphic

def graphic(img_u8: np.ndarray, shape: str, box_px, rgb, ground: Optional[np.ndarray] = None,
            width: float = 0.0) -> Tuple[np.ndarray, Dict]:
    """A flat shape in one colour: disc (an ellipse filling the box), rect, or ring / frame (an outline of
    ``width`` px). With ``ground`` it is laid only on those pixels: behind the figure."""
    H, W = img_u8.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in box_px)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if shape in ("disc", "ellipse", "ring"):
        cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, max(1.0, (x1 - x0) / 2), max(1.0, (y1 - y0) / 2)
        d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
        dist = (d - 1.0) * min(rx, ry)                       # about px from the edge, negative inside
    else:
        dx = np.maximum(x0 - xx, xx - x1)
        dy = np.maximum(y0 - yy, yy - y1)
        dist = np.maximum(dx, dy)
    if shape in ("ring", "frame") and width > 0:
        cover = np.clip(0.5 - (np.abs(dist + width / 2.0) - width / 2.0), 0.0, 1.0)
    else:
        cover = np.clip(0.5 - dist, 0.0, 1.0)                 # one px of anti-aliasing
    full = cover.copy()
    if ground is not None:
        g = ndi.gaussian_filter(ground.astype(np.float32), 0.7)
        cover = cover * np.clip(g * 1.4 - 0.2, 0.0, 1.0)
    c = np.asarray(rgb, np.float32)
    out = img_u8.astype(np.float32) * (1.0 - cover[..., None]) + c * cover[..., None]
    drawn = int((cover > 0.5).sum())
    box_n = max(1, int((full > 0.5).sum()))
    return (out + 0.5).astype(np.uint8), {"drawn": drawn, "box_share": drawn / box_n,
                                           "held_back": int(((full > 0.5) & (cover <= 0.5)).sum())}
