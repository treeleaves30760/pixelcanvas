"""Pure picture operations for a figure put in a new setting: cut-out, background, light, contact shadow.

Images are uint8 RGB ``H x W x 3`` at the delivered size; masks are bool; alpha is float32 0..1. Nothing here
touches the disk.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import change_ops as co

EDGE_SMOOTH = 0.004      # silhouette smoothing radius, share of the picture's diagonal
FEATHER = 1.0            # alpha feather (px) after smoothing
GRABCUT_SIDE = 600       # GrabCut runs on a copy this long on its long side


def _disk_hull(mask: np.ndarray) -> np.ndarray:
    import cv2
    out = np.zeros(mask.shape, np.uint8)
    pts = np.argwhere(mask)[:, ::-1].astype(np.int32)
    if len(pts) >= 3:
        cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
    return out > 0


def corner_pockets(fig: np.ndarray, max_share: float = 0.5) -> Tuple[np.ndarray, List[dict]]:
    """Pockets of ground the figure closes off against the frame: a ground area that touches the frame on two
    neighbouring edges only (a corner), is not the main ground and is smaller than ``max_share`` of the figure. A bust
    that runs off the bottom and the side of the picture closes such a corner; it is the figure's (a dark shoulder on
    a dark ground). Returns (pocket mask, [{share, corner}])."""
    from scipy import ndimage as ndi
    H, W = fig.shape
    gr = ~ndi.binary_closing(fig, iterations=3)
    gr = ndi.binary_opening(gr, iterations=max(1, int(0.01 * max(H, W))))
    lab, n = ndi.label(gr)
    out = np.zeros_like(fig)
    info = []
    if n < 2:
        return out, info
    sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    main = int(np.argmax(sizes)) + 1
    for i in range(1, n + 1):
        if i == main:
            continue
        c = lab == i
        t, b, l, r = c[0].any(), c[-1].any(), c[:, 0].any(), c[:, -1].any()
        edges = [k for k, v in zip("TBLR", (t, b, l, r)) if v]
        corner = {"BR": "bottom-right", "BL": "bottom-left", "TR": "top-right", "TL": "top-left"}.get(
            "".join(sorted(edges, key="TBLR".index)) if len(edges) == 2 else "", None)
        if corner is None or sizes[i - 1] > max_share * fig.sum():
            continue
        out |= c
        info.append({"share": round(float(c.mean()), 4), "corner": corner})
    if out.any():
        out = ndi.binary_dilation(out, iterations=max(1, int(0.01 * max(H, W)))) & ~fig | out
    return out, info


def cutout(img: np.ndarray, hint: np.ndarray, plus: Optional[np.ndarray] = None,
           minus: Optional[np.ndarray] = None, white_ground: Optional[np.ndarray] = None,
           corners: bool = True) -> Tuple[np.ndarray, dict]:
    """The figure's alpha. ``hint`` is the figure's plan regions (rough outlines), ``plus`` / ``minus`` areas that are
    always / never the figure. On a white ground (``white_ground``: the ground flooded from the frame) the figure is
    everything else; on a painted ground GrabCut is seeded with the regions, the frame outside their hull as ground.
    The silhouette is smoothed (no jagged cut) and feathered by one pixel."""
    import cv2
    from scipy import ndimage as ndi
    H, W = hint.shape
    plus = np.zeros_like(hint) if plus is None else plus
    minus = np.zeros_like(hint) if minus is None else minus
    seed = (hint | plus) & ~minus
    info: Dict = {"mode": "white" if white_ground is not None else "painted"}
    if white_ground is not None:
        raw = ~white_ground
    else:
        k = min(1.0, GRABCUT_SIDE / max(H, W))
        sw, sh = max(8, int(W * k)), max(8, int(H * k))
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        near = lambda m: cv2.resize(m.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0  # noqa: E731
        hs, ps, ms = near(seed), near(plus & ~minus), near(minus)
        r = max(2, int(0.02 * max(sw, sh)))
        gm = np.full((sh, sw), cv2.GC_PR_BGD, np.uint8)
        gm[ndi.binary_dilation(hs, iterations=r)] = cv2.GC_PR_FGD
        core = ndi.binary_erosion(hs, iterations=r)
        gm[core] = cv2.GC_FGD
        gm[~ndi.binary_dilation(_disk_hull(hs), iterations=r)] = cv2.GC_BGD
        gm[ps] = cv2.GC_FGD
        gm[ms] = cv2.GC_BGD
        if (gm == cv2.GC_FGD).any() and (gm == cv2.GC_BGD).any():
            bgm, fgm = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), gm, None, bgm, fgm, 6, cv2.GC_INIT_WITH_MASK)
            fs = (gm == cv2.GC_FGD) | (gm == cv2.GC_PR_FGD)
        else:
            fs = hs
        raw = cv2.resize(fs.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
    raw = (raw | plus) & ~minus
    lab, n = ndi.label(raw)
    keep = np.unique(lab[seed & raw])
    keep = keep[keep > 0]
    raw = np.isin(lab, keep) if len(keep) else raw
    raw = ndi.binary_fill_holes(raw)
    info["pockets"] = []
    if corners:
        pk, info["pockets"] = corner_pockets(raw)
        raw = (raw | pk) & ~minus
    sig = EDGE_SMOOTH * float(np.hypot(W, H))
    sm = ndi.gaussian_filter(raw.astype(np.float32), sig) > 0.5
    sm = ((sm | (plus & ndi.binary_erosion(plus, iterations=2))) & ~minus)
    sm = ndi.binary_fill_holes(sm)
    alpha = ndi.gaussian_filter(sm.astype(np.float32), FEATHER)
    alpha[minus] = np.minimum(alpha[minus], 0.0)
    info["raw"] = raw
    info["solid"] = sm
    info["smooth_px"] = round(sig, 1)
    info["lost"] = float((hint & ~sm).sum() / max(1, hint.sum()))
    info["share"] = float(sm.mean())
    return alpha.astype(np.float32), info


def ground_colour(img: np.ndarray, ground: np.ndarray) -> np.ndarray:
    lab = co.to_lab(img)
    return np.median(lab[ground], axis=0) if ground.any() else np.array([50.0, 0.0, 0.0])


def figure_colours(img: np.ndarray, alpha: np.ndarray, ground_lab: np.ndarray, hint: np.ndarray) -> Tuple[np.ndarray, dict]:
    """The figure's colours with the old ground taken off its edge: the feathered band and the ring just inside the
    silhouette that still shows the old ground get the colour of the figure next to them."""
    from scipy import ndimage as ndi
    H, W = alpha.shape
    solid = alpha > 0.5
    ring_w = max(2, int(round(EDGE_SMOOTH * float(np.hypot(W, H)))))
    ring = solid & ~ndi.binary_erosion(solid, iterations=ring_w)
    lab = co.to_lab(img)
    groundlike = np.linalg.norm(lab - ground_lab, axis=-1) < 10.0
    core = ndi.binary_erosion(hint, iterations=ring_w)
    bad = ((ring & groundlike & ~core) | ((alpha > 0.002) & (alpha < 0.98)))
    donors = solid & ~bad
    f = img.astype(np.float32)
    sig = max(1.5, ring_w / 2.0)
    num = ndi.gaussian_filter(f * donors[..., None], (sig, sig, 0))
    den = ndi.gaussian_filter(donors.astype(np.float32), sig)[..., None]
    est = np.where(den > 1e-3, num / np.maximum(den, 1e-6), f)
    # a pixel far from every donor: widen the reach
    far = bad & (den[..., 0] <= 1e-3)
    if far.any():
        sig2 = sig * 4
        num2 = ndi.gaussian_filter(f * donors[..., None], (sig2, sig2, 0))
        den2 = ndi.gaussian_filter(donors.astype(np.float32), sig2)[..., None]
        est = np.where(far[..., None] & (den2 > 1e-4), num2 / np.maximum(den2, 1e-6), est)
    out = f.copy()
    out[bad] = est[bad]
    ring_ground = float((ring & groundlike & ~core).sum() / max(1, ring.sum()))
    return np.clip(out, 0, 255).astype(np.uint8), {"ring_px": ring_w, "ring_ground": round(ring_ground, 4)}


def edge_spots(img: np.ndarray, alpha: np.ndarray, raw: np.ndarray, n: int = 3) -> List[dict]:
    """The ``n`` weakest places of the silhouette: where the cut runs between similar colours (low contrast: the edge
    is a guess) or where smoothing moved it most from what the colours said. Boxes in pixels."""
    from scipy import ndimage as ndi
    H, W = alpha.shape
    solid = alpha > 0.5
    edge = solid & ~ndi.binary_erosion(solid)
    # the frame is not an edge of the figure
    edge[:2, :] = edge[-2:, :] = False
    edge[:, :2] = edge[:, -2:] = False
    if not edge.any():
        return []
    w = max(8, int(0.025 * float(np.hypot(W, H))))
    lab = co.to_lab(img)
    s = w / 2.0
    a = solid.astype(np.float32)
    ins = ndi.gaussian_filter(lab * a[..., None], (s, s, 0)) / np.maximum(ndi.gaussian_filter(a, s), 1e-3)[..., None]
    outs = ndi.gaussian_filter(lab * (1 - a)[..., None], (s, s, 0)) / \
        np.maximum(ndi.gaussian_filter(1 - a, s), 1e-3)[..., None]
    contrast = np.linalg.norm(ins - outs, axis=-1)
    jag = ndi.uniform_filter((raw != solid).astype(np.float32), w)
    score = np.clip(1.0 - contrast / 20.0, 0, 1) + 2.0 * jag
    ys, xs = np.nonzero(edge)
    sc = score[ys, xs]
    order = np.argsort(-sc)
    picked: List[dict] = []
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        if any(abs(y - q["c"][1]) < 2 * w and abs(x - q["c"][0]) < 2 * w for q in picked):
            continue
        box = (max(0, x - w), max(0, y - w), min(W, x + w), min(H, y + w))
        picked.append({"c": (x, y), "box": box, "contrast": round(float(contrast[y, x]), 1),
                       "moved": round(float(jag[y, x]), 3), "score": round(float(sc[i]), 3)})
        if len(picked) >= n:
            break
    return picked


# ----------------------------------------------------------------------------------------------- background

def cover(src: np.ndarray, W: int, H: int) -> Tuple[np.ndarray, float]:
    """``src`` scaled to cover W x H and centred (the overflow cropped). Returns (picture, share of src cropped)."""
    import cv2
    h, w = src.shape[:2]
    s = max(W / w, H / h)
    nw, nh = max(W, int(round(w * s))), max(H, int(round(h * s)))
    big = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    x0, y0 = (nw - W) // 2, (nh - H) // 2
    crop = 1.0 - (W * H) / float(nw * nh)
    return np.clip(big[y0:y0 + H, x0:x0 + W], 0, 255).astype(np.uint8), crop


def gradient(W: int, H: int, c0: Sequence[int], c1: Sequence[int], angle: float = 90.0) -> np.ndarray:
    """A linear gradient from c0 to c1; angle 90 runs top to bottom, 0 left to right (degrees, y down)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = np.deg2rad(angle)
    d = (xx / max(1, W - 1) - 0.5) * np.cos(t) * W + (yy / max(1, H - 1) - 0.5) * np.sin(t) * H
    span = abs(np.cos(t)) * W + abs(np.sin(t)) * H
    u = np.clip(d / max(span, 1e-6) + 0.5, 0, 1)[..., None]
    a = co.to_lab(np.asarray([[c0]], np.uint8))[0, 0]
    b = co.to_lab(np.asarray([[c1]], np.uint8))[0, 0]
    return co.lab_to_u8(a * (1 - u) + b * u)


def composite(bg: np.ndarray, fig: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    return np.clip(bg.astype(np.float32) * (1 - a) + fig.astype(np.float32) * a + 0.5, 0, 255).astype(np.uint8)


def halo(out: np.ndarray, bg: np.ndarray, alpha: np.ndarray, ground_lab: np.ndarray) -> dict:
    """Old ground left around the figure: in the ring just inside the silhouette, the share of pixels of the old
    ground's colour that are unlike the figure just inside them and unlike the new background there (a dark
    shoulder that is dark through and through is the figure, not a ring)."""
    from scipy import ndimage as ndi
    H, W = alpha.shape
    solid = alpha > 0.5
    w = max(2, int(round(EDGE_SMOOTH * float(np.hypot(W, H)))))
    inner = ndi.binary_erosion(solid, iterations=w)
    ring = solid & ~inner
    lo, lb = co.to_lab(out), co.to_lab(bg)
    s = 1.5 * w
    est = ndi.gaussian_filter(lo * inner[..., None], (s, s, 0)) / \
        np.maximum(ndi.gaussian_filter(inner.astype(np.float32), s), 1e-3)[..., None]
    has = ndi.gaussian_filter(inner.astype(np.float32), s) > 1e-3
    old = np.linalg.norm(lo - ground_lab, axis=-1) < 10.0
    unlike = (np.linalg.norm(lo - est, axis=-1) > 10.0) | ~has
    new = np.linalg.norm(lb - ground_lab, axis=-1) < 20.0
    bad = ring & old & unlike & ~new
    return {"halo": round(float(bad.sum() / max(1, ring.sum())), 4)}


def bright_side(bg: np.ndarray, alpha: Optional[np.ndarray] = None) -> Tuple[str, float]:
    """Where the background's light sits: the side (left / right, or top when centred) of its lightest 3% of
    pixels (a sun, a window, a lit sky), and how far off the centre they are (share of the width, 0..0.5)."""
    L = co.to_lab(bg)[..., 0]
    W = bg.shape[1]
    m = L >= np.quantile(L, 0.97)
    cx = float(np.nonzero(m)[1].mean()) / max(1, W - 1) if m.any() else 0.5
    off = abs(cx - 0.5)
    return ("top" if off < 0.05 else ("left" if cx < 0.5 else "right")), round(off, 3)


def key_colour(bg: np.ndarray) -> np.ndarray:
    """The background's key colour: the mean of its lightest third."""
    L = co.to_lab(bg)[..., 0]
    m = L >= np.quantile(L, 0.67)
    return bg[m].reshape(-1, 3).mean(axis=0)


def light(fig: np.ndarray, alpha: np.ndarray, key: Sequence[float], amount: float, rim: Optional[str],
          rim_rgb: Sequence[float], rim_strength: float = 0.45, cool: Optional[Sequence[float]] = None) -> np.ndarray:
    """The figure lit by the setting: its colours multiplied toward the key colour by ``amount``, a rim of ``rim_rgb``
    on the edges that face ``rim`` (left, right, top), and the far side pushed toward ``cool``."""
    from scipy import ndimage as ndi
    H, W = alpha.shape
    f = fig.astype(np.float32) / 255.0
    k = np.asarray(key, np.float32) / 255.0
    kn = k / max(1e-3, float(k.max()))
    f = f * (1 - amount + amount * kn)
    diag = float(np.hypot(W, H))
    if rim:
        R = max(2.0, 0.004 * diag)
        edge = np.clip((alpha - ndi.gaussian_filter(alpha, R)) * 4, 0, 1)
        g = ndi.gaussian_filter(alpha, R / 2)
        if rim in ("left", "right"):
            d = ndi.sobel(g, axis=1)
            d = d if rim == "left" else -d
        else:
            d = ndi.sobel(g, axis=0)
        side = np.clip(d * 6, 0, 1)
        rr = (edge * side)[..., None] * rim_strength
        f = f + (np.asarray(rim_rgb, np.float32) / 255.0 - f) * rr
        if cool is not None and rim in ("left", "right"):
            xx = np.mgrid[0:H, 0:W][1].astype(np.float32) / max(1, W - 1)
            t = np.clip(((xx - 0.55) if rim == "left" else (0.45 - xx)) * 1.4, 0, 1)[..., None] * amount * 0.8
            cn = np.asarray(cool, np.float32) / 255.0
            cn = cn / max(1e-3, float(cn.max()))
            f = f * (1 - t + t * cn)
    return np.clip(f * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------------------------- contact shadow

def contact(alpha: np.ndarray) -> Optional[dict]:
    """Where the figure stands: its lowest pixels. None when it runs off the bottom of the picture (a bust)."""
    H, W = alpha.shape
    solid = alpha > 0.5
    rows = np.nonzero(solid.any(axis=1))[0]
    if not len(rows):
        return None
    bottom = int(rows[-1])
    if bottom >= H - 2:
        return None
    top = int(rows[0])
    band = solid[max(0, bottom - max(2, int(0.02 * H))):bottom + 1]
    xs = np.nonzero(band.any(axis=0))[0]
    cols = np.nonzero(solid.any(axis=0))[0]
    return {"y": bottom, "x0": int(xs[0]), "x1": int(xs[-1]), "fig_h": bottom - top + 1,
            "fig_w": int(cols[-1] - cols[0] + 1)}


def contact_shadow(alpha: np.ndarray, ground_y: int, direction: str, strength: float = 0.55,
                   length: float = 1.0) -> Tuple[np.ndarray, dict]:
    """A soft shadow on the ground under the figure, falling to ``direction`` (left, right, back, front). Returns
    (darkening 0..1, info); the darkening applies to the background only."""
    import cv2
    from scipy import ndimage as ndi
    H, W = alpha.shape
    c = contact(alpha)
    if c is None:
        return np.zeros((H, W), np.float32), {"contact": False}
    cw = c["x1"] - c["x0"] + 1
    cx = (c["x0"] + c["x1"]) / 2.0
    ax = 0.5 * (cw * 1.3 + 0.15 * c["fig_w"]) * (1.0 + 0.6 * length * (direction in ("left", "right")))
    ay = max(2.0, 0.035 * c["fig_h"])
    shift = {"left": -1, "right": 1}.get(direction, 0) * 0.45 * ax
    cy = ground_y + {"back": -0.5, "front": 0.5}.get(direction, 0.0) * ay
    sh = np.zeros((H, W), np.float32)
    cv2.ellipse(sh, (int(round(cx + shift)), int(round(cy))), (int(round(ax)), int(round(ay))), 0, 0, 360, 1.0, -1)
    sh = ndi.gaussian_filter(sh, max(1.5, 0.6 * ay)) * strength
    core = np.zeros((H, W), np.float32)
    cv2.ellipse(core, (int(round(cx)), int(round(ground_y))), (int(round(0.55 * cw)), int(round(max(1.0, 0.3 * ay)))),
                0, 0, 360, 1.0, -1)
    sh = np.maximum(sh, ndi.gaussian_filter(core, max(1.0, 0.3 * ay)) * min(0.85, strength + 0.25))
    return np.clip(sh, 0, 0.9), {"contact": True, "cx": cx + shift, "cy": cy, "ax": ax, "ay": ay, **c}


def shading_side(fig: np.ndarray, alpha: np.ndarray) -> Tuple[str, float]:
    """The lit side of the figure from its own shading: the lighter half (left / right) and by how much (L)."""
    L = co.to_lab(fig)[..., 0]
    solid = alpha > 0.5
    cols = np.nonzero(solid.any(axis=0))[0]
    if not len(cols):
        return "left", 0.0
    mid = (cols[0] + cols[-1]) // 2
    left, right = solid.copy(), solid.copy()
    left[:, mid:] = False
    right[:, :mid] = False
    lL = float(L[left].mean()) if left.any() else 50.0
    rL = float(L[right].mean()) if right.any() else 50.0
    return ("left" if lL >= rL else "right"), abs(lL - rL)
