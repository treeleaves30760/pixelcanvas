"""Media: painting processes that turn the scene buffers + a style spec into a picture.

Each medium reads only the buffer dict ``b`` (``medium.buffers``) and a Style, and is a process of its own: render
(light, tone, bloom), flat graphic (posterised local colour in value bands), oil (strokes along the form, coarse to
fine, on canvas), watercolour (washes in value bands on paper, pooled edges, granulation, reserved whites), ink
(contours and hatching that wraps the form), pixel art (a low-resolution grid in a small palette with outlines).
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from . import nd
from .core import (F, lin_to_srgb, parse_color, srgb3, blur, smoothstep, luma, rgb_to_hsv, hsv_to_rgb, noise2,
                   fbm2, load)


# ================================================================ shared helpers
def px(b, v):
    """Size given at 1200 px wide -> this resolution."""
    return v * b['albedo'].shape[1] / 1200.0


def lit(b, st, albedo=None):
    """Recompose the lit image (linear) from the separate light terms; the style can re-weight them."""
    alb = b['albedo'] if albedo is None else albedo
    sh = 1 - st.shadow * (1 - b['key_shadow'])
    key = b['key_color'][None, None] * (b['key_diff'] * sh)[..., None]
    out = alb * (key * st.key + b['amb'] * st.ambient + b['pts'] * st.key) + b['spec'] * st.key + b['emit']
    special = (b['transp'] > 0) | (b['id'] < 0) | (luma(b['emit']) > 0.03)
    if 'fog' in b:
        fg = b['fog'][..., None]
        out = out * (1 - fg) + b['fog_col'][None, None] * fg
    if albedo is None:
        out = np.where(special[..., None], b['beauty'], out)
    else:
        out = np.where((b['id'] < 0)[..., None], b['beauty'], out)
    return out.astype(F)


def light_amount(b, st):
    """Scalar irradiance (luminance) at each pixel: key * shadow + ambient + point lights."""
    sh = 1 - st.shadow * (1 - b['key_shadow'])
    k = luma(b['key_color'][None, None] * np.ones_like(b['albedo'])) * b['key_diff'] * sh * st.key
    return (k + luma(b['amb']) * st.ambient + luma(b['pts']) * st.key).astype(F)


def exposure(b):
    L = luma(b['beauty'])
    m = (b['emit'].sum(-1) == 0) & (b['id'] >= 0)
    if m.sum() < 100:
        m = np.ones_like(m)
    return 0.7 / max(float(np.percentile(L[m], 93)), 1e-4)


def aces(x):
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


def tone(b, lin, st, shift=True):
    """Linear -> display sRGB with the style's exposure, contrast, saturation, temperature, shadow colour."""
    x = lin * exposure(b) * st.exposure
    s = lin_to_srgb(aces(x))
    em = luma(b['emit']) > 0.03
    if em.any():
        # light sources: compress luminance only, so flames stay orange instead of clipping to white
        L = luma(x)[..., None] + 1e-6
        c = x * aces(L) / L
        c = c + np.clip(c.max(-1, keepdims=True) - 1, 0, None) * 0.08
        s = np.where(em[..., None], lin_to_srgb(np.clip(c, 0, 1)), s)
    if st.contrast != 1:
        L = luma(s)
        L2 = np.clip(0.5 + (L - 0.5) * st.contrast, 0, 1)
        s = s * ((L2 + 1e-4) / (L + 1e-4))[..., None]
    if st.saturation != 1:
        L = luma(s)[..., None]
        s = L + (s - L) * st.saturation
    if st.temperature:
        t = st.temperature
        s = s * np.array([1 + 0.06 * t, 1 + 0.01 * t, 1 - 0.08 * t], F)
    if shift and st.shadow_shift:
        shade = (1 - smoothstep(0.0, 0.35, b['key_diff'] * b['key_shadow'])) * (b['id'] >= 0)
        k = st.shadow_shift * shade[..., None]
        cool = np.array([0.0, -0.07, 0.05], F)
        s = s + k * cool * (0.3 + luma(s)[..., None])
    return np.clip(s, 0, 1).astype(F)


def screen_normal(b):
    cam = b['cam']
    right, up, fwd = cam[2], cam[3], cam[1]
    n = b['normal']
    return n @ right, -(n @ up), n @ fwd     # x right, y down, z into the screen


def direction_field(b, st, ref=None, sigma=6.0):
    """Stroke / hatch angle per pixel (radians, image coords): along the form (around the object) where the
    surface turns, along the pattern flow (grain, fur) where given, along isophotes of the reference elsewhere."""
    H, W = b['id'].shape
    nx, ny, nz = screen_normal(b)
    pl = np.hypot(nx, ny)
    # contour direction = perpendicular to the projected normal
    fa = np.arctan2(nx, -ny)
    vx, vy = np.cos(2 * fa) * pl, np.sin(2 * fa) * pl
    fl = b['flow']
    fm = np.hypot(fl[..., 0], fl[..., 1]) > 0.5
    aflow = np.arctan2(fl[..., 1], fl[..., 0])
    vx = np.where(fm, np.cos(2 * aflow) * 1.5, vx)
    vy = np.where(fm, np.sin(2 * aflow) * 1.5, vy)
    if ref is not None:
        L = blur(luma(ref), px(b, 3))
        gy, gx = np.gradient(L)
        jxx, jyy, jxy = blur(gx * gx, px(b, 4)), blur(gy * gy, px(b, 4)), blur(gx * gy, px(b, 4))
        ga = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2       # isophote direction
        coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)
        mag = np.sqrt(jxx + jyy)
        w = coh * np.clip(mag * 40, 0, 1) * (0.4 if st.direction == 'form' else 2.0)
        vx, vy = vx + np.cos(2 * ga) * w, vy + np.sin(2 * ga) * w
    a0 = math.radians(st.angle)
    vx, vy = vx + 0.12 * math.cos(2 * a0), vy + 0.12 * math.sin(2 * a0)
    if st.direction == 'fixed':
        vx, vy = np.full((H, W), math.cos(2 * a0), F), np.full((H, W), math.sin(2 * a0), F)
    s = px(b, sigma)
    vx, vy = blur(vx.astype(F), s), blur(vy.astype(F), s)
    return (0.5 * np.arctan2(vy, vx)).astype(F)


def edge_maps(b):
    """One-pixel edge maps: silhouettes (body changes / depth jumps, on the nearer side), inner part lines,
    creases (normal turns)."""
    gid, idm, d, n = b['gid'], b['id'], b['depth'], b['normal']
    H, W = gid.shape
    sil = np.zeros((H, W), bool)
    inner = np.zeros((H, W), bool)
    crease = np.zeros((H, W), bool)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        g2 = np.roll(np.roll(gid, dy, 0), dx, 1)
        i2 = np.roll(np.roll(idm, dy, 0), dx, 1)
        d2 = np.roll(np.roll(d, dy, 0), dx, 1)
        n2 = np.roll(np.roll(n, dy, 0), dx, 1)
        front = d <= d2
        jump = np.abs(d - d2) / np.maximum(d, 1e-3) > 0.04
        sil |= ((g2 != gid) | jump) & front & (gid >= 0)
        inner |= (i2 != idm) & (g2 == gid) & front
        crease |= ((n * n2).sum(-1) < 0.75) & (i2 == idm) & (idm >= 0)
    for a in (sil, inner, crease):
        a[0, :] = a[-1, :] = False
        a[:, 0] = a[:, -1] = False
    return sil, inner, crease & ~sil


def warp_mask(m, amp, scale, seed):
    H, W = m.shape
    dx = (noise2(H, W, scale, seed) - 0.5) * 2 * amp
    dy = (noise2(H, W, scale, seed + 7) - 0.5) * 2 * amp
    yy, xx = np.mgrid[0:H, 0:W].astype(F)
    return nd.sample(m.astype(F), yy + dy, xx + dx)


def draw_lines(mask, width, wobble=0.0, seed=0, scale=40):
    """Ink coverage (H, W) of lines through the mask pixels; `width` (H, W) or scalar, in px."""
    m = mask
    if wobble:
        m = warp_mask(mask, wobble, scale, seed) > 0.35
    return nd.near_lines(m, width)


def paper(H, W, color, tex, seed=3):
    """Cold-press paper: fibres and tooth. Returns (rgb, height)."""
    tooth = fbm2(H, W, 7, 3, seed) * 0.6 + fbm2(H, W, 2.2, 2, seed + 1) * 0.4
    fib = noise2(H, W, 40, seed + 2)
    h = (tooth - 0.5) * tex
    c = np.ones((H, W, 3), F) * srgb3(color)
    gy, gx = np.gradient(blur(tooth, 0.8))
    shade = 1 + (gx * -0.7 + gy * -0.7) * 0.9 * tex + (tooth - 0.5) * 0.05 * tex + (fib - 0.5) * 0.03 * tex
    return np.clip(c * shade[..., None], 0, 1).astype(F), tooth.astype(F)


def canvas_ground(H, W, color, tex, scale=1.0, seed=4):
    yy, xx = np.mgrid[0:H, 0:W].astype(F)
    p = 3.2 * scale
    weave = (np.sin(xx * 2 * np.pi / p) * np.sin(yy * 2 * np.pi / p * 1.07))
    thread = noise2(H, W, 30, seed) - 0.5
    h = (0.5 + 0.35 * weave + 0.3 * thread) * tex
    c = np.ones((H, W, 3), F) * srgb3(color)
    c = c * (1 + 0.08 * tex * (fbm2(H, W, 60, 3, seed + 1) - 0.5))[..., None]
    return np.clip(c, 0, 1).astype(F), h.astype(F)


def relief(img, height, strength, light=(-0.7, -0.7), sheen=0.0):
    gy, gx = np.gradient(height)
    d = (gx * light[0] + gy * light[1]) * strength
    out = img * (1 + d)[..., None]
    if sheen:
        out = out + sheen * np.clip(d, 0, None)[..., None] ** 2
    return np.clip(out, 0, 1).astype(F)


def kmeans(X, k, iters=14, seed=0):
    rng = np.random.default_rng(seed)
    S = X[rng.choice(len(X), min(len(X), 30000), replace=False)]
    C = [S[rng.integers(len(S))]]
    for _ in range(k - 1):
        d = np.min(((S[:, None] - np.array(C)[None]) ** 2).sum(-1), 1)
        C.append(S[rng.choice(len(S), p=d / d.sum())])
    C = np.array(C)
    for _ in range(iters):
        lab = np.argmin(((S[:, None] - C[None]) ** 2).sum(-1), 1)
        for j in range(k):
            if (lab == j).any():
                C[j] = S[lab == j].mean(0)
    return C


def to_lab(c):
    c = np.asarray(c, F)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], F)
    xyz = lin @ M.T / np.array([0.9505, 1.0, 1.089], F)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def nearest_color(img, C, lab=True):
    flat = img.reshape(-1, 3)
    A, B = (to_lab(flat), to_lab(C)) if lab else (flat, C)
    idx = np.empty(len(flat), np.int32)
    for i in range(0, len(flat), 200000):
        idx[i:i + 200000] = np.argmin(((A[i:i + 200000, None] - B[None]) ** 2).sum(-1), 1)
    return C[idx].reshape(img.shape).astype(F), idx.reshape(img.shape[:2])


def region_stats(labels, values, fn='median'):
    """Per-label statistic of an (H, W, C) array -> dict label -> value."""
    out = {}
    for l in np.unique(labels):
        m = labels == l
        v = values[m]
        out[l] = np.median(v, 0) if fn == 'median' else v.mean(0)
    return out


def focus_map(b, ref):
    """Where the eye goes: local contrast near the frame centre and the objects."""
    H, W = b['id'].shape
    L = luma(ref)
    c = np.abs(L - blur(L, px(b, 12)))
    yy, xx = np.mgrid[0:H, 0:W]
    cw = np.exp(-(((xx - W / 2) / (0.35 * W)) ** 2 + ((yy - H * 0.55) / (0.4 * H)) ** 2))
    obj = (b['id'] >= 0) & ~(b['gid'] == np.bincount(b['gid'][b['gid'] >= 0].ravel()).argmax())
    s = blur(c * cw * (0.3 + obj), px(b, 6))
    y, x = np.unravel_index(np.argmax(s), s.shape)
    r = 0.22 * W
    return np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (r * r))).astype(F), (x, y)


def edge_aa(img, b, sigma=0.6):
    sil, inner, _ = edge_maps(b)
    zone = nd.dilate(sil | inner, 1)
    return np.where(zone[..., None], blur(img, sigma), img).astype(F)


def detail_map(b, scale=2.5):
    """The drawing's own fine dark detail in the local colour (eyes, outlines, labels, window frames, stripes'
    edges): 0..1 where the albedo is much darker than its surroundings at a small scale."""
    L = np.log(luma(b['albedo']) + 0.01)
    d = blur(L, px(b, scale)) - L
    d2 = blur(L, px(b, scale * 3)) - L
    m = np.clip(np.maximum(d * 2.2, d2 * 1.2) - 0.08, 0, 1)
    return (m * (b['id'] >= 0)).astype(F)


# ================================================================ 1. render
def keep_colour(img, b, amount=1.0):
    """Each layer keeps its drawn local colour: per pixel the hue and saturation (Oklab a, b) of the drawing, and
    per layer the drawing's median lightness, with the medium's light and shade as the variation around it. Light
    changes value and adds a tinted shadow or highlight; it never bleaches or shifts a hue. Speculars and emission
    keep their own colour."""
    if b.get("drawn") is None or amount <= 0:
        return img
    from .core import srgb_to_oklab, oklab_to_srgb
    d = srgb_to_oklab(b["drawn"])
    r = srgb_to_oklab(img)
    gid = b["gid"]
    L = r[..., 0].copy()
    for k in np.unique(gid):
        own = gid == k
        if own.sum() < 16:
            continue
        L[own] += float(np.median(d[..., 0][own]) - np.median(r[..., 0][own]))
    hot = np.clip(luma(b["spec"]) * 3 + luma(b["emit"]) * 2, 0, 1)            # highlights and lights: their colour
    ab = d[..., 1:] * (1 - hot[..., None]) + r[..., 1:] * hot[..., None]
    shift = r[..., 0] - d[..., 0]                                               # a tinted shadow: a little cooler
    ab = ab + np.stack([-0.01 * np.clip(-shift, 0, 0.3), -0.03 * np.clip(-shift, 0, 0.3)], -1)
    out = np.concatenate([np.clip(L, 0, 1)[..., None], ab], -1)
    res = np.clip(oklab_to_srgb(out), 0, 1)
    em = np.clip(luma(b["emit"]) * 4, 0, 1)[..., None]     # a lamp, a flame, a glow: as the drawing shows it
    res = res * (1 - em) + b["drawn"] * em
    return (img * (1 - amount) + res * amount).astype(F)


def render(b, st):
    lin = lit(b, st)
    if st.bloom:
        k = exposure(b) * st.exposure
        hi = np.clip(b['emit'] * k - 0.9, 0, None)        # bloom only from real emission
        bl = sum(blur(hi, px(b, s)) * w for s, w in ((4, 0.5), (16, 0.35), (48, 0.25)))
        lin = lin + bl * st.bloom / k
    img = tone(b, lin, st, shift=False)
    img = edge_aa(img, b)
    H, W = b['id'].shape
    yy, xx = np.mgrid[0:H, 0:W]
    vig = 1 - 0.18 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    img = keep_colour(img, b, 1.0)
    return np.clip(img * (0.7 + 0.3 * vig)[..., None], 0, 1)


# ================================================================ 2. flat graphic
def flat(b, st):
    H, W = b['id'].shape
    idm = b['id']
    # a flat local colour per object (the median albedo), then value bands from the light amount
    # local colour per object: smoothed inside the object, then posterised to a few flat colours (1 for a
    # plain object, more for patterned / multi-coloured ones: giraffe patches, a motorcycle's parts)
    alb = np.zeros_like(b['albedo'])
    sa = px(b, 2 + 4 * st.simplify)
    for l in np.unique(idm):
        m = idm == l
        mf = m.astype(F)
        sm = blur(b['albedo'] * mf[..., None], sa) / np.maximum(blur(mf, sa), 1e-3)[..., None]
        X = sm[m]
        spread = float(np.sqrt(((X - X.mean(0)) ** 2).sum(1)).mean())
        n = 1 if spread < 0.03 else (3 if spread < 0.06 else (5 if spread < 0.12 else 7))
        if n == 1 or len(X) < 50:
            alb[m] = np.median(X, 0)
        else:
            C = kmeans(np.sqrt(np.clip(X, 0, 1)), n, iters=8)
            q, _ = nearest_color(np.sqrt(np.clip(X, 0, 1))[None], C, lab=False)
            alb[m] = q[0] ** 2
    lightv = light_amount(b, st)
    ref_lit = lit(b, st, albedo=np.ones_like(alb) * 0.6)
    k = exposure(b) * st.exposure
    v = luma(aces(ref_lit * k)) / 0.6
    s = px(b, 2 + 10 * st.simplify)
    vs = np.zeros_like(v)
    for l in np.unique(idm):                 # smooth inside each object only
        m = (idm == l).astype(F)
        vs += blur(v * m, s) / np.maximum(blur(m, s), 1e-3) * m
    v = vs
    nb = max(st.bands, 2)
    # thresholds: light / half / shadow
    th = np.quantile(v[idm >= 0], np.linspace(0, 1, nb + 1)[1:-1]) if nb > 3 else \
        [0.18, 0.42][:nb - 1]
    band = np.zeros((H, W), np.int32)
    for t in th:
        band += (v > t)
    level = np.array([0.45, 0.72, 1.0, 1.12, 1.2][:nb]) if nb <= 3 else np.linspace(0.45, 1.1, nb)
    if nb == 2:
        level = np.array([0.55, 1.0])
    base = tone(b, alb * 0.9 / k / 0.55 * 0.6, st.but(shadow_shift=0, contrast=1.0), shift=False)
    hsv = rgb_to_hsv(base)
    lv = level[band]
    shade = (band < nb - 1)
    hsv[..., 2] = np.clip(hsv[..., 2] * lv, 0, 1)
    hsv[..., 1] = np.clip(hsv[..., 1] * np.where(shade, 1.12, 0.95) * st.saturation, 0, 1)
    hsv[..., 0] = hsv[..., 0] + np.where(shade, 0.025 * st.shadow_shift * (nb - 1 - band), 0)
    img = hsv_to_rgb(hsv)
    # cool shadow hue toward blue-violet for shade bands
    img = img + (shade * st.shadow_shift * 0.05)[..., None] * np.array([-0.4, -0.2, 0.6], F)
    # emissive and sky keep their own tone, posterised
    special = (luma(b['emit']) > 0.03) | (idm < 0) | (b['transp'] > 0)
    sp = tone(b, lit(b, st), st, shift=False)
    sp = np.round(sp * 5) / 5
    em = luma(b['emit']) > 0.03
    if em.any():
        e = b['emit'] / (b['emit'].max(-1, keepdims=True) + 1e-6)
        eh = rgb_to_hsv(np.clip(e, 0, 1))
        eh[..., 2] = 1.0
        eh[..., 1] = np.round(np.clip(eh[..., 1] * 1.2, 0, 1) * 3) / 3
        sp = np.where(em[..., None], hsv_to_rgb(eh), sp)
    img = np.where(special[..., None], sp, img)
    # highlight shapes
    spec = blur(luma(b['spec']) * k, px(b, 1.5))
    hl = spec > 0.35
    img = np.where(hl[..., None], np.clip(base * 0.3 + 0.78, 0, 1), img)
    if st.line:
        sil, inner, _ = edge_maps(b)
        ink = draw_lines(sil, px(b, st.line))
        img = img * (1 - ink[..., None]) + srgb3(st.line_color) * ink[..., None]
    # the drawing's own detail (eyes, labels, frames) as flat dark line work
    det = smoothstep(0.25, 0.5, detail_map(b, 1.5))
    img = img * (1 - det[..., None]) + (img * 0.25 + srgb3(st.line_color) * 0.5) * det[..., None]
    # clean edges: smooth band boundaries
    return np.clip(edge_aa(img, b, 0.7), 0, 1)


# ================================================================ 3. oil / gouache
BRISTLES = None


def _bristles(rng, n=24, m=48):
    out = []
    for _ in range(n):
        p = np.convolve(rng.random(m), np.ones(3) / 3, 'same')
        p = (p - p.min()) / (p.max() - p.min() + 1e-6)
        out.append(p.astype(F))
    return out


def _stroke(canvas, height, cx, cy, ang, curv, L, w, col, alpha, prof, tex):
    H, W = canvas.shape[:2]
    half = L / 2 + w
    x0, x1 = int(max(cx - half, 0)), int(min(cx + half + 1, W))
    y0, y1 = int(max(cy - half, 0)), int(min(cy + half + 1, H))
    if x1 <= x0 or y1 <= y0:
        return
    ys = np.arange(y0, y1, dtype=F)[:, None] - cy
    xs = np.arange(x0, x1, dtype=F)[None, :] - cx
    ca, sa = math.cos(ang), math.sin(ang)
    u = xs * ca + ys * sa
    v = -xs * sa + ys * ca - 0.5 * curv * u * u
    t = u / (L / 2)
    hw = w / 2 * (1.0 - 0.22 * t)
    vn = v / hw
    env = smoothstep(1.0, 0.8, np.abs(t)) * smoothstep(1.0, 0.7, np.abs(vn))
    if not env.any():
        return
    pi = np.clip(((vn + 1) * 0.5 * (len(prof) - 1)).astype(np.int32), 0, len(prof) - 1)
    pr = prof[pi]
    dry = np.clip((1 - (t + 1) / 2) * 1.8 + pr * 0.9 - 0.35, 0, 1) if tex else 1.0
    a = alpha * env * (0.6 + 0.4 * pr if tex else 1.0) * dry
    a = a[..., None]
    c = col[None, None, :] * (1 + 0.1 * tex * (pr - 0.5))[..., None]
    sl = (slice(y0, y1), slice(x0, x1))
    canvas[sl] = canvas[sl] * (1 - a) + c * a
    height[sl] = height[sl] * (1 - a[..., 0]) + a[..., 0] * (0.35 + 0.65 * pr * env)


def oil(b, st, finish=True, seed=1):
    rng = np.random.default_rng(seed)
    H, W = b['id'].shape
    ref = tone(b, lit(b, st), st)
    if st.bands:
        # value banding (poster looks): quantise lightness in HSV, keep hue
        hsv = rgb_to_hsv(blur(ref, px(b, 2)))
        hsv[..., 2] = (np.floor(hsv[..., 2] * st.bands) + 0.5) / st.bands
        ref = hsv_to_rgb(hsv)
    if st.palette_colors:
        C = np.array([parse_color(c) for c in st.palette_colors], F)
        ref, _ = nearest_color(blur(ref, px(b, 1.5)), C)
    elif st.palette:
        C = kmeans(ref.reshape(-1, 3), st.palette, seed=seed)
        ref, _ = nearest_color(ref, C)
    angle = direction_field(b, st, ref, sigma=8)
    nx, ny, _ = screen_normal(b)
    flatness = 1 - np.clip(np.hypot(nx, ny) * 1.5, 0, 1)
    angle = angle + (noise2(H, W, px(b, 110), seed + 3) - 0.5) * 1.0 * flatness * (st.direction != 'fixed')
    canvas, hgt = canvas_ground(H, W, st.ground_color, st.texture, px(b, 1.0))
    hgt = hgt * 0.4
    ground_h = hgt.copy()
    tex = st.extra.get('impasto', 1.0)
    passes = st.extra.get('passes', 5)
    thr = st.extra.get('threshold', 0.05)
    R = px(b, st.mark_size)
    rmin = px(b, st.mark_min)
    radii = list(np.geomspace(R, rmin, passes))
    focus, _ = focus_map(b, ref)
    idm = b['gid']
    global BRISTLES
    BRISTLES = BRISTLES or _bristles(np.random.default_rng(7))
    n_strokes = 0
    for pi, r in enumerate(radii):
        last = pi == len(radii) - 1
        if last and not finish:
            break
        refi = blur(ref, max(r * 0.45 * (0.5 + st.simplify), 0.5))
        err = np.abs(blur(canvas, r * 0.6) - refi).sum(-1) / 3
        err = blur(err, r * 0.5)
        sp = r * 0.75
        gy, gx = np.mgrid[0:H:sp, 0:W:sp]
        gy = (gy + rng.uniform(-sp / 2, sp / 2, gy.shape)).ravel()
        gx = (gx + rng.uniform(-sp / 2, sp / 2, gx.shape)).ravel()
        gy, gx = np.clip(gy, 0, H - 1), np.clip(gx, 0, W - 1)
        iy, ix = gy.astype(int), gx.astype(int)
        e = err[iy, ix]
        if pi == 0:
            keep = np.ones(len(e), bool)
        else:
            t_here = thr * (1.0 + 0.5 * pi / len(radii)) * (1.4 - 0.9 * focus[iy, ix])
            keep = e > t_here
        order = rng.permutation(np.where(keep)[0])
        for j in order:
            cx, cy = gx[j], gy[j]
            yi, xi = iy[j], ix[j]
            a = angle[yi, xi] + rng.normal(0, 0.12)
            L = r * rng.uniform(2.2, 4.2) * (0.8 if last else 1.0)
            # keep the stroke inside its body: walk both ways until the id changes
            ca, sa = math.cos(a), math.sin(a)
            g0 = idm[yi, xi]
            ext = []
            for sgn in (1, -1):
                reach = L / 2
                for s in np.linspace(r * 0.5, L / 2, 5):
                    qx, qy = int(cx + sgn * ca * s), int(cy + sgn * sa * s)
                    if not (0 <= qx < W and 0 <= qy < H) or idm[qy, qx] != g0:
                        reach = max(s - r * 0.5, r * 0.3)
                        break
                ext.append(reach)
            Le = ext[0] + ext[1]
            shift = (ext[0] - ext[1]) / 2
            cxe, cye = cx + ca * shift, cy + sa * shift
            # bend with the field
            qx1, qy1 = int(np.clip(cxe + ca * Le / 2, 0, W - 1)), int(np.clip(cye + sa * Le / 2, 0, H - 1))
            qx0, qy0 = int(np.clip(cxe - ca * Le / 2, 0, W - 1)), int(np.clip(cye - sa * Le / 2, 0, H - 1))
            da = (angle[qy1, qx1] - angle[qy0, qx0] + np.pi / 2) % np.pi - np.pi / 2
            curv = float(np.clip(da / max(Le, 1), -0.08, 0.08))
            col = refi[yi, xi].copy()
            if st.jitter:
                # broken colour: value and a little warm/cool shift, never confetti
                vj = 1 + rng.normal(0, st.jitter)
                tj = rng.normal(0, st.jitter * 0.5)
                col = np.clip(col * vj + tj * np.array([0.5, 0.1, -0.5]), 0, 1)
            w = r * rng.uniform(0.85, 1.25)
            alpha = 0.92 if pi else 0.85
            _stroke(canvas, hgt, cxe, cye, a, curv, Le, w, col, alpha, BRISTLES[rng.integers(len(BRISTLES))], tex)
            n_strokes += 1
    if finish:
        # highlights: thick light dabs where the surface catches the key light
        k = exposure(b) * st.exposure
        spec = blur(luma(b['spec']) * k, px(b, 1))
        ys, xs = np.nonzero((spec > 0.4) & (rng.random(spec.shape) < 0.02))
        for y, x in zip(ys, xs):
            col = np.clip(ref[y, x] * 0.4 + 0.65, 0, 1)
            _stroke(canvas, hgt, x, y, angle[y, x], 0, px(b, rng.uniform(5, 9)), px(b, rng.uniform(2.5, 4)), col,
                    0.9, BRISTLES[rng.integers(len(BRISTLES))], tex)
        # edge control: soften (drag) silhouettes away from the focus, keep them sharp at the focus
        sil, _, _ = edge_maps(b)
        zone = blur(sil.astype(F), px(b, 2.5)) * (1 - focus) * (1 - st.edge)
        canvas = canvas * (1 - zone[..., None]) + blur(canvas, px(b, 2.5)) * zone[..., None]
    if finish:
        det = smoothstep(0.25, 0.55, detail_map(b, 1.5))
        det = blur(det, px(b, 0.7))
        canvas = canvas * (1 - 0.75 * det[..., None]) + blur(ref, px(b, 1)) * 0.75 * det[..., None] * 0.5
    out = relief(canvas, blur(hgt, 0.8) + ground_h * 0.5, 0.9 * tex + 0.5, sheen=0.12 * tex)
    oil.n_strokes = n_strokes
    return out


# ================================================================ 4. watercolour
def _soft_threshold(field, t, e):
    return smoothstep(t - e, t + e, field)


def watercolour(b, st, finish=True, seed=2):
    H, W = b['id'].shape
    rng = np.random.default_rng(seed)
    ex = st.extra
    ref = tone(b, lit(b, st), st)
    idm = b['id']
    # region-wise simplification (no colour leaks across silhouettes before we choose to bleed)
    sig = px(b, 2 + 10 * st.simplify)
    T = np.zeros_like(ref)
    for l in np.unique(idm):
        m = (idm == l).astype(F)
        T += blur(ref * m[..., None], sig) / np.maximum(blur(m, sig), 1e-3)[..., None] * m[..., None]
    T = np.clip(T, 0.003, 1)
    Lg = np.log(np.clip(luma(T), 1e-3, 1))
    logT = np.log(T)
    pap, tooth = paper(H, W, st.ground_color, st.texture)
    K = max(st.bands, 2)
    step = 0.32 / max(st.contrast, 0.3)
    # reserved whites: speculars and the brightest lights stay paper
    k = exposure(b) * st.exposure
    hl = blur(luma(b['spec']) * k, px(b, 1.2)) + np.clip(luma(ref) - 0.9, 0, 1) * 3
    reserve = smoothstep(0.5, 0.62, warp_mask(hl, px(b, 1.2), px(b, 8), 11))
    # wetness map: where edges get lost (wet-in-wet)
    wet = smoothstep(0.55, 0.7, fbm2(H, W, px(b, 120), 3, 21)) * ex.get('bleed', 0.35) / 0.35
    wet = np.clip(wet, 0, 1)
    # picture border: unfinished ragged edge
    yy, xx = np.mgrid[0:H, 0:W].astype(F)
    dborder = np.minimum.reduce([xx, W - 1 - xx, yy, H - 1 - yy]) / px(b, 1)
    bn = fbm2(H, W, px(b, 160), 3, 31) - 0.5
    bfield = dborder + bn * 80 + (fbm2(H, W, px(b, 18), 2, 32) - 0.5) * 10
    border_k = (lambda kb: smoothstep(6 + 14 * kb, 9 + 14 * kb, bfield - 10)) if ex.get('border') else         (lambda kb: 1.0)
    border = border_k(0)
    gran = ex.get('granulation', 0.5)
    pool = ex.get('pooling', 0.9)
    out = pap.copy()
    washes = []
    # local colour (the first wash) is separated from shading (the glazes): the fully lit colour per pixel
    bl = dict(b)
    bl['key_diff'] = np.full_like(b['key_diff'], 0.8)
    bl['key_shadow'] = np.ones_like(b['key_shadow'])
    bl['ao'] = np.ones_like(b['ao'])
    bl['amb'] = b['amb'] / np.maximum(b['ao'], 0.2)[..., None]
    Tl = tone(bl, lit(bl, st), st, shift=False)
    Tls = np.zeros_like(Tl)
    for l in np.unique(idm):
        m = (idm == l).astype(F)
        Tls += blur(Tl * m[..., None], sig) / np.maximum(blur(m, sig), 1e-3)[..., None] * m[..., None]
    Tls = np.clip(Tls, 0.003, 1)
    logTl = np.log(Tls)
    rel = np.clip(np.log(np.clip(luma(Tls), 1e-3, 1)) - Lg, 0, None)      # how much darker than fully lit
    band_cols = np.ones((K, H, W, 3), F)
    band_cols[0] = Tls ** 0.85
    noise_e = (fbm2(H, W, px(b, 25), 3, 41) - 0.5)
    ratio = logT - logTl                                                   # shading colour (log), per pixel
    for l in np.unique(idm):
        m = idm == l
        r = rel[m]
        rt = ratio[m]
        prev = np.zeros(3)
        for kb in range(1, K):
            selb = (r >= kb * step) & (r < (kb + 1) * step) if kb < K - 1 else (r >= kb * step)
            if selb.sum() < 10:
                continue
            mu = rt[selb].mean(0)
            band_cols[kb][m] = np.clip(np.exp(np.minimum(mu - prev, 0)), 0.6, 1)
            prev = mu
    for kb in range(K):
        if kb == 0:
            D = np.ones((H, W), F)
        else:
            e = px(b, 0.04) + 0.02 + (1 - st.edge) * 0.06 + wet * 0.12
            D = _soft_threshold(blur(rel, px(b, 1.5)) + noise_e * 0.08, kb * step, e)
        C = band_cols[kb]
        # pigment variation inside a wash: warm / cool and value drift at large scale
        var = fbm2(H, W, px(b, 140), 3, 60 + kb) - 0.5
        C = np.clip(C * (1 + var[..., None] * np.array([0.16, 0.03, -0.16], F) * (1.0 if kb == 0 else 0.6)), 0, 1)
        # bleeding: in wet zones the wash colour runs across region edges
        cb = blur(C * D[..., None], px(b, 7)) / np.maximum(blur(D, px(b, 7)), 1e-3)[..., None]
        wz = (wet * ex.get('bleed', 0.35) / 0.35)[..., None] if kb == 0 else wet[..., None]
        C = C * (1 - wz) + cb * wz
        if kb > 0:
            Db = blur(D, px(b, 5))
            D = D * (1 - wet) + Db * wet
        D = D * (1 - reserve) * border_k(kb)
        # pooling at dried edges, granulation in the paper tooth, uneven wash
        edge_pool = np.clip(D - blur(D, px(b, 2.5)), 0, None) * pool * (1 - wet)
        uneven = 1 + 0.25 * (fbm2(H, W, px(b, 60), 3, 50 + kb) - 0.5)
        dens = D * uneven * (1 + gran * (tooth - 0.5) * 1.0) + edge_pool * 1.8
        out = out * np.power(np.clip(C, 0.02, 1), dens[..., None])
        washes.append(D)
    em = luma(b['emit']) > 0.03
    if em.any():
        # light sources: core left as paper, a warm wet-in-wet glow wash around them
        e = b['emit'] / (b['emit'].max(-1, keepdims=True) + 1e-6)
        ecol = (blur(e * em[..., None], px(b, 6)) / np.maximum(blur(em.astype(F), px(b, 6)), 1e-3)[..., None])
        glow = np.clip(blur(em.astype(F), px(b, 40)) * 3.0, 0, 1) * (1 - smoothstep(0.3, 0.9, blur(em.astype(F),
                                                                                                  px(b, 3))))
        gc = np.clip(blur(ecol * glow[..., None], px(b, 20)) / np.maximum(blur(glow, px(b, 20)), 1e-3)[..., None],
                     0, 1)
        out = out * np.power(np.clip(gc * 0.9 + 0.1, 0.05, 1), (glow * 0.8)[..., None])
        out = np.where(em[..., None], out * 0 + pap * np.power(np.clip(ecol, 0.05, 1), 0.35), out)
    if finish:
        # dark accents at contacts and core shadows near the focus, crisp edges
        focus, _ = focus_map(b, ref)
        acc = (1 - b['ao']) * 1.4 + np.clip(0.18 - luma(ref), 0, 1) * 3
        acc = _soft_threshold(warp_mask(acc, px(b, 1.5), px(b, 10), 61), 0.35, 0.05) * (0.4 + 0.6 * focus)
        acc *= border * (b['id'] >= 0)
        accent_col = np.clip(np.exp(blur(logT, px(b, 3))) * 0.55, 0.05, 1)
        out = out * np.power(accent_col, (acc * 0.9 * (1 + 0.6 * (tooth - 0.5)))[..., None])
        # dry-brush strokes on textured / flow surfaces (grain): broken marks following the flow
        fl = np.hypot(b['flow'][..., 0], b['flow'][..., 1]) > 0.5
        if fl.any():
            ang = direction_field(b, st, None, sigma=4)
            yy2, xx2 = yy, xx
            ph = (xx2 * np.sin(ang) - yy2 * np.cos(ang)) / px(b, 3.5)
            streak = (np.sin(ph * 2 * np.pi + noise2(H, W, px(b, 20), 71) * 6) > 0.6) & (tooth > 0.52)
            mark = streak & fl & (fbm2(H, W, px(b, 50), 2, 72) > 0.5)
            out = out * (1 - 0.18 * (mark * border)[..., None] * (1 - np.exp(blur(logT, 2))))
        # splatter
        n = int(60 * W / 1200)
        for _ in range(n):
            x, y = rng.integers(0, W), rng.integers(0, H)
            if b['id'][y, x] < 0 or rng.random() > 0.6:
                continue
            rr = px(b, rng.uniform(1, 3.2))
            d2 = (xx - x) ** 2 + (yy - y) ** 2
            spot = (d2 < rr * rr) * 0.5
            out = out * np.power(np.clip(np.exp(logT[y, x]) * 0.8, 0.05, 1), spot[..., None])
    # the drawing's detail: a last dark glaze with a small brush (eyes, windows, labels survive)
    det = np.clip(detail_map(b, 1.5) * 1.6, 0, 1)
    det = warp_mask(det, px(b, 0.6), px(b, 6), 77) * border
    dcol = np.clip(np.exp(blur(logT, px(b, 1))) * 0.45, 0.04, 1)
    out = out * np.power(dcol, (det * 0.9)[..., None])
    # pencil underdrawing shows through the transparent washes
    if ex.get('pencil'):
        sil, inner, crease = edge_maps(b)
        lines = draw_lines(sil | inner, px(b, 1.1), wobble=px(b, 1.5), seed=81, scale=px(b, 60))
        lines2 = draw_lines(sil, px(b, 0.8), wobble=px(b, 2.5), seed=82, scale=px(b, 90))
        g = np.clip(lines * 0.8 + lines2 * 0.5, 0, 1) * ex['pencil'] * (0.5 + 0.5 * tooth)
        out = out * (1 - g[..., None] * np.array([0.55, 0.55, 0.5], F))
    watercolour.washes = washes
    return np.clip(out, 0, 1).astype(F)


# ================================================================ 5. ink
AXES = [np.array(a, F) / np.linalg.norm(a) for a in
        [(0, 1, 0), (1, 0.15, -0.35), (0.6, 0.6, 0.3), (-0.5, 0.2, 0.8), (0, 0, 1), (1, 0, 0)]]


def ink(b, st, seed=5):
    H, W = b['id'].shape
    ex = st.extra
    pap, tooth = paper(H, W, st.ground_color, st.texture, seed=9)
    # ink tone follows the light more than the local colour (albedo compressed toward grey)
    alb = 0.55 * (np.clip(b['albedo'], 0, 1) / 0.55) ** 0.6
    ref = tone(b, lit(b, st, albedo=alb), st, shift=False)
    T = blur(luma(ref), px(b, 1.0 + 3 * st.simplify))
    nb = max(st.bands, 2)
    idm = b['id']
    pos, depth = b['pos'], b['depth']
    cam = b['cam']
    H_, W_ = H, W
    tf = 1.0
    ink_cov = np.zeros((H, W), F)
    spacing = px(b, st.mark_size)
    layers = ex.get('hatch_layers', nb - 1)
    thresholds = np.linspace(0.86, 0.3, max(layers, 1))
    nz = (noise2(H, W, px(b, 30), 91) - 0.5)
    if 'hatch' in st.mark:
        for l in np.unique(idm):
            m = idm == l
            if l < 0 or (b['emit'][m].sum(-1) > 0).mean() > 0.5:
                continue
            ys, xs = np.nonzero(m)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            mm = m[y0:y1, x0:x1]
            P = pos[y0:y1, x0:x1]
            dmed = float(np.median(depth[m]))
            pxw = dmed * 2 * 0.3 / H        # world size of a pixel at that depth (approx fov 33deg)
            mn = b['normal'][m].mean(0)
            mn /= np.linalg.norm(mn) + 1e-9
            cand = [a for a in AXES if abs(a @ mn) < 0.75]
            Tl = T[y0:y1, x0:x1]
            for li in range(max(layers, 1)):
                a = cand[li % len(cand)]
                phi = (P @ a) / (spacing * pxw) + nz[y0:y1, x0:x1] * 0.6
                gyp, gxp = np.gradient(phi)
                g = np.hypot(gxp, gyp)
                g = np.where(mm, g, np.nan)
                gmed = np.nanmedian(g) if np.isfinite(g).any() else 1
                g = np.clip(np.nan_to_num(g, nan=gmed), gmed * 0.3, gmed * 3)
                dist_px = np.abs((phi % 1.0) - 0.5) / g
                t = thresholds[li]
                width = px(b, 1.6) * np.clip((t - Tl) / 0.18, 0, 1.4)
                line = smoothstep(width / 2 + 0.6, width / 2 - 0.4, dist_px) * (width > 0.15)
                ink_cov[y0:y1, x0:x1] = np.maximum(ink_cov[y0:y1, x0:x1], line * mm)
    if 'hatch' in st.mark:
        sky = (idm < 0) & (T < thresholds[0])
        if sky.any():
            yy, xx = np.mgrid[0:H, 0:W].astype(F)
            for li, t in enumerate(thresholds[:1]):
                ang = math.radians(8 + 75 * li)
                phi = (xx * math.sin(ang) + yy * math.cos(ang)) / (spacing * 1.2) + nz * 0.8
                dist_px = np.abs((phi % 1.0) - 0.5) * spacing * 1.2
                width = px(b, 1.4) * np.clip((t - T) / 0.18, 0, 1.4)
                line = smoothstep(width / 2 + 0.6, width / 2 - 0.4, dist_px) * (width > 0.15)
                ink_cov = np.maximum(ink_cov, line * sky)
    if 'wash' in st.mark:
        wc = srgb3(ex.get('wash_color', '#2a2a2a'))
        dens = np.zeros((H, W), F)
        wet = smoothstep(0.55, 0.7, fbm2(H, W, px(b, 100), 3, 23))
        for kb, t in enumerate(np.linspace(0.6, 0.22, nb)):
            D = _soft_threshold(-T + (noise2(H, W, px(b, 25), 93 + kb) - 0.5) * 0.05, -t, 0.015 + 0.08 * wet)
            D = D * (idm >= 0)
            pool_ = np.clip(D - blur(D, px(b, 2)), 0, None) * 1.5 * (1 - wet)
            dens += (D * (0.8 + 0.5 * (tooth - 0.5)) + pool_) * 0.45
        pap = pap * np.power(np.clip(wc + 0.25, 0, 1), dens[..., None])
    # spot blacks: the deepest shadows filled solid
    sb = ex.get('spot_black', 0)
    if sb:
        thr_b = min(float(np.quantile(T[idm >= 0], sb)), 0.12 + 0.6 * sb)
        dark = _soft_threshold(-T, -thr_b, 0.01)
        dark = warp_mask(dark, px(b, 1.2), px(b, 10), 95)
        solid = dark * (idm >= 0)
    else:
        solid = 0
    # the drawing's detail as pen lines (eyes, labels, frames)
    det = smoothstep(0.2, 0.45, detail_map(b, 1.5))
    ink_cov = np.maximum(ink_cov, det * 0.95)
    # contours with weight: heavier on silhouettes and on the shadow side
    if st.line:
        sil, inner, crease = edge_maps(b)
        shade = 1 - np.clip(T * 1.2, 0, 1)
        wnoise = 0.75 + 0.5 * noise2(H, W, px(b, 25), 97)
        w = np.where(sil, px(b, st.line) * (0.7 + 1.0 * shade), np.where(inner, px(b, st.line) * 0.6,
                                                                         px(b, st.line) * 0.45)) * wnoise
        lines = draw_lines(sil | inner | crease, w.astype(F), wobble=px(b, 1.0), seed=98, scale=px(b, 50))
        ink_cov = np.maximum(ink_cov, lines)
    # emissive things: ink leaves them paper, ringed by radiating strokes is too much -> keep as paper
    inkc = srgb3(st.line_color)
    ink_cov = np.maximum(ink_cov * (0.88 + 0.25 * (tooth - 0.5)), solid)
    out = pap * (1 - ink_cov[..., None]) + inkc * ink_cov[..., None]
    return np.clip(out, 0, 1).astype(F)


# ================================================================ 6. pixel art
def pixel(b, st, seed=6):
    H, W = b['id'].shape
    p = int(round(px(b, st.extra.get('pixel', 6))))
    h, w = H // p, W // p
    ref = tone(b, lit(b, st), st)[:h * p, :w * p]
    gid = b['gid'][:h * p, :w * p]
    idm = b['id'][:h * p, :w * p]
    dep = b['depth'][:h * p, :w * p]
    # majority body per block = centre sample; colour = mean over the block's pixels of that body
    cy, cx = p // 2, p // 2
    gl = gid[cy::p, cx::p][:h, :w]
    il = idm[cy::p, cx::p][:h, :w]
    dl = dep[cy::p, cx::p][:h, :w]
    blocks = ref.reshape(h, p, w, p, 3).transpose(0, 2, 1, 3, 4).reshape(h, w, p * p, 3)
    bid = idm.reshape(h, p, w, p).transpose(0, 2, 1, 3).reshape(h, w, p * p)
    same = (bid == il[..., None])[..., None]
    low = (blocks * same).sum(2) / np.maximum(same.sum(2), 1)
    # value bands per body with hue shift (cool shadows, warm lights)
    hsv = rgb_to_hsv(low)
    nb = max(st.bands, 2)
    v = hsv[..., 2]
    q = np.clip(np.floor(v * nb) / (nb - 1), 0, 1)
    vq = 0.12 + 0.88 * q
    dv = vq - v
    hsv[..., 2] = np.clip(v + dv * 0.85, 0, 1)
    hsv[..., 0] = hsv[..., 0] + np.where(dv < 0, 0.02, -0.01) * st.shadow_shift * np.abs(dv) * 4
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + 0.25 * (1 - q)), 0, 1)
    low = hsv_to_rgb(hsv)
    if st.extra.get('dither'):
        bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], F) / 16 - 0.5
        low = low + np.tile(bayer, (h // 4 + 1, w // 4 + 1))[:h, :w, None] * st.extra['dither'] / nb
    C = kmeans(low.reshape(-1, 3), st.palette or 20, seed=seed)
    low, lab = nearest_color(low, C)
    # clean orphans: a pixel that differs from all 4 neighbours inside the same body takes the majority
    for _ in range(1):
        nbrs = [np.roll(lab, s, a) for s, a in ((1, 0), (-1, 0), (1, 1), (-1, 1))]
        nbg = [np.roll(gl, s, a) for s, a in ((1, 0), (-1, 0), (1, 1), (-1, 1))]
        orphan = np.all([(n != lab) & (g == gl) for n, g in zip(nbrs, nbg)], 0)
        stack = np.stack(nbrs, -1)
        maj = np.array([np.bincount(r).argmax() for r in stack[orphan]], np.int32) if orphan.any() else []
        lab[orphan] = maj
        low = C[lab]
    # dark outlines on body silhouettes (selective: a darker shade of the body's own colour)
    if st.line:
        out_m = np.zeros((h, w), bool)
        for s, a in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            g2 = np.roll(gl, s, a)
            d2 = np.roll(dl, s, a)
            out_m |= (g2 != gl) & (dl < d2) & (gl >= 0)
        out_m[0, :] = out_m[-1, :] = out_m[:, 0] = out_m[:, -1] = False
        dark = low * np.array([0.35, 0.33, 0.42], F)
        darkq, _ = nearest_color(dark, np.vstack([C, C * 0.4]))
        bg = np.zeros((h, w), bool)
        for s, a in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            bg |= np.roll(gl, s, a) < 0
        darkq = np.where((bg & out_m)[..., None], np.array([0.08, 0.07, 0.1], F), darkq)
        low = np.where(out_m[..., None], darkq, low)
    img = np.repeat(np.repeat(low, p, 0), p, 1)
    out = np.zeros((H, W, 3), F)
    out[:h * p, :w * p] = img
    if h * p < H:
        out[h * p:] = out[h * p - 1:h * p]
    if w * p < W:
        out[:, w * p:] = out[:, w * p - 1:w * p]
    return out


MEDIA = dict(render=render, flat=flat, oil=oil, watercolour=watercolour, ink=ink, pixel=pixel)
LABELS = dict(render='render', flat='flat graphic', oil='oil', watercolour='watercolour', ink='ink',
              pixel='pixel art')


def font(size):
    for f in ('C:/Windows/Fonts/segoeui.ttf', 'C:/Windows/Fonts/arial.ttf'):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def sheet(images, labels, cols, path, tile=(600, 400)):
    rows = (len(images) + cols - 1) // cols
    bar = 34
    S = Image.new('RGB', (cols * tile[0] + (cols + 1) * 8, rows * (tile[1] + bar) + (rows + 1) * 8), (24, 24, 26))
    dr = ImageDraw.Draw(S)
    f = font(20)
    for i, (img, lab) in enumerate(zip(images, labels)):
        r, c = divmod(i, cols)
        x, y = 8 + c * (tile[0] + 8), 8 + r * (tile[1] + bar + 8)
        im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize(tile, Image.LANCZOS)
        S.paste(im, (x, y + bar))
        dr.text((x + 4, y + 5), lab, fill=(235, 235, 235), font=f)
    S.save(path)


def with_text(img, b, medium=None):
    """Text layers stay flat and sharp: laid over the finished medium (in ink, text on the light paper is inked;
    light text on a dark hatched area keeps its colour)."""
    if b.get("text_P") is None:
        return img
    A = b["text_A"][..., None]
    P = b["text_P"]
    if medium == "ink":
        paper = blur(luma(img), px(b, 14)) > 0.45
        dark = ((luma(np.clip(P / np.maximum(A, 1e-3), 0, 1)) < 0.5) | paper)[..., None]
        P = np.where(dark, A * np.array([0.11, 0.09, 0.07], F), P)
    return np.clip(img * (1 - A) + P, 0, 1).astype(F)
