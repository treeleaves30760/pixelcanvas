"""Small image operations for the medium package: numpy, with scipy.ndimage as an identical speed-up when present."""
import math

import numpy as np

from ..filters import gaussian_blur as _gauss, sample as _sample

try:
    from scipy import ndimage as _ndi
except ImportError:                                      # pragma: no cover - numpy path
    _ndi = None

F = np.float32


def blur(a, s):
    """Gaussian blur of (H, W) or (H, W, C); s <= 0 returns the array."""
    if s <= 0:
        return np.asarray(a, F)
    return _gauss(a, float(s))


def sample(a, ys, xs):
    """Bilinear read of a at float pixel coordinates (edges clamp)."""
    return _sample(a, xs, ys, 1)


def _shift(m, dy, dx, fill=False):
    out = np.full_like(m, fill)
    h, w = m.shape
    ys0, ys1 = max(dy, 0), h + min(dy, 0)
    xs0, xs1 = max(dx, 0), w + min(dx, 0)
    if ys1 > ys0 and xs1 > xs0:
        out[ys0:ys1, xs0:xs1] = m[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def dilate(m, iterations=1):
    m = np.asarray(m, bool)
    if _ndi is not None:
        return _ndi.binary_dilation(m, iterations=iterations)
    for _ in range(iterations):
        m = m | _shift(m, 1, 0) | _shift(m, -1, 0) | _shift(m, 0, 1) | _shift(m, 0, -1)
    return m


def erode(m, iterations=1):
    m = np.asarray(m, bool)
    if _ndi is not None:
        return _ndi.binary_erosion(m, iterations=iterations)
    for _ in range(iterations):
        m = m & _shift(m, 1, 0, True) & _shift(m, -1, 0, True) & _shift(m, 0, 1, True) & _shift(m, 0, -1, True)
        m[0, :] = m[-1, :] = False
        m[:, 0] = m[:, -1] = False
    return m


def opening(m, iterations=1):
    return dilate(erode(m, iterations), iterations)


def closing(m, iterations=1):
    m = np.pad(np.asarray(m, bool), iterations)
    out = erode(dilate(m, iterations), iterations)
    return out[iterations:-iterations, iterations:-iterations] if iterations else out


def label(m):
    """Connected parts (4-neighbours) -> (labels, count)."""
    m = np.asarray(m, bool)
    if _ndi is not None:
        lab, n = _ndi.label(m)
        return lab, int(n)
    h, w = m.shape
    lab = np.where(m, np.arange(1, h * w + 1).reshape(h, w), 0)
    big = h * w + 1
    while True:
        cur = np.where(m, lab, big)
        nb = cur.copy()
        nb[1:] = np.minimum(nb[1:], cur[:-1])
        nb[:-1] = np.minimum(nb[:-1], cur[1:])
        nb[:, 1:] = np.minimum(nb[:, 1:], cur[:, :-1])
        nb[:, :-1] = np.minimum(nb[:, :-1], cur[:, 1:])
        new = np.where(m, nb, 0)
        if np.array_equal(new, lab):
            break
        lab = new
    u = np.unique(lab[lab > 0])
    remap = np.zeros(big + 1, np.int64)
    remap[u] = np.arange(1, len(u) + 1)
    return remap[lab], int(len(u))


def fill_holes(m):
    m = np.asarray(m, bool)
    if _ndi is not None:
        return _ndi.binary_fill_holes(m)
    lab, n = label(np.pad(~m, 1))
    outside = lab == lab[0, 0]
    return ~outside[1:-1, 1:-1]


def edt(feature_free):
    """Euclidean distance from every True pixel to the nearest False pixel (scipy's distance_transform_edt)."""
    f = np.asarray(feature_free, bool)
    if _ndi is not None:
        return _ndi.distance_transform_edt(f).astype(F)
    from ..illustrate import _edt_sq
    lim = max(f.shape)
    return np.sqrt(_edt_sq(~f, lim)).astype(F)


def near_lines(mask, width, wobble_mask=None):
    """Ink coverage of lines through mask pixels with a per-pixel width (H, W) or a scalar, in px: each pixel takes
    the width of its nearest mask pixel (found within the widest line's reach)."""
    m = np.asarray(mask, bool)
    if not m.any():
        return np.zeros(m.shape, F)
    if np.ndim(width) == 2:
        reach = int(math.ceil(float(np.max(width[m])) / 2 + 2))
    else:
        reach = int(math.ceil(float(width) / 2 + 2))
    if _ndi is not None:
        dist, (iy, ix) = _ndi.distance_transform_edt(~m, return_indices=True)
        w = width[iy, ix] if np.ndim(width) == 2 else width
    else:
        h, wd = m.shape
        yy, xx = np.mgrid[0:h, 0:wd]
        sy = np.where(m, yy, -10 ** 6)
        sx = np.where(m, xx, -10 ** 6)
        best = np.where(m, 0.0, np.inf)
        for _ in range(reach):
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                ny, nx = _shift(sy, dy, dx, -10 ** 6), _shift(sx, dy, dx, -10 ** 6)
                d = (ny - yy) ** 2 + (nx - xx) ** 2
                better = (ny > -10 ** 5) & (d < best)
                best = np.where(better, d, best)
                sy, sx = np.where(better, ny, sy), np.where(better, nx, sx)
        dist = np.sqrt(best)
        ok = sy > -10 ** 5
        w = (width[np.clip(sy, 0, h - 1), np.clip(sx, 0, wd - 1)] * ok) if np.ndim(width) == 2 else width
    t = np.clip((np.asarray(w / 2 + 0.7, F) - dist) / 1.2, 0, 1)
    return (t * t * (3 - 2 * t) * (np.asarray(w) > 0.05)).astype(F)


def membrane(mask, max_side=160):
    """Inflate a mask: solve laplace(F) = -1 inside, 0 outside (conjugate gradients on a reduced grid) and return
    2 * sqrt(F) in pixels of the mask: a hemisphere of radius r over a disc of radius r."""
    from PIL import Image
    mask = np.asarray(mask, bool)
    ys, xs = np.nonzero(mask)
    if len(ys) < 12:
        return np.zeros(mask.shape, F)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = mask[y0:y1, x0:x1]
    f = max(1.0, max(sub.shape) / max_side)
    h, w = max(int(round(sub.shape[0] / f)), 3), max(int(round(sub.shape[1] / f)), 3)
    m = np.asarray(Image.fromarray(sub.astype(F), mode="F").resize((w, h), Image.BOX), F) > 0.5
    m = np.pad(m, 1)
    if m.sum() < 4:
        return np.zeros(mask.shape, F)
    mf = m.astype(np.float64)

    def lap(v):                                          # 4 v - neighbours, Dirichlet 0 outside the mask
        v = v * mf
        out = 4 * v
        out[1:] -= v[:-1]
        out[:-1] -= v[1:]
        out[:, 1:] -= v[:, :-1]
        out[:, :-1] -= v[:, 1:]
        return out * mf
    x = np.zeros_like(mf)
    r = mf - lap(x)
    p = r.copy()
    rs = float((r * r).sum())
    for _ in range(int(4 * max(m.shape)) + 20):
        Ap = lap(p)
        a = rs / max(float((p * Ap).sum()), 1e-20)
        x += a * p
        r -= a * Ap
        rn = float((r * r).sum())
        if rn < 1e-10 * mf.sum():
            break
        p = r + (rn / rs) * p
        rs = rn
    Hs = (2 * np.sqrt(np.clip(x[1:-1, 1:-1], 0, None)) * f).astype(F)
    Hf = np.asarray(Image.fromarray(Hs, mode="F").resize((sub.shape[1], sub.shape[0]), Image.BICUBIC), F)
    out = np.zeros(mask.shape, F)
    out[y0:y1, x0:x1] = np.clip(Hf, 0, None) * sub
    return out


def resize(a, W, H, mode="box"):
    from PIL import Image
    a = np.asarray(a, F)
    m = {"box": Image.BOX, "bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC, "nearest": Image.NEAREST}[mode]
    if a.ndim == 3:
        return np.stack([resize(a[..., c], W, H, mode) for c in range(a.shape[2])], -1)
    return np.asarray(Image.fromarray(a, mode="F").resize((int(W), int(H)), m), F)
