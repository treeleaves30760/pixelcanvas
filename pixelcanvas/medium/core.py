"""Shared helpers of the media: colour conversion, 2D noise fields, small colour ops."""
import numpy as np
from PIL import Image

from .nd import blur  # noqa: F401  (re-exported for the media)

F = np.float32


def srgb_to_lin(c):
    c = np.asarray(c, F)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(F)


def lin_to_srgb(c):
    c = np.clip(np.asarray(c, F), 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055).astype(F)


def parse_color(c):
    """'#rrggbb', 'rrggbb', a colour name or (r, g, b) in 0-1 or 0-255 -> sRGB 0-1 tuple."""
    if isinstance(c, str):
        from ..color import rgb
        return tuple(float(v) for v in rgb(c))
    c = tuple(float(v) for v in c)
    if max(c) > 1.0:
        c = tuple(v / 255.0 for v in c)
    return c


def srgb3(c):
    return np.array(parse_color(c), F)


def noise2(h, w, scale, seed=0):
    """Smooth value noise in [0, 1] with feature size ``scale`` px."""
    rng = np.random.default_rng(seed)
    scale = max(float(scale), 1.0)
    gh, gw = int(h / scale) + 3, int(w / scale) + 3
    g = rng.random((gh, gw)).astype(F)
    big = np.asarray(Image.fromarray(g, mode="F").resize((max(int(gw * scale), w), max(int(gh * scale), h)),
                                                          Image.BICUBIC), F)
    return np.clip(big[:h, :w], 0, 1)


def fbm2(h, w, scale, octaves=4, seed=0):
    tot, amp, norm = np.zeros((h, w), F), 1.0, 0.0
    for o in range(octaves):
        s = max(scale / (2 ** o), 1.0)
        tot += amp * noise2(h, w, s, seed + 31 * o)
        norm += amp
        amp *= 0.5
    return tot / norm


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def luma(img):
    img = np.asarray(img)
    return (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(F)


def rgb_to_hsv(c):
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    mx, mn = np.max(c, -1), np.min(c, -1)
    d = mx - mn + 1e-9
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) / 6.0
    s = np.where(mx > 1e-6, (mx - mn) / (mx + 1e-9), 0)
    return np.stack([h % 1.0, s, mx], -1).astype(F)


def hsv_to_rgb(c):
    h, s, v = c[..., 0] % 1.0, np.clip(c[..., 1], 0, 1), c[..., 2]
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    sel = [np.stack(x, -1) for x in ((v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q))]
    out = np.zeros(c.shape, F)
    for k in range(6):
        out = np.where((i == k)[..., None], sel[k], out)
    return out.astype(F)


def save(img, path):
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)


def load(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), F) / 255.0


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]], np.float32)
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]], np.float32)


def srgb_to_oklab(c):
    """sRGB (0-1, ... x 3) -> Oklab (L, a, b)."""
    lms = srgb_to_lin(np.clip(np.asarray(c, np.float32), 0, 1)) @ _M1.T
    return (np.cbrt(np.maximum(lms, 0)) @ _M2.T).astype(np.float32)


def oklab_to_srgb(lab):
    """Oklab -> sRGB (0-1, clipped)."""
    lms = np.asarray(lab, np.float32) @ np.linalg.inv(_M2).T.astype(np.float32)
    lin = (lms ** 3) @ np.linalg.inv(_M1).T.astype(np.float32)
    return np.clip(lin_to_srgb(np.clip(lin, 0, None)), 0, 1).astype(np.float32)
