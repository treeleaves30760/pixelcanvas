# -*- coding: utf-8 -*-
"""Canvas: a float32 (H, W, 3) image. Every method returns self (chainable); editing ``cv.buf`` directly also works."""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from .color import F32, rgb, as_color, mix, clamp01, grid
from . import filters as _f
from . import vision as _vision
from . import layers as _layers

_RES = getattr(Image, "Resampling", Image)

__all__ = ["Canvas"]

_CAPTURE = []        # active manifest captures (pixelcanvas.manifest); each gets the coverage of every paint


def _bbox(a):
    """(y0, y1, x0, x1) of the non-zero part of a 2D array, or None when it is all zero."""
    rows = np.flatnonzero(a.any(1))
    if not len(rows):
        return None
    cols = np.flatnonzero(a.any(0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _crop_color(color, shape, sl):
    """The part of ``color`` (single colour, colour map or Canvas) that lands on ``buf[sl]``; None if unknown."""
    c = as_color(color)
    if c.ndim == 1:
        return c
    if c.shape == shape:
        return c[sl]
    try:
        return np.broadcast_to(c, shape)[sl]
    except ValueError:
        return None


def _mo(mode, opacity, owner):
    """Resolve the (mode, opacity) pair: a number in the mode slot is the opacity (``cv.paint(c, mask, 0.9)``)."""
    if isinstance(mode, (int, float, np.number)) and not isinstance(mode, (bool, np.bool_)):
        if float(opacity) != 1.0:
            raise TypeError("%s: got opacity twice (%r as the third argument and opacity=%r); pass one number"
                            % (owner, mode, opacity))
        return "normal", float(mode)
    return _f.check_mode(mode, owner), opacity


class Canvas:
    """float32 (H, W, 3) canvas. x points right, y points down; ``cv.buf[y, x]`` is a pixel."""

    def __init__(self, width, height, color=(0.0, 0.0, 0.0)):
        self.buf = np.empty((int(height), int(width), 3), F32)
        self.buf[:] = rgb(color)
        _vision._track(self)
        _layers._track(self)

    # ---- create / export ----
    @classmethod
    def load(cls, path):
        """Load an image file as a canvas (RGB, 0~1)."""
        im = Image.open(path).convert("RGB")
        cv = cls(1, 1)
        cv.buf = np.asarray(im, F32) / 255.0
        return cv

    @classmethod
    def from_array(cls, arr):
        """Wrap an (H, W) or (H, W, 3) array: floats 0-1 as they are, uint8 0-255 (e.g. ``np.asarray(pil_image)``)
        scaled to 0-1."""
        if getattr(arr, "dtype", None) == np.uint8:
            arr = np.asarray(arr, F32) / 255.0
        arr = np.asarray(arr, F32)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, -1)
        cv = cls(1, 1)
        cv.buf = arr.copy()
        return cv

    @property
    def w(self):
        return self.buf.shape[1]

    @property
    def h(self):
        return self.buf.shape[0]

    @property
    def shape(self):
        return self.buf.shape

    @property
    def manifest(self):
        """What was drawn, where (list of dicts, paint order); see ``pixelcanvas.manifest``."""
        m = self.__dict__.get("_manifest")
        if m is None:
            m = self.__dict__["_manifest"] = []
        return m

    def copy(self):
        return Canvas.from_array(self.buf)

    def to_pil(self):
        return Image.fromarray((clamp01(self.buf) * 255 + 0.5).astype(np.uint8))

    def save(self, path, preview=None, preview_width=960, quality=92):
        """Save (the extension picks the format; missing folders are created). ``preview`` also writes a smaller copy for quick viewing."""
        im = self.to_pil()
        for target in (path, preview):
            folder = os.path.dirname(os.fspath(target)) if target else ""
            if folder:
                os.makedirs(folder, exist_ok=True)
        if str(path).lower().endswith((".jpg", ".jpeg")):
            im.save(path, quality=quality)
        else:
            im.save(path)
        if preview:
            pw = int(preview_width)
            ph = max(1, round(self.h * pw / self.w))
            im.resize((pw, ph), _RES.LANCZOS).save(preview, quality=88)
        return self

    def stats(self):
        """Print size and brightness statistics (a high ">1 fraction" means likely blown highlights); returns self.

        The same numbers are kept in ``self.last_stats`` (width, height, hdr_min, hdr_max, hdr_mean, over1_fraction).
        """
        b = self.buf
        over = float(np.mean(b.max(-1) > 1))
        self.last_stats = {"width": int(self.w), "height": int(self.h), "hdr_min": round(float(b.min()), 4),
                           "hdr_max": round(float(b.max()), 4), "hdr_mean": round(float(b.mean()), 4),
                           "over1_fraction": round(over, 5)}
        print(f"{self.w}x{self.h}  min={b.min():.3f} max={b.max():.3f} mean={b.mean():.3f}  fraction of pixels > 1: {over:.4f}")
        return self

    # ---- coordinates ----
    def xy(self):
        """Pixel coordinate grids (X, Y)."""
        return grid(self.h, self.w)

    def uv(self):
        """Normalised coordinates (U, V) in 0~1."""
        X, Y = self.xy()
        return X / self.w, Y / self.h

    # ---- pixel access ----
    def get(self, x, y):
        """Read pixels (x, y may be integers or integer arrays)."""
        return self.buf[y, x].copy()

    def put(self, x, y, color):
        """Write pixels. x, y may be integer arrays (many points at once); color is one colour or one per point (N, 3)."""
        self.buf[y, x] = as_color(color)
        return self

    def region(self, x0, y0, x1, y1):
        """A numpy view of a sub-region (edits apply to the canvas)."""
        return self.buf[y0:y1, x0:x1]

    # ---- painting ----
    def paint(self, color, mask=None, mode="normal", opacity=1.0):
        """Blend ``color`` (one colour or an (H, W, 3) colour map) through ``mask`` (alpha) with blend ``mode``.

        Only the bounding box of the non-zero part of the mask is touched, so small objects on a big canvas are cheap.
        ``mode`` is normal / add / multiply / screen / overlay / softlight / lighten / darken; a number in that
        position is the opacity, so ``cv.paint(color, mask, 0.9)`` paints at 90%. A tuple colour is floats 0~1
        (HDR above 1 allowed) or integers 0~255.
        """
        mode, opacity = _mo(mode, opacity, "Canvas.paint")
        if mask is None:
            a = float(opacity)
            if mode == "normal":
                _layers._note(self, a)
            for cap in _CAPTURE:
                cap.add(self, a, None)
            self.buf = _f.blend(self.buf, color, a, mode)
            return self
        a = np.asarray(mask, F32) * F32(opacity) if float(opacity) != 1.0 else np.asarray(mask, F32)
        buf = self.buf
        if a.ndim == 2 and a.shape == buf.shape[:2] and buf.dtype == F32 and buf.ndim == 3 and buf.flags.writeable:
            bb = _bbox(a)
            if bb is None:
                return self
            y0, y1, x0, x1 = bb
            sl = (slice(y0, y1), slice(x0, x1))
            col = _crop_color(color, buf.shape, sl)
            if col is not None:
                sub = a[sl]
                if mode == "normal":
                    _layers._note(self, sub, (x0, y0, x1, y1))
                for cap in _CAPTURE:
                    cap.add(self, sub, (x0, y0, x1, y1))
                buf[sl] = _f.blend(buf[sl], col, sub, mode)
                return self
        if mode == "normal":
            _layers._note(self, a)
        for cap in _CAPTURE:
            cap.add(self, a, None)
        self.buf = _f.blend(self.buf, color, a, mode)
        return self

    def fill(self, color, mask=None, mode="normal", opacity=1.0):
        """Same as ``paint``."""
        return self.paint(color, mask, mode, opacity)

    def gradient(self, stops, axis="y", mask=None, mode="normal", opacity=1.0):
        """Linear gradient: ``stops = [(position 0~1, colour), ...]``; axis = 'y' (top to bottom) or 'x' (left to right)."""
        from ._args import check_choice
        check_choice("Canvas.gradient", "axis", axis, ("y", "x"))
        U, V = self.uv()
        return self.paint(mix(stops, V if axis == "y" else U), mask, mode, opacity)

    def linear(self, p0, p1, stops, mask=None, mode="normal", opacity=1.0):
        """Linear gradient in any direction: position 0 at point ``p0`` (x, y) → 1 at ``p1``."""
        mode, opacity = _mo(mode, opacity, "Canvas.linear")
        X, Y = self.xy()
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        t = ((X - p0[0]) * dx + (Y - p0[1]) * dy) / max(dx * dx + dy * dy, 1e-9)
        return self.paint(mix(stops, t), mask, mode, opacity)

    def text(self, text, xy=None, size=None, color="#ffffff", **kw):
        """Draw styled text (see ``pixelcanvas.text.draw_text``). Returns self; the bbox is in ``cv.last_bbox``.
        Positions and sizes are pixels, or fractions of the canvas when every value is within 0–1: ``xy=(0.5, 0.1)``,
        ``size=0.08`` (of the height), ``box=(0.1, 0.8, 0.9, 0.95)``, ``max_width=0.6`` (of the width)."""
        from .text import draw_text
        xy, size, kw = self._fractions(xy, size, kw)
        self.last_bbox = draw_text(self, text, xy, size, color, **kw)
        return self

    def _fractions(self, xy, size, kw):
        """cv.text in canvas fractions: values all within 0–1 are scaled to pixels (a pixel value is never < 1)."""
        def frac(vals):
            v = [float(a) for a in vals]
            return all(0.0 <= a <= 1.0 for a in v) and any(a not in (0.0, 1.0) for a in v)
        if xy is not None and len(xy) == 2 and frac(xy):
            xy = (xy[0] * self.w, xy[1] * self.h)
        if size is not None and not isinstance(size, str) and 0 < float(size) < 1:
            size = float(size) * self.h
        b = kw.get("box")
        if b is not None and len(b) == 4 and frac(b):
            kw = dict(kw, box=(b[0] * self.w, b[1] * self.h, b[2] * self.w, b[3] * self.h))
        mw = kw.get("max_width")
        if mw is not None and not isinstance(mw, bool) and 0 < float(mw) < 1:
            kw = dict(kw, max_width=float(mw) * self.w)
        return xy, size, kw

    def radial(self, cx, cy, r, stops, mask=None, mode="normal", opacity=1.0):
        """Radial gradient: stop position 0 at the centre, 1 at radius r."""
        mode, opacity = _mo(mode, opacity, "Canvas.radial")
        X, Y = self.xy()
        t = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / float(r)
        return self.paint(mix(stops, t), mask, mode, opacity)

    def glow(self, cx, cy, sigma, color, strength=1.0, mode="add", mask=None):
        """Gaussian glow (added by default, so it can push values above 1 = HDR)."""
        mode = _f.check_mode(mode, "Canvas.glow")
        X, Y = self.xy()
        g = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma * sigma)).astype(F32) * strength
        if mask is not None:
            g = g * np.asarray(mask, F32)
        return self.paint(color, g, mode)

    def paste(self, other, x=0, y=0, mask=None, mode="normal", opacity=1.0):
        """Paste another canvas / array at (x, y) (clipped to the canvas); ``mask`` has the size of ``other``."""
        mode, opacity = _mo(mode, opacity, "Canvas.paste")
        src = other.buf if isinstance(other, Canvas) else np.asarray(other, F32)
        sh, sw = src.shape[:2]
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + sw, self.w), min(y + sh, self.h)
        if x0 >= x1 or y0 >= y1:
            return self
        sub = src[y0 - y:y1 - y, x0 - x:x1 - x]
        if mask is None:
            a = float(opacity)
        else:
            a = np.asarray(mask, F32)[y0 - y:y1 - y, x0 - x:x1 - x] * float(opacity)
            if a.ndim == 2 and a.shape == sub.shape[:2]:
                bb = _bbox(a)
                if bb is None:
                    return self
                by0, by1, bx0, bx1 = bb
                a, sub = a[by0:by1, bx0:bx1], sub[by0:by1, bx0:bx1]
                x0, y0, x1, y1 = x0 + bx0, y0 + by0, x0 + bx1, y0 + by1
        if mode == "normal":
            _layers._note(self, a, (x0, y0, x1, y1))
        for cap in _CAPTURE:
            cap.add(self, a, (x0, y0, x1, y1))
        self.buf[y0:y1, x0:x1] = _f.blend(self.buf[y0:y1, x0:x1], sub, a, mode)
        if isinstance(other, Canvas):                    # what a sticker canvas recorded belongs to this picture
            from . import manifest as _m
            _m.pasted(self, other, x, y, mask)
        return self

    # ---- sampling / geometry ----
    def sample(self, X, Y):
        """Sample the current picture at float coordinates (reflections, distortion)."""
        return _f.sample(self.buf, X, Y)

    def displace(self, dx, dy):
        """Warp the picture by a displacement field."""
        X, Y = self.xy()
        self.buf = _f.sample(self.buf, X + dx, Y + dy)
        return self

    def resize(self, w, h, method="lanczos"):
        """Resize the picture to w x h (method: see ``filters.resize``)."""
        self.buf = _f.resize(self.buf, w, h, method)
        return self

    def crop(self, x0, y0, x1, y1):
        self.buf = self.buf[y0:y1, x0:x1].copy()
        return self

    def flip(self, axis="y"):
        from ._args import check_choice
        check_choice("Canvas.flip", "axis", axis, ("y", "x"))
        self.buf = self.buf[::-1].copy() if axis == "y" else self.buf[:, ::-1].copy()
        return self

    # ---- drawing layers for Pillow ImageDraw / pycairo ----
    def draw(self, ss=2, name=None, kind="object", opacity=1.0, **finish):
        """A supersampled Pillow drawing layer: ``with cv.draw(ss=2, name="chef") as d:`` hands out an
        ImageDraw-compatible ``d`` on a layer ``ss`` times the canvas size. Draw as you would anywhere (``d.ellipse``,
        ``d.polygon``, ``d.line``, ``d.rounded_rectangle``, translucent ``fill=(r, g, b, a)`` blends); on exit it is
        downsampled (anti-aliased) and painted through ``cv.paint``, so layers and ``--check`` see it, and
        ``name=`` records its coverage. Coordinates: ``d.p(u, v)`` = layer pixels of canvas fractions,
        ``d.k(v)`` = layer pixels of a length in fractions of the height, ``d.font(size)`` a font at layer scale.
        Finish what was drawn in the same call (options of ``illustrate.Sketch.paint``): ``shadow=True``,
        ``shade=0.4`` (form shadow away from the light), ``highlight=0.3`` (rim light), ``line="#2a2030"`` or True,
        ``line_width=0.004``, ``glow="#ffcf70"``, ``light=(-0.6, -0.8)``. ``d.mask`` is the coverage afterwards."""
        return PillowLayer(self, ss, name, kind, opacity, finish)

    def cairo(self, ss=1, name=None, kind="object", opacity=1.0, **finish):
        """The same layer for pycairo (when installed): ``with cv.cairo() as d:`` gives ``d.ctx`` (a cairo Context
        in canvas pixels, already scaled by ``ss``) plus ``d.p`` / ``d.k``; composited through ``cv.paint`` on exit."""
        return CairoLayer(self, ss, name, kind, opacity, finish)

    # ---- post-processing (all wrappers around apply) ----
    def apply(self, fn, *args, **kw):
        """``cv.apply(fn, ...)`` → ``cv.buf = fn(cv.buf, ...)``."""
        self.buf = np.asarray(fn(self.buf, *args, **kw), F32)
        return self

    def blur(self, sigma):
        return self.apply(_f.gaussian_blur, sigma)

    def bloom(self, threshold=0.75, sigma=30, strength=0.4):
        return self.apply(_f.bloom, threshold, sigma, strength)

    def vignette(self, strength=0.35, radius=0.55, softness=0.6):
        return self.apply(_f.vignette, strength, radius, softness)

    def grain(self, amount=0.02, seed=0, mono=True):
        return self.apply(_f.grain, amount, seed, mono)

    def levels(self, black=0.0, white=1.0, gamma=1.0):
        return self.apply(_f.levels, black, white, gamma)

    def contrast(self, amount=0.2):
        return self.apply(_f.contrast, amount)

    def saturation(self, s=1.0):
        return self.apply(_f.saturation, s)

    def soft_clip(self, knee=0.8):
        return self.apply(_f.soft_clip, knee)

    def filmic(self):
        return self.apply(_f.filmic)

    def split_tone(self, shadows="#405070", highlights="#f0c080", amount=0.1):
        return self.apply(_f.split_tone, shadows, highlights, amount)

    def clip(self):
        self.buf = clamp01(self.buf)
        return self


# ---------------------------------------------------------------------------------------------- drawing layers

class _LayerBase:
    """Shared by PillowLayer / CairoLayer: coordinate helpers and the composite on exit."""

    _FINISH = ("shadow", "shade", "highlight", "volume", "spec", "line", "line_width", "glow", "light",
               "shade_color")

    def __init__(self, cv, ss, name, kind, opacity, finish=None):
        from ._args import reject_unknown
        self.finish = reject_unknown("Canvas.draw", dict(finish or {}), self._FINISH)
        self.cv, self.ss, self.name, self.kind, self.opacity = cv, max(int(ss), 1), name, kind, float(opacity)
        self.W, self.H = cv.w * self.ss, cv.h * self.ss

    def p(self, u, v):
        """Layer pixels of a point given as canvas fractions."""
        return (u * self.W, v * self.H)

    def pts(self, points):
        """Layer pixels of a list of fraction points."""
        return [self.p(u, v) for u, v in points]

    def k(self, v):
        """Layer pixels of a length given as a fraction of the canvas height."""
        return v * self.H

    def _finish(self, rgb, alpha):
        """rgb premultiplied over black, alpha 0..1, both at layer size → paint at canvas size."""
        if self.ss > 1:
            rgb = _f.resize(rgb, self.cv.w, self.cv.h, "box")
            alpha = _f.resize(alpha, self.cv.w, self.cv.h, "box")
        alpha = np.clip(alpha, 0, 1).astype(F32)
        col = np.clip(rgb / np.maximum(alpha, 1e-6)[..., None], 0, None).astype(F32)
        self.alpha = self.mask = alpha
        if self.finish:
            from .illustrate import Sketch
            fin = dict(self.finish)
            sk = Sketch(self.cv, light=fin.pop("light", (-0.6, -0.8)))
            sk.paint(alpha, col, opacity=self.opacity, **fin)
        else:
            self.cv.paint(col, alpha, opacity=self.opacity)
        if self.name:
            from . import manifest as _m
            ext = getattr(self, "_ext", None)                 # intended box in layer px, parts off the layer included
            if ext is not None:
                s = float(self.ss) if isinstance(self, PillowLayer) else 1.0
                ext = tuple(v / s for v in ext)
            _m.note(self.cv, self.kind, self.name, mask=alpha, extent=ext)


class PillowLayer(_LayerBase):
    """``cv.draw()``: Pillow ImageDraw calls on a supersampled layer with correct coverage for translucent inks."""

    def __enter__(self):
        from PIL import ImageDraw, ImageFont  # noqa: F401
        self.rgb = Image.new("RGB", (self.W, self.H), (0, 0, 0))
        self.cov = Image.new("L", (self.W, self.H), 0)
        self._d = ImageDraw.Draw(self.rgb, "RGBA")
        self._c = ImageDraw.Draw(self.cov)
        return self

    def font(self, size, name=None, bold=False, text=None):
        """A Pillow font ``size`` (fraction of the canvas height) tall at layer scale; ``name`` as ``find_font``."""
        from PIL import ImageFont
        from .text import find_font
        path = find_font(name, bold=bold, text=text)
        px = max(int(round(self.k(size))), 1)
        if path is None:
            return ImageFont.load_default()
        path, _, var = str(path).partition("#")
        f = ImageFont.truetype(path, px)
        if var:
            try:
                f.set_variation_by_name(var)
            except Exception:                            # noqa: BLE001
                pass
        return f

    @staticmethod
    def _ink_alpha(ink):
        if ink is None:
            return None
        if isinstance(ink, str):
            s = ink.lstrip("#")
            return int(s[6:8], 16) if ink.startswith("#") and len(s) == 8 else 255
        if isinstance(ink, (tuple, list)) and len(ink) == 4:
            return int(ink[3])
        return 255

    def __getattr__(self, attr):
        if attr.startswith("_") or "_d" not in self.__dict__:
            raise AttributeError(attr)
        fn = getattr(self._d, attr)
        if not callable(fn):
            return fn

        def call(*a, **k):
            if isinstance(k.get("width"), float):         # Pillow wants int widths; d.k() gives floats
                k["width"] = max(int(round(k["width"])), 1)
            fn(*a, **k)
            self._grow_ext(attr, a, k)
            inks = {}
            for key in ("fill", "outline"):
                if key in k:
                    inks[key] = self._ink_alpha(k[key])
            a2 = list(a)
            if not inks and len(a2) >= 2 and attr not in ("text", "multiline_text"):
                inks["fill"] = self._ink_alpha(a2[1])
                a2[1] = 255
            if attr in ("text", "multiline_text") and "fill" not in k:
                if len(a2) >= 3:
                    inks["fill"] = self._ink_alpha(a2[2])
                    a2[2] = 255
                else:
                    inks["fill"] = 255
            k2 = dict(k)
            for key, al in inks.items():
                k2[key] = None if al is None else 255
            if all(al in (None, 255) for al in inks.values()):
                getattr(self._c, attr)(*a2, **k2)
                return
            for key, al in inks.items():              # translucent ink: screen its own coverage in
                if al is None:
                    continue
                tmp = Image.new("L", self.cov.size, 0)
                kk = {kk_: (255 if kk_ == key else None) if kk_ in inks else v for kk_, v in k2.items()}
                getattr(ImageDraw.Draw(tmp), attr)(*a2, **kk)
                tmp = tmp.point(lambda v, al=al: v * al // 255)
                self.cov = ImageChops.screen(self.cov, tmp)
            self._c = ImageDraw.Draw(self.cov)
        return call

    _SHAPES = ("ellipse", "rectangle", "rounded_rectangle", "polygon", "line", "pieslice", "chord", "arc", "point",
               "regular_polygon", "circle")

    def _grow_ext(self, attr, a, k):
        """Widen the intended box by the geometry of one call, including parts outside the layer."""
        xy = k.get("xy", a[0] if a else None)
        try:
            if attr in ("text", "multiline_text"):
                kw = {key: k[key] for key in ("font", "anchor", "spacing", "align", "stroke_width") if key in k}
                text = k.get("text", a[1] if len(a) > 1 else "")
                fnb = self._d.multiline_textbbox if attr == "multiline_text" else self._d.textbbox
                box = fnb(tuple(xy), text, **kw)
            elif attr in ("regular_polygon", "circle"):
                if attr == "circle":
                    (cx, cy), r = xy, k.get("radius", a[1] if len(a) > 1 else 0)
                else:
                    cx, cy, r = (xy[0][0], xy[0][1], xy[1]) if len(xy) == 2 else xy
                box = (cx - r, cy - r, cx + r, cy + r)
            elif attr in self._SHAPES:
                v = np.asarray(xy, np.float64).reshape(-1, 2)
                w = float(k.get("width", 0) or 0) / 2.0 if attr in ("line", "arc", "point") else 0.0
                box = (v[:, 0].min() - w, v[:, 1].min() - w, v[:, 0].max() + w, v[:, 1].max() + w)
            else:
                return
        except Exception:                                # noqa: BLE001 - an unusual call: the mask's box is used
            return
        e = getattr(self, "_ext", None)
        self._ext = tuple(float(x) for x in box) if e is None else (
            min(e[0], box[0]), min(e[1], box[1]), max(e[2], box[2]), max(e[3], box[3]))

    def paste(self, im, xy=(0, 0), mask=None):
        """Paste a PIL image (RGBA uses its alpha) onto the layer and its coverage."""
        im = im.convert("RGBA")
        m = im.getchannel("A") if mask is None else mask
        self.rgb.paste(im.convert("RGB"), tuple(int(v) for v in xy), m)
        self._grow_ext("rectangle", ([xy[0], xy[1], xy[0] + im.width, xy[1] + im.height],), {})
        self.cov.paste(255, (int(xy[0]), int(xy[1]), int(xy[0]) + im.width, int(xy[1]) + im.height), m)
        self._d = ImageDraw.Draw(self.rgb, "RGBA")
        self._c = ImageDraw.Draw(self.cov)

    def __exit__(self, *exc):
        if exc[0] is not None:
            return False
        a = np.asarray(self.cov, F32) / 255.0
        c = np.asarray(self.rgb, F32) / 255.0
        self._finish(c * 1.0, a)              # rgb drawn over black == premultiplied
        return False


class CairoLayer(_LayerBase):
    """``cv.cairo()``: a pycairo context on an ARGB32 surface, composited like ``cv.draw``."""

    def __enter__(self):
        try:
            import cairo
        except ImportError as e:                         # pragma: no cover - depends on the machine
            raise ImportError("cv.cairo() needs pycairo (pip install pycairo); cv.draw() gives the same layer "
                              "for Pillow ImageDraw") from e
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.W, self.H)
        self.ctx = cairo.Context(self.surface)
        self.ctx.scale(self.ss, self.ss)
        self.W, self.H = self.cv.w, self.cv.h           # helpers answer in context units = canvas pixels
        return self

    def __exit__(self, *exc):
        if exc[0] is not None:
            return False
        self.surface.flush()
        h, w = self.cv.h * self.ss, self.cv.w * self.ss
        buf = np.frombuffer(self.surface.get_data(), np.uint8).reshape(h, self.surface.get_stride() // 4, 4)[:, :w]
        buf = buf.astype(F32) / 255.0
        self._finish(buf[..., 2::-1].copy(), buf[..., 3].copy())   # BGRA premultiplied
        return False
