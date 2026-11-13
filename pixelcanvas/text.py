# -*- coding: utf-8 -*-
"""Typography: fonts with per-glyph fallback, kerning, rich text, wrapping, fitting, effects and a text audit.

Everything follows the usual "mask first, then colour" rule:

* ``draw_text(cv, text, xy, ...)``   → paints styled text on a Canvas and returns its bounding box
  (``cv.text(...)`` forwards every keyword here).
* ``text_mask(h, w, text, xy, ...)`` → float32 (H, W) anti-aliased coverage mask.
* ``text_bbox`` / ``fit_text_size``  → measure before drawing.
* ``text_on_arc`` / ``text_on_path`` → badges, seals and curved labels.
* ``find_font`` / ``list_fonts`` / ``missing_glyphs`` → font lookup by name or style word
  ("serif", "condensed bold", "handwriting" ...); any glyph a font lacks is drawn from a fallback font.
* ``audit(cv)``                      → typography issues (size, contrast, clipping, overlaps, missing glyphs).

Anchors are two letters: horizontal ``l`` / ``m`` / ``r`` + vertical ``t`` / ``m`` / ``b`` (also ``a`` = top,
``s`` = baseline of the first line, ``d`` = bottom). Sizes and positions are canvas pixels
(multiply 1080p constants by ``S = H / 1080``).
"""
from __future__ import annotations

import glob
import math
import os
import re
import struct
import sys
from functools import lru_cache

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

from .color import F32, as_color, clamp01, mix, rgb, smoothstep
from . import manifest as _manifest
from ._args import (reject_unknown as _reject_unknown, unknown_name as _unknown_name,
                    check_choice as _check_choice, signature_options as _sig_options, strict as _strict)

_RES = getattr(Image, "Resampling", Image)
_TRANSFORM = getattr(Image, "Transform", Image)

__all__ = ["find_font", "list_fonts", "load_font", "has_cjk", "wrap_text", "text_bbox", "text_mask",
           "fit_text_size", "draw_text", "missing_glyphs", "text_on_arc", "text_on_path", "parse_markup",
           "audit_text", "contrast_ratio"]

USER_FONT_DIR = os.path.join(os.path.expanduser("~"), ".pixelcanvas", "fonts")

# --------------------------------------------------------------------------------------------- scripts

_CJK_RE = re.compile(r"[\u2e80-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef\u3000-\u303f]")
_KANA_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]")
_IGNORABLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufe0e\ufe0f\u2060"), None)


def has_cjk(text):
    """True if the text contains CJK / full-width characters (needs a CJK-capable font)."""
    return bool(_CJK_RE.search(str(text)))


def _is_cjk_char(ch):
    o = ord(ch)
    return (0x2e80 <= o <= 0x9fff or 0xac00 <= o <= 0xd7af or 0xf900 <= o <= 0xfaff or 0xff00 <= o <= 0xffef
            or 0x3000 <= o <= 0x303f or 0x20000 <= o <= 0x3ffff)


# --------------------------------------------------------------------------------------------- font files

def _font_dirs():
    dirs = []
    if sys.platform.startswith("win"):
        windir = os.environ.get("WINDIR", "C:/Windows")
        dirs += [os.path.join(windir, "Fonts"),
                 os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Windows", "Fonts")]
    elif sys.platform == "darwin":
        dirs += ["/System/Library/Fonts", "/System/Library/Fonts/Supplemental", "/Library/Fonts",
                 os.path.expanduser("~/Library/Fonts")]
    else:
        dirs += ["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"),
                 os.path.expanduser("~/.local/share/fonts")]
    dirs = [USER_FONT_DIR] + dirs
    extra = os.environ.get("PIXELCANVAS_FONT_DIR")
    if extra:
        dirs = extra.split(os.pathsep) + dirs
    return [d for d in dirs if d and os.path.isdir(d)]


@lru_cache(maxsize=1)
def _font_index():
    """{lower-case file stem: path} for every .ttf/.otf/.ttc under the font directories (user dirs first)."""
    idx = {}
    for d in _font_dirs():
        for ext in ("ttf", "otf", "ttc", "TTF", "OTF", "TTC"):
            for p in glob.glob(os.path.join(d, "**", f"*.{ext}"), recursive=True):
                idx.setdefault(os.path.splitext(os.path.basename(p))[0].lower(), p)
    return idx


def _norm(s):
    return re.sub(r"[\s_\-\[\],.]+", "", str(s).lower())


@lru_cache(maxsize=1)
def _norm_index():
    out = {}
    for k, p in _font_index().items():
        out.setdefault(_norm(k), p)
    return out


def refresh_fonts():
    """Forget the cached font list (call after installing fonts in a running process)."""
    _font_index.cache_clear()
    _norm_index.cache_clear()
    _sfnt.cache_clear()
    _FALLBACK_SCAN.clear()
    _chain.cache_clear()
    _family_file.cache_clear()


def _split_spec(spec):
    """'path#Bold Condensed' / 'path#1' → (path, ttc index, variation hint)."""
    s = str(spec)
    hint, index = None, 0
    if "#" in s:
        s, tail = s.rsplit("#", 1)
        if tail.isdigit():
            index = int(tail)
        else:
            hint = tail
    return s, index, hint


# --------------------------------------------------------------------------------------------- sfnt tables

def _popcount(v):
    return bin(v).count("1")


class _SFNT:
    """Minimal OpenType reader: cmap (coverage, char → glyph), units/em, cap height, GPOS / kern pair kerning."""

    def __init__(self, path, index=0):
        self.path, self.index = path, index
        self.cmap, self.upem, self.cap, self.xh = {}, 1000, 0.7, 0.5
        self.tables = {}
        self._kern = None
        try:
            with open(path, "rb") as fh:
                head = fh.read(12)
                base = 0
                if head[:4] == b"ttcf":
                    n = struct.unpack(">I", head[8:12])[0]
                    fh.seek(12 + 4 * min(index, n - 1))
                    base = struct.unpack(">I", fh.read(4))[0]
                fh.seek(base)
                hdr = fh.read(12)
                num = struct.unpack(">H", hdr[4:6])[0]
                dirb = fh.read(16 * num)
                for i in range(num):
                    tag = dirb[16 * i:16 * i + 4].decode("latin-1")
                    self.tables[tag] = struct.unpack(">II", dirb[16 * i + 8:16 * i + 16])
                h = self._read(fh, "head")
                if len(h) >= 20:
                    self.upem = struct.unpack(">H", h[18:20])[0] or 1000
                o2 = self._read(fh, "OS/2")
                if len(o2) >= 90 and struct.unpack(">H", o2[:2])[0] >= 2:
                    xh, cap = struct.unpack(">hh", o2[86:90])
                    if cap > 0:
                        self.cap = cap / self.upem
                    if xh > 0:
                        self.xh = xh / self.upem
                self.cmap = self._parse_cmap(self._read(fh, "cmap"))
        except Exception:
            pass

    def _read(self, fh, tag):
        if tag not in self.tables:
            return b""
        off, ln = self.tables[tag]
        fh.seek(off)
        return fh.read(ln)

    @staticmethod
    def _parse_cmap(d):
        if len(d) < 4:
            return {}
        n = struct.unpack(">H", d[2:4])[0]
        recs = []
        for i in range(n):
            pid, eid, off = struct.unpack(">HHI", d[4 + 8 * i:12 + 8 * i])
            fmt = struct.unpack(">H", d[off:off + 2])[0]
            rank = {(3, 10): 0, (0, 6): 1, (0, 4): 1, (3, 1): 2, (0, 3): 3, (0, 2): 4, (0, 1): 4, (0, 0): 5}.get((pid, eid), 9)
            if fmt in (4, 12):
                recs.append((0 if fmt == 12 else 1, rank, off, fmt))
        if not recs:
            return {}
        out = {}
        for _, _, off, fmt in sorted(recs)[:2]:
            try:
                part = _SFNT._fmt12(d, off) if fmt == 12 else _SFNT._fmt4(d, off)
            except Exception:
                part = {}
            for k, v in part.items():
                out.setdefault(k, v)
        return out

    @staticmethod
    def _fmt4(d, o):
        segx2 = struct.unpack(">H", d[o + 6:o + 8])[0]
        n = segx2 // 2
        ends = np.frombuffer(d, ">u2", n, o + 14).astype(np.int64)
        starts = np.frombuffer(d, ">u2", n, o + 16 + segx2).astype(np.int64)
        deltas = np.frombuffer(d, ">u2", n, o + 16 + 2 * segx2).astype(np.int64)
        ro_off = o + 16 + 3 * segx2
        ros = np.frombuffer(d, ">u2", n, ro_off).astype(np.int64)
        u8 = np.frombuffer(d, np.uint8)
        cps, gids = [], []
        for i in range(n):
            s, e = starts[i], ends[i]
            if s > e or s == 0xFFFF:
                continue
            c = np.arange(s, e + 1)
            if ros[i] == 0:
                g = (c + deltas[i]) & 0xFFFF
            else:
                addr = ro_off + 2 * i + ros[i] + 2 * (c - s)
                ok = addr + 1 < len(u8)
                addr = np.where(ok, addr, 0)
                g = u8[addr].astype(np.int64) * 256 + u8[addr + 1]
                g = np.where(ok & (g != 0), (g + deltas[i]) & 0xFFFF, 0)
            cps.append(c)
            gids.append(g)
        if not cps:
            return {}
        c = np.concatenate(cps)
        g = np.concatenate(gids)
        keep = g != 0
        return dict(zip(c[keep].tolist(), g[keep].tolist()))

    @staticmethod
    def _fmt12(d, o):
        ng = struct.unpack(">I", d[o + 12:o + 16])[0]
        arr = np.frombuffer(d, ">u4", ng * 3, o + 16).astype(np.int64).reshape(ng, 3)
        lens = arr[:, 1] - arr[:, 0] + 1
        lens = np.clip(lens, 0, 0x110000)
        total = int(lens.sum())
        if total <= 0 or total > 400000:
            arr = arr[lens < 70000]
            lens = arr[:, 1] - arr[:, 0] + 1
            total = int(lens.sum())
        grp = np.repeat(np.arange(len(arr)), lens)
        offs = np.arange(total) - np.repeat(np.cumsum(lens) - lens, lens)
        c = arr[grp, 0] + offs
        g = arr[grp, 2] + offs
        keep = g != 0
        return dict(zip(c[keep].tolist(), g[keep].tolist()))

    # ---- kerning
    def kern_units(self, a, b):
        ga, gb = self.cmap.get(ord(a)), self.cmap.get(ord(b))
        if not ga or not gb:
            return 0
        if self._kern is None:
            try:
                self._kern = self._load_kern()
            except Exception:
                self._kern = lambda g1, g2: 0
        return self._kern(ga, gb)

    def _load_kern(self):
        with open(self.path, "rb") as fh:
            gpos = self._read(fh, "GPOS")
            kern = self._read(fh, "kern") if not gpos else b""
        if gpos:
            f = self._gpos_kern(gpos)
            if f is not None:
                return f
            with open(self.path, "rb") as fh:
                kern = self._read(fh, "kern")
        if kern:
            return self._old_kern(kern)
        return lambda g1, g2: 0

    @staticmethod
    def _old_kern(d):
        ver, n = struct.unpack(">HH", d[:4])
        if ver != 0:
            return lambda g1, g2: 0
        pairs, o = {}, 4
        for _ in range(n):
            _, ln, cov = struct.unpack(">HHH", d[o:o + 6])
            if (cov >> 8) == 0 and (cov & 1):
                np_ = struct.unpack(">H", d[o + 6:o + 8])[0]
                arr = np.frombuffer(d, ">u2", np_ * 3, o + 14).reshape(np_, 3)
                vals = arr[:, 2].astype(np.int16)
                for l_, r_, v in zip(arr[:, 0].tolist(), arr[:, 1].tolist(), vals.tolist()):
                    pairs.setdefault((l_ << 16) | r_, v)
            o += ln
        return lambda g1, g2: pairs.get((g1 << 16) | g2, 0)

    @staticmethod
    def _coverage(d, o):
        fmt = struct.unpack(">H", d[o:o + 2])[0]
        n = struct.unpack(">H", d[o + 2:o + 4])[0]
        if fmt == 1:
            gl = np.frombuffer(d, ">u2", n, o + 4).tolist()
            return {g: i for i, g in enumerate(gl)}
        out = {}
        arr = np.frombuffer(d, ">u2", n * 3, o + 4).reshape(n, 3).tolist()
        for s, e, si in arr:
            for g in range(s, e + 1):
                out[g] = si + g - s
        return out

    @staticmethod
    def _classdef(d, o):
        fmt = struct.unpack(">H", d[o:o + 2])[0]
        out = {}
        if fmt == 1:
            start, n = struct.unpack(">HH", d[o + 2:o + 6])
            vals = np.frombuffer(d, ">u2", n, o + 6).tolist()
            for i, v in enumerate(vals):
                if v:
                    out[start + i] = v
        elif fmt == 2:
            n = struct.unpack(">H", d[o + 2:o + 4])[0]
            for s, e, c in np.frombuffer(d, ">u2", n * 3, o + 4).reshape(n, 3).tolist():
                if c:
                    for g in range(s, e + 1):
                        out[g] = c
        return out

    def _gpos_kern(self, d):
        u16 = lambda o: struct.unpack(">H", d[o:o + 2])[0]  # noqa: E731
        feat_list, lookup_list = u16(4 + 2), u16(4 + 4)
        nf = u16(feat_list)
        idx = set()
        for i in range(nf):
            rec = feat_list + 2 + 6 * i
            if d[rec:rec + 4] == b"kern":
                fo = feat_list + u16(rec + 4)
                cnt = u16(fo + 2)
                idx.update(u16(fo + 4 + 2 * j) for j in range(cnt))
        if not idx:
            return None
        lookups = []
        for li in sorted(idx):
            lo = lookup_list + u16(lookup_list + 2 + 2 * li)
            ltype, n = u16(lo), u16(lo + 4)
            subs = []
            for s in range(n):
                so = lo + u16(lo + 6 + 2 * s)
                t = ltype
                if t == 9:
                    t = u16(so + 2)
                    so = so + struct.unpack(">I", d[so + 4:so + 8])[0]
                if t == 2:
                    sub = self._pairpos(d, so)
                    if sub is not None:
                        subs.append(sub)
            if subs:
                lookups.append(subs)
        if not lookups:
            return None
        cache = {}

        def kern(g1, g2):
            key = (g1 << 16) | g2
            if key in cache:
                return cache[key]
            total = 0
            for subs in lookups:
                for sub in subs:
                    r = sub(g1, g2)
                    if r is not None:
                        total += r
                        break
            cache[key] = total
            return total
        return kern

    def _pairpos(self, d, so):
        u16 = lambda o: struct.unpack(">H", d[o:o + 2])[0]  # noqa: E731
        fmt = u16(so)
        cov = self._coverage(d, so + u16(so + 2))
        vf1, vf2 = u16(so + 4), u16(so + 6)
        s1, s2 = 2 * _popcount(vf1), 2 * _popcount(vf2)
        xoff = 2 * _popcount(vf1 & 3) if vf1 & 4 else None
        if fmt == 1:
            sets = {}

            def f1(g1, g2):
                ci = cov.get(g1)
                if ci is None:
                    return None
                if ci not in sets:
                    ps = so + u16(so + 10 + 2 * ci)
                    n = u16(ps)
                    rs = 2 + s1 + s2
                    m = {}
                    for k in range(n):
                        r = ps + 2 + k * rs
                        m[u16(r)] = struct.unpack(">h", d[r + 2 + xoff:r + 4 + xoff])[0] if xoff is not None else 0
                    sets[ci] = m
                v = sets[ci].get(g2)
                return v
            return f1
        if fmt == 2:
            cd1 = self._classdef(d, so + u16(so + 8))
            cd2 = self._classdef(d, so + u16(so + 10))
            c2n = u16(so + 14)
            rs = s1 + s2

            def f2(g1, g2):
                if g1 not in cov:
                    return None
                if xoff is None:
                    return 0
                r = so + 16 + (cd1.get(g1, 0) * c2n + cd2.get(g2, 0)) * rs
                return struct.unpack(">h", d[r + xoff:r + xoff + 2])[0]
            return f2
        return None


@lru_cache(maxsize=256)
def _sfnt(path, index=0):
    return _SFNT(path, index)


def _info(spec):
    if spec is None:
        return None
    p, i, _ = _split_spec(spec)
    return _sfnt(p, i)


# --------------------------------------------------------------------------------------------- style words

_WEIGHT_WORDS = {"bold": "bold", "black": "black", "heavy": "black", "extrabold": "black", "ultra": "black",
                 "light": "light", "thin": "light", "extralight": "light", "medium": "medium",
                 "semibold": "bold", "regular": "regular", "normal": "regular", "book": "regular"}
_ITALIC_WORDS = {"italic", "oblique", "slanted"}
_STYLE_ALIASES = {"sans-serif": "sans", "sansserif": "sans", "grotesk": "sans",
                  "slab-serif": "slab", "slabserif": "slab", "monospace": "mono", "code": "mono",
                  "typewriter": "mono", "poster": "display", "headline": "display",
                  "narrow": "condensed", "compressed": "condensed", "round": "rounded", "soft": "rounded",
                  "hand": "handwriting", "handwritten": "handwriting", "marker": "handwriting", "casual": "handwriting",
                  "brush": "script", "cursive": "script", "calligraphy": "script", "elegant": "script",
                  "classic": "serif", "geometric": "sans"}

# family → {variant: [normalised file stems]}; variant = weight [+ "i" for italic]. "vf" families take weights from
# their variation axes. A stem may carry a '#hint' (variation name, e.g. Bahnschrift's condensed instances).
_FAMILIES = {
    "inter": {"regular": ["intervf"], "i": ["interitalicvf"]},
    "segoe": {"regular": ["segoeui"], "bold": ["segoeuib"], "black": ["seguibl"], "light": ["segoeuil", "segoeuisl"],
              "medium": ["seguisb"], "i": ["segoeuii"], "boldi": ["segoeuiz"], "blacki": ["seguibli"], "lighti": ["seguili"]},
    "helvetica": {"regular": ["helveticaneue", "helvetica"]},
    "sfpro": {"regular": ["sfns", "sfpro", "sfprodisplayregular"]},
    "arial": {"regular": ["arial"], "bold": ["arialbd", "arialbold"], "black": ["ariblk", "arialblack"],
              "i": ["ariali", "arialitalic"], "boldi": ["arialbi", "arialbolditalic"]},
    "montserrat": {"regular": ["montserratvf", "montserratregular"], "medium": ["montserratmedium"], "i": ["montserratitalicvf"]},
    "spacegrotesk": {"regular": ["spacegroteskvf"]},
    "notosans": {"regular": ["notosansvf", "notosansregular"], "bold": ["notosansbold"], "i": ["notosansitalicvf", "notosansitalic"]},
    "dejavusans": {"regular": ["dejavusans"], "bold": ["dejavusansbold"], "i": ["dejavusansoblique"], "boldi": ["dejavusansboldoblique"]},
    "liberationsans": {"regular": ["liberationsansregular"], "bold": ["liberationsansbold"], "i": ["liberationsansitalic"],
                       "boldi": ["liberationsansbolditalic"]},
    "roboto": {"regular": ["robotoregular", "roboto"], "bold": ["robotobold"], "black": ["robotoblack"], "light": ["robotolight"],
               "i": ["robotoitalic"]},
    "ebgaramond": {"regular": ["ebgaramondvf"], "i": ["ebgaramonditalicvf"]},
    "lora": {"regular": ["loravf"], "i": ["loraitalicvf"]},
    "georgia": {"regular": ["georgia"], "bold": ["georgiab", "georgiabold"], "i": ["georgiai", "georgiaitalic"],
                "boldi": ["georgiaz", "georgiabolditalic"]},
    "cambria": {"regular": ["cambria"], "bold": ["cambriab"], "i": ["cambriai"], "boldi": ["cambriaz"]},
    "palatino": {"regular": ["pala", "palatino"], "bold": ["palab"], "i": ["palai"], "boldi": ["palabi"]},
    "times": {"regular": ["times", "timesnewroman"], "bold": ["timesbd", "timesnewromanbold"],
              "i": ["timesi", "timesnewromanitalic"], "boldi": ["timesbi", "timesnewromanbolditalic"]},
    "garamond": {"regular": ["gara", "garamond"], "bold": ["garabd"], "i": ["garait"]},
    "notoserif": {"regular": ["notoserifvf", "notoserifregular"], "bold": ["notoserifbold"], "i": ["notoserifitalic"]},
    "dejavuserif": {"regular": ["dejavuserif"], "bold": ["dejavuserifbold"], "i": ["dejavuserifitalic"]},
    "liberationserif": {"regular": ["liberationserifregular"], "bold": ["liberationserifbold"], "i": ["liberationserifitalic"]},
    "playfair": {"regular": ["playfairdisplayvf"], "i": ["playfairdisplayitalicvf"]},
    "dmserif": {"regular": ["dmserifdisplayregular"], "i": ["dmserifdisplayitalic"]},
    "abril": {"regular": ["abrilfatfaceregular"]},
    "cinzel": {"regular": ["cinzelvf"]},
    "robotoslab": {"regular": ["robotoslabvf"]},
    "zillaslab": {"regular": ["zillaslabregular"], "bold": ["zillaslabbold"], "light": ["zillaslablight"],
                  "medium": ["zillaslabmedium"], "i": ["zillaslabitalic"], "boldi": ["zillaslabbolditalic"]},
    "arvo": {"regular": ["arvoregular"], "bold": ["arvobold"], "i": ["arvoitalic"], "boldi": ["arvobolditalic"]},
    "rockwell": {"regular": ["rock", "rockwell"], "bold": ["rockb"], "i": ["rocki"]},
    "jetbrainsmono": {"regular": ["jetbrainsmonovf"], "i": ["jetbrainsmonoitalicvf"]},
    "spacemono": {"regular": ["spacemonoregular"], "bold": ["spacemonobold"], "i": ["spacemonoitalic"], "boldi": ["spacemonobolditalic"]},
    "cascadia": {"regular": ["cascadiamono", "cascadiacode"]},
    "consolas": {"regular": ["consola"], "bold": ["consolab"], "i": ["consolai"], "boldi": ["consolaz"]},
    "menlo": {"regular": ["menlo", "sfmono", "monaco"]},
    "dejavumono": {"regular": ["dejavusansmono"], "bold": ["dejavusansmonobold"]},
    "liberationmono": {"regular": ["liberationmonoregular"], "bold": ["liberationmonobold"]},
    "courier": {"regular": ["cour", "couriernew"], "bold": ["courbd"], "i": ["couri"]},
    "bebas": {"regular": ["bebasneueregular"]},
    "anton": {"regular": ["antonregular"]},
    "archivoblack": {"regular": ["archivoblackregular"]},
    "bungee": {"regular": ["bungeeregular"]},
    "monoton": {"regular": ["monotonregular"]},
    "impact": {"regular": ["impact"]},
    "arialblack": {"regular": ["ariblk", "arialblack"]},
    "segoeblack": {"regular": ["seguibl"], "i": ["seguibli"]},
    "oswald": {"regular": ["oswaldvf"]},
    "barlowcondensed": {"regular": ["barlowcondensedregular"], "bold": ["barlowcondensedbold"],
                        "black": ["barlowcondensedblack"], "light": ["barlowcondensedlight"],
                        "medium": ["barlowcondensedmedium"], "i": ["barlowcondenseditalic"], "boldi": ["barlowcondensedbolditalic"]},
    "bahnschrift": {"regular": ["bahnschrift#Condensed"]},
    "arialnarrow": {"regular": ["arialn"], "bold": ["arialnb"], "i": ["arialni"], "boldi": ["arialnbi"]},
    "dejavucondensed": {"regular": ["dejavusanscondensed"], "bold": ["dejavusanscondensedbold"]},
    "liberationnarrow": {"regular": ["liberationsansnarrowregular"], "bold": ["liberationsansnarrowbold"]},
    "nunito": {"regular": ["nunitovf"], "i": ["nunitoitalicvf"]},
    "fredoka": {"regular": ["fredokavf"]},
    "mplusrounded": {"regular": ["mplusrounded1cregular"], "bold": ["mplusrounded1cbold"], "black": ["mplusrounded1cblack"],
                     "light": ["mplusrounded1clight"], "medium": ["mplusrounded1cmedium"]},
    "arialrounded": {"regular": ["arlrdbd", "arialroundedmtbold", "arialroundedbold"]},
    "caveat": {"regular": ["caveatvf"]},
    "permanentmarker": {"regular": ["permanentmarkerregular"]},
    "indieflower": {"regular": ["indieflowerregular"]},
    "inkfree": {"regular": ["inkfree"]},
    "segoeprint": {"regular": ["segoepr"], "bold": ["segoeprb"]},
    "chalkboard": {"regular": ["chalkboard", "chalkboardse"]},
    "markerfelt": {"regular": ["markerfelt"]},
    "noteworthy": {"regular": ["noteworthy"]},
    "comic": {"regular": ["comic", "comicsansms"], "bold": ["comicbd"], "i": ["comici"]},
    "greatvibes": {"regular": ["greatvibesregular"]},
    "dancingscript": {"regular": ["dancingscriptvf"]},
    "pacifico": {"regular": ["pacificoregular"]},
    "lobster": {"regular": ["lobsterregular"]},
    "segoescript": {"regular": ["segoesc"], "bold": ["segoescb"]},
    "snell": {"regular": ["snellroundhand"]},
    "zapfino": {"regular": ["zapfino"]},
    "frenchscript": {"regular": ["frscript"]},
    "corsiva": {"regular": ["mtcorsva", "applechancery"]},
    "gabriola": {"regular": ["gabriola"]},
}

_STYLE_FAMILIES = {
    "sans": ["inter", "segoe", "helvetica", "sfpro", "arial", "notosans", "montserrat", "roboto", "dejavusans", "liberationsans"],
    "serif": ["ebgaramond", "lora", "georgia", "cambria", "palatino", "times", "garamond", "notoserif", "dejavuserif",
              "liberationserif", "playfair", "dmserif", "abril", "cinzel"],
    "slab": ["robotoslab", "zillaslab", "arvo", "rockwell"],
    "mono": ["jetbrainsmono", "spacemono", "cascadia", "consolas", "menlo", "dejavumono", "liberationmono", "courier"],
    "display": ["bebas", "anton", "archivoblack", "abril", "playfair", "dmserif", "bungee", "impact", "arialblack", "segoeblack",
                "oswald"],
    "condensed": ["oswald", "barlowcondensed", "bebas", "anton", "bahnschrift", "arialnarrow", "impact", "dejavucondensed",
                  "liberationnarrow"],
    "rounded": ["nunito", "fredoka", "mplusrounded", "arialrounded"],
    "handwriting": ["caveat", "permanentmarker", "indieflower", "inkfree", "segoeprint", "chalkboard", "markerfelt",
                    "noteworthy", "comic"],
    "script": ["greatvibes", "dancingscript", "pacifico", "lobster", "segoescript", "snell", "zapfino", "frenchscript",
               "corsiva", "gabriola"],
}
_STYLE_WORDS = set(_STYLE_FAMILIES) | set(_STYLE_ALIASES) | {"cjk", "cjk-serif"}
_STYLE_BACKUP = {"slab": "serif", "rounded": "sans", "handwriting": "script", "script": "serif", "display": "sans",
                 "condensed": "sans", "mono": "sans"}

# CJK families per language: (sans regular, sans bold, serif) lists of normalised stems
_CJK = {
    "tc": (["notosanstcvf", "msjh", "notosanscjktcregular", "notosanscjkregular", "pingfang", "notosanshkvf", "msyh",
            "notosansscvf", "simhei", "wqymicrohei", "wqyzenhei", "droidsansfallbackfull", "arialuni"],
           ["notosanstcvf", "msjhbd", "notosanscjktcbold", "notosanscjkbold", "pingfang", "msyhbd", "notosansscvf"],
           ["notoseriftcvf", "notoserifhkvf", "notoserifcjktcregular", "notoserifcjkregular", "mingliu", "pmingliu",
            "songti", "simsun", "wqyzenhei", "droidsansfallbackfull", "msjh"]),
    "sc": (["notosansscvf", "msyh", "notosanscjkscregular", "notosanscjkregular", "pingfang", "simhei", "wqymicrohei", "wqyzenhei", "droidsansfallbackfull", "msjh"],
           ["notosansscvf", "msyhbd", "notosanscjkscbold", "pingfang", "simhei", "msjhbd"],
           ["notoserifscvf", "notoserifcjkscregular", "songti", "simsun", "notoseriftcvf"]),
    "ja": (["notosansjpvf", "yugothm", "yugothr", "hiraginosans", "hiraginokakugothicpron", "notosanscjkjpregular",
            "msgothic", "wqymicrohei", "droidsansfallbackfull", "msjh"],
           ["notosansjpvf", "yugothb", "hiraginosans", "notosanscjkjpbold", "msgothic", "msjhbd"],
           ["notoserifjpvf", "yumin", "hiraginomincho", "hiraginominchopron", "notoserifcjkjpregular", "msmincho"]),
    "ko": (["notosanskrvf", "malgun", "applesdgothicneo", "notosanscjkkrregular", "nanumgothic", "notosanscjkregular", "droidsansfallbackfull"],
           ["notosanskrvf", "malgunbd", "applesdgothicneo", "notosanscjkkrbold", "nanumgothicbold"],
           ["notoserifkrvf", "notoserifcjkkrregular", "applemyungjo", "nanummyeongjo", "batang"]),
}
_CJK_BRUSH = ["kaiu", "stkaiti", "kaiti", "simkai", "biaukai", "notoseriftcvf"]

_SYMBOL_STEMS = ["seguisym", "notosanssymbols2regular", "notosanssymbolsvf", "notosansmathregular", "cambria#1",
                 "seguiemj", "notoemojivf", "segoeui", "arial", "notosansvf", "dejavusans", "l10646", "arialuni",
                 "notosanssymbolsregular", "liberationsansregular", "applesymbols", "stixtwomath", "symbola", "seguihis", "ebrima", "nirmala", "leelawui", "gadugi",
                 "sylfaen", "himalaya", "mvboli", "javatext", "notosansregular", "freeserif", "droidsansfallbackfull", "wqyzenhei",
                 "unifont"]


def _stem_path(stem):
    s, _, hint = stem.partition("#")
    p = _norm_index().get(s)
    if p is None:
        return None
    return p + ("#" + hint if hint else "")


def _weight(bold):
    if bold is True:
        return "bold"
    if not bold:
        return "regular"
    return _WEIGHT_WORDS.get(str(bold).lower().replace(" ", "").replace("-", ""), "bold")


@lru_cache(maxsize=512)
def _family_file(fam, weight, italic):
    v = _FAMILIES.get(fam, {})
    order = []
    if italic:
        order += [weight + "i" if weight != "regular" else "i"]
    order += [weight]
    if weight == "black":
        order += ["bold"]
    if italic:
        order += ["i"]
    order += ["regular"]
    for key in order:
        for stem in v.get(key, []):
            p = _stem_path(stem)
            if p:
                return p
    return None


def _resolve_style(words, weight="regular", italic=False):
    words = [_STYLE_ALIASES.get(w, w) for w in words]
    scores = {}
    for w in words:
        fams = _STYLE_FAMILIES.get(w, [])
        for r, fam in enumerate(fams):
            scores[fam] = scores.get(fam, 0.0) + 1.0 + 0.9 * (len(fams) - r) / len(fams)
    for fam in sorted(scores, key=lambda f: -scores[f]):
        p = _family_file(fam, weight, italic)
        if p:
            return p
    for w in words:
        b = _STYLE_BACKUP.get(w)
        if b and b not in words:
            return _resolve_style([b], weight, italic)
    if "sans" not in words:
        return _resolve_style(["sans"], weight, italic)
    return None


def _cjk_lang(text, lang=None):
    if lang:
        l_ = str(lang).lower()
        if l_.startswith(("ja", "jp")):
            return "ja"
        if l_.startswith(("ko", "kr")):
            return "ko"
        if l_ in ("sc", "zh-hans", "zh-cn", "zh_cn", "cn", "zh-sg"):
            return "sc"
        if l_.startswith(("tc", "zh")):
            return "tc"
    if text is not None:
        t = str(text)
        if _KANA_RE.search(t):
            return "ja"
        if _HANGUL_RE.search(t):
            return "ko"
    return "tc"


def _cjk_font(lang="tc", weight="regular", kind="sans"):
    sans, bold, serif = _CJK[lang]
    if kind == "brush":
        lst = _CJK_BRUSH + serif
    elif kind == "serif":
        lst = serif
    else:
        lst = (bold if weight in ("bold", "black") else []) + sans
    for stem in lst:
        p = _stem_path(stem)
        if p:
            return p
    for other in ("tc", "sc", "ja", "ko"):
        if other != lang:
            for stem in _CJK[other][0]:
                p = _stem_path(stem)
                if p:
                    return p
    return None


def _style_kind(spec_or_style):
    """'serif' / 'brush' / 'sans' — which CJK family suits a Latin font."""
    s = _norm(spec_or_style or "")
    for fam in _STYLE_FAMILIES["handwriting"] + _STYLE_FAMILIES["script"]:
        for stems in _FAMILIES[fam].values():
            if any(st.split("#")[0] in s for st in stems):
                return "brush"
    for fam in _STYLE_FAMILIES["serif"] + _STYLE_FAMILIES["slab"]:
        for stems in _FAMILIES[fam].values():
            if any(st.split("#")[0] and s.endswith(st.split("#")[0]) for st in stems):
                return "serif"
    if any(k in s for k in ("serif", "mincho", "ming", "song", "times", "georgia", "garamond")) and "sans" not in s:
        return "serif"
    return "sans"


def find_font(name=None, bold=False, style="sans", text=None, italic=False, lang=None):
    """Return a font spec (path, optionally ``path#variation``), or None (→ Pillow's built-in font).

    * ``name``: a path, a file stem / family substring (``"georgia"``, ``"Noto Serif TC"``), or style words:
      ``sans`` ``serif`` ``slab`` ``mono`` ``display`` ``condensed`` ``rounded`` ``handwriting`` ``script``
      plus ``bold`` / ``black`` / ``heavy`` / ``light`` / ``italic`` — e.g. ``"condensed bold"``, ``"serif italic"``.
    * ``style``: style word used when ``name`` is None (also ``"cjk"`` / ``"cjk-serif"``).
    * ``text``: CJK text picks a CJK font (Japanese kana → Japanese font, Hangul → Korean); ``lang`` forces one
      (``"ja"``, ``"ko"``, ``"sc"``, ``"tc"``).
    * Env overrides: ``PIXELCANVAS_FONT`` (Latin default), ``PIXELCANVAS_CJK_FONT`` (CJK default),
      ``PIXELCANVAS_FONT_DIR`` (extra search directories). Fonts in ``~/.pixelcanvas/fonts`` are searched first.
    """
    weight = _weight(bold)
    if name is not None and not isinstance(name, str):
        return name
    if name:
        if os.path.isfile(_split_spec(name)[0]):
            return str(name)
        words = [w for w in re.split(r"[\s,+/]+", name.lower().strip()) if w]
        mods = [w for w in words if w in _WEIGHT_WORDS or w in _ITALIC_WORDS]
        rest = [w for w in words if w not in mods]
        for w in mods:
            if w in _ITALIC_WORDS:
                italic = True
            elif _WEIGHT_WORDS[w] != "regular" or weight == "regular":
                weight = _WEIGHT_WORDS[w]
        if rest and all(w in _STYLE_WORDS for w in rest):
            st = [_STYLE_ALIASES.get(w, w) for w in rest]
            if text is not None and has_cjk(text) and "cjk" not in st[0]:
                return _cjk_font(_cjk_lang(text, lang), weight,
                                 "serif" if st[0] in ("serif", "slab") else "brush" if st[0] in ("script", "handwriting") else "sans")
            if st[0].startswith("cjk"):
                return _cjk_font(_cjk_lang(text, lang), weight, "serif" if st[0] == "cjk-serif" else "sans")
            return _resolve_style(st, weight, italic)
        if not rest:
            name = None
        else:
            name = " ".join(rest)
            key = name.replace(" ", "")
            idx = _font_index()
            if key in idx:
                return idx[key]
            nkey = _norm(name)
            nidx = _norm_index()
            if nkey in nidx and weight == "regular" and not italic:
                return nidx[nkey]
            hits = sorted((k for k in nidx if nkey in k), key=len)
            if not hits:
                return None

            def score(k):
                s = 0.0
                isb = any(t in k for t in ("bold", "black", "heavy")) or k.endswith(("bd", "b", "z", "bi"))
                isi = "italic" in k or "oblique" in k or k.endswith(("i", "z", "bi", "it"))
                s += 2.0 if (weight in ("bold", "black")) == isb else 0.0
                s += 2.0 if italic == isi else 0.0
                if weight == "black" and ("black" in k or "heavy" in k):
                    s += 1.0
                if weight == "light" and ("light" in k or "thin" in k):
                    s += 1.0
                if "vf" in k:
                    s += 1.5
                return -s, len(k)
            return nidx[sorted(hits, key=score)[0]]
    if text is not None and has_cjk(text) and not style.startswith("cjk"):
        style = "cjk-serif" if style in ("serif", "slab") else "cjk"
    env = os.environ.get("PIXELCANVAS_CJK_FONT" if style.startswith("cjk") else "PIXELCANVAS_FONT")
    if env and os.path.isfile(env):
        return env
    if style.startswith("cjk"):
        return _cjk_font(_cjk_lang(text, lang), weight, "serif" if style == "cjk-serif" else "sans")
    st = _STYLE_ALIASES.get(style, style)
    if st in _STYLE_FAMILIES:
        return _resolve_style([st], weight, italic)
    return _resolve_style(["sans"], weight, italic)


def list_fonts(pattern=""):
    """Sorted list of (stem, path) for installed fonts whose file name contains ``pattern``."""
    pat = pattern.lower().replace(" ", "")
    return sorted((k, p) for k, p in _font_index().items() if pat in k.replace(" ", ""))


# --------------------------------------------------------------------------------------------- loading

_VAR_NAMES = {"regular": ["Regular", "Text", "Book", "Normal", "Roman"],
              "medium": ["Medium", "SemiBold", "Semibold", "Regular"],
              "bold": ["Bold", "SemiBold", "Semibold", "Text Bold", "Bold Text", "ExtraBold"],
              "black": ["Black", "ExtraBold", "Heavy", "UltraBold", "Bold"],
              "light": ["Light", "ExtraLight", "SemiLight", "Semilight", "Thin", "DemiLight"]}
_HEAVY_STEMS = ("black", "heavy", "blk", "impact", "anton", "bebas", "bungee", "abril", "archivo", "seguibl", "monoton")


@lru_cache(maxsize=512)
def _load2(spec, size, weight="regular"):
    """(Pillow font, synthetic-bold amount as a fraction of size)."""
    size = max(1, int(size))
    if spec is None:
        try:
            return ImageFont.load_default(size), 0.0
        except TypeError:
            return ImageFont.load_default(), 0.0
    path, index, hint = _split_spec(spec)
    f = ImageFont.truetype(path, size, index=index)
    synth = 0.0
    names = []
    try:
        names = [n.decode() if isinstance(n, bytes) else n for n in f.get_variation_names()]
    except Exception:
        names = []
    if names:
        tries = _VAR_NAMES.get(weight, _VAR_NAMES["regular"])
        cands = ([f"{t} {hint}" for t in tries] + [hint] if hint else []) + tries
        done = False
        for nm in cands:
            if nm in names:
                try:
                    f.set_variation_by_name(nm)
                    done = True
                    break
                except Exception:
                    pass
        if not done and weight in ("bold", "black"):
            synth = 0.022 if weight == "bold" else 0.04
    else:
        style = (f.getname()[1] or "").lower()
        stem = os.path.basename(path).lower()
        heavy = any(w in style for w in ("bold", "black", "heavy", "semibold", "demi")) or any(h in stem for h in _HEAVY_STEMS)
        if weight in ("bold", "black") and not heavy:
            synth = 0.022 if weight == "bold" else 0.04
        elif weight == "black" and "black" not in style and "heavy" not in style and not any(h in stem for h in _HEAVY_STEMS):
            synth = 0.018
    return f, synth


def _load(path, size, bold):
    """Backwards-compatible loader: Pillow font only."""
    return _load2(path, size, _weight(bold))[0]


_WARNED = set()


def _warn_once(msg):
    if msg not in _WARNED:
        _WARNED.add(msg)
        print("pixelcanvas: " + msg, file=sys.stderr)


def load_font(font=None, size=40, bold=False, text=None, style="sans"):
    """Pillow font object. ``font`` = path / name / style words / None (auto, CJK-aware via ``text``) / an ImageFont."""
    if isinstance(font, (ImageFont.FreeTypeFont, ImageFont.ImageFont)):
        return font
    path = find_font(font, bold=bold, style=style, text=text)
    if path is None and text is not None and has_cjk(text):
        _warn_once("no CJK font found: CJK text will render as empty boxes. Run `python tools/fonts.py --install "
                   "noto-sans-tc` (or set PIXELCANVAS_CJK_FONT=/path/to/font)")
    return _load2(path, max(1, int(round(size))), _weight(bold))[0]


@lru_cache(maxsize=256)
def _is_italic(spec):
    if spec is None:
        return False
    try:
        f = _load2(spec, 20)[0]
        style = (f.getname()[1] or "").lower()
    except Exception:
        style = ""
    return "italic" in style or "oblique" in style


@lru_cache(maxsize=256)
def _italic_sibling(spec):
    if spec is None:
        return None
    path, index, hint = _split_spec(spec)
    stem = _norm(os.path.splitext(os.path.basename(path))[0])
    cands = [stem + "i", stem + "italic", stem.replace("regular", "italic"), stem.replace("vf", "italicvf"),
             stem.replace("bd", "bi"), stem[:-1] + "z" if stem.endswith("b") else stem + "z", stem + "it",
             stem.replace("bold", "bolditalic")]
    for c in cands:
        if c != stem:
            p = _norm_index().get(c)
            if p and _is_italic(p):
                return p + ("#" + hint if hint else "")
    return None


# --------------------------------------------------------------------------------------------- fallback

_FALLBACK_SCAN = {}


def _covers(spec, cp):
    if spec is None:
        return 32 <= cp < 127
    return cp in _info(spec).cmap


@lru_cache(maxsize=256)
def _chain(primary, weight="regular", kind="sans", lang="tc"):
    """Fonts tried in order for each glyph: the chosen font, CJK fonts for every language, symbol fonts."""
    out = [primary]
    seen = {primary}
    langs = [lang] + [l_ for l_ in ("tc", "sc", "ja", "ko") if l_ != lang]
    for l_ in langs:
        p = _cjk_font(l_, weight, kind if kind != "brush" or l_ in ("tc", "sc") else "sans")
        if p and p not in seen:
            out.append(p)
            seen.add(p)
    for stem in _SYMBOL_STEMS:
        p = _stem_path(stem)
        if p and p not in seen:
            out.append(p)
            seen.add(p)
    return tuple(out)


def _scan_for(cp):
    """Last resort: search every installed font for a code point (cached)."""
    if cp in _FALLBACK_SCAN:
        return _FALLBACK_SCAN[cp]
    found = None
    for p in sorted(set(_font_index().values())):
        try:
            if cp in _sfnt(p, 0).cmap:
                found = p
                break
        except Exception:
            continue
    _FALLBACK_SCAN[cp] = found
    return found


_SKIP_COVER = set(" \t\n\r\u00a0\u3000\u200b")


def _font_for(cp, chain):
    for spec in chain:
        if _covers(spec, cp):
            return spec
    return _scan_for(cp)


def _split_runs(text, chain):
    """[(text, font spec or None if nothing has the glyph)] — consecutive characters sharing a font."""
    runs = []
    cur, buf = None, []
    for ch in text:
        if ch in _SKIP_COVER or unicodedata_category(ch).startswith("M"):
            spec = cur if cur is not None else chain[0]
        else:
            spec = _font_for(ord(ch), chain)
            if spec is None:
                spec = ("__missing__", chain[0])
        if buf and spec != cur:
            runs.append(("".join(buf), cur))
            buf = []
        cur = spec
        buf.append(ch)
    if buf:
        runs.append(("".join(buf), cur))
    return runs


def unicodedata_category(ch):
    import unicodedata
    return unicodedata.category(ch)


def missing_glyphs(text, font=None, fallback=True):
    """Characters of ``text`` that no font can draw (with ``fallback=False``: that ``font`` lacks)."""
    text = str(text).translate(_IGNORABLE)
    spec = find_font(font, text=text) if (font is None or isinstance(font, str)) else font
    chain = _chain(spec, "regular", _style_kind(spec), _cjk_lang(text)) if fallback else (spec,)
    out = []
    for ch in dict.fromkeys(text):
        if ch in _SKIP_COVER or ch.isspace() or unicodedata_category(ch)[0] == "M" or unicodedata_category(ch) in ("Cc", "Cf", "Cs"):
            continue
        if fallback:
            if _font_for(ord(ch), chain) is None:
                out.append(ch)
        elif not _covers(spec, ord(ch)):
            out.append(ch)
    return out


# --------------------------------------------------------------------------------------------- shaping

_MS = 4                 # measure fonts at 4x so hinted advances stay precise at small sizes


@lru_cache(maxsize=4096)
def _shape(text, spec, px, weight, track, tabular=False):
    """Per-character x offsets (px), total advance (px) for one fallback run, kerning + tracking applied."""
    f = _load2(spec, round(px * _MS), weight)[0]
    scale = px / max(round(px * _MS), 1)
    info = _info(spec)
    kscale = px / info.upem if info else 0.0
    xs, x = [], 0.0
    digit_w = None
    if tabular:
        digit_w = max(f.getlength(d) for d in "0123456789") * scale
    prev = None
    for ch in text:
        if prev is not None and info is not None:
            x += info.kern_units(prev, ch) * kscale
        adv = f.getlength(ch) * scale
        if digit_w is not None and ch.isdigit():
            xs.append(x + (digit_w - adv) / 2)
            x += digit_w + track
        else:
            xs.append(x)
            x += adv + track
        prev = ch
    return tuple(xs), x


@lru_cache(maxsize=1024)
def _metrics(spec, px, weight="regular"):
    """(ascent, descent, cap height, x height) in px."""
    f = _load2(spec, round(px * _MS), weight)[0]
    a, d = f.getmetrics()
    sc = px / max(round(px * _MS), 1)
    info = _info(spec)
    cap = info.cap * px if info else 0.7 * px
    xh = info.xh * px if info else 0.5 * px
    return a * sc, d * sc, cap, xh


@lru_cache(maxsize=2048)
def _ink(spec, px, weight, ch):
    """Ink box (l, t, r, b) of one character relative to its baseline origin, px.

    Measured from the rendered glyph: Pillow's ``getbbox`` starts at the pen origin and spans at least the advance,
    which hides side bearings and the true position of punctuation.
    """
    n = max(round(px * _MS), 1)
    b = _ink_units(spec, n, weight, ch)
    if b is None:
        return (0.0, -px * 0.7, px * 0.5, 0.0)
    sc = px / n
    return tuple(v * sc for v in b)


@lru_cache(maxsize=4096)
def _ink_units(spec, n, weight, ch):
    f = _load2(spec, n, weight)[0]
    try:
        m, off = f.getmask2(ch, anchor="ls")
        bb = m.getbbox()
        if bb is None:
            b = f.getbbox(ch, anchor="ls")
            return (float(b[0]), float(b[1]), float(b[0]), float(b[1]))
        return (float(off[0] + bb[0]), float(off[1] + bb[1]), float(off[0] + bb[2]), float(off[1] + bb[3]))
    except Exception:
        return None


# --------------------------------------------------------------------------------------------- rich text

_NAMED = {"red": "#e53935", "orange": "#fb8c00", "yellow": "#fdd835", "green": "#43a047", "blue": "#1e88e5",
          "purple": "#8e24aa", "pink": "#ec407a", "white": "#ffffff", "black": "#000000", "gray": "#9e9e9e",
          "grey": "#9e9e9e", "gold": "#f2c14e", "cyan": "#00bcd4", "magenta": "#d81b60", "teal": "#00897b",
          "navy": "#1a237e", "cream": "#f6efe0", "brown": "#6d4c41"}
_MARKUP_HINT = re.compile(r"\*\*.+?\*\*|\{[^{}|]*\|[^{}]*\}|~~.+?~~|==.+?==|__.+?__", re.S)


def _parse_attrs(spec):
    st = {}
    for tok in re.split(r"[\s,;]+", spec.strip()):
        if not tok:
            continue
        k, _, v = tok.partition("=")
        k = k.lower()
        if tok.startswith("#"):
            if re.fullmatch(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})", tok):
                st["color"] = tok
        elif k in _NAMED and not v:
            st["color"] = _NAMED[k]
        elif k in ("b", "bold"):
            st["bold"] = v or True
        elif k in ("black", "heavy", "light"):
            st["bold"] = k
        elif k in ("i", "italic", "em"):
            st["italic"] = True
        elif k in ("u", "underline"):
            st["underline"] = v or True
        elif k in ("s", "strike", "del"):
            st["strike"] = True
        elif k in ("hl", "highlight", "mark"):
            st["highlight"] = v or "#ffe45c"
        elif k in ("size", "sz"):
            st["size"] = v
        elif k in ("color", "c"):
            st["color"] = _NAMED.get(v, v)
        elif k in ("font", "f"):
            st["font"] = v.replace("_", " ")
        elif k in ("caps", "upper", "uppercase"):
            st["case"] = "upper"
        elif k in ("lower", "lowercase"):
            st["case"] = "lower"
        elif k in ("sc", "smallcaps"):
            st["case"] = "smallcaps"
        elif k in ("track", "tracking"):
            st["tracking"] = float(v)
        elif k in ("off",):
            st["off"] = True
    return st


def parse_markup(text):
    """Light markup → [(text, style)] spans. ``**bold**`` ``*italic*`` ``__underline__`` ``~~strike~~``
    ``==highlight==`` ``{#ff0|colour}`` ``{size=1.5|bigger}`` ``{b #e33 u|several}`` ``{font=georgia|face}``
    ``{sc|small caps}`` ``{hl=#aef|marker}``; ``\\*`` escapes."""
    spans = []
    stack = [{}]
    toggles = {"**": "bold", "*": "italic", "__": "underline", "~~": "strike", "==": "highlight"}
    state = {v: False for v in toggles.values()}
    buf = []

    def cur():
        st = {}
        for s in stack:
            st.update(s)
        for k, on in state.items():
            if on:
                st[k] = "#ffe45c" if k == "highlight" else True
        return st

    def flush():
        if buf:
            spans.append(("".join(buf), cur()))
            buf.clear()
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "\\" and i + 1 < n:
            buf.append(text[i + 1])
            i += 2
            continue
        if c == "{":
            j = text.find("|", i)
            k = text.find("}", i)
            if j != -1 and (k == -1 or j < k) and "{" not in text[i + 1:j]:
                flush()
                stack.append(_parse_attrs(text[i + 1:j]))
                i = j + 1
                continue
        if c == "}" and len(stack) > 1:
            flush()
            stack.pop()
            i += 1
            continue
        matched = False
        for tok in ("**", "__", "~~", "==", "*"):
            if text.startswith(tok, i):
                key = toggles[tok]
                opening = not state[key]
                if opening and text.find(tok, i + len(tok)) == -1:
                    break
                if tok == "*" and opening and (i + 1 >= n or text[i + 1].isspace()):
                    break
                flush()
                state[key] = opening
                i += len(tok)
                matched = True
                break
        if matched:
            continue
        buf.append(c)
        i += 1
    flush()
    return spans


# --------------------------------------------------------------------------------------------- layout model

class _Sty:
    """Resolved style of a piece of text."""
    __slots__ = ("font", "weight", "italic", "synth_i", "px", "color", "ul", "st", "hl", "track", "tabular", "chain",
                 "key", "off")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def ident(self):
        return (self.font, self.weight, self.italic, self.px, self.track, self.tabular)


class _Box:
    __slots__ = ("segs", "w", "pre", "pre_sty", "brk", "pen", "text")

    def __init__(self):
        self.segs, self.w, self.pre, self.pre_sty, self.brk, self.pen, self.text = [], 0.0, 0.0, None, True, 0.0, ""


_NO_LINE_START = set("，。、！？；：）」』】》〉,.!?;:)]}%·ー…‥〕］｝〟’”ゝゞヽヾぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ々")
_NO_LINE_END = set("（「『【《〈〔［｛〝‘“([{")
_SHORT_WORDS = {"a", "an", "the", "of", "to", "in", "on", "at", "by", "for", "and", "or", "&", "with", "as", "is",
                "de", "la", "le", "el", "y", "et", "und", "der", "die", "das", "i"}
_BREAK_AFTER_PUNCT = set(",;:—–!?.)，。、！？；：")


class _Block:
    """Laid-out text: glyph placements in canvas pixels, decorations, line info and bbox."""

    def __init__(self):
        self.pl = []              # (text, spec, px, weight, synth, M (a,b,c,d) or None, tx, ty, key)
        self.decor = []           # (x0, y0, x1, y1, M, (ox, oy), key)  rectangles in local coords
        self.hl = []              # (x0, y0, x1, y1, M, (ox, oy), color)
        self.bbox = (0.0, 0.0, 0.0, 0.0)
        self.quad = None
        self.px = 40.0
        self.lines = []           # (baseline, cap height) in block-local y
        self.rot = None           # (angle deg, pivot) applied after layout
        self.fonts, self.missing = set(), []
        self._cache = {}

    # ---- geometry
    def translate(self, dx, dy):
        self.pl = [(t, s, p, w, sy, M, tx + dx, ty + dy, k) for (t, s, p, w, sy, M, tx, ty, k) in self.pl]
        self.decor = [(a, b, c, d, M, (o[0] + dx, o[1] + dy), k) for (a, b, c, d, M, o, k) in self.decor]
        self.hl = [(a, b, c, d, M, (o[0] + dx, o[1] + dy), k) for (a, b, c, d, M, o, k) in self.hl]
        x0, y0, x1, y1 = self.bbox
        self.bbox = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
        if self.quad is not None:
            self.quad = [(x + dx, y + dy) for x, y in self.quad]
        if self.rot is not None:
            self.rot = (self.rot[0], (self.rot[1][0] + dx, self.rot[1][1] + dy))
        self.lines = [(b + dy, c, l0 + dx, l1 + dx) for (b, c, l0, l1) in self.lines]
        self._cache = {}

    def rotate(self, deg, pivot):
        """Rotate counter-clockwise by ``deg`` around ``pivot``."""
        if not deg:
            return
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        R = (c, s, -s, c)                       # y down: visually counter-clockwise
        px_, py_ = pivot

        def rp(x, y):
            dx, dy = x - px_, y - py_
            return px_ + c * dx + s * dy, py_ - s * dx + c * dy
        new = []
        for (t, sp, p, w, sy, M, tx, ty, k) in self.pl:
            M2 = _mat_mul(R, M or (1, 0, 0, 1))
            nx, ny = rp(tx, ty)
            new.append((t, sp, p, w, sy, M2, nx, ny, k))
        self.pl = new
        self.decor = [(a_, b_, c_, d_, _mat_mul(R, M or (1, 0, 0, 1)), rp(*o), k) for (a_, b_, c_, d_, M, o, k) in self.decor]
        self.hl = [(a_, b_, c_, d_, _mat_mul(R, M or (1, 0, 0, 1)), rp(*o), k) for (a_, b_, c_, d_, M, o, k) in self.hl]
        x0, y0, x1, y1 = self.bbox
        q = [rp(x0, y0), rp(x1, y0), rp(x1, y1), rp(x0, y1)]
        self.quad = q
        xs, ys = [p[0] for p in q], [p[1] for p in q]
        self.bbox = (min(xs), min(ys), max(xs), max(ys))
        self.rot = (deg, pivot)
        self._cache = {}

    def corners(self):
        if self.quad is not None:
            return self.quad
        x0, y0, x1, y1 = self.bbox
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    # ---- rendering
    def masks(self, H, W, aa=2, stroke=0.0):
        """(union mask (H, W), {key: mask}) — anti-aliased, supersampled ``aa`` times."""
        ck = (H, W, aa, round(stroke, 3))
        if ck in self._cache:
            return self._cache[ck]
        res = _render_block(self, H, W, aa, stroke)
        self._cache[ck] = res
        return res

    def ink_extent(self, stroke=0.0):
        xs, ys = [], []
        for (t, sp, p, w, sy, M, tx, ty, k) in self.pl:
            adv = p * (0.8 * len(t) + 0.6)
            pad = stroke + p * (0.35 + sy)
            pts = [(-pad, -1.25 * p - pad), (adv + pad, -1.25 * p - pad), (-pad, 0.55 * p + pad), (adv + pad, 0.55 * p + pad)]
            for (u, v) in pts:
                if M is None:
                    xs.append(tx + u)
                    ys.append(ty + v)
                else:
                    xs.append(tx + M[0] * u + M[1] * v)
                    ys.append(ty + M[2] * u + M[3] * v)
        for lst in (self.decor, self.hl):
            for (a, b, c, d, M, o, k) in lst:
                for (u, v) in ((a - stroke, b - stroke), (c + stroke, b - stroke), (a - stroke, d + stroke), (c + stroke, d + stroke)):
                    M_ = M or (1, 0, 0, 1)
                    xs.append(o[0] + M_[0] * u + M_[1] * v)
                    ys.append(o[1] + M_[2] * u + M_[3] * v)
        if not xs:
            return None
        return min(xs), min(ys), max(xs), max(ys)


def _mat_mul(A, B):
    return (A[0] * B[0] + A[1] * B[2], A[0] * B[1] + A[1] * B[3], A[2] * B[0] + A[3] * B[2], A[2] * B[1] + A[3] * B[3])


def _render_block(block, H, W, aa, stroke):
    aa = max(int(aa), 1)
    ext = block.ink_extent(stroke)
    full = np.zeros((H, W), F32)
    if ext is None:
        return full, {}
    x0, y0 = max(int(math.floor(ext[0])) - 2, 0), max(int(math.floor(ext[1])) - 2, 0)
    x1, y1 = min(int(math.ceil(ext[2])) + 2, W), min(int(math.ceil(ext[3])) + 2, H)
    if x1 <= x0 or y1 <= y0:
        return full, {}
    cw, ch = x1 - x0, y1 - y0
    ox, oy = x0 * aa, y0 * aa
    imgs = {}

    def img_for(key):
        if key not in imgs:
            imgs[key] = Image.new("L", (cw * aa, ch * aa), 0)
        return imgs[key]
    for (t, sp, p, w, sy, M, tx, ty, key) in block.pl:
        if not t.strip():
            continue
        f = _load2(sp, round(p * aa), w)[0]
        sw = (stroke + sy * p) * aa
        sw = int(round(sw)) if sw >= 0.5 else 0
        im = img_for(key)
        X, Y = tx * aa - ox, ty * aa - oy
        if M is None or (abs(M[0] - 1) < 1e-6 and abs(M[1]) < 1e-6 and abs(M[2]) < 1e-6 and abs(M[3] - 1) < 1e-6):
            ImageDraw.Draw(im).text((X, Y), t, fill=255, font=f, anchor="ls", stroke_width=sw, stroke_fill=255)
        else:
            _draw_affine(im, t, f, sw, M, X, Y)
    for (a, b, c, d, M, o, key) in block.decor:
        im = img_for(key)
        M_ = M or (1, 0, 0, 1)
        s = stroke
        pts = [(a - s, b - s), (c + s, b - s), (c + s, d + s), (a - s, d + s)]
        poly = [((o[0] + M_[0] * u + M_[1] * v) * aa - ox, (o[1] + M_[2] * u + M_[3] * v) * aa - oy) for u, v in pts]
        ImageDraw.Draw(im).polygon(poly, fill=255)
    groups = {}
    for key, im in imgs.items():
        if aa > 1:
            im = im.resize((cw, ch), _RES.BOX)
        m = np.zeros((H, W), F32)
        m[y0:y1, x0:x1] = np.asarray(im, F32) / 255.0
        groups[key] = m
    if len(groups) == 1:
        full = next(iter(groups.values()))
    elif groups:
        full = np.clip(sum(groups.values()), 0, 1).astype(F32)
    return full, groups


def _draw_affine(im, t, f, sw, M, X, Y):
    l_, t_, r_, b_ = f.getbbox(t, anchor="ls", stroke_width=sw)
    m = 3
    tw, th = int(r_ - l_) + 2 * m, int(b_ - t_) + 2 * m
    if tw <= 0 or th <= 0:
        return
    tile = Image.new("L", (tw, th), 0)
    ImageDraw.Draw(tile).text((m - l_, m - t_), t, fill=255, font=f, anchor="ls", stroke_width=sw, stroke_fill=255)
    a, b, c, d = M
    lx0, ly0 = l_ - m, t_ - m
    cs = [(lx0, ly0), (lx0 + tw, ly0), (lx0, ly0 + th), (lx0 + tw, ly0 + th)]
    tx = [X + a * u + b * v for u, v in cs]
    ty = [Y + c * u + d * v for u, v in cs]
    bx0, by0 = int(math.floor(min(tx))), int(math.floor(min(ty)))
    bx1, by1 = int(math.ceil(max(tx))), int(math.ceil(max(ty)))
    det = a * d - b * c
    if abs(det) < 1e-9:
        return
    ia, ib, ic, id_ = d / det, -b / det, -c / det, a / det
    cx, cy = bx0 - X, by0 - Y
    data = (ia, ib, ia * cx + ib * cy - lx0, ic, id_, ic * cx + id_ * cy - ly0)
    out = tile.transform((bx1 - bx0, by1 - by0), _TRANSFORM.AFFINE, data, resample=_RES.BICUBIC)
    W_, H_ = im.size
    cx0, cy0, cx1, cy1 = max(bx0, 0), max(by0, 0), min(bx1, W_), min(by1, H_)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    region = im.crop((cx0, cy0, cx1, cy1))
    part = out.crop((cx0 - bx0, cy0 - by0, cx1 - bx0, cy1 - by0))
    im.paste(ImageChops.lighter(region, part), (cx0, cy0))


# --------------------------------------------------------------------------------------------- building blocks

def _resolve_px(v, base):
    if v is None:
        return base
    if isinstance(v, str):
        s = v.strip().lower()
        if s.endswith("px"):
            return float(s[:-2])
        if s.endswith("%"):
            return base * float(s[:-1]) / 100
        return base * float(s)
    return base * float(v)


def _span_list(text, markup):
    """text / list of (text, style) / list of str → [(text, style dict)]."""
    if isinstance(text, (list, tuple)) and text and all(isinstance(t, (list, tuple)) and len(t) == 2 for t in text):
        out = []
        for t, st in text:
            if isinstance(st, str):
                st = {"color": _NAMED.get(st, st)} if (st.startswith("#") or st in _NAMED) else _parse_attrs(st)
            out.append((str(t), dict(st or {})))
        return out
    if isinstance(text, (list, tuple)):
        text = "\n".join(str(t) for t in text)
    text = str(text)
    if markup is None:
        markup = bool(_MARKUP_HINT.search(text))
    if markup:
        return parse_markup(text)
    return [(text, {})]


def _apply_case(text, case):
    if not case:
        return text
    if case == "upper":
        return text.upper()
    if case == "lower":
        return text.lower()
    if case == "title":
        return " ".join(w[:1].upper() + w[1:] for w in text.split(" "))
    return text


class _Ctx:
    """Base style for a layout call."""

    def __init__(self, size, font, bold, italic, tracking, case, underline, strike, highlight, tabular, lang, plain):
        self.size, self.font, self.bold, self.italic = float(size), font, bold, italic
        self.tracking, self.case = tracking or 0.0, case
        self.underline, self.strike, self.highlight, self.tabular = underline, strike, highlight, tabular
        self.lang = _cjk_lang(plain, lang)
        self.plain = plain
        letters = [c for c in plain if c.isalpha() or _is_cjk_char(c)]
        cjk = sum(1 for c in letters if _is_cjk_char(c))
        self.mostly_cjk = cjk > 0 and cjk >= 0.5 * len(letters)
        self._cache = {}

    def sty(self, d):
        key = tuple(sorted((k, str(v)) for k, v in d.items()))
        if key in self._cache:
            return self._cache[key]
        font = d.get("font", self.font)
        bold = d.get("bold", self.bold)
        italic = d.get("italic", self.italic)
        if isinstance(font, str) and not os.path.isfile(_split_spec(font)[0]):
            for w in font.lower().split():          # "serif bold italic": weight / slant words count too
                if w in _ITALIC_WORDS:
                    italic = True
                elif w in _WEIGHT_WORDS and _WEIGHT_WORDS[w] != "regular" and not bold:
                    bold = _WEIGHT_WORDS[w]
        weight = _weight(bold)
        px = _resolve_px(d.get("size"), self.size)
        if isinstance(font, (ImageFont.FreeTypeFont,)):
            spec = getattr(font, "path", None)
        else:
            spec = find_font(font, bold=bold, text=self.plain if (font is None and self.mostly_cjk) else None,
                             italic=bool(italic), lang=self.lang)
        synth_i = False
        if italic and spec is not None and not _is_italic(spec):
            sib = _italic_sibling(spec)
            if sib:
                spec = sib
            else:
                synth_i = True
        chain = _chain(spec, weight, _style_kind(spec) if font is not None else "sans", self.lang)
        col = d.get("color")
        try:
            col = tuple(float(v) for v in rgb(_NAMED.get(col, col))) if col is not None else None
        except (ValueError, TypeError):
            col = None
        ul = d.get("underline", self.underline)
        hl = d.get("highlight", self.highlight)
        if hl is True:
            hl = "#ffe45c"
        s = _Sty(font=spec, weight=weight, italic=bool(italic), synth_i=synth_i, px=px, color=col,
                 ul=ul, st=d.get("strike", self.strike), hl=hl,
                 track=(float(d.get("tracking", 0.0)) + self.tracking) * px, tabular=self.tabular, chain=chain,
                 off=bool(d.get("off", False)))
        self._cache[key] = s
        return s


def _char_pieces(spans, ctx):
    """[(char, _Sty)] after case transforms (small caps become smaller capitals)."""
    out = []
    for text, d in spans:
        case = d.get("case", ctx.case)
        text = text.translate(_IGNORABLE)
        if case == "smallcaps":
            st_big = ctx.sty(d)
            d2 = dict(d)
            d2["size"] = (_resolve_px(d.get("size"), ctx.size) * 0.78) / ctx.size
            d2["tracking"] = float(d.get("tracking", 0.0)) + 0.05
            st_small = ctx.sty(d2)
            for ch in text:
                if ch.islower():
                    out.append((ch.upper(), st_small))
                else:
                    out.append((ch, st_big))
            continue
        st = ctx.sty(d)
        for ch in _apply_case(text, case):
            out.append((ch, st))
    return out


def _seg_width(text, sty):
    w = 0.0
    for t, spec in _split_runs(text, sty.chain):
        spec = spec[1] if isinstance(spec, tuple) else spec
        w += _shape(t, spec, round(sty.px, 3), sty.weight, round(sty.track, 4), bool(sty.tabular))[1]
    return w


def _tokenize(chars):
    """[(char, sty)] of one paragraph → wrap boxes."""
    boxes = []
    cur = None
    pending, pend_sty = "", None
    prev_ch = ""

    def add(ch, st, new, brk=True):
        nonlocal cur, pending, pend_sty
        if new or cur is None:
            cur = _Box()
            cur.brk = brk
            if pending:
                cur.pre = _seg_width(pending, pend_sty)
                cur.pre_sty = pend_sty
                cur.brk = True
            pending, pend_sty = "", None
            boxes.append(cur)
        if cur.segs and cur.segs[-1][1] is st:
            cur.segs[-1] = (cur.segs[-1][0] + ch, st)
        else:
            cur.segs.append((ch, st))
        cur.text += ch
    for ch, st in chars:
        if ch == " " or ch == "\t":
            pending += ch
            pend_sty = st
            prev_ch = ch
            continue
        if _is_cjk_char(ch) and ch != "\u3000" or ch in _NO_LINE_START and prev_ch and _is_cjk_char(prev_ch):
            if ch in _NO_LINE_START and cur is not None and not pending:
                add(ch, st, False)
            elif prev_ch in _NO_LINE_END and cur is not None and not pending:
                add(ch, st, False)
            else:
                add(ch, st, True)
        else:
            latin_cont = cur is not None and not pending and cur.text and not (_is_cjk_char(cur.text[-1]) and
                                                                               cur.text[-1] not in _NO_LINE_END)
            if latin_cont and prev_ch in "-/–—" and len(cur.text) > 2 and ch.isalnum():
                add(ch, st, True)
            elif latin_cont or (cur is not None and not pending and (ch in _NO_LINE_START or prev_ch in _NO_LINE_END)):
                add(ch, st, False)
            else:
                add(ch, st, True)
        prev_ch = ch
    for b in boxes:
        b.w = sum(_seg_width(t, s) for t, s in b.segs)
    for i in range(1, len(boxes)):
        prev = boxes[i - 1].text
        if prev and prev[-1] in "-/" and boxes[i].pre == 0:
            boxes[i].pen = 18.0                      # inside a hyphenated word / path: allowed, not preferred
        elif prev and prev[-1] in _BREAK_AFTER_PUNCT:
            boxes[i].pen = -12.0
        elif prev.lower() in _SHORT_WORDS:
            boxes[i].pen = 30.0
    seps = {i for i, b in enumerate(boxes) if b.text in _LIST_SEPS and b.pre > 0 and i + 1 < len(boxes)
            and boxes[i + 1].pre > 0}
    if seps:                                     # a list ("A · B · C"): break after a separator, never inside an item
        for i in range(1, len(boxes)):
            boxes[i].pen = -20.0 if i - 1 in seps else (_NO_BREAK_PEN if i in seps else _IN_ITEM_PEN)
    return boxes


_LIST_SEPS = {"·", "•", "|", "∙", "⋅", "/", "//", "✦", "◆", "★"}
_IN_ITEM_PEN = 900.0
_NO_BREAK_PEN = 5000.0


def _list_lines(boxes, widths):
    """Fewest lines for a separator list when every item fits on a line (None when one item is too wide)."""
    if not any(b.pen == -20.0 for b in boxes[1:]):
        return None
    items, start = [], 0
    for j in range(1, len(boxes) + 1):
        if j == len(boxes) or boxes[j].pen == -20.0:
            items.append((start, j))
            start = j
    n, k, w = 1, 0, None
    for i, j in items:
        iw = boxes[i].w + sum(boxes[t].pre + boxes[t].w for t in range(i + 1, j))
        Wk = widths(k)
        if Wk is None:
            return None
        if iw > Wk + 1e-6:
            return None
        if w is None:
            w = iw
        elif w + boxes[i].pre + iw <= Wk + 1e-6:
            w += boxes[i].pre + iw
        else:
            n, k, w = n + 1, k + 1, iw
    return n


def _wrap_greedy(boxes, widths):
    lines, i, k, n = [], 0, 0, len(boxes)
    while i < n:
        Wk = widths(k)
        w = boxes[i].w
        j = i + 1
        while j < n:
            if boxes[j].brk and Wk is not None and w + boxes[j].pre + boxes[j].w > Wk + 1e-6:
                break
            w += boxes[j].pre + boxes[j].w
            j += 1
        lines.append((i, j))
        i, k = j, k + 1
    return lines


def _wrap_dp(boxes, widths, nlines, balanced, justify_px=None):
    """Minimum-raggedness line breaking with exactly ``nlines`` lines (None if impossible).

    ``justify_px`` (the font size) switches the cost to justified setting: the badness of a line is how far its word
    gaps would have to stretch, so lines with few spaces are not stretched into rivers.
    """
    n = len(boxes)
    cum = [0.0]
    for b in boxes:
        cum.append(cum[-1] + b.pre + b.w)
    csp = [0]
    for b in boxes:
        csp.append(csp[-1] + (1 if b.pre > 0 else 0))
    INF = float("inf")
    f = [[INF] * (n + 1) for _ in range(nlines + 1)]
    back = [[-1] * (n + 1) for _ in range(nlines + 1)]
    f[0][0] = 0.0
    starts = [i for i in range(n) if i == 0 or boxes[i].brk]
    for k in range(1, nlines + 1):
        Wk = widths(k - 1)
        for j in range(1, n + 1):
            if j < n and not boxes[j].brk:
                continue
            last = j == n
            if last and k != nlines:
                continue
            for i in reversed(starts):
                if i >= j:
                    continue
                w = cum[j] - cum[i] - boxes[i].pre
                if w > Wk * 1.0005 and j - i > 1:
                    break
                if f[k - 1][i] == INF:
                    continue
                slack = max(Wk - w, 0.0) / max(Wk, 1e-6)
                if last and not balanced:
                    c = 0.0
                    if j - i == 1 and nlines > 1:
                        c += 60.0
                    elif slack > 0.85 and nlines > 1:
                        c += 25.0
                elif justify_px:
                    spaces = csp[j] - csp[i + 1]
                    if spaces <= 0:
                        c = 100.0 * slack * slack + 20.0 * (slack > 0.02)
                    else:
                        r = max(Wk - w, 0.0) / (spaces * 0.3 * justify_px)
                        c = 25.0 * r * r
                else:
                    c = 100.0 * slack * slack
                if j < n:
                    c += boxes[j].pen
                v = f[k - 1][i] + c
                if v < f[k][j]:
                    f[k][j] = v
                    back[k][j] = i
    if f[nlines][n] == INF:
        return None
    out, j = [], n
    for k in range(nlines, 0, -1):
        i = back[k][j]
        out.append((i, j))
        j = i
    return out[::-1]


def _line_width(boxes, i, j):
    if j <= i:
        return 0.0
    w = boxes[i].w + sum(boxes[t].pre + boxes[t].w for t in range(i + 1, j))
    last = boxes[j - 1].segs[-1][1] if boxes[j - 1].segs else None
    return w - (last.track if last is not None else 0.0)


_LIST_RE = re.compile(r"^(\s*)([-*•–]\s+|\d{1,3}[.)]\s+|[a-zA-Z][.)]\s+(?=\S))")


def _layout_h(spans, ctx, max_width, line_spacing, align, list_style=None, drop_cap=None, tabs=None,
              balance=False, para_spacing=0.0, glyph_key=None, indent=0.0):
    """Horizontal layout in block-local coordinates (top-left of the block = (0, 0))."""
    chars = _char_pieces(spans, ctx)
    paras, cur = [], []
    for ch, st in chars:
        if ch == "\n":
            paras.append(cur)
            cur = []
        else:
            cur.append((ch, st))
    paras.append(cur)
    px = ctx.size
    base_sty = ctx.sty({})
    asc, desc, cap, xh = _metrics(base_sty.font, px, base_sty.weight)
    lh = px * line_spacing
    block = _Block()
    block.px = px
    lines_out = []                     # (para idx, boxes, i, j, x_indent, is_last_of_para, width_limit, marker)
    for pi, pchars in enumerate(paras):
        marker = None
        hang = 0.0
        text_line = "".join(c for c, _ in pchars)
        if list_style is not False:
            m = _LIST_RE.match(text_line)
            if m and (list_style or m.group(2).strip() in ("-", "•", "–") or m.group(2)[0].isdigit()):
                if not (m.group(2).strip() == "*" and not list_style):
                    mk = m.group(2).strip()
                    if mk in ("-", "*", "•", "–"):
                        mk = {"dash": "–", "check": "✓", "arrow": "→"}.get(list_style, "•")
                    marker = (mk, pchars[0][1] if pchars else base_sty)
                    pchars = pchars[m.end():]
        if list_style and marker is None and pchars and list_style not in (True,):
            mk = {"bullet": "•", "dash": "–", "check": "✓", "arrow": "→"}.get(list_style)
            if list_style == "number":
                mk = f"{pi + 1}."
            if mk:
                marker = (mk, pchars[0][1])
        if marker is not None:
            hang = max(_seg_width("10." if marker[0][:1].isdigit() else marker[0], marker[1]) + px * 0.45, px * 1.1)
        dc = None
        if drop_cap and pi == 0 and pchars and pchars[0][0].strip():
            dcl = drop_cap if isinstance(drop_cap, int) else int(drop_cap.get("lines", 3))
            dcl = max(2, dcl)
            dch, dst = pchars[0]
            if dch in "“\"'‘" and len(pchars) > 1:
                dch = pchars[0][0] + pchars[1][0]
                pchars = pchars[2:]
            else:
                pchars = pchars[1:]
            d_opts = drop_cap if isinstance(drop_cap, dict) else {}
            dsty_d = {"size": 1.0}
            if d_opts.get("font"):
                dsty_d["font"] = d_opts["font"]
            if d_opts.get("color"):
                dsty_d["color"] = d_opts["color"]
            if d_opts.get("bold") is not None:
                dsty_d["bold"] = d_opts["bold"]
            s1 = ctx.sty(dsty_d)
            capr = _metrics(s1.font, px, s1.weight)[2] / px
            target = (dcl - 1) * lh + cap
            dsty_d["size"] = target / max(capr, 0.3) / px
            dsty = ctx.sty(dsty_d)
            dw = _seg_width(dch, dsty)
            dlsb = _ink(dsty.font, dsty.px, dsty.weight, dch[0])[0]
            dc = (dch, dsty, dcl, dw - dlsb + px * 0.25, dlsb)
        if "\t" in "".join(c for c, _ in pchars):
            lines_out.append((pi, ("tabs", pchars), 0, 0, hang, True, None, marker, dc))
            continue
        boxes = _tokenize(pchars)
        if not boxes:
            lines_out.append((pi, [], 0, 0, 0.0, True, None, marker, dc))
            continue

        def widths(k, hang=hang, dc=dc):
            if max_width is None:
                return None
            ind = hang + indent
            if dc is not None and k < dc[2]:
                ind += dc[3]
            return max(max_width - ind, px)
        lines = _wrap_greedy(boxes, widths)
        nl = _list_lines(boxes, widths) if max_width is not None and len(lines) > 1 else None
        if nl is not None and len(boxes) <= 160:
            dp = _wrap_dp(boxes, widths, max(nl, len(lines)), True)
            if dp:                               # a separator at the end of a line is dropped (as on a poster)
                lines = [(i, j - 1) if j < len(boxes) and j - 1 > i and boxes[j - 1].text in _LIST_SEPS else (i, j)
                         for i, j in dp]
        elif max_width is not None and len(boxes) <= 160 and (len(lines) > 1 or balance):
            if balance and len(lines) >= 1:
                dp = _wrap_dp(boxes, widths, len(lines), True)
            else:
                dp = _wrap_dp(boxes, widths, len(lines), False, px if align == "justify" else None)
            if dp:
                lines = dp
        for li, (i, j) in enumerate(lines):
            ind = hang + indent + (dc[3] if dc is not None and li < dc[2] else 0.0)
            lines_out.append((pi, boxes, i, j, ind, li == len(lines) - 1, widths(li), marker if li == 0 else None,
                              dc if li == 0 else None))
    # ---- vertical metrics
    y_base = []
    y = 0.0
    prev_para = None
    for n_, item in enumerate(lines_out):
        pi, boxes, i, j = item[0], item[1], item[2], item[3]
        ratio = 1.0
        if isinstance(boxes, list) and j > i:
            ratio = max([s.px for b in boxes[i:j] for _, s in b.segs] + [px]) / px
        if n_ == 0:
            y = asc * ratio
        else:
            y += lh * max(ratio, 1.0) + (para_spacing * lh if pi != prev_para else 0.0)
        y_base.append(y)
        prev_para = pi
    total_h = (y_base[-1] if y_base else 0.0) + desc
    # ---- horizontal placement
    widths_out = []
    placed = []
    for n_, item in enumerate(lines_out):
        pi, boxes, i, j, ind, last, wl, marker, dc = item
        if isinstance(boxes, tuple):
            segs = _place_tabs(boxes[1], tabs, px)
            w = max([x + _seg_width(t, s) for t, s, x in segs] + [0.0])
            placed.append((n_, [(t, s, ind + x) for t, s, x in segs], ind + w, marker, dc, ind))
            widths_out.append(ind + w)
            continue
        if not boxes or j <= i:
            placed.append((n_, [], ind, marker, dc, ind))
            widths_out.append(ind)
            continue
        w = _line_width(boxes, i, j)
        extra = 0.0
        if align == "justify" and not last and wl is not None and j - i > 1:
            extra = max(wl - w, 0.0)
            spaces = sum(1 for t in range(i + 1, j) if boxes[t].pre > 0)
            if spaces:                                   # never open word gaps more than ~0.4 em
                extra = min(extra, spaces * px * 0.4)
        gaps = j - i - 1
        segs = []
        x = ind
        for t in range(i, j):
            if t > i:
                x += boxes[t].pre + (extra / gaps if gaps else 0.0)
                if boxes[t].pre > 0 and boxes[t - 1].segs and boxes[t].segs:
                    s_prev, s_next = boxes[t - 1].segs[-1][1], boxes[t].segs[0][1]
                    if s_prev.ul and s_next.ul:
                        segs.append((None, ("ul", s_prev), x - boxes[t].pre - (extra / gaps if gaps else 0.0), x))
                    if s_prev.hl and s_next.hl and s_prev.hl == s_next.hl:
                        segs.append((None, ("hl", s_prev), x - boxes[t].pre - (extra / gaps if gaps else 0.0), x))
            for st_text, sty in boxes[t].segs:
                segs.append((st_text, sty, x))
                x += _seg_width(st_text, sty)
        placed.append((n_, segs, None, marker, dc, ind))
        widths_out.append(x - (segs[-1][1].track if segs and segs[-1][0] is not None else 0.0))
    bw = max(widths_out) if widths_out else 0.0
    if align == "justify" and max_width is not None and len(lines_out) > 1:
        bw = max(bw, min(max_width, bw))
    # ---- emit
    gcount = [0]
    left_edge = 0.0                 # leftmost ink (hung quotes / brackets, negative side bearings) in block coordinates
    for (n_, segs, _w, marker, dc, ind), lwid in zip(placed, widths_out):
        base = y_base[n_]
        al = align if align in ("left", "center", "right") else "left"
        if align == "justify":
            al = "left"
        dx = {"left": 0.0, "center": (bw - lwid) / 2, "right": bw - lwid}[al]
        first = next((s for s in segs if s[0]), None)
        if al == "left" and first is not None and ind == 0 and not dc:
            ch0 = first[0][0]
            l_ = _ink(_first_spec(ch0, first[1]), first[1].px, first[1].weight, ch0)[0]
            if ch0 in "“\"'‘«(" and first[1].px > 0:
                dx -= _ink(_first_spec(ch0, first[1]), first[1].px, first[1].weight, ch0)[2] * 0.85
            else:
                dx -= max(min(l_, first[1].px * 0.08), -first[1].px * 0.05)
        if first is not None and not dc:
            ch1 = first[0][0]
            left_edge = min(left_edge, dx + first[2] +
                            _ink(_first_spec(ch1, first[1]), first[1].px, first[1].weight, ch1)[0])
        if al == "right":
            lastseg = next((s for s in reversed(segs) if s[0]), None)
            if lastseg is not None:
                chn = lastseg[0][-1]
                sp = _first_spec(chn, lastseg[1])
                ink = _ink(sp, lastseg[1].px, lastseg[1].weight, chn)
                adv = _shape(chn, sp, round(lastseg[1].px, 3), lastseg[1].weight, 0.0)[1]
                if chn in ".,”’»)-" :
                    dx += (adv - ink[0]) * 0.7
                else:
                    dx += max(min(adv - ink[2], lastseg[1].px * 0.08), -lastseg[1].px * 0.05)
        if marker is not None:
            mk, msty = marker
            mw = _seg_width(mk, msty)
            mx = dx + indent
            if mk[:1].isdigit():
                hang = max(_seg_width("10.", msty) + px * 0.45, px * 1.1)
                mx = dx + indent + hang - px * 0.45 - mw
            _emit(block, mk, msty, mx, base, None)
        if dc is not None:
            dch, dsty, dcl, dwid, dlsb = dc
            _emit(block, dch, dsty, dx + ind - dwid - dlsb, y_base[min(n_ + dcl - 1, len(y_base) - 1)], None)
        for seg in segs:
            if seg[0] is None:
                kind, sty = seg[1]
                if kind == "ul":
                    t_ = max(sty.px * 0.06, 0.8)
                    block.decor.append((dx + seg[2], base + sty.px * 0.11, dx + seg[3], base + sty.px * 0.11 + t_, None,
                                        (0.0, 0.0), sty.color))
                else:
                    block.hl.append((dx + seg[2], base - sty.px * 0.74, dx + seg[3], base + sty.px * 0.2, None, (0.0, 0.0),
                                     sty.hl))
                continue
            text_, sty, x = seg
            gk = None
            if glyph_key is not None:
                x2 = dx + x
                for chunk_x, chunk in _per_char(text_, sty, x2):
                    k = glyph_key(gcount[0]) if chunk.strip() else None
                    if chunk.strip():
                        gcount[0] += 1
                    _emit(block, chunk, sty, chunk_x, base, k)
            else:
                _emit(block, text_, sty, dx + x, base, gk)
            w_ = _seg_width(text_, sty)
            if sty.ul:
                t_ = max(sty.px * 0.06, 0.8)
                block.decor.append((dx + x, base + sty.px * 0.11, dx + x + w_ - sty.track, base + sty.px * 0.11 + t_, None,
                                    (0.0, 0.0), sty.color))
            if sty.st:
                t_ = max(sty.px * 0.055, 0.8)
                ym = base - _metrics(sty.font, sty.px, sty.weight)[3] * 0.55
                block.decor.append((dx + x - sty.px * 0.03, ym - t_ / 2, dx + x + w_ - sty.track + sty.px * 0.03, ym + t_ / 2,
                                    None, (0.0, 0.0), sty.color))
            if sty.hl:
                block.hl.append((dx + x - sty.px * 0.12, base - sty.px * 0.74, dx + x + w_ - sty.track + sty.px * 0.12,
                                 base + sty.px * 0.2, None, (0.0, 0.0), sty.hl))
        block.lines.append((base, cap, dx, dx + lwid))
    block.bbox = (min(left_edge - 0.5, 0.0) if left_edge < -0.5 else 0.0, 0.0, bw, total_h)
    return block


def _first_spec(ch, sty):
    for t, spec in _split_runs(ch, sty.chain):
        return spec[1] if isinstance(spec, tuple) else spec
    return sty.font


def _per_char(text, sty, x0):
    """[(x, char)] positions for every character of a segment."""
    out = []
    x = x0
    for t, spec in _split_runs(text, sty.chain):
        spec = spec[1] if isinstance(spec, tuple) else spec
        xs, adv = _shape(t, spec, round(sty.px, 3), sty.weight, round(sty.track, 4), bool(sty.tabular))
        for c, xx in zip(t, xs):
            out.append((x + xx, c))
        x += adv
    return out


def _place_tabs(pchars, tabs, px):
    """Tab-separated cells → [(text, sty, x)]; ``tabs`` = stop positions (px from the block left), each a number
    (left-aligned) or (x, "r") right-aligned / (x, ".") decimal-aligned. Default: a stop every 6 em."""
    cells, cur = [], []
    for ch, st in pchars:
        if ch == "\t":
            cells.append(cur)
            cur = []
        else:
            cur.append((ch, st))
    cells.append(cur)
    stops = list(tabs or [])
    out = []
    x = 0.0
    for ci, cell in enumerate(cells):
        segs = []
        for ch, st in cell:
            if segs and segs[-1][1] is st:
                segs[-1] = (segs[-1][0] + ch, st)
            else:
                segs.append((ch, st))
        w = sum(_seg_width(t, s) for t, s in segs)
        if ci > 0:
            if ci - 1 < len(stops):
                stop = stops[ci - 1]
            else:
                stop = (math.floor(x / (6 * px)) + 1) * 6 * px
            mode = "l"
            if isinstance(stop, (tuple, list)):
                stop, mode = float(stop[0]), str(stop[1])[:1].lower()
            stop = float(stop)
            if mode == "r":
                x = stop - w
            elif mode in (".", "d"):
                txt = "".join(t for t, _ in segs)
                k = txt.find(".")
                pre = txt if k < 0 else txt[:k]
                wpre = _seg_width(pre, segs[0][1]) if segs and pre else 0.0
                x = stop - wpre
            elif mode == "c":
                x = stop - w / 2
            else:
                x = max(stop, x)
        for t, s in segs:
            out.append((t, s, x))
            x += _seg_width(t, s)
    return out


def _emit(block, text, sty, x, base, key):
    """Turn a styled segment into glyph placements (fallback runs, kerning, tracking, synthetic styles)."""
    if key is None:
        key = sty.color if not sty.off else ("off",)
    elif sty.off:
        key = ("off",)
    M = (1.0, -0.21, 0.0, 1.0) if sty.synth_i else None
    for t, spec in _split_runs(text, sty.chain):
        if isinstance(spec, tuple):                 # nothing has this glyph
            block.missing.extend(c for c in t if not c.isspace())
            spec = spec[1]
        block.fonts.add(spec)
        xs, adv = _shape(t, spec, round(sty.px, 3), sty.weight, round(sty.track, 4), bool(sty.tabular))
        synth = _load2(spec, round(sty.px * _MS), sty.weight)[1]
        # group characters into sub-runs that Pillow can draw in one call (no kerning / tracking inside)
        start = 0
        for idx in range(1, len(t) + 1):
            split = idx == len(t)
            if not split:
                natural = xs[idx - 1] + _shape(t[idx - 1], spec, round(sty.px, 3), sty.weight, 0.0)[1]
                split = abs(xs[idx] - natural) > 0.02 * max(sty.px, 1) / 8 or sty.tabular
            if split:
                sub = t[start:idx]
                if sub.strip():
                    block.pl.append((sub, spec, sty.px, sty.weight, synth, M, x + xs[start], base, key))
                start = idx
        x += adv


# --------------------------------------------------------------------------------------------- vertical

_V_ROTATE = set("（）「」『』【】《》〈〉〔〕［］｛｝()[]{}<>—―…‥～〜ーｰ-_=＝|｜:;：；→←")
_V_PUNCT = set("、。，．,.")
_V_SMALL = set("ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ")
_V_OPEN = set("（「『【《〈〔［｛")
_V_CLOSE = set("）」』】》〉〕］｝")
_V_HALF_AFTER = set("、。，．") | _V_CLOSE         # their ink sits in the first half of the cell
_V_HALF_BEFORE = set("、。，．") | _V_OPEN | _V_CLOSE


def _layout_v(spans, ctx, max_height, line_spacing):
    """Vertical CJK: top-to-bottom columns, right-to-left. Block-local coordinates."""
    chars = _char_pieces(spans, ctx)
    px = ctx.size
    colw = px * max(line_spacing, 1.0)
    cols, cur, y = [], [], 0.0
    for k, (ch, st) in enumerate(chars):
        if ch == "\n":
            cols.append(cur)
            cur, y = [], 0.0
            continue
        rot = ch in _V_ROTATE or (ord(ch) < 0x2e80 and not ch.isspace() and ch not in _V_PUNCT)
        if rot:
            sp = _first_spec(ch, st)
            adv = _shape(ch, sp, round(st.px, 3), st.weight, 0.0)[1] + st.track
        else:
            adv = st.px + st.track
        cell = adv
        nxt = chars[k + 1][0] if k + 1 < len(chars) else ""
        if ch in _V_HALF_AFTER and nxt in _V_HALF_BEFORE and adv > st.px * 0.8:
            adv = st.px * 0.5 + st.track            # consecutive punctuation: close the double blank (、「 」。)
        if max_height is not None and cur and y + adv > max_height + 1e-6 and ch not in _NO_LINE_START:
            carry = []
            while cur and cur[-1][0] in _NO_LINE_END and len(cur) > 1:
                carry.insert(0, cur.pop())
            cols.append(cur)
            cur, y = [], 0.0
            for c_ in carry:
                cur.append((c_[0], c_[1], y, c_[3], c_[4], c_[5]))
                y += c_[3]
        cur.append((ch, st, y, adv, rot, cell))
        y += adv
    cols.append(cur)
    block = _Block()
    block.px = px
    ncol = len(cols)
    bw = colw * (ncol - 1) + px
    bh = max([c[-1][2] + c[-1][3] - c[-1][1].track for c in cols if c] + [0.0])
    for ci, col in enumerate(cols):
        cx = bw - px / 2 - ci * colw
        for ch, st, y0, adv, rot, cell in col:
            if ch.isspace():
                continue
            sp = _first_spec(ch, st)
            if isinstance(sp, tuple):
                sp = sp[1]
            block.fonts.add(sp)
            if _font_for(ord(ch), st.chain) is None:
                block.missing.append(ch)
            gadv = _shape(ch, sp, round(st.px, 3), st.weight, 0.0)[1]
            ref = _ink(sp, st.px, st.weight, "中" if _covers(sp, ord("中")) else "H")
            em_c = (ref[1] + ref[3]) / 2              # em-box centre relative to the baseline
            synth = _load2(sp, round(st.px * _MS), st.weight)[1]
            key = st.color
            if rot:
                # rotate 90° clockwise about the glyph centre (advance / 2, em centre)
                M = (0.0, -1.0, 1.0, 0.0)
                gc = (gadv / 2, em_c)
                C = (cx, y0 + cell / 2)
                tx = C[0] - (M[0] * gc[0] + M[1] * gc[1])
                ty = C[1] - (M[2] * gc[0] + M[3] * gc[1])
                block.pl.append((ch, sp, st.px, st.weight, synth, M, tx, ty, key))
            else:
                tx = cx - gadv / 2
                ty = y0 + st.px / 2 - em_c
                if ch in _V_PUNCT:
                    # place the ink, not the glyph box: centred in its cell for Traditional Chinese, in the
                    # top-right corner for Japanese / Simplified Chinese / Korean
                    il, it, ir, ib = _ink(sp, st.px, st.weight, ch)
                    if getattr(ctx, "lang", "tc") == "tc":
                        gx, gy = cx, y0 + adv / 2
                    else:
                        half = max(ir - il, ib - it) / 2
                        gx = cx + max(st.px * 0.5 - half - st.px * 0.12, 0.0)
                        gy = y0 + min(half + st.px * 0.12, st.px * 0.5)
                    tx = gx - (il + ir) / 2
                    ty = gy - (it + ib) / 2
                elif ch in _V_SMALL:
                    tx += st.px * 0.08
                    ty -= st.px * 0.08
                block.pl.append((ch, sp, st.px, st.weight, synth, None, tx, ty, key))
    block.bbox = (0.0, 0.0, bw, bh)
    return block


# --------------------------------------------------------------------------------------------- public measuring

def _auto_tracking(plain, size, H):
    if H is None:
        return 0.0
    letters = [c for c in plain if c.isalpha()]
    latin = letters and all(ord(c) < 0x2e80 for c in letters)
    rel = size / max(H, 1)
    if latin and len(letters) >= 3 and all(c.isupper() for c in letters) and rel < 0.045:
        return 0.06
    if latin and rel >= 0.09:
        return -0.012
    return 0.0


def _auto_leading(plain, size, H):
    rel = size / max(H or 1080, 1)
    cjk = has_cjk(plain)
    if rel >= 0.09:
        return 1.12 if cjk else 1.02
    if rel >= 0.05:
        return 1.25 if cjk else 1.12
    return 1.5 if cjk else 1.3


def _plain(spans):
    return "".join(t for t, _ in spans)


def _make_block(text, xy, size, font, bold, anchor, align, max_width, line_spacing, tracking, *, italic=False,
                markup=None, case=None, underline=False, strike=False, highlight=None, tabular=False, lang=None,
                list_style=None, drop_cap=None, tabs=None, balance=False, para_spacing=0.0, vertical=False,
                max_height=None, glyph_key=None, rotate=0.0, indent=0.0):
    spans = _span_list(text, markup)
    plain = _plain(spans)
    if isinstance(font, (ImageFont.FreeTypeFont,)):
        font = getattr(font, "path", None)
    ctx = _Ctx(size, font, bold, italic, tracking, case, underline, strike, highlight, tabular, lang, plain)
    if vertical:
        block = _layout_v(spans, ctx, max_height, line_spacing)
    else:
        block = _layout_h(spans, ctx, max_width, line_spacing, align, list_style, drop_cap, tabs, balance, para_spacing,
                          glyph_key, indent)
    bw, bh = block.bbox[2], block.bbox[3]
    _check_anchor(anchor)
    ha, va = (anchor + "a")[0], (anchor + "a")[1]
    x, y = float(xy[0]), float(xy[1])
    asc = block.lines[0][0] if block.lines else (_metrics(ctx.sty({}).font, size)[0])
    x0 = x - {"l": 0.0, "m": bw / 2, "r": bw}.get(ha, 0.0)
    y0 = y - {"t": 0.0, "a": 0.0, "m": bh / 2, "b": bh, "d": bh, "s": asc}.get(va, 0.0)
    if vertical and va == "s":
        y0 = y
    block.translate(x0, y0)
    if align in ("center", "right") or (align is None and ha in "mr"):
        pass
    if rotate:
        block.rotate(rotate, (x, y))
    block.plain = plain
    return block


_ANCHORS = tuple(h + v for h in "lmr" for v in "atmbsd") + tuple("lmr")
_ALIGNS = ("left", "center", "right", "justify")


def _check_anchor(anchor):
    if anchor not in _ANCHORS:
        _check_choice("text", "anchor", anchor, _ANCHORS)
    return anchor


def _resolve_align(align, anchor):
    if align:
        if align not in _ALIGNS:
            _check_choice("text", "align", align, _ALIGNS)
        return align
    return {"l": "left", "m": "center", "r": "right"}.get((anchor or "la")[0], "left")


def text_bbox(text, xy=(0, 0), size=40, font=None, bold=False, anchor="la", align=None, max_width=None,
              line_spacing=1.2, tracking=0.0, **kw):
    """Bounding box (x0, y0, x1, y1) the text would occupy — measure before drawing.
    Accepts the layout keywords of ``draw_text`` (``italic``, ``vertical``, ``rotate``, ``markup``, ``case`` ...)."""
    kw = {k: v for k, v in kw.items() if k in _LAYOUT_KW}
    b = _make_block(text, xy, size, font, bold, anchor, _resolve_align(align, anchor), max_width,
                    1.2 if line_spacing is None else line_spacing, tracking or 0.0, **kw)
    return b.bbox


_LAYOUT_KW = {"italic", "markup", "case", "underline", "strike", "highlight", "tabular", "lang", "list_style",
              "drop_cap", "tabs", "balance", "para_spacing", "vertical", "max_height", "rotate", "indent"}


def fit_text_size(text, box_w, box_h=None, font=None, bold=False, max_size=None, min_size=6, wrap=False,
                  line_spacing=1.2, tracking=0.0, **kw):
    """Largest font size (px) such that the text fits inside ``box_w`` x ``box_h``.
    ``wrap=True`` lets the text wrap to ``box_w`` (paragraphs); ``balance=True`` evens out the lines."""
    hi = int(max_size or (box_h if box_h else box_w))
    lo = int(min_size)
    best = lo
    kw = {k: v for k, v in kw.items() if k in _LAYOUT_KW and k != "rotate"}
    vertical = kw.get("vertical")
    while lo <= hi:
        mid = (lo + hi) // 2
        if vertical:
            kw2 = dict(kw, max_height=box_h if wrap else None)
            x0, y0, x1, y1 = _make_block(text, (0, 0), mid, font, bold, "la", "left", None, line_spacing,
                                         tracking or 0.0, **kw2).bbox
        else:
            x0, y0, x1, y1 = _make_block(text, (0, 0), mid, font, bold, "la", "left", box_w if wrap else None,
                                         line_spacing, tracking or 0.0, **kw).bbox
        if (x1 - x0) <= box_w + 0.01 and (box_h is None or (y1 - y0) <= box_h + 0.01):
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def wrap_text(text, font, max_width, tracking_px=0.0):
    """Greedy line-wrapping with a Pillow font. Latin wraps at spaces; CJK can break between any two characters.
    Explicit ``\\n`` always breaks."""
    out = []

    def lw(line):
        if not line:
            return 0.0
        return float(font.getlength(line)) + tracking_px * max(len(line) - 1, 0)
    for para in str(text).split("\n"):
        if max_width is None or lw(para) <= max_width:
            out.append(para)
            continue
        tokens = re.findall(r"[\u2e80-\u9fff\uac00-\ud7af\uf900-\uffef]|[^\s\u2e80-\u9fff\uac00-\ud7af\uf900-\uffef]+|\s+", para)
        line = ""
        for tok in tokens:
            trial = line + tok
            if line and lw(trial.rstrip()) > max_width and not tok.isspace() and tok[0] not in _NO_LINE_START:
                out.append(line.rstrip())
                line = tok.lstrip()
            else:
                line = trial
        out.append(line.rstrip())
    return out


def text_mask(h, w, text, xy, size=40, font=None, bold=False, anchor="la", align=None, max_width=None,
              line_spacing=1.2, tracking=0.0, stroke=0.0, aa=2, font_path=None, **kw):
    """Anti-aliased text coverage mask, float32 (H, W).

    ``text`` may contain ``\\n``; ``max_width`` (px) enables word / CJK wrapping; ``align`` = left / center / right /
    justify; ``tracking`` = extra letter spacing as a fraction of ``size``; ``stroke`` (px) grows the glyphs
    (outline = text_mask(stroke=s) - text_mask()). Layout keywords of ``draw_text`` work too
    (``italic``, ``rotate``, ``vertical``, ``markup`` ...). ``font_path`` = ``font`` (kept for old code).
    """
    font = font or font_path
    kw = {k: v for k, v in kw.items() if k in _LAYOUT_KW}
    b = _make_block(text, xy, size, font, bold, anchor, _resolve_align(align, anchor), max_width,
                    1.2 if line_spacing is None else line_spacing, tracking or 0.0, **kw)
    return b.masks(int(h), int(w), aa, stroke)[0]


# --------------------------------------------------------------------------------------------- colour helpers

def _srgb_lum(c):
    c = np.clip(np.asarray(c, np.float64), 0, 1)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]


def contrast_ratio(a, b):
    """WCAG contrast ratio between two colours (1 … 21). Body text wants ≥ 4.5, large text ≥ 3."""
    la, lb = float(_srgb_lum(rgb(a))), float(_srgb_lum(rgb(b)))
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _shift(a, dx, dy):
    """Shift a 2-D (or 3-D) array by integer pixels, filling with zeros (no wrap-around)."""
    dx, dy = int(round(dx)), int(round(dy))
    out = np.zeros_like(a)
    H, W = a.shape[:2]
    if abs(dx) >= W or abs(dy) >= H:
        return out
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[yd, xd] = a[ys, xs]
    return out


def _blur(a, s):
    from .filters import gaussian_blur
    return gaussian_blur(a, max(float(s), 0.0)) if s > 0.05 else a


def _crop_box(H, W, bbox, pad):
    x0, y0, x1, y1 = bbox
    return (max(int(x0 - pad), 0), max(int(y0 - pad), 0), min(int(math.ceil(x1 + pad)), W), min(int(math.ceil(y1 + pad)), H))


def _bg_under(cv, mask, bbox, px):
    """Mean colour and luminance stats of the canvas around / under the text."""
    H, W = cv.h, cv.w
    x0, y0, x1, y1 = _crop_box(H, W, bbox, px * 0.25)
    if x1 <= x0 or y1 <= y0:
        return np.array([0.5, 0.5, 0.5], F32), 0.2, 0.0
    sub = clamp01(cv.buf[y0:y1, x0:x1])
    m = mask[y0:y1, x0:x1]
    reg = _blur(m, max(px * 0.15, 1.0)) + 0.02
    wsum = float(reg.sum())
    col = (sub * reg[..., None]).reshape(-1, 3).sum(0) / max(wsum, 1e-6)
    lum = _srgb_lum(sub)
    mu = float((lum * reg).sum() / wsum)
    sd = float(np.sqrt(max(((lum - mu) ** 2 * reg).sum() / wsum, 0.0)))
    return col.astype(F32), mu, sd


# --------------------------------------------------------------------------------------------- effects

_GOLD = [(0.0, "#fff3c4"), (0.22, "#f6d36b"), (0.48, "#b98222"), (0.52, "#d9a53c"), (0.75, "#ffe7a0"), (1.0, "#8c5a12")]
_SILVER = [(0.0, "#ffffff"), (0.25, "#dfe5ea"), (0.48, "#7d8892"), (0.53, "#b9c3cc"), (0.78, "#f2f6f9"), (1.0, "#6e7882")]
_ROSE = [(0.0, "#fff0ea"), (0.25, "#f2b9a0"), (0.5, "#a85f4a"), (0.55, "#d48d73"), (0.8, "#ffd9c8"), (1.0, "#7a3e2e")]
_CHROME = [(0.0, "#e9f6ff"), (0.45, "#6fa3c9"), (0.5, "#1b2733"), (0.56, "#c9a27a"), (0.8, "#fff4e0"), (1.0, "#8a6a4a")]

EFFECTS = ("long_shadow", "extrude", "emboss", "letterpress", "gold", "silver", "metal", "chrome", "rose_gold", "foil",
           "neon", "sticker", "knockout", "outline_only", "stripes", "glitch", "chalk", "stencil")


_CASES = ("upper", "lower", "title", "smallcaps")
_LIST_STYLES = ("bullet", "number", "dash", "check", "arrow")
_PLATES = ("pill", "box", "scrim", "blur", "glass")
_GRADIENT_DIRS = ("x", "y", "line", "lines")


def _check_style_options(shadow, glow, plate, case, list_style, gradient_dir, box_anchor):
    """Validate enumerated values and the keys of the ``shadow`` / ``glow`` / ``plate`` dicts of ``draw_text``;
    returns them with aliased key spellings normalised."""
    if case:
        _check_choice("draw_text", "case", case, _CASES)
    if isinstance(list_style, str):
        _check_choice("draw_text", "list_style", list_style, _LIST_STYLES)
    if isinstance(gradient_dir, str):
        _check_choice("draw_text", "gradient_dir", gradient_dir, _GRADIENT_DIRS)
    if isinstance(box_anchor, str) and (len(box_anchor) != 2 or box_anchor[0] not in "lmr" or box_anchor[1] not in "tmb"):
        _check_choice("draw_text", "box_anchor", box_anchor, tuple(h + v for h in "lmr" for v in "tmb"))
    if isinstance(shadow, dict):
        shadow = _reject_unknown("draw_text shadow", shadow, ("dx", "dy", "blur", "color", "opacity"))
    if isinstance(glow, dict):
        glow = _reject_unknown("draw_text glow", glow, ("radius", "color", "strength"))
    if isinstance(plate, str):
        _check_choice("draw_text", "plate", plate, _PLATES)
    elif isinstance(plate, dict):
        plate = _reject_unknown("draw_text plate", plate, ("kind", "pad", "pad_x", "radius", "color", "opacity", "soft",
                                                           "blur", "shadow", "border", "border_width"))
        if "kind" in plate:
            _check_choice("draw_text", "plate kind", plate["kind"], _PLATES)
    return shadow, glow, plate


def _norm_effects(effect):
    if not effect:
        return []
    if isinstance(effect, (str, dict)):
        effect = [effect]
    out = []
    for e in effect:
        if isinstance(e, str):
            out.append({"kind": e.lower().replace("-", "_").replace(" ", "_")})
        else:
            e = dict(e)
            e["kind"] = str(e.get("kind") or e.get("name")).lower().replace("-", "_").replace(" ", "_")
            out.append(e)
    alias = {"metallic": "gold", "gold_foil": "foil", "hollow": "outline_only", "outline": "outline_only",
             "3d": "extrude", "deboss": "letterpress", "die_cut": "sticker", "cutout": "knockout",
             "knock_out": "knockout", "retro": "stripes", "rgb_split": "glitch", "rosegold": "rose_gold",
             "longshadow": "long_shadow"}
    for e in out:
        e["kind"] = alias.get(e["kind"], e["kind"])
        if e["kind"] not in EFFECTS:
            raise _unknown_name("text effect", e["kind"], EFFECTS)
    return out


class _FX:
    """State shared by the effect painters for one draw."""

    def __init__(self, cv, block, fill, groups, px, aa, seed):
        self.cv, self.block, self.fill, self.groups, self.px, self.aa, self.seed = cv, block, fill, groups, px, aa, seed
        self.H, self.W = cv.h, cv.w
        self._ly = None

    def stroke(self, r):
        return self.block.masks(self.H, self.W, self.aa, r)[0]

    def local_y(self):
        """Block-local y (unrotated) for every pixel."""
        if self._ly is None:
            Y = np.arange(self.H, dtype=F32)[:, None] * np.ones((1, self.W), F32)
            if self.block.rot:
                deg, (px_, py_) = self.block.rot
                a = math.radians(deg)
                X = np.arange(self.W, dtype=F32)[None, :] * np.ones((self.H, 1), F32)
                Y = py_ + math.sin(a) * (X - px_) + math.cos(a) * (Y - py_)
            self._ly = Y.astype(F32)
        return self._ly

    def band_t(self):
        """0 at the cap height → 1 at the baseline of each line (for metal / stripes)."""
        Y = self.local_y()
        lines = self.block.lines
        if not lines:
            x0, y0, x1, y1 = self.block.bbox
            return np.clip((Y - y0) / max(y1 - y0, 1), 0, 1)
        t = np.zeros_like(Y)
        bounds = [-1e9] + [(lines[i][0] + lines[i + 1][0] - lines[i + 1][1]) / 2 for i in range(len(lines) - 1)] + [1e9]
        for i, (base, cap, _, _) in enumerate(lines):
            sel = (Y >= bounds[i]) & (Y < bounds[i + 1])
            t = np.where(sel, (Y - (base - cap)) / max(cap, 1.0), t)
        return t.astype(F32)

    def shade(self, sigma, light=(-0.6, -0.8)):
        """Bevel shading from the mask's blurred gradient: > 0 lit edges, < 0 shadowed edges."""
        h = _blur(self.fill, sigma)
        gy, gx = np.gradient(h)
        if self.block.rot:
            a = math.radians(self.block.rot[0])
            lx = light[0] * math.cos(a) + light[1] * math.sin(a)
            ly = -light[0] * math.sin(a) + light[1] * math.cos(a)
            light = (lx, ly)
        s = -(gx * light[0] + gy * light[1]) * sigma * 2.2
        return np.clip(s, -1, 1).astype(F32)


def _metal_map(fx, stops, streak=True):
    t = fx.band_t()
    col = mix(stops, np.clip(t * 0.9 + 0.05, 0, 1))
    sh = fx.shade(max(fx.px * 0.035, 1.0))
    col = col * (1.0 + 0.45 * sh)[..., None]
    if streak:
        x0, y0, x1, y1 = fx.block.bbox
        X = np.arange(fx.W, dtype=F32)[None, :]
        Y = np.arange(fx.H, dtype=F32)[:, None]
        d = (X - (x0 + 0.28 * (x1 - x0))) + (Y - y0) * 0.7
        w = max((x1 - x0) * 0.05, fx.px * 0.3)
        col = col + (np.exp(-(d / w) ** 2) * 0.55)[..., None] + (np.exp(-((d - 2.6 * w) / (0.4 * w)) ** 2) * 0.3)[..., None]
    return col.astype(F32)


def _fx_long_shadow(fx, e, base_col):
    ang = math.radians(float(e.get("angle", 45)))
    L = float(e.get("length", fx.px * 1.2))
    col = e.get("color", None)
    if col is None:
        bg, _, _ = _bg_under(fx.cv, fx.fill, fx.block.bbox, fx.px)
        col = bg * 0.55
    op = float(e.get("opacity", 0.9))
    ux, uy = math.cos(ang), math.sin(ang)
    x0, y0, x1, y1 = _crop_box(fx.H, fx.W, fx.block.bbox, L + fx.px)
    M = fx.fill[y0:y1, x0:x1]
    acc = np.zeros_like(M)
    dist = np.zeros_like(M)
    steps = int(max(min(L, 160), 1))
    for k in range(1, steps + 1):
        d = L * k / steps
        sm = _shift(M, ux * d, uy * d)
        new = np.clip(sm - acc, 0, 1)
        dist += new * (d / L)
        acc = np.maximum(acc, sm)
    fade = 1.0 - dist / np.maximum(acc, 1e-4) if e.get("fade", True) else 1.0
    a = acc * np.clip(fade, 0, 1) ** 0.8 * op
    full = np.zeros((fx.H, fx.W), F32)
    full[y0:y1, x0:x1] = a
    fx.cv.paint(col, full)


def _fx_extrude(fx, e, base_col):
    ang = math.radians(float(e.get("angle", 60)))
    depth = float(e.get("depth", fx.px * 0.14))
    side = e.get("color")
    if side is None:
        bc = np.asarray(base_col, F32)
        side = bc.reshape(-1, 3).mean(0) * 0.5 if bc.ndim > 1 else bc * 0.5
    side = rgb(side)
    ux, uy = math.cos(ang), math.sin(ang)
    x0, y0, x1, y1 = _crop_box(fx.H, fx.W, fx.block.bbox, depth + fx.px)
    M = fx.fill[y0:y1, x0:x1]
    sh = fx.shade(max(fx.px * 0.02, 1.0), light=(-0.3, -0.95))[y0:y1, x0:x1]
    acc = np.zeros_like(M)
    shade = np.zeros_like(M)
    dist = np.zeros_like(M)
    steps = int(max(depth, 1))
    for k in range(1, steps + 1):
        d = depth * k / steps
        sm = _shift(M, ux * d, uy * d)
        new = np.clip(sm - acc, 0, 1)
        shade += new * _shift(sh, ux * d, uy * d)
        dist += new * (d / depth)
        acc = np.maximum(acc, sm)
    nrm = np.maximum(acc, 1e-4)
    t = dist / nrm
    s = shade / nrm
    col = side[None, None, :] * (1.05 - 0.45 * t)[..., None] * (1.0 + 0.5 * s)[..., None]
    full_a = np.zeros((fx.H, fx.W), F32)
    full_a[y0:y1, x0:x1] = acc
    full_c = np.zeros((fx.H, fx.W, 3), F32)
    full_c[y0:y1, x0:x1] = col
    if e.get("shadow", True):
        shd = _blur(_shift(full_a, ux * depth * 0.4, uy * depth * 0.4), depth * 0.5)
        fx.cv.paint("#000000", shd * 0.35)
    fx.cv.paint(full_c, full_a)
    edge = e.get("edge", True)
    if edge:
        ring = np.clip(fx.stroke(max(fx.px * 0.012, 0.8)) - fx.fill, 0, 1)
        fx.cv.paint(side * 0.35, ring * 0.8)


def _fx_sticker(fx, e, base_col):
    w = float(e.get("width", fx.px * 0.1))
    col = e.get("color", "#ffffff")
    raw = fx.stroke(w)
    backing = smoothstep(0.35, 0.62, _blur(raw, w * 0.45))
    backing = np.maximum(backing, raw)
    if e.get("shadow", True):
        sh = _blur(_shift(backing, fx.px * 0.03, fx.px * 0.06), fx.px * 0.06)
        fx.cv.paint("#000000", sh * 0.45)
    line = e.get("line", None)
    if line:
        ring = np.clip(_blur(backing, 0.6) * 1.0 - 0, 0, 1)
        lw = float(e.get("line_width", max(fx.px * 0.015, 1.0)))
        grown = smoothstep(0.02, 0.25, _blur(backing, lw))
        fx.cv.paint(line, np.clip(grown - backing, 0, 1) + 0 * ring)
    fx.cv.paint(col, backing)


def _plate_mask(fx_or_cv, block, kind, pad_x, pad_y, radius=None):
    cv = fx_or_cv
    H, W = cv.h, cv.w
    from . import sdf
    x0, y0, x1, y1 = block.bbox if block.rot is None else _unrotated_bbox(block)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = (x1 - x0) / 2 + pad_x, (y1 - y0) / 2 + pad_y
    if radius is None:
        radius = hh if kind == "pill" else min(hh, hw) * 0.18
    X, Y = cv.xy()
    if block.rot is not None:
        deg, (px_, py_) = block.rot
        a = math.radians(deg)
        dx, dy = X - px_, Y - py_
        X = px_ + math.cos(a) * dx - math.sin(a) * dy
        Y = py_ + math.sin(a) * dx + math.cos(a) * dy
    d = sdf.sd_box(X, Y, cx, cy, hw, hh, radius=min(radius, hh, hw))
    return d


def _unrotated_bbox(block):
    deg, (px_, py_) = block.rot
    a = math.radians(deg)
    pts = []
    for (x, y) in block.corners():
        dx, dy = x - px_, y - py_
        pts.append((px_ + math.cos(a) * dx - math.sin(a) * dy, py_ + math.sin(a) * dx + math.cos(a) * dy))
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _fx_knockout(fx, e, base_col, color):
    from . import sdf
    pad = float(e.get("pad", fx.px * 0.12))
    d = _plate_mask(fx.cv, fx.block, e.get("shape", "box"), pad * 2.0, pad, e.get("radius"))
    plate = sdf.fill(d)
    col = e.get("color", color if not isinstance(color, str) or color != "auto" else "#ffffff")
    fx.cv.paint(col, plate * np.clip(1.0 - fx.fill, 0, 1) * float(e.get("opacity", 1.0)))


def _fx_neon_glow(fx, e, color, on):
    r = float(e.get("radius", fx.px * 0.45))
    gcol = rgb(e.get("color", color if not isinstance(color, str) or color.startswith("#") else "#ff3cac"))
    if np.asarray(gcol).ndim > 1:
        gcol = np.asarray(gcol).reshape(-1, 3).mean(0)
    halo = _blur(on, r * 0.2) * 1.0 + _blur(on, r * 0.7) * 0.85 + _blur(on, r * 2.2) * 0.55
    fx.cv.paint(gcol, halo * float(e.get("strength", 1.0)), "add")
    return gcol


def _off_mask(fx):
    return fx.groups.get(("off",), np.zeros((fx.H, fx.W), F32))


def _painted_fill(fx, e_kinds, effects, col, alpha):
    """Return (colour map, alpha) after fill-modifying effects."""
    for e in effects:
        k = e["kind"]
        if k in ("gold", "silver", "metal", "chrome", "rose_gold", "foil"):
            stops = {"gold": _GOLD, "foil": _GOLD, "silver": _SILVER, "metal": _SILVER, "chrome": _CHROME,
                     "rose_gold": _ROSE}[k]
            if e.get("stops"):
                stops = e["stops"]
            col = _metal_map(fx, stops, e.get("streak", True))
            if k == "foil":
                rng = np.random.default_rng(fx.seed)
                n = rng.random((fx.H, fx.W)).astype(F32)
                n = _blur(n, max(fx.px * 0.008, 0.5))
                col = col * (0.85 + 0.35 * (n - n.mean()) / (n.std() + 1e-6) * 0.25 + 0.15)[..., None]
        elif k == "stripes":
            t = fx.band_t()
            n = int(e.get("count", 3))
            gaps = np.zeros_like(t)
            for i in range(n):
                c = 0.5 + 0.5 * (i + 0.5) / n
                th = 0.04 + 0.07 * (i + 1) / n
                gaps = np.maximum(gaps, smoothstep(th * 0.5 + 0.02, th * 0.5, np.abs(t - c)))
            alpha = alpha * (1 - gaps)
        elif k == "chalk":
            rng = np.random.default_rng(fx.seed)
            g = rng.random((fx.H, fx.W)).astype(F32)
            g = _blur(g, 0.7)
            g = (g - g.mean()) / (g.std() + 1e-6)
            s = rng.random((fx.H, fx.W)).astype(F32)
            s = _blur(s, max(fx.px * 0.05, 1.5))
            s = (s - s.mean()) / (s.std() + 1e-6)
            core = _blur(alpha, max(fx.px * 0.03, 1.0))
            v = 0.55 * g + 0.35 * s + 1.6 * (core - 0.5)
            alpha = alpha * smoothstep(-0.9, 0.6, v) * float(e.get("opacity", 0.92))
        elif k == "stencil":
            ly = fx.local_y()
            gaps = np.zeros_like(alpha)
            bw = float(e.get("bridge", 0.075))
            for (base, cap, _, _) in fx.block.lines:
                xh = _metrics(fx.block.pl[0][1] if fx.block.pl else None, fx.px)[3] if fx.block.pl else cap * 0.7
                yc = base - xh * 0.5
                gaps = np.maximum(gaps, smoothstep(fx.px * bw * 0.5 + 0.7, fx.px * bw * 0.5 - 0.7, np.abs(ly - yc)))
            alpha = alpha * (1 - gaps)
            rng = np.random.default_rng(fx.seed)
            sp = (rng.random((fx.H, fx.W)) > 0.82).astype(F32)
            spray = _blur(alpha, max(fx.px * 0.025, 0.8)) * sp * 0.35
            alpha = np.clip(alpha * 0.97 + spray * (1 - alpha), 0, 1)
    return col, alpha


def _paint_block(cv, block, color="#ffffff", gradient=None, outline=0.0, outline_color="#000000", shadow=None,
                 glow=None, opacity=1.0, mode="normal", aa=2, effect=None, plate=None, texture=None,
                 gradient_dir="y", seed=0, kind="text", record=True, clip_note=None):
    """Paint a laid-out block with every style option. Returns the bbox."""
    H, W = cv.h, cv.w
    px = block.px
    fill, groups = block.masks(H, W, aa)
    effects = _norm_effects(effect)
    kinds = [e["kind"] for e in effects]
    fx = _FX(cv, block, fill, groups, px, aa, seed)
    # ---- plate and highlighter marks behind the text
    if plate:
        _paint_plate(cv, block, plate, color, fill)
    if block.hl:
        _paint_highlights(cv, block, aa, opacity)
    # ---- automatic ink
    auto = isinstance(color, str) and color == "auto"
    if auto:
        bg, mu, sd = _bg_under(cv, fill, block.bbox, px)
        cw, cb = (1.05) / (mu + 0.05), (mu + 0.05) / 0.05
        color = "#ffffff" if cw >= cb else "#141414"
        if sd > 0.12 and not shadow and not plate and not outline:
            shadow = dict(dx=0, dy=px * 0.03, blur=px * 0.12, opacity=0.6,
                          color="#000000" if color == "#ffffff" else "#ffffff")
    # ---- base colour map
    if texture is not None:
        col = _texture_map(texture, block, H, W)
    elif gradient is not None:
        x0, y0, x1, y1 = block.bbox
        if gradient_dir == "x":
            X = np.arange(W, dtype=F32)[None, :] * np.ones((H, 1), F32)
            col = mix(gradient, (X - x0) / max(x1 - x0, 1.0))
        elif gradient_dir in ("line", "lines"):
            col = mix(gradient, np.clip(fx.band_t(), 0, 1))
        elif isinstance(gradient_dir, (int, float)):
            a = math.radians(gradient_dir)
            X, Y = cv.xy()
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            proj = (X - cx) * math.cos(a) + (Y - cy) * math.sin(a)
            half = abs((x1 - x0) / 2 * math.cos(a)) + abs((y1 - y0) / 2 * math.sin(a))
            col = mix(gradient, proj / max(2 * half, 1.0) + 0.5)
        else:
            Y = np.arange(H, dtype=F32)[:, None] * np.ones((1, W), F32)
            col = mix(gradient, (Y - y0) / max(y1 - y0, 1.0))
    else:
        col = as_color(color)
    colored = [k for k in groups if isinstance(k, tuple) and len(k) == 3]
    if colored:
        cmap = np.broadcast_to(np.asarray(col, F32), (H, W, 3)).astype(F32).copy()
        den = np.maximum(fill, 1e-6)
        for k in colored:
            w_ = np.clip(groups[k] / den, 0, 1)[..., None]
            cmap = cmap * (1 - w_) + np.asarray(k, F32)[None, None, :] * w_
        col = cmap
    base_col = col
    alpha = fill
    glitch = next((e for e in effects if e["kind"] == "glitch"), None)
    if glitch is not None:
        alpha = _glitch_slices(fx, glitch, fill)
    # ---- glow (and neon)
    neon = next((e for e in effects if e["kind"] == "neon"), None)
    off = _off_mask(fx) if neon is not None else None
    on = np.clip(alpha - (off if off is not None else 0), 0, 1)
    if glow:
        g = dict(radius=px * 0.35, color=color if gradient is None and not isinstance(color, np.ndarray) else
                 (gradient[-1][1] if gradient else "#ffffff"), strength=1.0)
        g.update(glow if isinstance(glow, dict) else {})
        if isinstance(g["color"], str) and g["color"] == "auto":
            g["color"] = "#ffffff"
        halo = _blur(on, g["radius"] * 0.5) * 0.6 + _blur(on, g["radius"]) * 0.8
        cv.paint(g["color"], halo * g["strength"] * opacity, "add")
    neon_col = None
    if neon is not None:
        neon_col = _fx_neon_glow(fx, neon, color, on)
    # ---- under effects
    for e in effects:
        k = e["kind"]
        if k == "long_shadow":
            _fx_long_shadow(fx, e, base_col)
        elif k == "extrude":
            _fx_extrude(fx, e, base_col)
        elif k == "sticker":
            _fx_sticker(fx, e, base_col)
    if shadow:
        s = dict(dx=px * 0.05, dy=px * 0.07, blur=px * 0.06, color="#000000", opacity=0.55)
        s.update(shadow if isinstance(shadow, dict) else {})
        base = alpha if not outline else block.masks(H, W, aa, outline)[0]
        sh = _shift(base, s["dx"], s["dy"])
        if s["blur"] > 0.3:
            sh = _blur(sh, s["blur"])
        cv.paint(s["color"], sh * s["opacity"] * opacity)
    if outline:
        ring = block.masks(H, W, aa, outline)[0]
        if glitch is not None:
            ring = np.maximum(ring, alpha)
        cv.paint(outline_color, ring * opacity, mode)
    if "knockout" in kinds:
        _fx_knockout(fx, next(e for e in effects if e["kind"] == "knockout"), base_col, color)
    # ---- fill
    col, alpha = _painted_fill(fx, kinds, effects, col, alpha)
    hollow = next((e for e in effects if e["kind"] == "outline_only"), None)
    emb = next((e for e in effects if e["kind"] in ("emboss", "letterpress")), None)
    if glitch is not None:
        _glitch_ghosts(fx, glitch, alpha, opacity)
    if neon is not None:
        _neon_fill(fx, neon, alpha, off, neon_col, opacity)
    elif hollow is not None:
        w = float(hollow.get("width", max(px * 0.045, 1.0)))
        ring = np.clip(block.masks(H, W, aa, w)[0] - fill, 0, 1)
        cv.paint(col, ring * opacity, mode)
    elif "knockout" in kinds:
        pass
    elif emb is not None:
        _emboss(fx, emb, col, alpha, opacity, mode)
    else:
        cv.paint(col, alpha * opacity, mode)
    if any(k in kinds for k in ("gold", "silver", "metal", "chrome", "rose_gold", "foil")):
        edge = np.clip(fill - _blur(fill, max(px * 0.012, 0.7)) * 0 - 0, 0, 1)
        del edge
    if block.missing:
        chars = "".join(dict.fromkeys(block.missing))
        _warn_once(f"no installed font can draw {chars!r} ({', '.join(f'U+{ord(c):04X}' for c in chars)}); they render "
                   f"as boxes. Try `python tools/fonts.py --install noto-symbols` (or noto-cjk / all)")
    if record:
        _record(cv, block, fill, kind, effects, clip_note)
    return block.bbox


def _paint_highlights(cv, block, aa, opacity):
    """Highlighter-marker rectangles: slightly slanted ends, soft edges, painted under the glyphs."""
    H, W = cv.h, cv.w
    by_col = {}
    for item in block.hl:
        by_col.setdefault(item[6], []).append(item)
    for col, items in by_col.items():
        ext = []
        polys = []
        for (a, b, c, d, M, o, _) in items:
            sk = (d - b) * 0.12
            pts = [(a + sk, b), (c + sk * 0.4, b + (d - b) * 0.06), (c - sk * 0.2, d), (a - sk * 0.3, d - (d - b) * 0.05)]
            M_ = M or (1, 0, 0, 1)
            q = [(o[0] + M_[0] * u + M_[1] * v, o[1] + M_[2] * u + M_[3] * v) for u, v in pts]
            polys.append(q)
            ext += q
        x0 = max(int(min(p[0] for p in ext)) - 2, 0)
        y0 = max(int(min(p[1] for p in ext)) - 2, 0)
        x1 = min(int(max(p[0] for p in ext)) + 3, W)
        y1 = min(int(max(p[1] for p in ext)) + 3, H)
        if x1 <= x0 or y1 <= y0:
            continue
        k = 3
        im = Image.new("L", ((x1 - x0) * k, (y1 - y0) * k), 0)
        dr = ImageDraw.Draw(im)
        for q in polys:
            dr.polygon([((x - x0) * k, (y - y0) * k) for x, y in q], fill=255)
        im = im.resize((x1 - x0, y1 - y0), _RES.BOX)
        m = np.zeros((H, W), F32)
        m[y0:y1, x0:x1] = np.asarray(im, F32) / 255.0
        cv.paint(col, m * 0.88 * opacity)


def _glitch_slices(fx, e, fill):
    rng = np.random.default_rng(fx.seed + 7)
    out = fill.copy()
    x0, y0, x1, y1 = [int(v) for v in fx.block.bbox]
    y0, y1 = max(y0, 0), min(y1, fx.H)
    if y1 - y0 < 4:
        return out
    for _ in range(int(e.get("slices", 5))):
        h = int(rng.integers(max(2, (y1 - y0) // 30), max(3, (y1 - y0) // 7)))
        ys = int(rng.integers(y0, max(y0 + 1, y1 - h)))
        dx = int(rng.normal(0, fx.px * float(e.get("amount", 0.08))))
        out[ys:ys + h] = _shift(fill[ys:ys + h], dx, 0)
    return out


def _glitch_ghosts(fx, e, alpha, opacity):
    off = float(e.get("offset", fx.px * 0.045))
    bg, mu, _ = _bg_under(fx.cv, alpha, fx.block.bbox, fx.px)
    c1, c2 = rgb(e.get("colors", ("#ff2a6d", "#05d9e8"))[0]), rgb(e.get("colors", ("#ff2a6d", "#05d9e8"))[1])
    fx.cv.paint(c1, _shift(alpha, -off, 0) * 0.85 * opacity, "screen" if mu < 0.5 else "multiply")
    fx.cv.paint(c2, _shift(alpha, off, off * 0.2) * 0.85 * opacity, "screen" if mu < 0.5 else "multiply")


def _neon_fill(fx, e, alpha, off, neon_col, opacity):
    cv, px = fx.cv, fx.px
    on = np.clip(alpha - (off if off is not None else 0), 0, 1)
    if e.get("tube", False):
        w = float(e.get("width", px * 0.06))
        edge = _blur(on, w * 0.8)
        tube = np.clip(1.0 - np.abs(edge - 0.5) * 2.6, 0, 1) ** 1.5
        core = np.clip(1.0 - np.abs(edge - 0.5) * 7.0, 0, 1)
        cv.paint(neon_col * 1.4, tube * opacity)
        cv.paint((1.6, 1.55, 1.5), core * opacity * 0.9, "normal")
    else:
        core = smoothstep(0.35, 0.95, _blur(on, max(px * 0.03, 0.8)))
        cv.paint(neon_col * 1.35, on * opacity)
        cv.paint((1.7, 1.65, 1.6), core * on * opacity * 0.85)
    if off is not None and off.max() > 0:
        cv.paint(neon_col * 0.16 + 0.04, off * opacity)
        hl = np.clip(_shift(off, 0, -max(px * 0.02, 1)) * 0 + smoothstep(0.6, 1.0, _blur(off, px * 0.02)) * 0.15, 0, 1)
        cv.paint("#ffffff", hl * opacity * 0.5)


def _emboss(fx, e, col, alpha, opacity, mode):
    cv, px = fx.cv, fx.px
    d = float(e.get("depth", max(px * 0.03, 1.2)))
    soft = max(d * 0.6, 0.7)
    inv = 1.0 - alpha
    if e["kind"] == "letterpress":
        cv.paint(col, alpha * opacity * float(e.get("ink", 0.95)), mode)
        inner = alpha * _blur(_shift(inv, d, d), soft)
        cv.paint("#000000", inner * 0.6 * opacity)
        outer = inv * _blur(_shift(alpha, d * 0.8, d * 0.8), soft)
        cv.paint("#ffffff", outer * 0.35 * opacity)
    else:
        if e.get("fill", False):
            cv.paint(col, alpha * opacity, mode)
        drop = inv * _blur(_shift(alpha, d, d * 1.2), soft * 1.6)
        cv.paint("#000000", drop * 0.45 * opacity)
        lit = alpha * _blur(_shift(inv, d, d), soft)
        dark = alpha * _blur(_shift(inv, -d, -d), soft)
        cv.paint("#ffffff", np.clip(dark - lit * 0.3, 0, 1) * 0.55 * opacity)
        cv.paint("#000000", np.clip(lit - dark * 0.3, 0, 1) * 0.4 * opacity)
        cv.paint("#ffffff", alpha * 0.06 * opacity)


def _texture_map(texture, block, H, W):
    if isinstance(texture, str):
        im = Image.open(texture).convert("RGB")
        arr = np.asarray(im, F32) / 255.0
    else:
        arr = as_color(texture)
    if arr.ndim == 3 and arr.shape[:2] == (H, W):
        return arr
    x0, y0, x1, y1 = [int(round(v)) for v in block.bbox]
    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
    th, tw = arr.shape[:2]
    s = max(bw / tw, bh / th)
    from .filters import resize
    r = resize(arr, max(int(tw * s + 1), 1), max(int(th * s + 1), 1), "bilinear")
    out = np.zeros((H, W, 3), F32)
    cx0, cy0 = max(x0, 0), max(y0, 0)
    cx1, cy1 = min(x0 + r.shape[1], W), min(y0 + r.shape[0], H)
    if cx1 > cx0 and cy1 > cy0:
        out[cy0:cy1, cx0:cx1] = r[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    return out


def _paint_plate(cv, block, plate, color, fill):
    from . import sdf
    px = block.px
    p = {"kind": plate} if isinstance(plate, str) else dict(plate if isinstance(plate, dict) else {"kind": "box"})
    kind = p.get("kind", "box")
    pad = float(p.get("pad", px * 0.32))
    pad_x = float(p.get("pad_x", pad * 1.5 if kind == "pill" else pad * 1.15))
    d = _plate_mask(cv, block, "pill" if kind == "pill" else "box", pad_x, pad, p.get("radius"))
    pc = p.get("color")
    if pc is None:
        if isinstance(color, str) and color != "auto":
            ink = float(_srgb_lum(rgb(color)))
            pc = "#101114" if ink > 0.35 else "#fbfaf6"
        else:
            bg, mu, _ = _bg_under(cv, fill, block.bbox, px)
            pc = "#101114" if mu > 0.35 else "#fbfaf6"
            if isinstance(color, str) and color == "auto":
                pc = "#101114" if mu < 0.35 else "#fbfaf6"
    op = float(p.get("opacity", 0.88 if kind in ("box", "pill") else 0.6))
    if kind in ("scrim", "blur", "glass"):
        soft = float(p.get("soft", px * 0.9))
        m = sdf.fill(d, aa=soft) if kind == "scrim" else sdf.fill(d, aa=max(px * 0.05, 1))
        if kind in ("blur", "glass"):
            from .filters import gaussian_blur
            x0, y0, x1, y1 = _crop_box(cv.h, cv.w, block.bbox, pad_x + px)
            sub = cv.buf[y0:y1, x0:x1]
            bl = np.zeros_like(cv.buf)
            bl[y0:y1, x0:x1] = gaussian_blur(sub, float(p.get("blur", px * 0.45)))
            cv.paint(bl, m)
            cv.paint(pc, m * float(p.get("opacity", 0.35)))
            cv.paint("#ffffff", sdf.outline(d, max(px * 0.02, 1.0)) * 0.25)
        else:
            cv.paint(pc, m * op)
        return
    if p.get("shadow", False):
        cv.paint("#000000", _blur(sdf.fill(d), px * 0.25) * 0.35)
    cv.paint(pc, sdf.fill(d) * op)
    if p.get("border"):
        cv.paint(p["border"], sdf.outline(d, float(p.get("border_width", max(px * 0.04, 1.0)))))


# --------------------------------------------------------------------------------------------- audit

def _record(cv, block, fill, kind, effects, note):
    H, W = cv.h, cv.w
    log = getattr(cv, "text_log", None)
    if log is None:
        try:
            cv.text_log = log = []
        except Exception:
            return
    ys, xs = np.nonzero(fill > 0.15)
    if len(xs):
        ink = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    else:
        ink = tuple(int(round(v)) for v in block.bbox)
    lb = block.bbox
    clipped = lb[0] < -0.5 or lb[1] < -0.5 or lb[2] > W + 0.5 or lb[3] > H + 0.5
    ext = block.ink_extent() or lb
    if ext[0] < -1 or ext[2] > W + 1 or ext[1] < -1 or ext[3] > H + 1:
        # ink estimate is generous; confirm with the real mask touching an edge
        if len(xs) and (xs.min() == 0 or ys.min() == 0 or xs.max() == W - 1 or ys.max() == H - 1):
            clipped = True
    contrast, worst, worst_box, busy = _measure_contrast(cv, fill, ink, block.px)
    decorative = any(e["kind"] in ("emboss", "outline_only", "glitch", "chalk") for e in effects)
    overlaps = []
    mine = fill[ink[1]:ink[3], ink[0]:ink[2]] > 0.04
    for i, prev in enumerate(log):
        if not prev.get("audit", True) or "_mask" not in prev:
            continue
        a, b = prev["bbox"], ink
        ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        if ix1 > ix0 and iy1 > iy0:
            pm = prev["_mask"][iy0 - a[1]:iy1 - a[1], ix0 - a[0]:ix1 - a[0]]
            mm = mine[iy0 - b[1]:iy1 - b[1], ix0 - b[0]:ix1 - b[0]]
            both = int(np.logical_and(pm, mm).sum())
            if both > max(12, 0.004 * min(int(prev["_mask"].sum()), int(mine.sum()) or 1)):
                overlaps.append(i)
    fonts = sorted({os.path.basename(_split_spec(f)[0]) if f else "pillow-default" for f in block.fonts})
    log.append(dict(text=getattr(block, "plain", "")[:120], bbox=ink, layout_bbox=tuple(round(v, 1) for v in lb),
                    size=round(block.px, 1), fonts=fonts, contrast=None if contrast is None else round(contrast, 2),
                    min_contrast=None if worst is None else round(worst, 2), worst_box=worst_box, busy=busy,
                    clipped=bool(clipped),
                    overlaps=overlaps, missing=sorted(set(block.missing)), kind=kind,
                    effects=[e["kind"] for e in effects], decorative=decorative, note=note, canvas=(W, H), audit=True,
                    _mask=fill[ink[1]:ink[3], ink[0]:ink[2]] > 0.2))
    _manifest.drawn_text(cv, len(log) - 1)               # the scene manifest keeps a reference, not a copy


def _ratio(a, b):
    return (np.maximum(a, b) + 0.05) / (np.minimum(a, b) + 0.05)


def _measure_contrast(cv, fill, ink, px):
    """(overall ratio, worst local ratio, bbox of the worst part, busy) of text ink against the pixels around it.

    The overall ratio compares the median glyph luminance with the median of a thin ring around the glyphs. The
    local ratio does the same per tile of about one em, so text that is only partly over a dark or light shape is
    caught; ``busy`` is True when a noticeable share of the ring matches the ink (a textured background).
    """
    H, W = cv.h, cv.w
    x0, y0, x1, y1 = _crop_box(H, W, ink, max(px * 0.2, 3))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None, None, None, False
    m = fill[y0:y1, x0:x1]
    inner = m > 0.85
    if inner.sum() < 4:
        inner = m > 0.5
    if inner.sum() < 2:
        return None, None, None, False
    near = _blur(m, max(px * 0.06, 1.2)) > 0.08
    ring = near & (m < 0.04)
    if ring.sum() < 4:
        return None, None, None, False
    lum = _srgb_lum(cv.buf[y0:y1, x0:x1])
    lt = float(np.median(lum[inner]))
    rl = lum[ring]
    lb = float(np.median(rl))
    ratio = float(_ratio(lt, lb))
    # busy background: the share of ring pixels whose value is close to the ink
    close = _ratio(rl, lt) < 1.6
    busy = bool(close.mean() > 0.12) and ratio >= 1.6
    # local contrast per ~1 em tile
    T = max(px * 1.1, 6.0)
    hh, ww = m.shape
    ty = (np.arange(hh) // T).astype(np.int64)
    tx = (np.arange(ww) // T).astype(np.int64)
    nx = int(tx[-1]) + 1
    tid = ty[:, None] * nx + tx[None, :]
    nt = int(tid.max()) + 1
    ci = np.bincount(tid[inner], minlength=nt)
    si = np.bincount(tid[inner], weights=lum[inner], minlength=nt)
    cr = np.bincount(tid[ring], minlength=nt)
    sr = np.bincount(tid[ring], weights=lum[ring], minlength=nt)
    ok = (ci >= max(4, 0.02 * T * T)) & (cr >= max(6, 0.04 * T * T))
    worst, wbox = ratio, None
    if ok.any():
        loc = np.where(ok, _ratio(si / np.maximum(ci, 1), sr / np.maximum(cr, 1)), np.inf)
        k = int(np.argmin(loc))
        if loc[k] < worst:
            worst = float(loc[k])
            ky, kx = divmod(k, nx)
            wbox = (int(x0 + kx * T), int(y0 + ky * T), int(min(x0 + (kx + 1) * T, x1)), int(min(y0 + (ky + 1) * T, y1)))
    return ratio, worst, wbox, busy


def audit_text(cv, min_size=None, margin=0.025, verbose=False):
    """Typography issues on a canvas: list of dicts (kind, text, bbox, message, fix).

    Checks every ``cv.text`` / ``draw_text`` / ``text_on_arc`` call: too small for the canvas, low WCAG contrast,
    clipped by the canvas, overlapping earlier text, touching the margin, missing glyphs. ``verbose`` prints them.
    """
    log = getattr(cv, "text_log", []) or []
    H, W = cv.h, cv.w
    short = min(H, W)
    min_px = min_size if min_size is not None else max(5.0, 0.014 * short)
    mpx = margin * short
    issues = []

    def add(kind, e, msg, fix):
        issues.append(dict(kind=kind, text=e["text"], bbox=e["bbox"], message=msg, fix=fix))
    for i, e in enumerate(log):
        if not e.get("audit", True):
            continue
        x0, y0, x1, y1 = e["bbox"]
        if e["size"] < min_px and e["kind"] != "path":
            add("small", e, f"{e['size']:.0f}px text is hard to read on a {W}x{H} canvas (minimum ~{min_px:.0f}px)",
                f"size={min_px * 1.25:.0f} or larger")
        if e["contrast"] is not None and not e["decorative"]:
            need = 3.0 if e["size"] >= 0.035 * short else 4.5
            if round(e["contrast"], 1) < need:
                add("contrast", e, f"contrast {e['contrast']:.1f}:1 against the background (needs {need:.1f}:1)",
                    'color="auto", plate="pill" / plate="scrim", or shadow=True')
            elif e.get("min_contrast") is not None and e["min_contrast"] < need * 0.75 and not e["effects"]:
                add("contrast", e, f"part of this text has only {e['min_contrast']:.1f}:1 contrast (needs {need:.1f}:1) "
                    f"at {e.get('worst_box')}: it crosses a shape of similar value",
                    'move it clear of the shape, color="auto" with plate="scrim" / plate="box", or outline=...')
            elif e.get("busy") and e["size"] < 0.06 * short:
                add("contrast", e, "parts of the background behind this text match its colour (busy background)",
                    'plate="scrim" or plate="box", or shadow=dict(blur=...)')
        if e["clipped"]:
            add("clipped", e, f"text runs off the canvas (bbox {e['layout_bbox']})",
                "move it inside, use a smaller size or box=..., or pass clip_ok=True if intended")
        elif x0 < mpx or y0 < mpx or x1 > W - mpx or y1 > H - mpx:
            add("margin", e, f"text touches the canvas edge (closer than {mpx:.0f}px)",
                f"keep text ≥ {0.05 * short:.0f}px from the edges (layout.Layout(cv, margin=0.06))")
        for j in e["overlaps"]:
            add("overlap", e, f"overlaps earlier text #{j} {log[j]['text'][:30]!r} at {log[j]['bbox']}",
                "move one of them, or set both with layout.stack(...)")
        if e["missing"]:
            chars = "".join(e["missing"])
            add("missing_glyphs", e, f"no installed font can draw {chars!r}",
                "python tools/fonts.py --install noto-symbols (or noto-cjk for CJK)")
        if e.get("note"):
            add("adjusted", e, e["note"], "check the placement; pass clip_ok=True to allow text off the canvas")
    if verbose:
        if not issues:
            print("text audit: no issues")
        for it in issues:
            print(f"[{it['kind']}] {it['text'][:40]!r} at {it['bbox']}: {it['message']} → {it['fix']}")
    return issues


audit = audit_text


# --------------------------------------------------------------------------------------------- arcs & paths

def _glyph_stream(text, ctx, markup=None, glyph_key=None):
    """[(char, spec, px, weight, synth, x, adv, key, synth_italic)] along a straight baseline."""
    spans = _span_list(text, markup)
    chars = _char_pieces(spans, ctx)
    out = []
    x = 0.0
    prev = None
    gi = 0
    for ch, st in chars:
        if ch == "\n":
            ch = " "
        sp = _first_spec(ch, st)
        miss = isinstance(sp, tuple)
        if miss:
            sp = sp[1]
        info = _info(sp)
        if prev is not None and info is not None and prev[1] == sp and prev[2] == st.px:
            x += info.kern_units(prev[0], ch) * st.px / info.upem
        adv = _shape(ch, sp, round(st.px, 3), st.weight, 0.0)[1]
        synth = _load2(sp, round(st.px * _MS), st.weight)[1]
        key = st.color
        if glyph_key is not None and ch.strip():
            key = glyph_key(gi) or key
            gi += 1
        if st.off:
            key = ("off",)
        out.append((ch, sp, st.px, st.weight, synth, x, adv, key, st.synth_i, miss))
        x += adv + st.track
        prev = (ch, sp, st.px)
    return out, x - (chars[-1][1].track if chars else 0.0)


def _path_block(stream, total, sample, px, valign_shift):
    """Place a glyph stream along a curve. ``sample(s)`` → (x, y, tx, ty) at arc length s."""
    block = _Block()
    block.px = px
    for (ch, sp, p, w, synth, x, adv, key, si, miss) in stream:
        if not ch.strip():
            continue
        block.fonts.add(sp)
        if miss:
            block.missing.append(ch)
        s = x + adv / 2
        X, Y, tx, ty = sample(s)
        nx, ny = -ty, tx                                # local +y (down) direction
        M = (tx, nx, ty, ny)
        if si:
            M = _mat_mul(M, (1.0, -0.21, 0.0, 1.0))
        bx, by = X + nx * valign_shift, Y + ny * valign_shift
        ox = bx - (M[0] * adv / 2)
        oy = by - (M[2] * adv / 2)
        block.pl.append((ch, sp, p, w, synth, M, ox, oy, key))
    ext = block.ink_extent()
    if ext is not None:
        block.bbox = _tight_bbox(block)
    return block


def _tight_bbox(block):
    xs, ys = [], []
    for (t, sp, p, w, sy, M, tx, ty, k) in block.pl:
        adv = _shape(t, sp, round(p, 3), w, 0.0)[1]
        a, d, cap, xh = _metrics(sp, p, w)
        M_ = M or (1, 0, 0, 1)
        for (u, v) in ((0, -a), (adv, -a), (0, d), (adv, d)):
            xs.append(tx + M_[0] * u + M_[1] * v)
            ys.append(ty + M_[2] * u + M_[3] * v)
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    return min(xs), min(ys), max(xs), max(ys)


_STYLE_KW = {"color", "gradient", "outline", "outline_color", "shadow", "glow", "opacity", "mode", "aa", "effect",
             "plate", "texture", "gradient_dir", "seed"}


def text_on_arc(cv, text, center, radius, size=None, angle=-90.0, side="auto", upright=False, font=None, bold=False,
                italic=False, tracking=None, color="#ffffff", align="center", markup=None, case=None, lang=None,
                letter_colors=None, audit=True, repeat=None, **style):
    """Set text along a circle — badges, seals, stamps, curved labels. Returns the bbox.

    * ``center`` (x, y), ``radius`` px: the circle through the middle of the letters (default size 0.2 x radius).
    * ``angle``: where the text is centred, degrees (-90 = top, 90 = bottom, 0 = right).
    * ``side``: ``"outside"`` reads clockwise with letter tops pointing out (top of a seal), ``"inside"`` reads
      left-to-right along the bottom with tops pointing to the centre; ``"auto"`` picks by ``angle``.
    * ``upright=True`` keeps every letter vertical. ``align`` = center / start / end / ``"ring"`` (spread evenly
      round the whole circle; text that is too long is shrunk to fit).
    * ``repeat=True`` (or a separator string, default ``" · "``) repeats the text round the whole ring:
      ``text_on_arc(cv, "LIVE", c, r, repeat=True)`` → LIVE · LIVE · LIVE · … closing seamlessly.
    * Style keywords of ``draw_text`` apply (``outline``, ``shadow``, ``glow``, ``effect``, ``gradient`` ...).
    """
    R = float(radius)
    size = size or R * 0.2
    ring = align == "ring" or bool(repeat)
    if tracking is None:
        tracking = 0.08 if all(not c.isalpha() or c.isupper() for c in str(text)) else 0.03
    if side == "auto":
        side = "inside" if 0 < (angle % 360) < 180 and not ring else "outside"
    circ = 2 * math.pi * R
    plain = str(text)

    def build(t, sz):
        c = _Ctx(sz, font, bold, italic, tracking, case, False, False, None, False, lang, t)
        s, tot = _glyph_stream(t, c, markup, _letter_key(letter_colors, t))
        return c, s, tot
    if repeat:
        sep = repeat if isinstance(repeat, str) else " · "
        unit = plain.rstrip() + sep
        _, _, ulen = build(unit, size)
        n = max(1, int(circ // max(ulen, 1e-6)))
        plain = unit * n
    ctx, stream, total = build(plain, size)
    if ring and stream:
        # natural length incl. one tracking gap where the ring closes
        nat = total + ctx.sty({}).track
        if nat > circ * 0.995:
            size *= circ * 0.97 / nat
            ctx, stream, total = build(plain, size)
            nat = total + ctx.sty({}).track
        # spare length goes mostly into word spaces (and the gap where the ring closes), a little into letters
        wts = [2.5 if c.isspace() else 1.0 for c, *_ in stream]
        wts[-1] = 2.5
        u = (circ - nat) / sum(wts)
        acc, out = 0.0, []
        for (c, sp, p, w, sy, x, adv, key, si, miss), wt in zip(stream, wts):
            out.append((c, sp, p, w, sy, x + acc, adv, key, si, miss))
            acc += wt * u
        stream = out
        total = circ - wts[-1] * u - ctx.sty({}).track
        align = "center"
    a0 = math.radians(angle)
    cx, cy = center
    sty = ctx.sty({})
    cap = _metrics(sty.font, size, sty.weight)[2]
    off = {"center": -total / 2, "start": 0.0, "end": -total}.get(align, -total / 2)
    sign = 1.0 if side == "outside" else -1.0

    def sample(s):
        th = a0 + sign * (s + off) / R
        X, Y = cx + R * math.cos(th), cy + R * math.sin(th)
        if side == "outside":
            tx, ty = -math.sin(th), math.cos(th)
        else:
            tx, ty = math.sin(th), -math.cos(th)
        if upright:
            return X, Y, 1.0, 0.0
        return X, Y, tx, ty
    block = _path_block(stream, total, sample, size, cap / 2)
    block.plain = plain
    if audit is False:
        style["record"] = False
    style = {k: v for k, v in style.items() if k in _STYLE_KW or k == "record"}
    return _paint_block(cv, block, color=color, kind="arc", **style)


def text_on_path(cv, text, points, size=40, offset=0.0, align="start", font=None, bold=False, italic=False,
                 tracking=0.0, color="#ffffff", upright=False, baseline=0.0, markup=None, case=None, lang=None,
                 letter_colors=None, audit=True, **style):
    """Set text along any polyline (list of (x, y), e.g. ``bezier_points(...)``). Returns the bbox.

    ``align`` = start / center / end along the path, ``offset`` px shifts it along; the path is the baseline
    (``baseline`` px moves the text off it, positive = below). ``upright=True`` keeps letters vertical.
    """
    P = np.asarray(points, np.float64)
    seg = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    L = cum[-1]
    plain = str(text)
    ctx = _Ctx(size, font, bold, italic, tracking, case, False, False, None, False, lang, plain)
    stream, total = _glyph_stream(text, ctx, markup, _letter_key(letter_colors, plain))
    start = {"start": 0.0, "center": (L - total) / 2, "end": L - total}.get(align, 0.0) + offset

    def at(s):
        s = min(max(s, 0.0), L)
        i = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(seg) - 1))
        t = (s - cum[i]) / max(seg[i], 1e-9)
        return P[i] + (P[i + 1] - P[i]) * t

    def sample(s):
        s = s + start
        p = at(s)
        h = max(size * 0.35, 1.0)
        a, b = at(s - h), at(s + h)
        d = b - a
        n = math.hypot(d[0], d[1]) or 1.0
        tx, ty = d[0] / n, d[1] / n
        if s > L:                                   # extend straight past the end
            p = P[-1] + np.array([tx, ty]) * (s - L)
        elif s < 0:
            p = P[0] + np.array([tx, ty]) * s
        if upright:
            return p[0], p[1], 1.0, 0.0
        return p[0], p[1], tx, ty
    block = _path_block(stream, total, sample, size, baseline)
    block.plain = plain
    if audit is False:
        style["record"] = False
    style = {k: v for k, v in style.items() if k in _STYLE_KW or k == "record"}
    return _paint_block(cv, block, color=color, kind="path", **style)


def _letter_key(letter_colors, plain, off=None):
    if not letter_colors and not off:
        return None
    n = max(sum(1 for c in plain if c.strip()), 1)
    stops = None
    cols = None
    if letter_colors:
        if all(isinstance(c, (tuple, list)) and len(c) == 2 and isinstance(c[0], (int, float)) for c in letter_colors):
            stops = letter_colors
        else:
            cols = [tuple(float(v) for v in rgb(_NAMED.get(c, c))) for c in letter_colors]
    off = set(off or [])

    def key(i):
        if i in off:
            return ("off",)
        if stops is not None:
            return tuple(float(v) for v in mix(stops, i / max(n - 1, 1)))
        if cols:
            return cols[i % len(cols)]
        return None
    return key


# --------------------------------------------------------------------------------------------- styled drawing

def draw_text(cv, text, xy=None, size=None, color="#ffffff", font=None, bold=False, anchor="la", align=None,
              max_width=None, line_spacing=None, tracking=None, box=None, box_anchor="mm", padding=0.0,
              gradient=None, outline=0.0, outline_color="#000000", shadow=None, glow=None, opacity=1.0,
              mode="normal", aa=2, *, italic=False, rotate=0.0, vertical=False, markup=None, effect=None, plate=None,
              texture=None, letter_colors=None, gradient_dir="y", case=None, underline=False, strike=False,
              highlight=None, tabular=False, list_style=None, drop_cap=None, tabs=None, balance=False,
              para_spacing=0.0, lang=None, max_height=None, arc=None, path=None, clip_ok=False, seed=0, audit=True,
              indent=0.0):
    """Paint styled text onto a Canvas and return its bounding box (x0, y0, x1, y1).

    Placement — ``xy`` + ``anchor`` (+ ``size``), or ``box=(x0, y0, x1, y1)``: auto-sized to the box (minus
    ``padding``) and placed by ``box_anchor``; ``max_width=True`` with a box wraps (paragraph mode),
    ``balance=True`` wraps into even lines (headlines). ``rotate`` degrees counter-clockwise around ``xy``;
    ``vertical=True`` sets CJK top-to-bottom in right-to-left columns; ``arc=(cx, cy, r)`` or ``path=[...]`` sets
    it on a curve (see ``text_on_arc`` / ``text_on_path``). Text is kept inside the canvas unless ``clip_ok=True``.

    Text: ``text`` may be a string, light markup (``**bold** *italic* {#f33|red} {size=1.5|big} ==marker==``),
    a list of ``(text, style)`` spans, or a list of lines. Lines starting with ``- `` / ``• `` / ``1. `` become
    lists with hanging indents (``list_style="bullet" | "number" | "dash" | "check"`` forces one).
    ``align="justify"``, ``case="upper" | "lower" | "title" | "smallcaps"``, ``underline``, ``strike``,
    ``highlight="#ffe45c"``, ``drop_cap=3``, ``tabs=[x, (x, "r"), (x, ".")]`` with ``\\t``, ``tabular=True``
    (fixed-width digits), ``italic``, ``bold=True | "black" | "light"``, ``font`` = name / path / style words
    (``"serif"``, ``"condensed bold"``, ``"handwriting"`` ...). Missing glyphs come from fallback fonts.

    Style (drawn back to front: plate → glow → under-effects → shadow → outline → fill → over-effects):
      * ``color`` fill (HDR > 1 allowed) or ``"auto"`` (black / white by contrast with what is underneath);
        ``gradient=[(0, c0), (1, c1)]`` with ``gradient_dir`` = "y" / "x" / "line" / degrees; ``texture`` = image /
        array / Canvas filling the letters; ``letter_colors=[c0, c1, ...]`` (cycled) or stops (per-letter gradient).
      * ``outline`` px + ``outline_color``; ``shadow=dict(dx, dy, blur, color, opacity)`` or True;
        ``glow=dict(radius, color, strength)`` or True.
      * ``plate="pill" | "box" | "scrim" | "blur"`` or ``dict(kind, color, opacity, pad, radius, border)``.
      * ``effect`` = name, dict(kind=..., options) or a list: long_shadow, extrude, emboss, letterpress, gold,
        silver, chrome, rose_gold, foil, neon (``off=[2, 5]`` letters unlit, ``tube=True``), sticker, knockout,
        outline_only, stripes, glitch, chalk, stencil.
    Every call is recorded in ``cv.text_log``; ``audit_text(cv)`` reports problems.
    Options are checked: an unknown name raises a TypeError that names the closest option; both spellings
    (``colour`` / ``color``, ``w`` / ``width`` ...) are accepted.
    """
    H, W = cv.h, cv.w
    shadow, glow, plate = _check_style_options(shadow, glow, plate, case, list_style, gradient_dir, box_anchor)
    spans = _span_list(text, markup)
    plain = _plain(spans)
    style = dict(color=color, gradient=gradient, outline=outline, outline_color=outline_color, shadow=shadow,
                 glow=glow, opacity=opacity, mode=mode, aa=aa, effect=effect, plate=plate, texture=texture,
                 gradient_dir=gradient_dir, seed=seed)
    effects = _norm_effects(effect)
    neon = next((e for e in effects if e["kind"] == "neon"), None)
    off = None
    if neon is not None and neon.get("off"):
        off = neon["off"]
    gk = _letter_key(letter_colors, plain, off)
    if arc is not None:
        cx, cy, r = arc[:3]
        kw = dict(arc[3]) if len(arc) > 3 else {}
        return text_on_arc(cv, text, (cx, cy), r, size=size, font=font, bold=bold, italic=italic, tracking=tracking,
                           markup=markup, case=case, lang=lang, letter_colors=letter_colors, audit=audit, **kw, **style)
    if path is not None:
        return text_on_path(cv, text, path, size=size or 40, font=font, bold=bold, italic=italic,
                            tracking=tracking or 0.0, markup=markup, case=case, lang=lang, letter_colors=letter_colors,
                            audit=audit, **style)
    lay = dict(italic=italic, markup=markup, case=case, underline=underline, strike=strike, highlight=highlight,
               tabular=tabular, lang=lang, list_style=list_style, drop_cap=drop_cap, tabs=tabs, balance=balance,
               para_spacing=para_spacing, vertical=vertical, indent=indent)
    if box is not None:
        bx0, by0, bx1, by1 = (float(v) for v in box)
        bw, bh = bx1 - bx0 - 2 * padding, by1 - by0 - 2 * padding
        wrap = max_width is True or balance
        ls = line_spacing if line_spacing is not None else (1.2 if not has_cjk(plain) else 1.35)
        tr = tracking if tracking is not None else 0.0
        if size is None:
            size = fit_text_size(spans if not isinstance(text, str) else text, bw, bh, font, bold, wrap=wrap,
                                 line_spacing=ls, tracking=tr, **lay)
            if line_spacing is None or tracking is None:
                ls2 = line_spacing if line_spacing is not None else _auto_leading(plain, size, H)
                tr2 = tracking if tracking is not None else _auto_tracking(plain, size, H)
                if (ls2, tr2) != (ls, tr):
                    size = fit_text_size(spans if not isinstance(text, str) else text, bw, bh, font, bold, wrap=wrap,
                                         line_spacing=ls2, tracking=tr2, **lay)
                ls, tr = ls2, tr2
        line_spacing, tracking = ls, tr
        if vertical:
            max_height = bh if wrap else max_height
            max_width = None
        else:
            max_width = bw if wrap else None
        ha, va = box_anchor[0], box_anchor[1]
        xy = ({"l": bx0 + padding, "m": (bx0 + bx1) / 2, "r": bx1 - padding}[ha],
              {"t": by0 + padding, "m": (by0 + by1) / 2, "b": by1 - padding}[va])
        anchor = ha + va
    if xy is None:
        raise ValueError("draw_text needs xy, box, arc or path")
    size = float(size or 40)
    if max_width is True:
        max_width = None
    if line_spacing is None:
        line_spacing = _auto_leading(plain, size, H) if (max_width is not None or "\n" in plain or isinstance(text, list)) else 1.2
    if tracking is None:
        tracking = _auto_tracking(plain, size, H)
    al = _resolve_align(align, anchor)
    if vertical and max_height is None and box is None:
        max_height = None
    src = spans if not isinstance(text, str) else text
    block = _make_block(src, xy, size, font, bold, anchor, al, max_width, line_spacing, tracking, glyph_key=gk,
                        rotate=rotate, max_height=max_height, **lay)
    note = None
    if not clip_ok:
        block, note = _keep_inside(block, cv, src, xy, size, font, bold, anchor, al, max_width, line_spacing,
                                   tracking, gk, rotate, max_height, lay)
    style["record"] = audit is not False
    return _paint_block(cv, block, kind="vertical" if vertical else "text", clip_note=note, **style)


def _keep_inside(block, cv, src, xy, size, font, bold, anchor, al, max_width, line_spacing, tracking, gk, rotate,
                 max_height, lay):
    H, W = cv.h, cv.w
    x0, y0, x1, y1 = block.bbox
    bw, bh = x1 - x0, y1 - y0
    note = None
    if bw > W * 1.001 or bh > H * 1.001:
        f = min(W * 0.96 / max(bw, 1), H * 0.96 / max(bh, 1))
        size = max(size * f, 4)
        mw = max_width * 1.0 if max_width is not None else None
        block = _make_block(src, xy, size, font, bold, anchor, al, mw, line_spacing, tracking, glyph_key=gk,
                            rotate=rotate, max_height=max_height, **lay)
        x0, y0, x1, y1 = block.bbox
        note = f"text was larger than the canvas; size reduced to {size:.0f}px"
    dx = (-x0 if x0 < 0 else 0.0) + (W - x1 if x1 > W else 0.0)
    dy = (-y0 if y0 < 0 else 0.0) + (H - y1 if y1 > H else 0.0)
    if abs(dx) > 0.5 or abs(dy) > 0.5:
        block.translate(dx, dy)
        note = (note + "; " if note else "") + f"moved ({dx:+.0f}, {dy:+.0f})px to stay inside the canvas"
    return block, note


# --------------------------------------------------------------------------------------------- checked options

def _strict_layout(fn, owner, extra):
    """``fn`` takes layout keywords through ``**kw`` (``italic``, ``markup`` ...): allow the signature plus those."""
    import functools
    names = _sig_options(fn)
    allowed = names + [e for e in sorted(extra) if e not in names]
    known = frozenset(allowed)

    @functools.wraps(fn)
    def wrapper(*a, **k):
        if k and not known.issuperset(k):
            k = _reject_unknown(owner, k, allowed)
        return fn(*a, **k)
    return wrapper


text_bbox = _strict_layout(text_bbox, "text_bbox", _LAYOUT_KW)
fit_text_size = _strict_layout(fit_text_size, "fit_text_size", _LAYOUT_KW)
text_mask = _strict_layout(text_mask, "text_mask", _LAYOUT_KW)
text_on_arc = _strict_layout(text_on_arc, "text_on_arc", _STYLE_KW)
for _n in ("find_font", "list_fonts", "load_font", "has_cjk", "wrap_text", "draw_text", "missing_glyphs",
           "text_on_path", "parse_markup", "audit_text", "contrast_ratio"):
    globals()[_n] = _strict(globals()[_n], _n)
del _n


# ---- did-you-mean for names that do not exist (pixelcanvas._args.module_getattr)
from ._args import module_getattr as _module_getattr  # noqa: E402

_HINTS = {
    "text": "use cv.text(string, (x, y), size) on a Canvas; for a mask use text.text_mask(h, w, text, xy, size)",
    "draw": "use text.draw_text(cv, string, xy, size) or cv.text(string, (x, y), size)",
    "render": "use text.draw_text(cv, string, xy, size) or cv.text(string, (x, y), size); text_mask(h, w, text, xy, size) gives a mask",
    "measure": "use text.text_bbox(text, xy, size) -> (x0, y0, x1, y1)",
    "size": "use text.fit_text_size(text, box_w, box_h) for the largest size that fits, or text.text_bbox for measurements",
    "textsize": "use text.text_bbox(text, xy, size) -> (x0, y0, x1, y1)",
    "bbox": "use text.text_bbox(text, xy, size)",
    "wrap": "use text.wrap_text(text, font, max_width)",
    "font": "use text.find_font(name) for a font spec or text.load_font(font, size) for a Pillow font; text.list_fonts() lists them",
    "fonts": "use text.list_fonts(pattern='')",
    "getfont": "use text.find_font(name) or text.load_font(font, size)",
    "load": "use text.load_font(font, size)",
    "arc": "use text.text_on_arc(cv, text, center, radius, size)",
    "curve": "use text.text_on_path(cv, text, points, size) or text.text_on_arc(cv, text, center, radius, size)",
    "path": "use text.text_on_path(cv, text, points, size)",
    "mask": "use text.text_mask(h, w, text, xy, size)",
    "check": "use text.audit_text(cv) for legibility / clipping issues, text.missing_glyphs(text) for coverage",
    "coverage": "use text.missing_glyphs(text) -> characters no installed font can draw",
    "contrast": "use text.contrast_ratio(a, b) (WCAG ratio of two colours)",
    "markup": "use text.parse_markup(text)",
    "cjk": "use text.has_cjk(text)",
}
__getattr__ = _module_getattr(__name__, _HINTS)
