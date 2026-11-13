# -*- coding: utf-8 -*-
"""spec: pictures from a small JSON description ("pcscene" version 1), drawn by the kits.

A spec names things from the catalogue and says where they go; this module checks it, repairs what is
unambiguous, places everything and draws it through the kits. ``tools/compose.py`` is the command line.

>>> from pixelcanvas import spec
>>> s, notes = spec.loads('{"pcscene": 1, "kind": "nature", "layers": [{"draw": "oak", "at": "left"}]}')
>>> norm, problems = spec.validate(s)
>>> cv = spec.render(norm, 960, 540)

Format (every field except ``kind`` is optional)::

    {"pcscene": 1, "prompt": "the request, verbatim", "kind": "poster", "size": [1080, 1350],
     "palette": "midnight gold", "background": {"backdrop": "spotlight"},
     "layers": [{"draw": "moon", "at": "centre", "size": "large", "color": "gold"},
                {"text": "Blue Hour", "role": "title"}],
     "style": null}

Whole-picture kinds use one section instead of layers: ``diagram``, ``chart``, ``dashboard``, ``ui``,
``scene3d`` (kinds ``room`` and ``scene3d``). ``EXAMPLES[kind]`` is a complete spec for every kind.
"""
from __future__ import annotations

import copy
import difflib
import importlib
import inspect
import json
import math
import os
import re
import sys

import numpy as np

__all__ = ["VERSION", "KINDS", "EXAMPLES", "Problem", "SpecError", "loads", "load", "dumps", "validate", "render",
           "names", "lookup", "example", "schema_text", "set_value", "add_layer", "remove_layer", "next_steps",
           "scene_module_source", "parse_color"]

VERSION = 1

# --------------------------------------------------------------------------------------------- kinds
# group: how the picture is built; check: the --check kind; size: default canvas
KINDS = {
    "poster":     dict(group="design", check="design", size=(1080, 1350), desc="poster, flyer, event or cover with a picture and big type"),
    "card":       dict(group="design", check="design", size=(1200, 630), desc="social card, banner, thumbnail, quote card"),
    "wallpaper":  dict(group="design", check="design", size=(1920, 1080), desc="background or wallpaper, optional text"),
    "icons":      dict(group="design", check="design", size=(1200, 800), desc="a sheet of icons or pictograms in a grid"),
    "scene":      dict(group="scene", check="painting", size=(1920, 1080), desc="any illustrated scene outdoors: sky, ground, things on it"),
    "nature":     dict(group="scene", check="painting", size=(1920, 1080), desc="trees, flowers, rocks, weather"),
    "animals":    dict(group="scene", check="painting", size=(1920, 1080), desc="animals in a setting"),
    "people":     dict(group="scene", check="painting", size=(1920, 1080), desc="people in a setting"),
    "landscape":  dict(group="scene", check="painting", size=(1920, 1080), desc="mountains, hills, sea, desert, city skyline"),
    "still_life": dict(group="still_life", check="painting", size=(1920, 1080), desc="objects on a table: fruit, mugs, vases, bottles"),
    "room":       dict(group="scene3d", check="painting", size=(1440, 1080), desc="a furnished room in 3D (isometric by default)"),
    "scene3d":    dict(group="scene3d", check="painting", size=(1920, 1080), desc="3D objects: products, toys, houses, cars"),
    "diagram":    dict(group="diagram", check="diagram", size=(1920, 1080), desc="flowchart or architecture diagram: boxes and arrows"),
    "chart":      dict(group="chart", check="diagram", size=(1600, 900), desc="one chart: bar, line, pie, area, scatter ..."),
    "dashboard":  dict(group="dashboard", check="design", size=(1920, 1080), desc="KPI tiles and several charts"),
    "ui":         dict(group="ui", check="design", size=(1600, 1000), desc="app or website screen mock-up"),
}
KIND_ALIASES = {"flyer": "poster", "cover": "poster", "book cover": "poster", "event": "poster", "banner": "card",
                "thumbnail": "card", "social card": "card", "quote": "card", "quote card": "card", "social": "card",
                "background": "wallpaper", "backdrop": "wallpaper", "icon": "icons", "logo": "icons",
                "illustration": "scene", "painting": "scene", "picture": "scene", "outdoor": "scene",
                "animal": "animals", "person": "people", "portrait": "people", "plants": "nature", "garden": "nature",
                "still life": "still_life", "stilllife": "still_life", "table": "still_life", "interior": "room",
                "bedroom": "room", "isometric": "room", "3d": "scene3d", "product": "scene3d", "flowchart": "diagram",
                "architecture": "diagram", "graph": "chart", "bar chart": "chart", "plot": "chart",
                "infographic": "dashboard", "app": "ui", "website": "ui", "screen": "ui", "mockup": "ui",
                "mountains": "landscape", "seascape": "landscape"}
SECTIONS = ("diagram", "chart", "dashboard", "ui", "scene3d")
SECTION_KIND = {"diagram": "diagram", "chart": "chart", "dashboard": "dashboard", "ui": "ui", "scene3d": "scene3d"}

TOP_KEYS = ("pcscene", "prompt", "kind", "size", "palette", "background", "layers", "style", "post", "seed",
            "theme") + SECTIONS
TOP_ALIASES = {"version": "pcscene", "request": "prompt", "description": "prompt", "type": "kind",
               "canvas": "size", "dimensions": "size", "resolution": "size", "colors": "palette",
               "colours": "palette", "colour_palette": "palette", "color_palette": "palette", "bg": "background",
               "backdrop": "background", "objects": "layers", "items": "layers", "elements": "layers",
               "things": "layers", "content": "layers", "contents": "layers", "children": "layers",
               "medium": "style", "art_style": "style", "filter": "style", "effects": "post", "3d": "scene3d",
               "scene_3d": "scene3d", "graph": "chart", "plot": "chart", "screen": "ui", "app": "ui",
               "flowchart": "diagram", "architecture": "diagram", "colour_scheme": "palette",
               "color_scheme": "palette", "scheme": "palette", "colors_scheme": "palette", "theme_colors": "palette",
               "dimension": "size", "format": "size", "aspect": "size", "image_size": "size", "canvas_size": "size",
               "art": "style", "look": "style", "category": "kind", "template": "kind", "picture_type": "kind"}
TEXT_TOP = ("title", "subtitle", "kicker", "details", "footer", "caption", "badge", "byline", "author", "quote",
            "headline", "tagline", "date", "body")

ROLES = ("title", "subtitle", "kicker", "details", "body", "caption", "label", "badge", "footer", "byline",
         "quote", "date", "meta")
ROLE_ALIASES = {"headline": "title", "heading": "title", "name": "title", "tagline": "subtitle",
                "sub": "subtitle", "subheading": "subtitle", "eyebrow": "kicker", "overline": "kicker",
                "info": "details", "detail": "details", "when": "details", "where": "details", "time": "details",
                "venue": "details", "location": "details", "text": "body", "paragraph": "body",
                "description": "body", "note": "caption", "credit": "byline", "author": "byline", "by": "byline",
                "signature": "byline", "tag": "badge", "pill": "badge", "sticker": "badge", "cta": "badge",
                "button": "badge", "url": "footer", "website": "footer", "copyright": "footer"}

# size words scale the thing's natural size (a "large" bird is still a bird); "full" fills the height
SIZE_WORDS = {"tiny": 0.4, "very small": 0.55, "small": 0.72, "little": 0.72, "medium": 1.0, "normal": 1.0,
              "regular": 1.0, "large": 1.4, "big": 1.4, "very large": 1.75, "huge": 2.0, "giant": 2.4}
FORMAT_WORDS = {"portrait": (1080, 1350), "poster": (1080, 1350), "story": (1080, 1920), "phone": (1080, 1920),
                "vertical": (1080, 1920), "square": (1080, 1080), "landscape": (1920, 1080), "wide": (1920, 1080),
                "widescreen": (1920, 1080), "hd": (1920, 1080), "thumbnail": (1280, 720), "banner": (1500, 500),
                "card": (1200, 630), "a4": (2480, 3508), "desktop": (1920, 1080)}


class SpecError(Exception):
    """The spec could not be read at all (not JSON even after repairs)."""


class Problem:
    """One thing wrong with a spec: where (``layers[1].draw``), what, and the fix."""

    def __init__(self, path, message, fix=None, severity="error"):
        self.path, self.message, self.fix, self.severity = path, message, fix, severity

    def __str__(self):
        return f"{self.path}: {self.message}" + (f" -> {self.fix}" if self.fix else "")

    def as_dict(self):
        return {"path": self.path, "message": self.message, "fix": self.fix, "severity": self.severity}

    def __repr__(self):
        return f"Problem({str(self)!r})"


# ============================================================================================= sloppy JSON
_FENCE = re.compile(r"```[a-zA-Z0-9_-]*\s*\n?(.*?)```", re.S)
_LITERALS = {"true": "true", "false": "false", "null": "null", "True": "true", "False": "false", "None": "null",
             "TRUE": "true", "FALSE": "false", "NULL": "null", "none": "null", "nil": "null", "undefined": "null"}


def _tokens(text):
    """Tolerant tokenizer: yields (type, value) with type in { } [ ] : , str num lit word."""
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in " \t\r\n﻿":
            i += 1
        elif c == "/" and text.startswith("//", i) or c == "#":
            j = text.find("\n", i)
            yield ("comment", None)
            i = n if j < 0 else j
        elif c == "/" and text.startswith("/*", i):
            j = text.find("*/", i + 2)
            yield ("comment", None)
            i = n if j < 0 else j + 2
        elif c in "{}[]:,":
            yield (c, c)
            i += 1
        elif c == "=" :                                   # key = value (Python / TOML habits)
            yield (":", "=")
            i += 1
        elif c in "\"'“”‘’":
            q = {"“": "”", "‘": "’"}.get(c, c)
            closers = {q} | ({"”", "“", '"'} if c in "“”" else set())
            j, buf = i + 1, []
            while j < n and text[j] not in closers:
                if text[j] == "\\" and j + 1 < n:
                    nx = text[j + 1]
                    buf.append({"n": "\n", "t": "\t", '"': '"', "'": "'", "\\": "\\", "/": "/"}.get(nx, "\\" + nx))
                    j += 2
                    continue
                if text[j] == "\n" and q == "'":            # an apostrophe, not a string: stop at the line end
                    break
                buf.append(text[j])
                j += 1
            yield ("str", "".join(buf), c)
            i = j + 1
        else:
            m = re.match(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?(?![\w.])", text[i:])
            if m:
                yield ("num", m.group(0))
                i += len(m.group(0))
                continue
            m = re.match(r"[^\s{}\[\]:,\"']+(?:[ \t]+[^\s{}\[\]:,\"'=]+)*", text[i:])
            word = m.group(0) if m else c
            yield ("word", word.strip())
            i += max(1, len(word))


def _repair(text):
    """Turn sloppy JSON into strict JSON text. Returns (json_text, [what was repaired])."""
    notes = []
    src = text
    fence = _FENCE.search(src)
    if fence:
        src = fence.group(1)
        notes.append("removed the Markdown code fence")
    a, b = src.find("{"), src.rfind("}")
    if a > 0 and src[:a].strip() or (0 <= b < len(src) - 1 and src[b + 1:].strip()):
        if a >= 0 and b > a:
            src = src[a:b + 1]
            notes.append("ignored the text before / after the JSON object")
    out = []                       # strict tokens as strings
    stack = []                     # '{' / '['
    expect_key = []                # per object level: True when the next string is a key
    kinds = set()

    def last():
        return out[-1] if out else None

    def value_start():
        if out and last() not in ("{", "[", ",", ":"):
            if stack:
                out.append(",")
                kinds.add("added a missing comma")
        if stack and stack[-1] == "{" and expect_key and expect_key[-1]:
            pass

    toks = list(_tokens(src))
    k = 0
    while k < len(toks):
        t = toks[k]
        typ = t[0]
        if typ == "comment":
            kinds.add("removed comments")
        elif typ in "{[":
            value_start()
            out.append(typ)
            stack.append(typ)
            expect_key.append(typ == "{")
        elif typ in "}]":
            if last() == ",":
                out.pop()
                kinds.add("removed trailing commas")
            want = "}" if stack and stack[-1] == "{" else "]"
            if stack:
                if typ != want:
                    kinds.add(f"closed a bracket with {want!r}")
                out.append(want)
                stack.pop()
                expect_key.pop()
            else:
                kinds.add("dropped an extra closing bracket")
        elif typ == ",":
            if last() in ("{", "[", ",") or not out:
                kinds.add("removed an extra comma")
            else:
                out.append(",")
                if stack and stack[-1] == "{":
                    expect_key[-1] = True
        elif typ == ":":
            if t[1] == "=":
                kinds.add("changed '=' to ':'")
            out.append(":")
            if expect_key:
                expect_key[-1] = False
        else:
            is_key = bool(stack) and stack[-1] == "{" and (k + 1 < len(toks) and toks[k + 1][0] == ":")
            value_start()
            if typ == "str":
                if t[2] != '"':
                    kinds.add("changed single or curly quotes to double quotes")
                out.append(json.dumps(t[1], ensure_ascii=False))
            elif typ == "num":
                v = t[1]
                if v.startswith(".") or v.startswith("-."):
                    v = v.replace(".", "0.", 1)
                if v.endswith("."):
                    v += "0"
                if is_key:
                    out.append(json.dumps(v))
                else:
                    out.append(v)
            else:
                w = t[1]
                if not is_key and w in _LITERALS:
                    if w not in ("true", "false", "null"):
                        kinds.add("changed Python-style True / False / None to true / false / null")
                    out.append(_LITERALS[w])
                else:
                    kinds.add("quoted bare keys" if is_key else "quoted bare words")
                    out.append(json.dumps(w, ensure_ascii=False))
            if stack and stack[-1] == "{" and expect_key:
                expect_key[-1] = not is_key and expect_key[-1]
        k += 1
    while stack:
        if last() == ",":
            out.pop()
        out.append("}" if stack.pop() == "{" else "]")
        kinds.add("closed unclosed brackets")
    notes += sorted(kinds)
    return "".join(x if x not in (",", ":") else x + " " for x in out), notes


def loads(text):
    """Parse a spec from text that may be sloppy JSON. Returns (dict, [repairs made, in plain words])."""
    if not isinstance(text, str):
        raise SpecError("the spec must be text")
    t = text.strip()
    if not t:
        raise SpecError("the spec file is empty: write one JSON object, e.g. python tools/compose.py --example poster")
    try:
        data = json.loads(t)
        notes = []
    except ValueError:
        fixed, notes = _repair(t)
        try:
            data = json.loads(fixed)
        except ValueError as e:
            try:                                           # two objects one after the other: keep the first
                data, end = json.JSONDecoder().raw_decode(fixed)
                notes.append("kept the first JSON object and ignored the text after it")
            except ValueError:
                if re.search(r"^\s*\w+\s*:\s*\S", t, re.M) and "{" not in t:
                    raise SpecError("this looks like YAML; a spec is JSON: one object that starts with { and ends "
                                    "with }, keys and texts in double quotes")
                raise SpecError(f"not valid JSON even after repairs ({str(e).split(':')[0]}): a spec is one object "
                                "that starts with { and ends with }")
    if isinstance(data, list) and data and all(isinstance(x, dict) for x in data):
        if len(data) == 1:
            data = data[0]
            notes.append("unwrapped the one-element list around the spec")
        else:
            data = {"layers": data}
            notes.append("treated the top-level list as the layers")
    if not isinstance(data, dict):
        if re.search(r"^\s*[A-Za-z_]\w*\s*:\s*\S", t, re.M) and not t.lstrip().startswith(("{", "[")):
            raise SpecError("this looks like YAML; a spec is JSON: one object that starts with { and ends "
                            "with }, keys and texts in double quotes")
        raise SpecError("the spec must be one JSON object { ... }")
    known = set(TOP_KEYS) | set(SECTIONS) | {"layers"}
    if len(data) == 1:                                 # {"spec": {...}} / {"pcscene": {...}}: one wrapper too many
        (k, v), = data.items()
        if isinstance(v, dict) and k not in ("layers",) + SECTIONS and (set(v) & known):
            data = v
            notes.append(f"unwrapped the spec from {k!r}")
    return data, notes


def load(path):
    """Read and parse a spec file. Returns (dict, repairs)."""
    with open(path, encoding="utf-8-sig") as f:
        return loads(f.read())


def dumps(spec):
    """Spec dict -> tidy JSON text (layers one per line, so a person or a model can read it)."""
    def short(v):
        return json.dumps(v, ensure_ascii=False)
    lines = ["{"]
    keys = list(spec)
    for i, k in enumerate(keys):
        v = spec[k]
        comma = "," if i < len(keys) - 1 else ""
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            lines.append(f"  {short(k)}: [")
            for j, x in enumerate(v):
                lines.append(f"    {short(x)}" + ("," if j < len(v) - 1 else ""))
            lines.append(f"  ]{comma}")
        elif isinstance(v, dict) and len(short(v)) > 100:
            lines.append(f"  {short(k)}: {{")
            sub = list(v)
            for j, kk in enumerate(sub):
                vv = v[kk]
                if isinstance(vv, list) and len(short(vv)) > 90 and all(not isinstance(x, (dict, list)) or len(short(x)) < 90 for x in vv):
                    lines.append(f"    {short(kk)}: [")
                    for q, x in enumerate(vv):
                        lines.append(f"      {short(x)}" + ("," if q < len(vv) - 1 else ""))
                    lines.append("    ]" + ("," if j < len(sub) - 1 else ""))
                else:
                    lines.append(f"    {short(kk)}: {short(vv)}" + ("," if j < len(sub) - 1 else ""))
            lines.append(f"  }}{comma}")
        else:
            lines.append(f"  {short(k)}: {short(v)}{comma}")
    lines.append("}")
    return "\n".join(lines) + "\n"


# ============================================================================================= colours
_MODS = {"deep": -0.35, "dark": -0.4, "darker": -0.5, "light": 0.4, "pale": 0.55, "soft": 0.3, "bright": 0.0,
         "muted": 0.15, "dusty": 0.15, "warm": 0.0, "cool": 0.0, "vivid": 0.0, "rich": -0.2, "faded": 0.35,
         "pastel": 0.5}
_EXTRA_COLORS = {"ginger": "#d9772b", "rust": "#b7472a", "sand": "#d8c08f", "forest green": "#2f5d3a",
                 "sky blue": "#87ceeb", "midnight": "#141b3a", "charcoal": "#33363b", "cream": "#f3ead6",
                 "off white": "#f4f1ea", "brass": "#c9a24a", "copper": "#b87333", "bronze": "#a97142",
                 "sage": "#9caf88", "sage green": "#9caf88", "mint": "#a8e6cf", "peach": "#ffcba4",
                 "coral": "#ff7f50", "blush": "#e8b4b8", "ochre": "#cc8a2e", "olive": "#708238",
                 "wine": "#722f37", "plum": "#8e4585", "lavender": "#b8a6d9", "royal blue": "#2f56c8",
                 "deep blue": "#123a7a", "dark blue": "#13284f", "night blue": "#101a3a", "golden": "#d4a937",
                 "gold": "#d4a937", "silver": "#c0c4cc", "brown": "#7a5236", "tan": "#d2b48c", "beige": "#e6d8bd",
                 "terracotta": "#c8664a", "mustard": "#d8a92b", "teal": "#2a8c8c", "navy": "#1d2a5a",
                 "grey": "#8a8f98", "gray": "#8a8f98", "white": "#f6f4ef", "black": "#16171a"}


def _hex(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in c[:3])


def _base_color(word):
    from .color import rgb
    w = word.strip().lower().replace("_", " ").replace("-", " ")
    if w in _EXTRA_COLORS:
        return np.asarray(rgb(_EXTRA_COLORS[w]), float)
    try:
        return np.asarray(rgb(w.replace(" ", "")), float)
    except Exception:
        pass
    try:
        from PIL import ImageColor
        return np.asarray(ImageColor.getrgb(w.replace(" ", ""))[:3], float) / 255.0
    except Exception:
        pass
    try:
        from . import catalog
        cs = catalog._colors()
        if w in cs:
            v = cs[w]
            return np.asarray(rgb(v if isinstance(v, str) else v[0]), float)
    except Exception:
        pass
    return None


def parse_color(value):
    """A colour from a CSS name, '#hex', 'deep blue', 'soft sage green', [r, g, b] (0-1 or 0-255).
    Returns '#rrggbb' or None when it is not a colour."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) in (3, 4) and all(isinstance(x, (int, float)) for x in value):
        v = np.asarray(value[:3], float)
        if v.max() > 1.0:
            v = v / 255.0
        return _hex(v)
    if not isinstance(value, str):
        return None
    s = value.strip().lower()
    if re.fullmatch(r"#?[0-9a-f]{3}|#?[0-9a-f]{6}|#?[0-9a-f]{8}", s):
        s = s if s.startswith("#") else "#" + s
        from .color import rgb
        return _hex(rgb(s[:7] if len(s) > 7 else s))
    s = s.replace("colour", "").replace("color", "").strip()
    c = _base_color(s)
    if c is not None:
        return _hex(c)
    words = s.replace("-", " ").split()
    mod = 0.0
    rest = []
    for w in words:
        if w in _MODS and len(words) > 1:
            mod += _MODS[w]
        elif w not in ("and", "ish", "toned", "tone"):
            rest.append(w)
    for i in range(len(rest)):                         # longest colour phrase from the left: "sage green"
        c = _base_color(" ".join(rest[i:]))
        if c is not None:
            break
    else:
        c = None
        for w in rest:
            c = _base_color(w)
            if c is not None:
                break
    if c is None:
        return None
    if mod > 0:
        c = c + (1 - c) * min(mod, 0.8)
    elif mod < 0:
        c = c * (1 + max(mod, -0.7))
    return _hex(c)


# ============================================================================================= the drawable names
class Thing:
    """A drawable name: which kit draws it and how it sits in a picture."""

    __slots__ = ("name", "kit", "target", "cat", "size", "aspect", "summary", "options")

    def __init__(self, name, kit, target, cat, size, aspect=1.0, summary="", options=()):
        self.name, self.kit, self.target, self.cat = name, kit, target, cat
        self.size, self.aspect, self.summary, self.options = size, aspect, summary, tuple(options)

    def __repr__(self):
        return f"Thing({self.name!r}, {self.kit}:{self.target}, {self.cat})"


# still-life objects drawn with pixelcanvas.form: name -> (function, size, options)
FORM_OBJECTS = {
    "orange": ("ball", 0.12, dict(base="#f07a18", material="fruit")),
    "apple": ("ball", 0.13, dict(base="#c8302a", material="fruit")),
    "green apple": ("ball", 0.1, dict(base="#8cbf3f", material="fruit")),
    "lemon": ("ball", 0.1, dict(base="#f2c84b", material="fruit", stretch=1.35)),
    "lime": ("ball", 0.07, dict(base="#7cb342", material="fruit")),
    "peach": ("ball", 0.12, dict(base="#f6a46b", material="fruit")),
    "plum": ("ball", 0.07, dict(base="#6b2f5a", material="fruit")),
    "tomato": ("ball", 0.08, dict(base="#d63a2a", material="fruit")),
    "egg": ("ball", 0.07, dict(base="#f4ede2", material="ceramic", stretch=0.78)),
    "ball": ("ball", 0.12, dict(base="#2f6fb8", material="plastic")),
    "pearl": ("ball", 0.04, dict(base="#f1ece4", material="glossy")),
    "mug": ("cylinder", 0.24, dict(base="#e9e1d3", material="ceramic", handle=True, open=True, fill=0.85,
                                    fill_color="#3b1f12")),
    "cup": ("cylinder", 0.2, dict(base="#f2efe9", material="ceramic", handle=True, open=True, fill=0.8,
                                    fill_color="#5a3320")),
    "glass": ("cylinder", 0.26, dict(base="#d8eef0", material="glass", open=True)),
    "can": ("cylinder", 0.18, dict(base="#c8302a", material="metal")),
    "candle": ("cylinder", 0.22, dict(base="#f3ead6", material="wax")),
    "vase": ("lathe", 0.4, dict(base="#3f6f8f", material="ceramic",
                                profile=[(0.3, 0), (0.42, 0.25), (0.38, 0.55), (0.18, 0.85), (0.22, 1.0)])),
    "bottle": ("lathe", 0.46, dict(base="#5f9a72", material="glass",
                                   profile=[(0.24, 0), (0.26, 0.05), (0.26, 0.6), (0.1, 0.8), (0.08, 1.0)])),
    "jar": ("lathe", 0.2, dict(base="#d8e6dc", material="glass",
                               profile=[(0.4, 0), (0.44, 0.08), (0.44, 0.85), (0.36, 0.92), (0.36, 1.0)])),
    "bowl": ("lathe", 0.11, dict(base="#e7dfd0", material="ceramic", open=True,
                                 profile=[(0.35, 0), (0.6, 0.25), (0.95, 1.0)])),
    "pot": ("lathe", 0.16, dict(base="#c8664a", material="ceramic", open=True,
                                profile=[(0.32, 0), (0.42, 0.9), (0.46, 1.0)])),
    "book": ("cuboid", 0.05, dict(base="#a33a2a", material="matte", dims=(0.28, 0.2, 1.0))),
    "box": ("cuboid", 0.14, dict(base="#c58b4c", material="matte", dims=(1.0, 0.8, 1.0))),
    "gift": ("cuboid", 0.14, dict(base="#c8284f", material="matte", dims=(1.0, 0.9, 1.0))),
    "cube": ("cuboid", 0.12, dict(base="#e0b44a", material="matte", dims=(1.0, 1.0, 1.0))),
    "block": ("cuboid", 0.1, dict(base="#e0b44a", material="matte", dims=(1.0, 1.0, 1.0))),
    "donut": ("torus", 0.06, dict(base="#d9a066", material="matte")),
    "ring": ("torus", 0.04, dict(base="#d4a937", material="metal")),
}
FORM_ALIASES = {"oranges": "orange", "coffee": "mug", "coffee mug": "mug", "teacup": "cup", "tea cup": "cup",
                "wine bottle": "bottle", "flower vase": "vase", "tin": "can", "present": "gift", "parcel": "box",
                "crate": "box", "doughnut": "donut", "plate": "bowl"}

# elements: category and how ``size`` (fraction of the height) maps to the kit's ``size`` px at 1080
_FULL_FRAME = {"rain", "snow", "fog", "stars", "aurora", "god rays", "dust motes", "bokeh", "heat haze",
               "volumetric fog", "caustics", "underwater", "wet ground", "lens flare", "cloud bands", "cirrus",
               "storm clouds", "snow cap", "droplets", "sky", "ocean", "lake", "river", "fireworks", "embers",
               "sparks"}
_BACK_FRAME = {"sky", "aurora", "stars", "cloud bands", "cirrus", "storm clouds", "underwater", "ocean", "lake"}
_GROUND_FX = {"campfire", "fireplace", "torch", "lamp", "lantern", "candle", "waterfall", "puddle", "splash",
              "smoke", "chimney smoke", "steam", "flame", "explosion", "ripples", "breaking wave"}
_SKY_FX = {"sun", "moon", "cumulus", "shooting star", "rainbow", "lightning", "glow", "spotlight", "neon",
           "window light"}
_FX_ALIASES = {"cloud": "cumulus", "clouds": "cumulus", "fluffy cloud": "cumulus", "fire": "campfire",
               "bonfire": "campfire", "starry sky": "stars", "full moon": "moon", "crescent moon": "moon",
               "sunshine": "sun", "snowfall": "snow", "mist": "fog", "light rays": "god rays",
               "sun rays": "god rays", "thunder": "lightning", "firework": "fireworks"}

_PEOPLE_WORDS = ("person", "man", "woman", "boy", "girl", "child", "kid", "baby", "lady", "guy", "people",
                 "friend", "worker", "player", "singer", "dancer", "teacher", "farmer", "nurse", "pilot",
                 "saxophonist", "pianist", "guitarist", "drummer", "couple", "mother", "father", "family",
                 "old man", "old woman", "grandmother", "grandfather", "student", "chef", "doctor")
_BIG_ANIMALS = {"horse", "cow", "elephant", "giraffe", "bear", "lion", "tiger", "deer", "zebra", "whale", "shark",
                "dolphin", "wolf"}
_SMALL_ANIMALS = {"bee", "butterfly", "dragonfly", "ladybird", "snail", "spider", "mouse", "frog", "lizard"}

_REG = None
_ALIAS = None


def _norm(name):
    s = str(name).strip().lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^(a|an|the|some|one)\s+", "", s)
    return s


def _singular(w):
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "xes", "sses")):
        return w[:-2]
    if w.endswith("ves") and len(w) > 4:
        return w[:-3] + "f"
    if w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3:
        return w[:-1]
    return w


def _build_registry():
    reg, alias = {}, {}

    def add(thing, *aliases):
        if thing.name in reg and reg[thing.name] is not thing and reg[thing.name].kit != thing.kit:
            word = {"plant": "tree" if thing.size >= 0.4 else "plant", "rock": "rock", "animal": "animal",
                    "bird": "bird", "person": "person", "object": "object", "icon": "icon",
                    "pictogram": "pictogram"}.get(thing.cat, thing.kit)
            thing.name = f"{thing.name} {word}"
        if thing.name not in reg:
            reg[thing.name] = thing
        for a in aliases:
            a = _norm(a)
            if a and a not in reg and a not in alias:
                alias[a] = thing.name

    try:
        from . import creature as C
        groups = getattr(C, "_KINDS", {})
        for sp in C.species():
            g = groups.get(sp, "quad")
            if g == "bird" or sp in ("butterfly", "bee", "dragonfly"):
                cat, size, asp = "bird", 0.06, 1.3
            elif g == "sea":
                cat, size, asp = "animal", 0.2, 1.6
            elif sp in _SMALL_ANIMALS or g == "small":
                cat, size, asp = "animal", 0.05, 1.2
            else:
                cat, size, asp = "animal", 0.4 if sp in _BIG_ANIMALS else 0.24, 1.35
            add(Thing(sp, "creature", sp, cat, size, asp, f"{sp} (animal)",
                      ("pose", "coat", "facing", "view", "style", "head_turn", "tail", "ears", "mouth", "eyes")))
        for dname in ("_ALIASES", "_BIRD_ALIASES", "_SEA_ALIASES", "_SMALL_ALIASES"):
            for a, t in getattr(C, dname, {}).items():
                if t in reg:
                    add(reg[t], a)
        add(reg.get("sparrow") or next(iter(reg.values())), "bird", "songbird", "little bird")
        if "dog" in reg:
            add(reg["dog"], "puppy", "doggy")
        if "cat" in reg:
            add(reg["cat"], "kitten", "kitty")
    except Exception:
        pass
    try:
        from . import figure as Fg
        for p in getattr(Fg, "PEOPLE", {}):
            add(Thing(p.replace("_", " "), "figure", p, "person", 0.5, 0.36, f"{p.replace('_', ' ')} (person)",
                      ("pose", "describe", "facing", "view", "style", "expression", "item")))
        add(Thing("person", "figure", None, "person", 0.5, 0.36, "any person: say who in \"describe\"",
                  ("pose", "describe", "facing", "view", "style", "expression", "item")), *_PEOPLE_WORDS)
    except Exception:
        pass
    try:
        from . import form  # noqa: F401
        for n, (fn, size, opt) in FORM_OBJECTS.items():
            asp = {"ball": 1.0, "cylinder": 0.7, "lathe": 0.6, "cuboid": 1.4, "torus": 2.0}.get(fn, 1.0)
            add(Thing(n, "form", n, "object", size, asp, f"{n} (object with volume)", ("material",)))
        for a, t in FORM_ALIASES.items():
            if t in reg:
                add(reg[t], a)
    except Exception:
        pass
    try:
        from . import flora as Fl
        sub = getattr(Fl, "_REG", {})
        for n in Fl.names():
            kind = sub.get(n, (None, "plant"))[1]
            size = {"tree": 0.62, "shrub": 0.2, "ground": 0.1, "flower": 0.12, "potted": 0.25, "leaf": 0.15,
                    "rock": 0.1}.get(kind, 0.3)
            asp = {"tree": 0.8, "shrub": 1.4, "ground": 2.0, "flower": 0.35, "potted": 0.6, "rock": 1.6}.get(kind, 0.8)
            cat = "rock" if kind == "rock" else "plant"
            add(Thing(n, "flora", n, cat, size, asp, f"{n} ({kind})", ("season", "light", "shadow", "moss", "kind",
                                                                        "bloom", "leaves", "palette", "wind")))
        for a, t in getattr(Fl, "_ALIASES", {}).items():
            if t in reg:
                add(reg[t], a)
        for a, t in (("tree", "oak"), ("oak tree", "oak"), ("flower", "daisy"), ("houseplant", "potted monstera"),
                     ("potted plant", "potted monstera"), ("plant", "potted monstera"), ("stone", "rock"),
                     ("boulder", "rock"), ("pine tree", "pine"), ("palm tree", "palm"), ("cactus", "saguaro"),
                     ("cherry tree", "cherry"), ("willow tree", "willow"),
                     ("birch tree", "birch"), ("maple tree", "maple"), ("shrub", "bush"), ("roses", "rose")):
            if t in reg:
                add(reg[t], a)
    except Exception:
        pass
    try:
        from . import elements as E
        for n in E.names():
            if n in _FULL_FRAME:
                cat = "fullframe"
            elif n in _GROUND_FX:
                cat = "effect"
            else:
                cat = "sky"
            size = {"sun": 0.12, "moon": 0.16, "cumulus": 0.2, "rainbow": 0.6, "lightning": 0.6, "campfire": 0.22,
                    "candle": 0.15, "lamp": 0.3, "lantern": 0.12}.get(n, 0.25)
            asp = {"cumulus": 2.2, "rainbow": 1.8}.get(n, 1.0)
            add(Thing(n, "elements", n, cat, size, asp, f"{n} (effect)", ()))
        for a, t in _FX_ALIASES.items():
            if t in reg:
                add(reg[t], a)
    except Exception:
        pass
    try:
        P = importlib.import_module("pixelcanvas.pictograms")
        for n in P.names():
            add(Thing(n, "pictogram", n, "pictogram", 0.3, 1.0, f"{n} (pictogram)",
                      ("style", "accent", "rotate", "flip", "detail_color")))
        for n in P.names():
            try:
                for a in (P.info(n) or {}).get("aliases", ()):
                    if re.match(r"^[\x00-\x7f]+$", str(a)):
                        add(reg[n], a)
            except Exception:
                pass
    except Exception:
        pass
    try:
        from . import vector as V
        for n in V.icon_names():
            nn = n.replace("-", " ")
            th = Thing(f"{nn} icon", "icon", n, "icon", 0.18, 1.0, f"{nn} (line icon)", ("stroke_width", "fill"))
            add(th, f"icon {nn}", f"icon:{nn}")
            if nn not in reg and nn not in alias:
                alias[nn] = th.name
    except Exception:
        pass
    return reg, alias


def _registry():
    global _REG, _ALIAS
    if _REG is None:
        _REG, _ALIAS = _build_registry()
    return _REG, _ALIAS


def names(query=None, n=40):
    """Every drawable name (or those matching ``query``) as [(name, summary)]."""
    reg, alias = _registry()
    if not query:
        return [(k, t.summary) for k, t in reg.items()]
    q = _norm(query)
    toks = q.split()
    out, seen = [], set()

    def push(name, how=""):
        if name in seen or name not in reg:
            return
        seen.add(name)
        out.append((name, reg[name].summary + how))

    t0 = lookup(q)[0]
    if t0 is not None:
        push(t0.name, f"  (best match for {q!r})" if t0.name != q else "")

    if q in reg:
        push(q)
    if q in alias:
        push(alias[q], f"  (also called {q!r})")
    for k in reg:
        if q in k or (toks and all(t in k for t in toks)):
            push(k)
    for a, k in alias.items():
        if q in a:
            push(k, f"  (also called {a!r})")
    for k, t in reg.items():
        if any(tok in t.summary.lower() for tok in toks if len(tok) > 3):
            push(k)
    for k in difflib.get_close_matches(q, list(reg) + list(alias), n=6, cutoff=0.6):
        push(alias.get(k, k))
    if q.endswith("s"):
        for name, s in names(_singular(q), n):
            push(name)
    return out[:n]


def lookup(name, kind=None):
    """Resolve a ``draw`` name. Returns (Thing or None, extras dict, [suggestions], note).

    ``extras`` carries words split off the name ("ginger cat" -> coat ginger; "red apple" -> color)."""
    reg, alias = _registry()
    raw = str(name)
    prefer_icons = kind == "icons"
    s = _norm(raw)
    s = re.sub(r"\s*#?\d+$", "", s) or s                      # "bird 2" -> "bird"
    if ":" in s:
        kit, _, rest = s.partition(":")
        rest = rest.strip()
        kit = kit.strip()
        if kit in ("icon", "icons"):
            t = reg.get(f"{rest} icon")
            return (t, {}, [], "") if t else (None, {}, _suggest(rest), "")
        for t in reg.values():
            if t.kit.startswith(kit) and t.name == rest:
                return t, {}, [], ""
        s = rest
    if prefer_icons and f"{s} icon" in reg:
        return reg[f"{s} icon"], {}, [], ""
    for cand in (s, _singular(s)):
        if cand in reg:
            t = reg[cand]
            if t.kit in ("pictogram", "icon") and cand in alias and reg[alias[cand]].kit not in ("pictogram", "icon"):
                t = reg[alias[cand]]                     # a painted kit wins over a flat pictogram ("bird")
            return t, {}, [], ("" if cand == s else f"read {raw!r} as {cand!r}")
    for cand in (_singular(s), s) if _singular(s) != s and _singular(s) in alias else (s, _singular(s)):
        if cand in alias:
            return reg[alias[cand]], {}, [], ""
    words = s.split()
    for i in range(1, len(words)):                     # "ginger cat", "big old oak tree", "red apple"
        tail = " ".join(words[i:])
        for cand in (tail, _singular(tail)):
            t = reg.get(cand) or (reg.get(alias[cand]) if cand in alias else None)
            if t is None:
                continue
            head = " ".join(words[:i])
            extras = _split_attributes(t, head)
            if extras is not None:
                return t, extras, [], f"read {raw!r} as {t.name!r} with {', '.join(f'{k} {v!r}' for k, v in extras.items()) or 'no extra options'}"
    for i in range(1, len(words)):                     # "coffee cup" -> cup (the extra words are reported)
        tail = " ".join(words[i:])
        for cand in (tail, _singular(tail)):
            t = reg.get(cand) or (reg.get(alias[cand]) if cand in alias else None)
            if t is not None and t.kit != "icon":
                return t, {}, [], f"drew {raw!r} as {t.name!r} (no kit draws {raw!r}; ignored {' '.join(words[:i])!r})"
    return None, {}, _suggest(s), ""


_ATTR_WORDS = {"big": ("size", "large"), "large": ("size", "large"), "small": ("size", "small"),
               "little": ("size", "small"), "tiny": ("size", "tiny"), "huge": ("size", "huge"),
               "giant": ("size", "huge"), "tall": ("size", "large"), "old": None, "young": None, "cute": None,
               "fluffy": None, "pretty": None, "lovely": None, "beautiful": None, "wild": None, "flying": ("pose", "fly"),
               "sitting": ("pose", "sit"), "sleeping": ("pose", "sleep"), "running": ("pose", "run"),
               "standing": ("pose", "stand"), "walking": ("pose", "walk"), "lying": ("pose", "lie"),
               "perched": ("pose", "perch"), "swimming": ("pose", "swim"), "jumping": ("pose", "jump"),
               "waving": ("pose", "wave"), "dancing": ("pose", "dance")}


def _split_attributes(thing, head):
    """Words before a name -> options, or None when a word is not understood."""
    out = {}
    coats = ()
    if thing.kit == "creature":
        try:
            from . import creature as C
            coats = tuple(getattr(C, "COATS", {}))
        except Exception:
            coats = ()
    if head in coats:
        return {"coat": head}
    words = head.split()
    rest = []
    for w in words:
        if w in coats:
            out["coat"] = w
        elif w in _ATTR_WORDS:
            if _ATTR_WORDS[w]:
                k, v = _ATTR_WORDS[w]
                if k == "pose" and thing.kit not in ("creature", "figure"):
                    continue
                out[k] = v
        else:
            rest.append(w)
    if rest:
        col = parse_color(" ".join(rest))
        if col is None:
            if thing.kit == "figure":
                out["describe"] = head
                return out
            if thing.kit == "flora" and len(rest) == 1:
                return None
            return None
        out["coat" if thing.kit == "creature" and " ".join(rest) in coats else "color"] = " ".join(rest)
    return out


def _suggest(s, n=5):
    reg, alias = _registry()
    pool = list(reg) + list(alias)
    close = difflib.get_close_matches(s, pool, n=n, cutoff=0.55)
    words = [w for w in s.split() if len(w) > 2]
    for w in words:
        close += [k for k in difflib.get_close_matches(w, pool, n=3, cutoff=0.7)]
        close += [k for k in pool if k.startswith(w)][:2]
    out = []
    for c in close:
        c = alias.get(c, c)
        if c not in out:
            out.append(c)
    return out[:n]


# ============================================================================================= placement words
H_WORDS = {"far left": 0.1, "left edge": 0.08, "left": 0.24, "left third": 0.33, "centre": 0.5, "center": 0.5,
           "middle": 0.5, "right third": 0.67, "right": 0.76, "far right": 0.9, "right edge": 0.92}
V_WORDS = {"very top": 0.08, "top": 0.16, "upper": 0.3, "top third": 0.3, "upper third": 0.3, "centre": 0.5,
           "center": 0.5, "middle": 0.5, "lower": 0.68, "lower third": 0.7, "bottom third": 0.72, "bottom": 0.86,
           "very bottom": 0.93}
DEPTH_WORDS = {"far": 0.12, "far away": 0.08, "distance": 0.1, "in the distance": 0.1, "distant": 0.1,
               "horizon": 0.04, "on the horizon": 0.04, "background": 0.15, "back": 0.2, "middle ground": 0.5,
               "midground": 0.5, "near": 0.8, "front": 0.85, "foreground": 0.88, "close": 0.85,
               "close up": 0.9}
SKY_WORDS = ("in the sky", "sky", "high", "overhead", "in the air", "air", "above everything", "flying")
GROUND_WORDS = ("on the ground", "ground", "on the floor", "floor", "on the grass", "grass", "on the field",
                "standing", "on the beach", "beach")
REL_WORDS = ("in front of", "on top of", "to the left of", "to the right of", "left of", "right of", "next to",
             "beside", "behind", "under", "underneath", "below", "beneath", "above", "over", "on", "near", "by",
             "inside", "in")
REL_CANON = {"on top of": "on", "to the left of": "left of", "to the right of": "right of", "beside": "next to",
             "by": "next to", "near": "next to", "underneath": "under", "below": "under", "beneath": "under",
             "over": "above", "inside": "in"}
_FILLER = {"the", "a", "of", "side", "corner", "edge", "area", "part", "ish", "slightly", "bit", "little", "at",
           "somewhere", "around", "towards", "toward", "to", "position", "placed", "place", "and", "half",
           "screen", "canvas", "picture", "image", "page", "frame", "in", "on"}
ARRANGE = ("row", "cluster", "scatter", "column", "circle", "grid")
PLACEMENT_HELP = ("a word (centre, left, top right, bottom third, far, foreground, in the sky, on the ground), "
                  "a relation (left of cat, on table, under tree, behind tree, in front of house), "
                  "[u, v] fractions or a box [u0, v0, u1, v1]")
_NOT_TARGETS = ("the sky", "sky", "the air", "air", "the distance", "distance", "front", "the middle", "middle",
                "the centre", "the center", "centre", "center", "the foreground", "foreground", "the background",
                "background", "the ground", "ground", "the floor", "floor", "the grass", "grass", "the left",
                "the right", "left", "right", "top", "the top", "bottom", "the bottom", "the field", "field",
                "the beach", "beach", "the horizon", "horizon")


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _place_words_only(text):
    """True when every word is a placement word ("sky top left"), so it names no layer."""
    vocab = set(_FILLER)
    for table in (H_WORDS, V_WORDS, DEPTH_WORDS, {w: 1 for w in SKY_WORDS}, {w: 1 for w in GROUND_WORDS}):
        for k in table:
            vocab |= set(k.split())
    return all(w in vocab for w in text.split())


def parse_at(value):
    """A placement value -> (dict, error or None). The dict has ``mode`` point / box / words / auto / pixels."""
    if value is None or value == "" or value == "auto":
        return {"mode": "auto"}, None
    if isinstance(value, dict):
        lower = {str(k).lower(): v for k, v in value.items()}
        for a, b in (("x", "y"), ("u", "v")):
            if a in lower and b in lower and _num(lower[a]) and _num(lower[b]):
                return parse_at([lower[a], lower[b]])
        return None, "use [u, v] fractions or words, not an object"
    if isinstance(value, str) and re.fullmatch(r"\s*[\[(]?\s*-?[\d.]+\s*[, ]\s*-?[\d.]+(\s*[, ]\s*-?[\d.]+\s*[, ]\s*-?[\d.]+)?\s*[\])]?\s*", value):
        value = [float(x) for x in re.findall(r"-?[\d.]+", value)]
    if isinstance(value, (list, tuple)):
        if all(_num(x) for x in value) and len(value) in (2, 4):
            vals = [float(x) for x in value]
            note = None
            if max(abs(x) for x in vals) > 1.6:
                if max(abs(x) for x in vals) <= 100:
                    vals = [x / 100 for x in vals]
                    note = "read the numbers as percent"
                else:
                    return {"mode": "pixels", "values": vals}, None
            d = {"mode": "point" if len(vals) == 2 else "box", "values": vals}
            if note:
                d["note"] = note
            if len(vals) == 4:
                u0, v0, u1, v1 = vals
                if u1 <= u0 or v1 <= v0:
                    if u1 > 0 and v1 > 0 and u0 + u1 <= 1.05 and v0 + v1 <= 1.05:
                        d["values"] = [u0, v0, u0 + u1, v0 + v1]
                        d["note"] = "read [u, v, width, height] as a box"
                    else:
                        return None, "a box is [u0, v0, u1, v1] with u1 > u0 and v1 > v0"
            return d, None
        if len(value) == 1:
            return parse_at(value[0])
        if all(isinstance(x, str) for x in value):
            return parse_at(" ".join(value))
        return None, "use [u, v] (two numbers 0-1) or a box [u0, v0, u1, v1]"
    if not isinstance(value, str):
        return None, "not understood"
    text = _norm(value).replace(",", " ").replace(".", " ").replace("/", " ")
    text = re.sub(r"\bish\b|-ish", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # "held by sam", "in sam's hands", "with the girl": a thing someone holds stands next to them
    text = re.sub(r"^(?:held by|carried by|played by|holding by|in the hands of|in the arms of|with)\s+(.+)$",
                  r"next to \1", text)
    text = re.sub(r"^in (?:the )?(.+?)'?s? (?:hands?|arms|lap|paws)$", r"next to \1", text)
    d = {"mode": "words", "text": text}
    for r in sorted(REL_WORDS, key=len, reverse=True):
        m = re.search(rf"(?:^|\s){re.escape(r)}\s+(.+)$", text)
        if not m:
            continue
        target = re.sub(r"^(the|a|an)\s+", "", m.group(1)).strip()
        full = m.group(1).strip()
        if full in _NOT_TARGETS or target in _NOT_TARGETS or target.split()[0] in H_WORDS \
                or target.split()[0] in ("top", "bottom", "upper", "lower", "sky", "air") or _place_words_only(target):
            continue
        d["rel"] = REL_CANON.get(r, r)
        d["target"] = target
        text = text[:m.start()].strip()
        break
    words = " " + text + " "
    found = {}
    centre = 0
    for group, table in (("sky", {w: 1 for w in SKY_WORDS}), ("ground", {w: 1 for w in GROUND_WORDS}),
                         ("h", H_WORDS), ("v", V_WORDS), ("depth", DEPTH_WORDS)):
        for phrase in sorted(table, key=len, reverse=True):
            while True:
                m = re.search(rf"\s{re.escape(phrase)}(?=\s)", words)
                if not m:
                    break
                words = words[:m.start()] + " " + words[m.end():]
                if phrase in ("centre", "center", "middle"):
                    centre += 1
                    continue
                found.setdefault(group, table[phrase])
    left = [w for w in words.split() if w not in _FILLER]
    if left:
        return None, f"{' '.join(left)!r} not understood"
    if centre:
        if "h" not in found and "v" not in found:
            found["h"], found["v"] = 0.5, 0.5
        elif "h" not in found:
            found["h"] = 0.5
        elif "v" not in found:
            found["h"] = 0.5 + (found["h"] - 0.5) * 0.55            # "centre left" -> a little left of centre
            if centre > 1:
                found["v"] = 0.5
    d.update(found)
    if not found and "rel" not in d:
        return {"mode": "auto"}, None
    return d, None


def parse_size(value):
    """A size -> (fraction of the canvas height or None for the default, error, note)."""
    if value is None or value == "" or value == "auto" or value == "default":
        return None, None, None
    if isinstance(value, bool):
        return None, "use a word (tiny small medium large huge) or a number 0-1", None
    if _num(value):
        v = float(value)
        if v <= 0:
            return None, "must be more than 0", None
        if v <= 1.6:
            return v, None, None
        if v <= 100:
            return v / 100, None, "read the number as percent of the height"
        return None, "use a fraction of the canvas height (0.3 = 30%)", None
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(_num(x) for x in value):
        return parse_size(max(value))
    if isinstance(value, str):
        s = _norm(value)
        if s.endswith("%"):
            try:
                return float(s[:-1]) / 100, None, None
            except ValueError:
                pass
        try:
            return parse_size(float(s))
        except ValueError:
            pass
        if s in ("full", "full height", "whole"):
            return 0.95, None, None
        if s in SIZE_WORDS:
            return ("rel", SIZE_WORDS[s]), None, None
        s2 = s.replace("very ", "").replace("extra ", "").replace("quite ", "").replace("fairly ", "")
        if s2 in SIZE_WORDS:
            return ("rel", SIZE_WORDS[s2] * (1.25 if "very" in s or "extra" in s else 1.0)), None, None
        close = difflib.get_close_matches(s, list(SIZE_WORDS), n=1, cutoff=0.6)
        if close:
            return ("rel", SIZE_WORDS[close[0]]), None, f"read size {value!r} as {close[0]!r}"
    return None, f"{value!r} not understood: use tiny, small, medium, large, huge or a number 0-1", None


def parse_canvas_size(value, kind):
    """[w, h] / "1080x1350" / "portrait" -> ((w, h), error)."""
    default = KINDS.get(kind, KINDS["scene"])["size"]
    if value is None or value == "":
        return default, None
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(x, str) for x in value) \
            and all(re.fullmatch(r"\s*\d+(\.\d+)?\s*(px)?\s*", x) for x in value):
        value = [float(re.sub(r"[^\d.]", "", x)) for x in value]
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(_num(x) for x in value):
        w, h = int(value[0]), int(value[1])
        if 64 <= w <= 8000 and 64 <= h <= 8000:
            return (w, h), None
        return default, "width and height must be 64-8000 pixels"
    if isinstance(value, dict):
        lw = {str(k).lower(): v for k, v in value.items()}
        w, h = lw.get("width", lw.get("w")), lw.get("height", lw.get("h"))
        if _num(w) and _num(h):
            return parse_canvas_size([w, h], kind)
    if isinstance(value, str):
        s = value.strip().lower()
        m = re.fullmatch(r"(\d+)\s*(?:px)?\s*(?:[x×*,:/ ]|by)\s*(\d+)\s*(?:px|pixels)?", s)
        if m:
            return parse_canvas_size([int(m.group(1)), int(m.group(2))], kind)
        if s in FORMAT_WORDS:
            return FORMAT_WORDS[s], None
    return default, f"use [width, height] such as [1920, 1080], or a word: {', '.join(FORMAT_WORDS)}"


# ============================================================================================= validation
def _close(word, options, n=3):
    return difflib.get_close_matches(str(word).lower(), [str(o) for o in options], n=n, cutoff=0.5)


def _did_you_mean(word, options):
    c = _close(word, options)
    return f"did you mean {' or '.join(repr(x) for x in c)}?" if c else f"use one of: {', '.join(map(str, list(options)[:14]))}"


_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10, "a pair of": 2, "pair of": 2, "couple of": 2, "a couple of": 2,
                 "several": 4, "many": 7, "a few": 3, "few": 3, "a dozen": 12, "dozen": 12}
LAYER_KEYS = ("draw", "text", "python", "at", "size", "color", "count", "arrange", "role", "id", "label",
              "behind", "in_front_of", "on", "under", "left_of", "right_of", "next_to", "above", "font", "align",
              "box", "plate", "seed", "opacity", "describe", "pose", "coat", "facing", "view", "style",
              "material", "ink", "bold", "shadow", "depth")
LAYER_ALIASES = {"object": "draw", "thing": "draw", "subject": "draw", "item": "draw", "what": "draw",
                 "shape": "draw", "image": "draw", "picture": "draw", "animal": "draw", "species": "draw",
                 "plant": "draw", "effect": "draw", "icon": "draw", "pictogram": "draw", "prefab": "draw",
                 "type": "draw", "name": "draw", "content": "text", "words": "text", "string": "text",
                 "caption": "text", "title": "text", "label_text": "text", "position": "at", "pos": "at",
                 "where": "at", "place": "at", "location": "at", "placement": "at", "xy": "at", "center": "at",
                 "centre": "at", "scale": "size", "height": "size", "width": "size", "colour": "color",
                 "fill": "color", "tint": "color", "colors": "color", "colours": "color", "number": "count",
                 "quantity": "count", "n": "count", "amount": "count", "how_many": "count", "arrangement": "arrange",
                 "layout": "arrange", "pattern": "arrange", "kind": "role", "description": "describe",
                 "desc": "describe", "who": "describe", "looks": "describe", "outfit": "describe",
                 "posture": "pose", "action": "pose", "direction": "facing", "face": "facing", "fur": "coat",
                 "breed": "coat", "front_of": "in_front_of", "infront": "in_front_of", "in front of": "in_front_of",
                 "left of": "left_of", "right of": "right_of", "next to": "next_to", "beside": "next_to",
                 "near": "next_to", "below": "under", "beneath": "under", "over": "above", "on_top_of": "on",
                 "family": "font", "typeface": "font", "alignment": "align"}
REL_KEYS = {"behind": "behind", "in_front_of": "in front of", "on": "on", "under": "under", "left_of": "left of",
            "right_of": "right of", "next_to": "next to", "above": "above"}


def _resolve_kind(raw, spec, problems):
    if raw is None:
        for sec in SECTIONS:
            if sec in spec and spec[sec]:
                return SECTION_KIND[sec]
        guess = None
        try:
            from . import catalog
            guess = catalog.suggest_kind(str(spec.get("prompt") or ""))
        except Exception:
            guess = None
        guess = KIND_ALIASES.get(guess, guess) if guess else None
        if guess not in KINDS:
            lay = spec.get("layers") or []
            texty = [l for l in lay if isinstance(l, dict) and "text" in l]
            guess = "poster" if texty and len(texty) == len(lay) else "scene"
        problems.append(Problem("kind", f"missing; using {guess!r}", f'add "kind": "{guess}"', "note"))
        return guess
    k = _norm(raw)
    if k in KINDS:
        return k
    k2 = k.replace(" ", "_")
    if k2 in KINDS:
        return k2
    if k in KIND_ALIASES:
        return KIND_ALIASES[k]
    for w in k.split():
        if w in KINDS:
            return w
        if w in KIND_ALIASES:
            return KIND_ALIASES[w]
    problems.append(Problem("kind", f"unknown kind {raw!r}", _did_you_mean(raw, KINDS)))
    return "scene"


_KIND_SIGNALS = ((r"\b(mobile )?app screen|\bmobile app\b|\bwebsite\b|\blanding page\b|\blogin screen\b|\bui mock", "ui"),
                 (r"\bflow ?chart\b|\bdiagram\b|\barchitecture\b|\bprocess flow\b", "diagram"),
                 (r"\bdashboard\b", "dashboard"),
                 (r"\b(bar|line|pie|donut|area|scatter) (chart|graph)\b|\bchart of\b|\bgraph of\b", "chart"),
                 (r"\bposter\b|\bflyer\b", "poster"),
                 (r"\b(isometric )?(bedroom|living room|kitchen|room)\b", "room"))


def _kind_against_prompt(kind, prompt, problems):
    """'A mobile app screen ...' drawn as kind 'icons' is the wrong start: say which kind fits."""
    if not isinstance(prompt, str) or not prompt.strip():
        return
    head = " ".join(prompt.lower().split()[:7])
    hits = sorted((m.start(), want) for pat, want in _KIND_SIGNALS for m in [re.search(pat, head)] if m)
    for _, want in hits[:1]:                           # the first thing the request names is what it is
        if True:
            ok = {want} | ({"scene3d"} if want == "room" else set()) | ({"chart"} if want == "dashboard" else set())
            if kind not in ok and KINDS[kind]["group"] != KINDS[want]["group"]:
                problems.append(Problem("kind", f"{kind!r} does not fit the request (it asks for a {want})",
                                        f'set "kind": "{want}" and copy the shape of compose.py --example {want}'))
            return


def _canon_keys(d, path, canon, aliases, problems, notes, extra_ok=()):
    """Rename alias keys; unknown keys go to out["_unknown"]. Returns a new dict."""
    out = {}
    for k, v in d.items():
        kk = str(k).strip()
        kl = kk.lower().replace("-", "_").replace(" ", "_")
        if kl in canon or kl in extra_ok:
            key = kl
        elif kl in aliases:
            key = aliases[kl]
            notes.append(f"{path}: read {k!r} as {key!r}")
        elif kl.replace("_", " ") in aliases:
            key = aliases[kl.replace("_", " ")]
            notes.append(f"{path}: read {k!r} as {key!r}")
        else:
            out.setdefault("_unknown", {})[kk] = v
            continue
        if key in out:
            if key == kl:
                out.setdefault("_dropped", {})[key] = out[key]
            else:
                continue
        out[key] = v
    return out


def _style_name(value, problems, path="style"):
    from . import stylize
    if value in (None, "", False, "none", "None"):
        return None
    s = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    if s in stylize.STYLES:
        return s
    p = stylize.pick(str(value))
    if p:
        return p
    problems.append(Problem(path, f"unknown style {value!r}", _did_you_mean(s, stylize.STYLES)))
    return None


def _palette(value, problems):
    if value in (None, "", False):
        return None
    try:
        from . import palette as P
        if isinstance(value, (list, tuple)):
            cols = []
            for c in value:
                h = parse_color(c)
                if h is None:
                    problems.append(Problem("palette", f"{c!r} is not a colour", "use CSS names or '#rrggbb'"))
                else:
                    cols.append(h)
            return P.palette(cols) if cols else None
        return P.palette(str(value))
    except Exception as e:
        problems.append(Problem("palette", f"{value!r} not understood ({e})",
                                'use a name ("sunset", "forest"), a description ("deep blue and gold") or a list of colours'))
        return None


POST_KEYS = ("vignette", "grain", "bloom", "soft_clip", "contrast", "saturation")


def validate(spec, base_dir=None):
    """Check a spec and fill in defaults. Returns (normalised dict, [Problem]).

    Problems with severity "error" stop the drawing; "warn" and "note" lines say what was read or repaired."""
    problems, notes = [], []
    if not isinstance(spec, dict):
        return None, [Problem("spec", "must be one JSON object")]
    s = _canon_keys(spec, "spec", TOP_KEYS + TEXT_TOP, TOP_ALIASES, problems, notes)
    for k, v in (s.pop("_unknown", None) or {}).items():
        sec = _close(k, TOP_KEYS + TEXT_TOP)
        problems.append(Problem(k, "unknown field", f"did you mean {sec[0]!r}?" if sec else
                                f"top-level fields: {', '.join(TOP_KEYS)}"))
    s.pop("_dropped", None)
    kind = _resolve_kind(s.get("kind"), s, problems)
    _kind_against_prompt(kind, s.get("prompt"), problems)
    info = KINDS[kind]
    group = info["group"]
    ver = s.get("pcscene", None)
    if ver not in (None, 1, "1", 1.0):
        problems.append(Problem("pcscene", f"version {ver!r} is not supported", 'use "pcscene": 1', "note"))
    prompt = s.get("prompt")
    if prompt is not None and not isinstance(prompt, str):
        prompt = str(prompt)
    size, err = parse_canvas_size(s.get("size"), kind)
    if err:
        problems.append(Problem("size", f"{s.get('size')!r} not understood", err))
    seed = s.get("seed", 1)
    if isinstance(seed, str) and re.fullmatch(r"\s*-?\d+\s*", seed):
        seed = int(seed)
        notes.append("seed: read the text as a number")
    if not _num(seed):
        problems.append(Problem("seed", "must be a whole number", '"seed": 1'))
        seed = 1
    pal = _palette(s.get("palette"), problems)
    style_from = "spec"
    style = _style_name(s.get("style"), problems) if s.get("style") not in (None, "") else None
    if style is None and s.get("style") in (None, "") and prompt:
        try:
            from . import stylize
            style = stylize.pick(prompt)
            style_from = "prompt"
        except Exception:
            style = None
    post = s.get("post", None)
    if post not in (None, True, False) and not isinstance(post, dict):
        problems.append(Problem("post", "use an object such as {\"vignette\": 0.3, \"grain\": 0.01} or false"))
        post = None
    if isinstance(post, dict):
        for k in [k for k in post if k not in POST_KEYS]:
            problems.append(Problem(f"post.{k}", "unknown option", _did_you_mean(k, POST_KEYS)))
            post = {kk: vv for kk, vv in post.items() if kk != k}
    norm = {"kind": kind, "group": group, "check": info["check"], "prompt": prompt or "", "size": list(size),
            "seed": int(seed), "palette": pal, "palette_spec": s.get("palette"), "style": style,
            "style_from": style_from, "post": post, "theme": s.get("theme"), "notes": notes,
            "base_dir": base_dir}
    layers_in = s.get("layers")
    if layers_in is None:
        layers_in = []
    if isinstance(layers_in, dict):
        layers_in = [layers_in]
        notes.append("layers: wrapped the single layer in a list")
    if not isinstance(layers_in, list):
        problems.append(Problem("layers", "must be a list of layers [ {...}, {...} ]"))
        layers_in = []
    extra_text = []
    for k in TEXT_TOP:
        if k in s and s[k] not in (None, ""):
            role = k if k in ROLES else ROLE_ALIASES.get(k, "body")
            vals = s[k] if isinstance(s[k], list) else [s[k]]
            for v in vals:
                extra_text.append({"text": v if not isinstance(v, (list, tuple)) else " · ".join(map(str, v)),
                                   "role": role})
            notes.append(f"{k}: moved into a text layer with role {role!r}")
    norm["background"] = _validate_background(s.get("background"), group, kind, problems, notes, prompt or "")
    norm["layers"] = []
    for i, raw in enumerate(list(layers_in) + extra_text):
        path = f"layers[{i}]" if i < len(layers_in) else f"{extra_text[i - len(layers_in)]['role']}"
        L = _validate_layer(raw, path, i, kind, group, problems, notes, base_dir)
        if L is not None:
            norm["layers"].append(L)
    _design_water(norm, group, notes)
    _unique_labels(norm["layers"])
    _check_targets(norm["layers"], problems)
    _guitar_player(norm, notes)
    norm["group"] = norm.get("group", group)
    _apply_prompt_relations(norm, notes)
    _apply_prompt_poses(norm, notes)
    for sec in SECTIONS:
        if sec in s and s[sec] not in (None, False, {}, []):
            want = SECTION_KIND[sec]
            if sec == "scene3d" and kind in ("room", "scene3d"):
                pass
            elif kind != want and not (sec == "chart" and kind == "dashboard"):
                problems.append(Problem(sec, f"a {sec!r} section belongs to kind {want!r}, not {kind!r}",
                                        f'set "kind": "{want}" or remove "{sec}"'))
                continue
            norm[sec] = _validate_section(sec, s[sec], problems, notes, kind, norm)
    need = {"diagram": "diagram", "chart": "chart", "dashboard": "dashboard", "ui": "ui", "scene3d": "scene3d",
            "room": "scene3d"}.get(kind)
    if need and need not in norm:
        things3d = [L for L in norm["layers"] if L["type"] == "thing3d"]
        if kind == "dashboard" and "chart" in norm:
            norm["dashboard"] = _validate_section("dashboard", {"charts": [s.get("chart")]}, problems, notes, kind, norm)
        elif need == "scene3d" and things3d:
            objs = [dict(L["raw"]) for L in things3d]
            norm["scene3d"] = _validate_section("scene3d", {"room": kind == "room", "objects": objs}, problems,
                                                notes, kind, norm)
            notes.append(f"layers: drew the {len(objs)} layers as scene3d objects")
        elif kind == "room":
            norm["scene3d"] = _validate_section("scene3d", {"room": True, "objects": []}, problems, notes, kind, norm)
            problems.append(Problem("scene3d", "missing: the room is empty",
                                    'add "scene3d": {"room": {}, "objects": [{"draw": "bed"}, ...]} '
                                    "(see compose.py --example room)"))
        else:
            ex = json.dumps(EXAMPLES.get(kind, {}).get(need, {}), ensure_ascii=False)[:160]
            problems.append(Problem(need, f"kind {kind!r} needs a {need!r} section", f'add "{need}": {ex}'))
    norm["layers"] = [L for L in norm["layers"] if L["type"] != "thing3d"]
    if group in ("diagram", "chart", "dashboard", "ui"):
        things = [L for L in norm["layers"] if L["type"] == "thing"]
        if things:
            problems.append(Problem("layers", f"kind {kind!r} is drawn from its {need!r} section; "
                                    f"{len(things)} picture layers are left out", "remove them or change the kind",
                                    "warn"))
            norm["layers"] = [L for L in norm["layers"] if L["type"] != "thing"]
    had_layers = bool(s.get("layers"))
    if group in ("design", "scene", "still_life") and not norm["layers"] and not had_layers:
        ex = json.dumps({"text": "Your Title", "role": "title"} if kind in ("poster", "card") else
                        (EXAMPLES.get(kind, {}).get("layers") or [{}])[0], ensure_ascii=False)
        problems.append(Problem("layers", "missing: nothing to draw",
                                f'add "layers": [{ex}, ...] (see compose.py --example {kind})'))
    elif kind in ("poster", "card") and not any(L["type"] in ("text", "python") for L in norm["layers"])             and not any(isinstance(x, dict) and "text" in x for x in (layers_in or [])):
        problems.append(Problem("layers", f"a {kind} needs its words: no text layer",
                                'add {"text": "...", "role": "title"} to "layers" with the title words from the request'))
    if not prompt:
        problems.append(Problem("prompt", "missing", 'add "prompt": "<the request, verbatim>" so --check can compare',
                                "note"))
    return norm, problems


def _guitar_player(norm, notes):
    """"one of them playing the guitar": the person plays it (pose "guitar", the guitar in their hands) instead of
    a separate guitar layer next to them."""
    figs = [L for L in norm["layers"] if L["type"] == "thing" and L["thing"].kit == "figure"]
    if not figs:
        return
    prompt = (norm.get("prompt") or "").lower()
    gl = [L for L in norm["layers"] if L["type"] == "thing" and "guitar" in L["thing"].name and L.get("count", 1) == 1]
    if not gl and not re.search(r"\b(play|playing|plays|strum|strumming)\s+(?:the\s+|a\s+|an\s+)?guitar", prompt):
        return
    if any(L["opts"].get("pose") == "guitar" for L in figs):
        target = None
    else:
        target = None
        for G in gl:                                   # the person the guitar relates to
            tl = (G.get("at") or {}).get("target_layer")
            target = next((F for F in figs if F["label"] == tl), None) or target
        if target is None:                             # the person whose description mentions the guitar
            target = next((F for F in figs if "guitar" in str(F["opts"].get("describe") or "").lower()), figs[0])
        target["opts"]["pose"] = "guitar"
        notes.append(f"{target['path']}: plays the guitar -> \"pose\": \"guitar\" (the guitar is in their hands)")
    for G in gl:
        norm["layers"].remove(G)
        notes.append(f"{G['path']}: the guitar is drawn in the player's hands, not as its own layer")


_WATER_WORDS = ("sea", "the sea", "ocean", "waves", "water", "lake", "harbour", "harbor", "bay", "beach")


def _design_water(norm, group, notes):
    """In a poster, card or wallpaper, "sea" / "ocean" is the setting, not a wave symbol: it becomes the ocean
    backdrop (when no backdrop is set) instead of a layer."""
    bg = norm["background"]
    if group != "design" or bg.get("backdrop") or bg.get("landscape"):
        return
    seaside = re.search(r"\b(sea|ocean|harbou?r|bay|beach|coast|lake)\b", (norm.get("prompt") or "").lower())
    for L in list(norm["layers"]):
        nm = _norm(L.get("name", ""))
        if L["type"] == "thing" and L.get("count", 1) == 1 and \
                (nm in _WATER_WORDS or (seaside and nm in ("wave", "waves", "ocean wave", "sea wave"))):
            bg["backdrop"] = "ocean"
            norm["layers"].remove(L)
            notes.append(f"{L['path']}: {L.get('name')!r} is the setting of a {norm['kind']}: drawn as "
                         f"\"background\": {{\"backdrop\": \"ocean\"}}")
            return


def _validate_background(bg, group, kind, problems, notes, prompt):
    """Background -> dict with keys: sky, ground, horizon, far, backdrop, color, gradient, table, wall, texture,
    landscape."""
    out = {}
    if bg in (None, "", True) or bg == {}:
        bg = {}
    if isinstance(bg, list):
        cols = [parse_color(c) for c in bg]
        if cols and all(cols):
            bg = {"gradient": cols}
            notes.append("background: read the list as a gradient")
        else:
            problems.append(Problem("background", "a list must hold colours (a gradient)",
                                    '"background": {"gradient": ["navy", "black"]}'))
            bg = {}
    if isinstance(bg, str):
        bg = _background_word(bg, problems, notes)
    if not isinstance(bg, dict):
        problems.append(Problem("background", "not understood",
                                'use {"sky": "sunset"} or {"backdrop": "mesh"} or {"color": "navy"}'))
        bg = {}
    keys = ("sky", "ground", "horizon", "far", "backdrop", "color", "gradient", "table", "wall", "texture",
            "landscape", "palette", "water", "options")
    aliases = {"colour": "color", "floor": "ground", "terrain": "ground", "surface": "ground", "distance": "far",
               "background": "far", "scenery": "landscape", "pattern": "backdrop", "type": "backdrop",
               "name": "backdrop", "style": "backdrop", "kind": "backdrop", "colors": "gradient",
               "colours": "gradient", "time": "sky", "weather": "sky", "time_of_day": "sky", "walls": "wall",
               "material": "texture", "sea": "water", "lake": "water", "ocean": "water", "fill": "color"}
    b = _canon_keys(bg, "background", keys, aliases, problems, notes)
    b.pop("_dropped", None)
    for k, v in (b.pop("_unknown", None) or {}).items():
        problems.append(Problem(f"background.{k}", "unknown field", _did_you_mean(k, keys)))
    from . import elements, backdrop, texture
    sky_alias = {"sunrise": "dawn", "evening": "dusk", "night sky": "night", "blue": "day", "clear": "day",
                 "afternoon": "day", "daytime": "day", "cloudy": "overcast", "stormy": "storm", "golden hour": "golden",
                 "midday": "noon", "starry": "night", "starry night": "night", "sunny": "day"}
    bd = b.get("backdrop")
    if isinstance(bd, str) and not b.get("sky") and _norm(bd).replace(" ", "_") not in backdrop.names():
        sk = sky_alias.get(_norm(bd), _norm(bd))
        if sk in elements.SKIES:                        # {"backdrop": "sunset"} means the sky
            b["sky"] = b.pop("backdrop")
            notes.append(f"background.backdrop: {bd!r} is a sky -> read as \"sky\": {bd!r}")
            if group == "scene" and not b.get("ground") and not b.get("color"):
                b["ground"] = "grass"
    if "sky" in b and b["sky"] not in (None, False, ""):
        sk = _norm(b["sky"])
        sk = sky_alias.get(sk, sk)
        if sk not in elements.SKIES:                    # "night stars", "clear blue morning"
            hit = next((sky_alias.get(w, w) for w in re.findall(r"[a-z]+", sk)
                        if sky_alias.get(w, w) in elements.SKIES), None)
            if hit:
                notes.append(f"background.sky: one sky word: read {b['sky']!r} as {hit!r}"
                             + (" (stars are a layer: {\"draw\": \"stars\"})" if "star" in sk else ""))
                sk = hit
        if sk not in elements.SKIES:
            problems.append(Problem("background.sky", f"unknown sky {b['sky']!r}", _did_you_mean(sk, elements.SKIES)))
        else:
            out["sky"] = sk
    if "color" in b and b["color"] not in (None, ""):
        c = parse_color(b["color"])
        if c is None:
            problems.append(Problem("background.color", f"{b['color']!r} is not a colour", "use a CSS name or '#rrggbb'"))
        else:
            out["color"] = c
    if "gradient" in b and b["gradient"]:
        g = b["gradient"] if isinstance(b["gradient"], list) else re.split(r"\s+(?:to|and)\s+|,", str(b["gradient"]))
        cols = [parse_color(c) for c in g]
        if not all(cols) or len(cols) < 2:
            problems.append(Problem("background.gradient", "use a list of two or more colours", '["#0b1d3a", "#1c3f7a"]'))
        else:
            out["gradient"] = cols
    if "backdrop" in b and b["backdrop"]:
        name = _norm(b["backdrop"]).replace(" ", "_")
        if name in backdrop.names():
            out["backdrop"] = name
        else:
            problems.append(Problem("background.backdrop", f"unknown backdrop {b['backdrop']!r}",
                                    _did_you_mean(name, backdrop.names())))
    if "landscape" in b and b["landscape"]:
        name = _norm(b["landscape"]).replace(" ", "_")
        name = {"mountain": "mountains", "hill": "hills", "sea": "ocean", "desert": "dunes", "city": "skyline",
                "trees": "forest", "stars": "starry"}.get(name, name)
        if name in LANDSCAPES:
            out["landscape"] = name
        else:
            problems.append(Problem("background.landscape", f"unknown landscape {b['landscape']!r}",
                                    _did_you_mean(name, LANDSCAPES)))
    if "ground" in b and b["ground"] not in (None, False, ""):
        g = _ground_name(b["ground"])
        if g is None and isinstance(b["ground"], str):  # "forest floor with logs", "campsite", "snowy field"
            low = _norm(b["ground"])
            for w in re.findall(r"[a-z]+", low):
                w2 = {"floor": "dirt", "earth": "dirt", "soil": "dirt", "mud": "dirt", "camp": "dirt",
                      "campsite": "dirt", "path": "dirt", "trail": "dirt", "snowy": "snow", "grassy": "grass",
                      "sandy": "sand", "rocky": "stone", "rocks": "stone", "field": "field", "park": "grass"}.get(w, w)
                g = _ground_name(w2)
                if g is not None:
                    notes.append(f"background.ground: read {b['ground']!r} as {w2!r}")
                    break
        if g is None:
            problems.append(Problem("background.ground", f"{b['ground']!r} not understood",
                                    "use grass, meadow, sand, snow, dirt, stone, water, a texture name or a colour"))
        else:
            out["ground"] = g
    elif b.get("ground") is False:
        out["ground"] = None
    if b.get("water"):
        out["water"] = True
    if "far" in b:
        f = b["far"]
        f = None if f in (None, False, "none", "") else _norm(f)
        f = {"trees": "forest", "woods": "forest", "hill": "hills", "mountain": "mountains", "skyline": "city",
             "buildings": "city"}.get(f, f)
        if f is not None and f not in FAR_KINDS:       # "dark trees", "distant snowy mountains"
            far_words = {"trees": "forest", "tree": "forest", "woods": "forest", "wood": "forest", "forest": "forest",
                         "pines": "forest", "hill": "hills", "hills": "hills", "mountain": "mountains",
                         "mountains": "mountains", "peaks": "mountains", "city": "city", "buildings": "city",
                         "skyline": "city", "town": "city"}
            hit = next((far_words[w] for w in re.findall(r"[a-z]+", f) if w in far_words), None)
            if hit:
                notes.append(f"background.far: read {b['far']!r} as {hit!r}")
                f = hit
        if f is not None and f not in FAR_KINDS:
            problems.append(Problem("background.far", f"{b['far']!r} not understood", _did_you_mean(f, FAR_KINDS)))
        else:
            out["far"] = f
    if "horizon" in b:
        h = b["horizon"]
        if isinstance(h, str):
            h = {"low": 0.72, "high": 0.42, "middle": 0.58, "centre": 0.55, "center": 0.55}.get(_norm(h), None)
        if not _num(h) or not 0.05 <= float(h) <= 0.95:
            problems.append(Problem("background.horizon", "use a number 0.3-0.8 (0.6 = 60% down) or low / middle / high"))
        else:
            out["horizon"] = float(h)
    if "table" in b and b["table"] not in (None, False):
        t = b["table"]
        t = "wood" if t is True else _norm(t)
        t = {"wooden": "wood", "marble": "stone", "linen": "cloth", "tablecloth": "cloth"}.get(t, t)
        if t not in ("wood", "stone", "cloth") and parse_color(t) is None:
            problems.append(Problem("background.table", f"{b['table']!r} not understood", "use wood, marble, cloth or a colour"))
        else:
            out["table"] = t if t in ("wood", "stone", "cloth") else parse_color(t)
    if "wall" in b and b["wall"]:
        c = parse_color(b["wall"])
        if c is None:
            problems.append(Problem("background.wall", f"{b['wall']!r} is not a colour", "use a CSS name or '#rrggbb'"))
        else:
            out["wall"] = c
    if "texture" in b and b["texture"]:
        name = _norm(b["texture"]).replace(" ", "_")
        if name in texture.names():
            out["texture"] = name
        else:
            problems.append(Problem("background.texture", f"unknown texture {b['texture']!r}",
                                    _did_you_mean(name, texture.names())))
    if b.get("palette"):
        out["palette"] = b["palette"]
    if isinstance(b.get("options"), dict):
        out["options"] = b["options"]
    if group == "scene":
        if not any(k in out for k in ("backdrop", "color", "gradient", "texture")):
            if "landscape" not in out:
                out.setdefault("sky", _sky_from_prompt(prompt))
                out.setdefault("ground", GROUND_KINDS["grass"] if kind != "landscape" else GROUND_KINDS["meadow"])
                out.setdefault("far", "hills" if kind != "landscape" else "mountains")
    elif group == "still_life":
        out.setdefault("table", "wood")
    return out


LANDSCAPES = ("mountains", "hills", "dunes", "ocean", "forest", "skyline", "synthwave", "starry", "clouds",
              "underwater")
FAR_KINDS = ("hills", "mountains", "forest", "city")
GROUND_KINDS = {"grass": ("lawn", "#6f9a3c"), "meadow": ("meadow", "#7fa046"), "lawn": ("lawn", "#6f9a3c"),
                "field": ("meadow", "#9aae52"), "sand": ("sand", "#e0c890"), "beach": ("sand", "#e8d4a2"),
                "desert": ("sand", "#d9b77a"), "snow": ("snow", "#ffffff"), "dirt": ("soil", "#8a6a48"),
                "soil": ("soil", "#7a5a3a"), "earth": ("dry_earth", None), "mud": ("mud", None),
                "stone": ("flagstone", None), "rock": ("granite", None), "road": ("asphalt", None),
                "street": ("asphalt", None), "path": ("gravel", None), "gravel": ("gravel", None),
                "moss": ("moss", "#94b45a"), "water": ("water", None), "wood": ("planks", None),
                "floor": ("planks", None), "tiles": ("tiles", None), "cobblestone": ("cobblestone", None)}


def _ground_name(v):
    s = _norm(v) if isinstance(v, str) else v
    if isinstance(s, str) and s in GROUND_KINDS:
        return GROUND_KINDS[s]
    c = parse_color(v)
    if c is not None:
        return ("color", c)
    if not isinstance(s, str):
        return None
    for ph, val in (("forest floor", "soil"), ("campsite", "soil"), ("camp", "soil"), ("woodland", "soil"),
                    ("leaves", "soil"), ("snowy", "snow"), ("grassy", "grass"), ("sandy", "sand")):
        if ph in s:
            return GROUND_KINDS[val]
    for w in s.split():
        if w in GROUND_KINDS:
            return GROUND_KINDS[w]
    from . import texture
    name = s.replace(" ", "_")
    if name in texture.names():
        return (name, None)
    return None


def _sky_from_prompt(prompt):
    p = (prompt or "").lower()
    for words, sky in ((("sunset", "dusk", "evening"), "sunset"), (("sunrise", "dawn"), "dawn"),
                       (("night", "starry", "moonlit", "stars"), "night"), (("golden hour",), "golden"),
                       (("storm", "thunder"), "storm"), (("overcast", "cloudy", "rainy", "rain"), "overcast"),
                       (("twilight", "blue hour"), "twilight"), (("morning",), "morning")):
        if any(re.search(rf"\b{w}\b", p) for w in words):
            return sky
    return "day"


def _background_word(word, problems, notes):
    from . import elements, backdrop, texture
    w = _norm(word)
    w2 = w.replace(" ", "_")
    if w in elements.SKIES or w in ("sunrise", "evening", "starry night", "night sky"):
        return {"sky": w}
    if w2 in backdrop.names():
        return {"backdrop": w2}
    if w2 in LANDSCAPES or w in ("mountain", "desert", "sea", "city"):
        return {"landscape": w}
    if w in ("table", "wooden table", "kitchen table"):
        return {"table": "wood"}
    c = parse_color(w)
    if c is not None:
        return {"color": c}
    if w2 in texture.names():
        return {"texture": w2}
    parts = [x for x in re.split(r"\s+(?:to|and)\s+|,", w) if x.strip()]
    if len(parts) > 1 and all(parse_color(x) for x in parts):
        return {"gradient": [parse_color(x) for x in parts]}
    problems.append(Problem("background", f"{word!r} not understood",
                            'use {"sky": "sunset"}, {"backdrop": "mesh"}, {"color": "navy"} or {"gradient": ["navy", "black"]}'))
    return {}


_PLACEHOLDERS = {"name", "where", "kind", "size", "colour", "color", "word", "thing", "what", "object", "n", "a",
                 "xxx", "todo", "tbd", "example", "something", "placeholder", "draw"}


def _is_placeholder(text, text_layer=False):
    """'NAME', '<the title>', 'WHERE', '...' left in a spec by copying the instructions."""
    t = str(text).strip()
    if not t or set(t) <= set(".…-_ "):
        return True
    if re.fullmatch(r"<[^<>]{1,60}>", t) or re.fullmatch(r"\{[A-Za-z_ ]{1,30}\}", t):
        return True
    return not text_layer and t.isupper() and len(t) > 1 and _norm(t) in _PLACEHOLDERS


def _validate_layer(raw, path, index, kind, group, problems, notes, base_dir):
    if isinstance(raw, str):
        raw = {"draw": raw}
        notes.append(f"{path}: read the bare name as {{\"draw\": {raw['draw']!r}}}")
    if not isinstance(raw, dict):
        problems.append(Problem(path, "a layer must be an object such as {\"draw\": \"oak\", \"at\": \"left\"}"))
        return None
    d = _canon_keys(raw, path, LAYER_KEYS, LAYER_ALIASES, problems, notes)
    d.pop("_dropped", None)
    unknown = d.pop("_unknown", {}) or {}
    L = {"index": index, "path": path, "raw": raw, "opts": {}}
    if "python" in d:
        ref = str(d["python"])
        if ":" not in ref:
            problems.append(Problem(f"{path}.python", "use \"file.py:function\"", '"python": "my_shapes.py:draw_logo"'))
            return None
        L["type"] = "python"
        L["ref"] = ref
        L["label"] = str(d.get("id") or d.get("label") or ref.split(":")[1])
    elif "text" in d and "draw" not in d:
        t = d["text"]
        if isinstance(t, list):
            t = "\n".join(map(str, t))
        if t is None or str(t).strip() == "":
            problems.append(Problem(f"{path}.text", "is empty", "write the words to draw, exactly as requested"))
            return None
        if _is_placeholder(t, text_layer=True):
            problems.append(Problem(f"{path}.text", f"{t!r} is a placeholder from the instructions",
                                    "write the exact words from the request"))
            return None
        L["type"] = "text"
        L["text"] = str(t)
        qm = re.fullmatch(r"\s*(['\"‘“])(.+)(['\"’”])\s*", L["text"])
        if qm and len(qm.group(2)) > 1:                # 'Wind Lanterns' copied with the request's quotes
            L["text"] = qm.group(2)
            notes.append(f"{path}: wrote the text without the quote marks around it: {L['text']!r}")
            t = L["text"]
        role = _norm(d.get("role") or "body")
        role = ROLE_ALIASES.get(role, role)
        if role not in ROLES:
            problems.append(Problem(f"{path}.role", f"unknown role {d.get('role')!r}", _did_you_mean(role, ROLES)))
            role = "body"
        L["role"] = role
        L["label"] = str(d.get("id") or d.get("label") or str(t)[:24])
        flow_roles = ("title", "subtitle", "kicker", "badge", "details", "date", "byline", "footer", "body", "quote")
        if kind in ("poster", "card") and role in flow_roles and (d.get("at") is not None or d.get("box") is not None):
            # a small model's own boxes put a tiny title in a corner; the template sets type far better
            where = {k_: d.pop(k_) for k_ in ("at", "box") if d.get(k_) is not None}
            notes.append(f"{path}: text in a {kind} is laid out by its role ({role}); ignored "
                         + ", ".join(f"\"{k_}\": {v_!r}" for k_, v_ in where.items()) + " - leave it out")
        for k in ("font", "align", "bold", "plate", "shadow"):
            if k in d:
                L["opts"][k] = d[k]
        if "ink" in d and "color" not in d:
            d["color"] = d["ink"]
    elif "draw" in d:
        name = d["draw"]
        if isinstance(name, list):
            problems.append(Problem(f"{path}.draw", "one name per layer", "split it into several layers, or use \"count\""))
            return None
        name = str(name)
        m = re.match(r"^(\d+|" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")\s+(.+)$", _norm(name))
        if m and "count" not in d:
            n = int(m.group(1)) if m.group(1).isdigit() else _NUMBER_WORDS[m.group(1)]
            d["count"] = n
            notes.append(f"{path}: read {name!r} as count {n} of {m.group(2)!r}")
            name = m.group(2)
        if kind in ("room", "scene3d"):
            L["type"] = "thing3d"
            L["name"] = name
            L["label"] = str(d.get("id") or d.get("label") or name)
            return L
        if group in ("scene", "still_life") and _norm(name) in ("snowflake", "snowflakes", "snow flakes", "falling snow",
                                                                 "raindrop", "raindrops", "falling rain"):
            new = "rain" if "rain" in _norm(name) else "snow"
            notes.append(f"{path}: read {name!r} as the {new!r} effect over the whole picture")
            name = new
            d.pop("count", None)
        if group == "scene" and _norm(name) in ("star", "stars", "starry sky", "night stars", "twinkling stars"):
            notes.append(f"{path}: read {name!r} as the 'stars' effect: a starry sky over the whole picture")
            name = "stars"
            for k_ in ("count", "at", "size"):
                d.pop(k_, None)
        if _is_placeholder(name):
            problems.append(Problem(f"{path}.draw", f"{name!r} is a placeholder from the instructions",
                                    "write a real thing from the request, e.g. \"cat\" (list names: compose.py --names WORD)"))
            return None
        thing, extras, sugg, note = lookup(name, kind)
        if (thing is None or (len(_norm(name).split()) > 2 and thing.kit != "figure")) and \
                any(re.search(rf"\b{w}\b", _norm(name)) for w in _PEOPLE_WORDS):
            thing, extras, note = _registry()[0].get("person"), {"describe": name}, ""
        if thing is None:
            fix = (f"did you mean {' or '.join(repr(x) for x in sugg[:3])}?" if sugg else
                   "no kit draws it: pick a name from compose.py --names WORD, or remove the layer")
            problems.append(Problem(f"{path}.draw", f"unknown {name!r}", fix))
            return None
        if note:
            notes.append(f"{path}: {note}")
        L["type"] = "thing"
        L["thing"] = thing
        L["name"] = name
        L["label"] = str(d.get("id") or d.get("label") or _norm(name))
        for k, v in extras.items():
            d.setdefault(k, v)
    else:
        found = ", ".join(map(repr, unknown)) if unknown else "nothing"
        problems.append(Problem(path, f"no \"draw\" or \"text\" (found {found})",
                                "name what to draw: {\"draw\": \"oak\"}, or write {\"text\": \"...\", \"role\": \"title\"}"))
        return None
    at, err = parse_at(d.get("at"))
    if err:
        problems.append(Problem(f"{path}.at", f"{d.get('at')!r}: {err}", "use " + PLACEMENT_HELP))
        at = {"mode": "auto"}
    if at.get("note"):
        notes.append(f"{path}.at: {at['note']}")
    for key, rel in REL_KEYS.items():
        if key in d and d[key] not in (None, ""):
            at = dict(at) if at.get("mode") == "words" else {"mode": "words", "text": ""}
            at["rel"], at["target"] = REL_CANON.get(rel, rel), _norm(d[key])
    if "box" in d:
        b, err = parse_at(d["box"])
        if err or b.get("mode") != "box":
            problems.append(Problem(f"{path}.box", "use [u0, v0, u1, v1] fractions", "[0.1, 0.1, 0.5, 0.4]"))
        else:
            at = b
    L["at"] = at
    sz, err, note = parse_size(d.get("size"))
    if err:
        problems.append(Problem(f"{path}.size", err))
    if note:
        notes.append(f"{path}.size: {note}")
    L["size"], L["size_rel"] = (None, sz[1]) if isinstance(sz, tuple) else (sz, 1.0)
    L["color"] = None
    if L.get("thing") is not None and L["thing"].kit == "creature" and d.get("color") and not d.get("coat"):
        from . import creature as _C
        cw = _norm(d["color"]) if isinstance(d["color"], str) else ""
        coats = getattr(_C, "COATS", {})
        if cw in coats or _norm(cw.replace("brown", "chocolate")) in coats:
            d["coat"] = cw if cw in coats else "chocolate"
            notes.append(f"{path}: read color {d['color']!r} as coat {d['coat']!r}")
            d.pop("color")
        else:
            problems.append(Problem(f"{path}.color", f"animals take a coat, not a colour",
                                    f"use \"coat\": one of {', '.join(list(coats)[:12])}", "warn"))
            d.pop("color")
    if "color" in d and d["color"] not in (None, ""):
        col = d["color"]
        if isinstance(col, list) and col and not all(_num(x) for x in col):
            col = col[0]
        c = parse_color(col)
        if c is None:
            if L.get("thing") is not None and L["thing"].kit == "creature":
                d.setdefault("coat", d["color"])
            else:
                problems.append(Problem(f"{path}.color", f"{d['color']!r} is not a colour", "use a CSS name ('gold') or '#rrggbb'"))
        L["color"] = c
    n = d.get("count", 1)
    if isinstance(n, str):
        n = _NUMBER_WORDS.get(_norm(n), n)
        try:
            n = int(n)
        except (TypeError, ValueError):
            problems.append(Problem(f"{path}.count", f"{d.get('count')!r} is not a number", '"count": 3'))
            n = 1
    if not _num(n) or n < 1 or n > 60:
        problems.append(Problem(f"{path}.count", "use a whole number 1-60"))
        n = 1
    L["count"] = int(n)
    arr = d.get("arrange")
    if arr is not None:
        a = _norm(arr)
        a = {"line": "row", "group": "cluster", "random": "scatter", "spread": "scatter", "pile": "cluster",
             "ring": "circle", "stack": "column", "vertical": "column", "horizontal": "row"}.get(a, a)
        if a not in ARRANGE:
            problems.append(Problem(f"{path}.arrange", f"unknown arrangement {arr!r}", _did_you_mean(a, ARRANGE)))
            a = None
        arr = a
    L["arrange"] = arr
    for k in ("seed",):
        if k in d and _num(d[k]):
            L[k] = int(d[k])
    for k in ("opacity", "depth"):
        if k in d and _num(d[k]):
            L[k] = float(d[k])
    if L["type"] == "thing":
        th = L["thing"]
        for k in ("describe", "pose", "coat", "facing", "view", "style", "material"):
            if k in d and d[k] not in (None, ""):
                L["opts"][k] = d[k]
        for k, v in unknown.items():
            kl = k.lower()
            if kl in th.options:
                L["opts"][kl] = v
            else:
                opts = list(th.options) + ["at", "size", "color", "count", "arrange"]
                problems.append(Problem(f"{path}.{k}", f"unknown option for {th.name!r}", _did_you_mean(kl, opts)))
        _check_options(L, path, problems)
    else:
        for k in unknown:
            problems.append(Problem(f"{path}.{k}", "unknown field",
                                    _did_you_mean(k, ("text", "role", "at", "size", "color", "font", "align", "box"))))
    return L


_POSE_WORDS = {"sitting": "sit", "standing": "stand", "flying": "fly", "lying": "lie", "sleeping": "sleep",
               "running": "run", "walking": "walk", "perched": "perch", "perching": "perch", "swimming": "swim",
               "jumping": "jump", "lying down": "lie", "waving": "wave", "dancing": "dance", "pointing": "point",
               "playing guitar": "guitar", "cheering": "cheer", "talking": "talk", "kneeling": "kneel",
               "leaning": "lean", "grazing": "graze", "sniffing": "sniff", "crouching": "crouch",
               "galloping": "gallop", "trotting": "trot", "resting": "rest", "thinking": "thinking",
               "reaching": "reach", "holding": "hold", "asleep": "sleep", "napping": "sleep", "dozing": "sleep",
               "curled up": "sleep", "curled": "sleep", "relaxed": "rest", "relaxing": "rest", "lounging": "lie",
               "laying": "lie", "laying down": "lie", "seated": "sit", "eating": "graze", "prowling": "walk",
               "hunting": "crouch", "pouncing": "jump", "leaping": "jump", "soaring": "glide", "gliding": "glide",
               "pecking": "peck", "standing up": "stand", "sitting down": "sit", "hopping": "jump"}


def _check_options(L, path, problems):
    th, o = L["thing"], L["opts"]
    if th.kit in ("creature", "figure") and "pose" in o:
        mod = importlib.import_module(f"pixelcanvas.{th.kit}")
        poses = list(mod.POSES)
        p = _norm(o["pose"])
        p = _POSE_WORDS.get(p, p).replace(" ", "_")
        if th.kit == "creature" and p == "run":
            p = "gallop"
        if p not in poses:
            for alt in {"sleep": ("lie", "rest", "sit"), "rest": ("lie", "sleep", "sit"), "lie": ("rest", "sleep"),
                        "glide": ("fly",), "perch": ("stand", "sit")}.get(p, ()):
                if alt in poses:
                    p = alt
                    break
        if p not in poses:                             # "sit, hold" / "sitting relaxed, playing guitar": a pose word
            low = _norm(o["pose"])                     # inside the phrase (the first one that is a pose)
            found = None
            for w in sorted(_POSE_WORDS, key=len, reverse=True):
                if re.search(r"\b" + re.escape(w) + r"\b", low) and _POSE_WORDS[w] in poses:
                    found = (low.index(w), _POSE_WORDS[w])
                    break
            for w in re.findall(r"[a-z_]+", low):
                if w in poses and (found is None or low.index(w) < found[0]):
                    found = (low.index(w), w)
                    break
            if found:
                problems.append(Problem(f"{path}.pose", f"one pose per layer: read {o['pose']!r} as {found[1]!r}",
                                        f'write "pose": "{found[1]}"', "note"))
                p = found[1]
        if p not in poses:
            close = _close(p, poses, n=2)
            problems.append(Problem(f"{path}.pose", f"unknown pose {o['pose']!r} for {th.name!r}",
                                    (f"did you mean {' or '.join(repr(c) for c in close)}? " if close else "")
                                    + f"poses: {', '.join(poses)}"))
            o.pop("pose")
        else:
            o["pose"] = p
    if th.kit == "creature":
        from . import creature as C
        if "describe" in o:
            desc = _norm(o.pop("describe"))
            coats = sorted(getattr(C, "COATS", {}), key=len, reverse=True)
            hit = next((c for c in coats if re.search(rf"\b{re.escape(c)}\b", desc)), None)
            if hit and "coat" not in o:
                o["coat"] = hit
            pw = next((p for w, p in _POSE_WORDS.items() if re.search(rf"\b{w}\b", desc)), None)
            if pw and "pose" not in o and pw in C.POSES:
                o["pose"] = pw
            problems.append(Problem(f"{path}.describe", "animals take \"coat\" and \"pose\", not \"describe\"",
                                    f"read it as {', '.join(f'{k} {o[k]!r}' for k in ('coat', 'pose') if k in o) or 'nothing'}; "
                                    f"coats: {', '.join(list(getattr(C, 'COATS', {}))[:10])}", "warn"))
        if "coat" in o:
            c = _norm(o["coat"])
            coats = getattr(C, "COATS", {})
            if c not in coats:
                col = parse_color(c)
                o.pop("coat")
                if col is not None:
                    L["color"] = L["color"] or col
                else:
                    problems.append(Problem(f"{path}.coat", f"unknown coat {c!r}", _did_you_mean(c, coats)))
            else:
                o["coat"] = c
    if "facing" in o:
        f = _norm(o["facing"])
        if f not in ("left", "right"):
            problems.append(Problem(f"{path}.facing", "use left or right"))
            o.pop("facing")
        else:
            o["facing"] = f
    if th.kit == "pictogram" and "style" in o:
        try:
            P = importlib.import_module("pixelcanvas.pictograms")
            st = _norm(o["style"])
            if st not in P.STYLES:
                problems.append(Problem(f"{path}.style", f"unknown pictogram style {o['style']!r}",
                                        _did_you_mean(st, P.STYLES)))
                o.pop("style")
            else:
                o["style"] = st
        except Exception:
            pass
    if th.kit == "form" and "material" in o:
        from . import form
        mats = list(getattr(form, "MATERIALS", {}))
        if mats and o["material"] not in mats:
            problems.append(Problem(f"{path}.material", f"unknown material {o['material']!r}",
                                    _did_you_mean(o["material"], mats)))
            o.pop("material")
    if th.kit not in ("creature", "figure", "pictogram", "form"):
        for k in list(o):
            if k in ("pose", "coat", "describe", "view"):
                problems.append(Problem(f"{path}.{k}", f"{th.name!r} takes no {k!r}", f"remove \"{k}\"", "warn"))
                o.pop(k)


def _unique_labels(layers):
    seen = {}
    for L in layers:
        base = L["label"]
        n = seen.get(base, 0)
        seen[base] = n + 1
        if n:
            L["label"] = f"{base} {n + 1}"


def _words_of(L):
    th = L.get("thing")
    words = {_norm(L["label"]), _norm(L.get("name", ""))}
    if th is not None:
        words |= {th.name, th.cat, th.target or ""}
        if th.cat == "plant" and th.size >= 0.4:
            words.add("tree")
        if th.cat in ("bird", "animal"):
            words.add("animal")
        if th.cat == "bird":
            words.add("bird")
        if th.kit == "figure":
            words |= {"person", "man", "woman", "people", "friend", "friends", "couple", "family", "child", "kid"}
    return {w for w in words if w}


def _find_target(layers, name, exclude=None):
    t = _norm(name)
    t = re.sub(r"^(the|a|an)\s+", "", t)
    cands = [L for L in layers if L is not exclude and L["type"] in ("thing", "python")]
    for L in cands:
        if t in (_norm(L["label"]), _norm(L.get("name", ""))):
            return L
    m = re.match(r"^(first|second|third|fourth|1st|2nd|3rd|4th|other|another)\s+(.+)$", t)
    if m:                                              # "the second person" -> the second layer that is a person
        k = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "other": 1, "another": 1, "third": 2, "3rd": 2,
             "fourth": 3, "4th": 3}[m.group(1)]
        same = [L for L in cands if m.group(2) in _words_of(L) or _singular(m.group(2)) in _words_of(L)]
        if len(same) > k:
            return same[k]
    for L in cands:                                    # "the man" -> the person described as "a man in ..."
        desc = _norm(str((L.get("opts") or {}).get("describe") or ""))
        if desc and re.search(r"\b" + re.escape(_singular(t)) + r"\b", desc):
            return L
    for L in cands:
        words = _words_of(L)
        if t in words or _singular(t) in words or any(t == w.split()[-1] or _singular(t) == w.split()[-1]
                                                       for w in words):
            return L
    return None


def _check_targets(layers, problems):
    for L in layers:
        at = L.get("at") or {}
        if at.get("rel") and at.get("target"):
            tgt = at["target"]
            T = _find_target(layers, tgt, L)
            if T is None and _norm(tgt) in ("table", "the table", "desk", "the desk"):
                at["target_layer"] = "@table"
                continue
            if T is None:
                labels = [x["label"] for x in layers if x is not L and x["type"] in ("thing", "python")]
                problems.append(Problem(f"{L['path']}.at", f"{at['rel']} {tgt!r}: no layer called {tgt!r}",
                                        (f"use one of: {', '.join(repr(x) for x in labels)}" if labels else
                                         "add that layer first, or use a placement word")))
                L["at"] = {"mode": "auto"}
            else:
                at["target_layer"] = T["label"]


# ============================================================================================= sections
DIAGRAM_KEYS = ("nodes", "edges", "groups", "theme", "direction", "title", "subtitle", "routing", "style")
CHART_KEYS = ("type", "data", "title", "subtitle", "x", "xlabel", "ylabel", "theme", "fmt", "yfmt", "xfmt",
              "value_labels", "legend", "highlight", "smooth", "stacked", "horizontal", "colors", "color", "box",
              "options")
DASH_KEYS = ("title", "subtitle", "theme", "kpis", "charts", "right")
UI_KEYS = ("template", "screen", "theme", "accent", "content", "style")
S3D_KEYS = ("room", "objects", "lighting", "camera", "ground", "background", "sky", "light")
CHART_TYPES = ("bar", "barh", "line", "area", "pie", "donut", "scatter", "hist", "stacked", "heatmap", "radar", "spark")
_CHART_TYPE_ALIASES = {"column": "bar", "columns": "bar", "bars": "bar", "bar chart": "bar", "horizontal bar": "barh",
                       "line chart": "line", "lines": "line", "pie chart": "pie", "doughnut": "donut",
                       "histogram": "hist", "points": "scatter", "dots": "scatter", "stacked bar": "stacked",
                       "spider": "radar", "sparkline": "spark", "area chart": "area"}
_CAMERAS = ("isometric", "three-quarter", "front", "side", "top", "low", "high")


def _sub_keys(d, path, keys, aliases, problems, notes):
    if not isinstance(d, dict):
        problems.append(Problem(path, "must be an object { ... }"))
        return {}
    out = _canon_keys(d, path, keys, aliases, problems, notes)
    out.pop("_dropped", None)
    for k in (out.pop("_unknown", None) or {}):
        problems.append(Problem(f"{path}.{k}", "unknown field", _did_you_mean(k, keys)))
    return out


def _validate_section(sec, d, problems, notes, kind, norm):
    if sec == "diagram":
        return _validate_diagram(d, problems, notes)
    if sec == "chart":
        return _validate_chart(d, "chart", problems, notes)
    if sec == "dashboard":
        if not isinstance(d, dict):
            problems.append(Problem("dashboard", "must be an object"))
            return {}
        out = _sub_keys(d, "dashboard", DASH_KEYS, {"header": "title", "kpi": "kpis", "tiles": "kpis",
                                                     "metrics": "kpis", "stats": "kpis", "graphs": "charts",
                                                     "panels": "charts", "chips": "right"}, problems, notes)
        kpis = []
        for i, k in enumerate(out.get("kpis") or []):
            p = f"dashboard.kpis[{i}]"
            if isinstance(k, (list, tuple)) and len(k) >= 2:
                k = {"label": k[0], "value": k[1], **({"delta": k[2]} if len(k) > 2 else {})}
            if not isinstance(k, dict):
                problems.append(Problem(p, "use {\"label\": \"Revenue\", \"value\": 128400, \"delta\": 12.4}"))
                continue
            kk = _sub_keys(k, p, ("label", "value", "delta", "spark", "fmt", "note", "icon", "color", "good"),
                           {"name": "label", "title": "label", "change": "delta", "trend": "spark",
                            "sparkline": "spark", "number": "value"}, problems, notes)
            if "label" not in kk or "value" not in kk:
                problems.append(Problem(p, "needs label and value"))
                continue
            kpis.append(kk)
        charts = []
        for i, c in enumerate(out.get("charts") or []):
            if c:
                charts.append(_validate_chart(c, f"dashboard.charts[{i}]", problems, notes))
        out["kpis"], out["charts"] = kpis, [c for c in charts if c]
        if not kpis and not out["charts"]:
            problems.append(Problem("dashboard", "add kpis and / or charts", "see compose.py --example dashboard"))
        _theme_check(out, "dashboard", problems)
        return out
    if sec == "ui":
        if isinstance(d, str):
            d = {"template": d}
        out = _sub_keys(d, "ui", UI_KEYS, {"screen_type": "screen", "device": "screen", "name": "template",
                                           "page": "template", "kind": "template", "type": "template",
                                           "data": "content", "text": "content", "colour": "accent",
                                           "color": "accent"}, problems, notes)
        from . import ui
        content = out.get("content") if isinstance(out.get("content"), dict) else {}
        t = _norm(out.get("template") or ("home" if content.get("cards") else "landing")).replace(" ", "_")
        t = {"home_screen": "home", "greeting": "home", "cards": "home", "today": "home"}.get(t, t)
        if t not in ui.TEMPLATES and t != "home":
            problems.append(Problem("ui.template", f"unknown template {out.get('template')!r}",
                                    _did_you_mean(t, list(ui.TEMPLATES) + ["home"])
                                    + ' ("home": a greeting and a list of cards)'))
            t = "landing"
        cards_ = content.get("cards") if isinstance(content, dict) else None
        if t == "home" and isinstance(cards_, (str, dict)) and cards_:
            notes.append("ui.content.cards: read as a list of cards (write \"cards\": [{\"title\": ...}, ...])")
        elif t == "home" and cards_ is not None and not isinstance(cards_, list):
            problems.append(Problem("ui.content.cards", "must be a list",
                                    '"cards": ["Today\'s plan", {"title": "Sleep", "text": "7 h 20 min"}]'))
        out["template"] = t
        scr = _norm(out.get("screen") or ("browser" if t in ("dashboard", "landing", "pricing", "kanban") else "phone"))
        scr = {"web": "browser", "website": "browser", "mobile": "phone", "iphone": "phone", "android": "phone",
               "desktop": "browser", "ipad": "tablet"}.get(scr, scr)
        if scr not in ui.DEVICES:
            problems.append(Problem("ui.screen", f"unknown screen {out.get('screen')!r}", _did_you_mean(scr, ui.DEVICES)))
            scr = "phone"
        out["screen"] = scr
        if out.get("accent") is not None:
            c = parse_color(out["accent"])
            if c is None:
                problems.append(Problem("ui.accent", f"{out['accent']!r} is not a colour"))
            out["accent"] = c
        if out.get("theme") not in (None, "light", "dark"):
            problems.append(Problem("ui.theme", "use light or dark"))
            out["theme"] = None
        if out.get("content") is not None and not isinstance(out["content"], dict):
            problems.append(Problem("ui.content", "must be an object of texts, e.g. {\"title\": \"Overview\"}"))
            out["content"] = None
        return out
    if sec == "scene3d":
        return _validate_scene3d(d, problems, notes, kind)
    return d


def _theme_check(out, path, problems):
    from . import vector
    th = out.get("theme")
    if th is None:
        return
    t = _norm(th)
    if t not in vector.THEMES:
        problems.append(Problem(f"{path}.theme", f"unknown theme {th!r}", _did_you_mean(t, vector.THEMES)))
        out["theme"] = None
    else:
        out["theme"] = t


def _validate_chart(d, path, problems, notes):
    if not isinstance(d, dict):
        problems.append(Problem(path, "must be an object {\"type\": \"bar\", \"data\": {...}}"))
        return None
    out = _sub_keys(d, path, CHART_KEYS, {"kind": "type", "chart": "type", "chart_type": "type", "values": "data",
                                          "series": "data", "labels": "x", "categories": "x", "x_label": "xlabel",
                                          "y_label": "ylabel", "unit": "ylabel", "units": "ylabel",
                                          "y_axis": "ylabel", "x_axis": "xlabel", "format": "fmt",
                                          "number_format": "fmt", "show_values": "value_labels",
                                          "labels_on_bars": "value_labels", "colour": "color",
                                          "heading": "title", "name": "title"}, problems, notes)
    t = _norm(out.get("type") or "bar")
    t = _CHART_TYPE_ALIASES.get(t, t)
    if t not in CHART_TYPES:
        problems.append(Problem(f"{path}.type", f"unknown chart type {out.get('type')!r}", _did_you_mean(t, CHART_TYPES)))
        t = "bar"
    out["type"] = t
    data = out.get("data")
    if data is None:
        problems.append(Problem(f"{path}.data", "missing", '"data": {"Q1": 120, "Q2": 150}'))
    elif isinstance(data, list) and data and all(isinstance(x, dict) for x in data):
        # [{"label": "Q1", "value": 120}, ...] -> {"Q1": 120}
        conv = {}
        for x in data:
            lab = next((x[k] for k in ("label", "name", "x", "category", "quarter", "month") if k in x), None)
            val = next((x[k] for k in ("value", "y", "amount", "count", "revenue") if k in x), None)
            if lab is None or val is None:
                conv = None
                break
            conv[str(lab)] = val
        if conv:
            out["data"] = conv
            notes.append(f"{path}.data: read the list of objects as {{label: value}}")
        else:
            problems.append(Problem(f"{path}.data", "use {\"label\": number} pairs", '{"Q1": 120, "Q2": 150}'))
    elif isinstance(data, list) and data and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in data) \
            and t not in ("scatter",):
        out["data"] = {str(a): b for a, b in data}
        notes.append(f"{path}.data: read the pairs as {{label: value}}")
    if isinstance(out.get("data"), dict):
        bad = [k for k, v in out["data"].items() if not (_num(v) or (isinstance(v, list) and all(_num(x) for x in v)))]
        for k in bad:
            v = out["data"][k]
            try:
                out["data"][k] = float(str(v).replace(",", "").replace("$", "").replace("%", ""))
                notes.append(f"{path}.data[{k!r}]: read {v!r} as a number")
            except ValueError:
                problems.append(Problem(f"{path}.data[{k!r}]", f"{v!r} is not a number"))
    if "fmt" in out and "yfmt" not in out:
        out["yfmt"] = out.pop("fmt")
    if isinstance(out.get("color"), str):
        c = parse_color(out["color"])
        out.pop("color")
        if c:
            out["colors"] = [c]
    if isinstance(out.get("colors"), list):
        cs = [parse_color(c) for c in out["colors"]]
        out["colors"] = [c for c in cs if c] or None
        if out["colors"] is None:
            out.pop("colors")
    _theme_check(out, path, problems)
    return out


_ICON_HINTS = (("database", "database"), ("postgres", "database"), ("mysql", "database"), ("sql", "database"),
               ("redis", "layers"), ("cache", "layers"), ("queue", "inbox"), ("kafka", "inbox"), ("web", "monitor"),
               ("browser", "globe"), ("frontend", "monitor"), ("mobile", "phone"), ("app", "phone"),
               ("ios", "phone"), ("android", "phone"), ("gateway", "shield"), ("auth", "lock"), ("login", "key"),
               ("user", "user"), ("customer", "users"), ("order", "cart"), ("payment", "credit-card"),
               ("billing", "credit-card"), ("search", "search"), ("email", "mail"), ("mail", "mail"),
               ("notification", "bell"), ("storage", "folder"), ("file", "file"), ("cdn", "cloud"),
               ("cloud", "cloud"), ("server", "server"), ("api", "server"), ("service", "server"),
               ("analytics", "chart"), ("monitor", "activity"), ("log", "list"), ("ai", "bot"), ("ml", "bot"),
               ("model", "bot"), ("worker", "gear"), ("scheduler", "clock"), ("load balancer", "network"),
               ("dns", "globe"), ("internet", "globe"), ("client", "monitor"))


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_") or "node"


def _validate_diagram(d, problems, notes):
    if isinstance(d, list):
        d = {"nodes": d}
    out = _sub_keys(d, "diagram", DIAGRAM_KEYS, {"boxes": "nodes", "items": "nodes", "components": "nodes",
                                                  "services": "nodes", "steps": "nodes", "links": "edges",
                                                  "arrows": "edges", "connections": "edges", "flows": "edges",
                                                  "clusters": "groups", "tiers": "groups", "layout": "direction",
                                                  "orientation": "direction", "heading": "title"}, problems, notes)
    nodes, ids = [], {}
    for i, n in enumerate(out.get("nodes") or []):
        p = f"diagram.nodes[{i}]"
        if isinstance(n, str):
            n = {"label": n}
        if not isinstance(n, dict):
            problems.append(Problem(p, "use a name or {\"id\": \"api\", \"label\": \"API\"}"))
            continue
        nn = _sub_keys(n, p, ("id", "label", "icon", "shape", "group", "color", "sublabel", "rank"),
                       {"name": "label", "text": "label", "title": "label", "colour": "color", "type": "shape",
                        "description": "sublabel", "subtitle": "sublabel", "key": "id"}, problems, notes)
        label = str(nn.get("label") or nn.get("id") or "")
        if not label:
            problems.append(Problem(p, "needs a label"))
            continue
        nid = _slug(nn.get("id") or label)
        while nid in ids:
            nid += "_2"
        nn["id"], nn["label"] = nid, label
        low = label.lower()
        if "icon" not in nn:
            for word, ic in _ICON_HINTS:
                if re.search(rf"\b{re.escape(word)}", low):
                    nn["icon"] = ic
                    break
        if "shape" not in nn and re.search(r"database|postgres|mysql|\bdb\b|sql|mongo|storage", low):
            nn["shape"] = "cylinder"
        if "shape" not in nn and (low.rstrip().endswith("?") or re.match(r"(if|is|does|do|has|can|check if)\b", low)):
            nn["shape"] = "diamond"                    # a question is a decision
            nn.pop("icon", None)
        if "shape" not in nn and re.fullmatch(r"(start|end|stop|begin|finish|done)\b.*", low) and len(low) < 16:
            nn["shape"] = "pill"                       # the start and the end of a flowchart
        ids[nid] = nn
        for alias in {label.lower(), nid, _slug(label)}:
            ids.setdefault(alias, nn)
        nodes.append(nn)
    groups = out.get("groups") or {}
    gl = []
    if isinstance(groups, list):
        tmp = {}
        for g in groups:
            if isinstance(g, dict) and ("label" in g or "name" in g or "id" in g):
                tmp[str(g.get("label") or g.get("name") or g.get("id"))] = g.get("nodes") or g.get("members") or g.get("items") or []
        groups = tmp
    if not isinstance(groups, dict):
        problems.append(Problem("diagram.groups", "use {\"Services\": [\"auth\", \"orders\"]}"))
        groups = {}

    def ref(x, path):
        k = str(x).lower()
        hit = ids.get(k) or ids.get(_slug(x))
        if hit is None:
            close = difflib.get_close_matches(k, [n["label"].lower() for n in nodes] + [n["id"] for n in nodes], n=1, cutoff=0.6)
            if close:
                hit = ids.get(close[0])
                notes.append(f"{path}: read {x!r} as {hit['label']!r}")
            else:
                for n in nodes:                           # "auth" -> "Auth service"
                    if k and (k in n["label"].lower().split() or n["label"].lower().startswith(k)):
                        hit = n
                        notes.append(f"{path}: read {x!r} as {n['label']!r}")
                        break
        return hit

    for gname, members in groups.items():
        mem = []
        for m in members if isinstance(members, list) else [members]:
            hit = ref(m, f"diagram.groups[{gname!r}]")
            if hit is None:
                problems.append(Problem(f"diagram.groups[{gname!r}]", f"no node {m!r}",
                                        f"nodes: {', '.join(repr(n['label']) for n in nodes)}"))
            else:
                mem.append(hit["id"])
        if len(mem) == 1 and len(nodes) > 2:           # a frame around one box is noise
            notes.append(f"diagram.groups[{gname!r}]: a group of one node is left out")
            continue
        gl.append((str(gname), mem))
    edges = []
    for i, e in enumerate(out.get("edges") or []):
        p = f"diagram.edges[{i}]"
        if isinstance(e, str):
            parts = [x.strip() for x in re.split(r"\s*(?:->|→|=>|-->)\s*", e) if x.strip()]
            if len(parts) >= 2:
                e = [parts[0], parts[1]]
            else:
                problems.append(Problem(p, "use [\"from\", \"to\", \"label\"]"))
                continue
        if isinstance(e, dict):
            ee = _sub_keys(e, p, ("from", "to", "label", "style", "arrow"),
                           {"source": "from", "a": "from", "start": "from", "target": "to", "b": "to", "end": "to",
                            "text": "label", "name": "label"}, problems, notes)
            e = [ee.get("from"), ee.get("to"), ee.get("label"), ee.get("style"), ee.get("arrow")]
        if not isinstance(e, (list, tuple)) or len(e) < 2:
            problems.append(Problem(p, "use [\"from\", \"to\"] or [\"from\", \"to\", \"label\"]"))
            continue
        e = list(e)
        label = e[2] if len(e) > 2 else None
        for k_ in (1, 0):                              # "Yes -> Refill tank" / "No: Brew" as a node: label + node
            if isinstance(e[k_], str) and str(e[k_]).lower() not in ids:
                m_ = re.match(r"^\s*(yes|no|true|false|ok|fail|pass|otherwise|else)\s*(?:->|=>|:|-|,|\))\s*(.+)$",
                              e[k_], re.I)
                if m_ and ref(m_.group(2), p) is not None:
                    e[k_] = m_.group(2)
                    if label in (None, ""):
                        label = m_.group(1).capitalize()
                    notes.append(f"{p}: read {m_.group(0).strip()!r} as the edge label {label!r} to {e[k_]!r}")
        a_list = e[0] if isinstance(e[0], list) else [e[0]]
        b_list = e[1] if isinstance(e[1], list) else [e[1]]
        style = e[3] if len(e) > 3 else None
        ok = True
        res = []
        for side, lst in (("from", a_list), ("to", b_list)):
            got = []
            for x in lst:
                hit = ref(x, p)
                if hit is None:
                    problems.append(Problem(f"{p}.{side}", f"no node {x!r}",
                                            f"nodes: {', '.join(repr(n['label']) for n in nodes)}"))
                    ok = False
                else:
                    got.append(hit["id"])
            res.append(got)
        if ok:
            for a in res[0]:
                for b in res[1]:
                    edges.append((a, b, None if label in (None, "") else str(label),
                                  style if style in ("solid", "dashed", "dotted") else "solid"))
    if len(edges) >= 2:                                # a step with no arrow: joined from the step listed before it
        linked = {a for a, b, *_ in edges} | {b for a, b, *_ in edges}
        for k, n in enumerate(nodes):
            if n["id"] in linked or k == 0:
                continue
            prev = nodes[k - 1]
            if prev["id"] in linked:
                edges.append((prev["id"], n["id"], None, "solid"))
                linked.add(n["id"])
                notes.append(f"diagram: {n['label']!r} had no arrow; joined it from {prev['label']!r} (the node "
                             f"before it) - write [\"{prev['label']}\", \"{n['label']}\"] in edges")
    if not nodes:
        problems.append(Problem("diagram.nodes", "no nodes", '"nodes": ["Web app", "API", "Database"]'))
    dr = out.get("direction")
    if dr is not None:
        dd = {"left to right": "LR", "lr": "LR", "horizontal": "LR", "top to bottom": "TB", "tb": "TB",
              "vertical": "TB", "td": "TB", "rl": "RL", "bt": "BT", "auto": "auto"}.get(_norm(dr), None)
        if dd is None:
            problems.append(Problem("diagram.direction", f"{dr!r} not understood", "use LR, TB or auto"))
        out["direction"] = dd or "auto"
    _theme_check(out, "diagram", problems)
    out["nodes"], out["edges"], out["groups"] = nodes, edges, gl
    return out


_PREFAB_ALIASES = {"desk": "table", "dining table": "table", "coffee table": "table", "lamp": "desk_lamp",
                   "desk lamp": "desk_lamp", "table lamp": "desk_lamp", "ceiling lamp": "pendant_lamp",
                   "pendant lamp": "pendant_lamp", "bookcase": "bookshelf", "book shelf": "bookshelf",
                   "shelf": "bookshelf", "books": "bookshelf", "plant": "potted_plant", "potted plant": "potted_plant",
                   "houseplant": "potted_plant", "flower pot": "potted_plant", "carpet": "rug", "couch": "sofa",
                   "armchair": "chair", "stool": "chair", "computer": "laptop", "notebook": "laptop",
                   "coffee cup": "mug", "cup": "mug", "person": "character", "man": "character",
                   "woman": "character", "kid": "character", "dog": "quadruped", "cat": "quadruped",
                   "bear": "quadruped", "sheep": "quadruped", "home": "house", "cottage": "house",
                   "automobile": "car", "lorry": "truck", "spaceship": "rocket", "block": "toy_block",
                   "speaker": "speaker", "bottle": "bottle", "vase": "vase", "single bed": "bed", "double bed": "bed",
                   "pine tree": "tree", "oak": "tree", "bush": "tree", "coffee maker": "coffee_machine",
                   "bench": "bench"}
_QUAD = {"dog": "dog", "cat": "cat", "bear": "bear", "sheep": "sheep"}


def _prefabs():
    from . import solid
    try:
        from . import catalog
        names = catalog._solid_prefab_names(solid)
    except Exception:
        names = []
    if not names:
        names = ["mug", "cup", "bottle", "vase", "speaker", "laptop", "coffee_machine", "table", "chair", "sofa",
                 "bed", "desk_lamp", "pendant_lamp", "bookshelf", "book", "rug", "potted_plant", "room", "counter",
                 "house", "tree", "car", "truck", "rocket", "robot", "character", "quadruped", "toy_block"]
    return [n for n in names if n not in ("room", "wall", "window_frame", "model") and callable(getattr(solid, n, None))]


def _clay_names():
    try:
        from . import clay
        return list(getattr(clay, "PROPS", ())), list(getattr(clay, "SPECIES", ()))
    except Exception:
        return [], []


def _validate_scene3d(d, problems, notes, kind):
    if isinstance(d, list):
        d = {"objects": d}
    out = _sub_keys(d, "scene3d", S3D_KEYS, {"items": "objects", "layers": "objects", "things": "objects",
                                              "furniture": "objects", "models": "objects", "light": "lighting",
                                              "lights": "lighting", "view": "camera", "angle": "camera",
                                              "floor": "ground", "bg": "background"}, problems, notes)
    from . import solid
    prefabs = _prefabs()
    props, animals = _clay_names()
    objs = []
    for i, o in enumerate(out.get("objects") or []):
        p = f"scene3d.objects[{i}]"
        if isinstance(o, str):
            o = {"draw": o}
        if not isinstance(o, dict):
            problems.append(Problem(p, "use {\"draw\": \"bed\", \"at\": \"back right\"}"))
            continue
        oo = _sub_keys(o, p, ("draw", "at", "on", "color", "size", "rotate", "id", "count", "facing", "pose"),
                       {"prefab": "draw", "name": "draw", "object": "draw", "type": "draw", "item": "draw",
                        "position": "at", "where": "at", "place": "at", "colour": "color", "scale": "size",
                        "rotation": "rotate", "turn": "rotate", "angle": "rotate", "yaw": "rotate",
                        "on_top_of": "on", "label": "id", "material": "color"}, problems, notes)
        name = _norm(oo.get("draw") or "")
        if not name:
            problems.append(Problem(p, "needs \"draw\": a prefab name", f"prefabs: {', '.join(prefabs[:20])}"))
            continue
        m = re.match(r"^(\d+|" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")\s+(.+)$", name)
        if m and "count" not in oo:
            oo["count"] = int(m.group(1)) if m.group(1).isdigit() else _NUMBER_WORDS[m.group(1)]
            name = m.group(2)
        key = name.replace(" ", "_")
        src = None
        if key in prefabs:
            src = ("solid", key)
        elif name in _PREFAB_ALIASES and _PREFAB_ALIASES[name] in prefabs:
            src = ("solid", _PREFAB_ALIASES[name])
        elif _singular(name) in _PREFAB_ALIASES or _singular(key) in prefabs:
            k2 = _PREFAB_ALIASES.get(_singular(name), _singular(key))
            if k2 in prefabs:
                src = ("solid", k2)
        if src is None and key in props:
            src = ("clay_prop", key)
        if src is None and key in animals:
            src = ("clay_animal", key)
        if src is None:
            words = name.split()
            for w in reversed(words):
                for cand in (w, _singular(w)):
                    k2 = _PREFAB_ALIASES.get(cand, cand)
                    if k2 in prefabs:
                        src = ("solid", k2)
                        rest = " ".join(x for x in words if x != w)
                        if rest and parse_color(rest) and "color" not in oo:
                            oo["color"] = rest
                        notes.append(f"{p}: read {name!r} as {k2!r}")
                        break
                if src:
                    break
        if src is None:
            pool = prefabs + list(_PREFAB_ALIASES) + props + animals
            close = difflib.get_close_matches(name, pool, n=3, cutoff=0.5)
            close = [_PREFAB_ALIASES.get(c, c) for c in close]
            problems.append(Problem(f"{p}.draw", f"unknown 3D object {oo.get('draw')!r}",
                                    (f"did you mean {' or '.join(repr(c) for c in dict.fromkeys(close))}?" if close else
                                     f"3D objects: {', '.join(prefabs)}")))
            continue
        ob = {"path": p, "src": src, "name": name, "label": str(oo.get("id") or name), "raw": o}
        if name in _QUAD and src == ("solid", "quadruped"):
            ob["kind"] = _QUAD[name]
        if oo.get("color") not in (None, ""):
            c = parse_color(oo["color"])
            if c is None:
                problems.append(Problem(f"{p}.color", f"{oo['color']!r} is not a colour"))
            ob["color"] = c
        at, err = parse_at(oo.get("at"))
        if err:
            problems.append(Problem(f"{p}.at", f"{oo.get('at')!r}: {err}",
                                    "use words (back left, centre, front right, left wall, on desk) or [x, z] metres"))
            at = {"mode": "auto"}
        if at.get("mode") == "words" and at.get("rel") == "on" and "on" not in oo:
            oo["on"] = at.get("target")
        ob["at"] = at
        if oo.get("on"):
            ob["on"] = _norm(oo["on"])
        sz = oo.get("size")
        if sz is not None:
            v, err, _ = parse_size(sz)
            if err:
                problems.append(Problem(f"{p}.size", err))
            else:
                if isinstance(v, tuple):
                    ob["scale"] = v[1]
                elif _num(sz) and 0.2 <= float(sz) <= 5:
                    ob["scale"] = float(min(2.5, max(0.7, float(sz))))
                    if ob["scale"] != float(sz):
                        notes.append(f"{p}.size: a 3D size is a scale (1 = natural size); read {sz} as {ob['scale']}")
                elif v is not None:
                    ob["scale"] = v / 0.25
        if oo.get("rotate") is not None:
            if _num(oo["rotate"]):
                ob["rotate"] = float(oo["rotate"])
            else:
                problems.append(Problem(f"{p}.rotate", "use degrees, e.g. 90"))
        if oo.get("pose"):
            ob["pose"] = oo["pose"]
        n = oo.get("count", 1)
        ob["count"] = int(n) if _num(n) and 1 <= n <= 30 else 1
        objs.append(ob)
    _unique_labels(objs)
    labels = {o["label"]: o for o in objs}
    for o in objs:
        if o.get("on"):
            tgt = o["on"]
            hit = None
            for q in objs:
                if q is o:
                    continue
                if tgt in (q["label"], q["name"], q["src"][1], q["src"][1].replace("_", " ")) or \
                        _PREFAB_ALIASES.get(tgt) == q["src"][1] or _singular(tgt) == q["name"]:
                    hit = q
                    break
            if hit is None:
                problems.append(Problem(f"{o['path']}.on", f"no object called {tgt!r}",
                                        f"objects: {', '.join(repr(x) for x in labels)}"))
                o.pop("on")
            else:
                o["on_obj"] = hit["label"]
    room = out.get("room", kind == "room")
    if room is True or room == {}:
        room = {}
    elif room in (None, False):
        room = None
    elif isinstance(room, str):
        room = {"walls": room}
    if isinstance(room, dict):
        rr = _sub_keys(room, "scene3d.room", ("walls", "floor", "window", "size", "wall2", "door"),
                       {"wall": "walls", "wall_color": "walls", "colour": "walls", "color": "walls",
                        "floor_color": "floor", "windows": "window", "width": "size", "door_side": "door"},
                       problems, notes)
        for k in ("walls", "wall2"):
            if rr.get(k) is not None:
                c = parse_color(rr[k])
                if c is None:
                    problems.append(Problem(f"scene3d.room.{k}", f"{rr[k]!r} is not a colour"))
                rr[k] = c
        if rr.get("floor") is not None:
            f = _norm(rr["floor"])
            if f in ("wood", "planks", "wooden", "parquet", "hardwood"):
                rr["floor"] = "wood"
            elif f in ("carpet", "tiles", "tile", "concrete", "marble", "checker"):
                rr["floor"] = f
            elif parse_color(f):
                rr["floor"] = parse_color(f)
            else:
                problems.append(Problem("scene3d.room.floor", f"{rr['floor']!r} not understood", "use wood, tiles, marble or a colour"))
                rr["floor"] = "wood"
        room = rr
    out["room"] = room
    lt = _norm(out.get("lighting") or ("interior" if room is not None else "studio"))
    lt = {"warm": "interior", "cosy": "interior", "cozy": "interior", "indoor": "interior", "day": "sunny",
          "daylight": "sunny", "sun": "sunny", "evening": "sunset", "soft": "soft_box", "softbox": "soft_box",
          "dark": "night", "moody": "dramatic"}.get(lt, lt).replace(" ", "_")
    if lt not in solid.LIGHTING:
        problems.append(Problem("scene3d.lighting", f"unknown lighting {out.get('lighting')!r}", _did_you_mean(lt, solid.LIGHTING)))
        lt = "studio"
    out["lighting"] = lt
    cam = _norm(out.get("camera") or ("isometric" if kind == "room" else "three-quarter"))
    cam = {"iso": "isometric", "3/4": "three-quarter", "three quarter": "three-quarter", "perspective": "three-quarter",
           "front view": "front", "top down": "top", "overhead": "top", "birds eye": "top", "side view": "side",
           "eye level": "front"}.get(cam, cam)
    if cam not in _CAMERAS:
        problems.append(Problem("scene3d.camera", f"unknown camera {out.get('camera')!r}", _did_you_mean(cam, _CAMERAS)))
        cam = "three-quarter"
    out["camera"] = cam
    for k in ("ground", "background"):
        v = out.get(k)
        if v is None:
            continue
        if isinstance(v, list):
            cs = [parse_color(c) for c in v]
            if all(cs):
                out[k] = cs
                continue
        c = parse_color(v) if not (k == "ground" and _norm(v) in ("wood", "checker", "marble", "none")) else _norm(v)
        if c is None:
            problems.append(Problem(f"scene3d.{k}", f"{v!r} not understood", "use a colour or a list of two colours"))
            out.pop(k)
        else:
            out[k] = c
    if room is not None and 0 < len(objs) < 4:         # a nearly empty room: add what such a room has
        keys = {o["src"][1] for o in objs}
        extra = []
        table = next((o for o in objs if o["src"][1] == "table"), None)
        if table is not None and not keys & {"desk_lamp", "laptop"}:
            extra.append({"draw": "desk_lamp", "on": table["label"]} if "desk" in table["name"] else
                         {"draw": "vase", "on": table["label"]})
        if "rug" not in keys:
            extra.append({"draw": "rug"})
        if "potted_plant" not in keys:
            extra.append({"draw": "potted_plant"})
        for j, e in enumerate(extra):
            ob = {"path": f"scene3d.objects[{len(objs)}]", "src": ("solid", e["draw"]), "name": e["draw"].replace("_", " "),
                  "label": e["draw"].replace("_", " "), "raw": e, "at": {"mode": "auto"}, "count": 1, "added": True}
            if e.get("on"):
                ob["on"], ob["on_obj"] = e["on"], e["on"]
            if e["draw"] == "rug":
                ob["color"] = "#7f9bb5"
            objs.append(ob)
        if extra:
            notes.append("scene3d.objects: a room with few things looks empty; added "
                         + ", ".join(e["draw"].replace("_", " ") + (" on the " + e["on"] if e.get("on") else "")
                                     for e in extra))
    out["objects"] = objs
    if not objs:
        problems.append(Problem("scene3d.objects", "no objects", '"objects": [{"draw": "bed"}, {"draw": "desk"}]',
                                "warn" if room is not None else "error"))
    return out


# ============================================================================================= rendering
LAST = {"errors": [], "placed": {}, "order": [], "warnings": []}


class _Ctx:
    def __init__(self, cv, norm, seed, log):  # noqa: D401
        self.cv, self.norm, self.seed, self.log = cv, norm, seed, log
        self.W, self.H = cv.w, cv.h
        self.S = cv.h / 1080
        self.area = (0.0, 0.0, 1.0, 1.0)
        self.horizon = None
        self.table = None
        self.light = (-0.62, -0.78)
        self.placed = {}           # label -> {"bbox": [u0, v0, u1, v1], "base": (u, v), "cat": ...}
        self.theme = None
        self.grounded = False
        self.text_area = None
        self.card_panel = None


def _err(L, msg, fix=None, severity="fail"):
    LAST["errors"].append(Problem(L["path"] if isinstance(L, dict) else str(L), msg, fix, severity))


def _kit_error(exc):
    msg = str(exc).split("\n")[0]
    return msg[:300]


def render(norm, width=None, height=None, seed=None, log=None, base_dir=None):
    """Draw a validated spec (``validate(spec)[0]``) -> Canvas. Errors inside a layer are collected in
    ``spec.LAST["errors"]`` and the other layers are still drawn."""
    from .canvas import Canvas
    log = log or (lambda m: None)
    LAST["errors"], LAST["placed"], LAST["order"], LAST["warnings"] = [], {}, [], []
    LAST["bg_dark"] = True
    w, h = (width or norm["size"][0]), (height or norm["size"][1])
    seed = norm.get("seed", 1) if seed is None else seed
    cv = Canvas(int(w), int(h))
    ctx = _Ctx(cv, norm, int(seed), log)
    if base_dir:
        norm["base_dir"] = base_dir
    group = norm["group"]
    try:
        if group == "design":
            _render_design(ctx)
        elif group == "scene":
            _render_scene(ctx)
        elif group == "still_life":
            _render_still_life(ctx)
        elif group == "scene3d":
            _render_scene3d(ctx)
            _render_overlay_text(ctx)
        elif group in ("diagram", "chart", "dashboard", "ui"):
            _render_section(ctx, group)
            _render_overlay_text(ctx)
    finally:
        LAST["placed"] = ctx.placed
    _render_post(ctx)
    try:                                               # a small luminance map: what lies under each text
        step = max(1, min(cv.h, cv.w) // 96)
        b = np.clip(np.asarray(cv.buf[::step, ::step, :3], np.float32), 0, 1)
        LAST["lum"] = b @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    except Exception:
        LAST["lum"] = None
    if norm.get("style"):
        log("style")
        from . import stylize
        try:
            stylize.apply(cv, norm["style"], seed=ctx.seed, prompt=norm.get("prompt") or None)
        except Exception as e:
            _err("style", f"style {norm['style']!r} failed: {_kit_error(e)}")
    return cv


# --------------------------------------------------------------------------------------------- themes / colours
def _color_list(text):
    """'deep blue and gold' -> ['#123a7a', '#d4a937'] (colours named in order), or []."""
    if not isinstance(text, str):
        return []
    out = []
    for part in re.split(r"\s*(?:,|&|/|\+|;|\band\b|\bwith\b|\bplus\b)\s*", text.lower()):
        toks = [t for t in part.replace("-", " ").split() if t not in ("palette", "colors", "colours", "tones", "accents", "accent")]
        i = 0
        while i < len(toks):
            hit = None
            for j in range(len(toks), i, -1):
                phrase = " ".join(toks[i:j])
                if _base_color(phrase) is not None or (j - i > 1 and toks[i] in _MODS and _base_color(" ".join(toks[i + 1:j])) is not None):
                    hit = j
                    break
            if hit is None:
                i += 1
                continue
            out.append(parse_color(" ".join(toks[i:hit])))
            i = hit
    return [c for c in out if c]


def _lum(c):
    from .color import rgb
    r, g, b = rgb(c)[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a, b):
    from .color import rgb

    def rl(c):
        c = np.asarray(rgb(c)[:3], float)
        c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    x, y = sorted((rl(a), rl(b)), reverse=True)
    return (x + 0.05) / (y + 0.05)


def _mixhex(a, b, t):
    from .color import rgb
    return _hex(np.asarray(rgb(a)[:3]) * (1 - t) + np.asarray(rgb(b)[:3]) * t)


def _design_theme(norm):
    """A layout theme dict from the spec's theme / palette / background colour."""
    from . import layout
    base = norm.get("theme")
    if isinstance(base, str) and base in layout.THEMES:
        th = layout.theme(base)
    else:
        th = layout.theme("midnight")
    bgc = norm["background"].get("color")
    cols = _color_list(norm.get("palette_spec")) if isinstance(norm.get("palette_spec"), str) else \
        [parse_color(c) for c in (norm.get("palette_spec") or []) if parse_color(c)] if isinstance(norm.get("palette_spec"), list) else []
    pal = norm.get("palette")
    bg = accent = None
    if cols:
        bg = cols[0]
        accent = cols[1] if len(cols) > 1 else None
    elif pal is not None and not (isinstance(base, str) and base in layout.THEMES):
        bg, accent = pal.bg, pal.accent
    if bgc:
        if bg and bg != bgc and accent is None:
            accent = bg
        bg = bgc
    if bg is None:
        return th
    dark = _lum(bg) < 0.45
    ink = "#f7f2e8" if dark else "#1c1f1a"
    if accent is None or abs(_lum(accent) - _lum(bg)) < 0.12:
        accent = (pal.accent if pal is not None and abs(_lum(pal.accent) - _lum(bg)) >= 0.12 else
                  ("#e2bd68" if dark else _mixhex(bg, "#000000", 0.55)))
    for _ in range(8):                                 # accent text must read on the background
        if _contrast(accent, bg) >= 6.0:
            break
        accent = _mixhex(accent, ink, 0.25)
    def _readable(mix, need):                          # ink mixed toward bg, but still clear on a lit backdrop
        while mix > 0.0 and _contrast(_mixhex(ink, bg, mix), bg) < need:
            mix -= 0.05
        return _mixhex(ink, bg, max(0.0, mix))

    th.update(bg=[(0, _mixhex(bg, "#ffffff" if not dark else bg, 0.08 if not dark else 0.0)),
                  (1, _mixhex(bg, "#000000", 0.35 if dark else 0.08))],
              ink=ink, soft=_readable(0.22, 7.0), muted=_readable(0.45, 6.0), accent=accent,
              accent2=_mixhex(accent, ink, 0.4), panel=_mixhex(bg, ink, 0.1), dark=dark)
    if not dark:
        th["title_font"] = "serif bold" if th.get("title_font") == "display" else th["title_font"]
    return th


# --------------------------------------------------------------------------------------------- backgrounds
def _backdrop_palette(norm, bg):
    """Palette for a backdrop: explicit colours from the spec (dark, base, accents), a palette name, or None."""
    spec_p = bg.get("palette") or norm.get("palette_spec")
    cols = []
    if isinstance(spec_p, str):
        cols = _color_list(spec_p)
        if not cols:
            return spec_p
    elif isinstance(spec_p, list):
        cols = [parse_color(c) for c in spec_p if parse_color(c)]
    if bg.get("color"):
        cols = [bg["color"]] + [c for c in cols if c != bg["color"]]
    if not cols and bg.get("sky") in ("night", "twilight", "dusk", "evening"):   # "sky": "night" with a backdrop
        return ["#070b1c", "#14204a", "#2b3a6e", "#9fb2e0"] if bg["sky"] in ("night", "twilight") else             ["#1d1a3a", "#4a3a6e", "#c0607a", "#f2a65a"]
    if not cols:
        return None
    base = cols[0]
    dark = _lum(base) < 0.45
    return [_mixhex(base, "#000000", 0.45) if dark else _mixhex(base, "#ffffff", 0.35), base] + cols[1:]


def _paint_flat_background(ctx, bg):
    cv, norm = ctx.cv, ctx.norm
    from . import backdrop as B, texture as T, elements
    pal = norm.get("palette")
    if bg.get("backdrop"):
        try:
            p = _backdrop_palette(norm, bg)
            n0 = len(getattr(cv, "manifest", None) or [])
            B.backdrop(cv, bg["backdrop"], palette=p, seed=ctx.seed, **(bg.get("options") or {}))
            for e in (getattr(cv, "manifest", None) or [])[n0:]:   # a backdrop is never the cropped subject
                if isinstance(e, dict):
                    e.setdefault("meta", {})["backdrop"] = True
            return True
        except Exception as e:
            _err("background.backdrop", f"backdrop failed: {_kit_error(e)}")
    if bg.get("landscape"):
        try:
            p = _backdrop_palette(norm, bg)
            B.backdrop(cv, bg["landscape"], palette=p, seed=ctx.seed)
            return True
        except Exception as e:
            _err("background.landscape", f"landscape failed: {_kit_error(e)}")
    if bg.get("gradient"):
        g = bg["gradient"]
        cv.gradient([(i / max(1, len(g) - 1), c) for i, c in enumerate(g)])
        return True
    if bg.get("texture"):
        cv.paint(T.texture(bg["texture"], cv.h, cv.w, seed=ctx.seed))
        return True
    if bg.get("sky") and ctx.norm["group"] == "design":
        elements.sky(cv, bg["sky"], horizon=1.0)
        return True
    return False


def _fit_theme_to_canvas(cv, th):
    """Text colours must read on the painted background (a lit backdrop is lighter than its base colour)."""
    buf = np.clip(np.asarray(cv.buf, np.float32), 0, 1)
    step = max(1, min(buf.shape[0], buf.shape[1]) // 120)
    px = buf[::step, ::step].reshape(-1, 3)
    lum = px @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    dark = bool(th.get("dark", True))
    ref = _hex(px[np.argsort(lum)[int((0.9 if dark else 0.1) * (len(lum) - 1))]])
    far = "#ffffff" if dark else "#000000"
    for key in ("ink", "accent", "accent2", "soft", "muted"):
        col = th.get(key)
        if not isinstance(col, str):
            continue
        for _ in range(6):
            if _contrast(col, ref) >= 4.8:
                break
            col = _mixhex(col, far, 0.3)
        th[key] = col


def _render_design(ctx):
    from . import layout
    cv, norm = ctx.cv, ctx.norm
    th = _design_theme(norm)
    ctx.theme = th
    LAST["bg_dark"] = bool(th.get("dark", True))
    texts = [L for L in norm["layers"] if L["type"] == "text"]
    pics = [L for L in norm["layers"] if L["type"] != "text"]
    kind = norm["kind"]
    ctx.log("background")
    bg = norm["background"]
    if not _paint_flat_background(ctx, bg):
        if bg.get("color"):
            layout._bg(cv, th, ctx.seed)
        else:
            layout._bg(cv, th, ctx.seed)
    _fit_theme_to_canvas(cv, th)
    # where the picture goes
    W, H = cv.w, cv.h
    flow = [L for L in texts if L["at"].get("mode") == "auto"]
    if kind == "poster" and flow:
        ctx.area = _poster_picture_area(ctx, flow, th)
    elif kind == "card" and flow:
        _card_panel(ctx, th, pics)
    elif kind == "icons":
        ctx.area = (0.06, 0.18 if flow else 0.08, 0.94, 0.94)
    else:
        ctx.area = (0.06, 0.06, 0.94, 0.94)
    if bg.get("ground") or bg.get("horizon"):
        ctx.grounded = True
        ctx.horizon = bg.get("horizon", 0.7)
    if kind == "icons":
        _grid_auto(ctx, pics)
    back = [L for L in pics if L["type"] == "thing" and L["thing"].cat == "fullframe" and L["thing"].target in _BACK_FRAME]
    for L in back:                                     # a sky or starfield is part of the background
        _draw_layer(ctx, L)
    pics = [L for L in pics if L not in back]
    if kind == "poster" and pics:
        _poster_stage(ctx, pics, th)
    _draw_layers(ctx, pics)
    _draw_texts(ctx, texts, th)


def _poster_stage(ctx, pics, th):
    """Poster picture: one focal subject that fills the free area, the others smaller beside it, all standing on
    one baseline; a lit disc behind the focal subject and a soft floor shadow tie them together."""
    from . import sdf
    cv, S = ctx.cv, ctx.S
    x0, y0, x1, y1 = ctx.area
    ah = y1 - y0
    things = [L for L in pics if L["type"] == "thing" and L["thing"].cat not in ("fullframe", "sky")
              and L["at"].get("mode") in ("auto", "words") and not L["at"].get("target_layer")]
    flyers = [L for L in things if _flies(L)]
    if flyers:                                         # kites, balloons and birds fly: above the stage, or across
        alone = len(flyers) == len(things)             # the whole picture area when nothing stands on a stage
        things = [L for L in things if L not in flyers]
        for L in flyers:
            L["at"].setdefault("v", 0.45 if alone else 0.2)
            if L.get("count", 1) > 1 and not L.get("arrange"):
                L["arrange"] = "scatter"
            if L["size"] is None:
                k = (0.4 if L.get("count", 1) > 1 else 0.6) if alone else (0.22 if L.get("count", 1) > 1 else 0.3)
                L["size"], L["size_rel"] = (y1 - y0) * k * L.get("size_rel", 1.0), 1.0
    if not things:
        return
    focal = max(things, key=lambda L: (L.get("size_rel", 1.0) if L["size"] is None else L["size"] / 0.3, -L["index"]))
    base = y1 - 0.01
    sides = len(things) > 1
    for L in things:
        rel = L.get("size_rel", 1.0)
        k = max(0.82, min(0.94, 0.62 * rel)) if L is focal else max(0.4, min(0.6, 0.5 * rel))
        h = ah * k
        aspect = max(0.2, float(L["thing"].aspect or 1.0))
        room = (x1 - x0) * ((0.5 if sides else 0.9) if L is focal else 0.24)
        h = min(h, room * ctx.W / ctx.H / aspect)
        if L is not focal and L["at"].get("h") is not None:
            L["at"]["h"] = 0.1 if L["at"]["h"] < 0.5 else 0.9   # flank the focal subject
            L["at"]["poster_side"] = True
        if L["size"] is None:
            L["size"], L["size_rel"] = h, 1.0
        centre_word = str(L["at"].get("text", "")).strip().lower() in ("centre", "center", "middle", "the centre",
                                                                        "the center", "in the middle")
        if L["thing"].cat not in ("bird", "icon") and ("v" not in L["at"] or centre_word)                 and L["at"].get("depth") is None:
            L["at"]["poster_base"] = base
        if L is focal:                                 # the focal subject stands centre stage
            L["at"]["h"] = 0.5
            L["at"]["poster_base"] = base
            L["at"].pop("v", None)
    fh = focal["size"]
    fu = x0 + (x1 - x0) * focal["at"].get("h", 0.5)
    for L in pics:                                     # a sun or moon becomes the halo behind the focal subject
        if L["type"] == "thing" and L["thing"].cat == "sky" and L["at"].get("mode") in ("auto", "words")                 and not L["at"].get("target_layer") and L.get("count", 1) == 1:
            L["at"] = {"mode": "point", "values": [fu, base - fh * 0.62], "centre": True}
            L["size"], L["size_rel"] = min(L["size"] or 1.0, fh * 0.3), 1.0
    X, Y = cv.xy()
    cx, cy = fu * ctx.W, (base - fh * 0.52) * ctx.H
    r = min(fh * 0.56 * ctx.H, (x1 - x0) * ctx.W * 0.48)
    acc = th["accent"]
    ctx.log("poster stage")
    bgc = th["bg"][0][1] if isinstance(th.get("bg"), list) else "#202020"
    dark = bool(th.get("dark", True))
    cv.glow(cx, cy, r * 0.9, acc, strength=0.22 if dark else 0.1)
    disc = sdf.fill(sdf.sd_circle(X, Y, cx, cy, r))
    cv.radial(cx, cy - r * 0.35, r * 1.35, [(0, _mixhex(acc, bgc, 0.35 if dark else 0.6)),
                                            (1, _mixhex(acc, bgc, 0.8 if dark else 0.85))], mask=disc, opacity=0.9)
    cv.paint(acc, sdf.fill(sdf.sd_ring(X, Y, cx, cy, r * 1.07, 1.6 * S)), opacity=0.5)
    fy = base * ctx.H
    floor = np.clip(1 - np.maximum(sdf.sd_ellipse(X, Y, cx, fy, (x1 - x0) * ctx.W * 0.42, 14 * S), 0) / (22 * S), 0, 1)
    cv.paint("#000000", floor ** 1.5, opacity=0.32)


_FLYER_WORDS = ("kite", "balloon", "butterfly", "airplane", "aeroplane", "plane", "bird", "drone", "bat", "dragonfly",
                "bee", "ufo", "glider", "paper plane")


def _flies(L):
    th = L.get("thing")
    if th is None:
        return False
    if th.cat == "bird" or L["opts"].get("pose") == "fly":
        return True
    name = f"{th.name} {L.get('name', '')}".lower()
    return any(re.search(r"\b" + w + r"s?\b", name) for w in _FLYER_WORDS)


def _card_panel(ctx, th, pics):
    """Card layout: an accent panel that holds the illustration (or an ornament), text column beside it."""
    from . import sdf
    cv, S = ctx.cv, ctx.S
    W, H = cv.w, cv.h
    wide = W >= H * 1.15
    if wide:
        panel = (0.6, 0.08, 0.955, 0.92)
        ctx.text_area = (0.075, 0.1, 0.55, 0.9)
    else:
        panel = (0.06, 0.045, 0.94, 0.5 if H > W * 1.15 else 0.46)
        ctx.text_area = (0.08, panel[3] + 0.04, 0.92, 0.95)
    ctx.card_panel = panel
    acc, bg = th["accent"], th["bg"][0][1] if isinstance(th.get("bg"), list) else "#f4efe6"
    dark = bool(th.get("dark", True))
    if dark:
        tint = _mixhex(bg, "#ffffff", 0.08)
        deep = _mixhex(tint, acc, 0.5)
    else:
        tint = _mixhex(acc, bg, 0.72)
        deep = _mixhex(acc, bg, 0.45)
    X, Y = cv.xy() if callable(getattr(cv, "xy", None)) else np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    x0, y0, x1, y1 = panel[0] * W, panel[1] * H, panel[2] * W, panel[3] * H
    cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    r = 0.045 * min(W, H)
    ctx.log("card panel")
    d = sdf.sd_box(X, Y, cx, cy + 10 * S, hw, hh, radius=r)
    cv.paint("#000000", np.clip(1 - np.maximum(d, 0) / (28 * S), 0, 1) ** 2 * 0.22)   # soft drop shadow
    box = sdf.fill(sdf.sd_box(X, Y, cx, cy, hw, hh, radius=r))
    cv.paint(tint, box)
    cv.linear((x0, y0), (x1, y1), [(0, _mixhex(tint, "#ffffff", 0.18 if not dark else 0.06)),
                                   (1, _mixhex(tint, deep if not dark else "#000000", 0.35 if not dark else 0.2))], mask=box)
    rad = min(hw, hh) * 0.82
    disc = sdf.fill(sdf.sd_circle(X, Y, cx, cy + hh * 0.04, rad)) * box
    cv.paint(_mixhex(tint, "#ffffff" if not dark else acc, 0.35 if not dark else 0.22), disc, opacity=0.7)
    ring = sdf.fill(sdf.sd_ring(X, Y, cx, cy + hh * 0.04, rad * 1.18, 2.2 * S)) * box
    cv.paint(deep, ring, opacity=0.35)
    rng = np.random.default_rng(ctx.seed + 17)
    for _ in range(5):                                  # a few confetti dots inside the panel
        px, py = rng.uniform(x0 + r, x1 - r), rng.uniform(y0 + r, y1 - r)
        if math.hypot(px - cx, py - cy) < rad * 1.05:
            continue
        cv.paint(acc, sdf.fill(sdf.sd_circle(X, Y, px, py, rng.uniform(4, 9) * S)) * box, opacity=0.55)
    if not pics:                                        # no illustration: the title's initial as a monogram
        title = next((L["text"] for L in ctx.norm["layers"] if L["type"] == "text" and L["role"] == "title"), "")
        glyph = next((ch for ch in str(title) if ch.isalnum()), "")
        if glyph:
            cv.text(glyph.upper(), (cx, cy + hh * 0.04), size=rad * 1.25, color=acc, anchor="mm", bold=True,
                    font=th.get("title_font"), audit=False)
    pad = 0.08
    ctx.area = (panel[0] + pad * (panel[2] - panel[0]), panel[1] + pad * (panel[3] - panel[1]),
                panel[2] - pad * (panel[2] - panel[0]), panel[3] - pad * 0.6 * (panel[3] - panel[1]))
    ctx.card_ink = acc


def _card_texts(ctx, flow, th):
    """Card text column: kicker, title, other text, an accent rule, then the author line with an initial."""
    from . import layout, sdf
    cv, S = ctx.cv, ctx.S
    W, H = cv.w, cv.h
    tx0, ty0, tx1, ty1 = ctx.text_area
    wide = W >= H * 1.15
    foot_roles = ("byline", "footer", "date", "meta")
    head = [L for L in flow if L["role"] not in foot_roles]
    foot = [L for L in flow if L["role"] in foot_roles]
    foot_h = 0.13 if foot else 0.0
    rule_gap = 0.075
    hb = (tx0, ty0, tx1, ty1 - foot_h - rule_gap)
    boxes = layout.stack(cv, _px(ctx, hb), _text_items(head, th, ctx), align="left",
                         valign="bottom" if wide else "top", theme=th) if head else []
    bottom = max((b[3] for b in (boxes or []) if b is not None), default=hb[3] * H) / H
    ry = min(bottom + 0.035, ty1 - foot_h - 0.02)
    X, Y = cv.xy() if callable(getattr(cv, "xy", None)) else np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rw = 0.09 * W if wide else 0.14 * W
    cv.paint(th["accent"], sdf.fill(sdf.sd_box(X, Y, tx0 * W + rw / 2, ry * H, rw / 2, 3.2 * S, radius=3.2 * S)))
    if not foot:
        return
    fy0 = ry + 0.045
    fb = (tx0, fy0, tx1, min(1.0, fy0 + foot_h))
    by = next((L for L in foot if L["role"] == "byline"), None)
    if by is not None:
        name = re.sub(r"^(by|von|par|por)\s+", "", by["text"].strip(), flags=re.I)
        initial = (name[:1] or "?").upper()
        d = 0.075 * H
        ax, ay = tx0 * W + d / 2, (fy0 + foot_h / 2) * H - (0.012 * H if len(foot) > 1 else 0)
        cv.paint(th["accent"], sdf.fill(sdf.sd_circle(X, Y, ax, ay, d / 2)))
        ink = "#ffffff" if _contrast("#ffffff", th["accent"]) >= _contrast("#1c1f1a", th["accent"]) else "#1c1f1a"
        cv.text(initial, (ax, ay), size=d * 0.5, color=ink, anchor="mm", bold=True, audit=False)
        fb = (tx0 + d / W + 0.015, fy0, tx1, fb[3])
    items = _text_items(foot, th, ctx)
    fs = 30 * min(W, H) / 1080
    for it, L in zip(items, sorted(foot, key=lambda L: (_ROLE_ORDER.get(L["role"], 5), L["index"]))):
        it["role"] = "body"
        it["size"] = fs if L["role"] == "byline" else fs * 0.9
        if not L.get("color") and L["role"] != "byline":
            it["color"] = th["soft"]
        if L["role"] == "byline":
            it["bold"] = True
            if not L.get("color"):
                it["color"] = th["ink"]
    layout.stack(cv, _px(ctx, fb), items, align="left", valign="middle", theme=th)


def _grid_auto(ctx, pics):
    auto = [L for L in pics if L["at"].get("mode") == "auto" and L["count"] == 1]
    n = len(auto)
    if not n:
        return
    x0, y0, x1, y1 = ctx.area
    aspect = (x1 - x0) * ctx.W / max(1, (y1 - y0) * ctx.H)
    cols = max(1, min(n, int(round(math.sqrt(n * aspect)))))
    rows = int(math.ceil(n / cols))
    cw, ch = (x1 - x0) / cols, (y1 - y0) / rows
    for i, L in enumerate(auto):
        r, c = divmod(i, cols)
        cx = x0 + (c + 0.5) * cw
        cy = y0 + (r + 0.5) * ch
        L["at"] = {"mode": "point", "values": [cx, cy], "centre": True}
        if L["size"] is None:
            L["size"] = min(ch * 0.62, cw * ctx.W / ctx.H * 0.62)


def _poster_picture_area(ctx, texts, th):
    """Dry-run the poster template on a scratch canvas to find its free picture box (fractions)."""
    from .canvas import Canvas
    from . import layout
    content = _poster_content(texts)
    if not content.get("title"):
        return (0.07, 0.3, 0.93, 0.8)
    scratch = Canvas(ctx.W, ctx.H)
    try:
        from . import manifest
        on = manifest._STATE.get("on")
        manifest._STATE["on"] = False
    except Exception:
        on = None
    try:
        boxes = layout.template("poster", scratch, theme=th, background=False, seed=ctx.seed, **content)
        b = boxes["image"]
        area = (b[0] / ctx.W, b[1] / ctx.H, b[2] / ctx.W, b[3] / ctx.H)
    except Exception:
        area = (0.07, 0.3, 0.93, 0.8)
    finally:
        if on is not None:
            manifest._STATE["on"] = on
    if area[3] - area[1] < 0.12:
        area = (area[0], area[1], area[2], min(0.95, area[1] + 0.2))
    return area


def _poster_content(texts):
    c = {"details": []}
    extra = []
    for L in texts:
        r, t = L["role"], L["text"]
        if r == "title" and not c.get("title"):
            c["title"] = t
        elif r in ("subtitle",) and not c.get("subtitle"):
            c["subtitle"] = t
        elif r == "subtitle":                          # a tagline split in two layers reads as one line
            c["subtitle"] = c["subtitle"] + " " + t
        elif r in ("kicker", "badge") and not c.get("kicker"):
            c["kicker"] = t
        elif r in ("footer", "byline") and not c.get("footer"):
            c["footer"] = t
        elif r in ("details", "date", "meta", "label", "caption", "body"):
            c["details"].append(t)
        elif r == "title":
            c["subtitle"] = (c.get("subtitle") + " · " + t) if c.get("subtitle") else t
        else:
            extra.append(t)
    if extra:
        c["details"] += extra
    if len(c["details"]) > 4:
        c["details"] = c["details"][:3] + [" · ".join(c["details"][3:])]
    if not c["details"]:
        c.pop("details")
    return c


_STACK_ROLE = {"title": "title", "subtitle": "subtitle", "kicker": "kicker", "badge": "kicker", "details": "body",
               "body": "body", "caption": "caption", "label": "caption", "footer": "footer", "byline": "subtitle",
               "quote": "quote", "date": "meta", "meta": "meta"}
_ROLE_ORDER = {"badge": 0, "kicker": 1, "title": 2, "quote": 2, "subtitle": 3, "body": 4, "details": 5, "date": 6,
               "byline": 7, "caption": 8, "label": 8, "meta": 9, "footer": 10}


def _text_items(texts, th, ctx):
    items = []
    for L in sorted(texts, key=lambda L: (_ROLE_ORDER.get(L["role"], 5), L["index"])):
        role = _STACK_ROLE.get(L["role"], "body")
        it = {"text": L["text"], "role": role}
        if L.get("color"):
            it["color"] = L["color"]
        elif th is not None:
            it["color"] = {"kicker": th["accent"], "subtitle": th["soft"], "meta": th["soft"],
                           "caption": th["muted"], "footer": th["muted"]}.get(role, th["ink"])
            if L["role"] == "byline":
                it["color"] = th["soft"]
        if L["role"] == "badge":
            it["plate"] = dict(kind="pill", color=th["accent"] if th else "#c8284f")
            it["color"] = th["bg"][-1][1] if th and isinstance(th.get("bg"), list) else "#ffffff"
            it["bold"] = True
        if th is not None and role in ("title", "quote"):
            it["font"] = th["title_font"]
        for k in ("font", "bold", "plate"):
            if k in L["opts"]:
                it[k] = L["opts"][k]
        if L["opts"].get("shadow"):
            it["shadow"] = dict(dx=0, dy=4 * ctx.S, blur=16 * ctx.S, opacity=0.5)
        items.append(it)
    return items


def _draw_texts(ctx, texts, th, scene=False):
    """Text layers: explicit boxes / points, then the flow (template or stack)."""
    from . import layout
    cv = ctx.cv
    if not texts:
        return
    fixed = [L for L in texts if L["at"].get("mode") not in ("auto",)]
    flow = [L for L in texts if L["at"].get("mode") == "auto"]
    kind = ctx.norm["kind"]
    if flow:
        ctx.log("text")
        try:
            if kind == "card" and not scene and getattr(ctx, "card_panel", None):
                _card_texts(ctx, flow, th)
            elif kind == "poster" and not scene and any(L["role"] == "title" for L in flow):
                layout.template("poster", cv, theme=th, background=False, seed=ctx.seed, **_poster_content(flow))
            else:
                if scene:
                    tops = [L for L in flow if L["role"] in ("title", "subtitle", "kicker", "badge", "quote")]
                    bots = [L for L in flow if L not in tops]
                    shadow = True
                    for grp, box, va in ((tops, (0.06, 0.04, 0.94, 0.26), "top"),
                                         (bots, (0.06, 0.8, 0.94, 0.96), "bottom")):
                        if grp:
                            items = _text_items(grp, th, ctx)
                            for it in items:
                                it.setdefault("color", "#fffaf0")
                                if shadow and "plate" not in it:
                                    it["shadow"] = dict(dx=0, dy=3 * ctx.S, blur=14 * ctx.S, opacity=0.6)
                            layout.stack(cv, _px(ctx, box), items, align="center", valign=va, theme=th)
                elif kind == "wallpaper" and not ctx.text_area:
                    _wallpaper_texts(ctx, flow, th)
                else:
                    box = ctx.text_area or ((0.08, 0.08, 0.92, 0.92) if kind != "icons" else (0.06, 0.03, 0.94, 0.16))
                    align = "left" if ctx.text_area and ctx.W >= ctx.H else "center"
                    va = "middle" if kind != "icons" else "middle"
                    layout.stack(cv, _px(ctx, box), _text_items(flow, th, ctx), align=align, valign=va, theme=th)
        except Exception as e:
            for L in flow:
                _err(L, f"text layout failed: {_kit_error(e)}")
    for L in fixed:
        ctx.log(f"text {L['label']}")
        at = L["at"]
        items = _text_items([L], th, ctx)
        try:
            if at["mode"] == "box":
                box = _px(ctx, at["values"])
            else:
                u, v = _word_point(ctx, at, centre=True)
                hh = (L["size"] or {"title": 0.14, "subtitle": 0.07}.get(L["role"], 0.05) * L.get("size_rel", 1.0))
                box = _px(ctx, (max(0.02, u - 0.42), max(0.0, v - hh / 2), min(0.98, u + 0.42), min(1.0, v + hh / 2)))
            if scene:
                for it in items:
                    it.setdefault("color", "#fffaf0")
                    it["shadow"] = dict(dx=0, dy=3 * ctx.S, blur=14 * ctx.S, opacity=0.6)
            al = L["opts"].get("align", "center")
            al = al if al in ("left", "center", "right") else "center"
            layout.stack(cv, box, items, align=al, valign="middle", theme=th)
        except Exception as e:
            _err(L, f"text failed: {_kit_error(e)}")


def _calm_band(cv, bands):
    """The band (x0, y0, x1, y1 fractions) with the least detail on the canvas."""
    buf = np.clip(np.asarray(cv.buf, np.float32), 0, 1)
    lum = buf @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    gy = np.abs(np.diff(lum, axis=0))[:, :-1] + np.abs(np.diff(lum, axis=1))[:-1, :]
    H, W = gy.shape
    def busy(b):
        x0, y0, x1, y1 = b
        return float(gy[int(y0 * H):max(int(y0 * H) + 1, int(y1 * H)), int(x0 * W):max(int(x0 * W) + 1, int(x1 * W))].mean())
    return min(bands, key=busy)


def _wallpaper_texts(ctx, flow, th):
    """Wallpaper type: a quiet line of type in the calmest band (the sky above a landscape, the middle of an
    abstract backdrop), sized as a wallpaper word rather than a poster headline, in ink that reads on what is
    under it."""
    from . import layout
    cv = ctx.cv
    bg = ctx.norm["background"]
    scenic = str(bg.get("backdrop") or bg.get("landscape") or "") in LANDSCAPES or bg.get("sky")
    bands = ([(0.12, 0.12, 0.88, 0.40), (0.12, 0.20, 0.88, 0.48)] if scenic else
             [(0.2, 0.33, 0.8, 0.67), (0.12, 0.12, 0.88, 0.40), (0.12, 0.60, 0.88, 0.88)])
    box = _calm_band(cv, bands)
    buf = np.clip(np.asarray(cv.buf, np.float32), 0, 1)
    x0, y0, x1, y1 = box
    under = buf[int(y0 * cv.h):int(y1 * cv.h), int(x0 * cv.w):int(x1 * cv.w)].reshape(-1, 3)
    ref = _hex(under.mean(0)) if under.size else "#808080"
    light_ink = _contrast("#fffaf2", ref) >= _contrast("#1b1d24", ref)
    ink = "#fffaf2" if light_ink else "#1b1d24"
    items = _text_items(flow, th, ctx)
    for it in items:
        if not any(L["text"] == it["text"] and L.get("color") for L in flow):
            it["color"] = ink
        if it["role"] in ("title", "quote") and not any(L["opts"].get("font") for L in flow):
            it["font"] = "serif"                       # a quiet word, not a poster headline
        if light_ink and "plate" not in it:
            it["shadow"] = dict(dx=0, dy=3 * ctx.S, blur=18 * ctx.S, opacity=0.45)
    layout.stack(cv, _px(ctx, box), items, align="center", valign="middle", theme=th)


def _px(ctx, box):
    x0, y0, x1, y1 = box
    return (x0 * ctx.W, y0 * ctx.H, x1 * ctx.W, y1 * ctx.H)


def _render_scene(ctx):
    cv, norm = ctx.cv, ctx.norm
    bg = norm["background"]
    from . import elements, flora, texture
    flat = any(bg.get(k) for k in ("backdrop", "color", "gradient", "texture"))
    ctx.horizon = bg.get("horizon", 0.62 if norm["kind"] != "landscape" else 0.6)
    ctx.grounded = True
    sun = [L for L in norm["layers"] if L["type"] == "thing" and L["thing"].target in ("sun", "moon")]
    if sun:
        at = sun[0]["at"]
        u = at.get("h", at["values"][0] if at.get("mode") == "point" else 0.7)
        ctx.light = (float(np.clip((u - 0.5) * 2, -0.9, 0.9)) or -0.3, -0.75)
    try:
        from . import form
        form.set_light(direction=ctx.light, elevation=40)
    except Exception:
        pass
    ctx.log("sky")
    if flat:
        _paint_flat_background(ctx, bg)
    elif bg.get("landscape"):
        _paint_flat_background(ctx, bg)
    else:
        try:
            elements.sky(cv, bg.get("sky", "day"), horizon=ctx.horizon)
        except Exception as e:
            _err("background.sky", f"sky failed: {_kit_error(e)}")
    sky_fx = [L for L in norm["layers"] if L["type"] == "thing" and L["thing"].cat == "fullframe"
              and L["thing"].target in ("stars", "aurora", "cirrus", "cloud bands", "storm clouds")]
    rest = [L for L in norm["layers"] if L not in sky_fx]
    for L in sky_fx:
        _draw_layer(ctx, L)
    if not flat and not bg.get("landscape"):
        if bg.get("far"):
            ctx.log("distance")
            _paint_far(ctx, bg["far"], bg.get("sky", "day"))
        if bg.get("ground"):
            ctx.log("ground")
            _paint_ground(ctx, bg["ground"])
    elif bg.get("ground"):
        ctx.log("ground")
        _paint_ground(ctx, bg["ground"])
    _draw_layers(ctx, [L for L in rest if L["type"] != "text"])
    _draw_texts(ctx, [L for L in rest if L["type"] == "text"], _scene_theme(ctx), scene=True)


def _scene_theme(ctx):
    from . import layout
    th = layout.theme("midnight")
    th.update(ink="#fffaf0", soft="#f3ead8", muted="#e8dcc6")
    return th


def _sky_haze(sky):
    return {"sunset": "#e8a07a", "dusk": "#9a7fa0", "dawn": "#e9b7a0", "night": "#2a3350", "twilight": "#4d5a86",
            "golden": "#e8c38a", "storm": "#7d8794", "overcast": "#b7bec6", "morning": "#c9d9e6"}.get(sky, "#b9cde0")


def _paint_far(ctx, far, sky):
    from .noise import fbm1d
    from .color import clamp01
    cv = ctx.cv
    W, H, hz = cv.w, cv.h, ctx.horizon
    X, Y = cv.xy()
    haze = _sky_haze(sky)
    night = sky in ("night", "twilight", "storm")
    if far in ("hills", "mountains"):
        tints = ["#6f8f6a", "#4f7350", "#3b5e3e"] if far == "hills" else ["#8090a8", "#5f6f88", "#4a5a70"]
        if night:
            tints = ["#2c3550", "#232b42", "#1b2236"]
        amp = 0.05 if far == "hills" else 0.17
        for i, col in enumerate(tints[:2]):
            col = _mixhex(col, haze, 0.55 - 0.25 * i)
            base = hz - amp * (1.2 - 0.45 * i)
            ridge = (base + amp * 0.5 * fbm1d(W, (520 - 160 * i) * ctx.S * (0.6 if far == "mountains" else 1.0), 5,
                                               seed=ctx.seed + 7 + i)) * H
            cv.paint(col, clamp01(Y - ridge[None, :] + 0.5))
    elif far == "forest":
        from . import flora
        try:
            flora.forest(cv, region=(-0.05, hz - 0.06, 1.05, hz + 0.01), n=45, height=(0.05, 0.12),
                         haze=(haze, 0.85 if night else 0.5), seed=ctx.seed + 2, light=ctx.light,
                         **_far_forest_kw(ctx.norm))
            if night:                                  # trees at night are dark shapes, not sunlit green
                band = clamp01((Y / H - (hz - 0.2)) / 0.05) * clamp01((hz + 0.012 - Y / H) / 0.01)
                cv.paint("#0d1428", band, opacity=0.72)
        except Exception as e:
            _err("background.far", f"forest failed: {_kit_error(e)}")
    elif far == "city":
        rng = np.random.default_rng(ctx.seed + 5)
        col = _mixhex("#4a5466" if not night else "#1c2233", haze, 0.45)
        x = 0.0
        m = np.zeros((H, W), np.float32)
        while x < 1.0:
            bw = rng.uniform(0.025, 0.07)
            bh = rng.uniform(0.05, 0.2)
            x0, x1 = int(x * W), int(min(1.0, x + bw) * W)
            y0 = int((hz - bh) * H)
            m[max(0, y0):int(hz * H) + 2, x0:x1] = 1.0
            x += bw + rng.uniform(0.0, 0.01)
        cv.paint(col, m)


_CONIFER = re.compile(r"\b(pines?|firs?|spruces?|conifers?|coniferous|evergreens?|taiga|christmas trees?)\b")


def _far_forest_kw(norm):
    """The far forest: conifers when the request names them (a snowy pine forest), bare trees in other winters."""
    prompt = (norm.get("prompt") or "").lower()
    if _CONIFER.search(prompt):
        return {"species": ("spruce", "fir"), **({"season": "winter"} if _wintry(norm) else {})}
    return {"season": "winter", "species": ("oak", "maple")} if _wintry(norm) else {}


def _wintry(norm):
    """Snow on the ground or winter / snow in the request: trees are drawn bare and frosted."""
    g = str((norm.get("background") or {}).get("ground") or "").lower()
    return "snow" in g or "ice" in g or bool(re.search(r"\b(winter|snow|snowy|wintry|frost|frosty)\b",
                                                         (norm.get("prompt") or "").lower()))


def _paint_ground(ctx, ground):
    from . import texture
    from .color import clamp01
    cv = ctx.cv
    H, W, hz = cv.h, cv.w, ctx.horizon
    X, Y = cv.xy()
    mask = clamp01(Y - hz * H + 0.5)
    name, tint = ground
    try:
        if name == "color":
            from .color import rgb
            c = tint
            cv.paint(_mixhex(c, "#ffffff", 0.08), mask)
            cv.paint(_mixhex(c, "#000000", 0.25), mask * clamp01((Y / H - hz) / max(1e-3, 1 - hz)) * 0.6)
        else:
            texture.fill(cv, mask, name, uv=texture.plane_uv(H, W, horizon=hz - 0.1), tint=tint, light=ctx.light,
                         seed=ctx.seed)
            if name in ("lawn", "meadow", "moss") and cv.h >= 300:
                from . import flora
                flora.grass(cv, region=(0.0, hz + 0.01, 1.0, hz + 0.1), tufts=45, height=0.035,
                            seed=ctx.seed + 3, light=ctx.light)
        if (ctx.norm.get("background") or {}).get("sky") in ("night", "twilight"):
            cv.paint("#0b1124", mask, opacity=0.55)    # the ground at night is in shadow, not daylit
    except Exception as e:
        _err("background.ground", f"ground failed: {_kit_error(e)}")


def _render_still_life(ctx):
    cv, norm = ctx.cv, ctx.norm
    from . import form
    from .color import mix
    bg = norm["background"]
    ctx.light = (-0.85, -0.5)
    form.set_light(direction=ctx.light, elevation=32)
    ctx.log("wall")
    if not _paint_flat_background(ctx, bg):
        wall = bg.get("wall") or bg.get("color")
        if wall is None and norm.get("palette") is not None:
            wall = norm["palette"].bg
        wall = wall or "#d8c6ac"
        U, V = cv.uv()
        cv.paint(mix([(0.0, _mixhex(wall, "#ffffff", 0.12)), (1.0, _mixhex(wall, "#000000", 0.42))],
                     np.clip(U * 0.9 + V * 0.3, 0, 1)))
        cv.glow(0.2 * cv.w, 0.25 * cv.h, 0.25 * cv.h, "#ffe2b8", 0.25)
    ctx.log("table")
    t = bg.get("table", "wood")
    try:
        ctx.table = form.tabletop(cv, kind=t, top=0.6, front=0.96, horizon=0.42, seed=ctx.seed)
    except Exception as e:
        _err("background.table", f"table failed: {_kit_error(e)}")
        ctx.table = None
    ctx.horizon = 0.6
    ctx.grounded = True
    things = [L for L in norm["layers"] if L["type"] == "thing" and L["size"] is None]
    if things and len(things) == len([L for L in norm["layers"] if L["type"] == "thing"]):
        big = max(L["thing"].size * L.get("size_rel", 1.0) for L in things)
        if big < 0.2:                                  # only small fruit: draw the group larger so it fills the table
            k = min(1.9, 0.22 / max(big, 1e-3))
            for L in things:
                L["size_rel"] = L.get("size_rel", 1.0) * k
    _draw_layers(ctx, [L for L in norm["layers"] if L["type"] != "text"])
    _draw_texts(ctx, [L for L in norm["layers"] if L["type"] == "text"], _scene_theme(ctx), scene=True)


def _render_overlay_text(ctx):
    texts = [L for L in ctx.norm["layers"] if L["type"] == "text"]
    if texts:
        th = _design_theme(ctx.norm) if ctx.norm["group"] != "scene3d" else _scene_theme(ctx)
        _draw_texts(ctx, texts, th, scene=True)


# --------------------------------------------------------------------------------------------- placement
def _est_box(ctx, cat, u, v, h, aspect, anchor):
    """Predicted box (fractions) of a thing of height h at (u, v)."""
    wf = h * aspect * ctx.H / ctx.W
    if anchor == "base":
        return [u - wf / 2, v - h, u + wf / 2, v]
    return [u - wf / 2, v - h / 2, u + wf / 2, v + h / 2]


def _is_grounded(ctx, L):
    th = L["thing"] if L["type"] == "thing" else None
    if th is None:
        return False
    at = L["at"]
    if at.get("sky") or at.get("rel") == "above":
        return False
    if at.get("ground"):
        return True
    if th.cat in ("sky", "fullframe", "icon", "pictogram"):
        return bool(ctx.grounded and th.cat == "pictogram" and (at.get("depth") is not None or ctx.table is not None))
    if th.cat == "bird":
        return at.get("rel") in ("on",) or L["opts"].get("pose") in ("stand", "walk", "peck", "sit")
    return ctx.grounded


def _word_point(ctx, at, centre=False):
    x0, y0, x1, y1 = ctx.area
    u = at.get("h", 0.5)
    v = at.get("v", 0.5)
    if at.get("depth") is not None and "v" not in at:
        v = 0.3 + 0.6 * at["depth"]
    return x0 + (x1 - x0) * u, y0 + (y1 - y0) * v


def _ground_v(ctx, depth):
    hz = ctx.horizon if ctx.horizon is not None else 0.6
    if ctx.table is not None:
        return 0.6 + (0.96 - 0.6) * depth
    return hz + (0.975 - hz) * depth


def _depth_of(at):
    if at.get("depth") is not None:
        return at["depth"]
    if at.get("v") is not None:
        return float(np.clip((at["v"] - 0.15) / 0.75, 0.05, 0.95))
    return None


def _base_place(ctx, L, auto_slot):
    """(u, v, anchor) for one layer without count; v is the base for grounded things."""
    th = L["thing"] if L["type"] == "thing" else None
    at = L["at"]
    grounded = _is_grounded(ctx, L)
    mode = at.get("mode")
    if mode == "point":
        u, v = at["values"]
        if grounded and ctx.horizon is not None and ctx.table is None and ctx.norm["group"] == "scene" \
                and v < ctx.horizon + 0.02 and not at.get("centre"):
            v = ctx.horizon + 0.02 + 0.1 * max(0.0, v - 0.2)
            LAST["warnings"].append(f"{L['path']} {L['label']!r} stands on the ground, so its base moved down to v={v:.2f} "
                     f"(the ground starts at {ctx.horizon:.2f})")
        return u, v, ("centre" if at.get("centre") or not grounded else "base")
    if mode == "pixels":
        u, v = at["values"][0] / ctx.W, at["values"][1] / ctx.H
        return u, v, ("base" if grounded else "centre")
    if mode == "box":
        u0, v0, u1, v1 = at["values"]
        return (u0 + u1) / 2, (v1 if grounded else (v0 + v1) / 2), ("base" if grounded else "centre")
    x0, y0, x1, y1 = ctx.area
    hu = at.get("h")
    if hu is None:
        hu = auto_slot
    if ctx.norm["group"] == "design" and not at.get("poster_side"):   # "far left" in a design stays on the page
        hu = min(max(hu, 0.14), 0.86)
    if ctx.table is not None:                          # a still life is one group in the middle of the table
        hu = 0.27 + 0.46 * hu
    u = x0 + (x1 - x0) * hu
    if L.get("group_u") is not None:
        u = L["group_u"]
    if at.get("poster_base") is not None:             # poster: everything stands on the stage line
        return u, at["poster_base"], "base"
    if grounded:
        d = _depth_of(at)
        if d is None:
            d = L.get("depth", 0.55)
        return u, _ground_v(ctx, d), "base"
    # sky / floating
    if ctx.grounded and ctx.horizon is not None and (th is not None and th.cat in ("sky", "bird") or at.get("sky")):
        hz = ctx.horizon
        vw = at.get("v")
        if vw is None:
            vw = 0.3 if not at.get("sky") else 0.3
        v = 0.05 + (hz - 0.12) * min(vw, 0.95) * 1.1
        return u, min(v, hz - 0.06), "centre"
    v = at.get("v", 0.5)
    return u, y0 + (y1 - y0) * v, "centre"


def _relate(ctx, L, rel, T, h, aspect, grounded):
    """Position relative to a placed layer ``T`` (its record in ctx.placed)."""
    b = T["bbox"]
    tw, th_ = b[2] - b[0], b[3] - b[1]
    tcx = (b[0] + b[2]) / 2
    base_v = T["base"][1] if T.get("base") else b[3]
    wf = h * aspect * ctx.H / ctx.W
    if str(L["opts"].get("pose", "")) in ("sleep", "lie", "rest") and L.get("thing") is not None             and L["thing"].kit == "creature":
        wf *= 1.5                                      # lying down is wider than the standing outline
    gap = 0.015
    if rel == "on":
        if T.get("cat") == "plant" and th_ > 0.25:
            return tcx + 0.18 * tw, b[1] + 0.3 * th_, "base"
        return tcx, b[1] + 0.03 * th_, "base"
    if rel == "under":
        if T.get("cat") == "plant" and th_ > 0.25:
            # beside the trunk and a step in front of it, on the ground: never on the roots or the trunk base
            trunk = T["base"][0] if T.get("base") else tcx
            side = 1 if trunk < 0.6 else -1
            off = 0.07 * tw + 0.5 * wf + 0.035          # half the root flare + half the thing + a clear gap
            if not 0.04 < trunk + side * off < 0.96:
                side = -side
            return trunk + side * off, min(0.97, base_v + 0.06), "base"
        return tcx, b[3] + (0.02 + h if not grounded else 0.03), ("base" if grounded else "centre")
    if rel == "above":
        return tcx, b[1] - h / 2 - 0.03, "centre"
    if rel in ("left of", "right of", "next to"):
        side = -1 if rel == "left of" else 1
        if rel == "next to" and b[2] + wf + gap > 0.97:
            side = -1
        u = (b[0] - gap - wf / 2) if side < 0 else (b[2] + gap + wf / 2)
        if grounded:
            return u, base_v, "base"
        return u, (b[1] + b[3]) / 2, "centre"
    if rel == "behind":
        side = 1 if tcx < 0.5 else -1
        u = tcx + side * 0.3 * tw
        if grounded:
            return u, max((ctx.horizon or 0.5) + 0.01, base_v - 0.05), "base"
        return u, (b[1] + b[3]) / 2 - 0.03, "centre"
    if rel == "in front of":
        if grounded:
            return tcx + 0.05 * tw, min(0.985, base_v + 0.06), "base"
        return tcx, (b[1] + b[3]) / 2 + 0.03, "centre"
    if rel == "in":
        return tcx, (b[1] + b[3]) / 2 + h / 2, "base" if grounded else "centre"
    return tcx, base_v, "base"


def _default_size(ctx, L, v=None, grounded=False):
    th = L["thing"] if L["type"] == "thing" else None
    if L["size"] is not None:
        return L["size"]
    if L["at"].get("mode") == "box":
        u0, v0, u1, v1 = L["at"]["values"]
        return max(0.02, v1 - v0)
    s = th.size if th is not None else 0.25
    if ctx.norm["group"] == "design" and th is not None and th.cat in ("pictogram", "icon", "sky", "animal", "person", "plant", "object"):
        x0, y0, x1, y1 = ctx.area
        s = min(max(s, 0.3 * (y1 - y0)), 0.6 * (y1 - y0))
        if th.cat == "sky":
            s = 0.45 * (y1 - y0)
    if grounded and v is not None and ctx.horizon is not None and ctx.table is None and ctx.norm["group"] == "scene":
        depth = (v - ctx.horizon) / max(0.05, 0.975 - ctx.horizon)
        s *= 0.55 + 0.6 * float(np.clip(depth, 0, 1))
    return float(min(0.95, s * L.get("size_rel", 1.0)))


def _instances(ctx, L, u, v, anchor, h, aspect, grounded):
    """Expand ``count`` into positions [(u, v, h)]."""
    n = L["count"]
    if n <= 1:
        return [(u, v, h)]
    rng = np.random.default_rng(L.get("seed", ctx.seed) * 7919 + L["index"])
    arr = L["arrange"] or ("scatter" if (not grounded and L["thing"] is not None and L["thing"].cat in ("bird", "sky"))
                           else "row" if n <= 4 else "cluster")
    wf = h * aspect * ctx.H / ctx.W
    out = []
    if arr == "row":
        step = wf * 1.12
        tot = step * (n - 1)
        if tot > 0.9:
            step = 0.9 / (n - 1)
        for i in range(n):
            out.append((u + (i - (n - 1) / 2) * step, v + (0.004 * ((i % 2) * 2 - 1) if grounded else 0), h))
    elif arr == "column":
        step = h * 1.1
        for i in range(n):
            out.append((u, v + (i - (n - 1) / 2) * step, h))
    elif arr == "circle":
        r = max(wf, h * ctx.H / ctx.W) * 0.9 * max(1.0, n / 5)
        for i in range(n):
            a = 2 * math.pi * i / n
            out.append((u + r * math.cos(a), v + r * ctx.W / ctx.H * math.sin(a), h))
    elif arr == "grid":
        cols = int(math.ceil(math.sqrt(n)))
        for i in range(n):
            r_, c_ = divmod(i, cols)
            out.append((u + (c_ - (cols - 1) / 2) * wf * 1.2, v + (r_ - (n // cols - 1) / 2) * h * 1.15, h))
    elif arr == "cluster":
        pts = [(0, 0), (-0.75, 0.2), (0.75, 0.15), (-0.3, -0.35), (0.4, -0.3), (0.05, 0.45), (-1.2, -0.1), (1.2, -0.05)]
        for i in range(n):
            px, py = pts[i % len(pts)]
            k = i // len(pts)
            px += k * 0.5
            out.append((u + px * wf * 0.85, v + py * (0.05 if grounded else h * 0.6), h * (1 - 0.04 * (i % 3))))
        if grounded:
            out.sort(key=lambda p: p[1])
    else:   # scatter
        if grounded:
            lo = (ctx.horizon or 0.5) + 0.04
            for i in range(n):
                vv = float(np.clip(v + rng.uniform(-0.12, 0.12), lo, 0.97))
                out.append((float(np.clip(u + rng.uniform(-0.3, 0.3), 0.06, 0.94)), vv, h * rng.uniform(0.85, 1.1)))
            out.sort(key=lambda p: p[1])
        else:
            spread_u, spread_v = 0.14 + 0.03 * n, 0.07 + 0.01 * n
            if (L.get("at") or {}).get("h") is None and ctx.norm["group"] == "scene":
                spread_u = 0.36                        # "in the sky" alone: across the whole sky
            boxes = [p["bbox"] for p in ctx.placed.values() if isinstance(p, dict) and p.get("bbox")]
            pts = []
            for i in range(n):
                for _ in range(40):
                    cand = (u + rng.uniform(-spread_u, spread_u), v + rng.uniform(-spread_v, spread_v))
                    free = not any(b[0] < cand[0] < b[2] and b[1] < cand[1] < b[3] for b in boxes)
                    if free and all(abs(cand[0] - p[0]) > wf * 1.1 or abs(cand[1] - p[1]) > h * 1.1 for p in pts):
                        break
                pts.append(cand)
            top = ctx.horizon - 0.04 if ctx.horizon else 0.95
            out = [(float(np.clip(p[0], 0.04, 0.96)), float(np.clip(p[1], 0.04, top)), h * rng.uniform(0.85, 1.15))
                   for p in pts]
    if ctx.norm["group"] == "design" and out:          # a repeated thing in a design stays inside the picture area
        x0, _, x1, _ = ctx.area
        half = max(wf * hh / max(h, 1e-6) for (_, _, hh) in out) / 2
        lo, hi = min(p[0] for p in out) - half, max(p[0] for p in out) + half
        mid = (lo + hi) / 2
        span, room = hi - lo, 2 * max(min(mid - x0, x1 - mid), 0.18 * (x1 - x0))   # keep to its own side
        k = max(0.35, min(1.0, room / span)) if span > 0 else 1.0
        if k < 1.0:
            out = [(mid + (pu - mid) * k, pv, ph * k) for (pu, pv, ph) in out]
            lo, hi = mid - span * k / 2, mid + span * k / 2
        shift = (x0 - lo) if lo < x0 else (x1 - hi) if hi > x1 else 0.0
        out = [(pu + shift, pv, ph) for (pu, pv, ph) in out]
    return out


def _draw_layers(ctx, layers):
    """Place every picture layer, order them far -> near (relations honoured), draw each."""
    auto = [L for L in layers if L["at"].get("mode") == "auto" and L["type"] == "thing"
            and L["thing"].cat not in ("fullframe",)]
    slots = {}
    ground_auto = [L for L in auto if _is_grounded(ctx, L) or (ctx.table is not None and L["thing"].cat != "sky")]
    float_auto = [L for L in auto if L not in ground_auto]
    for grp in (ground_auto, float_auto):
        n = len(grp)
        for i, L in enumerate(grp):
            slots[L["label"]] = 0.5 if n == 1 else 0.2 + 0.6 * i / (n - 1)
    _group_people(ctx, layers, slots)
    plans = []
    for L in layers:
        if L["type"] == "python":
            plans.append((L, None))
            continue
        if L["type"] != "thing":
            continue
        plans.append((L, None))
    # order: fullframe sky fx first ... weather last
    def key(item):
        L = item[0]
        if L["type"] == "python":
            return (2, 0.5, L["index"])
        th = L["thing"]
        at = L["at"]
        if th.cat == "fullframe":
            return (0, 0, L["index"]) if th.target in _BACK_FRAME else (4, 0, L["index"])
        g = _is_grounded(ctx, L)
        if not g:                                      # clouds and sun far back; flying animals in front
            return (3, 0, L["index"]) if th.cat in ("bird", "animal") else (1, 0, L["index"])
        if at.get("mode") in ("point", "box"):
            v = at["values"][1] if at["mode"] == "point" else at["values"][3]
        else:
            d = _depth_of(at)
            v = _ground_v(ctx, 0.55 if d is None else d)
        return (2, v, L["index"])
    order = sorted(plans, key=key)
    # relations: move "behind X" before X, "on / in front of X" after X
    labels = [p[0]["label"] for p in order]
    for _ in range(2):
        for p in list(order):
            L = p[0]
            at = L.get("at") or {}
            tl = at.get("target_layer")
            if not tl or tl == "@table":
                continue
            names = [q[0]["label"] for q in order]
            if tl not in names:
                continue
            i, j = names.index(L["label"]), names.index(tl)
            if at.get("rel") == "behind" and i > j:
                order.insert(j, order.pop(i))
            elif at.get("rel") in ("on", "in front of", "under", "above", "left of", "right of", "next to", "in") and i < j:
                order.insert(j, order.pop(i))
    LAST["order"] = [p[0]["label"] for p in order]
    for L, _ in order:
        _draw_layer(ctx, L, slots.get(L["label"], 0.5))


def _group_people(ctx, layers, slots):
    """Two or more people placed by plain words ("left third", "right") or by nothing stand together as one
    group near the middle and turn towards each other, instead of standing alone at the far ends."""
    if ctx.norm["group"] != "scene" or ctx.table is not None:
        return
    ppl = []
    figs = {L["label"] for L in layers if L["type"] == "thing" and L["thing"].kit == "figure"}
    for L in layers:
        th = L.get("thing") if L["type"] == "thing" else None
        at = L.get("at") or {}
        if th is None or th.kit != "figure" or L["count"] > 1 or at.get("mode") not in ("auto", "words") \
                or at.get("sky"):
            continue
        u = at.get("h")
        if at.get("target_layer"):                     # "right of the first person" joins the group
            if at["target_layer"] not in figs or at.get("rel") not in ("left of", "right of", "next to"):
                continue
            u = 0.5 + (-0.01 if at["rel"] == "left of" else 0.01) * (1 + L["index"])
        ppl.append((slots.get(L["label"], 0.5) if u is None else u, L))
    if len(ppl) < 2:
        return
    ppl.sort(key=lambda p: (p[0], p[1]["index"]))
    n = len(ppl)
    mid = float(np.clip(np.mean([p[0] for p in ppl]), 0.3, 0.7))
    h = np.mean([_default_size(ctx, L, _ground_v(ctx, 0.55), True) for _, L in ppl])
    step = max(0.12, 0.62 * h * ctx.H / ctx.W)         # about one and a half body widths apart
    fire = next((L for L in layers if L["type"] == "thing" and re.search(r"fire", L["thing"].name)
                 and L["thing"].cat != "fullframe" and not (L.get("at") or {}).get("target_layer")
                 and (L.get("at") or {}).get("mode") in ("auto", "words")), None)
    gap = 0.0
    if fire is not None:                               # friends by a campfire sit on either side of it
        fu = fire["at"].get("h")
        mid = float(np.clip(slots.get(fire["label"], 0.5) if fu is None else fu, 0.3, 0.7))
        fire["at"] = {"mode": "words", "text": "between the people", "h": mid, "depth": 0.66}
        gap = 0.07
    for i, (_, L) in enumerate(ppl):
        k = i - (n - 1) / 2
        u = mid + k * step + (np.sign(k) * gap if k else 0.0)
        L["group_u"] = float(np.clip(u, 0.08, 0.92))
        if (L.get("at") or {}).get("target_layer"):
            L["at"] = {"mode": "words", "text": L["at"].get("text", "")}
        if not any(k in L["opts"] for k in ("facing", "view")):
            L["opts"]["facing"] = "right" if u < mid else "left"
            L["opts"]["view"] = "three_quarter"


def _draw_layer(ctx, L, slot=0.5):
    cv = ctx.cv
    ctx.log(L["label"])
    if L["type"] == "python":
        return _draw_python(ctx, L)
    th = L["thing"]
    if th.cat == "fullframe":
        return _draw_instance(ctx, L, th, None, None, None, "centre", L["label"])
    at = L["at"]
    grounded = _is_grounded(ctx, L)
    aspect = th.aspect
    try:
        if at.get("target_layer") and at["target_layer"] != "@table":
            T = ctx.placed.get(at["target_layer"])
            if T is None:
                u, v, anchor = _base_place(ctx, L, slot)
            else:
                h0 = _default_size(ctx, L)
                u, v, anchor = _relate(ctx, L, at["rel"], T, h0, aspect, grounded)
                # several things "next to" / "under" one target: line them up instead of stacking them
                used = ctx.__dict__.setdefault("rel_used", {}).setdefault((at["target_layer"], at["rel"]), [])
                if used and at["rel"] not in ("on", "in", "behind"):
                    wf = h0 * aspect * ctx.H / ctx.W
                    tc = (T["bbox"][0] + T["bbox"][2]) / 2
                    sign = 1 if u >= tc else -1
                    if at["rel"] == "next to" and len(used) % 2 == 1:
                        sign = -sign
                        u = tc - (u - tc)
                    far = [x for x in used if (x - tc) * sign > 0]
                    if far:
                        u = (max(far) if sign > 0 else min(far)) + sign * (wf * 0.9 + 0.015)
                    u = float(np.clip(u, 0.06, 0.94))
                used.append(u)
                if at.get("rel") == "on" and th.cat == "bird":
                    L["opts"].setdefault("pose", "perch")
                if at.get("rel") == "on":
                    grounded = True
                if at.get("h") is not None and at.get("rel") not in ("left of", "right of", "next to"):
                    u += (at["h"] - 0.5) * 0.2
        elif at.get("target_layer") == "@table" and ctx.table is not None:
            u, v, anchor = _base_place(ctx, L, slot)
            anchor = "base"
            grounded = True
        else:
            u, v, anchor = _base_place(ctx, L, slot)
        h = _default_size(ctx, L, v, grounded and anchor == "base")
        if at.get("mode") == "box" and th.kit not in ("creature",):
            u0, v0, u1, v1 = at["values"]
            h = min(v1 - v0, (u1 - u0) * ctx.W / ctx.H / max(0.2, aspect))
    except Exception as e:
        _err(L, f"placement failed: {_kit_error(e)}")
        return
    insts = _instances(ctx, L, u, v, anchor, h, aspect, grounded)
    boxes = []
    for i, (uu, vv, hh) in enumerate(insts):
        label = L["label"] if i == 0 else f"{L['label']} #{i + 1}"
        rec = _draw_instance(ctx, L, th, uu, vv, hh, anchor, label, i)
        if rec:
            boxes.append(rec)
    if boxes:
        bb = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
        ctx.placed[L["label"]] = {"bbox": bb, "base": (u, v if anchor == "base" else bb[3]), "cat": th.cat,
                                  "path": L["path"], "count": len(boxes), "at": (round(u, 3), round(v, 3)),
                                  "size": round(h, 3)}


def _new_entries(cv, n0):
    m = getattr(cv, "manifest", None) or []
    return [e for e in m[n0:] if isinstance(e, dict) and e.get("kind") != "text" and "bbox" in e]


def _draw_instance(ctx, L, th, u, v, h, anchor, label, i=0):
    cv = ctx.cv
    n0 = len(getattr(cv, "manifest", []) or [])
    seed = L.get("seed", ctx.seed) + 31 * L["index"] + 7 * i
    o = dict(L["opts"])
    col = L.get("color")
    try:
        k = th.kit
        if k == "creature":
            from . import creature
            pose = o.pop("pose", None)
            if anchor == "centre":
                if th.cat == "bird" and pose is None:
                    pose = "fly"
                v = v + h * 0.5
            facing = o.pop("facing", None) or ("left" if u > 0.55 else "right")
            if L["count"] > 1 and o.get("facing") is None and th.cat != "bird":
                facing = "right" if (i % 2 == 0) == (u < 0.5) else "left"
            kw = {kk: vv for kk, vv in o.items() if kk in ("coat", "view", "style", "head_turn", "tail", "ears",
                                                            "mouth", "eyes")}
            creature.creature(cv, th.target, at=(u, v), size=h, pose=pose, facing=facing, seed=seed,
                              shadow=anchor == "base", **kw)
        elif k == "figure":
            from . import figure
            desc = o.pop("describe", None) or th.target or L.get("name") or "person"
            if col and isinstance(desc, str) and not th.target:
                pass
            kw = {kk: vv for kk, vv in o.items() if kk in ("pose", "facing", "view", "style", "expression", "item")}
            if anchor == "centre":
                v = v + h * 0.5
            fig = figure.person(desc, height=h, **kw)
            fig.draw(cv, at=(u, v))
        elif k == "flora":
            from . import flora
            if th.cat == "rock":
                kw = {kk: vv for kk, vv in o.items() if kk in ("moss", "kind")}
                flora.rock(cv, at=(u, v if anchor == "base" else v + h / 2), size=h / 0.7, seed=seed,
                           light=ctx.light, **kw)
            else:
                kw = {kk: vv for kk, vv in o.items() if kk in ("season",)}
                target = th.target
                if "season" not in kw and _wintry(ctx.norm) and th.size >= 0.3:
                    kw["season"] = "winter"
                    if "birch" in str(target):                 # flora's birch keeps its leaves in winter
                        target = "snowy tree"
                    elif str(target) == "pine":                # a winter pine reads as a dark spruce spire
                        target = "spruce"
                flora.plant(cv, target, at=(u, v if anchor == "base" else v + h / 2), height=h, seed=seed,
                            light=ctx.light, **kw)
        elif k == "elements":
            from . import elements
            name = th.target
            if th.cat == "fullframe":
                if name == "stars":
                    elements.stars(cv, seed=seed, horizon=ctx.horizon)
                else:
                    elements.element(cv, name, seed=seed)
            elif name == "sun":
                kw = {"color": col} if col else {}
                elements.sun(cv, at=(u, v), size=h * 1080 / 2 * 0.6, seed=seed, **kw)
            elif name == "moon":
                kw = {"color": col} if col else {}
                elements.moon(cv, at=(u, v), size=h * 1080 / 3.0, seed=seed, phase=0.5 if "crescent" in L.get("name", "") else 1.0
                              if "full" in L.get("name", "") else 0.5, **kw)
            elif name == "cumulus":
                sky = (ctx.norm.get("background") or {}).get("sky", "day")
                tones = {"sunset": ("#ffd9bd", "#b07a8c", "#7a5878"), "dusk": ("#e9c3c8", "#7f6f96", "#5b5277"),
                         "dawn": ("#ffe1cf", "#b58d9e", "#86708a"), "golden": ("#fff0d0", "#c9a07e", "#9a7a66"),
                         "night": ("#5d6a8a", "#2c3450", "#1e2438"), "twilight": ("#9aa3c8", "#4d5682", "#363d63"),
                         "storm": ("#9aa2ad", "#5a6270", "#40464f")}.get(sky)
                kw = dict(lit=tones[0], shadow=tones[1], base=tones[2]) if tones else {}
                elements.cumulus(cv, at=(u, v + h * 0.45), size=h * 1080 / 0.46, seed=seed,
                                 light=(-ctx.light[0], ctx.light[1]), **kw)
                for e in (getattr(cv, "manifest", None) or [])[n0:]:   # the cloud's drawing window reaches far
                    if isinstance(e, dict) and e.get("bbox") and e.get("extent"):   # above it: report the cloud
                        e["extent"] = list(e["bbox"])
            else:
                at_ = (u, v) if anchor == "base" or th.cat == "effect" else (u, v)
                if th.cat == "effect" and anchor == "centre":
                    at_ = (u, v + h / 2)
                elements.element(cv, name, at=at_, size=h * 1080 / 1.2, seed=seed)
        elif k == "form":
            _draw_form(ctx, L, th, u, v, h, anchor, col, o)
        elif k == "pictogram":
            P = importlib.import_module("pixelcanvas.pictograms")
            if col is None and ctx.theme and ctx.norm["group"] == "design":
                picts = [x for x in ctx.norm["layers"] if x["type"] == "thing" and x["thing"].kit == "pictogram"]
                k_ = next((j for j, x in enumerate(picts) if x is L), 0)
                col = [ctx.theme["accent"], ctx.theme["soft"], ctx.theme["accent2"]][k_ % 3]
            color = col or _natural_colour(th) or ("#1d2633" if ctx.norm["group"] == "design" else "#6a7686")
            kw = {kk: vv for kk, vv in o.items() if kk in ("style", "accent", "rotate", "flip", "detail_color")}
            if "style" not in kw:
                kw["style"] = "duotone" if ctx.norm["group"] == "design" else "shaded"
            if ctx.norm["group"] != "design":
                kw.setdefault("shadow", anchor == "base")
            P.pictogram(cv, th.target, at=(u, v), size=h, color=color, anchor="bm" if anchor == "base" else "mm", **kw)
        elif k == "icon":
            from . import vector
            color = col or (ctx.theme["ink"] if ctx.theme else "#1d2633")
            px = h * ctx.H
            cx, cy = u * ctx.W, (v - h / 2 if anchor == "base" else v) * ctx.H
            vector.icon(th.target, color=color, stroke_width=2.0).draw(cv, box=(cx - px / 2, cy - px / 2, cx + px / 2, cy + px / 2))
        else:
            _err(L, f"no drawer for kit {k!r}")
            return None
    except Exception as e:
        _err(L, f"{th.name!r} could not be drawn: {_kit_error(e)}",
             "check the option names in this layer; compose.py --example shows working ones")
        return None
    new = _new_entries(cv, n0)
    if new:
        for e in new:
            if e.get("name") != label:
                al = list(e.get("aliases") or [])
                for a in (e.get("name"), th.name, th.target):
                    if a and a not in al and a != label:
                        al.append(str(a))
                e["aliases"] = al
                e["name"] = label
            meta = e.setdefault("meta", {})
            meta["layer"] = L["path"]
        b = [min(e["bbox"][0] for e in new), min(e["bbox"][1] for e in new), max(e["bbox"][2] for e in new),
             max(e["bbox"][3] for e in new)]
        return b
    if u is None:
        return None
    eb = _est_box(ctx, th.cat, u, v, h, th.aspect, anchor)
    try:
        from . import manifest
        manifest.note(cv, th.cat if th.cat in ("plant", "creature", "person", "effect", "object") else "object",
                      label, bbox=[max(0, eb[0]), max(0, eb[1]), min(1, eb[2]), min(1, eb[3])], alias=[th.name],
                      layer=L["path"])
    except Exception:
        pass
    return eb


_NATURAL = (("pear", "#b7c443"), ("grape", "#6d3b78"), ("apple", "#c8352b"), ("banana", "#f0cf3a"),
            ("lemon", "#f2d43b"), ("lime", "#7cb342"), ("cherry", "#b3122e"), ("strawberr", "#d8323c"),
            ("orange", "#ee8a22"), ("peach", "#f5a26b"), ("plum", "#6b2f5f"), ("melon", "#7fb24a"),
            ("carrot", "#e8762b"), ("tomato", "#d93a2b"), ("pepper", "#c9302c"), ("pumpkin", "#e07b24"),
            ("broccoli", "#4f8a3a"), ("leaf", "#5b9a3c"), ("tree", "#4f8a3a"), ("cactus", "#4f8a3a"),
            ("flower", "#d8547a"), ("rose", "#c8284f"), ("sun", "#f3c13a"), ("moon", "#e9e4cf"),
            ("star", "#f3d36b"), ("heart", "#d8324f"), ("fire", "#f08a24"), ("water", "#3f8fd1"),
            ("film", "#2f333b"), ("camera", "#2f333b"), ("saxophone", "#d4a937"), ("trumpet", "#d4a937"),
            ("guitar", "#a0522d"), ("violin", "#8b4513"), ("piano", "#22252b"), ("coffee", "#6f4a35"),
            ("cup", "#e9e3d8"), ("mug", "#e9e3d8"), ("wine", "#7a1f35"), ("bread", "#d9a35b"),
            ("croissant", "#d9a35b"), ("cheese", "#f1c84b"), ("rocket", "#d8dde3"), ("planet", "#d98b4a"))


def _natural_colour(th):
    """A believable body colour for a pictogram drawn without one (a pear is green, not navy)."""
    name = f"{th.name} {th.target}".lower()
    return next((c for k, c in _NATURAL if k in name), None)


def _draw_form(ctx, L, th, u, v, h, anchor, col, o):
    from . import form
    fn, size0, opts = FORM_OBJECTS[th.target]
    opts = dict(opts)
    nat = _natural_colour(th)
    if col and nat and isinstance(col, str) and col.startswith("#") and _lum(col) > 0.8:
        LAST["warnings"].append(f"{L['path']} {L['label']!r}: {col} is a washed-out {th.name}; drew it {nat} "
                                f"(give a stronger colour, or none)")
        col = nat
    base = col or opts.pop("base")
    opts.pop("base", None)
    mat = o.get("material") or opts.pop("material", "matte")
    opts.pop("material", None)
    if anchor == "centre":
        v = v + h / 2
    at = (u, v)
    if ctx.table is not None and anchor == "base":
        depth = float(np.clip((0.96 - v) / 0.36, 0.0, 1.0))
        at = ctx.table.place(u, depth)
        k = getattr(at, "k", 1.0) or 1.0
        h = h / max(k, 0.3)
    if fn == "ball":
        st = opts.pop("stretch", 1.0)
        r = h / 2
        rad = (r * st, r) if st != 1.0 else r
        form.ball(cv=ctx.cv, at=at, radius=rad, base=base, material=mat, cast=True)
    elif fn == "cylinder":
        r = h * 0.38
        form.cylinder(ctx.cv, at, r, h, base=base, material=mat, cast=True, **opts)
    elif fn == "lathe":
        prof = [(rr * h, hh * h) for rr, hh in opts.pop("profile")]
        form.lathe(ctx.cv, at, prof, base=base, material=mat, cast=True, **opts)
    elif fn == "cuboid":
        dx, dy, dz = opts.pop("dims")
        hh = h
        form.cuboid(ctx.cv, at, size=(hh / dy * dx * 0.8, hh / dy * dz * 0.6, hh), base=base, material=mat,
                    yaw=25, cast=True)
    elif fn == "torus":
        form.torus(ctx.cv, at, h * 1.2, h * 0.4, base=base, material=mat)


def _draw_python(ctx, L):
    """Escape hatch: call ``fn(cv, box, spec)`` from a sibling Python file."""
    ref = L["ref"]
    fname, _, func = ref.rpartition(":")
    base = ctx.norm.get("base_dir") or os.getcwd()
    path = fname if os.path.isabs(fname) else os.path.join(base, fname)
    if not os.path.exists(path):
        _err(L, f"python file {fname!r} not found next to the spec ({base})")
        return
    try:
        import importlib.util
        name = "pcspec_" + re.sub(r"\W", "_", os.path.splitext(os.path.basename(path))[0])
        spec_ = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(mod)
        fn = getattr(mod, func, None)
        if fn is None:
            _err(L, f"{fname} has no function {func!r}")
            return
        at = L["at"]
        if at.get("mode") == "box":
            box = at["values"]
        else:
            u, v = (at["values"] if at.get("mode") == "point" else _word_point(ctx, at, True))
            h = L["size"] or 0.3 * L.get("size_rel", 1.0)
            wf = h * ctx.H / ctx.W
            box = [u - wf / 2, v - h / 2, u + wf / 2, v + h / 2]
        n0 = len(getattr(ctx.cv, "manifest", []) or [])
        px = (box[0] * ctx.W, box[1] * ctx.H, box[2] * ctx.W, box[3] * ctx.H)
        fn(ctx.cv, px, L["raw"])
        new = _new_entries(ctx.cv, n0)
        if not new:
            from . import manifest
            manifest.note(ctx.cv, "object", L["label"], bbox=list(box), layer=L["path"])
        ctx.placed[L["label"]] = {"bbox": list(box), "base": ((box[0] + box[2]) / 2, box[3]), "cat": "object",
                                  "path": L["path"], "count": 1}
    except Exception as e:
        _err(L, f"{ref} raised {type(e).__name__}: {_kit_error(e)}")


# --------------------------------------------------------------------------------------------- sections
def _render_section(ctx, group):
    cv, norm = ctx.cv, ctx.norm
    if group == "diagram":
        d = norm.get("diagram")
        if not d:
            return
        from . import diagram
        ctx.log("diagram")
        try:
            W, H = cv.w, cv.h
            direction = d.get("direction") or ("LR" if W >= H else "TB")
            kw = {}
            if d.get("title"):
                kw["title"] = d["title"]
            if d.get("subtitle"):
                kw["subtitle"] = d["subtitle"]
            dg = diagram.Diagram(theme=d.get("theme") or "light", direction=direction, **kw)
            grouped = {}
            for gname, mem in d.get("groups") or []:
                for m in mem:
                    grouped.setdefault(m, gname)
            nodes = {n["id"]: n for n in d["nodes"]}
            done = set()
            for gname, mem in d.get("groups") or []:
                with dg.group(_slug(gname), gname):
                    for m in mem:
                        if m in done:
                            continue
                        _diagram_node(dg, nodes[m])
                        done.add(m)
            for nid, n in nodes.items():
                if nid not in done:
                    _diagram_node(dg, n)
            for a, b, label, style in d["edges"]:
                dg.edge(a, b, label=label, style=style)
            dg.draw(cv, log=ctx.log)
        except Exception as e:
            _err("diagram", f"diagram failed: {_kit_error(e)}")
    elif group == "chart":
        c = norm.get("chart")
        if not c:
            return
        ctx.log("chart")
        _draw_chart(ctx, c, None)
    elif group == "dashboard":
        _draw_dashboard(ctx, norm.get("dashboard") or {})
    elif group == "ui":
        u = norm.get("ui")
        if not u:
            return
        from . import ui
        ctx.log("screen")
        try:
            kw = {}
            if u.get("accent"):
                kw["accent"] = u["accent"]
            if u["template"] == "home":
                s = _ui_home(u, kw)
            else:
                s = ui.template(u["template"], screen=u["screen"], theme=u.get("theme") or "light",
                                content=u.get("content") or None, **kw)
            cv.gradient([(0, "#e8edf3"), (1, "#cfd8e3")] if (u.get("theme") or "light") == "light" else
                        [(0, "#1a1f2b"), (1, "#0d1017")])
            try:
                s.draw(cv, log=ctx.log)
            except Exception as e:                     # a value the screen cannot lay out must not cost the
                if u["template"] != "home":            # picture: the greeting and the card titles alone
                    raise
                c = u.get("content") or {}
                cards = [str(d.get("title") or d.get("label") or d.get("name") or "") if isinstance(d, dict)
                         else str(d) for d in (c.get("cards") if isinstance(c.get("cards"), list) else [])]
                plain = dict(u, content={"title": str(c.get("title") or c.get("greeting") or "Good morning"),
                                         "cards": [{"title": t} for t in cards if t]})
                _ui_home(plain, {}).draw(cv, log=ctx.log)
                LAST["warnings"].append(f"ui: drew the home screen with the greeting and the card titles only "
                                        f"({_kit_error(e)}); give each card a title, a text and a progress")
        except Exception as e:
            _err("ui", f"ui failed: {_kit_error(e)} - write the ui section in the shape of compose.py --example ui")


_CARD_ICONS = (("plan", "calendar"), ("today", "calendar"), ("calendar", "calendar"), ("schedule", "clock"),
               ("water", "drop"), ("drink", "drop"), ("hydrat", "drop"), ("sleep", "moon"), ("night", "moon"),
               ("step", "activity"), ("workout", "activity"), ("run", "activity"), ("heart", "heart"),
               ("mood", "smile"), ("weather", "cloud-sun"), ("music", "music"), ("money", "wallet"),
               ("budget", "wallet"), ("spend", "credit-card"), ("task", "check"), ("todo", "check"),
               ("message", "message"), ("mail", "mail"), ("photo", "image"), ("goal", "target"),
               ("energy", "bolt"), ("meal", "fire"), ("food", "fire"), ("focus", "target"), ("news", "file"),
               ("read", "bookmark"), ("book", "bookmark"), ("walk", "activity"), ("light", "bulb"),
               ("idea", "bulb"), ("photo", "camera"), ("shop", "cart"), ("friend", "users"), ("team", "users"))


def _ui_home(u, kw):
    """A phone home screen: date line and greeting, an avatar, then one card per entry of content.cards."""
    from . import ui
    c = u.get("content") or {}
    s = ui.Screen(u["screen"], theme=u.get("theme") or "light", **kw)
    greet = str(c.get("title") or c.get("greeting") or c.get("heading") or "Good morning")
    sub = str(c.get("subtitle") or c.get("date") or "Thursday, 12 November")
    user = str(c.get("user") or c.get("name") or (greet.split(",")[-1].strip() if "," in greet else "Sam"))
    cards = c.get("cards") or ["Today's plan", "Activity", "Sleep"]
    if isinstance(cards, str):                         # "Water the fern, Light check"
        cards = [x.strip() for x in re.split(r",|;|\band\b", cards) if x.strip()]
    elif isinstance(cards, dict):                      # {"Water the fern": "every 3 days"}
        cards = [{"title": k_, "text": v_ if isinstance(v_, str) else ""} for k_, v_ in cards.items()]
    with s.vstack(gap=16):
        with s.hstack(gap=12):
            with s.vstack(gap=2, w="fill"):
                s.small(sub)
                s.heading(greet, level=2)
            s.avatar(user, 46)
        progs = [float(d["progress"]) for d in cards if isinstance(d, dict) and _num(d.get("progress"))]
        if len(cards) <= 4:                            # a summary ring above the cards fills the screen like an app
            val = (sum(p / 100 if p > 1 else p for p in progs) / len(progs)) if progs else 0.68
            val = float(np.clip(val, 0.0, 1.0))
            summ = c.get("summary") if isinstance(c.get("summary"), str) else None
            with s.card(pad=16, gap=10):
                with s.hstack(gap=16):
                    s.ring(val, label=f"{round(val * 100)}%", size=72)
                    with s.vstack(gap=4, w="fill"):
                        s.title(summ or "Today")
                        s.small(f"{len(cards)} things on your list")
        for i, card in enumerate(cards[:6]):
            d = card if isinstance(card, dict) else {"title": str(card)}
            title = str(d.get("title") or d.get("label") or d.get("name") or f"Card {i + 1}")
            text = d.get("text") or d.get("subtitle") or d.get("value") or ""
            low = title.lower()
            icon = d.get("icon") or next((ic for k, ic in _CARD_ICONS if k in low), "sparkles")
            with s.card(pad=16, gap=10):
                with s.hstack(gap=14):
                    s.icon(icon if icon in ui_icon_names() else "sparkles", 26, color="accent")
                    with s.vstack(gap=2, w="fill"):
                        s.title(title)
                        if text:
                            s.small(str(text))
                    s.icon("chevron-right", 20)
                if d.get("progress") is not None and _num(d.get("progress")):
                    p = float(d["progress"])
                    s.progress(float(np.clip(p / 100 if p > 1 else p, 0.0, 1.0)))
        if len(cards) <= 3:
            with s.card(pad=16, gap=8):
                s.title("This week")
                s.chart("bar", {"M": 3, "T": 5, "W": 4, "T ": 6, "F": 5, "S": 7, "S ": 4}, height=90, axes=False)
        s.spacer()
    s.tab_bar([("home", "Home"), ("calendar", "Plan"), ("activity", "Health"), ("user", "Profile")], active=0)
    return s


def ui_icon_names():
    from . import vector
    return set(getattr(vector, "ICONS", {}) or ())


def _diagram_node(dg, n):
    kw = {k: n[k] for k in ("icon", "shape", "color", "sublabel", "rank") if n.get(k) is not None}
    try:
        dg.node(n["id"], n["label"], **kw)
    except Exception:
        kw.pop("icon", None)
        dg.node(n["id"], n["label"], **kw)


_CHART_PASS = ("subtitle", "xlabel", "ylabel", "yfmt", "xfmt", "value_labels", "legend", "highlight", "smooth",
               "stacked", "horizontal", "colors")


def _draw_chart(ctx, c, box, fig=None):
    from . import chart
    cv = ctx.cv
    data = c.get("data")
    if data is None:
        return
    kw = {k: c[k] for k in _CHART_PASS if k in c and c[k] is not None}
    if isinstance(c.get("options"), dict):
        kw.update(c["options"])
    kind = c["type"]
    if kind in ("bar", "barh", "stacked") and "value_labels" not in kw and isinstance(data, dict) and len(data) <= 12:
        kw["value_labels"] = True
    if "colors" in kw:
        cols = kw.pop("colors")
        kw["color"] = cols[0] if len(cols) == 1 else cols
    try:
        if fig is not None:
            ax = fig.axes(box, title=c.get("title"), **{k: v for k, v in kw.items() if k in ("subtitle", "xlabel", "ylabel", "yfmt", "xfmt", "legend")})
            series = {k: v for k, v in kw.items() if k not in ("subtitle", "xlabel", "ylabel", "yfmt", "xfmt", "legend")}
            _series(ax, kind, data, c.get("x"), series)
        else:
            theme = c.get("theme") or ctx.norm.get("theme") or "light"
            theme = theme if theme in ("light", "dark", "paper", "blueprint", "pastel", "neon", "mono") else "light"
            title = c.get("title")
            if box is None:                            # a whole picture: page, header band, chart on a card
                box = _chart_page(ctx, c, kind, data, theme, kw)
                title = None
            for attempt in range(3):
                try:
                    chart.quick(cv, kind, data, title=title, box=box, theme=theme, log=ctx.log,
                                x=c.get("x"), **kw)
                    break
                except TypeError as e:
                    m = re.search(r"'(\w+)'", str(e))
                    if m and m.group(1) in kw and attempt < 2:
                        LAST["warnings"].append(f"chart: option {m.group(1)!r} not used by a {kind} chart")
                        kw.pop(m.group(1))
                        continue
                    raise
    except Exception as e:
        _err("chart", f"chart failed: {_kit_error(e)}")


def _numeric_dict(data):
    return isinstance(data, dict) and bool(data) and all(_num(v) for v in data.values())


def _chart_summary(kind, data, fmt=None):
    """A one-line reading of the data for the header ('total 1,630 · highest Apr 520'), or None."""
    from . import chart
    if not _numeric_dict(data):
        return None
    keys = list(data)
    vals = [float(data[k]) for k in keys]
    f = (lambda v: chart.fmt_number(v, fmt)) if fmt else (lambda v: chart.fmt_number(v))
    if kind in ("bar", "barh", "column"):
        i = int(np.argmax(vals))
        return f"total {f(sum(vals))} · highest {keys[i]} {f(vals[i])}"
    if kind in ("line", "area") and len(vals) >= 2:
        a, b = vals[0], vals[-1]
        pct = f" ({(b - a) / abs(a) * 100:+.0f}%)" if a else ""
        return f"{keys[0]} {f(a)} → {keys[-1]} {f(b)}{pct}"
    if kind in ("pie", "donut"):
        i = int(np.argmax(vals))
        tot = sum(vals) or 1.0
        return f"largest share {keys[i]} {vals[i] / tot * 100:.0f}%"
    return None


def _chart_page(ctx, c, kind, data, theme, kw):
    """Paint the page and the header band of a one-chart picture; return the box for the chart card."""
    from . import chart
    cv = ctx.cv
    th = chart._theme(theme)
    bg = th.background if isinstance(th.background, str) else "#f5f6f8"
    cv.gradient([(0, _mixhex(bg, "#ffffff", 0.35)), (1, _mixhex(bg, "#000000", 0.04))])
    sub = kw.pop("subtitle", None)
    summary = _chart_summary(kind, data, kw.get("yfmt") or kw.get("fmt"))
    if kind in ("bar", "barh") and "highlight" not in kw and _numeric_dict(data) and len(data) >= 3:
        kw["highlight"] = max(data, key=lambda k: float(data[k]))
    if kind == "line" and _numeric_dict(data):
        kw.setdefault("fill", True)
        kw.setdefault("width", 3.2)
        kw.setdefault("smooth", True)
    kw.setdefault("card", True)
    title = c.get("title")
    if not title and not sub:
        return (0.025, 0.04, 0.975, 0.96)
    if title and not sub:
        sub, summary = summary, None
    try:
        chart.header(cv, title or sub, sub if title else None, right=[summary] if summary else None,
                     box=(0.025, 0.02, 0.975, 0.16), theme=theme, font_scale=1.35).draw(log=ctx.log)
    except Exception as e:                             # the chart itself is still drawn
        _err("chart", f"chart header failed: {_kit_error(e)}", severity="warn")
    return (0.02, 0.18, 0.98, 0.97)


def _series(ax, kind, data, x, kw):
    if kind in ("bar", "barh", "stacked"):
        cats = x if x is not None else (list(data) if isinstance(data, dict) and all(_num(v) for v in data.values()) else None)
        vals = data if cats is None or not isinstance(data, dict) or not all(_num(v) for v in data.values()) else [data[k] for k in cats]
        if isinstance(data, dict) and not all(_num(v) for v in data.values()):
            ax.bar(x, data, stacked=kind == "stacked", horizontal=kind == "barh", **{k: v for k, v in kw.items() if k in ("value_labels", "highlight")})
        else:
            ax.bar(cats, vals, horizontal=kind == "barh", **{k: v for k, v in kw.items() if k in ("value_labels", "highlight")})
    elif kind in ("line", "area", "spark"):
        if isinstance(data, dict) and all(_num(v) for v in data.values()):
            getattr(ax, "area" if kind == "area" else "line")(list(data), list(data.values()), **({"smooth": True} if kw.get("smooth") else {}))
        else:
            getattr(ax, "area" if kind == "area" else "line")(x if x is not None else list(range(len(next(iter(data.values())) if isinstance(data, dict) else data))), data)
    elif kind in ("pie", "donut"):
        pkw = {}
        if kind == "donut" and isinstance(data, dict) and data and all(_num(v) for v in data.values())                 and abs(sum(float(v) for v in data.values()) - 100) < 1.5:
            top = max(data, key=lambda k_: float(data[k_]))   # shares of 100: the hole names the largest one
            pkw["center"] = (f"{float(data[top]):g}%", str(top))
        ax.pie(data, donut=0.55 if kind == "donut" else 0.0, **pkw)
    elif kind == "scatter":
        ax.scatter(data.get("x"), data.get("y"))
    elif kind == "hist":
        ax.hist(data)
    else:
        ax.bar(list(data), list(data.values()))


def _draw_dashboard(ctx, d):
    from . import chart
    cv = ctx.cv
    theme = d.get("theme") or "dark"
    ctx.log("dashboard")
    try:
        fig = chart.figure(cv, theme=theme)
        top = 0.02
        if d.get("title"):
            fig.header(d["title"], d.get("subtitle"), right=d.get("right") if isinstance(d.get("right"), list) else None)
            top = 0.15
        kpis = d.get("kpis") or []
        charts = d.get("charts") or []
        if kpis:
            n = len(kpis)
            gap = 0.015
            w = (0.96 - gap * (n - 1)) / n
            for i, k in enumerate(kpis):
                x0 = 0.02 + i * (w + gap)
                kw = {kk: k[kk] for kk in ("delta", "spark", "fmt", "note", "icon", "good") if k.get(kk) is not None}
                fig.kpi((x0, top, x0 + w, top + 0.2), str(k["label"]), k["value"], **kw)
            top += 0.23
        if charts:
            n = len(charts)
            cols = min(n, 3 if n != 4 else 2)
            rows = int(math.ceil(n / cols))
            gap = 0.02
            cw = (0.96 - gap * (cols - 1)) / cols
            ch = (0.98 - top - gap * (rows - 1)) / rows
            for i, c in enumerate(charts):
                r, q = divmod(i, cols)
                box = (0.02 + q * (cw + gap), top + r * (ch + gap), 0.02 + q * (cw + gap) + cw, top + r * (ch + gap) + ch)
                _draw_chart(ctx, c, box, fig=fig)
        fig.draw(log=ctx.log)
    except Exception as e:
        _err("dashboard", f"dashboard failed: {_kit_error(e)}")


# --------------------------------------------------------------------------------------------- 3D
_SLOTS = {   # default spots in a room of half-size R: (x, z, rotate, wall)
    "bed": (0.55, -0.45, 0, "back"), "bookshelf": (-0.45, -0.92, 0, "back"), "table": (-0.88, 0.25, 90, "left"),
    "desk_lamp": None, "chair": (-0.45, 0.3, -100, None), "sofa": (0.3, -0.75, 0, "back"),
    "potted_plant": (-0.82, 0.85, 0, None), "rug": (0.1, 0.25, 0, None), "laptop": None,
    "pendant_lamp": (0.0, 0.0, 0, None), "speaker": (0.85, 0.8, -30, None), "coffee_machine": None,
    "tree": (0.0, 0.0, 0, None), "character": (0.2, 0.55, 20, None), "quadruped": (0.5, 0.6, -30, None),
    "floor_lamp": (0.82, -0.2, 0, None), "armchair": (-0.42, 0.6, 150, None), "side_table": (0.85, 0.35, 0, None),
}


def _render_scene3d(ctx):
    cv, norm = ctx.cv, ctx.norm
    s3 = norm.get("scene3d")
    if not s3:
        return
    from . import solid
    room = s3.get("room")
    R = 2.0
    try:
        light_kw = {}
        if s3["lighting"] == "interior":
            light = solid.Lighting("interior", sun=(-100, 33), sun_power=2.6, ambient=0.6, softness=0.25)
        else:
            light = solid.Lighting(s3["lighting"])
        ground = s3.get("ground")
        if room is not None:
            ground = None
        elif ground is None:
            ground = "#d9d4cc"
        elif isinstance(ground, list):
            ground = ground[0]
        sc = solid.Scene(lighting=light, ground=ground if ground != "none" else None)
        if room is None and isinstance(ground, str) and ground.startswith("#") and s3["camera"] != "top":
            sizes = []                                 # a studio sweep sized to the group, so the floor rises
            for ob in s3["objects"]:                   # into a wall behind it instead of a distant horizon
                m_, _ = _make_model(solid, ob, 0)
                if m_ is not None:
                    sizes.append(float(max(m_.size)))
            big = max(sizes or [1.0])
            sc.backdrop(ground, distance=max(0.6, 2.6 * big), radius=max(0.3, 1.0 * big))
        if room is not None:
            walls = room.get("walls") or "#efe2cf"
            wall2 = room.get("wall2") or _mixhex(walls, "#c9dde0", 0.35)
            floor = room.get("floor") or "wood"
            if floor == "wood":
                fl = solid.planks()
            elif floor in ("tiles", "checker", "tile"):
                fl = solid.checker(0.5)
            elif floor == "marble":
                fl = solid.marble()
            elif floor == "carpet" or floor == "concrete":
                fl = "#b9b0a3" if floor == "concrete" else "#a88f7a"
            else:
                fl = floor
            sc.add(solid.room(2 * R, 2 * R, 2.6, walls=walls, walls2=wall2, door=None,
                              window=("left", 0.3, 1.2, 1.2), floor=fl), "room")
        placed = {}
        feet = []                                     # floor footprints (x0, z0, x1, z1) of placed objects
        lamps = []
        used = []
        auto_i = 0
        studio = _studio_layout(solid, s3["objects"]) if room is None else {}
        for ob in s3["objects"]:
            ob["in_room"] = room is not None
        for ob in s3["objects"]:
            for i in range(ob.get("count", 1)):
                m, info = _make_model(solid, ob, i)
                if m is None:
                    continue
                key = ob["src"][1]
                rot = ob.get("rotate")
                x = z = None
                y = 0.0
                if (ob["path"], i) in studio and not (ob.get("on_obj") and ob["on_obj"] in placed):
                    x, z, r0 = studio[(ob["path"], i)]
                    rot = rot if rot is not None else r0
                elif ob.get("on_obj") and ob["on_obj"] in placed:
                    tm = placed[ob["on_obj"]]
                    lo, hi = tm.bounds()
                    y = float(hi[1])
                    cx, cz = float((lo[0] + hi[0]) / 2), float((lo[2] + hi[2]) / 2)
                    n_on = sum(1 for u_ in used if u_ == ob["on_obj"])
                    x = cx + (n_on - 0.5) * 0.35 * min(1.0, float(hi[0] - lo[0]))
                    z = cz + (0.15 if n_on % 2 else -0.1) * min(1.0, float(hi[2] - lo[2]))
                    used.append(ob["on_obj"])
                    if rot is None and placed.get(ob["on_obj"] + "#rot") is not None:
                        rot = placed[ob["on_obj"] + "#rot"]
                else:
                    at = ob["at"]
                    if at.get("mode") == "point":
                        a, b = at["values"]
                        if max(abs(a), abs(b)) <= 1.0 and room is not None:
                            x, z = (a - 0.5) * 2 * R * 0.9, (b - 0.5) * 2 * R * 0.9
                        else:
                            x, z = a, b
                    elif at.get("mode") == "words":
                        x, z, rot0 = _room_words(at, R, key)
                        rot = rot if rot is not None else rot0
                    else:
                        slot = _SLOTS.get(key)
                        if slot and room is not None and key not in [p.split("#")[0] for p in placed if "#" not in p]:
                            x, z, r0, _w = slot[0] * R, slot[1] * R, slot[2], slot[3]
                            rot = rot if rot is not None else r0
                        else:
                            ang = auto_i * 2.39996
                            rad = 0.35 * R * math.sqrt(auto_i + 1) if room is not None else 1.2 * math.sqrt(auto_i)
                            x, z = rad * math.cos(ang), rad * math.sin(ang) * (0.6 if room is not None else 1.0)
                            auto_i += 1
                    if ob.get("count", 1) > 1:
                        x += (i - (ob["count"] - 1) / 2) * 0.9 * max(0.4, float(m.size[0]))
                if rot:
                    m = m.rotate_y(rot)
                m0 = m
                m = m0.place(x=float(x), z=float(z), y=float(y))
                label = ob["label"] if ob.get("count", 1) == 1 else f"{ob['label']} {i + 1}"
                if y == 0.0 and key not in ("rug", "pendant_lamp", "room") and (ob["path"], i) not in studio:
                    m, moved = _free_floor(m0, m, x, z, feet, R if room is not None else None)
                    if moved:
                        LAST["warnings"].append(f"{ob['path']}: moved {label!r} to [{moved[0]:.1f}, {moved[1]:.1f}] "
                                                f"so it does not stand inside another object")
                    lo, hi = m.bounds()
                    feet.append((float(lo[0]), float(lo[2]), float(hi[0]), float(hi[2])))
                mm = sc.add(m, label)
                placed[ob["label"]] = mm
                placed[ob["label"] + "#rot"] = rot
                if key in ("desk_lamp", "pendant_lamp"):
                    lamps.append(mm)
        for lm in lamps:
            try:
                sc.light_from(lm["bulb"], color="#ffcf8a", power=0.9, radius=0.9)
            except Exception:
                pass
        cam = s3["camera"]
        if cam == "isometric":
            sc.camera = solid.Camera.isometric(sc)
        else:
            az, el = {"three-quarter": (30, 18), "front": (0, 8), "side": (90, 8), "top": (20, 70),
                      "low": (30, 4), "high": (35, 40)}.get(cam, (30, 18))
            sc.camera = solid.Camera.frame(sc, azimuth=az, elevation=el)
        bgc = s3.get("background")
        if bgc is None:
            bgc = [(0, "#dbe6ee"), (1, "#a9c0d0")] if room is not None else None
        elif isinstance(bgc, list):
            bgc = [(i / max(1, len(bgc) - 1), c) for i, c in enumerate(bgc)]
        sc.render(cv, log=ctx.log, background=bgc)
    except Exception as e:
        _err("scene3d", f"3D scene failed: {_kit_error(e)}")


def _studio_layout(solid, objs):
    """A product shot without a room: the things stand together as one group, the tallest at the centre and a
    little back, the others beside it and a little forward, a small gap between them (left-to-right order follows
    their "at" words or numbers). Returns {(path, instance): (x, z, rotate)}."""
    items = []
    for k, ob in enumerate(objs):
        if ob.get("on_obj"):
            continue
        at = ob.get("at") or {}
        hx = None
        if at.get("mode") == "point":
            hx = float(at["values"][0])                 # numbers keep their left-to-right order
        elif at.get("mode") == "words" and at.get("h") is not None:
            hx = float(at["h"]) - 0.5
        for i in range(ob.get("count", 1)):
            m, _ = _make_model(solid, ob, i)
            if m is None:
                continue
            lo, hi = m.bounds()
            w, hgt, d = (float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(hi[2] - lo[2]))
            items.append({"key": (ob["path"], i), "w": w, "h": hgt, "d": d, "hx": hx, "k": k + 0.01 * i})
    if not items:
        return {}
    if any(it["hx"] is not None for it in items):
        items.sort(key=lambda it: (it["hx"] if it["hx"] is not None else 0.0, it["k"]))
    else:                                              # tallest in the middle, the rest alternate right / left
        by = sorted(items, key=lambda it: -it["h"])
        row = [by[0]]
        for j, it in enumerate(by[1:]):
            if j % 2 == 0:
                row.append(it)
            else:
                row.insert(0, it)
        items = row
    med = float(np.median([max(it["w"], it["h"]) for it in items]))
    maxh = max(it["h"] for it in items)
    gap = 0.12 * med
    total = sum(it["w"] for it in items) + gap * (len(items) - 1)
    x = -total / 2
    out = {}
    for j, it in enumerate(items):
        cx = x + it["w"] / 2
        z = (0.45 - it["h"] / maxh) * 0.7 * med + (0.08 * med if j % 2 else 0.0)
        out[it["key"]] = (cx, z, None)
        x += it["w"] + gap
    return out


def _free_floor(m0, m, x, z, feet, R, gap=0.15):
    """Move a model (m0 placed at x, z = m) to the nearest floor spot that overlaps no footprint."""
    def hits(mm):
        lo, hi = mm.bounds()
        a = (float(lo[0]) - gap, float(lo[2]) - gap, float(hi[0]) + gap, float(hi[2]) + gap)
        if R is not None and (lo[0] < -R - 0.1 or hi[0] > R + 0.1 or lo[2] < -R - 0.1 or hi[2] > R + 0.1):
            return True
        return any(a[0] < f[2] and f[0] < a[2] and a[1] < f[3] and f[1] < a[3] for f in feet)
    if not hits(m):
        return m, None
    lo, hi = m.bounds()
    for r in np.arange(0.2, 4.01, 0.2):
        for k in range(16):
            ang = k * math.pi / 8
            nx, nz = x + r * math.cos(ang), z + r * math.sin(ang)
            cand = m0.place(x=float(nx), z=float(nz))
            if not hits(cand):
                return cand, (nx, nz)
    return m, None


def _room_words(at, R, key):
    """Words -> (x, z, rotation) inside a room: left wall = -x, back wall = -z."""
    h, d = at.get("h"), at.get("depth")
    v = at.get("v")
    text = at.get("text", "")
    x = 0.0 if h is None else (h - 0.5) * 2 * R * 0.8
    z = 0.0
    if d is not None:
        z = (d - 0.5) * 2 * R * 0.85
    elif v is not None:
        z = (v - 0.5) * 2 * R * 0.85
    rot = None
    if "back" in text or (v is not None and v < 0.3) or (d is not None and d < 0.3):
        z = -R + 0.5
    if "front" in text:
        z = R - 0.5
    if "wall" in text and "left" in text:
        x, rot = -R + 0.35, 90
    elif "wall" in text and "right" in text:
        x, rot = R - 0.35, -90
    elif "wall" in text:
        z = -R + 0.3
    if "corner" in text:
        x = (-R + 0.45) if (h is not None and h < 0.5) else (R - 0.45)
        z = -R + 0.45 if "back" in text or "front" not in text else R - 0.45
    return x, z, rot


def _make_model(solid, ob, i):
    src, key = ob["src"]
    col = ob.get("color")
    try:
        if src == "solid":
            fn = getattr(solid, key)
            params = inspect.signature(fn).parameters
            kw = {}
            if key == "quadruped" and ob.get("kind"):
                kw["kind"] = ob["kind"]
            if col:
                for p in ("color", "duvet", "body", "walls", "shirt", "leaves", "screen", "pot"):
                    if p in params:
                        kw[p] = col
                        break
            if key == "character" and ob.get("pose"):
                kw["pose"] = ob["pose"]
            if key == "bookshelf":
                kw.setdefault("seed", i + 1)
            m = fn(**kw)
        elif src == "clay_prop":
            from . import clay
            m = clay.prop(key, **({"color": col} if col else {}))
        else:
            from . import clay
            m = clay.animal(key, **({"color": col} if col else {}))
        sc_ = float(ob.get("scale") or 1.0) * (1.4 if key == "desk_lamp" else 1.0)
        if ob.get("in_room") and ob.get("scale") is None and float(max(m.size)) < 0.4:
            # a mug or a vase in a whole room is a few pixels: draw small things a little larger, more so on a table
            sc_ *= (2.0 if float(max(m.size)) < 0.25 else 1.5) if ob.get("on") else 1.5
        if sc_ != 1.0:
            m = m.scale(sc_)
        return m, None
    except Exception as e:
        _err(ob["path"], f"{key!r} could not be built: {_kit_error(e)}")
        return None, None


# --------------------------------------------------------------------------------------------- post
def _render_post(ctx):
    cv, norm = ctx.cv, ctx.norm
    post = norm.get("post")
    if post is False:
        return
    p = dict(post) if isinstance(post, dict) else {}
    group = norm["group"]
    defaults = {"design": dict(vignette=0.2, grain=0.008, soft_clip=0.92),
                "scene": dict(vignette=0.22, grain=0.008, soft_clip=0.9),
                "still_life": dict(vignette=0.3, grain=0.01, soft_clip=0.86),
                "scene3d": dict(vignette=0.12, grain=0.006)}.get(group, {})
    for k, v in defaults.items():
        p.setdefault(k, v)
    if not p:
        return
    ctx.log("post")
    try:
        if p.get("bloom"):
            cv.bloom(0.85, 24 * ctx.S, float(p["bloom"]))
        if p.get("vignette"):
            cv.vignette(float(p["vignette"]), 0.68)
        if p.get("soft_clip"):
            cv.soft_clip(float(p["soft_clip"]))
        if p.get("grain"):
            cv.grain(float(p["grain"]), ctx.seed)
    except Exception as e:
        _err("post", f"post failed: {_kit_error(e)}")


# ============================================================================================= examples
EXAMPLES = {
    "poster": {
        "pcscene": 1,
        "prompt": "A poster for a spring plant market called 'Green Days' on Saturday 10:00 at Riverside Park, green and cream",
        "kind": "poster",
        "size": [1080, 1350],
        "palette": "forest green and cream",
        "background": {"backdrop": "spotlight"},
        "layers": [
            {"draw": "potted monstera", "at": "centre", "size": "large"},
            {"draw": "potted cactus", "at": "far left", "size": "medium"},
            {"draw": "potted fern", "at": "far right", "size": "medium"},
            {"text": "Spring market", "role": "kicker"},
            {"text": "Green Days", "role": "title"},
            {"text": "Plants, pots and coffee", "role": "subtitle"},
            {"text": "Saturday 10:00", "role": "details"},
            {"text": "Riverside Park", "role": "details"}
        ],
        "style": None
    },
    "card": {
        "pcscene": 1,
        "prompt": "A social card that says 'Slow Mornings' by Ana Ruiz on a warm cream background",
        "kind": "card",
        "size": [1200, 630],
        "palette": "cream and terracotta",
        "background": {"color": "cream"},
        "layers": [
            {"draw": "coffee cup", "at": "centre", "size": "large", "color": "terracotta"},
            {"text": "Essay", "role": "kicker"},
            {"text": "Slow Mornings", "role": "title"},
            {"text": "by Ana Ruiz", "role": "byline"},
        ],
    },
    "wallpaper": {
        "pcscene": 1,
        "prompt": "A calm mountain wallpaper at dusk in deep teal and coral with the word 'Breathe'",
        "kind": "wallpaper",
        "size": [1920, 1080],
        "palette": "deep teal and coral",
        "background": {"backdrop": "mountains"},
        "layers": [
            {"text": "Breathe", "role": "title"},
            {"text": "in, hold, out", "role": "subtitle"},
        ],
    },
    "icons": {
        "pcscene": 1,
        "prompt": "A set of six app icons in navy",
        "kind": "icons",
        "size": [1200, 800],
        "background": {"color": "#f4f1ea"},
        "layers": [
            {"text": "App icons", "role": "title"},
            {"draw": "home icon", "color": "navy"},
            {"draw": "search icon", "color": "navy"},
            {"draw": "camera icon", "color": "navy"},
            {"draw": "calendar icon", "color": "navy"},
            {"draw": "mail icon", "color": "navy"},
            {"draw": "heart icon", "color": "navy"},
        ],
    },
    "scene": {
        "pcscene": 1,
        "prompt": "A girl flying a kite in a park on a sunny day, a dog beside her",
        "kind": "scene",
        "background": {"sky": "day", "ground": "grass", "far": "forest"},
        "layers": [
            {"draw": "cloud", "at": "top left"},
            {"draw": "oak", "at": "far right"},
            {"draw": "girl", "describe": "a girl in a yellow dress", "pose": "wave", "at": "left third"},
            {"draw": "dog", "at": "right of girl"},
            {"draw": "kite", "at": "above girl", "size": "small"},
        ],
    },
    "nature": {
        "pcscene": 1,
        "prompt": "An old oak on a hill with wildflowers and rocks on a summer afternoon",
        "kind": "nature",
        "background": {"sky": "day", "ground": "meadow", "far": "hills"},
        "layers": [
            {"draw": "cloud", "at": "top right"},
            {"draw": "old oak", "at": "left third", "size": "huge"},
            {"draw": "rock", "at": "right foreground"},
            {"draw": "poppy", "count": 5, "at": "foreground", "arrange": "scatter"},
            {"draw": "daisy", "count": 4, "at": "front left", "arrange": "row"},
        ],
    },
    "animals": {
        "pcscene": 1,
        "prompt": "A white rabbit sitting under a cherry tree on a spring morning, three butterflies in the sky",
        "kind": "animals",
        "background": {"sky": "morning", "ground": "grass", "far": "hills"},
        "layers": [
            {"draw": "cherry", "at": "left third", "size": "huge"},
            {"draw": "rabbit", "color": "white", "pose": "sit", "at": "under cherry"},
            {"draw": "butterfly", "count": 3, "at": "top right"},
            {"draw": "daisy", "count": 5, "at": "foreground", "arrange": "scatter"}
        ]
    },
    "people": {
        "pcscene": 1,
        "prompt": "Two friends talking in a park on a sunny afternoon, one waving, their dog sitting beside them",
        "kind": "people",
        "background": {"sky": "day", "ground": "grass", "far": "forest"},
        "layers": [
            {"draw": "cloud", "at": "top right"},
            {"draw": "oak", "at": "left", "size": "huge"},
            {"draw": "person", "describe": "a woman in a red coat", "pose": "wave", "at": "left third"},
            {"draw": "person", "describe": "a man in a blue jacket", "pose": "talk", "at": "right third"},
            {"draw": "dog", "pose": "sit", "at": "right of the man"},
            {"draw": "daisy", "count": 5, "at": "foreground", "arrange": "scatter"},
        ],
    },
    "landscape": {
        "pcscene": 1,
        "prompt": "Mountains at sunset with pine trees",
        "kind": "landscape",
        "background": {"sky": "sunset", "far": "mountains", "ground": "meadow", "horizon": 0.64},
        "layers": [
            {"draw": "sun", "at": "centre", "size": "small"},
            {"draw": "pine", "count": 3, "at": "left third", "arrange": "row", "size": "large"},
            {"draw": "pine", "count": 2, "at": "right", "arrange": "row"},
        ],
    },
    "still_life": {
        "pcscene": 1,
        "prompt": "A still life: three oranges, a blue vase and a white mug on a wooden table",
        "kind": "still_life",
        "background": {"table": "wood", "wall": "#d8c6ac"},
        "layers": [
            {"draw": "vase", "at": "back left", "color": "steelblue"},
            {"draw": "mug", "at": "centre", "color": "white"},
            {"draw": "orange", "count": 3, "at": "front right", "arrange": "cluster"},
        ],
    },
    "room": {
        "pcscene": 1,
        "prompt": "A bright isometric living room with a sofa, an armchair, a coffee table with a vase, a rug, a bookshelf, a floor lamp and a plant",
        "kind": "room",
        "scene3d": {
            "room": {"walls": "#e8eef0", "floor": "wood"},
            "objects": [
                {"draw": "sofa", "color": "#c8664a"},
                {"draw": "armchair", "color": "#d9c7a6"},
                {"draw": "rug", "color": "#6f8fb0"},
                {"draw": "table", "at": "centre", "size": 0.8},
                {"draw": "vase", "on": "table", "color": "#e0b04a"},
                {"draw": "bookshelf"},
                {"draw": "floor lamp"},
                {"draw": "plant"}
            ],
            "lighting": "interior",
            "camera": "isometric"
        }
    },
    "scene3d": {
        "pcscene": 1,
        "prompt": "A yellow mug, a green bottle and two red apples, studio product shot",
        "kind": "scene3d",
        "scene3d": {
            "objects": [
                {"draw": "bottle", "color": "seagreen"},
                {"draw": "mug", "color": "goldenrod"},
                {"draw": "apple", "count": 2},
            ],
            "lighting": "studio",
            "camera": "three-quarter",
            "ground": "#e9e1d6",
            "background": ["#f3eee8", "#d9cfc4"],
        },
    },
    "diagram": {
        "pcscene": 1,
        "prompt": "A data pipeline diagram: the app sends events to a queue, a worker reads the queue and writes to a warehouse, and a dashboard reads the warehouse.",
        "kind": "diagram",
        "diagram": {
            "title": "Event pipeline",
            "theme": "light",
            "nodes": ["App", "Event queue", "Worker", "Warehouse", "Dashboard"],
            "edges": [
                ["App", "Event queue", "events"],
                ["Event queue", "Worker"],
                ["Worker", "Warehouse", "writes"],
                ["Dashboard", "Warehouse", "reads"]
            ],
            "groups": {"Processing": ["Event queue", "Worker"]}
        }
    },
    "chart": {
        "pcscene": 1,
        "prompt": "A bar chart of monthly sign-ups titled 'Sign-ups 2026'",
        "kind": "chart",
        "chart": {
            "type": "bar",
            "title": "Sign-ups 2026",
            "data": {"Jan": 320, "Feb": 410, "Mar": 380, "Apr": 520},
            "ylabel": "sign-ups",
        },
    },
    "dashboard": {
        "pcscene": 1,
        "prompt": "A sales dashboard with revenue, orders and conversion, a revenue trend and sales by channel",
        "kind": "dashboard",
        "dashboard": {
            "title": "Store performance",
            "subtitle": "Q3 2026",
            "theme": "dark",
            "kpis": [
                {"label": "Revenue", "value": 128400, "delta": 12.4, "fmt": "$"},
                {"label": "Orders", "value": 1842, "delta": 8.1},
                {"label": "Conversion", "value": "3.9%", "delta": -0.4},
            ],
            "charts": [
                {"type": "line", "title": "Revenue by month", "yfmt": "$k",
                 "data": {"Apr": 31000, "May": 34000, "Jun": 33000, "Jul": 38000, "Aug": 41000, "Sep": 49000}},
                {"type": "donut", "title": "Sales by channel", "data": {"Search": 38, "Direct": 26, "Social": 21, "Email": 15}},
            ],
        },
    },
    "ui": {
        "pcscene": 1,
        "prompt": "A habit app home screen that says 'Hello, Maya' with cards for 'Morning run', 'Reading' and 'Sleep'",
        "kind": "ui",
        "ui": {"template": "home", "screen": "phone", "theme": "light", "accent": "#3a7d6b",
               "content": {"title": "Hello, Maya", "subtitle": "Tuesday, 3 March",
                           "cards": [{"title": "Morning run", "text": "5 km at 7:00", "progress": 40},
                                     {"title": "Reading", "text": "20 pages a day"},
                                     {"title": "Sleep", "text": "7 h 40 min last night"}]}},
    },
}


def example(kind):
    """A complete example spec (dict) for a kind."""
    k = KIND_ALIASES.get(_norm(kind), _norm(kind).replace(" ", "_"))
    if k not in EXAMPLES:                              # "film poster", "line chart", "mobile app screen"
        probs = []
        guess = _resolve_kind(str(kind), {}, probs)
        if not probs:
            k = guess
    if k not in EXAMPLES:
        raise KeyError(f"no example for {kind!r}; kinds: {', '.join(EXAMPLES)}")
    return copy.deepcopy(EXAMPLES[k])


def schema_text():
    """The format on one screen (what --schema prints)."""
    kinds = "\n".join(f"  {k:<11} {v['desc']}" for k, v in KINDS.items())
    return f"""pcscene version {VERSION}: one JSON object.

  "pcscene": 1
  "prompt": "the request, verbatim"            (the report checks the picture against it)
  "kind": KIND                                   (see below)
  "size": [width, height] or portrait / landscape / square / story / thumbnail
  "palette": "deep blue and gold" | "sunset" | ["navy", "gold"]
  "background": {{"sky": "sunset", "ground": "grass", "far": "hills"}}       scenes
                {{"backdrop": "spotlight"}} | {{"color": "sage"}} | {{"gradient": ["navy", "black"]}}   designs
                {{"table": "wood", "wall": "#d8c6ac"}}                             still life
  "layers": [ ... one object per thing ... ]
     {{"draw": NAME, "at": WHERE, "size": SIZE, "color": COLOUR, "count": 3, "arrange": "row"}}
     {{"text": "exact words", "role": "title"}}     roles: {', '.join(ROLES)}
     {{"python": "file.py:function", "at": [u0, v0, u1, v1]}}   calls function(cv, box, layer)
  "style": null or a style name (watercolor, oil, gouache, pixel_art ...)
  "post": {{"vignette": 0.3, "grain": 0.01, "bloom": 0.2}} or false

NAME   anything compose.py --names WORD prints (animals, plants, people, objects, effects, pictograms, icons).
       Options go in the same object: pose, coat, facing, describe (people), material, style (pictograms).
WHERE  centre, left, right, top left, bottom third, far, foreground, in the sky, on the ground,
       left of cat, right of tree, on table, under tree, behind tree, in front of house, above tree,
       [u, v] (0-1; the base of a thing on the ground, the centre of a thing in the sky) or [u0, v0, u1, v1].
SIZE   tiny, small, medium, large, huge, or a number = fraction of the picture height.

Whole-picture kinds use one section instead of layers:
  "diagram":   {{"nodes": ["Web app", "API"], "edges": [["Web app", "API", "HTTPS"]], "groups": {{"Clients": ["Web app"]}}, "theme": "dark"}}
  "chart":     {{"type": "bar", "data": {{"Q1": 120, "Q2": 150}}, "title": "Revenue", "ylabel": "thousand $"}}
  "dashboard": {{"title": "...", "kpis": [{{"label": "Orders", "value": 1842, "delta": 8.1}}], "charts": [{{chart}}, ...]}}
  "ui":        {{"template": "login", "screen": "phone", "content": {{"title": "Welcome"}}}}
  "scene3d":   {{"room": {{}}, "objects": [{{"draw": "bed"}}, {{"draw": "lamp", "on": "desk"}}], "camera": "isometric"}}

Kinds:
{kinds}
"""


# ============================================================================================= edits
def _split_path(path):
    parts = []
    for tok in re.split(r"\.", str(path).strip()):
        for m in re.finditer(r"([^\[\]]+)|\[(\d+)\]", tok):
            if m.group(1) is not None:
                parts.append(int(m.group(1)) if m.group(1).isdigit() else m.group(1))
            else:
                parts.append(int(m.group(2)))
    return parts


def _value(text):
    if not isinstance(text, str):
        return text
    t = text.strip()
    try:
        return json.loads(t)
    except ValueError:
        pass
    try:
        v, _ = loads(t) if t.startswith(("{", "[")) else (None, None)
        if v is not None:
            return v
    except SpecError:
        pass
    low = t.lower()
    if low in _LITERALS:
        return json.loads(_LITERALS[low])
    return t


def set_value(spec, path, value):
    """``set_value(spec, "layers.1.size", "0.5")`` -> edits the dict in place; returns a description."""
    parts = _split_path(path)
    if not parts:
        raise KeyError("empty path")
    if isinstance(parts[0], int):
        parts = ["layers"] + parts
    if parts[0] == "layers" and "layers" not in spec and "scene3d" in spec:
        parts = ["scene3d", "objects"] + parts[1:]
    cur = spec
    for i, p in enumerate(parts[:-1]):
        nxt = parts[i + 1]
        if isinstance(p, int):
            if not isinstance(cur, list) or p >= len(cur) or p < -len(cur):
                n = len(cur) if isinstance(cur, list) else 0
                raise KeyError(f"{'.'.join(map(str, parts[:i + 1]))}: there are {n} items (0-{n - 1})")
            cur = cur[p]
        else:
            if not isinstance(cur, dict):
                raise KeyError(f"{'.'.join(map(str, parts[:i]))} is not an object")
            if p not in cur or cur[p] is None:
                cur[p] = [] if isinstance(nxt, int) else {}
            cur = cur[p]
    last = parts[-1]
    v = _value(value)
    if isinstance(last, int):
        if not isinstance(cur, list) or last >= len(cur):
            raise KeyError(f"{path}: index {last} out of range")
        cur[last] = v
    else:
        if not isinstance(cur, dict):
            raise KeyError(f"{path}: the parent is not an object")
        if v is None or v == "":
            cur.pop(last, None)
            return f"removed {path}"
        cur[last] = v
    return f"set {path} = {json.dumps(v, ensure_ascii=False)}"


def add_layer(spec, layer):
    """Append a layer (dict or JSON text); for room / scene3d kinds it goes to scene3d.objects, for diagrams a
    string becomes a node. Returns a description."""
    v = _value(layer) if isinstance(layer, str) else layer
    if isinstance(v, str):
        v = {"draw": v}
    kind = _norm(spec.get("kind") or "")
    if kind in ("room", "scene3d") or ("scene3d" in spec and "layers" not in spec):
        s3 = spec.setdefault("scene3d", {})
        s3.setdefault("objects", []).append(v)
        return f"added scene3d.objects[{len(s3['objects']) - 1}] = {json.dumps(v, ensure_ascii=False)}"
    if kind == "diagram" and isinstance(spec.get("diagram"), dict) and isinstance(v, (str, dict)) and "text" not in v and "draw" not in v:
        spec["diagram"].setdefault("nodes", []).append(v)
        return f"added a diagram node {json.dumps(v, ensure_ascii=False)}"
    if not isinstance(v, dict):
        raise ValueError("--add takes one JSON object, e.g. '{\"draw\": \"moon\", \"at\": \"top right\"}'")
    spec.setdefault("layers", []).append(v)
    return f"added layers[{len(spec['layers']) - 1}] = {json.dumps(v, ensure_ascii=False)}"


def remove_layer(spec, name):
    """Remove a layer by index (``2`` / ``layers.2``) or by its draw / text / id. Returns a description."""
    key = str(name).strip()
    lists = []
    if isinstance(spec.get("layers"), list):
        lists.append(("layers", spec["layers"]))
    if isinstance(spec.get("scene3d"), dict) and isinstance(spec["scene3d"].get("objects"), list):
        lists.append(("scene3d.objects", spec["scene3d"]["objects"]))
    m = re.fullmatch(r"(?:layers\.?\[?)?(\d+)\]?", key)
    if m and lists:
        name_, lst = lists[0]
        i = int(m.group(1))
        if i >= len(lst):
            raise KeyError(f"{name_} has {len(lst)} items (0-{len(lst) - 1})")
        it = lst.pop(i)
        return f"removed {name_}[{i}] {json.dumps(it, ensure_ascii=False)}"
    k = _norm(key)
    for name_, lst in lists:
        for i, it in enumerate(lst):
            if not isinstance(it, dict):
                if _norm(it) == k:
                    lst.pop(i)
                    return f"removed {name_}[{i}] {it!r}"
                continue
            vals = [it.get(f) for f in ("id", "label", "draw", "text", "prefab", "name")]
            if any(isinstance(x, str) and (_norm(x) == k) for x in vals):
                lst.pop(i)
                return f"removed {name_}[{i}] {json.dumps(it, ensure_ascii=False)}"
    for name_, lst in lists:
        for i, it in enumerate(lst):
            vals = [it.get(f) for f in ("id", "label", "draw", "text")] if isinstance(it, dict) else [it]
            if any(isinstance(x, str) and k in _norm(x) for x in vals):
                lst.pop(i)
                return f"removed {name_}[{i}] {json.dumps(it, ensure_ascii=False)}"
    raise KeyError(f"no layer {name!r}")


# ============================================================================================= the NEXT block
def _layer_by_label(norm, label):
    base = str(label).split(" #")[0]
    for L in norm.get("layers", []):
        if L.get("label") == base:
            return L
    return None


def _free_spot(ctx_placed, box, avoid, horizon):
    """A nearby [u, v] base for ``box`` that does not overlap ``avoid`` (both fractions)."""
    w = box[2] - box[0]
    cands = []
    for du in (-0.25, 0.25, -0.4, 0.4, -0.15, 0.15):
        u = (box[0] + box[2]) / 2 + du
        if not 0.08 <= u <= 0.92:
            continue
        nb = [u - w / 2, box[1], u + w / 2, box[3]]
        ov = max(0.0, min(nb[2], avoid[2]) - max(nb[0], avoid[0]))
        cands.append((ov, abs(du), u))
    if not cands:
        return None
    cands.sort()
    return round(cands[0][2], 2)


def _prompt_things(prompt):
    """Drawable things the request names (outside quotes): [(word, Thing)]."""
    if not prompt:
        return []
    from .manifest import quoted
    plain = prompt
    try:
        for q in quoted(prompt):
            plain = plain.replace(q, " ")
    except Exception:
        pass
    reg, alias = _registry()
    out = []
    low = plain.lower()
    colour_ctx = r"(?:warm|deep|dark|light|bright|pale|soft|muted|rich|burnt|and|in|with)"
    for w in set(re.findall(r"[a-z]+", low)):            # a colour word used as a colour, not a fruit
        if parse_color(w) is not None and re.search(rf"\b{colour_ctx}\s+{w}\b|\b{w}\s+(?:and|tones?|accents?|"
                                                    rf"colou?rs?|palette|background|light|glow|sky)\b", low):
            low = re.sub(rf"\b{w}\b", " ", low)
    words = re.findall(r"[a-z]+", low)
    skip = {"room", "hour", "night", "day", "blue", "gold", "palette", "theme", "dark", "light", "poster", "card",
            "chart", "diagram", "background", "ground", "sky", "picture", "image", "set", "style", "title",
            "green", "soft", "deep", "red", "art", "lamp", "app", "service", "gateway", "database", "web",
            "mobile", "sun", "moon", "star", "stars", "clouds", "cloud", "water", "fire", "light", "wind",
            "time", "home", "friday", "music", "jazz", "calm", "pine", "table", "line", "bar", "text", "word",
            "words", "one", "two", "three", "four", "with", "and", "the", "for"}
    skip |= set(_ATTR_WORDS) | set(_POSE_WORDS) | set(_POSE_WORDS.values())
    i = 0
    seen = set()
    while i < len(words):
        hit = None
        for n in (2, 1):
            phrase = " ".join(words[i:i + n])
            if len(words[i:i + n]) < n:
                continue
            for cand in (phrase, _singular(phrase)):
                if cand in skip or len(cand) < 3:
                    continue
                t = reg.get(cand) or (reg.get(alias[cand]) if cand in alias else None)
                if t is not None and t.kit in ("creature", "flora", "figure", "form", "pictogram"):
                    hit = (cand, t, n)
                    break
            if hit:
                break
        if hit:
            if hit[1].name not in seen:
                seen.add(hit[1].name)
                out.append((hit[0], hit[1]))
            i += hit[2]
        else:
            i += 1
    return out


_STOP = {"a", "an", "the", "and", "or", "of", "for", "with", "in", "on", "at", "to", "by", "from", "make", "draw",
         "create", "picture", "image", "poster", "card", "social", "called", "named", "that", "says", "which",
         "this", "is", "are", "be", "use", "skill", "files", "file", "put", "deliver", "final", "palette", "theme",
         "background", "colour", "color", "colours", "colors", "style", "night", "day", "friday", "saturday",
         "sunday", "monday", "tuesday", "wednesday", "thursday", "soft", "deep", "dark", "light", "bright",
         "green", "blue", "gold", "red", "white", "black", "pink", "sage", "cream", "art", "pixelcanvas", "lite",
         "some", "very", "its", "their", "his", "her", "my", "your", "our", "into", "under", "over", "above",
         "below", "behind", "two", "three", "four", "one", "five", "set", "kind", "big", "small", "large",
         "sunny", "sunset", "sunrise", "cloudy", "rainy", "morning", "evening", "afternoon", "summer", "winter",
         "spring", "autumn", "cosy", "cozy", "calm", "beautiful", "nice", "cute", "isometric", "dark",
         "warm", "cool", "cold", "hot", "bold", "minimal", "minimalist", "modern", "clean", "simple", "elegant",
         "vintage", "retro", "film", "movie", "tagline", "title", "subtitle", "banner", "flyer", "cover", "text",
         "words", "font", "type", "typography", "navy", "teal", "purple", "yellow", "grey", "gray", "brown",
         "violet", "silver", "terracotta", "pastel", "neon", "vibrant", "muted", "flat", "layout", "design",
         "quote", "thumbnail", "wallpaper", "illustration", "scene", "high", "contrast", "event", "launch"}


def _content_words(prompt):
    from .manifest import quoted
    plain = prompt or ""
    try:
        for q in quoted(plain):
            plain = plain.replace(q, " ")
    except Exception:
        pass
    plain = plain.split("Use the pixelcanvas")[0]
    toks = re.findall(r"[A-Za-z][A-Za-z\-]+", plain)
    proper = {t.lower() for i, t in enumerate(toks) if i > 0 and t[0].isupper()}        # names: The Velvet Room
    words = [w for w in (t.lower() for t in toks) if w not in _STOP and w not in proper and len(w) > 2
             and w not in _POSE_WORDS and w not in _ATTR_WORDS
             and parse_color(w) is None and not re.fullmatch(r"\d+", w)]
    return list(dict.fromkeys(words))


def _leftovers(norm):
    """Layers copied unchanged from the kind's example that the request never mentions."""
    ex = EXAMPLES.get(norm.get("kind")) or {}
    if (norm.get("prompt") or "").strip() == (ex.get("prompt") or "").strip():
        return []
    words = set(_content_words(norm.get("prompt")))
    prompt_low = (norm.get("prompt") or "").lower()
    ex_draws = {_norm(L.get("draw")) for L in ex.get("layers", []) if isinstance(L, dict) and L.get("draw")}
    ex_texts = {str(L.get("text")).strip().lower() for L in ex.get("layers", []) if isinstance(L, dict) and L.get("text")}
    out = []
    s3 = norm.get("scene3d") or {}
    ex3 = {_norm(o.get("draw")) for o in ((ex.get("scene3d") or {}).get("objects") or []) if isinstance(o, dict)}
    for o in s3.get("objects") or []:
        nm = _norm(o.get("name", ""))
        alias_ = _PREFAB_ALIASES.get(nm, nm).replace("_", " ")
        if nm in ex3 and not o.get("added") and not ({nm, alias_} & words) and not any(_singular(w) in (nm, alias_) for w in words):
            out.append(f"{o['path']} {o['label']!r} is left over from the example and the request does not mention "
                       f"it -> replace it with an object from the request, or remove it")
    for L in norm.get("layers", []):
        decor = L["type"] == "thing" and (L["thing"].cat in ("sky", "fullframe", "rock") or
                                          (L["thing"].cat == "plant" and L["thing"].size < 0.2))
        if L["type"] == "thing" and _norm(L.get("name", "")) in ex_draws and not decor:
            names_ = set(_norm(L.get("name", "")).split()) | {L["thing"].name} | set(L["thing"].name.split())
            if L["thing"].kit == "figure":                 # "two friends", "a couple", "a man in a blue jacket"
                names_ |= _words_of(L) | set(_norm(str(L["opts"].get("describe") or "")).split())
                if (norm.get("prompt") or "") and re.search(r"\b(people|friends?|couple|family|kids|children|"
                                                            r"players|workers|team|crowd)\b", prompt_low):
                    continue
            if not (names_ & words) and not any(_singular(w) in names_ for w in words):
                out.append(f"{L['path']} {L['label']!r} is left over from the example and the request does not "
                           f"mention it -> replace it with a thing from the request, or remove it")
        elif L["type"] == "text" and L["text"].strip().lower() in ex_texts and L["text"].strip().lower() not in prompt_low:
            out.append(f"{L['path']} text {L['text']!r} is left over from the example -> write your own words or remove it")
    return out


_REL_PAT = (r"under|underneath|beneath|below|on top of|sitting on|standing on|perched on|on|behind|in front of|"
            r"next to|beside|near|left of|right of|above|over")


def _prompt_relations(norm):
    """(A, rel, B) for each relation the request names between two drawn layers ("a cat under the oak")."""
    prompt = (norm.get("prompt") or "").lower()
    things = [L for L in norm.get("layers", []) if L["type"] == "thing"]
    out = []
    generic = {"animal", "plant", "object", "sky", "effect", "rock", "person", "people", "pictogram", "icon"}
    best = {}
    for A in things:
        if A["thing"].cat in ("fullframe", "sky"):     # snow, rain and the sun relate to nothing
            continue
        wa = sorted({w for w in _words_of(A) if w not in generic and len(w) > 2}, key=len, reverse=True)
        for B in things:
            if B is A:
                continue
            wb = sorted({w for w in _words_of(B) if w not in generic and len(w) > 2}, key=len, reverse=True)
            hit = None
            for x in wa:
                for y in wb:
                    m = re.search(rf"\b{re.escape(x)}s?\b(?:\W+\w+){{0,5}}?\W+({_REL_PAT})\W+(?:\w+\W+){{0,2}}?{re.escape(y)}s?\b",
                                  prompt)
                    if m:
                        hit = (m.group(1), y)
                        break
                if hit:
                    break
            if not hit:
                continue
            rel = hit[0].replace("sitting on", "on").replace("standing on", "on").replace("perched on", "on")
            rel = REL_CANON.get(rel, rel)
            own = hit[1] in (B["label"], B.get("name"), B["thing"].name)     # "birch" beats the generic "tree"
            key = (id(A), rel)
            if key not in best or (own and not best[key][1]):
                best[key] = ((A, rel, B), own)
    return [v[0] for v in best.values()]


def _relation_lines(norm):
    """Relations named in the request between two drawn layers that the spec does not express."""
    out = []
    for A, rel, B in _prompt_relations(norm):
        at = A.get("at") or {}
        if at.get("rel") == rel and at.get("target_layer") == B["label"]:
            continue
        out.append(f"the request puts {A['label']!r} {rel} {B['label']!r} -> set \"at\": \"{rel} {B['label']}\" "
                   f"in {A['path']}")
    return out


def _apply_prompt_poses(norm, notes):
    """An animal or person without a pose takes the one the request gives it ("a cat sitting ...")."""
    prompt = (norm.get("prompt") or "").lower()
    if not prompt:
        return
    for L in norm.get("layers", []):
        if L["type"] != "thing" or L["thing"].kit not in ("creature", "figure") or L["opts"].get("pose"):
            continue
        poses = list(importlib.import_module(f"pixelcanvas.{L['thing'].kit}").POSES)
        for w in sorted({w for w in _words_of(L) if len(w) > 2}, key=len, reverse=True):
            m = re.search(rf"\b{re.escape(w)}s?\b\W+(?:\w+\W+){{0,2}}?(\w+ing|sits|stands|asleep|curled up|perched|seated|relaxed)\b",
                          prompt)
            if not m:
                continue
            word = {"sits": "sitting", "stands": "standing"}.get(m.group(1), m.group(1))
            p = _POSE_WORDS.get(word, word).replace(" ", "_")
            if p in poses:
                L["opts"]["pose"] = p
                notes.append(f"{L['path']}.pose: the request says {word!r} -> pose {p!r}")
            break


def _apply_prompt_relations(norm, notes):
    """A layer the request relates to another ("cat under the oak") but placed without a relation gets it."""
    if norm.get("group") not in ("scene", "still_life"):
        return
    for A, rel, B in _prompt_relations(norm):
        at = A.get("at") or {}
        if at.get("rel") or A.get("count", 1) > 1 or (B.get("at") or {}).get("target_layer") == A["label"]:
            continue
        if A["thing"].cat in ("sky", "fullframe") or B["thing"].cat in ("sky", "fullframe"):
            continue
        A["at"] = {"mode": "words", "text": f"{rel} {B['label']}", "rel": rel, "target": B["label"],
                   "target_layer": B["label"]}
        notes.append(f"{A['path']}.at: the request puts {A['label']!r} {rel} {B['label']!r}, so it is placed "
                     f"\"{rel} {B['label']}\"")


def _topic_line(norm):
    """When no picture layer relates to the request's words, name things that would."""
    words = _content_words(norm.get("prompt"))
    if not words:
        return None
    reg, alias = _registry()
    drawn = set()
    for L in norm.get("layers", []):
        if L["type"] == "thing":
            drawn |= _words_of(L) | {L["thing"].name}
    for v in (norm.get("background") or {}).values():
        for x in (v if isinstance(v, (list, tuple)) else [v]):
            if isinstance(x, str):
                drawn |= {x} | set(x.replace("_", " ").split())
    drawn |= {_singular(d) for d in drawn}
    if any(w in drawn or _singular(w) in drawn for w in words):
        return None
    cands, why = [], {}
    try:
        P = importlib.import_module("pixelcanvas.pictograms")
    except Exception:
        P = None
    for w in words[:8]:
        hits = []
        t = reg.get(w) or reg.get(_singular(w)) or (reg.get(alias[w]) if w in alias else None)
        if t is not None and t.kit != "icon":
            hits.append(t.name)
        if P is not None:
            try:
                for n in P.find(w, n=3):
                    inf = P.info(n) or {}
                    toks = set(str(n).split()) | {str(a).lower() for a in inf.get("aliases", ())} |                         {str(k).lower() for k in inf.get("keywords", ())}
                    if w not in toks and _singular(w) not in toks:
                        continue
                    if n in reg:
                        hits.append(n)
                    else:
                        hits += [k for k, th in reg.items() if th.kit == "pictogram" and th.target == n]
            except Exception:
                pass
        for h in hits:
            if h not in cands:
                cands.append(h)
                why.setdefault(h, w)
    if not cands or set(cands) & drawn:
        return None
    first = cands[0]
    others = ", ".join(repr(c) for c in cands[1:4])
    return (f"nothing in the picture shows {why[first]!r} from the request -> add {{\"draw\": \"{first}\"}}"
            + (f" (or {others})" if others else ""))


def next_steps(norm, report, render_errors=(), compose_cmd="compose.py SPEC"):
    """The NEXT block: at most five concrete spec edits, and the stop line. Returns (lines, n_actionable)."""
    items = []          # (priority, text)
    layers = norm.get("layers", []) if norm else []
    for p in render_errors:
        items.append((0, f"{p.path}: {p.message}" + (f" -> {p.fix}" if p.fix else "")))
    rows = report.get("objects") or []
    group = norm.get("group") if norm else None
    # hidden / clipped objects
    for r in rows:
        if r.get("kind") == "text":
            continue
        L = _layer_by_label(norm, r.get("name", ""))
        if L is None or L["type"] != "thing":
            continue
        ref = f"{L['path']} {L['label']!r}"
        vis = r.get("visible_pct", 100)
        if vis < 45 and r.get("area_pct", 0) > 0.05 and L["thing"].cat not in ("fullframe",):
            by = r.get("hidden_by")
            hb = None
            for q in rows:
                if by and q.get("name") == by:
                    hb = q["bbox"]
            u = _free_spot(None, r["bbox"], hb, norm) if hb else None
            if by:
                fix = (f"set \"at\" to [{u}, {round(r['bbox'][3], 2)}]" if u is not None else "move it or make it larger") + \
                    f" (or move {by!r})"
                items.append((1, f"{ref} is {100 - vis:.0f}% hidden by {by!r} -> {fix}"))
            else:
                items.append((1, f"{ref} is {100 - vis:.0f}% hidden -> move it or put it later in the list"))
        b = r["bbox"]
        inside = b[0] > 0.01 and b[1] > 0.01 and b[2] < 0.99 and b[3] < 0.99
        if r.get("clipped_pct", 0) > 30 and not inside and L.get("count", 1) == 1 \
                and L["thing"].cat != "sky":
            cu = round(min(0.85, max(0.15, (b[0] + b[2]) / 2)), 2)
            items.append((1, f"{ref} is {r['clipped_pct']:.0f}% outside the picture -> set \"at\" nearer the centre "
                             f"(u about {cu}) or a smaller \"size\""))
        if r.get("area_pct", 1) < 0.12 and L["thing"].cat in ("animal", "person", "plant", "object", "pictogram") \
                and L["count"] < 3 and not r.get("name", "").count("#"):
            items.append((3, f"{ref} is tiny ({r['area_pct']:.2f}% of the picture) -> set \"size\" to \"medium\" or larger"))
    # two picture things on top of each other (designs and scenes)
    things = []
    for r in rows:
        L = _layer_by_label(norm, r.get("name", "")) if norm else None
        if L is not None and L["type"] == "thing" and L["thing"].cat not in ("fullframe", "sky") \
                and "#" not in r.get("name", "") and L.get("count", 1) == 1:
            things.append((r, L))

    def _core(b):                                       # shadows and leaf tips make boxes look larger
        dx, dy = 0.1 * (b[2] - b[0]), 0.1 * (b[3] - b[1])
        return [b[0] + dx, b[1] + dy, b[2] - dx, b[3] - dy]

    for i, (ra, La) in enumerate(things):
        for rb, Lb in things[i + 1:]:
            if La is Lb or "effect" in (La["thing"].cat, Lb["thing"].cat):   # a fire's sparks and smoke are airy
                continue
            A, B = _core(ra.get("solid_bbox") or ra["bbox"]), _core(rb.get("solid_bbox") or rb["bbox"])
            iw = max(0.0, min(A[2], B[2]) - max(A[0], B[0]))
            ih = max(0.0, min(A[3], B[3]) - max(A[1], B[1]))
            aa, ab = (A[2] - A[0]) * (A[3] - A[1]), (B[2] - B[0]) * (B[3] - B[1])
            small = min(aa, ab)
            limit = 0.2 if group == "design" else 0.4
            if small <= 0 or iw * ih / small < limit:
                continue
            rel_a = (La.get("at") or {}).get("target_layer") == Lb["label"]
            rel_b = (Lb.get("at") or {}).get("target_layer") == La["label"]
            if rel_a or rel_b:
                continue                                    # "on", "under" ... overlap on purpose
            S_, Bg = (La, A) if aa <= ab else (Lb, B)
            O_ = Lb if S_ is La else La
            side = "left" if (Bg[0] + Bg[2]) / 2 < 0.5 else "right"
            items.append((1, f"{S_['path']} {S_['label']!r} sits on top of {O_['label']!r} "
                             f"({iw * ih / small * 100:.0f}% overlap) -> set \"at\" to \"{side}\" "
                             f"or \"size\" to \"small\""))
    # a picture thing over the words of a design
    if group == "design":
        texts = [r for r in rows if r.get("kind") == "text"]
        told = set()
        for r in rows:
            L = _layer_by_label(norm, r.get("name", "")) if norm else None
            if L is None or L["type"] != "thing" or L["thing"].cat in ("fullframe", "sky") or id(L) in told:
                continue
            A = _core(r["bbox"])
            for t in texts:
                B = t["bbox"]
                iw = max(0.0, min(A[2], B[2]) - max(A[0], B[0]))
                ih = max(0.0, min(A[3], B[3]) - max(A[1], B[1]))
                ta = max((B[2] - B[0]) * (B[3] - B[1]), 1e-6)
                if iw * ih / ta > 0.2:
                    told.add(id(L))
                    words = t.get("name", "text").replace("label ", "")
                    items.append((1, f"{L['path']} {L['label']!r} covers the text {words} -> set \"size\" to "
                                     f"\"small\"" + (" and \"count\" lower" if L.get("count", 1) > 1 else "")
                                     + " or \"at\" to \"centre\""))
                    break
    # text
    for t in report.get("text_issues") or []:
        rule = t.get("rule", "")
        val = t.get("value") or ""
        L = next((x for x in layers if x["type"] == "text" and x["text"][:40] == str(val)[:40]), None)
        ref = f"{L['path']} {L['text'][:30]!r}" if L else f"text {str(val)[:30]!r}"
        if rule == "text_contrast":
            ink = "white" if LAST.get("bg_dark", True) else "black"
            lum = LAST.get("lum")
            b = t.get("where") or t.get("bbox")
            if lum is not None and b and len(b) == 4:
                hh, ww = lum.shape
                y0, y1 = int(b[1] * hh), max(int(b[1] * hh) + 1, int(np.ceil(b[3] * hh)))
                x0, x1 = int(b[0] * ww), max(int(b[0] * ww) + 1, int(np.ceil(b[2] * ww)))
                patch = lum[max(0, y0):min(hh, y1), max(0, x0):min(ww, x1)]
                if patch.size:
                    ink = "black" if float(patch.mean()) > 0.5 else "white"
            items.append((1 if t.get("severity") == "fail" else 2,
                          f"{ref}: low contrast on the background -> set \"color\" to \"{ink}\" in that layer, "
                          f"or a plainer background"))
        elif rule in ("text_clipped", "text_overlap"):
            items.append((1, f"{ref}: {t.get('message', '')[:80]} -> shorten the text or give it a box "
                             f"\"at\": [0.08, 0.7, 0.92, 0.85]"))
        elif rule == "text_missing_glyphs":
            items.append((0, f"{ref}: some characters cannot be drawn -> remove emoji or symbols"))
        elif rule == "text_small" and t.get("severity") != "info":
            items.append((3, f"{ref}: text is small -> use role \"subtitle\" or \"title\""))
    # prompt
    for p in report.get("prompt_issues") or []:
        rule = p.get("rule", "")
        if rule == "prompt_text_missing":
            q = p.get("value")
            role = "title" if not any(x["type"] == "text" and x["role"] == "title" for x in layers) else "subtitle"
            if group in ("diagram",):
                items.append((0, f"the request quotes {q!r} but no text shows it -> set diagram.title to {json.dumps(q, ensure_ascii=False)}"))
            elif group in ("chart", "dashboard"):
                sec = "chart" if group == "chart" else "dashboard"
                items.append((0, f"the request quotes {q!r} but no text shows it -> set {sec}.title to {json.dumps(q, ensure_ascii=False)}"))
            elif group in ("ui",):
                qj = json.dumps(q, ensure_ascii=False)
                if ((norm or {}).get("ui") or {}).get("template") == "home":
                    items.append((0, f"the request quotes {q!r} but no text shows it -> use it as \"title\" in "
                                     f"ui.content (the greeting) or as a card: {{\"title\": {qj}}} in ui.content.cards"))
                else:
                    items.append((0, f"the request quotes {q!r} but no text shows it -> put it in ui.content, "
                                     f"e.g. \"content\": {{\"title\": {qj}}}, or use \"template\": \"home\""))
            else:
                items.append((0, f"the request quotes {q!r} but no text layer contains it -> add "
                                 f"{{\"text\": {json.dumps(q, ensure_ascii=False)}, \"role\": \"{role}\"}}"))
        elif rule == "prompt_style_missing":
            items.append((1, f"the request asks for {p.get('value')!r} -> set \"style\": \"{p.get('value')}\""))
        elif rule == "prompt_count":
            want, have = p.get("threshold"), p.get("value")
            m = re.search(r"says \S+ (\S+) but", p.get("message", ""))
            word = m.group(1) if m else "it"
            tgt = None
            if norm:
                tgt = next((x for x in layers if x["type"] == "thing" and _find_target([x], word) is not None), None)
            ref = f"{tgt['path']} {tgt['label']!r}" if tgt else word
            items.append((1, f"the request says {want} {word} but {have} are drawn -> set \"count\": {want} in {ref}"))
        elif rule in ("prompt_colour", "prompt_small"):
            m = re.search(r"the ([a-z][a-z' -]*?) (?:is|are) painted|names the ([a-z][a-z' -]*?), drawn at",
                          str(p.get("message", "")))
            who = (m.group(1) or m.group(2)).strip() if m else ""
            L = _layer_by_label(norm, who) if norm and who else None
            if L is None and norm and who:
                L = next((x for x in layers if x["type"] == "thing" and _find_target([x], who.split()[-1]) is not None),
                         None)
            ref = f"{L['path']} {L['label']!r}" if L else (repr(who) if who else "that thing")
            if rule == "prompt_colour":
                cw = p.get("threshold")
                items.append((1, f"the request says {cw} but {who or 'the thing'} is drawn another colour -> set "
                                 f"\"color\": \"{cw}\" in {ref}"))
            elif not any(ref in t and "is tiny" in t for _, t in items):
                items.append((1, f"{who or 'a thing'} is too small to see -> set \"size\" to \"large\" in {ref}"))
        elif rule == "prompt_relation":
            items.append((1, f"a place the request states is not met ({str(p.get('message', ''))[:90]}) -> "
                             f"change \"at\" in the layer that is out of place (e.g. \"left\", \"right\", \"on <layer>\")"))
        elif rule == "prompt_text_small":
            mq = re.search(r"quoted text (['\"])(.*?)\1 is drawn", str(p.get("message", "")))
            q = mq.group(2) if mq else ""
            L = next((x for x in layers if x["type"] == "text" and q and q in x["text"]), None)
            ref = f"{L['path']} {L['text'][:30]!r}" if L else f"the text {q!r}"
            items.append((1, f"{ref} is too small to read -> set \"role\" to \"title\" or \"subtitle\""))
    # left over from the example, and nothing that shows the topic
    if norm and norm.get("kind") in EXAMPLES and group in ("design", "scene", "still_life", "scene3d"):
        items += [(0, t) for t in _leftovers(norm)]
        topic = _topic_line(norm) if group != "scene3d" else None
        if topic:
            items.append((1, topic))
    # relations the request states ("a cat under an oak") that the layers do not
    if norm and group in ("scene", "still_life", "design"):
        items += [(1, t) for t in _relation_lines(norm)]
    # things the request names that no layer draws
    if norm and group in ("scene", "still_life", "design"):
        drawn = set()
        for L in layers:
            if L["type"] == "thing":
                drawn |= _words_of(L)
        for v in (norm.get("background") or {}).values():
            for x in (v if isinstance(v, (list, tuple)) else [v]):
                if isinstance(x, str):
                    drawn |= {x, _singular(x)} | set(x.replace("_", " ").split())
        quoted_txt = " ".join(x["text"].lower() for x in layers if x["type"] == "text")
        for word, th in _prompt_things(norm.get("prompt")):
            names_ = {th.name, word, _singular(word), th.cat}
            if th.cat == "plant" and th.size >= 0.4:
                names_.add("tree")
            if names_ & drawn or word in quoted_txt:
                continue
            if any(word == w.split()[-1] for w in drawn):
                continue
            items.append((2, f"the request names {word!r} but no layer draws it -> add {{\"draw\": \"{th.name}\"}}"))
    if norm and norm.get("group") == "scene3d" and norm.get("scene3d"):
        objs = norm["scene3d"]["objects"]
        have = set()
        for o in objs:
            have |= {o["name"], o["src"][1], o["src"][1].replace("_", " "), o["label"]}
        prompt = (norm.get("prompt") or "").lower()
        for word, key in sorted(_PREFAB_ALIASES.items(), key=lambda kv: -len(kv[0])):
            if any(word in str(h).replace("_", " ").split() for h in have):   # "lamp" in a drawn "floor lamp"
                continue
            if re.search(rf"\b{re.escape(word)}s?\b", prompt) and key not in have and word not in have \
                    and key in _prefabs() and word not in ("person", "man", "woman", "home", "speaker", "block", "cup", "kid"):
                items.append((2, f"the request names {word!r} but no object is a {key.replace('_', ' ')} -> add "
                                 f"{{\"draw\": \"{word}\"}} to scene3d.objects"))
                have.add(key)
        for key in _prefabs():
            w = key.replace("_", " ")
            if re.search(rf"\b{re.escape(w)}s?\b", prompt) and key not in have and w not in have:
                items.append((2, f"the request names {w!r} but no object draws it -> add {{\"draw\": \"{w}\"}} to scene3d.objects"))
                have.add(key)
    if norm and group == "diagram" and norm.get("diagram"):
        prompt = (norm.get("prompt") or "").lower()
        labels = " ".join(n["label"].lower() for n in norm["diagram"]["nodes"])
        for kw_ in ("postgres", "mysql", "redis", "kafka", "database", "gateway", "auth", "cache", "queue", "mobile",
                    "web", "orders", "payments", "users", "search", "cdn", "storage"):
            if re.search(rf"\b{kw_}", prompt) and kw_ not in labels:
                items.append((2, f"the request names {kw_!r} but no node does -> add it to diagram.nodes and connect it in diagram.edges"))
        if "dark" in prompt and (norm["diagram"].get("theme") or "light") != "dark":
            items.append((1, "the request asks for a dark theme -> set diagram.theme to \"dark\""))
        if not norm["diagram"]["edges"] and len(norm["diagram"]["nodes"]) > 1:
            items.append((1, "no edges -> add diagram.edges such as [\"Web app\", \"API gateway\"]"))
    if norm and group == "chart" and norm.get("chart"):
        prompt = norm.get("prompt") or ""
        c = norm["chart"]
        data = c.get("data") if isinstance(c.get("data"), dict) else {}
        for lab, num in re.findall(r"\b([A-Z][A-Za-z0-9]{0,8})\s*[:=]?\s+(\d+(?:\.\d+)?)\b", prompt):
            if data and lab not in data and lab.lower() not in [str(k).lower() for k in data]:
                if re.fullmatch(r"Q\d|[A-Z][a-z]{2}", lab):
                    items.append((1, f"the request has {lab} {num} but chart.data has no {lab!r} -> add \"{lab}\": {num}"))
            elif data:
                key = next((k for k in data if str(k).lower() == lab.lower()), None)
                if key is not None and _num(data[key]) and abs(float(data[key]) - float(num)) > 1e-6:
                    items.append((0, f"chart.data[{key!r}] is {data[key]} but the request says {num} -> set it to {num}"))
        if "thousand" in prompt.lower() and not any(c.get(k) for k in ("ylabel", "yfmt")):
            items.append((2, "the request gives units (thousands) -> set chart.ylabel, e.g. \"thousand $\""))
    # image critique (only the ones a spec can fix)
    for w in report.get("warnings") or []:
        if w.get("severity") != "fail":
            continue
        rule = w.get("rule", "")
        if rule in ("underexposed", "too_dark", "crushed_blacks"):
            items.append((2, f"the picture is too dark ({rule}) -> use a lighter background or sky (\"day\")"))
        elif rule in ("overexposed", "blown_highlights") and not norm.get("style"):
            items.append((3, f"the picture is too bright ({rule}) -> darker background colour"))
    # unique, sorted, at most five
    seen, out = set(), []
    for pr, t in sorted(items, key=lambda x: x[0]):
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    n_act = len(out)
    out = out[:5]
    return out, n_act


def scene_module_source(spec_path, root):
    """Python source of the small scene module render.py runs for a spec file."""
    return f'''# -*- coding: utf-8 -*-
"""Generated by compose.py from {os.path.basename(spec_path)}: edit the JSON file, not this one."""
import os
import sys

sys.path.insert(0, {root!r})
from pixelcanvas import spec as _spec

SPEC_FILE = {os.path.abspath(spec_path)!r}
_S, _REPAIRS = _spec.load(SPEC_FILE)
_N, _PROBLEMS = _spec.validate(_S, base_dir=os.path.dirname(SPEC_FILE))
PROMPT = _N["prompt"] or ""
KIND = _N["check"]
DEFAULT_SIZE = tuple(_N["size"])
DEFAULT_SEED = _N["seed"]


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    return _spec.render(_N, width, height, seed=seed, log=log, base_dir=os.path.dirname(SPEC_FILE))
'''
