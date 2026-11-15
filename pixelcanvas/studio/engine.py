# -*- coding: utf-8 -*-
"""studio engine: run one cell on its own canvas, cache its RGBA, place it (move / scale / rotate / flip), add
its effects (outline, shadow, glow, tint, blur) and composite the stack. No file layout here: the caller hands
in the picture dict, the cell sources and a cache folder."""
from __future__ import annotations

import builtins
import ctypes
import hashlib
import pickle
import threading
import re
import io
import json
import math
import os
import sys
import time
import traceback

import numpy as np

from ..canvas import Canvas
from .. import layers as _layers
from .. import manifest as _manifest
from .. import filters as _f

F32 = np.float32
ENGINE = 12                                   # bump when a change makes old cache files wrong
MAX_RENDER_PX = 3840 * 2160                  # a cell rendered larger than the picture (scale > 1) stays below this
_KIT_MODULES = ("flora", "elements", "solid", "texture", "form", "figure", "creature", "layout", "clay",
                "pictograms", "palette", "backdrop", "stylize", "scenery", "vector", "diagram", "chart",
                "drafting", "ui")


# ============================================================================ small helpers

def sha(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def cache_key(src, size, k, seed, look):
    blob = json.dumps([ENGINE, src, [int(size[0]), int(size[1])], float(k), seed, look], sort_keys=True,
                      ensure_ascii=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:20]


def render_k(scale, W, H):
    """Render factor of an object cell: 1 up to scale 1, then the next half step (1.5, 2, 2.5 ...; capped by
    MAX_RENDER_PX), so a scaled-up layer is drawn at least at its size and never blown up from a small render;
    scale 1.2 and 1.4 share one render."""
    if scale <= 1.0001:
        return 1
    k = math.ceil(scale * 2 - 1e-6) / 2.0
    while k > 1 and W * H * k * k > MAX_RENDER_PX:
        k -= 0.5
    return max(1.0, k)


def _alpha_box(A, thr=0.1):
    """(x0, y0, x1, y1) of A > thr in A's own pixels, or None."""
    m = A > thr
    rows = np.flatnonzero(m.any(1))
    if not len(rows):
        return None
    cols = np.flatnonzero(m.any(0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _json_safe(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    return str(v)


class _LazyKit:
    """``flora``, ``elements`` ... in a cell without an import: the module loads on first use."""

    def __init__(self, name):
        self.__dict__["_name"] = name

    def _mod(self):
        import importlib
        return importlib.import_module("pixelcanvas." + self.__dict__["_name"])

    def __getattr__(self, attr):
        v = getattr(self._mod(), attr)
        box = self.__dict__.get("_returns")
        return _recording(v, box) if box is not None and callable(v) and not isinstance(v, type) else v

    def __repr__(self):
        return f"<pixelcanvas.{self.__dict__['_name']} (loads on first use)>"


_TAKES_NAME = {}


def _recording(fn, box):
    """fn, keeping each returned object that has ``.anchors`` (a creature, a figure) in ``box``."""
    import functools
    import inspect
    takes_name = _TAKES_NAME.get(fn)                     # "yes", "no", or "maybe" (it takes **options)
    if takes_name is None:
        try:
            ps = list(inspect.signature(fn).parameters.values())
            takes_name = ("yes" if any(p.name == "name" for p in ps) else
                          "maybe" if any(p.kind == p.VAR_KEYWORD for p in ps) else "no")
        except (TypeError, ValueError):
            takes_name = "yes"
        try:
            _TAKES_NAME[fn] = takes_name
        except TypeError:
            pass

    @functools.wraps(fn)
    def call(*a, **kw):
        if takes_name == "no":                           # name= from habit: the section name is the layer name
            kw.pop("name", None)
        if takes_name == "maybe" and "name" in kw:
            try:
                r = fn(*a, **kw)
            except (TypeError, ValueError) as e:         # its options were checked before it drew anything
                if "'name'" not in str(e) and '"name"' not in str(e):
                    raise
                kw.pop("name", None)
                r = fn(*a, **kw)
        else:
            r = fn(*a, **kw)
        if r is not None and isinstance(getattr(r, "anchors", None), dict):
            box.append(r)
        return r
    return call


# ============================================================================ cell results

class CellResult:
    """What one cell drew: premultiplied RGBA (uint8) cropped to what it touched, in render pixels; alpha 0 with
    colour = light added (a glow)."""

    def __init__(self, rgba, off, size, k, anchors=None, texts=None, entries=None, styles=None, deps=None,
                 seconds=0.0, key=None, extra=None):
        self.rgba, self.off, self.size, self.k = rgba, tuple(off), tuple(size), float(k)
        self.extra = extra                     # a whole-picture cell (cv = scene(...)): its records, kept as they are
        self.anchors = dict(anchors or {})
        self.texts = list(texts or [])
        self.entries = list(entries or [])
        self.styles = list(styles or [])
        self.deps = list(deps or [])
        self.seconds, self.key = float(seconds), key
        self.records = None                    # Sketch.paint records (medium buffers); None = nothing recorded

    def empty(self):
        return self.rgba.size == 0 or int(self.rgba[..., 3].max()) == 0

    # ---- disk
    def save(self, path):
        meta = {"off": list(self.off), "size": list(self.size), "k": self.k, "anchors": self.anchors,
                "entries": self.entries, "styles": self.styles, "deps": self.deps, "seconds": self.seconds,
                "texts": [{k: v for k, v in t.items() if k != "_mask"} for t in self.texts]}
        arrays = {"rgba": self.rgba, "meta": np.frombuffer(json.dumps(_json_safe(meta)).encode("utf-8"), np.uint8)}
        for i, t in enumerate(self.texts):
            if t.get("_mask") is not None:
                arrays[f"t{i}"] = np.asarray(t["_mask"], bool)
        if self.records is not None:
            meta["records"] = [{k: v for k, v in r.items() if k not in ("mask", "color", "normals", "hollow", "painted")}
                               for r in self.records]
            for i, r in enumerate(self.records):
                arrays[f"rm{i}"] = r["mask"]
                arrays[f"rc{i}"] = np.asarray(r["color"], np.float32)
                if r.get("normals") is not None:
                    arrays[f"rn{i}"] = r["normals"]
                if r.get("hollow") is not None:
                    arrays[f"rh{i}"] = r["hollow"]
                if r.get("painted") is not None:
                    arrays[f"rp{i}"] = r["painted"]
            arrays["meta"] = np.frombuffer(json.dumps(_json_safe(meta)).encode("utf-8"), np.uint8)
        if self.extra is not None:                       # chart / diagram drawings and records: pickled beside it
            blob = pickle.dumps(self.extra)
            with open(path[:-4] + ".pkl.tmp", "wb") as f:
                f.write(blob)
            os.replace(path[:-4] + ".pkl.tmp", path[:-4] + ".pkl")
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path, key=None):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta"]).decode("utf-8"))
            texts = meta.get("texts") or []
            for i, t in enumerate(texts):
                t["_mask"] = z[f"t{i}"] if f"t{i}" in z.files else None
            rgba = np.ascontiguousarray(z["rgba"])
            records = None
            if meta.get("records") is not None:
                records = [dict(r, mask=z[f"rm{i}"], color=z[f"rc{i}"],
                                normals=z[f"rn{i}"] if f"rn{i}" in z.files else None,
                                hollow=z[f"rh{i}"] if f"rh{i}" in z.files else None,
                                painted=z[f"rp{i}"] if f"rp{i}" in z.files else None)
                           for i, r in enumerate(meta["records"])]
        extra = None
        if os.path.isfile(path[:-4] + ".pkl"):
            with open(path[:-4] + ".pkl", "rb") as f:
                extra = pickle.load(f)
        cr = cls(rgba, meta["off"], meta["size"], meta["k"], meta.get("anchors"), texts, meta.get("entries"),
                 meta.get("styles"), meta.get("deps"), meta.get("seconds", 0.0), key, extra)
        cr.records = records
        return cr


class Cache:
    """cache/<key>.npz files; ``None`` folder = memory only."""

    def __init__(self, folder=None):
        self.folder = folder
        self.mem = {}
        if folder:
            os.makedirs(folder, exist_ok=True)

    def path(self, key):
        return os.path.join(self.folder, key + ".npz") if self.folder else None

    def get(self, key):
        if not key:
            return None
        if key in self.mem:
            return self.mem[key]
        p = self.path(key)
        if p and os.path.isfile(p):
            try:
                cr = CellResult.load(p, key)
            except Exception:                            # noqa: BLE001 - a broken cache file is re-rendered
                return None
            self.mem[key] = cr
            return cr
        return None

    def put(self, key, cr):
        cr.key = key
        self.mem[key] = cr
        p = self.path(key)
        if p:
            try:
                cr.save(p)
            except Exception:                            # noqa: BLE001 - not picklable: kept in memory only
                for q in (p, p[:-4] + ".pkl"):
                    if os.path.isfile(q):
                        os.remove(q)


# ============================================================================ running a cell

class CellError(Exception):
    """A cell raised: ``info`` is the explanation (type, message, line, likely causes)."""

    def __init__(self, info):
        super().__init__(info.get("message", ""))
        self.info = info


def signature_line(api):
    """The first line ``studio api NAME`` prints: the call written with its arguments, or None."""
    try:
        from .api import _resolve, _sig
        got = _resolve(api)
        if got:
            return _sig(*got[0])[:400]
    except Exception:                                    # noqa: BLE001
        pass
    return None


def _call_target(node, ns, own):
    """(label, callable) a Call node's function resolves to in the section namespace, else None: a namespace name
    (creature, tabletop ...), kit.function (elements.moon), sk.method, cv.method. Names the section or its
    providers bind are left alone."""
    import ast
    from ..illustrate import Sketch
    f = node.func
    if isinstance(f, ast.Name):
        if f.id in own or f.id not in ns or f.id in ("box", "anchor", "scene"):
            return None
        v = ns[f.id]
        return (f.id, v) if callable(v) and not isinstance(v, _LazyKit) else None
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
        head, a = f.value.id, f.attr
        if head in own:
            return None
        if head == "sk":
            v = getattr(Sketch, a, None)
            return (f"sk.{a}", v) if callable(v) else None
        if head == "cv":
            v = getattr(Canvas, a, None)
            return (f"cv.{a}", v) if callable(v) else None
        if head in _KIT_MODULES and isinstance(ns.get(head), _LazyKit):
            try:
                v = getattr(ns[head]._mod(), a, None)
            except Exception:                            # noqa: BLE001
                return None
            return (f"{head}.{a}", v) if callable(v) and not isinstance(v, type) else None
    return None


def check_options(src, ns, own=frozenset()):
    """Every keyword option of the section's calls that the called pixelcanvas function does not take, found before
    the section runs: [(line, label, option, did-you-mean or None, signature)]. A call whose target or options
    cannot be known statically (``**options``, a name the section binds) is left to the run."""
    import ast
    import difflib
    import inspect
    try:
        tree = ast.parse(src or "")
    except (SyntaxError, ValueError):
        return []
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.keywords:
            continue
        t = _call_target(node, ns, own)
        if t is None:
            continue
        label, fn = t
        try:
            ps = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            continue
        if any(p.kind == p.VAR_KEYWORD for p in ps.values()):
            continue
        names = [n for n, p in ps.items() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and n != "self"]
        for kw in node.keywords:
            if kw.arg is None or kw.arg in names or kw.arg == "name":
                continue
            close = difflib.get_close_matches(kw.arg, names, n=1, cutoff=0.5)
            bad.append((node.lineno, getattr(kw, "col_offset", 0), label, kw.arg, close[0] if close else None,
                        signature_line(label)))
    return [(ln, lab, opt, dym, sg) for ln, _, lab, opt, dym, sg in sorted(bad, key=lambda t: (t[0], t[1]))]


def options_error(bad, src):
    """One CellError info for every unknown option of a section."""
    parts = [f"{lab}: {opt}=" + (f" (did you mean {dym}=?)" if dym else "") for _, lab, opt, dym, _ in bad]
    line = bad[0][0]
    lines = (src or "").splitlines()
    sigs = list(dict.fromkeys(s for *_, s in bad if s))
    return {"type": "TypeError", "message": "unknown options: " + "; ".join(parts), "line": line,
            "source": lines[line - 1].strip() if 0 < line <= len(lines) else "",
            "short": (f"unknown option{'s' if len(bad) > 1 else ''} " + "; ".join(parts)
                      + f" -> studio api {bad[0][1]}")[:240],
            "api": bad[0][1], "signature": sigs[0] if sigs else None, "signatures": sigs[:3],
            "causes": [f"{lab}() takes no option {opt!r}" + (f"; did you mean {dym!r}?" if dym else "")
                       for ln, lab, opt, dym, _ in bad[:6]], "likely_cause": []}


def explain(exc, tb, path, src):
    """Explain a cell error the way render.py explains a scene error: (info dict). ``info["short"]`` is one line."""
    info = None
    try:
        R = _render_module()
        old = R.SCENE_DIR
        R.SCENE_DIR = os.path.dirname(os.path.abspath(path))
        try:
            info, text = R.failure_report(exc, tb, os.path.basename(path))
        finally:
            R.SCENE_DIR = old
        info["text"] = text
    except Exception:                                    # noqa: BLE001 - never let the explainer fail the picture
        info = {"type": type(exc).__name__, "message": str(exc)[:500], "likely_cause": []}
    line = None
    if isinstance(exc, SyntaxError) and exc.lineno:
        line = exc.lineno
    else:
        for fr in traceback.extract_tb(tb):
            if os.path.normcase(os.path.abspath(fr.filename)) == os.path.normcase(os.path.abspath(path)):
                line = fr.lineno
    info["line"] = line
    src_line = ""
    if line:
        lines = src.splitlines()
        if 0 < line <= len(lines):
            src_line = lines[line - 1].strip()
    info["source"] = src_line
    msg = " ".join(str(exc.msg if isinstance(exc, SyntaxError) and exc.msg else exc).split()) or type(exc).__name__
    msg = re.sub(r"\s*(?:Options|Valid|Choices): .*$", "", msg).rstrip(" -\u2014") or msg   # keep the did-you-mean
    causes = [c for c in info.get("likely_cause") or [] if c]
    hint = ""
    for c in causes:
        if "did you mean" in c.lower() or "Did you mean" in c:
            hint = c
            break
    short = f"{type(exc).__name__}: {msg}"
    if hint and hint.lower() not in short.lower():
        m = re.search(r"[Dd]id you mean:? ([^?]+)\?", hint)
        if m and m.group(0).lower() not in short.lower():
            short += f" - did you mean {m.group(1).strip()}?"
    api = api_name(msg)
    extra = _extra_positional(msg, src_line)
    if extra:                                            # which argument is one too many, and the call written out
        api, text = extra
        short += f" - {text}"
    m = re.search(r"'(str|int|float|tuple|list|dict|NoneType|bool)' object has no attribute", msg)
    if m and src_line:                                   # a kit call without cv: name the call on the line
        call = next((c for c in re.findall(r"([A-Za-z_][\w.]*)\(", src_line)
                     if c.split(".")[0] in _KIT_MODULES + ("creature", "tabletop", "ball", "cylinder", "cuboid",
                                                          "lathe", "Ground")), None)
        if call:
            api = call
            first = re.search(re.escape(call) + r"\(\s*([^,)]*)", src_line)
            if first and first.group(1).strip() != "cv":
                short += f" - the first argument is cv: {call}(cv, ...)"
    if not api and src_line and not isinstance(exc, SyntaxError):
        frames = traceback.extract_tb(tb)
        inside = bool(frames) and os.path.normcase(os.path.abspath(frames[-1].filename)) != \
            os.path.normcase(os.path.abspath(path))
        if inside:                                       # raised inside a call on this line: the outermost one
            for c in re.findall(r"([A-Za-z_][\w.]*)\(", src_line):
                head = c.split(".")[0]
                if head in _KIT_MODULES or head in ("sk", "cv", "creature", "tabletop", "ball", "cylinder", "cuboid",
                                                    "lathe", "Ground", "grade", "paper", "limit_palette"):
                    api = c
                    break
    if api and "." not in api and src_line:             # 'potted' -> 'flora.potted' as the line writes it
        m = re.search(r"([A-Za-z_][\w]*)\." + re.escape(api) + r"\(", src_line)
        if m and m.group(1) in _KIT_MODULES:
            api = f"{m.group(1)}.{api}"
    if api:
        short += f" -> studio api {api}"
        info["signature"] = signature_line(api)          # the call written out: no extra api turn needed
    info["api"] = api
    info["short"] = short[:240]
    info["causes"] = causes
    return info


def _extra_positional(msg, src_line):
    """('sk.paint', 'the 3rd argument (0.6) is one too many: ...') for "takes 3 positional arguments but 4 were
    given", read from the call on the line, else None."""
    import ast
    m = re.search(r"([\w.]+)\(\) takes (?:from \d+ to )?(\d+) positional arguments? but (\d+)", msg)
    if not m or not src_line:
        return None
    fname, allowed = m.group(1).split(".")[-1], int(m.group(2))
    try:
        tree = ast.parse(src_line.rstrip(":"), mode="exec")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if name != fname:
            continue
        label = ast.unparse(f)
        method = isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in ("sk", "cv")
        ok = allowed - 1 if method else allowed
        if len(node.args) <= ok:
            continue
        bad = node.args[ok]
        nth = {1: "1st", 2: "2nd", 3: "3rd"}.get(ok + 1, f"{ok + 1}th")
        return label, (f"the {nth} argument ({ast.unparse(bad)[:40]}) is one too many: {label} takes {ok} "
                       f"positional argument{'s' if ok != 1 else ''}, the rest are options written name=value")
    return None


def api_name(msg):
    """The call a wrong-argument / unknown-name error is about, as ``studio api`` takes it (sk.paint, creature)."""
    m = (re.search(r"^([\w.]+): unknown option", msg) or re.search(r"([\w.]+)\(\) got an unexpected keyword", msg)
         or re.search(r"([\w.]+)\(\) missing \d+ required", msg) or re.search(r"([\w.]+)\(\) takes ", msg)
         or re.search(r"([\w.]+)\(\) got multiple values", msg))
    if m:
        name = m.group(1)
        if name.startswith("Sketch."):
            return "sk." + name.split(".", 1)[1]
        if name.startswith("Canvas."):
            return "cv." + name.split(".", 1)[1]
        return name
    m = re.search(r"name '(\w+)' is not defined", msg)
    if m:
        return m.group(1)
    m = re.search(r"'(\w+)' object has no attribute '(\w+)'", msg)
    if m:
        return {"Sketch": "sk.", "Canvas": "cv."}.get(m.group(1), m.group(1) + ".") + m.group(2)
    m = re.search(r"^(\w+) has no '(\w+)'", msg)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    return None


_RENDER = {}


def _render_module():
    """tools/render.py of the runtime (failure_report, build_report)."""
    if "R" in _RENDER:
        return _RENDER["R"]
    from .. import cli
    root = cli.runtime_root()
    tdir = os.path.join(root, "tools") if root else None
    if tdir and tdir not in sys.path:
        sys.path.insert(0, tdir)
    import render as R                                   # noqa: E402
    _RENDER["R"] = R
    return R


def namespace(cv, look, seed, layer, resolver=None, medium=None):
    """The names a cell sees (also a reference for the docs: ``names()``)."""
    import random as _random
    from .. import illustrate
    from ..illustrate import Sketch, paper, grade, limit_palette
    from .. import color as _color
    ns = {"__name__": f"studio_cell_{layer}", "__builtins__": builtins}
    exec("from pixelcanvas import *", ns)                 # Canvas, colours, masks, filters, text helpers
    from ..creature import creature
    from ..form import tabletop, Ground, ball, cylinder, cuboid, lathe
    light = tuple(look.get("light") or (-0.6, -0.8))
    returns = []
    for name in _KIT_MODULES:
        ns[name] = _LazyKit(name)
        ns[name].__dict__["_returns"] = returns
    ns.update(cv=cv, sk=Sketch(cv, light=light), np=np, W=cv.w, H=cv.h, S=cv.h / 1080.0, seed=seed,
              rng=np.random.default_rng(seed), light=light, medium=medium, manifest=_manifest,
              illustrate=illustrate, Sketch=Sketch, paper=paper, grade=grade, limit_palette=limit_palette,
              creature=_recording(creature, returns), tabletop=_recording(tabletop, returns),
              Ground=Ground, ball=_recording(ball, returns), cylinder=_recording(cylinder, returns),
              cuboid=_recording(cuboid, returns), lathe=_recording(lathe, returns), Canvas=Canvas, rgb=_color.rgb,
              mix=_color.mix, layer_name=layer, __returns__=returns, math=math, random=_random)
    _taught(ns, returns)
    if resolver is not None:
        ns["box"] = resolver.box_fn(layer)
        ns["anchor"] = resolver.anchor_fn(layer)
        ns["scene"] = resolver.scene_fn(layer, cv)
    else:
        def _none(*a, **k):
            raise NameError("box() and anchor() read other layers: they work inside a studio picture")
        ns["box"] = ns["anchor"] = ns["scene"] = _none
    return ns


# Names the reference pages teach without a module prefix (forms.md, creatures.md, figures.md, solid.md ...): a
# section has them like tabletop and ball. They stay out of NAMESPACE, so a section that binds one of them itself
# still provides it to the sections below.
TAUGHT = {"form": ("Frame", "set_light", "get_light", "contact_shadow", "contact_shadow_px", "cast_shadow",
                   "reflection", "Room", "shelf", "counter", "row", "disc", "book", "cone", "capsule", "torus"),
          "sdf": ("sd_circle", "sd_box", "sd_ellipse", "sd_segment", "sd_polygon", "sd_capsule", "sd_rounded_box",
                  "smooth_union"),
          "creature": ("Creature",), "figure": ("person", "Figure")}


def _taught(ns, returns):
    import importlib
    for mod, names in TAUGHT.items():
        try:
            m = importlib.import_module("pixelcanvas." + mod)
        except Exception:                                # noqa: BLE001
            continue
        for n in names:
            v = getattr(m, n, None)
            if v is not None and n not in ns:
                ns[n] = _recording(v, returns) if callable(v) and not isinstance(v, type) else v


NAMESPACE = ("cv", "sk", "np", "math", "random", "W", "H", "S", "seed", "rng", "light", "medium", "box", "anchor",
             "scene", "creature",
             "tabletop", "Ground", "ball", "cylinder", "cuboid", "lathe", "manifest", "paper", "grade",
             "limit_palette", "Sketch", "Canvas", "rgb", "mix") + _KIT_MODULES


# ============================================================================ names shared between sections
# A draft is one script cut into sections: a cell sees the names the sections before it bind (a helper, a palette,
# a seeded generator, a tabletop, a creature). Each cell still runs and caches on its own: the earlier sections it
# takes names from (its providers, found by static analysis) run first in its namespace, on a scratch canvas whose
# drawing is thrown away.

_BASE = {}
_SCAN = {}


def base_names():
    """The names every cell has without a provider: the cell namespace and the builtins."""
    if "names" not in _BASE:
        ns = namespace(Canvas(4, 4), {"light": (-0.6, -0.8)}, 1, "x")
        _BASE["names"] = frozenset(set(ns) | set(dir(builtins)))
    return _BASE["names"]


def core_names():
    """The studio's own names (cv, sk, rng, the kits, box ...) and the builtins: never taken from a section."""
    if "core" not in _BASE:
        _BASE["core"] = frozenset(set(NAMESPACE) | set(dir(builtins)) | {"__name__", "__builtins__", "__returns__",
                                                                         "layer_name"})
    return _BASE["core"]


def _targets(node, out):
    """Names a binding target stores (a, (a, b), [a, *b]); attributes and subscripts store none."""
    import ast
    if isinstance(node, ast.Name):
        out.add(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for e in node.elts:
            _targets(e, out)
    elif isinstance(node, ast.Starred):
        _targets(node.value, out)
    return out


def _stores(node, out):
    """Names a statement binds in its own scope (nested functions, lambdas and comprehensions left out)."""
    import ast
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        out.add(node.name)
        return out
    if isinstance(node, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        return out
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        for a in node.names:
            if a.name != "*":
                out.add(a.asname or a.name.split(".")[0])
        return out
    if isinstance(node, ast.ExceptHandler) and node.name:
        out.add(node.name)
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
        out.add(node.id)
    for ch in ast.iter_child_nodes(node):
        _stores(ch, out)
    return out


def _params(a):
    names = {x.arg for x in list(getattr(a, "posonlyargs", [])) + a.args + a.kwonlyargs}
    for x in (a.vararg, a.kwarg):
        if x is not None:
            names.add(x.arg)
    return names


def _loads(node, out, local=frozenset()):
    """Names an expression or statement reads that are not local to a function, lambda or comprehension in it."""
    import ast
    if isinstance(node, ast.Name):
        if isinstance(node.ctx, ast.Load) and node.id not in local:
            out.add(node.id)
        return out
    if isinstance(node, ast.Lambda):
        for d in node.args.defaults + [x for x in node.args.kw_defaults if x is not None]:
            _loads(d, out, local)
        _loads(node.body, out, local | _params(node.args))
        return out
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
        loc = set(local)
        for g in node.generators:
            _loads(g.iter, out, frozenset(loc))
            _targets(g.target, loc)
            for c in g.ifs:
                _loads(c, out, frozenset(loc))
        for e in ([node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]):
            _loads(e, out, frozenset(loc))
        return out
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for d in node.decorator_list + node.args.defaults + [x for x in node.args.kw_defaults if x is not None]:
            _loads(d, out, local)
        out |= _body_free(node.body, _params(node.args)) - local
        return out
    if isinstance(node, ast.ClassDef):
        for d in node.decorator_list + node.bases + [k.value for k in node.keywords]:
            _loads(d, out, local)
        out |= _body_free(node.body, set()) - local
        return out
    if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
        if node.target.id not in local:
            out.add(node.target.id)
    for ch in ast.iter_child_nodes(node):
        _loads(ch, out, local)
    return out


def _body_free(body, params):
    """Free names of a function (or class) body: what it reads minus its parameters and its own locals."""
    import ast
    glob = set()
    loc = set(params)
    for st in body:
        _stores(st, loc)
        for n in ast.walk(st):
            if isinstance(n, (ast.Global, ast.Nonlocal)):
                glob.update(n.names)
    loc -= glob
    out = set()
    for st in body:
        _loads(st, out, frozenset(loc))
    return out


def _scan_body(stmts, bound, free, deferred):
    """Walk module-level statements in order: a name read before this source binds it is free."""
    import ast
    for st in stmts:
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            got = _loads(st, set())
            deferred |= got                              # read when called: free unless bound anywhere here
            bound.add(st.name)
        elif isinstance(st, (ast.For, ast.AsyncFor)):
            free |= _loads(st.iter, set()) - bound
            _targets(st.target, bound)
            _scan_body(st.body + st.orelse, bound, free, deferred)
        elif isinstance(st, (ast.While, ast.If)):
            free |= _loads(st.test, set()) - bound
            _scan_body(st.body + st.orelse, bound, free, deferred)
        elif isinstance(st, (ast.With, ast.AsyncWith)):
            for it in st.items:
                free |= _loads(it.context_expr, set()) - bound
                if it.optional_vars is not None:
                    _targets(it.optional_vars, bound)
            _scan_body(st.body, bound, free, deferred)
        elif isinstance(st, ast.Try) or type(st).__name__ == "TryStar":
            _scan_body(st.body, bound, free, deferred)
            for h in st.handlers:
                if h.type is not None:
                    free |= _loads(h.type, set()) - bound
                if h.name:
                    bound.add(h.name)
                _scan_body(h.body, bound, free, deferred)
            _scan_body(st.orelse + st.finalbody, bound, free, deferred)
        else:
            free |= _loads(st, set()) - bound
            _stores(st, bound)


def scan(src):
    """(names a source binds at module level, names it reads before binding them) by static analysis; a source
    that does not parse binds and reads nothing (its own run reports the syntax error)."""
    import ast
    key = sha(src or "")
    if key in _SCAN:
        return _SCAN[key]
    try:
        tree = ast.parse(src or "")
    except (SyntaxError, ValueError):
        res = (frozenset(), frozenset())
    else:
        bound, free, deferred = set(), set(), set()
        _scan_body(tree.body, bound, free, deferred)
        free |= deferred - bound
        res = (frozenset(bound), frozenset(free))
    _SCAN[key] = res
    return res


def providers(cell, order, cells):
    """The earlier sections ``cell`` takes names from, transitively: [(provider cell, [names])] in program order.
    For each name the cell reads and does not bind first (base names left out), the latest earlier section that
    binds it at module level provides it."""
    pos = {c: i for i, c in enumerate(order)}
    base = core_names()
    chosen = {}
    stack, seen = [cell], set()
    while stack:
        c = stack.pop()
        if c in seen or c not in cells:
            continue
        seen.add(c)
        i = pos.get(c, len(order))
        for name in sorted(scan(cells[c])[1] - base):
            for j in range(i - 1, -1, -1):
                p = order[j]
                if p != cell and p in cells and name in scan(cells[p])[0]:
                    chosen.setdefault(p, set()).add(name)
                    stack.append(p)
                    break
    return [(p, sorted(chosen[p])) for p in order if p in chosen]


def binder_after(cell, order, cells, name):
    """The first section after ``cell`` that binds ``name`` at module level, else None."""
    if cell not in order:
        return None
    for p in order[order.index(cell) + 1:]:
        if p in cells and name in scan(cells[p])[0]:
            return p
    return None


_DARKENING = {"multiply", "darken", "overlay", "softlight"}


_ADDITIVE = {"add", "screen", "lighten"}


class CellTimeout(BaseException):
    """Raised inside a cell that runs longer than the picture's cell_timeout (a BaseException, so a cell's own
    except Exception cannot swallow it)."""


def _run_limited(code, ns, limit):
    """exec(code, ns) (or code() when it is a function) with a watchdog: after ``limit`` seconds CellTimeout is
    raised in this thread."""
    if not callable(code):
        code, src_code = (lambda: exec(src_code, ns)), code
    if not limit or limit <= 0:
        code()
        return
    tid = threading.get_ident()
    lock = threading.Lock()
    state = {"done": False}

    def fire():
        with lock:
            if not state["done"]:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), ctypes.py_object(CellTimeout))
    timer = threading.Timer(float(limit), fire)
    timer.daemon = True
    timer.start()
    try:
        code()
    finally:
        with lock:
            state["done"] = True
        timer.cancel()


def _timeout_info(limit):
    return {"type": "CellTimeout", "line": None, "source": "", "causes": [], "api": None,
            "short": (f"ran longer than {limit:g} s and was stopped: paint once per colour (np.maximum joins "
                      f"masks) or use one kit call; or allow more: set scene cell_timeout={max(2, int(limit * 2))}")}


def _seed_all(seed):
    import random as _random
    np.random.seed(int(seed) % (2 ** 32))
    _random.seed(int(seed))


def _provided(ns, pre, cv, look, seed, layer, resolver, medium):
    """Run the provider sections ``pre`` [(cell, names, src, path)] in ``ns`` on a scratch canvas (their drawing
    is thrown away), then give the cell its own cv, sk, rng and seeds back. -> {name: id(value)} the providers
    left, so nothing they made is counted as the cell's own."""
    from ..illustrate import Sketch
    scratch = Canvas(cv.w, cv.h, (0.0, 0.0, 0.0))
    light = tuple(look.get("light") or (-0.6, -0.8))
    before = {k: id(v) for k, v in ns.items()}
    ns.update(cv=scratch, sk=Sketch(scratch, light=light), rng=np.random.default_rng(seed))
    _seed_all(seed)
    for cell, names, src, path in pre:
        try:
            code = compile(src, path, "exec")
            exec(code, ns)
        except CellTimeout:
            raise
        except Exception as e:                           # noqa: BLE001 - named with the provider
            info = explain(e, e.__traceback__, path, src)
            need = ", ".join(f"`{n}`" for n in names)
            info["short"] = (f"needs {need} from section {cell}, which fails: {info.get('short', '')}")[:240]
            info["provider"] = cell
            info["line"], info["source"] = None, ""
            raise CellError(info) from None
    fresh = namespace(cv, look, seed, layer, resolver, medium)
    core = core_names()
    for k, v in fresh.items():                           # the studio's own names back; a helper a section wrote
        if k in core or id(ns.get(k)) == before.get(k):  # over a library name stays, as in one script
            ns[k] = v
    _seed_all(seed)
    return {k: id(v) for k, v in ns.items() if fresh.get(k) is not v}


def _name_error(info, e, layer, resolver):
    """A NameError said in plain words: names are shared top to bottom between sections."""
    import difflib
    name = getattr(e, "name", None)
    if not name:
        m = re.search(r"name '(\w+)' is not defined", str(e))
        name = m.group(1) if m else None
    if not name:
        return info
    later = resolver.defined_after(layer, name) if resolver is not None and hasattr(resolver, "defined_after") \
        else None
    if later:
        short = (f"NameError: `{name}` is defined in section {later}, which comes after this one: sections run top "
                 f"to bottom - define it in an earlier section or before the first section")
        api = None
    else:
        short = (f"NameError: `{name}` is not defined in this or an earlier section: sections run top to bottom "
                 f"and share names - define it before the first section or in an earlier one")
        known = sorted(set(base_names()) - set(dir(builtins)))
        near = difflib.get_close_matches(name, known, n=1, cutoff=0.75)
        api = near[0] if near else None
        if api:
            short += f" - did you mean {api}? -> studio api {api}"
    info["short"], info["api"] = short[:300], api
    info["causes"] = []                                  # the generic import advice does not apply here
    return info


def _exec_cell(src, path, size, seed, look, layer, resolver, medium, bg, deps, limit=None, pre=None):
    """One run of a cell on a canvas filled with ``bg`` -> (canvas, namespace, normal coverage, additive
    coverage, blend modes used, {name: id} the providers left)."""
    W, H = size
    cv = Canvas(W, H, bg)
    rec = _layers.LayerRecorder(None, post=None, halos="off")
    rec.attach(cv)
    rec.current, rec.before, rec.coverage = "cell", None, None
    modes = set()
    light = {"cov": None}
    paint, paste = Canvas.paint, Canvas.paste

    def spy_paint(self, color, mask=None, mode="normal", opacity=1.0):   # blend modes used, where light was added
        if self is cv:
            from ..canvas import _mo
            try:
                m, op = _mo(mode, opacity, "Canvas.paint")
            except Exception:                            # noqa: BLE001 - the real call reports it
                m, op = "normal", 1.0
            modes.add(m)
            if m in _ADDITIVE:
                if light["cov"] is None:
                    light["cov"] = np.zeros((H, W), F32)
                a = np.ones((H, W), F32) if mask is None else np.asarray(mask, F32)
                if a.ndim == 3:
                    a = a.max(-1)
                if a.shape == (H, W):
                    np.maximum(light["cov"], np.clip(a * F32(op), 0, 1), out=light["cov"])
        return paint(self, color, mask, mode, opacity)

    def spy_paste(self, other, x=0, y=0, mask=None, mode="normal", opacity=1.0):
        if self is cv and isinstance(mode, str):
            modes.add(mode)
            if mode in _ADDITIVE:                        # kit glows add light through paste in a window
                try:
                    src = other.buf if isinstance(other, Canvas) else np.asarray(other, F32)
                    sh, sw = src.shape[:2]
                    x0, y0, x1, y1 = max(int(x), 0), max(int(y), 0), min(int(x) + sw, W), min(int(y) + sh, H)
                    if x0 < x1 and y0 < y1:
                        if light["cov"] is None:
                            light["cov"] = np.zeros((H, W), F32)
                        if mask is None:
                            a = np.full((y1 - y0, x1 - x0), 1.0, F32)
                        else:
                            a = np.asarray(mask, F32)
                            a = (a.max(-1) if a.ndim == 3 else a)[y0 - int(y):y1 - int(y), x0 - int(x):x1 - int(x)]
                        tgt = light["cov"][y0:y1, x0:x1]
                        np.maximum(tgt, np.clip(a * F32(opacity), 0, 1), out=tgt)
                except Exception:                        # noqa: BLE001 - the real call reports it
                    pass
        return paste(self, other, x, y, mask, mode, opacity)
    if resolver is not None:
        resolver.deps_stack.append(deps)
    old_step = _manifest._STATE["step"]
    Canvas.paint, Canvas.paste = spy_paint, spy_paste
    held = {"provided": {}}
    from ..medium import record as _record
    paints = _record.start(cv)                           # what Sketch.paint knows: albedo, volume, shadow, glow
    try:
        ns = namespace(cv, look, seed, layer, resolver, medium)
        code = compile(src, path, "exec")

        def run():
            if pre:
                held["provided"] = _provided(ns, pre, cv, look, seed, layer, resolver, medium)
            own = set(scan(src)[0]) | set(held["provided"]) | {n for p in (pre or []) for n in (p[1] or [])}
            bad = check_options(src, ns, frozenset(own))
            if bad:                                      # every wrong option of the section in one error
                raise CellError(options_error(bad, src))
            _manifest.step(layer)
            _seed_all(seed)
            exec(code, ns)
        try:
            _run_limited(run, ns, limit)
        except CellTimeout:
            raise CellError(_timeout_info(limit)) from None
        except CellError:
            raise
        except Exception as e:                           # noqa: BLE001
            info = explain(e, e.__traceback__, path, src)
            if isinstance(e, NameError) and not isinstance(e, UnboundLocalError):
                info = _name_error(info, e, layer, resolver)
            raise CellError(info) from None
    except SyntaxError as e:
        raise CellError(explain(e, e.__traceback__, path, src)) from None
    finally:
        Canvas.paint, Canvas.paste = paint, paste
        _record.stop(paints)
        _manifest._STATE["step"] = old_step
        if rec in _layers._ATTACHED:
            _layers._ATTACHED.remove(rec)
        if resolver is not None:
            resolver.deps_stack.pop()
    out = ns.get("cv")
    if not (isinstance(out, Canvas) and out.shape == cv.shape):
        out = cv
    mine = out is cv
    held["records"] = None if (paints.overflow or not mine) else paints.items
    return (out, ns, (rec.coverage if mine else None), (light["cov"] if mine else None), modes, held["provided"],
            held["records"])


def run_cell(src, path, size, k, seed, look, layer, resolver=None, medium=None, limit=None, pre=None):
    """Run an object cell alone on a black canvas of ``size`` x ``k`` -> CellResult (raises CellError). ``pre``:
    the earlier sections it takes names from, [(cell, names, src, path)], run first on a scratch canvas.

    The result is premultiplied RGBA: where the cell painted (normal paint) the alpha is its coverage; where it
    only added light (glows) the alpha is 0, so the light adds to whatever lies below wherever the layer goes. A
    cell that darkens what is under it (multiply shadows of solids and animals) runs once more on white: the
    pair gives the exact alpha of those shadows, so contact shadows stay with the object."""
    W, H = int(round(size[0] * k)), int(round(size[1] * k))
    t0 = time.time()
    deps = []
    out, ns, cov, lit, modes, provided, records = _exec_cell(src, path, (W, H), seed, look, layer, resolver,
                                                             medium, (0.0, 0.0, 0.0), deps, limit, pre)
    if getattr(out, "_studio_whole", False):
        return _whole(out, (W, H), k, deps, t0)
    after = np.clip(np.asarray(out.buf, F32), 0.0, 1.0)
    white = None
    if modes & _DARKENING:
        out_w = _exec_cell(src, path, (W, H), seed, look, layer, resolver, medium, (1.0, 1.0, 1.0), [], limit,
                           pre)[0]
        white = np.clip(np.asarray(out_w.buf, F32), 0.0, 1.0)
    touched = after.max(-1) > _layers.EPS
    if cov is not None:
        touched |= cov > 0
    if white is not None:
        touched |= white.min(-1) < 1.0 - _layers.EPS
    bb = _alpha_box(touched.astype(F32), 0.5)
    if bb is not None:                                   # keep the faint ends of glows and shadows
        pad = 2
        bb = (max(0, bb[0] - pad), max(0, bb[1] - pad), min(W, bb[2] + pad), min(H, bb[3] + pad))
    if bb is None:
        rgba, off = np.zeros((0, 0, 4), np.uint8), (0, 0)
    else:
        x0, y0, x1, y1 = bb
        P = after[y0:y1, x0:x1]                          # on black, the picture IS the premultiplied colour
        c = np.zeros(P.shape[:2], F32) if cov is None else np.clip(cov[y0:y1, x0:x1], 0, 1)
        lit_here = (lit[y0:y1, x0:x1] > 0) if lit is not None else np.zeros(P.shape[:2], bool)
        light_only = (c <= 0) & lit_here
        if white is None:
            a = np.maximum(c, P.max(-1))                 # unpainted change: the least alpha that explains it
            a = np.where(lit_here, c, a)                 # where light was added, only paint occludes: a glow over
            #                                              faint paint (an ember's tail) stays light, with no edge
        else:                                            # black / white pair: alpha = 1 - (white - black)
            ws = white[y0:y1, x0:x1]
            a = np.maximum(c, np.clip((1.0 - (ws - P)).max(-1), 0.0, 1.0))
            dark = np.clip((1.0 - ws).max(-1), 0.0, 1.0)            # how much it darkens white: shadow alpha
            a = np.where(light_only, dark, a)
        if white is None:
            a = np.where(light_only, 0.0, a)
        a = np.where(a > _layers.EPS, a, 0.0).astype(F32)
        rgba = np.empty(P.shape[:2] + (4,), np.uint8)
        rgba[..., :3] = np.round(np.clip(P, 0, 1) * 255.0).astype(np.uint8)
        rgba[..., 3] = np.round(a * 255.0).astype(np.uint8)
        off = (x0, y0)
    texts = []
    for t in getattr(out, "text_log", None) or []:
        r = {kk: _json_safe(v) for kk, v in t.items() if kk != "_mask"}
        r["_mask"] = None if t.get("_mask") is None else np.asarray(t["_mask"], bool)
        texts.append(r)
    entries = []
    for e in _manifest.entries(out):
        if e.get("kind") in ("text", _manifest.CUE, "style"):
            continue
        entries.append({"kind": e["kind"], "name": e["name"], "aliases": list(e.get("aliases") or []),
                        "meta": _json_safe({kk: v for kk, v in (e.get("meta") or {}).items()
                                            if kk in _manifest.META_KEYS}),
                        "bbox": _json_safe(e.get("bbox")), "area": _json_safe(e.get("area")),
                        "solid": bool((e.get("meta") or {}).get("solid3d"))})     # a 3D scene's own objects
    got = {e["name"] for e in entries}
    for nm in getattr(out, "solid_unseen", None) or []:  # added to the 3D scene, nowhere in the frame
        if nm not in got:
            entries.append({"kind": "object", "name": nm, "aliases": [], "meta": {}, "bbox": None, "area": 0.0,
                            "solid": True, "unseen": True})
    anchors = {}
    mine_ = {kk: v for kk, v in ns.items() if provided.get(kk) != id(v)}    # not what a provider section made
    objs = [mine_.get(layer)] + [v for kk, v in mine_.items() if kk != layer and not kk.startswith("_")] + \
        list(ns.get("__returns__") or [])
    for v in objs:
        an = getattr(v, "anchors", None) if v is not None and not isinstance(v, (type, _LazyKit)) else None
        if isinstance(an, dict):
            for nm, p in an.items():
                try:
                    if nm not in anchors:
                        anchors[str(nm)] = [float(p[0]) / W, float(p[1]) / H]
                except (TypeError, ValueError, IndexError):
                    continue
    cr = CellResult(rgba, off, (W, H), k, anchors, texts, entries,
                    list(getattr(out, "styles_applied", []) or []), deps, time.time() - t0)
    cr.records = records
    return cr


def _whole(out, size, k, deps, t0):
    """A cell that set ``cv`` to a whole picture (``cv = scene("chart.py")``): opaque, with the picture's own
    records (manifest entries, text, vector drawings) kept for the check."""
    W, H = size
    buf = np.clip(np.asarray(out.buf, F32), 0.0, 1.0)
    rgba = np.empty((H, W, 4), np.uint8)
    rgba[..., :3] = np.round(buf * 255.0).astype(np.uint8)
    rgba[..., 3] = 255
    extra = {"manifest": list(getattr(out, "manifest", []) or []),
             "text_log": list(getattr(out, "text_log", None) or []),
             "drawings": list(out.__dict__.get("_drawings", []) or []),
             "fit_reports": list(getattr(out, "fit_reports", None) or []),
             "styles": list(getattr(out, "styles_applied", None) or [])}
    return CellResult(rgba, (0, 0), (W, H), k, {}, [], [], extra["styles"], deps, time.time() - t0, extra=extra)


def run_pass(src, path, buf, seed, look, layer, resolver=None, medium=None, pre=None):
    """Run a pass cell on a copy of the composite ``buf`` -> (new buf, styles applied)."""
    cv = Canvas.from_array(np.array(buf, F32))
    old_step = _manifest._STATE["step"]
    try:
        ns = namespace(cv, look, seed, layer, resolver, medium)
        code = compile(src, path, "exec")
        try:
            if pre:
                _provided(ns, pre, cv, look, seed, layer, resolver, medium)
            _seed_all(seed)
            exec(code, ns)
        except CellError:
            raise
        except Exception as e:                           # noqa: BLE001
            info = explain(e, e.__traceback__, path, src)
            if isinstance(e, NameError) and not isinstance(e, UnboundLocalError):
                info = _name_error(info, e, layer, resolver)
            raise CellError(info) from None
    except SyntaxError as e:
        raise CellError(explain(e, e.__traceback__, path, src)) from None
    finally:
        _manifest._STATE["step"] = old_step
    out = ns.get("cv")
    if not (isinstance(out, Canvas) and out.shape == cv.shape):
        out = cv
    return np.asarray(out.buf, F32), list(getattr(out, "styles_applied", []) or [])


# ============================================================================ placing a layer

class Placed:
    """A layer at its place on the target canvas: premultiplied colour ``P`` (h, w, 3), alpha ``A`` (h, w),
    top-left ``(x0, y0)`` in target pixels (may lie off the canvas); ``point(q)`` maps a render pixel there."""

    def __init__(self, P, A, x0, y0, point, scale_px):
        self.P, self.A, self.x0, self.y0 = P, A, int(x0), int(y0)
        self.point = point
        self.scale_px = scale_px                         # target px per render px
        bb = (_alpha_box(A) or _alpha_box(P.max(-1))) if A.size else None         # the solid part; light if none
        self.box = None if bb is None else (self.x0 + bb[0], self.y0 + bb[1], self.x0 + bb[2], self.y0 + bb[3])

    def full_alpha(self, W, H):
        out = np.zeros((H, W), F32)
        sl = _clip(self.x0, self.y0, self.A.shape[1], self.A.shape[0], W, H)
        if sl:
            (dy, dx), (sy, sx) = sl
            out[dy, dx] = self.A[sy, sx]
        return out


def _clip(x0, y0, w, h, W, H):
    """Slices (dst, src) of a w x h patch at (x0, y0) inside a W x H canvas, or None."""
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    return ((slice(cy0, cy1), slice(cx0, cx1)), (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0)))


def native_anchor(cr):
    """The point a layer scales, rotates and flips about, in render pixels: the other layer's landmark it hangs
    from when the cell placed itself with ``anchor()`` (a lantern stays at the fox's mouth), else the bottom
    centre of what it drew (a thing stays on the ground)."""
    for dep in cr.deps or []:
        if dep and dep[0] == "anchor" and dep[2] and len(dep[2]) == 2:
            try:
                return (float(dep[2][0]) * cr.size[0], float(dep[2][1]) * cr.size[1])
            except (TypeError, ValueError):
                break
    return bottom_centre(cr)


def bottom_centre(cr):
    """The bottom centre of what the cell drew, in render pixels."""
    A = cr.rgba[..., 3]
    bb = _alpha_box(A.astype(F32) / 255.0)
    if bb is None and cr.rgba.size:
        bb = _alpha_box(cr.rgba.max(-1).astype(F32) / 255.0)
    if bb is None:
        return (cr.size[0] / 2.0, cr.size[1] / 2.0)
    x0, y0, x1, y1 = bb
    ox, oy = cr.off
    return (ox + (x0 + x1) / 2.0, oy + y1)


def place(cr, p, Wt, Ht):
    """Move / scale / rotate / flip a cell result onto a Wt x Ht canvas -> Placed, with ``pivot``: where the point
    it scales about (``native_anchor``) lands, in target pixels."""
    pl = _place(cr, p, Wt, Ht)
    qx, qy = native_anchor(cr)
    pl.pivot = pl.point(qx, qy)
    return pl


def place_patch(cr, p, Wt, Ht, rgba, off):
    """A patch drawn by the same cell (a recorded paint: rgba uint8 at render offset ``off``) moved exactly like
    the layer -> (P, A, x0, y0) in target pixels."""
    import copy
    fake = copy.copy(cr)
    fake.rgba, fake.off = rgba, tuple(off)
    pl = _place(fake, p, Wt, Ht, anchor=native_anchor(cr))
    return pl.P, pl.A, pl.x0, pl.y0


def _place(cr, p, Wt, Ht, anchor=None):
    """Move / scale / rotate / flip a cell result onto a Wt x Ht canvas (anchored at the bottom centre of what it
    drew). Premultiplied resampling (no dark fringes); an unscaled, unrotated layer is copied pixel for pixel."""
    k = cr.size[0] / float(Wt)                          # render px per target px
    s = float(p.get("scale", 1.0) or 1.0)
    th = float(p.get("rotate", 0.0) or 0.0) % 360.0
    flip = bool(p.get("flip"))
    dx, dy = float(p.get("dx", 0.0) or 0.0) * Wt, float(p.get("dy", 0.0) or 0.0) * Ht
    ax, ay = anchor if anchor is not None else native_anchor(cr)
    ax, ay = ax / k, ay / k
    if p.get("crop"):                                    # the picture shows a region enlarged f times
        cx0, cy0, f = p["crop"]
        dx, dy = ax * (f - 1) + f * (dx - cx0 * Wt), ay * (f - 1) + f * (dy - cy0 * Ht)
        s *= f
    g = s / k
    rad = math.radians(th)
    cs, sn = math.cos(rad), math.sin(rad)

    def point(qx, qy):
        tx, ty = ax * (1 - s) + g * qx, ay * (1 - s) + g * qy
        if flip:
            tx = 2 * ax - tx
        if th:
            vx, vy = tx - ax, ty - ay
            tx, ty = ax + cs * vx - sn * vy, ay + sn * vx + cs * vy
        return tx + dx, ty + dy

    if cr.rgba.size == 0:
        return Placed(np.zeros((0, 0, 3), F32), np.zeros((0, 0), F32), 0, 0, point, g)
    rgba = cr.rgba.astype(F32) / 255.0                  # premultiplied colour + alpha
    A = rgba[..., 3].copy()
    P = rgba[..., :3].copy()
    qx0, qy0 = cr.off
    h, w = A.shape
    if abs(g - 1.0) > 1e-6:
        nw, nh = max(1, int(round(w * g))), max(1, int(round(h * g)))
        method = "lanczos"
        stack = _f.resize(np.concatenate([P, A[..., None]], -1), nw, nh, method)
        A = np.clip(stack[..., 3], 0.0, 1.0)
        P = np.clip(stack[..., :3], 0.0, 1.0)
        ox, oy = ax * (1 - s) + g * qx0, ay * (1 - s) + g * qy0          # pixel edges map to pixel edges
        w, h = nw, nh
    else:
        ox, oy = ax * (1 - s) + qx0, ay * (1 - s) + qy0
    if flip:
        P, A = P[:, ::-1], A[:, ::-1]
        ox = 2 * ax - ox - w
    if not th:
        return Placed(np.ascontiguousarray(P), np.ascontiguousarray(A), int(round(ox + dx)), int(round(oy + dy)),
                      point, g)
    # rotation about the anchor: sample the (padded) patch at the inverse-rotated pixel centres
    pad = 2
    P = np.pad(P, ((pad, pad), (pad, pad), (0, 0)))
    A = np.pad(A, ((pad, pad), (pad, pad)))
    ox, oy = ox - pad, oy - pad
    w, h = w + 2 * pad, h + 2 * pad
    corners = [(ox, oy), (ox + w, oy), (ox, oy + h), (ox + w, oy + h)]
    rc = [(ax + cs * (x - ax) - sn * (y - ay), ay + sn * (x - ax) + cs * (y - ay)) for x, y in corners]
    X0 = int(math.floor(min(c[0] for c in rc)))
    Y0 = int(math.floor(min(c[1] for c in rc)))
    X1 = int(math.ceil(max(c[0] for c in rc)))
    Y1 = int(math.ceil(max(c[1] for c in rc)))
    Yg, Xg = np.mgrid[Y0:Y1, X0:X1].astype(F32)
    vx, vy = Xg + 0.5 - ax, Yg + 0.5 - ay
    sx = ax + cs * vx + sn * vy - ox - 0.5
    sy = ay - sn * vx + cs * vy - oy - 0.5
    inside = (sx > -1) & (sx < w) & (sy > -1) & (sy < h)
    st = _f.sample(np.concatenate([P, A[..., None]], -1), sx, sy)
    st = st * inside[..., None]
    A2 = np.clip(st[..., 3], 0, 1)
    P2 = np.clip(st[..., :3], 0, 1)
    return Placed(P2.astype(F32), A2.astype(F32), int(round(X0 + dx)), int(round(Y0 + dy)), point, g)


# ============================================================================ effects

def _pad_patch(A, r):
    r = int(math.ceil(r))
    return np.pad(A, ((r, r), (r, r))), r


def _oval(h, w, cx, cy, rx, ry, edge=0.35):
    Y, X = np.mgrid[0:h, 0:w].astype(F32)
    rr = np.sqrt(((X + 0.5 - cx) / max(rx, 1e-3)) ** 2 + ((Y + 0.5 - cy) / max(ry, 1e-3)) ** 2)
    t = np.clip((rr - edge) / (1.0 - edge), 0, 1)
    return (1.0 - t * t * (3 - 2 * t)).astype(F32)


def tint(P, A, color, amount=1.0):
    """Recolour towards ``color`` keeping the shading (light parts lighter, dark parts darker)."""
    from ..color import rgb
    T = np.asarray(rgb(color), F32)[:3]
    a = np.maximum(A, 1e-6)[..., None]
    C = np.clip(P / a, 0, 1)
    painted = (A > 0.02)[..., None]
    wts = np.array([0.2126, 0.7152, 0.0722], F32)
    L = C @ wts
    solid = A > 0.5
    Lref = float(np.median(L[solid])) if solid.any() else float(np.median(L[A > 0.02])) if (A > 0.02).any() else 0.5
    Lref = max(Lref, 0.03)
    tl = max(float(T @ wts), 0.03)
    ratio = (L / Lref)[..., None]
    Lnew = tl * ratio
    dark = T * ratio
    light = T + (1 - T) * np.clip((Lnew - tl) / max(1 - tl, 1e-3), 0, 1)
    new = np.where(ratio <= 1.0, dark, light)
    C2 = C + (np.clip(new, 0, 1) - C) * float(amount)
    return np.where(painted, C2 * A[..., None], P).astype(F32)            # added light keeps its colour


def effects(pl, p, Wt, Ht, light, shadow_color="#1d1830"):
    """The pieces drawn under a placed layer: [(mode, colour or premult patch, alpha patch, x0, y0)]."""
    from ..color import rgb
    out = []
    if pl.A.size == 0 or pl.box is None:
        return out
    S = Ht / 1080.0
    lx, ly = light
    sh = p.get("shadow")
    if sh:
        if sh == "ground":
            x0, y0, x1, y1 = pl.box
            A = pl.A
            solid = A > 0.5                              # the footprint: where the opaque body meets the ground,
            cols_any = np.flatnonzero(solid.any(0))      # not the box (soft fur, a kit's own shadow reach lower)
            if len(cols_any):
                low = solid.shape[0] - 1 - np.argmax(solid[::-1, :], axis=0)       # lowest opaque row per column
                rows = np.flatnonzero(solid.any(1))
                hh = max(1, rows[-1] - rows[0] + 1)
                yb = int(low[cols_any].max())
                near = cols_any[low[cols_any] >= yb - max(2, int(0.12 * hh))]
                bx0, bx1 = pl.x0 + near[0], pl.x0 + near[-1] + 1
                y1 = pl.y0 + yb + 1
            else:
                bx0, bx1 = x0, x1
            bw = max(bx1 - bx0, 0.25 * (x1 - x0))
            rx = 0.62 * bw + 2
            ry = max(0.1 * bw, 2.0, 0.006 * Ht)
            cx = (bx0 + bx1) / 2.0 - lx * 0.12 * bw
            cy = y1 - 0.25 * ry
            X0, Y0 = int(math.floor(cx - rx - 2)), int(math.floor(cy - ry - 2))
            W_, H_ = int(math.ceil(2 * rx + 4)), int(math.ceil(2 * ry + 4))
            m = _oval(H_, W_, cx - X0, cy - Y0, rx, ry) * F32(0.5 * float(p.get("shadow_strength", 1.0)))
            out.append(("normal", rgb(shadow_color), m, X0, Y0))
        else:                                            # drop shadow: offset away from the light, soft
            d = 0.012 * Ht * float(p.get("shadow_size", 1.0))
            nrm = math.hypot(lx, ly) or 1.0
            ox, oy = -lx / nrm * d * 1.4, -ly / nrm * d * 1.4
            blur = max(0.8, 0.010 * Ht * float(p.get("shadow_size", 1.0)))
            Ap, r = _pad_patch(pl.A, abs(ox) + abs(oy) + 3 * blur + 2)
            m = _f.gaussian_blur(Ap, blur) * F32(0.4 * float(p.get("shadow_strength", 1.0)))
            out.append(("normal", rgb(shadow_color), m, int(round(pl.x0 - r + ox)), int(round(pl.y0 - r + oy))))
    gl = p.get("glow")
    if gl:
        rad = max(1.0, float(p.get("glow_size", 0.03)) * Ht)
        Ap, r = _pad_patch(pl.A, 3 * rad)
        m = np.clip(_f.gaussian_blur(Ap, rad) * F32(0.8 * float(p.get("glow_strength", 1.0))), 0, 1)
        out.append(("add", rgb(gl), m, pl.x0 - r, pl.y0 - r))
    ol = p.get("outline")
    if ol:
        from ..illustrate import grow
        wpx = max(1.0, float(p.get("outline_width", 0.004)) * Ht)
        Ap, r = _pad_patch(pl.A, wpx + 3)
        m = grow(Ap, wpx)
        out.append(("normal", rgb(ol), m, pl.x0 - r, pl.y0 - r))
    return out


def body(pl, p, Ht):
    """The layer's own premultiplied colour and alpha after tint / blur / opacity."""
    P, A = pl.P, pl.A
    if A.size == 0:
        return P, A, pl.x0, pl.y0
    x0, y0 = pl.x0, pl.y0
    if p.get("tint"):
        P = tint(P, A, p["tint"], float(p.get("tint_amount", 1.0)))
    b = float(p.get("blur", 0.0) or 0.0) * Ht
    if b > 0.3:
        r = int(math.ceil(3 * b))
        P = _f.gaussian_blur(np.pad(P, ((r, r), (r, r), (0, 0))), b)
        A = _f.gaussian_blur(np.pad(A, ((r, r), (r, r))), b)
        x0, y0 = x0 - r, y0 - r
    op = float(p.get("opacity", 1.0))
    if op < 0.999:
        P, A = P * F32(op), A * F32(op)
    return P, A, x0, y0


def over(buf, P, A, x0, y0, note_cv=None):
    """Composite a premultiplied patch over ``buf`` (in place)."""
    H, W = buf.shape[:2]
    sl = _clip(x0, y0, A.shape[1], A.shape[0], W, H)
    if not sl:
        return
    (dy, dx), (sy, sx) = sl
    a = A[sy, sx][..., None]
    p = P[sy, sx]
    e = np.maximum(p - a, 0.0)                           # light beyond what paint explains: screened, never clips
    b = buf[dy, dx]
    buf[dy, dx] = (p - e) + b * (1.0 - a) + e * (1.0 - np.clip(b, 0.0, 1.0))
    if note_cv is not None:
        _layers._note(note_cv, A[sy, sx], (dx.start, dy.start, dx.stop, dy.stop))


def paint_piece(buf, mode, color, m, x0, y0, note_cv=None):
    H, W = buf.shape[:2]
    sl = _clip(x0, y0, m.shape[1], m.shape[0], W, H)
    if not sl:
        return
    (dy, dx), (sy, sx) = sl
    a = m[sy, sx][..., None]
    c = np.asarray(color, F32)[:3]
    if mode == "add":
        buf[dy, dx] = buf[dy, dx] + c * a
    else:
        buf[dy, dx] = buf[dy, dx] * (1.0 - a) + c * a
        if note_cv is not None:
            _layers._note(note_cv, m[sy, sx], (dx.start, dy.start, dx.stop, dy.stop))
