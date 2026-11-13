# -*- coding: utf-8 -*-
"""pixelcanvas.pictograms: the long tail of nameable objects as clean pictograms (masks, styles, SVG).

Quick start::

    from pixelcanvas import Canvas
    from pixelcanvas.pictograms import pictogram, find
    cv = Canvas(960, 540, "#f4efe6")
    find("sax")                                               # ['saxophone', ...]
    p = pictogram(cv, "saxophone", at=(0.5, 0.5), size=0.6, color="#1d2633", style="duotone")
    p.mask, p.bbox, p.anchors                                 # full-frame mask, pixel box, pixel anchors

Data format (one dict per pictogram; a data file is any module of this package with ``PICTOGRAMS = [...]``,
merged automatically at first use)::

    dict(name="saxophone",                     # unique, lower case, words separated by spaces
         category="music",                     # music, arts, transport, buildings, ... (free text, lower case)
         aliases=["sax", "薩克斯風"],           # other English words, Traditional Chinese for common ones
         keywords=["jazz", "brass"],           # optional: search words that are not names
         desc="alto saxophone, bell right",    # optional: one line for describe()
         body=[...],                           # the silhouette (required)
         detail=[...],                         # optional: interior marks (second tone, or cut out of a silhouette)
         accent=[...],                         # optional: a small highlight area (accent colour)
         accent_color="#f2b134",               # optional: default colour of the accent area
         anchors={"bell": (70, 82)})           # optional: named points in box units

Geometry is drawn on a 100 x 100 box, x right, y down. Each role (body / detail / accent) is a path string or a
list of path strings. A path string is SVG path data with absolute ``M L C Q A Z`` commands only, filled with the
nonzero rule (like one SVG ``<path>`` element); the strings of a role are unioned in order, and a string that
starts with ``-`` erases what the strings before it in that role painted (wheel arches, gaps). The helpers
``circle``, ``ellipse``, ``ring``, ``rect``, ``poly``, ``bar``, ``arc``, ``smooth``, ``tube``, ``frame``,
``reverse``, ``rotate_points`` and ``cut`` return such strings (or points), so a data file can write
``body=[rect(10, 40, 80, 30, r=6), cut(circle(30, 72, 12)), circle(30, 72, 9)]``.

Conventions for a consistent set: keep the shape inside 4..96, rest it on y = 90 when it stands on the ground,
strokes about 6 units thick, corners rounded 2-4 units, vehicles facing right, details inset 3+ units from the edge.
"""
from __future__ import annotations

import difflib
import importlib
import math
import pkgutil
import re

import numpy as np

from ..color import F32, as_color, rgb, luma
from .. import vector as _v
from .. import filters as _f
from .. import manifest as _manifest
from .._args import strict as _strict, check_choice, unknown_name

__all__ = ["pictogram", "FITS", "mask", "drawing", "names", "find", "describe", "categories", "sheet", "info",
           "Pictogram", "STYLES", "circle", "ellipse", "ring", "rect", "poly", "bar", "arc", "smooth", "tube", "frame",
           "reverse", "rotate_points", "cut", "reload"]

STYLES = ("silhouette", "duotone", "line", "shaded", "sticker", "cutout")
ANCHORS = ("mm", "bm", "tm", "lm", "rm")
FITS = ("height", "box")
ROLES = ("body", "detail", "accent")
_KEYS = {"name", "category", "aliases", "keywords", "desc", "body", "detail", "accent", "accent_color", "anchors"}
_CMDS = set("MLCQAZ")
DEFAULT_COLOR = "#1d2633"
DEFAULT_ACCENT = "#f2a93b"
_FRAME = "M-50 -50 L150 -50 L150 150 L-50 150 Z "          # outer frame of an even-odd clip (erasers, cut details)


# ============================================================================================ path helpers
def _n(v):
    return _v._n(float(v))


def _orient(pts):
    """Clockwise on screen (positive shoelace area with y down)."""
    P = [tuple(map(float, p[:2])) for p in pts]
    a = sum(P[i][0] * P[(i + 1) % len(P)][1] - P[(i + 1) % len(P)][0] * P[i][1] for i in range(len(P)))
    return a >= 0


def circle(cx, cy, r):
    """Circle path (clockwise)."""
    return (f"M{_n(cx - r)} {_n(cy)} A{_n(r)} {_n(r)} 0 1 1 {_n(cx + r)} {_n(cy)} "
            f"A{_n(r)} {_n(r)} 0 1 1 {_n(cx - r)} {_n(cy)} Z")


def ellipse(cx, cy, rx, ry, rotate=0.0):
    """Ellipse path (clockwise), optionally rotated ``rotate`` degrees clockwise."""
    t = math.radians(rotate)
    dx, dy = rx * math.cos(t), rx * math.sin(t)
    return (f"M{_n(cx - dx)} {_n(cy - dy)} A{_n(rx)} {_n(ry)} {_n(rotate)} 1 1 {_n(cx + dx)} {_n(cy + dy)} "
            f"A{_n(rx)} {_n(ry)} {_n(rotate)} 1 1 {_n(cx - dx)} {_n(cy - dy)} Z")


def ring(cx, cy, r, w):
    """Annulus: outer radius ``r``, band width ``w`` (the hole is part of the same path)."""
    ri = r - w
    return circle(cx, cy, r) + (f" M{_n(cx - ri)} {_n(cy)} A{_n(ri)} {_n(ri)} 0 1 0 {_n(cx + ri)} {_n(cy)} "
                                f"A{_n(ri)} {_n(ri)} 0 1 0 {_n(cx - ri)} {_n(cy)} Z")


def rect(x, y, w, h, r=0.0):
    """Rectangle with corner radius ``r`` (one number or (top-left, top-right, bottom-right, bottom-left))."""
    rs = tuple(r) if isinstance(r, (tuple, list)) else (r, r, r, r)
    m = min(w, h) / 2.0
    tl, tr, br, bl = (max(0.0, min(float(v), m)) for v in rs)
    x1, y1 = x + w, y + h
    s = [f"M{_n(x + tl)} {_n(y)}", f"L{_n(x1 - tr)} {_n(y)}"]
    if tr:
        s.append(f"A{_n(tr)} {_n(tr)} 0 0 1 {_n(x1)} {_n(y + tr)}")
    s.append(f"L{_n(x1)} {_n(y1 - br)}")
    if br:
        s.append(f"A{_n(br)} {_n(br)} 0 0 1 {_n(x1 - br)} {_n(y1)}")
    s.append(f"L{_n(x + bl)} {_n(y1)}")
    if bl:
        s.append(f"A{_n(bl)} {_n(bl)} 0 0 1 {_n(x)} {_n(y1 - bl)}")
    s.append(f"L{_n(x)} {_n(y + tl)}")
    if tl:
        s.append(f"A{_n(tl)} {_n(tl)} 0 0 1 {_n(x + tl)} {_n(y)}")
    return " ".join(s) + " Z"


def poly(points, r=0.0):
    """Closed polygon with rounded corners: ``points`` are (x, y) or (x, y, radius) (that corner's own radius)."""
    P = [(float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else float(r)) for p in points]
    if not _orient(P):
        P = P[::-1]
    n = len(P)
    out = []
    for i in range(n):
        x, y, rr = P[i]
        if rr <= 0:
            out.append((x, y, None))
            continue
        px, py = P[i - 1][:2]
        nx, ny = P[(i + 1) % n][:2]
        la, lb = math.hypot(px - x, py - y), math.hypot(nx - x, ny - y)
        t = min(rr, la / 2, lb / 2)
        if t <= 1e-6:
            out.append((x, y, None))
            continue
        a = (x + (px - x) / la * t, y + (py - y) / la * t)
        b = (x + (nx - x) / lb * t, y + (ny - y) / lb * t)
        out.append((a, (x, y), b))
    s = []
    for i, o in enumerate(out):
        if o[2] is None and not isinstance(o[0], tuple):
            s.append(("M" if i == 0 else "L") + f"{_n(o[0])} {_n(o[1])}")
        else:
            a, c, b = o
            s.append(("M" if i == 0 else "L") + f"{_n(a[0])} {_n(a[1])}")
            s.append(f"Q{_n(c[0])} {_n(c[1])} {_n(b[0])} {_n(b[1])}")
    return " ".join(s) + " Z"


def bar(x1, y1, x2, y2, w, cap="round"):
    """A straight stroke of width ``w`` from (x1, y1) to (x2, y2) as a filled path; ``cap`` round or butt."""
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy) or 1e-9
    hw = w / 2.0
    nx, ny = -dy / L * hw, dx / L * hw
    a1, a2 = (x1 + nx, y1 + ny), (x2 + nx, y2 + ny)
    b2, b1 = (x2 - nx, y2 - ny), (x1 - nx, y1 - ny)
    if not _orient([a1, a2, b2, b1]):
        a1, b1, a2, b2 = b1, a1, b2, a2
    if cap == "round":
        return (f"M{_n(a1[0])} {_n(a1[1])} L{_n(a2[0])} {_n(a2[1])} A{_n(hw)} {_n(hw)} 0 0 1 {_n(b2[0])} {_n(b2[1])} "
                f"L{_n(b1[0])} {_n(b1[1])} A{_n(hw)} {_n(hw)} 0 0 1 {_n(a1[0])} {_n(a1[1])} Z")
    return (f"M{_n(a1[0])} {_n(a1[1])} L{_n(a2[0])} {_n(a2[1])} L{_n(b2[0])} {_n(b2[1])} "
            f"L{_n(b1[0])} {_n(b1[1])} Z")


def arc(cx, cy, r, a0, a1, w, cap="round"):
    """A thick circular arc (centre line radius ``r``, width ``w``) from angle ``a0`` to ``a1`` degrees, clockwise
    on screen (0 = right, 90 = down)."""
    if a1 < a0:
        a0, a1 = a1, a0
    if a1 - a0 >= 359.99:
        return ring(cx, cy, r + w / 2.0, w)
    ro, ri, hw = r + w / 2.0, r - w / 2.0, w / 2.0
    big = 1 if a1 - a0 > 180 else 0

    def pt(rr, a):
        t = math.radians(a)
        return cx + rr * math.cos(t), cy + rr * math.sin(t)
    o0, o1, i1, i0 = pt(ro, a0), pt(ro, a1), pt(ri, a1), pt(ri, a0)
    if cap == "round":
        c1 = f"A{_n(hw)} {_n(hw)} 0 0 1 {_n(i1[0])} {_n(i1[1])}"
        c0 = f"A{_n(hw)} {_n(hw)} 0 0 1 {_n(o0[0])} {_n(o0[1])}"
    else:
        c1, c0 = f"L{_n(i1[0])} {_n(i1[1])}", f"L{_n(o0[0])} {_n(o0[1])}"
    return (f"M{_n(o0[0])} {_n(o0[1])} A{_n(ro)} {_n(ro)} 0 {big} 1 {_n(o1[0])} {_n(o1[1])} {c1} "
            f"A{_n(ri)} {_n(ri)} 0 {big} 0 {_n(i0[0])} {_n(i0[1])} {c0} Z")


def smooth(points, closed=True, tension=0.5):
    """Smooth closed outline through ``points`` (Catmull-Rom as cubic ``C`` segments)."""
    P = [tuple(map(float, p)) for p in points]
    if closed and not _orient(P):
        P = P[::-1]
    return _v._catmull(P, closed, tension).to_d().replace("M", "M ").strip()


def tube(points, widths, tension=0.5):
    """A smooth band of varying width along a centre line (instrument tubes, necks, handles, feathers):
    ``points`` (x, y) of the centre line, ``widths`` one number or one per point; the ends are cut square."""
    P = np.asarray([tuple(map(float, p)) for p in points], np.float64)
    n = len(P)
    W_ = np.full(n, float(widths)) if np.isscalar(widths) else np.asarray(widths, np.float64)
    T = np.gradient(P, axis=0)
    T /= np.maximum(np.hypot(T[:, 0], T[:, 1]), 1e-9)[:, None]
    N = np.stack([-T[:, 1], T[:, 0]], -1)
    L = P + N * (W_ / 2)[:, None]
    R = P - N * (W_ / 2)[:, None]
    if not _orient(list(map(tuple, np.concatenate([L, R[::-1]])))):
        L, R = R, L
    a = _v._catmull([tuple(p) for p in L], False, tension).to_d()
    b = _v._catmull([tuple(p) for p in R[::-1]], False, tension).to_d()
    return a + " L" + b[1:] + " Z"


def reverse(d):
    """The same path traced backwards: added to another path string it becomes a hole in that string."""
    p = _v.Path(d)
    out = []
    for start, segs, closed in p.subs:
        pts = [start]
        for sg in segs:
            pts.append((sg[-2], sg[-1]))
        s = [f"M{_n(pts[-1][0])} {_n(pts[-1][1])}"]
        for i in range(len(segs) - 1, -1, -1):
            sg = segs[i]
            x, y = pts[i]
            if sg[0] == "L":
                s.append(f"L{_n(x)} {_n(y)}")
            else:
                s.append(f"C{_n(sg[3])} {_n(sg[4])} {_n(sg[1])} {_n(sg[2])} {_n(x)} {_n(y)}")
        out.append(" ".join(s) + (" Z" if closed else ""))
    return " ".join(out)


def frame(x, y, w, h, t, r=0.0):
    """A rectangular frame (border of width ``t``) as one path: outer rectangle with an inner hole."""
    ri = max(0.0, (max(r) if isinstance(r, (tuple, list)) else r) - t)
    return rect(x, y, w, h, r) + " " + reverse(rect(x + t, y + t, w - 2 * t, h - 2 * t, ri))


def rotate_points(points, angle, cx=50.0, cy=50.0):
    """Rotate (x, y[, r]) points by ``angle`` degrees clockwise about (cx, cy)."""
    t = math.radians(angle)
    c, s = math.cos(t), math.sin(t)
    return [(cx + (p[0] - cx) * c - (p[1] - cy) * s, cy + (p[0] - cx) * s + (p[1] - cy) * c) + tuple(p[2:])
            for p in points]


def cut(d):
    """Mark a path string as an eraser: it removes what the earlier strings of its role painted."""
    return "-" + d


# ============================================================================================ registry
_REG = None
_ORDER = []
_PATHS = {}


def _check_d(name, role, d):
    body = d[1:] if d.startswith("-") else d
    bad = set(re.findall(r"[A-Za-z]", body)) - _CMDS - {"e", "E"}
    if bad:
        raise ValueError(f"pictogram '{name}': {role} path uses {sorted(bad)}; only absolute M L C Q A Z")
    return d


def _norm_record(rec, src):
    if not isinstance(rec, dict):
        raise TypeError(f"{src}: each entry of PICTOGRAMS must be a dict")
    extra = set(rec) - _KEYS
    if extra:
        bad = sorted(extra)[0]
        close = difflib.get_close_matches(bad, sorted(_KEYS), n=1)
        raise ValueError(f"{src}: pictogram '{rec.get('name')}' has unknown key '{bad}'"
                         + (f" — did you mean '{close[0]}'?" if close else ""))
    name = str(rec.get("name", "")).strip().lower()
    if not name:
        raise ValueError(f"{src}: pictogram without a name")
    if not rec.get("body"):
        raise ValueError(f"{src}: pictogram '{name}' has no body")
    out = {"name": name, "category": str(rec.get("category", "misc")).strip().lower(),
           "aliases": [str(a) for a in rec.get("aliases", ())], "keywords": [str(k) for k in rec.get("keywords", ())],
           "desc": str(rec.get("desc", "")), "accent_color": rec.get("accent_color"),
           "anchors": {k: (float(v[0]), float(v[1])) for k, v in dict(rec.get("anchors") or {}).items()},
           "module": src}
    for role in ROLES:
        v = rec.get(role) or []
        v = [v] if isinstance(v, str) else list(v)
        out[role] = [_check_d(name, role, str(d).strip()) for d in v if str(d).strip()]
    return out


def _load():
    global _REG, _ORDER
    if _REG is not None:
        return _REG
    reg, order = {}, []
    for mi in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
        if mi.name.startswith("_"):
            continue
        mod = importlib.import_module(f"{__name__}.{mi.name}")
        for rec in getattr(mod, "PICTOGRAMS", None) or []:
            r = _norm_record(rec, mi.name)
            if r["name"] in reg:
                raise ValueError(f"pictogram '{r['name']}' is defined twice ({reg[r['name']]['module']}, {mi.name})")
            reg[r["name"]] = r
            order.append(r["name"])
    _REG, _ORDER = reg, order
    return reg


def reload():
    """Forget the loaded data (the next call re-reads every data file)."""
    global _REG
    _REG = None
    _PATHS.clear()


def _get(name):
    reg = _load()
    key = str(name).strip().lower().replace("_", " ")
    if key in reg:
        return reg[key]
    for r in reg.values():                                   # an exact alias also works
        if key in (a.lower() for a in r["aliases"]):
            return r
    hits = find(key, n=3)
    hint = ("Try find(%r): %s" % (key, ", ".join(hits))) if hits else "Try names() or find('...')"
    raise unknown_name("pictogram", name, list(reg), hint)


def _canonical(name):
    try:
        return _get(name)["name"]
    except Exception:                                    # noqa: BLE001
        return str(name)


def names(category=None):
    """Pictogram names in data order, or those of one ``category``."""
    reg = _load()
    if category is None:
        return list(_ORDER)
    cat = str(category).lower()
    if cat not in categories():
        raise unknown_name("category", category, categories())
    return [n for n in _ORDER if reg[n]["category"] == cat]


def categories():
    """Category names in data order."""
    reg = _load()
    return list(dict.fromkeys(reg[n]["category"] for n in _ORDER))


def info(name):
    """The normalised data record of a pictogram (dict: name, category, aliases, keywords, body, detail ...)."""
    return dict(_get(name))


def describe(name):
    """One line: name (category) — description; aliases."""
    r = _get(name)
    s = f"{r['name']} ({r['category']} pictogram)"
    if r["desc"]:
        s += f": {r['desc']}"
    if r["aliases"]:
        s += "; also: " + ", ".join(r["aliases"])
    return s


def _tok(s):
    return [t for t in re.findall(r"[a-z0-9]+|[^\x00-\x7f]+", str(s).lower()) if t]


def _stem(w):
    """A crude singular form: cherries -> cherry, boxes -> box, boats -> boat (non-ASCII words are left alone)."""
    if not w.isascii() or len(w) < 4:
        return w
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "sses", "xes", "zes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


# Words a request uses for a whole group of pictograms: each trigger word lists the names it stands for, best first.
_GROUPS = [
    (("weather", "forecast", "climate", "天氣"),
     ["sun behind cloud", "moon behind cloud", "rain cloud", "storm cloud", "snow cloud", "windy", "fog", "sun", "cloud",
      "lightning", "rainbow", "snowflake", "umbrella", "thermometer"]),
    (("task", "todo", "to-do", "chore", "errand", "agenda"),
     ["checklist", "kanban board", "clipboard", "notes", "calendar"]),
    (("veg", "veggie", "vegetable", "produce", "greens", "蔬菜"),
     ["eggplant", "broccoli", "onion", "lettuce", "cabbage", "radish", "garlic", "potato", "cucumber", "peas", "pumpkin",
      "chili pepper", "cauliflower", "sweet potato", "turnip", "carrot", "tomato", "pepper", "corn", "mushroom"]),
    (("fruit", "水果"),
     ["apple", "banana", "orange", "lemon", "strawberry", "cherries", "grapes", "watermelon slice", "pear", "pineapple",
      "avocado", "peach", "mango", "kiwi", "coconut", "blueberries"]),
    (("boat", "ship", "vessel", "watercraft", "sailing", "船"),
     ["sailboat", "rowing boat", "canoe", "motorboat", "fishing boat", "ferry", "cargo ship", "cruise ship", "submarine"]),
    (("cookware", "utensil", "kitchenware", "cooking"),
     ["pot", "frying pan", "ladle", "spatula", "whisk", "rolling pin", "colander", "kettle", "teapot", "cutting board", "knife"]),
    (("garden", "gardening", "yard", "花園", "庭園"),
     ["stone lantern", "bonsai", "watering can", "shovel", "wheelbarrow", "flower pot", "birdhouse", "bamboo", "lotus",
      "koi", "arched bridge", "bench", "torii gate"]),
    (("camping", "camp", "campsite", "outdoors", "露營"),
     ["tent", "campfire", "sleeping bag", "lantern", "picnic basket", "backpack", "hiking"]),
    (("biology", "microscopic", "organelle", "生物"),
     ["cell", "nucleus", "mitochondrion", "chloroplast", "dna helix", "leaf cross-section", "bacterium", "virus", "neuron",
      "seedling", "petri dish", "microscope", "plant cell"]),
    (("furniture", "interior", "家具"),
     ["armchair", "sofa", "chair", "table", "desk", "bed", "wardrobe", "bookshelf", "shelf", "dresser", "stool", "side table",
      "coat rack", "floor lamp", "desk lamp", "round rug"]),
    (("app", "apps", "productivity", "ui", "interface"),
     ["notes", "checklist", "chat bubble", "sun behind cloud", "calculator", "gear", "folder", "map", "camera", "music note",
      "envelope", "calendar", "magnifier", "bookmark", "user avatar", "photo", "sliders", "toggle switch"]),
    (("street", "road", "city", "urban", "街道"),
     ["bus stop shelter", "traffic light", "street lamp", "bench", "traffic cone", "fire hydrant", "pedestrian crossing",
      "billboard", "road sign", "bus stop sign", "roadworks barrier"]),
    (("fantasy", "storybook", "fairy", "fairytale", "magic", "奇幻", "魔法"),
     ["wizard tower", "castle", "dragon", "magic wand", "spell book", "crystal ball", "potion bottle", "cauldron",
      "knight helmet", "unicorn", "witch hat"]),
    (("gadget", "gadgets", "electronics", "device", "devices", "wearable"),
     ["smartwatch", "earbuds", "charging case", "phone", "laptop", "tablet", "drone", "game controller", "vr headset",
      "smart speaker", "power bank", "router", "usb drive"]),
    (("hygiene", "cleaning", "toiletries"),
     ["soap", "hand sanitizer", "toothbrush", "face mask", "bandage"]),
]


def find(query, n=5):
    """Best pictogram names for ``query``: names, aliases (English, Traditional Chinese), head nouns and parts of
    multi-word names, keywords, group words (weather, tasks, veg, boat, ...) and typos. Names that are not related
    to the query are never returned."""
    reg = _load()
    q = str(query).strip().lower().replace("_", " ")
    qt = _tok(q)
    if not q or not qt:
        return []
    qs = [_stem(t) for t in qt]
    qj = q.replace(" ", "").replace("-", "")
    scores = {}

    def bump(nm, v):
        scores[nm] = max(scores.get(nm, 0.0), v)

    for nm in _ORDER:
        r = reg[nm]
        terms = [nm] + [a.lower() for a in r["aliases"]]
        s = 0.0
        if q in terms:
            s = 300.0 if q == nm else 250.0
        else:
            joined = [t.replace(" ", "").replace("-", "") for t in terms]
            if len(qj) >= 4 and qj in joined:                  # "light house" -> lighthouse
                s = 200.0
            for t in terms:
                if q.isascii():
                    hit = len(q) >= 3 and (re.search(r"\b" + re.escape(q), t) is not None
                                           or (len(t) >= 3 and f" {t} " in f" {q} "))
                else:
                    hit = len(q) >= 2 and (q in t or (len(t) >= 2 and t in q))
                if hit:
                    s = max(s, 70.0 - abs(len(t) - len(q)) * 0.5)
            words, wstem = set(), set()
            for t in terms:
                for w in _tok(t):
                    words.add(w)
                    wstem.add(_stem(w))
            kw = set(_stem(w) for w in _tok(" ".join(r["keywords"])))
            own = set(_stem(w) for w in _tok(nm))
            head = _stem(_tok(nm)[-1]) if _tok(nm) else ""
            for w, ws in zip(qt, qs):
                if ws in wstem or w in words:
                    s += 38.0 if ws in own else 30.0
                    if len(qs) > 1 and ws == head and ws == qs[-1]:
                        s += 8.0                               # the head noun of the request names the thing
                elif ws in kw:
                    s += 14.0
                elif w == r["category"]:
                    s += 6.0
                elif len(ws) >= 4 and any(ws in x for x in own):
                    s += 24.0                                  # a part of a compound: sail -> sailboat
        if s > 0:
            bump(nm, s)
    for trig, members in _GROUPS:                              # group words stand for a list of pictograms
        if q in trig or any(t in trig or s_ in trig for t, s_ in zip(qt, qs)):
            for i, nm in enumerate(members):
                if nm in reg:
                    bump(nm, 62.0 - i * 0.4)
    if not scores:                                             # typos: only when nothing else matched
        for nm in _ORDER:
            r = reg[nm]
            terms = [nm] + [a.lower() for a in r["aliases"] if a.isascii()]
            best = max((difflib.SequenceMatcher(None, q, t).ratio() for t in terms if t[:1] == q[:1]), default=0.0)
            if len(q) >= 4 and best >= 0.8:
                bump(nm, best * 40.0)
                continue
            words = {w for t in terms for w in _tok(t) if len(w) >= 4}
            for w in qt:
                if len(w) >= 4 and any(x[:1] == w[:1] for x in words) and \
                        difflib.get_close_matches(w, list(words), n=1, cutoff=0.84):
                    bump(nm, 20.0)
    top = max(scores.values(), default=0.0)
    ranked = sorted(((k, v) for k, v in scores.items() if v >= min(0.4 * top, 34.0)), key=lambda kv: -kv[1])
    return [k for k, _ in ranked][:max(int(n), 0)]


# ============================================================================================ rasterising
def _path(d):
    p = _PATHS.get(d)
    if p is None:
        p = _PATHS[d] = _v.Path(d)
    return p


def _role_bbox(rec):
    """Box-unit bounding box of every painted (non-eraser) string."""
    key = ("bbox", rec["name"])
    if key in _PATHS:
        return _PATHS[key]
    xs0, ys0, xs1, ys1 = [], [], [], []
    for role in ROLES:
        for d in rec[role]:
            if d.startswith("-"):
                continue
            b = _path(d).bbox()
            xs0.append(b[0])
            ys0.append(b[1])
            xs1.append(b[2])
            ys1.append(b[3])
    b = (min(xs0), min(ys0), max(xs1), max(ys1))
    _PATHS[key] = b
    return b


def _raster_role(strings, M, w, h):
    """Coverage (h, w) of one role under the box→pixel matrix ``M``."""
    acc = np.zeros((h, w), F32)
    if not strings:
        return acc
    s = _v._mscale(M)
    tol = 0.12 / max(s, 1e-9)
    jobs, signs = [], []
    for d in strings:
        neg = d.startswith("-")
        p = _path(d[1:].strip() if neg else d)
        polys = [_v._apply(M, q)[None] for q, _ in p.flatten(tol) if len(q) >= 3]
        jobs.append((polys, False))
        signs.append(neg)
    res = _v._raster(jobs, (0, 0, w, h))
    for neg, r in zip(signs, res):
        if r is None:
            continue
        x0, y0, a = r
        sl = (slice(y0, y0 + a.shape[0]), slice(x0, x0 + a.shape[1]))
        if neg:
            acc[sl] *= 1.0 - a
        else:
            np.maximum(acc[sl], a, out=acc[sl])
    return acc


def _matrix(scale, rotate=0.0, flip=False, center=(50.0, 50.0)):
    """Box units → local units: flip about the box centre, rotate (degrees clockwise), scale."""
    m = _v._I
    if flip:
        m = _v._mul(_v._mul(_v._translate(center[0], 0), _v._scale(-1, 1)), _v._translate(-center[0], 0))
    if rotate:
        m = _v._mul(_v._rotate(rotate, *center), m)
    return _v._mul(_v._scale(scale), m)


def _transformed_bbox(rec, M):
    b = _role_bbox(rec)
    C = np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])
    P = _v._apply(M, C)
    return P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max()


def _layout(rec, h_px, at, size, rotate, flip, anchor, W, H, fit_box=False):
    """Matrix box units → canvas px, and the shape's px bbox (``fit_box``: ``size`` is a (w, h) box in canvas
    fractions the shape is scaled to fit, and ``anchor`` mm centres the shape itself)."""
    if fit_box:
        bw, bh = size
        M1 = _matrix(1.0, rotate, flip)
        a0, b0, a1, b1 = _transformed_bbox(rec, M1)
        k = min(bw * W / max(a1 - a0, 1e-9), bh * H / max(b1 - b0, 1e-9))
    else:
        k = size * h_px / 100.0
    M0 = _matrix(k, rotate, flip)
    x0, y0, x1, y1 = _transformed_bbox(rec, M0)
    ax, ay = at[0] * W, at[1] * H
    c = _v._apply(M0, np.array([[50.0, 50.0]]))[0]
    if fit_box:
        c = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
    if anchor == "mm":
        tx, ty = ax - c[0], ay - c[1]
    elif anchor == "bm":
        tx, ty = ax - c[0], ay - y1
    elif anchor == "tm":
        tx, ty = ax - c[0], ay - y0
    elif anchor == "lm":
        tx, ty = ax - x0, ay - c[1]
    else:
        tx, ty = ax - x1, ay - c[1]
    M = _v._mul(_v._translate(tx, ty), M0)
    return M, (x0 + tx, y0 + ty, x1 + tx, y1 + ty), k


# ============================================================================================ distance field
def _jfa(seed, band):
    """Distance (px) from every pixel to the nearest True pixel of ``seed`` (jump flooding), exact up to about
    ``band`` px; farther pixels get a large value."""
    h, w = seed.shape
    BIG = np.float32(1e6)
    if not seed.any():
        return np.full((h, w), BIG, F32)
    Ys, Xs = np.mgrid[0:h, 0:w].astype(F32)
    sx = np.where(seed, Xs, BIG).astype(F32)
    sy = np.where(seed, Ys, BIG).astype(F32)
    step = 1
    while step * 2 <= max(2, min(band, max(h, w))):
        step *= 2
    steps = []
    while step >= 1:
        steps.append(step)
        step //= 2
    steps.append(1)
    best = (Xs - sx) ** 2 + (Ys - sy) ** 2
    for st in steps:
        px = np.pad(sx, st, constant_values=BIG)
        py = np.pad(sy, st, constant_values=BIG)
        for dy in (-st, 0, st):
            for dx in (-st, 0, st):
                if dx == 0 and dy == 0:
                    continue
                tx = px[st + dy:st + dy + h, st + dx:st + dx + w]
                ty = py[st + dy:st + dy + h, st + dx:st + dx + w]
                dd = (Xs - tx) ** 2 + (Ys - ty) ** 2
                better = dd < best
                if better.any():
                    np.copyto(sx, tx, where=better)
                    np.copyto(sy, ty, where=better)
                    np.copyto(best, dd, where=better)
    return np.sqrt(best).astype(F32)


def _sdist(cov, band=64.0):
    """Signed distance (px, negative inside) of a coverage mask, sub-pixel exact along the anti-aliased edge and
    exact to about ``band`` px from it."""
    inside = cov >= 0.5
    band = int(math.ceil(band)) + 2
    d_in = _jfa(inside, band)                 # outside pixels: distance to the shape
    d_out = _jfa(~inside, band)               # inside pixels: distance to the outside
    d = np.where(inside, 0.5 - d_out, d_in - 0.5).astype(F32)
    edge = (cov > 0.002) & (cov < 0.998)
    d = np.where(edge & (np.abs(d) < 1.0), 0.5 - cov, d)
    return np.clip(d, -band, band).astype(F32)


def _fill(d, aa=1.0):
    return np.clip(0.5 - d / aa, 0, 1).astype(F32)


# ============================================================================================ the result
class Pictogram:
    """A drawn pictogram: ``mask`` (full frame), ``alpha`` + ``box`` (its crop), ``bbox`` (px, the shape only),
    ``anchors`` (px points: center, top, bottom, left, right + the data's named anchors), ``uv(key)``."""

    def __init__(self, cv, name, box, alpha, bbox, anchors, style):
        self.cv, self.name, self.box, self.alpha = cv, name, box, alpha
        self.bbox = tuple(float(v) for v in bbox)
        self.anchors = anchors
        self.style = style

    @property
    def mask(self):
        H, W = self.cv.h, self.cv.w
        out = np.zeros((H, W), F32)
        x0, y0, x1, y1 = self.box
        out[y0:y1, x0:x1] = self.alpha
        return out

    def __getitem__(self, key):
        return self.anchors[key]

    def uv(self, key):
        """Anchor as canvas fractions."""
        x, y = self.anchors[key]
        return x / self.cv.w, y / self.cv.h

    def __repr__(self):
        return f"<Pictogram {self.name} {self.style} bbox={tuple(round(v) for v in self.bbox)}>"


def _auto_detail(c):
    c = rgb(c)
    if float(luma(c)) < 0.55:
        return np.clip(c + (1.0 - c) * 0.62, 0, 1).astype(F32)
    return np.clip(c * 0.42, 0, 1).astype(F32)


def _col(c, shape):
    a = as_color(c)
    if a.ndim == 1:
        return np.broadcast_to(a.astype(F32), shape + (3,))
    return a


def _paste(cv, color, alpha, x0, y0, mode="normal"):
    if alpha is None or not np.any(alpha > 0):
        return
    cv.paste(np.ascontiguousarray(_col(color, alpha.shape)), x0, y0, mask=alpha, mode=mode)


def _light():
    try:
        from .. import form
        return form.get_light()
    except Exception:                                    # noqa: BLE001
        return None


def _shadow_dir():
    L = _light()
    if L is None:
        return 0.5, 0.85
    dx, dy = L.dir2
    return -dx, max(-dy, 0.35)


# ============================================================================================ drawing
def pictogram(cv, shape=None, at=(0.5, 0.5), size=0.3, color=DEFAULT_COLOR, style="silhouette", detail_color=None,
              accent=None, rotate=0.0, flip=False, shadow=False, anchor="mm", line_width=None, light=None,
              material="matte", log=None, fit="height", name=None, alias=None):
    """Paint pictogram ``shape`` and return a ``Pictogram`` (mask, bbox, anchors).

    name / alias: what the scene report calls it. A stand-in records under the request's word:
    ``pictogram(cv, "sailboat", name="boat")`` records "boat" (alias sailboat). ``pictogram(cv, name="sailboat")``
    with no shape still means the shape.

    at: (u, v) canvas fractions of the anchor point; size: height of the 100-unit box as a fraction of the canvas
    height; fit: "height" (the default, ``size`` as just described) or "box" (``size`` is a (w, h) box in canvas
    fractions, or one number for a square of that fraction of the canvas height; the shape is scaled to fill the
    box and centred on ``at``, so a wide pictogram takes the box's width); color: body colour (cutout: what shows through the hole); style: silhouette (one colour, details cut
    out), duotone (details in ``detail_color``), line (outlines at one weight, ``line_width`` box units, default
    4.5), shaded (a soft lit volume, ``light`` / ``material`` as in pixelcanvas.form), sticker (white die-cut
    border and soft shadow), cutout (the shape punched out of what is below, showing ``color``);
    accent: colour of the accent area (default: the pictogram's own, or ``color`` for a silhouette); rotate:
    degrees clockwise; flip: mirror left-right; shadow: soft drop shadow; anchor: mm (centre of the box), bm
    (bottom of the shape: stands on ``at``), tm, lm, rm; log: called with the name before painting."""
    if shape is None:                                    # old spelling: pictogram(cv, name="sailboat")
        shape = name
    rec = _get(shape)
    check_choice("pictogram", "style", style, STYLES)
    check_choice("pictogram", "anchor", anchor, ANCHORS)
    check_choice("pictogram", "fit", fit, FITS)
    if log:
        log(rec["name"])
    H, W = cv.h, cv.w
    try:
        u, v = float(at[0]), float(at[1])
    except (TypeError, IndexError, ValueError):
        raise ValueError(f"pictogram: at must be (u, v) canvas fractions, got {at!r}")
    if fit == "box":
        try:
            bw, bh = (float(size[0]), float(size[1])) if hasattr(size, "__len__") else (float(size) * H / W, float(size))
        except (TypeError, IndexError, ValueError):
            raise ValueError(f"pictogram: with fit='box' size must be (w, h) canvas fractions or one number, got {size!r}")
        M, bb, k = _layout(rec, H, (u, v), (bw, bh), float(rotate), bool(flip), anchor, W, H, True)
    else:
        M, bb, k = _layout(rec, H, (u, v), float(size), float(rotate), bool(flip), anchor, W, H)
    lw = (4.5 if line_width is None else float(line_width)) * k
    pad = 3.0 + (lw if style == "line" else 0.0)
    if style == "sticker":
        pad += 7.0 * k + 6.0 * k
    if shadow or style in ("sticker", "cutout", "shaded"):
        pad += 8.0 * k
    x0 = int(max(0, math.floor(bb[0] - pad)))
    y0 = int(max(0, math.floor(bb[1] - pad)))
    x1 = int(min(W, math.ceil(bb[2] + pad)))
    y1 = int(min(H, math.ceil(bb[3] + pad)))
    anchors = _anchors(rec, M, bb)
    if x1 <= x0 or y1 <= y0:
        return Pictogram(cv, rec["name"], (0, 0, 0, 0), np.zeros((0, 0), F32), bb, anchors, style)
    w, h = x1 - x0, y1 - y0
    Mc = _v._mul(_v._translate(-x0, -y0), M)
    body = _raster_role(rec["body"], Mc, w, h)
    det = _raster_role(rec["detail"], Mc, w, h)
    acc = _raster_role(rec["accent"], Mc, w, h)
    sil = np.maximum(body, acc)
    det_col = detail_color if detail_color is not None else _auto_detail(color)
    acc_col = accent if accent is not None else (rec["accent_color"] or DEFAULT_ACCENT)
    if isinstance(acc_col, str) and acc_col == "detail":     # steam, glints: the accent area takes the detail colour
        acc_col = det_col
    alpha = sil
    if style in ("line", "sticker", "shaded"):
        d = _sdist(sil, {"line": 2 * lw, "sticker": 8.0 * k, "shaded": 10.0 * k}[style])
    if shadow and style not in ("sticker", "cutout"):
        _drop_shadow(cv, sil, x0, y0, k)
    if style == "silhouette":
        alpha = sil * (1.0 - det)
        _paste(cv, color, alpha, x0, y0)
        if accent is not None:
            _paste(cv, accent, acc * (1.0 - det), x0, y0)
    elif style == "duotone":
        _paste(cv, color, body, x0, y0)
        _paste(cv, det_col, det * sil, x0, y0)
        _paste(cv, acc_col, acc, x0, y0)
    elif style == "line":
        dd = _sdist(det, lw) if det.any() else None
        ring_ = sil * (1.0 - _fill(d + lw))                        # an inside stroke: thin parts stay solid
        if dd is not None:
            ring_ = np.maximum(ring_, _fill(np.abs(dd) - lw / 2.0) * sil)
        if accent is not None and acc.any():
            _paste(cv, accent, acc * _fill(d + lw / 2.0) * (1 - det), x0, y0)
        _paste(cv, color, ring_, x0, y0)
        alpha = np.maximum(ring_, acc if accent is not None else 0 * acc)
    elif style == "shaded":
        # a small illustration: details and accents are lit with the body, a sky-to-ground tone runs down the
        # volume, the form is rounded over a wide band, and a soft shadow sits under it
        dc = rgb(det_col)
        base = np.empty((h, w, 3), F32)
        base[:] = rgb(color)
        base = base * (1 - det[..., None]) + dc * det[..., None]
        base = base * (1 - acc[..., None]) + rgb(acc_col) * acc[..., None]
        yy = np.linspace(0.0, 1.0, h, dtype=F32)[:, None, None]
        base = np.clip(base * (1.10 - 0.26 * yy), 0, 4).astype(F32)
        from .. import form
        _drop_shadow(cv, sil, x0, y0, k, strength=0.22)
        col = form.shade(d, base, light=light, material=material, profile="round", width=max(3.0, 13.0 * k),
                         contact=0.35)
        cv.paste(col, x0, y0, mask=sil)
    elif style == "sticker":
        bw = 6.0 * k
        border = _fill_holes(_fill(d - bw))
        _drop_shadow(cv, border, x0, y0, k, strength=0.38)
        _paste(cv, "#ffffff", border, x0, y0)
        _paste(cv, color, body, x0, y0)
        _paste(cv, det_col, det * sil, x0, y0)
        _paste(cv, acc_col, acc, x0, y0)
        alpha = border
    elif style == "cutout":
        _paste(cv, color, sil, x0, y0)
        dx, dy = _shadow_dir()                       # the cut edge throws a shadow onto what shows through
        sh = _shift(1.0 - sil, dx * 2.5 * k, dy * 2.5 * k)
        sh = _f.gaussian_blur(sh, max(0.8, 1.6 * k)) * sil
        _paste(cv, "#000000", np.clip(sh * 0.7, 0, 1), x0, y0)
        lit = _shift(sil, -dx * 1.2 * k, -dy * 1.2 * k) * (1 - sil)   # bright paper edge on the far side
        _paste(cv, "#ffffff", np.clip(_f.gaussian_blur(lit, 0.6) * 0.35, 0, 1), x0, y0)
    return Pictogram(cv, rec["name"], (x0, y0, x1, y1), np.clip(alpha, 0, 1).astype(F32), bb, anchors, style)


def _fill_holes(a):
    """Fill every enclosed hole of a mask (a die-cut sticker has one outer edge)."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(((a > 0.5) * 255).astype(np.uint8), "L")
    h, w = a.shape
    pad = Image.new("L", (w + 2, h + 2), 0)
    pad.paste(im, (1, 1))
    ImageDraw.floodfill(pad, (0, 0), 128)
    holes = np.asarray(pad, np.uint8)[1:-1, 1:-1] == 0
    return np.where(holes, 1.0, a).astype(F32)


def _shift(a, dx, dy):
    """Shift a crop by (dx, dy) px (bilinear, edges repeat)."""
    h, w = a.shape
    Y, X = np.mgrid[0:h, 0:w].astype(F32)
    xs, ys = np.clip(X - dx, 0, w - 1), np.clip(Y - dy, 0, h - 1)
    x0, y0 = np.floor(xs).astype(np.int32), np.floor(ys).astype(np.int32)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = xs - x0, ys - y0
    top = a[y0, x0] * (1 - fx) + a[y0, x1] * fx
    bot = a[y1, x0] * (1 - fx) + a[y1, x1] * fx
    return (top * (1 - fy) + bot * fy).astype(F32)


def _drop_shadow(cv, sil, x0, y0, k, strength=0.32):
    dx, dy = _shadow_dir()
    off = 3.0 * k
    s = _f.gaussian_blur(_shift(sil, dx * off, dy * off), max(0.8, 3.0 * k))
    _paste(cv, "#000000", np.clip(s * strength, 0, 1), x0, y0, mode="normal")


def _anchors(rec, M, bb):
    x0, y0, x1, y1 = bb
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    a = {"center": (cx, cy), "top": (cx, y0), "bottom": (cx, y1), "left": (x0, cy), "right": (x1, cy)}
    for k_, p in rec["anchors"].items():
        q = _v._apply(M, np.array([p], np.float64))[0]
        a[k_] = (float(q[0]), float(q[1]))
    return a


# ============================================================================================ masks, SVG, sheet
def mask(name, h, w, box=None, role="silhouette", rotate=0.0, flip=False):
    """Float mask (h, w) of pictogram ``name`` for custom painting (texture, gradient, photo).

    box: (x0, y0, x1, y1) px where the 100-unit box goes (default: the largest centred square); role:
    silhouette (body + accent, details cut out), solid (body + accent), body, detail or accent."""
    rec = _get(name)
    check_choice("mask", "role", role, ("silhouette", "solid", "body", "detail", "accent"))
    h, w = int(h), int(w)
    if box is None:
        s = min(h, w)
        box = ((w - s) / 2, (h - s) / 2, (w + s) / 2, (h + s) / 2)
    bx0, by0, bx1, by1 = (float(v) for v in box)
    sx, sy = (bx1 - bx0) / 100.0, (by1 - by0) / 100.0
    M = _matrix(1.0, rotate, flip)
    M = _v._mul(_v._mul(_v._translate(bx0, by0), _v._scale(sx, sy)), M)
    if role in ROLES:
        return _raster_role(rec[role], M, w, h)
    body = np.maximum(_raster_role(rec["body"], M, w, h), _raster_role(rec["accent"], M, w, h))
    if role == "solid":
        return body
    return body * (1.0 - _raster_role(rec["detail"], M, w, h))


def drawing(name, size=100, color=DEFAULT_COLOR, style="silhouette", detail_color=None, accent=None,
            background=None, line_width=None):
    """A ``vector.Drawing`` of the pictogram (``size`` px square) for SVG export: ``.save_svg(path)``.

    style: silhouette (details cut out with a clip), duotone, or line (outlines of every path at one weight)."""
    rec = _get(name)
    check_choice("drawing", "style", style, ("silhouette", "duotone", "line"))
    dwg = _v.Drawing(size, size, background=background, name=rec["name"])
    k = size / 100.0
    det_col = detail_color if detail_color is not None else "#" + "".join(
        "%02x" % int(round(float(c) * 255)) for c in _auto_detail(color))
    acc_col = accent if accent is not None else (rec["accent_color"] or DEFAULT_ACCENT)
    if acc_col == "detail":
        acc_col = det_col

    def add_role(g, strings, **st):
        """Paths in order; an eraser clips everything painted before it in the role (nested clip groups)."""
        items = []
        for d in strings:
            if d.startswith("-"):
                clip = g.path(_FRAME + d[1:].strip(), fill_rule="evenodd", add=False)
                sub = g.group(clip=clip)
                for it in items:
                    it.remove()
                    sub.children.append(it)
                    it.parent = sub
                items = [sub]
            else:
                items.append(g.path(d, **st))
        return items

    with dwg.group(rec["name"], scale=k) as g:
        if style == "line":
            sw = 4.5 if line_width is None else float(line_width)
            for d in rec["body"] + rec["accent"] + rec["detail"]:
                if not d.startswith("-"):
                    g.path(d, fill=None, stroke=color, stroke_width=sw, join="round", cap="round")
        else:
            silh = style == "silhouette"
            if silh and rec["detail"]:
                gg = g
                for batch in _disjoint([d for d in rec["detail"] if not d.startswith("-")]):
                    gg = gg.group(clip=g.path(_FRAME + " ".join(batch), fill_rule="evenodd", add=False))
                add_role(gg, rec["body"], fill=color)
                add_role(gg, rec["accent"], fill=accent if accent is not None else color)
                for d in rec["detail"]:                    # detail erasers give the body back (black keys)
                    if d.startswith("-"):
                        with g.group(clip=g.path(d[1:].strip(), add=False)) as gg:
                            add_role(gg, rec["body"], fill=color)
                            add_role(gg, rec["accent"], fill=accent if accent is not None else color)
            else:
                add_role(g, rec["body"], fill=color)
                if not silh:
                    add_role(g, rec["detail"], fill=det_col)
                add_role(g, rec["accent"], fill=acc_col if not silh else (accent or color))
    return dwg


def sheet(cv, names_=None, box=None, cols=None, style="duotone", color=DEFAULT_COLOR, label=True,
          label_color=None, tile_color=None, log=None, **kw):
    """A labelled contact sheet of pictograms in ``box`` (u0, v0, u1, v1 fractions; default the whole canvas).
    ``names_``: list of names (default all); extra keywords go to ``pictogram``. Returns the Pictogram list."""
    ns = list(names_) if names_ is not None else names()
    if not ns:
        return []
    u0, v0, u1, v1 = box if box is not None else (0.02, 0.02, 0.98, 0.98)
    W, H = cv.w, cv.h
    bw, bh = (u1 - u0) * W, (v1 - v0) * H
    n = len(ns)
    if cols is None:
        cols = max(1, int(round(math.sqrt(n * bw / max(bh, 1) * 0.9))))
    rows = int(math.ceil(n / cols))
    tw, th = bw / cols, bh / rows
    lab = th * 0.2 if label else 0.0
    side = min(tw, th - lab) * 0.86
    out = []
    lc = label_color or ("#5b6472" if float(luma(rgb(cv.buf[int(v0 * H), int(u0 * W)]))) > 0.5 else "#c9d0da")
    for i, nm in enumerate(ns):
        r, c = divmod(i, cols)
        cx = u0 * W + (c + 0.5) * tw
        cy = v0 * H + r * th + (th - lab) / 2
        if tile_color is not None:
            _tile(cv, cx, cy, side * 1.08, tile_color)
        p = pictogram(cv, nm, at=(cx / W, cy / H), size=side / H, color=color, style=style, log=log, **kw)
        out.append(p)
        if label:
            fs = max(6.0, min(lab * 0.62, tw * 0.98 / (0.5 * max(len(nm), 1))))
            cv.text(nm, (cx, v0 * H + (r + 1) * th - lab * 0.55), size=fs, color=lc, anchor="mm", audit=False,
                    clip_ok=True)
    return out


def _disjoint(strings):
    """Split path strings into batches whose boxes do not overlap (one even-odd clip per batch)."""
    batches = []
    for d in strings:
        b = _path(d).bbox()
        for bt in batches:
            if all(b[2] <= o[0] or o[2] <= b[0] or b[3] <= o[1] or o[3] <= b[1] for o in bt[1]):
                bt[0].append(d)
                bt[1].append(b)
                break
        else:
            batches.append(([d], [b]))
    return [bt[0] for bt in batches]


def _tile(cv, cx, cy, s, color):
    x0, y0 = int(max(0, cx - s / 2 - 1)), int(max(0, cy - s / 2 - 1))
    x1, y1 = int(min(cv.w, cx + s / 2 + 2)), int(min(cv.h, cy + s / 2 + 2))
    if x1 <= x0 or y1 <= y0:
        return
    M = _v._mul(_v._translate(cx - s / 2 - x0, cy - s / 2 - y0), _v._scale(s / 100.0))
    _paste(cv, color, _raster_role([rect(0, 0, 100, 100, 14)], M, x1 - x0, y1 - y0), x0, y0)


def _alias_of(name):
    try:
        r = _get(name)
    except Exception:                                    # noqa: BLE001
        return None
    return [r["name"]] + r["aliases"]


def _entry_name(a):
    if a.get("shape") is None:
        return _canonical(a.get("name"))
    return a.get("name") or _canonical(a.get("shape"))


def _entry_alias(a):
    al = a.get("alias")
    al = [al] if isinstance(al, str) else list(al or [])
    old_spelling = a.get("shape") is None
    own = _alias_of(a.get("name") if old_spelling else a.get("shape")) or []
    return al + [x for x in own if x not in al] if (al or (a.get("name") and not old_spelling)) else (own or None)


_manifest.register(globals(), "object", ["pictogram"], arg=_entry_name, alias=_entry_alias)
pictogram = _strict(pictogram, "pictogram")

mask = _strict(mask, "pictograms.mask")
drawing = _strict(drawing, "pictograms.drawing")
sheet = _strict(sheet, "pictograms.sheet")
