# -*- coding: utf-8 -*-
"""``studio api``: the exact arguments of any call a cell can make, in one command (no picture needed).

  studio api                      the short map: shapes, paint options, solids, text, which kit holds what
  studio api sk.paint creature    signature with defaults, the first doc lines, a usage example
  studio api steam                a bare word lists every call with that word in its name
"""
from __future__ import annotations

import difflib
import glob
import importlib
import inspect
import os
import re

from .engine import _KIT_MODULES

_SHAPES = ("circle", "ellipse", "box", "polygon", "blob", "stroke", "arc", "curve", "arrow")
_SOLIDS = ("creature", "tabletop", "Ground", "ball", "cylinder", "cuboid", "lathe")
_NS_FUNCS = {"creature": ("pixelcanvas.creature", "creature"), "tabletop": ("pixelcanvas.form", "tabletop"),
             "Ground": ("pixelcanvas.form", "Ground"), "ball": ("pixelcanvas.form", "ball"),
             "cylinder": ("pixelcanvas.form", "cylinder"), "cuboid": ("pixelcanvas.form", "cuboid"),
             "lathe": ("pixelcanvas.form", "lathe"), "paper": ("pixelcanvas.illustrate", "paper"),
             "grade": ("pixelcanvas.illustrate", "grade"), "limit_palette": ("pixelcanvas.illustrate", "limit_palette"),
             "Sketch": ("pixelcanvas.illustrate", "Sketch"), "Canvas": ("pixelcanvas.canvas", "Canvas")}
_STUDIO = {"box": 'box(layer) -> (x0, y0, x1, y1): where another layer is now, in fractions; the cell is drawn '
                  'again when that layer moves.  e.g. x0, y0, x1, y1 = box("table")',
           "anchor": 'anchor(layer, which="center") -> (x, y) in fractions: center top bottom left right top_left ... '
                     'and a creature\'s mouth nose head back tail paws.  e.g. mx, my = anchor("fox", "mouth")',
           "scene": 'scene(path) -> the whole picture of a scene file (tools/new_scene.py: chart, diagram, drawing, '
                    'UI) at this layer\'s size.  e.g. cv = scene("../sales.py")'}


def _mod(name):
    return importlib.import_module(name)


def _public(obj):
    names = getattr(obj, "__all__", None) or [n for n in dir(obj) if not n.startswith("_")]
    return [n for n in names if callable(getattr(obj, n, None))]


def _resolve(name):
    """'sk.paint' / 'paint' / 'creature' / 'elements.steam' / 'cv.text' -> [(label, object)]."""
    from ..illustrate import Sketch
    from ..canvas import Canvas
    n = name.strip().rstrip("()")
    if n.startswith("sk."):
        a = n[3:]
        return [(f"sk.{a}", getattr(Sketch, a))] if hasattr(Sketch, a) else []
    if n.startswith("cv."):
        a = n[3:]
        return [(f"cv.{a}", getattr(Canvas, a))] if hasattr(Canvas, a) else []
    if "." in n:
        mod, _, a = n.partition(".")
        try:
            m = _mod(f"pixelcanvas.{mod}")
        except ImportError:
            return []
        if "." in a:                                     # chart.Axes.bar
            cls, _, meth = a.partition(".")
            c = getattr(m, cls, None)
            return [(n, getattr(c, meth))] if c is not None and hasattr(c, meth) else []
        if hasattr(m, a):
            return [(n, getattr(m, a))]
        return _class_methods(m, mod, a)                 # chart.bar -> chart.Axes.bar (fig.axes(...).bar)
    if n in _STUDIO:
        return [(n, _STUDIO[n])]
    if n in _NS_FUNCS:
        mod, a = _NS_FUNCS[n]
        return [(n, getattr(_mod(mod), a))]
    out = []
    if hasattr(Sketch, n) and not n.startswith("_"):
        out.append((f"sk.{n}", getattr(Sketch, n)))
    if hasattr(Canvas, n) and not n.startswith("_"):
        out.append((f"cv.{n}", getattr(Canvas, n)))
    if n in _KIT_MODULES:
        out.append((n, _mod(f"pixelcanvas.{n}")))
    return out


_PAGES = {"chart": "charts.md", "diagram": "diagrams.md", "ui": "ui.md", "drafting": "drafting.md",
          "vector": "vector.md", "flora": "flora.md", "elements": "elements.md", "form": "forms.md",
          "figure": "figures.md", "creature": "creatures.md", "clay": "clay.md", "solid": "solid.md",
          "texture": "textures.md", "backdrop": "backdrops.md", "pictograms": "pictograms.md", "stylize": "styles.md"}


def _class_methods(m, mod, a):
    """``a`` as a method of the kit's own classes (a chart's Axes, a Diagram, a ui Screen)."""
    out = []
    for cname, c in sorted(vars(m).items()):
        if (isinstance(c, type) and getattr(c, "__module__", "") == m.__name__ and not cname.startswith("_")
                and a in vars(c) and callable(getattr(c, a))):
            out.append((f"{mod}.{cname}.{a}", getattr(c, a)))
    if not out:
        for cname, c in sorted(vars(m).items()):        # private bases (chart's _Part)
            if isinstance(c, type) and getattr(c, "__module__", "") == m.__name__ and a in vars(c) \
                    and callable(getattr(c, a)):
                out.append((f"{mod}.{cname}.{a}", getattr(c, a)))
                break
    return out


def page_section(mod, word):
    """'charts.md, section "## Bars"': where a kit's reference page explains ``word``."""
    from .. import cli
    page = _PAGES.get(mod, f"{mod}.md")
    path = os.path.join(cli.runtime_root() or "", "skills", "pixelcanvas", "references", page)
    try:
        text = open(path, encoding="utf-8").read()
    except OSError:
        return None
    head = None
    for ln in text.splitlines():
        if ln.startswith("#"):
            head = ln.strip()
        if re.search(r"\b" + re.escape(word) + r"\b", ln) and head:
            return f"skills/pixelcanvas/references/{page}, section {head!r}"
    return f"skills/pixelcanvas/references/{page}"


def _all_names():
    from ..illustrate import Sketch
    from ..canvas import Canvas
    names = [f"sk.{a}" for a in dir(Sketch) if not a.startswith("_")]
    names += [f"cv.{a}" for a in dir(Canvas) if not a.startswith("_")]
    names += list(_NS_FUNCS) + list(_STUDIO)
    for k in tuple(_KIT_MODULES) + ("chart", "diagram", "ui"):
        try:
            m = _mod(f"pixelcanvas.{k}")
            names += [f"{k}.{a}" for a in _public(m)]
            for c in vars(m).values():                   # chart.bar: a method of the kit's own classes
                if isinstance(c, type) and getattr(c, "__module__", "") == m.__name__:
                    names += [f"{k}.{a}" for a in vars(c) if not a.startswith("_") and callable(getattr(c, a))]
        except Exception:                                # noqa: BLE001 - a kit that fails to import is skipped
            continue
    return list(dict.fromkeys(names))


def _sig(label, obj):
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return f"{label}(...)"
    params = [p for p in sig.parameters.values() if p.name != "self"]
    return f"{label}({', '.join(str(p) for p in params)})"


def _doc(obj, lines=2):
    d = inspect.getdoc(obj) or ""
    out = []
    for ln in d.splitlines():
        if not ln.strip():
            if out:
                break
            continue
        out.append(ln.strip())
        if len(out) >= lines:
            break
    return " ".join(out)[:260]


def _none_options(obj, most=8):
    """For each option the signature leaves as None: what the docstring says it takes."""
    try:
        params = [p for p in inspect.signature(obj).parameters.values() if p.default is None]
    except (TypeError, ValueError):
        return []
    doc = inspect.getdoc(obj) or ""
    out = []
    for p in params:
        m = re.search(r"(?<![\w`])`{0,2}" + re.escape(p.name) + r"`{0,2}\s*[:=]\s*([^;\n]+)", doc)
        if m:
            out.append(f"  {p.name}=None: {m.group(1).strip().rstrip('.,')[:110]}")
        if len(out) >= most:
            break
    return out


def _example(label):
    """A one-line usage from the kit registry or a reference page."""
    short = label.split(".")[-1]
    pat = re.compile(r"(?<![\w.])" + re.escape(label if label.startswith(("sk.", "cv.")) or "." in label
                                               else short) + r"\(")
    found = []                                           # an example with an outline is the last choice: an
    try:                                                 # outline reads as a drawing, never as a photo
        from .. import catalog
        for e in catalog.entries():
            call = str(getattr(e, "call", "") or "")
            if pat.search(call) and "sc.add" not in call:
                found.append(call.replace("creature.creature(", "creature("))
                if not _outlined(found[-1]):
                    return found[-1]
    except Exception:                                    # noqa: BLE001
        pass
    from .. import cli
    root = cli.runtime_root()
    pages = sorted(glob.glob(os.path.join(root or "", "skills", "pixelcanvas", "references", "*.md")))
    pages.append(os.path.join(root or "", "skills", "pixelcanvas", "SKILL.md"))
    for p in pages:
        try:
            text = open(p, encoding="utf-8").read()
        except OSError:
            continue
        for block in re.findall(r"```python\n(.*?)```", text, re.S):
            for ln in block.splitlines():
                if pat.search(ln) and len(ln.strip()) < 150:
                    found.append(ln.strip())
                    if not _outlined(found[-1]):
                        return found[-1]
    return found[0] if found else None


def _outlined(code):
    return bool(re.search(r"\bline\s*=\s*(?:True|\"ink\"|'ink')|outline\s*=\s*(?:on|True)", code))


def describe(label, obj):
    if isinstance(obj, str):                             # box / anchor / scene
        return [obj]
    if inspect.ismodule(obj):
        names = _public(obj)
        return [f"{label}: {_doc(obj, 1)[:120]}", "  calls: " + ", ".join(names[:40])
                + (f" ... +{len(names) - 40}" if len(names) > 40 else ""), f"  studio api {label}.NAME for one"]
    out = [_sig(label, obj)]
    d = _doc(obj)
    if d:
        out.append("  " + d)
    out += _none_options(obj)
    ex = _example(label)
    if ex:
        out.append("  e.g. " + ex)
    if label == "creature":
        fn = obj
        try:
            out.append("  species: " + ", ".join(fn.species()))
            out.append("  poses: " + ", ".join(fn.QUAD_POSES) + " (four-legged animals: fox, dog, cat, horse ...)")
            out.append("  poses of birds, fish and small animals: " + ", ".join(fn.poses("sparrow")))
        except Exception:                                # noqa: BLE001
            pass
    return out


def overview():
    from ..illustrate import Sketch
    from ..canvas import Canvas
    shapes = [s for s in _SHAPES if hasattr(Sketch, s)]
    lines = ["shapes (sk., fractions of the height; x of the width) -> masks:"]
    lines.append("  " + "; ".join(_sig(s, getattr(Sketch, s)).replace("self, ", "") for s in shapes[:5]))
    lines.append("  " + "; ".join(_sig(s, getattr(Sketch, s)) for s in shapes[5:]))
    lines.append("  a * b clips two masks, np.maximum(a, b) joins them; with sk.at(x, y, size): draws in a local frame")
    lines.append(_sig("sk.paint", Sketch.paint))
    other = [m for m in dir(Sketch) if not m.startswith("_") and m not in shapes + ["paint", "at", "px", "size"]]
    lines.append("more sk.: " + ", ".join(other))
    lines.append(_sig("cv.text", Canvas.text) + "  (box=(x0, y0, x1, y1) fits the text; font=, bold=)")
    lines.append("solids and animals: " + "; ".join(_sig(n, getattr(_mod(_NS_FUNCS[n][0]), _NS_FUNCS[n][1]))
                                                   .split("(")[0] + "(" + ", ".join(
        str(p) for p in list(inspect.signature(getattr(_mod(_NS_FUNCS[n][0]), _NS_FUNCS[n][1])).parameters.values())
        [1:4]) + ", ...)" for n in _SOLIDS))
    lines.append("finish (a pass cell): grade(cv, mood, amount); paper(cv); limit_palette(cv, colors); cv.vignette()")
    lines.append("other layers: box(\"layer\") -> (x0, y0, x1, y1); anchor(\"layer\", \"mouth\") -> (x, y); "
                 "scene(\"chart.py\") -> a whole chart / diagram picture")
    lines.append("also in every section: np, math, random and rng (seeded per section), W, H, S, seed; names a "
                 "section above defines")
    kits = {"flora": "trees, shrubs, flowers, grass, rocks", "elements": "sky, sun, moon, clouds, fire, smoke, "
            "steam, rain, snow, water, light", "form": "2.5D objects, rooms, shadows", "figure": "posable people",
            "clay": "3D clay characters and props", "pictograms": "633 icons", "solid": "ray-marched 3D objects",
            "backdrop": "one-call backgrounds", "texture": "80 materials", "stylize": "43 art styles"}
    lines.append("kits (use as flora.tree(...)): " + "; ".join(f"{k}: {v}" for k, v in kits.items()))
    lines.append("studio api NAME [NAME ...] gives the exact arguments, e.g. studio api sk.paint creature elements.steam")
    return lines


def run(names):
    """Print the answer for ``names`` (or the map); returns an exit code."""
    from .commands import say
    if not names:
        for ln in overview():
            say(ln)
        return 0
    code = 0
    for name in names:
        found = _resolve(name)
        if not found:
            word = name.split(".")[-1].lower()
            allnames = _all_names()
            hits = [n for n in allnames if word in n.split(".")[-1].lower()]
            if hits and "." not in name:
                say(f"{name}: {len(hits)} call(s) with '{word}' in the name:")
                for h in hits[:12]:
                    obj = _resolve(h)
                    d = _doc(obj[0][1], 1)[:90] if obj and not isinstance(obj[0][1], str) else ""
                    say(f"  {h}" + (f" - {d}" if d else ""))
                if len(hits) > 12:
                    say(f"  ... +{len(hits) - 12} more")
                continue
            near = difflib.get_close_matches(name, allnames, n=4, cutoff=0.55)
            close = difflib.get_close_matches(name, allnames, n=1, cutoff=0.8)
            if not close and "." in name and name.split(".")[0] in _KIT_MODULES + ("chart", "diagram", "ui",
                                                                                   "drafting", "vector"):
                sec = page_section(name.split(".")[0], word)
                if sec:
                    say(f"{name}: not a call of its own; read {sec}")
                    continue
            near += [n for n in allnames if n.split(".")[-1] == word and n not in near][:3]
            say(f"{name}: unknown" + (f" - did you mean {' or '.join(near[:4])}?" if near else
                                      " (studio api lists the calls)"))
            code = 2
            continue
        for label, obj in found:
            for ln in describe(label, obj):
                say(ln)
    return code
