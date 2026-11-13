#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render a scene (scenes/<name>.py).

  python tools/render.py mountains_sunset                 full size -> outputs/mountains_sunset.png (+ _preview.jpg)
  python tools/render.py mountains_sunset --scale 0.5     draft (960x540, ~4x faster) -> outputs/mountains_sunset_x0.5.png
  python tools/render.py mountains_sunset --seed 7 --size 2560x1440 --out out.png
  python tools/render.py mountains_sunset --crop 700 300 1500 750      extra full-resolution crops (repeatable)
  python tools/render.py --list                           list scenes
  python <root>/tools/render.py ./art/poster.py --check   any scene file, from any folder -> ./art/outputs/
  python tools/render.py poster --deliver ./site/img/     also copy the final PNG (+ .svg / .psd / .ora / .gif)

A bare name is looked up in ./, ./scenes, $PIXELCANVAS_WORK and <root>/scenes. Results go to outputs/ next to the
scene (<project>/outputs for <project>/scenes/x.py), or to --outdir / env PIXELCANVAS_OUT.

Iterating faster (all write labelled images you can look at):

  --check                       also write <out>_check.jpg (guides | values | edges | thumbnail) and print the
                                warnings, most severe first
  --round                       also keep this draft as outputs/rounds/<scene>/rNN_preview.jpg (never overwritten);
                                `look.py rounds <scene>` turns all rounds into one labelled "how it improved" strip
                                (same as env PIXELCANVAS_ROUNDS=1)
  --layers                      also write editable layers, one per log("step"): <out>.psd (Photoshop /
                                Photopea / GIMP), <out>.ora (Krita / GIMP), <out>_layers/ (PNGs + layers.json)
  --stages                      also write <out>_stages.jpg: the canvas after every drawing step
                                (steps = each log("...") / sc.log("...") call in the scene)
  --set NAME=EXPR               override a module-level constant before rendering (repeatable), e.g.
                                --set SUN_X=0.7  --set "SKY[1]=(0.5, '#ff8040')"  --set "LAYERS[0]['fog']=0.6"
  --sweep NAME EXPR EXPR ...    render one variant per value -> <scene>_sweep_<NAME>.jpg contact sheet
  --seeds N N N ...             render one variant per seed  -> <scene>_seeds.jpg contact sheet
                                (sweeps default to --scale 0.33 unless --scale is given)
  --animate NAME START END      one frame per step with the constant NAME swept START -> END (NAME defaults to T;
                                several: "SUN_X:0.2:0.8,SUN_Y:0.7:0.3") -> outputs/<scene>.gif + <scene>_frames.jpg
                                (filmstrip of ~8 labelled frames). Options: --frames 24 --fps 12 --format gif|webp|apng
                                --bounce --hold N --ease linear|inout --loop (END == START, not rendered twice)
                                --dither (gif) --out path; default --scale 0.5
  --json                        print, as the last line, one JSON object: scene, size, scale, seed, seconds,
                                outputs, steps (name + seconds), warnings (structured), stats, and the --check
                                report (text_issues, objects, relations, object_issues, prompt_issues, kind)
  --svg                         also write outputs/<scene>.svg from the vector drawings the scene painted
  --dxf                         also write outputs/<scene>.dxf when the scene drew a drafting Sheet
  --style NAME                  apply a pixelcanvas.stylize style after the scene (preview any scene in any style)
  --styles a,b,c                write <out>_styles.jpg: the scene in each style, labelled (contact sheet)
  --prompt TEXT / --kind KIND   check against this request / judge as this kind (override PROMPT / KIND)

The --check report has four sections, fails first in each: WARNINGS (image critique; painterly rules are dropped
for diagrams / drawings and softened for designs), TEXT (text.audit_text), OBJECTS (what was drawn where, from
cv.manifest, plus relations such as "mug overlaps bottle"), PROMPT (only when the scene defines PROMPT: quoted text
that was not drawn, a named style that was not applied, counts that do not match). A scene may define
PROMPT = "the request, verbatim", KIND = "painting" | "design" | "diagram" | "drawing" and DEFAULT_SIZE = (w, h).

If the scene raises, the error is explained first (type, message, scene file line, likely cause), then the
traceback tail; the exit code is non-zero. Set PIXELCANVAS_TRACE=<file> to append one JSON line per run.
"""
from __future__ import annotations

import argparse
import copy
import difflib
import importlib
import inspect
import json
import math
import os
import re
import sys
import time
import traceback

for _s in (sys.stdout, sys.stderr):              # Windows consoles default to cp950 -> mojibake
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from pixelcanvas import cli as workspace  # noqa: E402  (scene lookup, output folders, --deliver)
import numpy as np  # noqa: E402

SCENE_DIR = os.path.join(ROOT, "scenes")         # folder of the scene being rendered (for error reports)


class UserError(Exception):
    """A mistake in the command line (unknown scene, bad --set ...): printed as one clear message."""


def _rel(p):
    try:
        return os.path.relpath(p).replace("\\", "/")
    except ValueError:                                # different drive on Windows
        return str(p).replace("\\", "/")


def list_scenes():
    d = os.path.join(ROOT, "scenes")
    return sorted(f[:-3] for f in os.listdir(d) if f.endswith(".py") and not f.startswith("_"))


def _eval(expr, ns):
    try:
        return eval(expr, dict(ns))
    except Exception:
        return expr                                   # bare words / '#ff8040' -> plain string


def constants(mod):
    """The UPPER_CASE module-level names of a scene (what --set / --sweep can change)."""
    plain = (int, float, str, bool, tuple, list, dict, set, type(None))
    return [k for k, v in vars(mod).items() if k.isupper() and not k.startswith("_") and isinstance(v, plain)]


def apply_override(mod, assignment):
    """'NAME=EXPR' or 'NAME[..][..]=EXPR' -> assign inside the scene module. Returns (root, original value)."""
    lhs, sep, rhs = assignment.partition("=")
    lhs, rhs = lhs.strip(), rhs.strip()
    root = re.match(r"[A-Za-z_]\w*", lhs)
    if not root or not sep:
        raise UserError(f"--set expects NAME=EXPR (e.g. --set SUN_X=0.7), got {assignment!r}")
    root = root.group(0)
    if not hasattr(mod, root):
        names = constants(mod)
        close = difflib.get_close_matches(root, names, n=3, cutoff=0.6)
        msg = f"unknown --set name {root!r} for scene {mod.__name__.split('.')[-1]!r}."
        if close:
            msg += f" Did you mean: {', '.join(close)}?"
        raise UserError(msg + f"\navailable constants: {', '.join(names) or '(none: the scene defines no UPPER_CASE constants)'}")
    original = copy.deepcopy(getattr(mod, root))
    ns = vars(mod)
    ns["__pc_value"] = _eval(rhs, ns)
    try:
        exec(f"{lhs} = __pc_value", ns)
    except Exception as e:
        setattr(mod, root, original)
        raise UserError(f"--set {assignment!r} failed: {type(e).__name__}: {e}. Current value of {root}: {original!r}"[:600])
    finally:
        ns.pop("__pc_value", None)
    return root, original


# ----------------------------------------------------------------------------- helpful failures

def _lookup_callable(qualified):
    """'Canvas.paint' / 'sky' / 'Scene.set_sun' -> the function object, searched in the pixelcanvas modules."""
    import pixelcanvas
    from pixelcanvas import scenery
    parts = qualified.split(".")
    for ns in (pixelcanvas, scenery):
        obj = ns
        try:
            for p in parts:
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    return None


def _all_public_names():
    import pixelcanvas
    from pixelcanvas import scenery
    names = set(getattr(pixelcanvas, "__all__", [])) | set(getattr(scenery, "__all__", []))
    names |= {n for n in dir(pixelcanvas) if not n.startswith("_")} | {n for n in dir(scenery) if not n.startswith("_")}
    return sorted(names)


def likely_causes(exc, frames):
    """Short, concrete explanations for the mistakes agents make most often."""
    msg = str(exc)
    out = []
    src = " ".join((f.line or "") for f in frames)
    if "could not be broadcast" in msg or "shape mismatch" in msg or (isinstance(exc, (ValueError, IndexError)) and "shape" in msg):
        shapes = re.findall(r"\(\d+(?:,\s*\d+)*,?\)", msg)
        out.append("a mask or colour map does not match the canvas size"
                   + (f" (shapes {' vs '.join(shapes[:2])})" if len(shapes) >= 2 else "")
                   + ". Masks must be float32 (H, W) with H, W = cv.h, cv.w (sc.H, sc.W in a Scene); colour maps (H, W, 3). "
                   "Build them from cv.xy() / sc.X, sc.Y instead of a fixed size, and give every helper the current width and height.")
    m = re.search(r"(?:^|\s)([\w.]+)\(\) got an unexpected keyword argument '(\w+)'", msg)
    if m:
        name, kw = m.group(1), m.group(2)
        fn = _lookup_callable(name.replace("Scene.", "Scene.")) or _lookup_callable(name.split(".")[-1])
        extra = ""
        if fn is not None:
            try:
                sig = inspect.signature(fn)
                params = [p for p in sig.parameters if p != "self"]
                close = difflib.get_close_matches(kw, params, n=2, cutoff=0.5)
                extra = f" Signature: {name}{sig}." + (f" Did you mean: {', '.join(close)}?" if close else "")
            except (TypeError, ValueError):
                pass
        out.append(f"unknown keyword {kw!r} for {name}().{extra} See skills/pixelcanvas/references/api.md for the options.")
    m = re.search(r"([\w.]+)\(\) missing (\d+) required (?:positional|keyword-only) arguments?: (.+)", msg)
    if m:
        fn = _lookup_callable(m.group(1)) or _lookup_callable(m.group(1).split(".")[-1])
        sig = ""
        if fn is not None:
            try:
                sig = f" Signature: {m.group(1)}{inspect.signature(fn)}."
            except (TypeError, ValueError):
                pass
        out.append(f"{m.group(1)}() needs {m.group(3)}.{sig}")
    m = re.search(r"'(\w+)' object has no attribute '(\w+)'", msg)
    if m and not m.group(1) == "NoneType":
        cls = _lookup_callable(m.group(1))
        names = [n for n in dir(cls) if not n.startswith("_")] if cls is not None else []
        close = difflib.get_close_matches(m.group(2), names, n=3, cutoff=0.5)
        out.append(f"{m.group(1)} has no attribute {m.group(2)!r}." + (f" Did you mean: {', '.join(close)}?" if close else
                   (f" Available: {', '.join(names[:25])}" if names else "")))
    m = re.search(r"name '(\w+)' is not defined", msg)
    if m:
        close = difflib.get_close_matches(m.group(1), _all_public_names(), n=3, cutoff=0.6)
        out.append(f"{m.group(1)!r} is not defined: add it to the imports (from pixelcanvas import *; from pixelcanvas.scenery import *)"
                   + (f" or fix the spelling (did you mean {', '.join(close)}?)." if close else "."))
    if "NoneType" in msg and re.search(r"\.(sun|D|haze|light)\b|sun\(|god_rays|haze", src + " " + msg):
        out.append("a recipe ran before the scene state it needs was set. Call sc.set_horizon(frac), sc.set_sun(x, y, light_dir=...) "
                   "and sc.set_haze(stops) right after creating the Scene, before sky(), sun(), mountain_layer(), water_surface() ...")
    elif "NoneType" in msg and not out:
        out.append("a value is None: a function returned nothing (many canvas methods return self, recipes may return None) or a "
                   "variable was never assigned. Check the name on the failing line.")
    if isinstance(exc, AssertionError):
        m = re.search(r"sc\.(sun|D|haze|light|horizon)\b", src)
        need = {"sun": "sc.set_sun(x, y, light_dir=...)", "D": "sc.set_sun(x, y, light_dir=...)", "haze": "sc.set_haze(stops)",
                "light": "sc.set_sun(x, y, light_dir=...)", "horizon": "sc.set_horizon(frac)"}
        if m:
            out.append(f"a recipe was called before the scene state it needs was set: call {need[m.group(1)]} right after creating "
                       "the Scene, before sky(), sun(), mountain_layer(), water_surface() ...")
    if isinstance(exc, ValueError) and "base 16" in msg:
        out.append("a colour string is not valid hex. Use '#rrggbb' or '#rgb', an (r, g, b) tuple in 0..1 (or ints 0..255), or an (H, W, 3) array.")
    if any(os.path.basename(f.filename) == "color.py" for f in frames) and not out:
        out.append("a colour argument was not understood. Use '#rrggbb' / '#rgb', an (r, g, b) tuple in 0..1 (ints 0..255 also work), "
                   "or an (H, W, 3) array; mixing 0..255 floats with 0..1 values gives blown-out or black results.")
    if isinstance(exc, ModuleNotFoundError):
        out.append(f"missing module {getattr(exc, 'name', '')!r}: only numpy and Pillow are available; use `from pixelcanvas import ...`.")
    if isinstance(exc, MemoryError):
        out.append("out of memory: render a smaller draft (--scale 0.5) or reduce the number of full-size float32 layers.")
    return out


def _bad_colour(tb):
    """The argument that color.rgb() choked on (read from the failing frame), or None."""
    bad = None
    while tb is not None:
        f = tb.tb_frame
        if os.path.basename(f.f_code.co_filename) == "color.py" and f.f_code.co_name in ("rgb", "as_color"):
            bad = f.f_locals.get("c", bad)
        tb = tb.tb_next
    return bad if isinstance(bad, (str, tuple, list, int, float)) else None


def failure_report(exc, tb, scene=None):
    """Build the short message that is printed before the traceback tail. Returns (dict, text)."""
    from types import SimpleNamespace
    frames = [SimpleNamespace(filename=f.filename, lineno=f.lineno, name=f.name, line=f.line or "")
              for f in traceback.extract_tb(tb)]
    scenes_dir = os.path.normcase(os.path.abspath(SCENE_DIR)) + os.sep
    mine = [f for f in frames if os.path.normcase(os.path.abspath(f.filename)).startswith(scenes_dir)]
    where = (mine or frames or [None])[-1]
    text = str(exc)[:500]
    info = {"type": type(exc).__name__, "message": text, "scene": scene}
    lines = [f"ERROR {type(exc).__name__}: {text}"]
    if isinstance(exc, SyntaxError) and exc.filename:
        info.update(file=_rel(exc.filename), line=exc.lineno, source=(exc.text or "").strip())
        lines.append(f"  at {_rel(exc.filename)} line {exc.lineno}: {(exc.text or '').strip()}")
    elif where is not None:
        info.update(file=_rel(where.filename), line=where.lineno, source=(where.line or "").strip())
        lines.append(f"  at {_rel(where.filename)} line {where.lineno}: {(where.line or '').strip()}")
        if mine and frames and frames[-1] is not where:
            inner = frames[-1]
            lines.append(f"  inside {_rel(inner.filename)} line {inner.lineno} in {inner.name}(): {(inner.line or '').strip()}")
    causes = likely_causes(exc, frames)
    bad = _bad_colour(tb)
    if bad is not None:
        causes = [f"the colour {bad!r} was not understood. Hex colours need 3 or 6 digits ('#f80' or '#ff8800')."
                  if isinstance(bad, str) else
                  f"the colour {bad!r} was not understood. Use '#rrggbb', an (r, g, b) tuple, or an (H, W, 3) array."] +                  [c for c in causes if "colour" not in c]
    info["likely_cause"] = causes
    for c in causes:
        lines.append(f"  likely cause: {c}")
    tail = "".join(traceback.format_exception(type(exc), exc, tb)).rstrip().splitlines()
    tail = tail[-9:]
    lines.append("  traceback (tail):")
    lines += ["    " + t for t in tail]
    return info, "\n".join(lines)


# ----------------------------------------------------------------------------- rendering

def render_one(mod, w, h, seed, log, stages=False, layers=False):
    from pixelcanvas import vision
    from pixelcanvas import layers as _layers
    if layers:
        _layers.enable(target_size=(w, h))
        inner_l = log

        def log(m):                                    # noqa: F811 - every log() call starts a new layer
            _layers.mark(m)
            inner_l(m)
    if stages:
        vision.enable_stages(target_size=(w, h))
        inner = log

        def log(m):                                    # noqa: F811 - every log() call marks a stage
            vision.stage(None, m)
            inner(m)
    kwargs = dict(width=w, height=h, log=log)
    if seed is not None:
        kwargs["seed"] = seed
    cv = mod.render(**kwargs)
    shots = vision.take_stages(cv) if stages else None
    if layers:
        stack = _layers.take(cv)
        stack, _ = _layers.reconcile(stack, cv)
        return cv, shots, stack
    return cv, shots


def _scene_seed(mod, seed):
    if seed is not None:
        return seed
    try:
        d = inspect.signature(mod.render).parameters["seed"].default
        return None if d is inspect.Parameter.empty else d
    except (KeyError, ValueError, TypeError, AttributeError):
        return getattr(mod, "DEFAULT_SEED", None)


def _next_round(folder):
    """Reserve r01, r02 ... without ever overwriting an existing file."""
    import look
    n = max([k for k, _ in look.list_rounds(folder)] or [0]) + 1
    while True:
        path = os.path.join(folder, f"r{n:02d}_preview.jpg")
        try:
            open(path, "xb").close()
            return n, path
        except FileExistsError:
            n += 1


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", nargs="?", help="scene name (looked up in ./, ./scenes, $PIXELCANVAS_WORK, <root>/scenes) "
                                             "or path to a scene file such as ./art/poster.py")
    ap.add_argument("--outdir", default=None, metavar="DIR",
                    help="folder for the results (default: outputs/ next to the scene; env PIXELCANVAS_OUT)")
    ap.add_argument("--deliver", default=None, metavar="PATH",
                    help="copy the final PNG (and the .svg / .psd / .ora / .gif made by this run) to PATH: a folder, "
                         "or a file name whose stem the copies share; folders are created")
    ap.add_argument("--scale", type=float, default=None, help="resolution factor: 0.5 = draft, 1 = full (default)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--size", default=None, help="WxH (default: the scene's DEFAULT_SIZE)")
    ap.add_argument("--out", default=None, help="output PNG (default outputs/<scene>[_x<scale>].png)")
    ap.add_argument("--crop", nargs=4, type=int, action="append", metavar=("X0", "Y0", "X1", "Y1"),
                    help="also save a full-resolution crop (repeatable)")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=EXPR", help="override a scene constant")
    ap.add_argument("--sweep", nargs="+", metavar=("NAME", "EXPR"), help="one variant per value -> contact sheet")
    ap.add_argument("--seeds", nargs="+", type=int, help="one variant per seed -> contact sheet")
    ap.add_argument("--stages", action="store_true", help="also save a sheet of every drawing step")
    ap.add_argument("--check", action="store_true", help="also save a diagnostic sheet + print warnings")
    ap.add_argument("--round", action="store_true",
                    help="keep this draft as outputs/rounds/<scene>/rNN_preview.jpg (also: env PIXELCANVAS_ROUNDS=1)")
    ap.add_argument("--layers", action="store_true",
                    help="also save editable layers (one per log() step): <out>.psd, <out>.ora, <out>_layers/")
    ap.add_argument("--json", action="store_true", help="print one JSON object as the last line")
    ap.add_argument("--svg", action="store_true", help="also write outputs/<scene>.svg from the scene's vector drawings")
    ap.add_argument("--dxf", action="store_true", help="also write outputs/<scene>.dxf from the scene's drafting sheet")
    ap.add_argument("--style", default=None, metavar="NAME", help="apply a stylize style after the scene")
    ap.add_argument("--styles", default=None, metavar="A,B,C", help="contact sheet of the scene in several styles")
    ap.add_argument("--prompt", default=None, metavar="TEXT", help="the request to check against (overrides PROMPT)")
    ap.add_argument("--kind", default=None, choices=KINDS, help="picture kind for the critique (overrides KIND)")
    ap.add_argument("--ref", default=None, metavar="REF.png",
                    help="--check: append the gaps to this style reference (REFERENCE block; scene: REFERENCE = ...)")
    ap.add_argument("--animate", nargs="+", metavar="NAME START END",
                    help="one frame per step with a constant swept START -> END: outputs/<scene>.gif + "
                         "<scene>_frames.jpg (NAME defaults to T; several: \"SUN_X:0.2:0.8,SUN_Y:0.7:0.3\")")
    ap.add_argument("--frames", type=int, default=24, help="--animate: number of frames (default 24)")
    ap.add_argument("--fps", type=float, default=12, help="--animate: frames per second (default 12)")
    ap.add_argument("--format", default=None, choices=("gif", "webp", "apng"), help="--animate: file format (default gif)")
    ap.add_argument("--bounce", action="store_true", help="--animate: append the reversed frames (ping-pong)")
    ap.add_argument("--hold", type=int, default=0, metavar="N", help="--animate: repeat the last frame N more times")
    ap.add_argument("--ease", default="linear", choices=("linear", "inout"), help="--animate: easing of the sweep")
    ap.add_argument("--loop", action="store_true",
                    help="--animate: START..END is one full cycle; the END frame (= START) is not rendered twice")
    ap.add_argument("--dither", action="store_true", help="--animate gif: Floyd-Steinberg dithering (default off)")
    ap.add_argument("--list", action="store_true", help="list scenes")
    return ap


# ----------------------------------------------------------------------------- report

KINDS = ("painting", "design", "diagram", "drawing")
_TEXT_SEV = {"missing_glyphs": "fail", "clipped": "fail", "contrast": "warn", "overlap": "warn", "small": "warn",
             "margin": "info", "adjusted": "info"}
_ORDER = {"fail": 0, "warn": 1, "info": 2}


def _sorted(ws):
    return sorted(ws, key=lambda w: _ORDER.get(w.get("severity"), 3))


def _at_threshold(message):
    """True for a contrast message whose shown ratio equals the shown requirement ("3.0:1 ... needs 3.0:1")."""
    m = re.search(r"(\d+(?:\.\d+)?):1.*?needs (\d+(?:\.\d+)?):1", message)
    return bool(m) and float(m.group(1)) >= float(m.group(2))


def text_issues(cv):
    """pixelcanvas.text.audit_text issues in the structured warning format."""
    from pixelcanvas.text import audit_text
    out = []
    W, H = cv.w, cv.h
    for it in audit_text(cv):
        if it["kind"] == "contrast" and _at_threshold(it["message"]):
            continue                                    # "3.0:1 (needs 3.0:1)": meets the threshold as shown
        b = it.get("bbox")
        where = [round(b[0] / W, 3), round(b[1] / H, 3), round(b[2] / W, 3), round(b[3] / H, 3)] if b else None
        out.append({"rule": "text_" + it["kind"], "severity": _TEXT_SEV.get(it["kind"], "warn"),
                    "message": f"{it['text'][:50]!r}: {it['message']}", "where": where, "value": it["text"][:80],
                    "threshold": None, "fix": it["fix"]})
    return _sorted(out)


def _contrast(a, b):
    def lum(c):
        c = np.clip(np.asarray(c, np.float32), 0, 1)
        c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return float(c @ np.array([0.2126, 0.7152, 0.0722], np.float32))
    la, lb = lum(a), lum(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def _shrunk(b, k=0.18):
    """A text line box without most of its leading (line boxes of stacked labels touch by design)."""
    h = b[3] - b[1]
    return (b[0], b[1] + k * h, b[2], b[3] - k * h)


def _hits(a, b, share=0.2):
    """Two text boxes overlap enough to collide; spans set side by side on one line (code tokens, coloured
    words) only when they overlap by half the narrower one."""
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    if not (iw > 0 and ih > 0 and small > 0 and iw * ih > share * small):
        return False
    h = max(a[3] - a[1], b[3] - b[1], 1e-6)
    if abs(a[1] - b[1]) < 0.15 * h and abs(a[3] - b[3]) < 0.15 * h:
        return iw > 0.5 * min(a[2] - a[0], b[2] - b[0])
    return True


def vector_text_issues(cv, labels=None, group_after=3):
    """The text audit for labels inside vector drawings (diagram, chart, ui, drafting): too small, clipped by
    the canvas or by its own node, overlapping other text, low contrast against what is under it. More than
    ``group_after`` issues of one kind become one summary line."""
    from pixelcanvas import manifest
    labels = manifest.vector_texts(cv) if labels is None else labels
    W, H = cv.w, cv.h
    short = min(W, H)
    min_px = max(4.0, 0.008 * short)                    # UI and chart captions are small by design
    buf = np.clip(np.asarray(cv.buf, np.float32), 0, 1)
    found = {}

    def add(kind, it, msg, fix):
        b = it["bbox"]
        found.setdefault(kind, []).append(
            {"rule": "text_" + kind, "severity": _TEXT_SEV.get(kind, "warn"), "message": f"{it['text'][:50]!r}: {msg}",
             "where": [round(b[0] / W, 3), round(b[1] / H, 3), round(b[2] / W, 3), round(b[3] / H, 3)],
             "value": it["text"][:80], "threshold": None, "fix": fix, "_it": it})
    raster = [(t.get("text", ""), t["bbox"]) for t in (getattr(cv, "text_log", None) or []) if t.get("bbox")]
    boxes = [_shrunk(it["bbox"]) for it in labels]
    for i, it in enumerate(labels):
        x0, y0, x1, y1 = it["bbox"]
        where = f"in {it['group']}" if it.get("group") else f"in drawing {it['drawing']}"
        if it["size"] < min_px:
            add("small", it, f"{it['size']:.0f}px label {where} is hard to read on a {W}x{H} canvas "
                f"(minimum ~{min_px:.0f}px)", "a larger font_size / size=, or fewer, larger elements")
        if x0 < -1 or y0 < -1 or x1 > W + 1 or y1 > H + 1:
            add("clipped", it, f"label {where} runs off the canvas", "move it inside or give the drawing more room")
        ob = it.get("owner_box")
        if ob is not None:
            tol = max(2.0, 0.15 * it["size"])
            if x0 < ob[0] - tol or y0 < ob[1] - tol or x1 > ob[2] + tol or y1 > ob[3] + tol:
                add("clipped", it, f"label sticks out of its own box ({it['group']})",
                    "shorten the label, add a line break, or widen the node (width=)")
        j = next((j for j in range(i) if _hits(boxes[i], boxes[j])), None)
        if j is not None:
            add("overlap", it, f"overlaps label {labels[j]['text'][:30]!r}",
                "move one of them (label_pos=, spacing=) or shorten it")
        t = next((t for t, o in raster if _hits(boxes[i], _shrunk(o))), None)
        if t is not None:
            add("overlap", it, f"overlaps text {str(t)[:30]!r} drawn with cv.text", "move one of them")
        col = it.get("color")
        if col is not None and not it.get("halo") and it.get("opacity", 1) >= 0.99:
            cx0, cy0 = max(0, int(x0)), max(0, int(y0))
            cx1, cy1 = min(W, int(math.ceil(x1))), min(H, int(math.ceil(y1)))
            if cx1 - cx0 >= 2 and cy1 - cy0 >= 2:
                px = buf[cy0:cy1, cx0:cx1].reshape(-1, 3)
                far = px[np.abs(px - np.asarray(col, np.float32)).max(-1) > 0.12]
                bg = np.median(far if len(far) >= 0.3 * len(px) else px, axis=0)   # ink never fills its box
                ratio = _contrast(col, bg)
                if round(ratio, 1) < 2.0:
                    add("contrast", it, f"contrast {ratio:.1f}:1 against what is under it (needs 2.0:1)",
                        "a lighter / darker label colour, or a plate behind it")
    out = []
    for kind, lst in found.items():
        if len(lst) <= group_after:
            out += lst
            continue
        lst = sorted(lst, key=lambda w: w["_it"]["size"]) if kind == "small" else lst
        ex = "; ".join(f"{w['_it']['text'][:24]!r} {w['_it']['size']:.0f}px" if kind == "small" else
                       repr(w["_it"]["text"][:24]) for w in lst[:3])
        what = {"small": f"are smaller than ~{min_px:.0f}px on this {W}x{H} canvas",
                "clipped": "are clipped (off the canvas or out of their box)", "overlap": "overlap other text",
                "contrast": "have low contrast against what is under them"}.get(kind, kind)
        out.append({"rule": "text_" + kind, "severity": lst[0]["severity"],
                    "message": f"{len(lst)} labels {what} (e.g. {ex}).", "where": None,
                    "value": len(lst), "threshold": None, "fix": lst[0]["fix"]})
    for w in out:
        w.pop("_it", None)
    return _sorted(out)


def _label_summary(labels):
    if not labels:
        return ""
    names = sorted({l["drawing"] for l in labels})
    return f"{len(labels)} label{'s' if len(labels) != 1 else ''} in {len(names)} drawing{'s' if len(names) != 1 else ''}"


def style_checks(cv, warns, image=None):
    """Critique of a styled picture: painterly rules measured inside a paper border / mat (detected), the rules
    each applied style triggers by being itself downgraded to info, and ``mostly_blank``."""
    from pixelcanvas import manifest, vision
    styles = list(getattr(cv, "styles_applied", []) or [])
    if not styles:
        return warns, None
    img = np.asarray(vision.to_pil(image), np.float32) / 255 if image is not None else np.clip(cv.buf, 0, 1)
    out, area = list(warns), None
    box = manifest.picture_area(img)
    if box is not None:
        x0, y0, x1, y1 = box
        H, W = img.shape[:2]
        from PIL import Image
        inner = vision.critique(Image.fromarray((img[y0:y1, x0:x1] * 255 + 0.5).astype(np.uint8), "RGB"))
        rules = set(manifest._AREA_RULES)

        def back(w):
            wh = w.get("where")
            sx, sy = (x1 - x0) / W, (y1 - y0) / H
            if isinstance(wh, list) and len(wh) == 4:
                wh = [round(x0 / W + wh[0] * sx, 4), round(y0 / H + wh[1] * sy, 4),
                      round(x0 / W + wh[2] * sx, 4), round(y0 / H + wh[3] * sy, 4)]
            elif isinstance(wh, list) and len(wh) == 2:
                wh = [round(x0 / W + wh[0] * sx, 4), round(y0 / H + wh[1] * sy, 4)]
            return dict(w, where=wh)
        out = [w for w in out if w.get("rule") not in rules] + [back(w) for w in inner if w.get("rule") in rules]
        area = [round(x0 / W, 3), round(y0 / H, 3), round(x1 / W, 3), round(y1 / H, 3)]
        out.append(vision._warn("picture_area", "info", f"{styles[-1]} frames the picture with a uniform border; "
                                "the painterly checks measure the picture inside it.", area, area, None,
                                "nothing to fix"))
    log = [r for r in (getattr(cv, "style_log", None) or []) if r.get("_before") is not None]
    if log:                                             # composition: judged on the picture before the style
        from PIL import Image
        comp = set(manifest.COMPOSITION_RULES)
        pre = vision.critique(Image.fromarray(log[0]["_before"], "RGB"))
        out = [w for w in out if w.get("rule") not in comp] + [w for w in pre if w.get("rule") in comp]
    out = manifest.style_relax(out, styles)
    blank = manifest.mostly_blank(img, getattr(cv, "style_log", None))
    if blank:
        out.insert(0, blank)
    for rec in getattr(cv, "style_log", None) or []:
        for n in rec.get("notes", []):
            out.append(vision._warn("style_option_clamped", "info", n + ".", None, rec["style"], None,
                                    f"use a value inside the range (stylize.options({rec['style']!r}))"))
    return _sorted(out), area


# words scenes use for KIND -> the critique kind they mean
KIND_WORDS = {"dashboard": "design", "poster": "design", "ui": "design", "screen": "design", "chart": "design",
              "infographic": "design", "flyer": "design", "card": "design", "banner": "design", "icon": "design",
              "icons": "design", "logo": "design", "thumbnail": "design", "menu": "design", "slide": "design",
              "cover": "design", "sticker": "design", "stickers": "design", "typography": "design",
              "illustration": "painting", "photo": "painting", "landscape": "painting", "portrait": "painting",
              "3d": "painting", "render": "painting", "scene": "painting", "still_life": "painting",
              "flowchart": "diagram", "architecture": "diagram", "figure": "diagram", "timeline": "diagram",
              "plan": "drawing", "floorplan": "drawing", "floor_plan": "drawing", "blueprint": "drawing",
              "engineering": "drawing", "technical": "drawing", "drafting": "drawing", "sheet": "drawing"}


def resolve_kind(value):
    """A scene's KIND -> one of KINDS (common words such as 'dashboard' or 'poster' map to their kind). Anything
    else raises UserError with the closest kinds."""
    if value is None:
        return None
    k = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if k in KINDS:
        return k
    if k in KIND_WORDS:
        return KIND_WORDS[k]
    from pixelcanvas._args import check_choice
    try:
        check_choice("KIND", "kind", k, list(KINDS) + sorted(KIND_WORDS))
    except ValueError as e:
        raise UserError(f"{e}\nKIND is one of {', '.join(KINDS)} (it decides which critique rules apply)")
    return None


def scene_kind(mod, cv):
    """(kind, source): the scene's KIND when given ('dashboard' and similar words map to their kind), else
    inferred from the manifest and the pixels."""
    from pixelcanvas import manifest
    k = getattr(mod, "KIND", None)
    if isinstance(k, str) and k.strip():
        try:
            r = resolve_kind(k)
        except UserError:
            r = None
        if r:
            return r, "KIND"
    return manifest.infer_kind(cv), "inferred"


def _prompt_ok_line(cv, prompt, rows):
    """What the PROMPT section checked and found in order (only the checks the request gives material for)."""
    from pixelcanvas import manifest
    done = manifest.prompt_status(cv, prompt, rows)
    if not done:
        return "no quoted text, named style or count in the request to compare"
    return "; ".join(done)


def text_block(cv, issues, labels, prompt, data=None):
    """Lines of the TEXT section: the issues; each quoted string of the request with drawn / missing; and,
    when there is no issue, what was audited (never a bare 'none' while text exists)."""
    from pixelcanvas import manifest, vision
    L = ["  " + vision.format_warning(w) for w in issues]
    qs = manifest.quoted_status(cv, prompt) if prompt else []
    if data is not None:
        data["quoted_text"] = [{"text": q, "drawn": ok, "closest": near} for q, ok, near in qs]
    for q, ok, near in qs:
        L.append(f"  quoted {q!r}: " + ("drawn" if ok else "missing" + (f" (closest drawn: {near[:40]!r})"
                                                                            if near else "")))
    n_draws = len(getattr(cv, "text_log", []) or [])
    if not issues:
        if n_draws or labels:
            L.append(f"  no problems in {n_draws + len(labels)} text item{'s' if n_draws + len(labels) != 1 else ''}"
                     " (size, contrast, clipping, overlap)")
        else:
            L.append("  no text drawn")
    if not qs and n_draws:                               # what the raster text says, to compare with the request
        seen = []
        for t in getattr(cv, "text_log", []) or []:
            s = " ".join(str(t.get("text", "")).split())
            if s and s not in seen:
                seen.append(s)
        shown = ", ".join(repr(s[:30]) for s in seen[:10])
        L.append(f"  drawn: {shown}" + (f" … +{len(seen) - 10} more" if len(seen) > 10 else ""))
    return L


_SPEC_FIX = {       # what to change, in the words of a JSON spec author (mode="spec"): no Python to write
    "prompt_colour": "change the colour of that layer in the spec (its color / palette value), or add the layer the "
                     "request names",
    "prompt_relation": "move the layer: change its at / position in the spec so it sits where the request puts it",
    "prompt_count": "add or remove layers in the spec until the number matches the request",
    "prompt_uncounted": "add a layer for each counted thing the request names",
}
_PY_ADVICE = re.compile(r"manifest\.note|cv\.\w+\(|look\.py|render\.py|stylize\.apply|new_scene|\.py\b")


def _spec_wording(issues):
    """The same issues with Python advice replaced by what a spec author edits (mode="spec")."""
    out = []
    for w in issues:
        fix = str(w.get("fix", ""))
        if _PY_ADVICE.search(fix):
            fix = _SPEC_FIX.get(w.get("rule"), "change the matching layer in the spec, then re-render")
            w = dict(w, fix=fix)
        out.append(w)
    return out


FIX_LINES = 12                                       # the FIX block, header included, never runs longer
_FIX_ORDER = ("prompt_text_missing", "prompt_unrecorded", "prompt_small", "prompt_cut", "prompt_count",
              "prompt_colour", "prompt_relation", "prompt_covered", "prompt_text_place", "prompt_text_small",
              "prompt_hidden", "prompt_crowded", "prompt_style_missing")
# a requested item the report cannot confirm (a checked request lists these first, in the request's order)
ITEM_RULES = {"prompt_text_missing", "prompt_unrecorded", "prompt_small", "prompt_cut", "prompt_count",
              "prompt_colour", "prompt_relation", "prompt_covered", "prompt_text_place", "prompt_text_small"}
# request issues a checked request keeps above the block as info (too often staging, not a fault)
_ITEM_SOFT = {"prompt_hidden", "prompt_crowded", "prompt_style_missing"}
# critiques of the picture itself: never in the FIX block of a checked request (kept above as info)
_CRITIQUE_OFF = {"object_at_edge", "subject_cropped", "object_tiny", "object_hidden"}
REMINDER = ("the report confirms places, counts, sizes and colours, not that a shape reads as the thing: look at "
            "each named item in the picture and ask whether a stranger would name it")


def _fix_line(w, mode="scene"):
    """One line of the FIX block for an issue: ``[warn] cup: painted green, the request says blue - ...``."""
    short = w.get("short")
    if short and not (mode == "spec" and _PY_ADVICE.search(short)):
        text = short
    else:
        msg = str(w.get("message", "")).rstrip(".")
        fix = re.split(r";|\(| or (?=[a-z]+ )", str(w.get("fix", "")), maxsplit=1)[0].strip().rstrip(",.")
        if mode == "spec" and _PY_ADVICE.search(fix):
            fix = ""
        text = f"{w['rule']}: {msg}" + (f" - {fix}" if fix else "")
    text = " ".join(text.split())
    if len(text) > 170:
        text = text[:167].rstrip() + "..."
    return f"[{w['severity']}] {text}"


def _fix_rank(w):
    r = w.get("rule", "")
    sev = _ORDER.get(w.get("severity"), 3)
    if r in _FIX_ORDER:
        grp = _FIX_ORDER.index(r)
    elif r.startswith("prompt_"):
        grp = len(_FIX_ORDER)
    elif r.startswith("text_"):
        grp = len(_FIX_ORDER) + 1
    elif r.startswith("object_") or r.startswith("connection"):
        grp = len(_FIX_ORDER) + 2
    else:
        grp = len(_FIX_ORDER) + 3                      # picture critique last
    return (sev, grp)


def _item_pos(w, prompt):
    """Where the item of a request issue first appears in the request (its order in the FIX block)."""
    low = (prompt or "").lower()
    item = w.get("item")
    if not item:
        m = re.match(r"\s*(?:the request names an? |not recorded: |the )?('[^']+'|[a-z][a-z -]*?)(?::| is | floats|"
                     r" takes|,| \()", str(w.get("short") or ""), re.I)
        item = m.group(1).strip("'") if m else ""
    item = item.lower()
    k = low.find(item) if item else -1
    if k < 0 and item:
        k = low.find(item.split()[-1][:-1] if item.endswith("s") else item.split()[-1])
    return k if k >= 0 else len(low)


def fix_block(issues, mode="scene", prompt=None, checked=False, round_=1):
    """(fix list, printed lines): every [fail] and [warn] issue, at most ``FIX_LINES`` lines with the header.
    A checked request lists the requested items it cannot confirm first, in the request's order; other requests
    order fails first, the request's things before text, objects and the picture critique."""
    todo = [w for w in issues if w.get("severity") in ("fail", "warn")]
    if checked:
        items = sorted([w for w in todo if w.get("rule") in ITEM_RULES],
                       key=lambda w: (w.get("rule") == "prompt_crowded", _item_pos(w, prompt)))
        rest = sorted([w for w in todo if w.get("rule") not in ITEM_RULES], key=_fix_rank)
        todo = items + rest
    else:
        items = []
        todo = sorted(todo, key=_fix_rank)
    fix = [_fix_line(w, mode) for w in todo]
    if not fix:
        return [], ["FIX: nothing flagged - look at the picture once, then deliver"]
    nf = sum(1 for w in todo if w["severity"] == "fail")
    head = f"FIX (worst first): {nf} fail, {len(todo) - nf} warn - change these, render once more, look"
    if checked and items:
        n = len(items)
        head = (f"FIX: {n} requested item{'s' if n != 1 else ''} still unconfirmed - fix these and render once more"
                if round_ >= 2 else
                f"FIX: {n} requested item{'s' if n != 1 else ''} unconfirmed - fix these, render once more, look")
    elif checked:
        head = "FIX: every requested item is confirmed - deliver (the lines below are optional)"
    room = FIX_LINES - 1
    lines = fix if len(fix) <= room else fix[:room - 1] + [f"... +{len(fix) - room + 1} more in the sections above"]
    return fix, [head] + lines


def _names_in(w, names):
    """True when an object issue (object_tiny / object_hidden) is about one of the recorded ``names``."""
    msg = str(w.get("message", ""))
    return any(msg.startswith(n + " (") or f" of {n} is visible" in msg for n in names)


def _unrecorded_items(cov, prompt, rows, words=None):
    """A checked request: one issue per confident thing (``manifest.confident_thing``) that nothing recorded, no
    label and no word of the scene answers to, in the request's order: "the request names a brass key: nothing
    recorded answers to it - draw it and pass name="key". When the scene records fewer things than the request
    names, one line instead: "the request names 6 things, 2 are recorded: ..."."""
    from pixelcanvas import manifest, vision
    records = any(r.get("kind") in ("object", "plant", "creature", "person") and not r.get("backdrop") for r in rows)
    sure = [c for c in cov if manifest.confident_thing(c["phrase"], prompt, rows)]
    miss = [c for c in sure if c["status"] == "missing" and not manifest.answered_by_words(c["phrase"], words or set())]
    rec = {r["name"].split(" #")[0].lower() for r in rows
           if r.get("kind") in ("object", "plant", "creature", "person") and not r.get("backdrop")}
    if len(miss) >= 2 and len(rec) < len(sure):
        shown = ", ".join(c["phrase"] for c in miss[:3]) + (f" +{len(miss) - 3} more" if len(miss) > 3 else "")
        w = vision._warn("prompt_unrecorded", "warn" if records else "info",
                         f"The request names {len(sure)} things and the scene records {len(rec)}; nothing answers to "
                         f"{shown}.", None, [c["phrase"] for c in miss], None,
                         "pass name= to the call that paints each thing (or manifest.note) so the report can check it")
        w["short"] = (f"the request names {len(sure)} things, {len(rec)} are recorded: record each with name= so they "
                      f"can be checked (not found: {shown})")
        w["item"] = miss[0]["phrase"]
        return [w]
    out = []
    for c in miss:
        if c["status"] != "missing":
            continue
        name = manifest._one(c["phrase"].split()[-1])
        mats = [st.split(" (")[0] for st in c.get("states", []) if st.split(" (")[0] in manifest._MATERIALS]
        what = " ".join(mats + [c["phrase"]])
        last = c["phrase"].split()[-1].lower()
        many = manifest._one(last) != last
        low = what.lower()
        art = "" if many else ("an " if low[:1] in "aeiou" and not re.match(r"u[bcdfgjklmnpqrstvwxz][aeiou]|eu|uni|one",
                                                                              low) else "a ")
        it = "them" if many else "it"
        w = vision._warn("prompt_unrecorded", "warn" if records else "info", f"The request names {art}{what}, and no "
                         f"recorded object answers to {it}.", None, c["phrase"], None,
                         f'draw {it} and pass name="{name}" to the call that paints {it} (or manifest.note)')
        w["short"] = (f'the request names {art}{what}: nothing recorded answers to {it} - draw {it} and pass '
                      f'name="{name}"')
        w["item"] = c["phrase"]
        out.append(w)
    return out


def _unrecorded_issue(cov, rows, prompt=None, words=None):
    """One issue naming every confident thing of the request that nothing recorded, no label and no word of the
    scene answers to: a warn when the scene records its things (so a missing one is likely missing), else info."""
    from pixelcanvas import manifest, vision
    miss = [c["phrase"] for c in cov if c["status"] == "missing" and manifest.confident_thing(c["phrase"], prompt, rows)
            and not manifest.answered_by_words(c["phrase"], words or set())]
    if not miss:
        return None
    records = any(r.get("kind") in ("object", "plant", "creature", "person") and not r.get("backdrop") for r in rows)
    sev = "warn" if records else "info"
    shown = ", ".join(miss[:5]) + (f" +{len(miss) - 5} more" if len(miss) > 5 else "")
    w = vision._warn("prompt_unrecorded", sev,
                     f"The request names {shown}, and no recorded object answers to "
                     f"{'it' if len(miss) == 1 else 'them'}.", None, miss, None,
                     "draw each one; when it is drawn by hand, pass name=... to the call that paints it (or "
                     "manifest.note) so the report can check its size, colour and place")
    w["short"] = (f"not recorded: {shown} - draw {'it' if len(miss) == 1 else 'them'}, or pass name= when "
                  "painting so the report can check")
    return w


def build_report(cv, mod, warns, sheet_path=None, image=None, preview=None, mode="scene", png=None, round_=1):
    """Everything --check prints, as (dict, text). ``image``: the saved PNG (critique of a styled picture);
    ``preview``: the preview JPG (a note when it is smaller than the picture); ``png``: the picture's path,
    printed (absolute) as the line before the FIX block. ``mode``: ``"scene"`` (default) for a scene file;
    ``"spec"`` for a JSON spec (tools/compose.py): advice names spec edits, never Python or look.py. The text
    ends with the FIX block: what to change next, worst first (``fix`` in the dict, same order)."""
    from pixelcanvas import manifest, vision
    kind, src = scene_kind(mod, cv)
    prompt = getattr(mod, "PROMPT", None)
    prompt = prompt if isinstance(prompt, str) and prompt.strip() else None
    warns, area = style_checks(cv, warns, image)
    warns = manifest.prompt_relax(manifest.relax(warns, kind), prompt)
    warns = manifest.horizon_relax(warns, cv.buf, kind, prompt)
    labels = manifest.vector_texts(cv)
    texts = _sorted(text_issues(cv) + vector_text_issues(cv, labels))
    rows = manifest.summary(cv)
    rel = manifest.relations(cv, rows=rows)
    obj = _sorted(manifest.object_checks(cv, rows))
    warns = manifest.reconcile_cropped(warns, rows, obj)
    conns = manifest.connections(cv)
    cissues = _sorted(manifest.connection_checks(cv))
    pr = _sorted(manifest.prompt_checks(cv, prompt, rows)) if prompt else []
    cov = manifest.coverage(cv, prompt, rows) if prompt else []
    if kind in ("diagram", "drawing"):                 # parts of a labelled figure are small by design
        pr = [w for w in pr if w["rule"] not in ("prompt_small", "prompt_hidden", "prompt_cut", "prompt_covered",
                                                 "prompt_crowded")]
    checked = bool(prompt) and manifest.checked_request(prompt)
    try:
        source = open(mod.__file__, encoding="utf-8").read() if getattr(mod, "__file__", None) else None
    except (OSError, UnicodeDecodeError):
        source = None
    words = manifest.scene_words(cv, rows, source) if prompt else set()
    if checked:
        pr = [dict(w, severity="info") if w["rule"] in _ITEM_SOFT and w.get("severity") == "warn" else w for w in pr]
        pr = _sorted(pr + _unrecorded_items(cov, prompt, rows, words))
        soft = []                                       # the picture critique stays above, as info
        for w in warns:
            soft.append(dict(w, severity="info") if w.get("severity") in ("warn", "fail") and
                        w.get("rule") not in ("mostly_blank",) else w)
        warns = _sorted(soft)
        obj = [dict(w, severity="info") if w["rule"] in _CRITIQUE_OFF and w.get("severity") == "warn" else w
               for w in obj]
    else:
        unrec = _unrecorded_issue(cov, rows, prompt, words) if prompt else None
        if unrec:
            pr = _sorted(pr + [unrec])
    named = {n for w in pr if w["rule"] in ("prompt_small", "prompt_hidden") for n in w.get("names", [])}
    if named:                                           # one line per thing: the request-sized one wins
        obj = [w for w in obj if not (w["rule"] in ("object_tiny", "object_hidden") and _names_in(w, named))]
    cols = manifest.colour_checks(cv, prompt, rows)[0] if prompt else []
    asked = manifest.request_summary(cv, prompt, rows) if prompt else None
    if mode == "spec":
        warns, texts, obj, cissues, pr = (_spec_wording(x) for x in (warns, texts, obj, cissues, pr))
    allw = warns + texts + obj + cissues + pr
    n = {s: sum(1 for w in allw if w["severity"] == s) for s in ("fail", "warn", "info")}
    data = {"kind": kind, "kind_source": src, "prompt": prompt, "warnings": warns, "text_issues": texts,
            "objects": rows, "relations": rel, "object_issues": obj, "prompt_issues": pr, "counts": n,
            "styles_applied": list(getattr(cv, "styles_applied", []) or []),
            "vector_labels": [{"text": l["text"], "drawing": l["drawing"], "group": l["group"],
                               "size_px": round(l["size"], 1),
                               "bbox": [round(l["bbox"][0] / cv.w, 4), round(l["bbox"][1] / cv.h, 4),
                                        round(l["bbox"][2] / cv.w, 4), round(l["bbox"][3] / cv.h, 4)]}
                              for l in labels],
            "connections": [{k: c[k] for k in ("from", "to", "label", "style", "arrow")} for c in conns],
            "connection_issues": cissues, "coverage": cov, "colours": cols,
            "as_requested": asked}
    if area:
        data["picture_area"] = area
    fix, fix_lines = fix_block(allw, mode, prompt=prompt, checked=checked, round_=round_)
    data["fix"] = fix
    data["checked"] = checked

    def fmt(w):                                         # an issue with a FIX line: the change is printed there
        if w.get("short") and w.get("severity") in ("fail", "warn"):
            return f"[{w['severity']}] {w['rule']}: {w['message']}  (-> FIX)"
        return vision.format_warning(w)
    L = [f"check: {sheet_path}   kind: {kind} ({'from KIND' if src == 'KIND' else 'inferred; set KIND = ... to change'})"
         if sheet_path else f"kind: {kind} ({src})"]
    if preview and cv.w > 960:
        L.append(f"preview is 960 px wide (picture is {cv.w}x{cv.h}): zoom into details with look.py zoom")
    L.append("WARNINGS (image critique)")
    L += ["  " + fmt(w) for w in warns] or ["  none"]
    n_draws = len(getattr(cv, "text_log", []) or [])
    head = f"{n_draws} text draw{'s' if n_draws != 1 else ''}"
    if labels:
        head += f" + {_label_summary(labels)}"
    L.append(f"TEXT (text audit of {head})")
    L += text_block(cv, texts, labels, prompt, data)
    if labels and not manifest.is_text_only(rows):
        lab_rows = [{"kind": "text", "name": l["text"], "bbox": [l["bbox"][0] / cv.w, l["bbox"][1] / cv.h,
                                                                  l["bbox"][2] / cv.w, l["bbox"][3] / cv.h],
                     "size": [0, (l["bbox"][3] - l["bbox"][1]) / cv.h],
                     "centre": [(l["bbox"][0] + l["bbox"][2]) / 2 / cv.w, (l["bbox"][1] + l["bbox"][3]) / 2 / cv.h]}
                    for l in labels]
        order = manifest.reading_order(lab_rows, (cv.w, cv.h), limit=10)
        data["label_reading_order"] = order
        L.append("  labels in reading order (top to bottom: text, position, height):")
        L += ["    " + t for t in order]
    L.append(f"OBJECTS ({len(rows)} recorded; centre / size are fractions of the canvas; exact, from cv.manifest)")
    L += ["  " + fmt(w) for w in obj]
    fits = list(getattr(cv, "fit_reports", []) or [])
    if fits:
        data["fit_reports"] = fits
        L += ["  sheet scale: " + f for f in fits]
    if manifest.is_text_only(rows):                     # typography only: reading order instead of table + relations
        order = manifest.reading_order(rows, (cv.w, cv.h))
        data["reading_order"] = order
        L.append("  text only; reading order (top to bottom: text, position, height):")
        L += ["    " + t for t in order]
    else:
        L += ["  " + t for t in manifest.table(rows)]
        if rel:
            L.append("  relations:")
            L += ["    - " + r for r in rel]
    if conns or cissues:
        L.append(f"CONNECTIONS ({len(conns)} edge{'s' if len(conns) != 1 else ''}; from -> to, dashed = ⇢)")
        L += ["  " + fmt(w) for w in cissues]
        cap = 16
        L += ["  " + c["text"] for c in conns[:cap]]
        if len(conns) > cap:
            L.append(f"  … +{len(conns) - cap} more (all of them are in --json)")
    if prompt:
        L.append(f"PROMPT {prompt[:100]!r}")
        L += ["  " + fmt(w) for w in pr]
        L.append("  " + _prompt_ok_line(cv, prompt, rows))
        if cov:
            L.append("  COVERAGE (subjects of the request and what answers to them):")
            L += ["    " + t for t in manifest.coverage_lines(cov)]
        L += ["    [info] " + t for t in manifest.state_lines(cov)]
        if cols and not (asked and asked["lines"]):
            L.append("  COLOURS (each colour the request names, against the colour painted inside that object):")
            L += ["    " + c["line"] for c in cols[:12]]
            if len(cols) > 12:
                L.append(f"    … +{len(cols) - 12} more (all of them are in --json)")
        if asked and asked["lines"]:
            L.append("  AS REQUESTED (count, colour, place and text the request spells out, against the picture; "
                     "✗ names what to change):")
            L += ["    " + t for t in asked["lines"][:16]]
    if checked:
        L.append(REMINDER)
    if png:
        L.append(f"picture: {os.path.abspath(png)}")
    L += fix_lines
    return data, "\n".join(L)


def write_svg(scene, folder):
    """SVG(s) of the scene's vector drawings -> list of paths."""
    import sys as _sys
    vec = _sys.modules.get("pixelcanvas.vector")
    drawings = vec.last_drawings() if vec else []
    if not drawings:
        return []
    def covers(d):
        info = getattr(d, "_pc_drawn", None)
        if not info:
            return False
        box, (W, H) = info
        return box is None or tuple(int(round(v)) for v in box) == (0, 0, W, H)

    if len(drawings) > 1:
        cover = [d for d in drawings if covers(d)]
        if len(cover) == 1:
            drawings = cover
    if len(drawings) == 1:
        p = os.path.join(folder, f"{scene}.svg")
        drawings[0].save_svg(p)
        return [p]
    out = []
    for i, d in enumerate(drawings, 1):
        p = os.path.join(folder, f"{scene}_{i}.svg")
        d.save_svg(p)
        out.append(p)
    return out


def write_dxf(scene, folder):
    import sys as _sys
    dr = _sys.modules.get("pixelcanvas.drafting")
    sheets = dr.last_sheets() if dr else []
    out = []
    for i, sh in enumerate(sheets, 1):
        p = os.path.join(folder, f"{scene}.dxf" if len(sheets) == 1 else f"{scene}_{i}.dxf")
        sh.save_dxf(p)
        out.append(p)
    return out


def _style_name(name):
    from pixelcanvas import stylize
    if name in stylize.STYLES:
        return name
    close = difflib.get_close_matches(name, list(stylize.STYLES), n=3, cutoff=0.5)
    picked = stylize.pick(name)
    raise UserError(f"unknown style {name!r}." + (f" Did you mean: {', '.join(close)}?" if close else
                    (f" Closest by keyword: {picked}." if picked else ""))
                    + f"\nstyles: {', '.join(stylize.STYLES)}")


def parse_animate(mod, tokens):
    """['T','0','1'] | ['0','1'] | ['A:0:1,B:2:3'] -> [(NAME, start, end)] (floats; start / end may be expressions)."""
    toks = list(tokens)
    triples = []
    if all(":" in t for t in toks):
        for t in toks:
            for part in t.split(","):
                bits = part.split(":")
                if len(bits) != 3:
                    raise UserError(f"--animate expects NAME:START:END, got {part!r}")
                triples.append(tuple(b.strip() for b in bits))
    elif len(toks) == 3:
        triples.append(tuple(toks))
    elif len(toks) == 2:
        if not hasattr(mod, "T"):
            raise UserError("--animate START END needs a constant named T in the scene; use --animate NAME START END "
                            f"(constants: {', '.join(constants(mod)) or 'none'})")
        triples.append(("T", toks[0], toks[1]))
    else:
        raise UserError('--animate NAME START END   (e.g. --animate T 0 1, or --animate "SUN_X:0.2:0.8,SUN_Y:0.7:0.3")')
    out = []
    for name, a, b in triples:
        if not hasattr(mod, name):
            apply_override(mod, f"{name}=0")              # raises the helpful "unknown name" error
        try:
            fa, fb = float(_eval(a, vars(mod))), float(_eval(b, vars(mod)))
        except (TypeError, ValueError):
            raise UserError(f"--animate {name}: START and END must be numbers, got {a!r} and {b!r}")
        out.append((name, fa, fb))
    return out


def frame_values(specs, n, ease="linear", loop=False):
    """[(t, {NAME: value})] for ``n`` frames. ``loop`` leaves out the END frame (it equals START in a cycle)."""
    n = max(2, int(n))
    out = []
    for i in range(n):
        t = i / n if loop else i / (n - 1)
        e = t * t * (3 - 2 * t) if ease == "inout" else t
        out.append((t, {name: a + (b - a) * e for name, a, b in specs}))
    return out


def animation_sequence(frames, bounce=False, hold=0):
    """Frame order of the file: the frames, then (bounce) the reversed ones without repeating either end,
    then the last frame ``hold`` more times."""
    seq = list(frames)
    if bounce and len(frames) > 2:
        seq += frames[-2:0:-1]
    if hold > 0:
        seq += [seq[-1]] * hold
    return seq


def run_animation(args, mod, rec, w, h, t0):
    """--animate: one render per frame -> GIF / WebP / APNG + a labelled filmstrip contact sheet."""
    import look
    from pixelcanvas import manifest
    specs = parse_animate(mod, args.animate)
    n = max(2, args.frames)
    plan = frame_values(specs, n, args.ease, args.loop)
    ext = args.format or (os.path.splitext(args.out)[1].lstrip(".").lower() if args.out else "") or "gif"
    fmt = {"gif": "gif", "webp": "webp", "apng": "apng", "png": "apng"}.get(ext, "gif")
    out = args.out or os.path.join(args.outdir, f"{args.scene}.{fmt}")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    originals = {name: copy.deepcopy(getattr(mod, name)) for name, _, _ in specs}
    ends = {name: (a, b) for name, a, b in specs}
    quiet = lambda m: None                              # noqa: E731
    frames, labels = [], []
    names = ", ".join(f"{k}: {a:g} -> {b:g}" for k, a, b in specs)
    print(f"animate {args.scene}  {n} frames  {names}  {w}x{h}", flush=True)
    try:
        for i, (t, vals) in enumerate(plan):
            for name, v in vals.items():
                orig = originals[name]
                if isinstance(orig, int) and not isinstance(orig, bool) and all(float(x).is_integer() for x in ends[name]):
                    v = int(round(v))
                    vals[name] = v
                setattr(mod, name, v)
            manifest.reset()
            cv, _ = render_one(mod, w, h, args.seed, quiet)
            if args.style:
                from pixelcanvas import stylize
                stylize.apply(cv, args.style, seed=rec["seed"] or 0)
            frames.append(cv.to_pil().convert("RGB"))
            first = next(iter(vals.items()))
            labels.append(f"{i + 1}/{n}  {first[0]}={first[1]:.3g}" + (f" +{len(vals) - 1}" if len(vals) > 1 else ""))
            print(f"[{time.time() - t0:5.1f}s] frame {i + 1}/{n}  " + ", ".join(f"{k}={v:.3g}" for k, v in vals.items()), flush=True)
    finally:
        for name, v in originals.items():
            setattr(mod, name, v)
    seq = animation_sequence(frames, args.bounce, args.hold)
    info = look.save_animation(seq, out, fps=args.fps, fmt=fmt, dither=args.dither)
    pick = sorted({round(k * (n - 1) / 7) for k in range(8)}) if n > 8 else list(range(n))
    strip = look.frames_sheet([frames[i] for i in pick], [labels[i] for i in pick],
                              title=f"{args.scene}  {w}x{h}  {len(seq)} frames @ {args.fps:g} fps  ({names})")
    strip_path = os.path.splitext(out)[0] + "_frames.jpg"
    strip.save(strip_path, quality=88)
    secs = time.time() - t0
    rec["outputs"]["animation"] = _rel(out)
    rec["outputs"]["filmstrip"] = _rel(strip_path)
    rec["animation"] = {"format": fmt, "frames": len(seq), "rendered": n, "fps": args.fps,
                        "duration_s": round(len(seq) / args.fps, 2), "bytes": info["bytes"],
                        "constants": {k: [a, b] for k, a, b in specs}, "bounce": bool(args.bounce), "hold": args.hold,
                        "loop": bool(args.loop), "ease": args.ease}
    rec["warnings"] = []
    mb = info["bytes"] / 1e6
    print(f"[{secs:5.1f}s] {fmt:5s}  {_rel(out)}   {len(seq)} frames, {len(seq) / args.fps:.1f} s, {mb:.2f} MB")
    print(f"[{secs:5.1f}s] sheet   {_rel(strip_path)}   <- look at this: do neighbouring frames change smoothly, "
          "does the last frame lead back into the first?", flush=True)
    if args.deliver:
        _deliver(args, rec, [out])
    if info["bytes"] > 8e6:
        wn = {"rule": "anim_size", "severity": "warn", "message": f"the animation is {mb:.1f} MB (above ~8 MB)",
              "where": None, "value": info["bytes"], "threshold": 8e6,
              "fix": "lower --scale, use fewer --frames, or --format webp"}
        rec["warnings"].append(wn)
        print(f"warning: {wn['message']}; {wn['fix']}", flush=True)
    return 0


def run(args, rec):
    """Do the work. ``rec`` collects outputs / warnings / steps for the JSON line and the trace."""
    from pixelcanvas import vision
    if args.list or not args.scene:
        print("scenes:", ", ".join(list_scenes()))
        mine = workspace.list_scenes(ROOT, include_root=False)
        if mine:
            print("workspace scenes:", ", ".join(f"{n} ({_rel(p)})" for n, p in mine.items()))
        return 0

    global SCENE_DIR
    path = workspace.resolve_scene(args.scene, ROOT)
    if path is None:
        names = sorted(set(list_scenes()) | set(workspace.list_scenes(ROOT)))
        key = os.path.splitext(os.path.basename(args.scene))[0]
        close = difflib.get_close_matches(key, names, n=3, cutoff=0.4)
        where = "file not found" if workspace.looks_like_path(args.scene) else \
            "searched " + ", ".join(_rel(d) for d in workspace.scene_dirs(ROOT))
        raise UserError(f"unknown scene {args.scene!r} ({where})." + (f" Closest matches: {', '.join(close)}." if close else "")
                        + f"\navailable scenes: {', '.join(names)}\nnew scene: python {_rel(os.path.join(ROOT, 'tools', 'new_scene.py'))} "
                          f"{key} --kind KIND [--dir DIR]")
    args.scene = os.path.splitext(os.path.basename(path))[0]
    SCENE_DIR = os.path.dirname(path)
    args.outdir = workspace.output_dir(path, ROOT, args.outdir)
    rec["scene"] = args.scene
    rec["scene_file"] = path
    mod = workspace.import_scene(path, ROOT)
    for s in args.set:
        apply_override(mod, s)
    if args.prompt:
        mod.PROMPT = args.prompt
    if args.kind:
        mod.KIND = args.kind
    if isinstance(getattr(mod, "KIND", None), str) and mod.KIND.strip():
        resolve_kind(mod.KIND)                        # a misspelt KIND stops here with the closest kinds

    multi = bool(args.sweep or args.seeds)
    scale = args.scale if args.scale is not None else (0.33 if multi else 0.5 if args.animate else 1.0)
    if not hasattr(mod, "DEFAULT_SIZE"):
        print(f"note: {mod.__name__.split('.')[-1]} defines no DEFAULT_SIZE; using 1920x1080 (add DEFAULT_SIZE = (w, h) to the scene)", file=sys.stderr)
    w, h = getattr(mod, "DEFAULT_SIZE", (1920, 1080))
    if args.size:
        try:
            w, h = (int(v) for v in args.size.lower().split("x"))
        except ValueError:
            raise UserError(f"--size expects WxH such as 1920x1080, got {args.size!r}")
    w, h = int(round(w * scale)), int(round(h * scale))
    os.makedirs(args.outdir, exist_ok=True)
    rec.update(size=[w, h], scale=scale, seed=_scene_seed(mod, args.seed))

    from pixelcanvas import manifest
    if args.style:
        args.style = _style_name(args.style)
    style_list = [_style_name(s.strip()) for s in args.styles.split(",") if s.strip()] if args.styles else []
    for modname, fn in (("pixelcanvas.vector", "clear_drawings"), ("pixelcanvas.drafting", "clear_sheets")):
        if args.svg and modname.endswith("vector"):
            importlib.import_module(modname)
        if args.dxf and modname.endswith("drafting"):
            importlib.import_module(modname)
        if modname in sys.modules and hasattr(sys.modules[modname], fn):
            getattr(sys.modules[modname], fn)()
    manifest.reset()

    t0 = time.time()
    steps = []                                          # [name, time at which the step began]

    def log(m):
        t = time.time() - t0
        steps.append([m, t])
        manifest.step(m)
        print(f"[{t:5.1f}s] {m}", flush=True)

    outputs = rec["outputs"]

    def finish_steps(t_end):
        out = []
        for i, (name, t) in enumerate(steps):
            nxt = steps[i + 1][1] if i + 1 < len(steps) else t_end
            out.append({"name": name, "seconds": round(max(0.0, nxt - t), 3)})
        return out

    if args.animate:
        return run_animation(args, mod, rec, w, h, t0)

    if multi:
        quiet = lambda m: None                          # noqa: E731
        ims, labels = [], []
        if args.seeds:
            for sd in args.seeds:
                cv, _ = render_one(mod, w, h, sd, quiet)
                ims.append(cv.to_pil()); labels.append(f"seed {sd}")
                print(f"[{time.time() - t0:5.1f}s] seed {sd} done", flush=True)
            name = f"{args.scene}_seeds"
        else:
            key, *values = args.sweep
            if not values:
                raise UserError("--sweep NAME EXPR [EXPR ...]  (e.g. --sweep SUN_X 0.3 0.5 0.7)")
            for v in values:
                root, orig = apply_override(mod, f"{key}={v}")
                try:
                    cv, _ = render_one(mod, w, h, args.seed, quiet)
                finally:
                    setattr(mod, root, orig)
                ims.append(cv.to_pil()); labels.append(f"{key} = {v}")
                print(f"[{time.time() - t0:5.1f}s] {key}={v} done", flush=True)
            name = f"{args.scene}_sweep_{re.sub(r'[^A-Za-z0-9_]+', '_', key).strip('_')}"
        out = args.out or os.path.join(args.outdir, f"{name}.jpg")
        sheet = vision.contact_sheet(ims, labels, tile_width=min(640, ims[0].width), title=f"{args.scene}  {w}x{h}")
        sheet.save(out, quality=90)
        outputs["sheet"] = _rel(out)
        rec["variants"] = labels
        print(f"[{time.time() - t0:5.1f}s] sheet   {_rel(out)}   <- look at this and pick a variant", flush=True)
        if args.deliver:
            _deliver(args, rec, [out])
        return 0

    res = render_one(mod, w, h, args.seed, log, stages=args.stages, layers=args.layers)
    cv, shots = res[0], res[1]
    if args.style:
        from pixelcanvas import stylize
        log(f"style   {args.style}")
        stylize.apply(cv, args.style, seed=rec["seed"] or 0)
    manifest.step(None)
    t_render = time.time() - t0
    rec["steps"] = finish_steps(t_render)

    tag = "" if scale == 1 else f"_x{scale:g}"
    out = args.out or os.path.join(args.outdir, f"{args.scene}{tag}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    base = os.path.splitext(out)[0]                     # deliverables (layers, svg) sit next to the picture
    # the preview and the check sheet stay in the work folder, so --check --out <delivery path> writes exactly
    # that PNG there (--stages / --styles / --crop / --layers / --svg, when asked for, go next to it)
    diag = os.path.join(args.outdir, os.path.basename(base)) if args.out else base
    preview = diag + "_preview.jpg"
    cv.save(out, preview=preview)
    picture = cv.to_pil()                               # the saved pixels, reused by every check (no re-read)
    cv.stats()
    outputs["png"], outputs["preview"] = _rel(out), _rel(preview)
    log(f"saved   {_rel(out)}")
    log(f"preview {_rel(preview)}   <- look at this")
    if args.crop:
        from PIL import Image
        im = Image.open(out)
        outputs["crops"] = []
        for x0, y0, x1, y1 in args.crop:
            p = f"{base}_crop_{x0}_{y0}_{x1}_{y1}.jpg"
            im.crop((x0, y0, x1, y1)).save(p, quality=90)
            outputs["crops"].append(_rel(p))
            log(f"crop    {_rel(p)}")
    if shots:
        sheet = vision.contact_sheet([im for _, im in shots], [f"{i + 1}. {lab}" for i, (lab, _) in enumerate(shots)],
                                     tile_width=min(480, w), title=f"{args.scene}: canvas after each step")
        sheet.save(base + "_stages.jpg", quality=88)
        outputs["stages"] = _rel(base + "_stages.jpg")
        log(f"stages  {_rel(base + '_stages.jpg')}   ({len(shots)} steps)")
    elif args.stages:
        log("stages: none captured (the scene never called log(...))")
    if args.layers:
        from pixelcanvas import layers as _layers
        stack = res[2]
        paths = _layers.save_all(stack, base, final=cv.buf)
        names_ = ", ".join(L.name for L in stack)
        outputs["layers"] = _rel(base + "_layers")
        for p in paths:
            if p.lower().endswith(".psd"):
                outputs["psd"] = _rel(p)
            elif p.lower().endswith(".ora"):
                outputs["ora"] = _rel(p)
        log(f"layers  {len(stack)} (bottom->top): {names_}")
        for p in paths:
            log(f"        {_rel(p)}")

    folder = os.path.dirname(os.path.abspath(out))
    if style_list:
        from pixelcanvas import stylize
        ims = [cv.to_pil()]
        for s in style_list:
            c2 = cv.copy()
            stylize.apply(c2, s, seed=rec["seed"] or 0)
            ims.append(c2.to_pil())
            print(f"[{time.time() - t0:5.1f}s] style {s} done", flush=True)
        sheet = vision.contact_sheet(ims, ["original"] + style_list, tile_width=min(640, w),
                                     title=f"{args.scene} in {len(style_list)} styles")
        sheet.save(base + "_styles.jpg", quality=90)
        outputs["styles"] = _rel(base + "_styles.jpg")
        log(f"styles  {_rel(base + '_styles.jpg')}   <- pick one, then --style NAME or stylize.apply(cv, NAME)")
    if args.svg:
        paths = write_svg(args.scene, folder)
        if paths:
            outputs["svg"] = [_rel(p) for p in paths] if len(paths) > 1 else _rel(paths[0])
            for p in paths:
                log(f"svg     {_rel(p)}")
        else:
            log("svg     none: the scene drew no vector Drawing (diagram, chart, drafting, vector kits draw them)")
    if args.dxf:
        paths = write_dxf(args.scene, folder)
        if paths:
            outputs["dxf"] = [_rel(p) for p in paths] if len(paths) > 1 else _rel(paths[0])
            for p in paths:
                log(f"dxf     {_rel(p)}")
        else:
            log("dxf     none: the scene drew no drafting Sheet (sheet.draw(cv) records it)")

    want_round = args.round or os.environ.get("PIXELCANVAS_ROUNDS", "").lower() in ("1", "true", "yes", "on")
    need_warnings = args.check or args.json or want_round or bool(os.environ.get("PIXELCANVAS_TRACE"))
    warns = []
    sheet_path = None
    if args.check:
        import look
        sheet_path, warns = look.check_sheet(out, diag + "_check.jpg")
        outputs["check"] = _rel(sheet_path)
    elif need_warnings:
        warns = vision.critique(picture)
    report = None
    if need_warnings:
        data, report = build_report(cv, mod, warns, _rel(sheet_path) if sheet_path else None, image=picture,
                                    preview=preview, png=out, round_=_check_round(args) if args.check else 1)
        report = __import__("pixelcanvas.reference", fromlist=["add_to_report"]).add_to_report(
            report, args.ref or getattr(mod, "REFERENCE", None), out, scene=getattr(mod, "__file__", None))
        warns = data["warnings"]
        rec.update({k: v for k, v in data.items() if k != "warnings"})
    rec["warnings"] = warns
    if want_round:
        import look
        import shutil
        folder = look.rounds_dir(args.scene) if os.environ.get("PIXELCANVAS_ROUNDS_DIR") else \
            os.path.join(args.outdir, "rounds", args.scene)
        os.makedirs(folder, exist_ok=True)
        n, rpath = _next_round(folder)
        shutil.copyfile(preview, rpath)
        entry = {"round": n, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "scale": scale, "seed": rec["seed"],
                 "set": list(args.set), "seconds": round(t_render, 2), "warnings": look.summary_counts(warns)}
        with open(os.path.join(folder, "rounds.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        outputs["round"] = _rel(rpath)
        log(f"round   {_rel(rpath)}   (round {n}; `python tools/look.py rounds {args.scene}` shows them all)")
    try:
        stats = vision.image_stats(picture)
        stats.update(getattr(cv, "last_stats", None) or {})
    except Exception:
        stats = {}
    rec["stats"] = stats
    if args.deliver:
        extra = []
        for key in ("svg", "psd", "ora"):
            v = outputs.get(key)
            extra += v if isinstance(v, list) else [v] if v else []
        _deliver(args, rec, [out] + extra)
    if args.check and report:
        print(report, flush=True)                       # last: the report ends with the FIX block
    return 0


def _check_round(args):
    """How many times this scene was rendered with --check (kept in <outdir>/.check/<scene>.rounds): the FIX
    header says "still unconfirmed" from the second render on."""
    f = os.path.join(args.outdir, ".check", f"{args.scene}.rounds")
    try:
        n = int(open(f, encoding="utf-8").read().strip() or 0) + 1
    except (OSError, ValueError):
        n = 1
    try:
        os.makedirs(os.path.dirname(f), exist_ok=True)
        with open(f, "w", encoding="utf-8") as fh:
            fh.write(str(n))
    except OSError:
        pass
    return n


def _deliver(args, rec, files):
    """--deliver: copy the results to the requested place (same stem, original extensions)."""
    files = [f if os.path.isabs(f) else os.path.abspath(f) for f in files if f]
    paths = workspace.deliver(files, args.deliver, args.scene)
    rec["outputs"]["delivered"] = [os.path.abspath(p) for p in paths]
    for p in paths:
        print(f"deliver {os.path.abspath(p)}", flush=True)
    prev = rec["outputs"].get("preview")
    if prev:
        print(f"preview {os.path.abspath(prev)}   (look at this before you hand the picture over)", flush=True)


def main(argv=None):
    args = build_parser().parse_args(argv)
    t_start = time.time()
    rec = {"scene": args.scene, "size": None, "scale": args.scale, "seed": args.seed, "seconds": None, "outputs": {},
           "steps": [], "warnings": [], "stats": {}}
    code, err = 0, None
    try:
        code = run(args, rec)
    except UserError as e:
        err = {"type": "UserError", "message": str(e)}
        print(f"ERROR {e}", file=sys.stderr)
        code = 2
    except KeyboardInterrupt:
        err = {"type": "KeyboardInterrupt", "message": "interrupted"}
        print("interrupted", file=sys.stderr)
        code = 130
    except BaseException as e:                         # noqa: BLE001 - scene code can raise anything
        if isinstance(e, SystemExit):
            raise
        err, text = failure_report(e, e.__traceback__, args.scene)
        print(text, file=sys.stderr, flush=True)
        code = 1
    rec["seconds"] = round(time.time() - t_start, 3)
    if os.environ.get("PIXELCANVAS_TRACE"):
        from pixelcanvas import vision
        a = {k: v for k, v in vars(args).items() if v not in (None, False, [])}
        vision.trace_event("render", "render", a, list(rec["outputs"].values()) if rec["outputs"] else [],
                           rec["seconds"], rec["warnings"], scene=args.scene, ok=code == 0, error=err)
    if args.json and not (args.list or not args.scene):
        doc = dict(rec)
        if err:
            doc["error"] = err
        print(json.dumps(doc, ensure_ascii=False, separators=(",", ":"), default=str))
    return code


if __name__ == "__main__":
    sys.exit(main())
