# -*- coding: utf-8 -*-
"""The ``pixelcanvas`` command and the workspace rules shared by every tool.

  pixelcanvas new NAME --kind poster [--dir DIR]     create a scene from a template (tools/new_scene.py)
  pixelcanvas render SCENE --scale 0.5 --check       render a scene by name or path (tools/render.py)
  pixelcanvas look | layers | styles | fonts | kit   the other tools/<name>.py (any tool that exists)
  pixelcanvas doctor [--json]                        check Python, numpy, Pillow, fonts, write access, a test render
  pixelcanvas mcp                                    run the stdio MCP server (pixelcanvas.mcp_server)
  pixelcanvas version                                version and runtime root
  pixelcanvas info IMAGE [--json] / preview IMAGE SMALL.jpg [WIDTH]

Workspace rules (used by render.py, new_scene.py, look.py, styles.py and the MCP server):

* the runtime root is the checkout that holds this package (``<root>/tools``, ``<root>/scenes``) or, in an
  installed wheel, ``pixelcanvas/_runtime`` inside the package;
* a scene is a path (``./art/poster.py``) or a bare name, looked up in ./, ./scenes, $PIXELCANVAS_WORK and
  <root>/scenes, in that order;
* outputs go next to the scene: ``<dir>/outputs`` (``<project>/outputs`` for a scene in ``<project>/scenes``), or
  to ``--outdir`` / $PIXELCANVAS_OUT.
"""
from __future__ import annotations

import json
import os
import re
import sys

PKG_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------------------------- runtime root


def _is_root(path):
    return bool(path) and os.path.isfile(os.path.join(path, "tools", "render.py")) \
        and os.path.isdir(os.path.join(path, "scenes"))


def runtime_root():
    """Folder that holds tools/ and scenes/: the source checkout, else the copy bundled in the wheel."""
    for cand in (os.path.dirname(PKG_DIR), os.path.join(PKG_DIR, "_runtime"), os.environ.get("PIXELCANVAS_HOME")):
        if cand and _is_root(os.path.expanduser(cand)):
            return os.path.abspath(os.path.expanduser(cand))
    return None


def is_checkout(root):
    """True when ``root`` is a source checkout (not the read-only copy inside an installed package)."""
    return bool(root) and os.path.isfile(os.path.join(root, "pixelcanvas", "__init__.py"))


def _same(a, b):
    try:
        return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))
    except (OSError, ValueError):
        return False


def _inside(path, folder):
    a = os.path.normcase(os.path.realpath(path))
    b = os.path.normcase(os.path.realpath(folder))
    return a == b or a.startswith(b.rstrip("\\/") + os.sep)


# --------------------------------------------------------------------------------------------- scenes


def scene_dirs(root=None):
    """Folders searched for a bare scene name, in order: ./, ./scenes, $PIXELCANVAS_WORK, <root>/scenes."""
    root = root or runtime_root()
    cwd = os.getcwd()
    out = [cwd, os.path.join(cwd, "scenes")]
    work = os.environ.get("PIXELCANVAS_WORK")
    if work:
        out.append(os.path.abspath(os.path.expanduser(work)))
    if root:
        out.append(os.path.join(root, "scenes"))
    seen, uniq = set(), []
    for d in out:
        key = os.path.normcase(os.path.realpath(d))
        if key not in seen and os.path.isdir(d):
            seen.add(key)
            uniq.append(d)
    return uniq


def looks_like_path(spec):
    return any(c in spec for c in "/\\") or spec.endswith(".py") or spec.startswith(".")


def _is_scene_file(path):
    """A .py file that defines render(...) (cheap text test, nothing is imported)."""
    if not (path.endswith(".py") and os.path.isfile(path)):
        return False
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return re.search(r"^def render\s*\(", f.read(), re.M) is not None
    except OSError:
        return False


def resolve_scene(spec, root=None):
    """Scene name or path -> absolute path of the scene file, or None.

    A path (``./art/poster.py``, ``art/poster``, ``scenes/x``) is taken as given (``.py`` may be left out); a bare
    name is looked up in ``scene_dirs()``."""
    if not spec:
        return None
    spec = os.path.expanduser(str(spec))
    if looks_like_path(spec) or os.path.isabs(spec):
        for cand in (spec, spec + ".py"):
            if os.path.isfile(cand) and cand.endswith(".py"):
                return os.path.abspath(cand)
        base = os.path.splitext(os.path.basename(spec.rstrip("/\\")))[0]
        if spec.replace("\\", "/").startswith("scenes/") and base:     # "scenes/<name>" = the name
            return resolve_scene(base, root)
        return None
    name = spec[:-3] if spec.endswith(".py") else spec
    for d in scene_dirs(root):
        p = os.path.join(d, name + ".py")
        if os.path.isfile(p) and (os.path.basename(d) == "scenes" or _same(d, os.environ.get("PIXELCANVAS_WORK") or "\0")
                                  or _is_scene_file(p)):
            return os.path.abspath(p)
    return None


def list_scenes(root=None, include_root=True):
    """{name: path} of every scene the lookup can reach (first match wins, like ``resolve_scene``)."""
    root = root or runtime_root()
    out = {}
    for d in scene_dirs(root):
        is_root_dir = root and _same(d, os.path.join(root, "scenes"))
        if is_root_dir and not include_root:
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".py") or f.startswith("_"):
                continue
            p = os.path.join(d, f)
            if f[:-3] not in out and (is_root_dir or _is_scene_file(p)):
                out[f[:-3]] = p
    return out


def is_builtin_scene(path, root=None):
    root = root or runtime_root()
    return bool(root) and _same(os.path.dirname(os.path.abspath(path)), os.path.join(root, "scenes"))


def _writable(folder):
    probe = folder
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            return False
        probe = parent
    return os.access(probe, os.W_OK)


def output_dir(scene_path=None, root=None, outdir=None):
    """Where a tool writes its results.

    ``outdir`` (``--outdir``) > $PIXELCANVAS_OUT > next to the scene: ``<dir>/outputs``, or ``<project>/outputs``
    for a scene in ``<project>/scenes`` (the checkout's own scenes write to ``<root>/outputs`` as always). Without a
    scene: ``<root>/outputs`` when working inside the checkout, else ``./outputs``."""
    if outdir:
        return os.path.abspath(os.path.expanduser(outdir))
    env = os.environ.get("PIXELCANVAS_OUT")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    root = root or runtime_root()
    cwd_out = os.path.join(os.getcwd(), "outputs")
    if scene_path:
        d = os.path.dirname(os.path.abspath(scene_path))
        if is_builtin_scene(scene_path, root):
            out = os.path.join(root, "outputs")
            return out if is_checkout(root) and _writable(out) else cwd_out
        if os.path.basename(d) == "scenes":
            return os.path.join(os.path.dirname(d), "outputs")
        return os.path.join(d, "outputs")
    if root and is_checkout(root) and _inside(os.getcwd(), root):
        return os.path.join(root, "outputs")
    return cwd_out


def default_scene_dir(root=None):
    """Where ``new_scene.py`` writes when no --dir is given: $PIXELCANVAS_WORK, the checkout's scenes/ when working
    inside the checkout, else the current folder."""
    work = os.environ.get("PIXELCANVAS_WORK")
    if work:
        return os.path.abspath(os.path.expanduser(work))
    root = root or runtime_root()
    if root and is_checkout(root) and _inside(os.getcwd(), root):
        return os.path.join(root, "scenes")
    return os.getcwd()


def _scenes_package(extra_dirs, root):
    """Make ``import scenes.<other>`` work for scenes inside and outside the runtime (one shared package)."""
    import importlib
    import types
    if root and root not in sys.path:
        sys.path.insert(0, root)
    pkg = sys.modules.get("scenes")
    if pkg is None:
        try:
            pkg = importlib.import_module("scenes")
        except ImportError:
            pkg = None
        if pkg is None or not hasattr(pkg, "__path__"):
            pkg = types.ModuleType("scenes")
            pkg.__path__ = []
            sys.modules["scenes"] = pkg
    want = [os.path.abspath(d) for d in extra_dirs if d and os.path.isdir(d)]
    have = [str(p) for p in list(pkg.__path__)]
    if root:
        want.append(os.path.join(root, "scenes"))
    new = []
    for p in want + have:
        if not any(_same(p, q) for q in new):
            new.append(p)
    pkg.__path__ = new
    return pkg


def import_scene(path, root=None):
    """Import a scene file as ``scenes.<stem>``; its folder joins sys.path and the ``scenes`` package path, so the
    scene may ``import pixelcanvas``, its own helpers, and ``scenes.<other>``."""
    import importlib
    import importlib.util
    root = root or runtime_root()
    path = os.path.abspath(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    folder = os.path.dirname(path)
    importlib.invalidate_caches()
    if is_builtin_scene(path, root):
        _scenes_package([], root)
        return importlib.import_module(f"scenes.{stem}")
    if folder not in sys.path:
        sys.path.insert(0, folder)
    _scenes_package([folder] + scene_dirs(root)[1:], root)
    name = f"scenes.{stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    old = sys.modules.get(name)
    sys.modules[name] = mod
    no_pyc = sys.dont_write_bytecode
    sys.dont_write_bytecode = True                       # no __pycache__ folders in the user's project
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        if old is not None:
            sys.modules[name] = old
        else:
            sys.modules.pop(name, None)
        raise
    finally:
        sys.dont_write_bytecode = no_pyc
    return mod


def deliver(files, dest, stem):
    """Copy result files to ``dest`` (a folder, or a file path whose stem names the copies). ``files`` is a list of
    paths; every copy keeps its extension and takes the delivery stem (``_1``, ``_2`` ... when an extension
    repeats). Returns the written paths."""
    import shutil
    dest = os.path.abspath(os.path.expanduser(dest))
    if dest.endswith(("/", "\\")) or os.path.isdir(dest) or not os.path.splitext(dest)[1]:
        folder, base = dest.rstrip("/\\"), stem
    else:
        folder, base = os.path.dirname(dest), os.path.splitext(os.path.basename(dest))[0]
    os.makedirs(folder, exist_ok=True)
    by_ext = {}
    for f in files:
        by_ext.setdefault(os.path.splitext(f)[1].lower(), []).append(f)
    out = []
    for ext, group in by_ext.items():
        for i, f in enumerate(group, 1):
            tgt = os.path.join(folder, f"{base}{ext}" if len(group) == 1 else f"{base}_{i}{ext}")
            if not _same(f, tgt):
                shutil.copyfile(f, tgt)
            out.append(tgt)
    return out


# --------------------------------------------------------------------------------------------- tools

ALIASES = {"new": "new_scene", "install-skill": "install_skill", "install": "install_skill", "font": "fonts",
           "style": "styles", "layer": "layers"}
_HIDDEN = {"pc", "release_check"}                     # entry shim and the maintainers' release check


def tools_dir(root=None):
    root = root or runtime_root()
    return os.path.join(root, "tools") if root else None


def available_tools(root=None):
    """{command: path} for every tools/<name>.py (discovered, so new tools join automatically)."""
    d = tools_dir(root)
    out = {}
    if d and os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".py") and not f.startswith("_") and f[:-3] not in _HIDDEN:
                out[f[:-3]] = os.path.join(d, f)
    return out


def tool_path(name, root=None):
    tools = available_tools(root)
    name = name.replace("-", "_") if name.replace("-", "_") in tools else name
    return tools.get(ALIASES.get(name, name))


def run_tool(name, argv, root=None):
    """Run tools/<name>.py in this process as if it were started from the command line; returns its exit code."""
    import runpy
    path = tool_path(name, root)
    if path is None:
        raise KeyError(name)
    tdir = os.path.dirname(path)
    old_argv, old_path = sys.argv[:], sys.path[:]
    sys.argv = [path] + list(argv)
    sys.path[:0] = [tdir, os.path.dirname(tdir)]
    try:
        runpy.run_path(path, run_name="__main__")
        return 0
    except SystemExit as e:
        code = e.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(code, file=sys.stderr)
        return 1
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path


# --------------------------------------------------------------------------------------------- doctor


def _version_of(mod):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "?")
    except Exception:                                   # noqa: BLE001
        return None


def doctor_report(root=None):
    """List of checks: {name, status ok|warn|fail, detail}."""
    import platform
    import tempfile
    import time
    root = root or runtime_root()
    rows = []

    def add(name, status, detail):
        rows.append({"name": name, "status": status, "detail": detail})

    py_ok = sys.version_info >= (3, 9)
    add("python", "ok" if py_ok else "fail", f"{platform.python_version()} ({sys.executable})"
        + ("" if py_ok else "; pixelcanvas needs Python 3.9 or newer"))
    for mod, label, need in (("numpy", "numpy", True), ("PIL", "Pillow", True), ("scipy", "scipy", False)):
        v = _version_of(mod)
        if v:
            add(label, "ok", v)
        else:
            add(label, "fail" if need else "info",
                f"missing: pip install {label}" if need else "not installed (optional speed-up: pip install scipy)")
    from . import __version__
    add("pixelcanvas", "ok", f"{__version__} ({PKG_DIR})")
    if root:
        n_tools = len(available_tools(root))
        n_kinds = len([f for f in os.listdir(os.path.join(root, "scenes")) if f.startswith("_template_")])
        add("runtime", "ok", f"{root} ({'checkout' if is_checkout(root) else 'installed package'}; "
                             f"{n_tools} tools, {n_kinds} templates)")
    else:
        add("runtime", "fail", "tools/ and scenes/ not found: reinstall the package or clone the repository")
    try:
        from . import text as T
        fonts = T.list_fonts()
        sans = T.find_font()
        add("fonts", "ok" if fonts else "warn",
            f"{len(fonts)} installed; default {os.path.basename(str(sans)) if sans else 'Pillow built-in'}"
            + ("" if fonts else "; install a TrueType font (e.g. Noto Sans) for good text"))
        sample = "漢字 日本語 한국어"
        missing = T.missing_glyphs(sample)
        cjk = T.find_font(text="漢字")
        if missing:
            add("CJK fonts", "warn", f"no glyphs for {''.join(sorted(set(missing)))!r}: install Noto Sans CJK "
                                      "(apt install fonts-noto-cjk) or run: pixelcanvas fonts --help")
        else:
            add("CJK fonts", "ok", f"Chinese / Japanese / Korean covered ({os.path.basename(str(cjk)) if cjk else 'fallback chain'})")
    except Exception as e:                              # noqa: BLE001
        add("fonts", "warn", f"font scan failed: {type(e).__name__}: {e}")
    out = output_dir(root=root)
    add("write access", "ok" if _writable(out) else "fail", f"outputs go to {out}"
        + ("" if _writable(out) else " (not writable: pass --outdir or set PIXELCANVAS_OUT)"))
    vlm = os.environ.get("PIXELCANVAS_VLM", "").strip()
    add("vision model", "ok" if vlm else "info",
        f"PIXELCANVAS_VLM={vlm}" if vlm else
        "not configured (optional, for look.py ask: PIXELCANVAS_VLM=ollama:<model> or openai:<model>@<base_url>)")
    try:
        from .canvas import Canvas
        t0 = time.time()
        cv = Canvas(480, 270)
        cv.gradient([(0, "#0b1a3a"), (1, "#f6c28f")])
        cv.glow(330, 90, 40, "#fff0c0", strength=1.2)
        cv.text("pixelcanvas", (240, 170), 40, "#ffffff", anchor="mm")
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "selftest.png")
            cv.soft_clip(0.8).save(p)
            ok = os.path.getsize(p) > 1000
        dt = time.time() - t0
        add("self-test render", "ok" if ok and dt < 10 else "warn", f"480x270 gradient + glow + text in {dt:.2f} s")
    except Exception as e:                              # noqa: BLE001
        add("self-test render", "fail", f"{type(e).__name__}: {e}")
    return rows


def cmd_doctor(argv):
    rows = doctor_report()
    if "--json" in argv:
        print(json.dumps({"ok": not any(r["status"] == "fail" for r in rows), "checks": rows}, ensure_ascii=False))
    else:
        w = max(len(r["name"]) for r in rows)
        for r in rows:
            print(f"{r['status']:>4}  {r['name']:<{w}}  {r['detail']}")
        fails = sum(r["status"] == "fail" for r in rows)
        print("all set: pixelcanvas new my_picture --kind poster" if not fails else f"{fails} problem(s) to fix above")
    return 1 if any(r["status"] == "fail" for r in rows) else 0


# --------------------------------------------------------------------------------------------- main


def _usage():
    names = ", ".join(sorted(set(available_tools()) | {"new"}))
    return f"""usage: pixelcanvas <command> [args]

  kit find "REQUEST"                     which kits and calls draw a request (kit list shows every kit)
  new NAME --prompt "REQUEST"            create a scene for a request (--kind KIND picks a template; --list)
  render SCENE [--scale 0.5] [--check]   render a scene (name or path); --deliver PATH copies the result
  compose SPEC.json --check              draw from a JSON spec (compose --example poster writes one)
  look VIEW IMAGE ...                    inspect a picture (guides, zoom, values, critique, report, ask ...)
  match REF.png [DRAFT.png]              a reference picture's style sheet; with a draft, its gaps to the reference
  styles list                            the art styles (styles pick "REQUEST" chooses one)
  install-skill [claude|agents|codex|cursor|opencode]   give the skills to an agent
  doctor [--json]                        check the installation
  mcp                                    run the MCP server on stdin / stdout
  version                                print the version and the runtime root
  info IMAGE [--json]                    size and brightness statistics
  preview IMAGE SMALL.jpg [WIDTH]        write a downscaled preview (default width 960)

tools: {names}   (pixelcanvas <tool> --help for each)"""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    for s in (sys.stdout, sys.stderr):                 # Windows consoles default to a legacy code page
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0 if argv else 1
    cmd, rest = argv[0], argv[1:]
    if cmd in ("version", "--version", "-V"):
        from . import __version__
        print(f"pixelcanvas {__version__}")
        print(f"root {runtime_root()}")
        return 0
    if cmd == "doctor":
        return cmd_doctor(rest)
    if cmd == "mcp":
        from . import mcp_server
        return mcp_server.main(rest)
    if cmd in ("info", "preview"):
        return _image_cmd(cmd, rest)
    if tool_path(cmd) is None:
        print(f"unknown command {cmd!r}\n\n{_usage()}", file=sys.stderr)
        return 2
    prog = sys.argv[0] if sys.argv else ""
    if prog.endswith("pc.py"):
        prog = os.path.abspath(prog).replace("\\", "/")
        os.environ.setdefault("PIXELCANVAS_CMD", f'python "{prog}"' if " " in prog else f"python {prog}")
    else:
        os.environ.setdefault("PIXELCANVAS_CMD", "pixelcanvas")
    return run_tool(cmd, rest)


def _image_cmd(cmd, rest):
    from .canvas import Canvas
    as_json = "--json" in rest
    rest = [a for a in rest if a != "--json"]
    if not rest:
        print(_usage())
        return 1
    cv = Canvas.load(rest[0])
    if cmd == "info":
        if as_json:
            from . import vision
            print(json.dumps({**vision.image_stats(cv), "file": rest[0]}))
        else:
            cv.stats()
    else:
        out = rest[1] if len(rest) > 1 else "preview.jpg"
        pw = int(rest[2]) if len(rest) > 2 else 960
        cv.resize(pw, max(1, round(cv.h * pw / cv.w))).save(out)
        print("saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
