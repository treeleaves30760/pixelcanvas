# -*- coding: utf-8 -*-
"""pixelcanvas as a Model Context Protocol server (stdio, standard library only).

  python -m pixelcanvas.mcp_server          (also: pixelcanvas mcp, python <root>/tools/pc.py mcp)

Speaks JSON-RPC 2.0, one message per line on stdin / stdout. Both protocol eras are served:

* modern (2026-07-28): every request carries ``_meta["io.modelcontextprotocol/protocolVersion"]``;
  ``server/discover`` lists the supported versions; results carry ``resultType``;
* legacy (2025-11-25, 2025-06-18, 2025-03-26, 2024-11-05): ``initialize`` handshake; a client asking for an unknown
  version gets the newest legacy version back.

Methods: server/discover, initialize, ping, tools/list, tools/call, resources/list, resources/read,
resources/templates/list, prompts/list. Scenes are created and rendered in the server's working directory
(or $PIXELCANVAS_WORK); every tool runs the matching tools/*.py in a child process, so a failing scene comes back as
a readable tool error and nothing but protocol messages reaches stdout.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import subprocess
import sys
import time

MODERN = ("2026-07-28",)
LEGACY = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
SUPPORTED = MODERN + LEGACY
META_VERSION = "io.modelcontextprotocol/protocolVersion"
SERVER_NAME = "pixelcanvas"
MAX_SIDE = 1024
TIMEOUT = float(os.environ.get("PIXELCANVAS_MCP_TIMEOUT", "900"))

INSTRUCTIONS = """pixelcanvas draws pictures with code (numpy + Pillow): posters, illustrations, landscapes, diagrams,
charts, UI mock-ups, technical drawings, 3D objects, typography in any language.
The loop: new_scene(name, kind) -> edit the file (read_scene / write_scene) -> render(scene) and look at the
preview image -> fix the 2-4 worst problems -> render again. Use kinds() to pick a template, find("...") to find
the right kit call, reference("api") for the API. Every render returns the --check report (WARNINGS / TEXT /
OBJECTS / PROMPT) as text and the preview as an image."""

# ----------------------------------------------------------------------------------------------- helpers


def _version():
    try:
        from . import __version__
        return __version__
    except Exception:                                            # noqa: BLE001
        return "0"


def _root():
    from .cli import runtime_root
    return runtime_root()


def _workdir():
    work = os.environ.get("PIXELCANVAS_WORK")
    return os.path.abspath(os.path.expanduser(work)) if work else os.getcwd()


class ToolError(Exception):
    """A problem the model should read (bad argument, failed render): returned as an isError tool result."""


def _run(tool, args, timeout=None):
    """Run tools/<tool>.py with ``args`` in a child process from the work folder -> (code, stdout, stderr)."""
    from .cli import tool_path
    path = tool_path(tool, _root())
    if path is None:
        raise ToolError(f"tool {tool!r} is not available in this installation")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = pkg_parent + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    try:
        r = subprocess.run([sys.executable, path] + [str(a) for a in args], cwd=_workdir(), env=env,
                           stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout or TIMEOUT)
    except subprocess.TimeoutExpired:
        raise ToolError(f"{tool} did not finish within {int(timeout or TIMEOUT)} s; render a smaller draft "
                        "(scale 0.5 or 0.33) or simplify the scene")
    return r.returncode, r.stdout, r.stderr


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(_workdir(), p)


def _image_block(path, max_side=MAX_SIDE):
    """A JPEG image content block (longest side <= max_side) of the picture at ``path``, or None."""
    try:
        from PIL import Image
        im = Image.open(_abs(path))
        im.load()
    except Exception:                                            # noqa: BLE001
        return None
    im = im.convert("RGB")
    s = max(im.size)
    if s > max_side:
        k = max_side / s
        im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))),
                       getattr(Image, "Resampling", Image).LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return {"type": "image", "data": base64.b64encode(buf.getvalue()).decode("ascii"), "mimeType": "image/jpeg"}


def _text(t):
    return {"type": "text", "text": t}


def _split_json(stdout):
    """render.py --json prints one JSON object as the last line: (text without it, parsed object or None)."""
    lines = stdout.rstrip().splitlines()
    for i in range(len(lines) - 1, -1, -1):
        ln = lines[i].strip()
        if ln.startswith("{") and ln.endswith("}"):
            try:
                return "\n".join(lines[:i] + lines[i + 1:]), json.loads(ln)
            except ValueError:
                break
    return stdout, None


def _scene_name(name):
    name = str(name or "").strip()
    if name.endswith(".py"):
        name = name[:-3]
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ToolError(f"scene name {name!r}: use lower-case letters, digits and _ (e.g. night_market)")
    return name


def _scene_path(scene):
    from .cli import resolve_scene, scene_dirs
    cwd = os.getcwd()
    try:
        os.chdir(_workdir())
        p = resolve_scene(str(scene), _root())
        dirs = scene_dirs(_root())
    finally:
        os.chdir(cwd)
    if p is None:
        raise ToolError(f"scene {scene!r} not found (searched {', '.join(dirs)}); create it with new_scene")
    return p


def _clip(t, n=12000):
    t = t.strip()
    return t if len(t) <= n else t[:n // 2] + "\n... (output shortened) ...\n" + t[-n // 2:]


# ----------------------------------------------------------------------------------------------- tools

def t_kinds(a):
    code, out, err = _run("new_scene", ["--list"])
    if code:
        raise ToolError(err or out)
    return [_text(out.strip() + "\n\nCreate one with new_scene(name, kind).")]


def t_new_scene(a):
    name = _scene_name(a.get("name"))
    args = [name, "--kind", a.get("kind") or "landscape"]
    if a.get("prompt"):
        args += ["--prompt", a["prompt"]]
    if a.get("size"):
        args += ["--size", a["size"]]
    folder = _abs(a["dir"]) if a.get("dir") else _workdir()
    args += ["--dir", folder]
    code, out, err = _run("new_scene", args)
    if code:
        raise ToolError((err or out).strip())
    path = os.path.join(folder, name + ".py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    return [_text(f"{out.strip()}\n\nfile: {path}\n\n```python\n{src}\n```\n"
                  "Change the plan docstring, PROMPT and the UPPER_CASE tunables, then write_scene() the new code "
                  f"and render(\"{name}\").")]


def t_write_scene(a):
    name = _scene_name(a.get("name"))
    code_src = a.get("code")
    if not isinstance(code_src, str) or not code_src.strip():
        raise ToolError("code is empty: pass the full Python source of the scene")
    folder = _abs(a["dir"]) if a.get("dir") else os.path.dirname(_existing(name)) if _existing(name) else _workdir()
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name + ".py")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(code_src if code_src.endswith("\n") else code_src + "\n")
    msg = f"wrote {path} ({len(code_src.splitlines())} lines)"
    try:
        compile(code_src, path, "exec")
    except SyntaxError as e:
        raise ToolError(f"{msg}, but it does not compile: line {e.lineno}: {e.msg}\n  {(e.text or '').rstrip()}")
    if not re.search(r"^def render\s*\(", code_src, re.M):
        msg += "\nnote: a scene needs def render(width, height, seed=0, log=print) returning a Canvas"
    return [_text(msg + f"\nnext: render(\"{name}\")")]


def _existing(name):
    try:
        return _scene_path(name)
    except ToolError:
        return None


def t_read_scene(a):
    path = _scene_path(a.get("name"))
    with open(path, encoding="utf-8") as f:
        return [_text(f"file: {path}\n\n```python\n{f.read()}\n```")]


def t_render(a):
    path = _scene_path(a.get("scene"))
    args = [path, "--json", "--scale", str(a.get("scale", 0.5))]
    if a.get("check", True):
        args.append("--check")
    for flag in ("layers", "svg"):
        if a.get(flag):
            args.append("--" + flag)
    if a.get("style"):
        args += ["--style", a["style"]]
    for s in a.get("set") or []:
        args += ["--set", s]
    if a.get("seed") is not None:
        args += ["--seed", int(a["seed"])]
    if a.get("deliver"):
        args += ["--deliver", _abs(a["deliver"])]
    t0 = time.time()
    code, out, err = _run("render", args)
    text, doc = _split_json(out)
    if code:
        raise ToolError(_clip((err.strip() + "\n" + text.strip()).strip()))
    outs = (doc or {}).get("outputs", {})
    lines = [ln for ln in text.splitlines() if not re.match(r"\[\s*\d+\.\d+s\] (?!saved|preview|svg|layers|deliver)", ln)]
    head = f"rendered {os.path.basename(path)} in {time.time() - t0:.1f} s -> {outs.get('png', '?')}"
    content = [_text(_clip(head + "\n" + "\n".join(lines)))]
    img = _image_block(outs.get("preview") or outs.get("png") or "")
    if img:
        content.append(img)
    return content


LOOK_VIEWS = ("guides", "zoom", "values", "edges", "grid", "probe", "diff", "sheet", "critique", "report", "check",
              "stats", "compare")


def t_look(a):
    view = a.get("view") or "guides"
    if view not in LOOK_VIEWS:
        raise ToolError(f"unknown view {view!r}; use one of {', '.join(LOOK_VIEWS)}")
    image = a.get("image")
    if not image:
        raise ToolError("image is required (a path such as outputs/poster_x0.5.png)")
    if not os.path.exists(_abs(image)):
        raise ToolError(f"image not found: {image}")
    extra = [str(x) for x in (a.get("args") or [])]
    code, out, err = _run("look", [view, _abs(image)] + extra)
    if code:
        raise ToolError(_clip((err or out).strip()))
    saved = re.findall(r"^saved (.+?)\s*$", out + "\n" + err, re.M)
    text = "\n".join(ln for ln in out.splitlines() if not ln.startswith("saved ")).strip()
    content = [_text(_clip(text or ("wrote " + ", ".join(saved) if saved else "done")))]
    for p in saved[:4]:
        img = _image_block(p)
        if img:
            content.append(img)
    return content


def _references():
    root = _root()
    d = os.path.join(root, "skills", "pixelcanvas", "references") if root else ""
    pages = {}
    if d and os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".md"):
                pages[f[:-3]] = os.path.join(d, f)
    skill = os.path.join(root, "skills", "pixelcanvas", "SKILL.md") if root else ""
    if skill and os.path.isfile(skill):
        pages["SKILL"] = skill
    return pages


def t_find(a):
    q = str(a.get("query") or "").strip()
    if not q:
        raise ToolError("query is required, e.g. find(\"campfire\")")
    from .cli import tool_path
    if tool_path("kit", _root()):
        code, out, err = _run("kit", ["find", q])
        if code == 0 and out.strip():
            return [_text(_clip(out))]
    words = [w for w in re.findall(r"\w+", q.lower()) if len(w) > 1]
    hits = []
    for page, p in _references().items():
        with open(p, encoding="utf-8") as f:
            for i, ln in enumerate(f, 1):
                low = ln.lower()
                score = sum(w in low for w in words)
                if score:
                    hits.append((score, page, i, ln.strip()))
    hits.sort(key=lambda h: (-h[0], h[1], h[2]))
    if not hits:
        return [_text(f"nothing matches {q!r}. Pages: {', '.join(_references())}; read one with reference(page).")]
    rows = [f"{page}.md:{i}: {ln[:160]}" for _, page, i, ln in hits[:25]]
    return [_text(f"matches for {q!r} in the reference pages (read a page with reference(page)):\n" + "\n".join(rows))]


def t_reference(a):
    pages = _references()
    page = str(a.get("page") or "").strip()
    key = page[:-3] if page.endswith(".md") else page
    key = {"skill": "SKILL", "": "SKILL"}.get(key.lower(), key)
    if key not in pages:
        lower = {k.lower(): k for k in pages}
        key = lower.get(key.lower(), key)
    if key not in pages:
        raise ToolError(f"no reference page {page!r}; pages: {', '.join(pages)}")
    with open(pages[key], encoding="utf-8") as f:
        return [_text(f.read())]


def t_ask(a):
    image, q = a.get("image"), a.get("question")
    if not image or not q:
        raise ToolError("image and question are required")
    code, out, err = _run("look", ["ask", _abs(image), q])
    if code:
        raise ToolError(_clip((err or out).strip()))
    return [_text(out.strip())]


def t_animate(a):
    path = _scene_path(a.get("scene"))
    name = a.get("name") or "T"
    args = [path, "--json", "--animate", name, str(a.get("start", 0)), str(a.get("end", 1)),
            "--frames", str(int(a.get("frames", 24))), "--fps", str(a.get("fps", 12)),
            "--scale", str(a.get("scale", 0.5))]
    code, out, err = _run("render", args)
    text, doc = _split_json(out)
    if code:
        raise ToolError(_clip((err.strip() + "\n" + text.strip()).strip()))
    outs = (doc or {}).get("outputs", {})
    content = [_text(_clip(text))]
    img = _image_block(outs.get("filmstrip") or "")
    if img:
        content.append(img)
    return content


def t_layers(a):
    scene = a.get("scene")
    folder = _abs(scene) if scene and os.path.isdir(_abs(scene)) else None
    log = ""
    if folder is None:
        path = _scene_path(scene)
        code, out, err = _run("render", [path, "--json", "--layers", "--scale", str(a.get("scale", 0.5))])
        text, doc = _split_json(out)
        if code:
            raise ToolError(_clip((err.strip() + "\n" + text.strip()).strip()))
        folder = _abs(((doc or {}).get("outputs") or {}).get("layers", ""))
        log = "\n".join(ln for ln in text.splitlines() if "layers" in ln or ".psd" in ln or ".ora" in ln)
    code, info, err = _run("layers", ["info", folder])
    if code:
        raise ToolError(_clip((err or info).strip()))
    content = [_text(_clip(f"{log}\nfolder: {folder}\n{info}".strip()))]
    hide = [str(h) for h in (a.get("hide") or [])]
    if hide:
        out_png = folder.rstrip("/\\") + "_hidden.png"
        code, out, err = _run("layers", ["compose", folder, "--hide"] + hide + ["--out", out_png])
        if code:
            raise ToolError(_clip((err or out).strip()))
        content.append(_text(f"composite without {', '.join(hide)}: {out_png}"))
        img = _image_block(out_png)
    else:
        code, out, err = _run("layers", ["sheet", folder])
        img = _image_block(out.strip().splitlines()[-1]) if code == 0 and out.strip() else None
    if img:
        content.append(img)
    return content


S = lambda d, **kw: dict({"type": "string", "description": d}, **kw)            # noqa: E731
N = lambda d, **kw: dict({"type": "number", "description": d}, **kw)            # noqa: E731
B = lambda d, **kw: dict({"type": "boolean", "description": d}, **kw)           # noqa: E731


def _schema(props, required=()):
    return {"type": "object", "properties": props, "required": list(required), "additionalProperties": False}


TOOLS = [
    ("kinds", "List the starter templates (kinds) with one line each: poster, landscape, diagram, chart, ui, "
              "isometric, product ... Pick the closest before new_scene.", _schema({}), t_kinds),
    ("new_scene", "Create a scene file from a template and return its source. The template already renders a "
                  "finished picture; replace its content, keep its structure.",
     _schema({"name": S("file name: lower-case letters, digits and _ (e.g. night_market)"),
              "kind": S("template kind from kinds(), e.g. poster, landscape, diagram (default landscape)"),
              "prompt": S("the user's request, verbatim; stored as PROMPT and checked by render"),
              "size": S("canvas size WxH, e.g. 1080x1350 (default: the template's)"),
              "dir": S("folder for the file, relative to the work folder (default: the work folder)")},
             ["name"]), t_new_scene),
    ("write_scene", "Save the full Python source of a scene (replaces the file). The scene defines DEFAULT_SIZE, "
                    "UPPER_CASE tunables and render(width, height, seed=0, log=print) -> Canvas.",
     _schema({"name": S("scene name (file name without .py)"), "code": S("complete Python source of the scene"),
              "dir": S("folder for the file (default: where the scene already is, else the work folder)")},
             ["name", "code"]), t_write_scene),
    ("read_scene", "Return the source of a scene (by name or path).",
     _schema({"name": S("scene name or path")}, ["name"]), t_read_scene),
    ("render", "Render a scene and return the check report (WARNINGS / TEXT / OBJECTS / PROMPT) plus the preview "
               "image. Draft at scale 0.5 while iterating; fix the [fail] and worst [warn] lines, then render again.",
     _schema({"scene": S("scene name or path"),
              "scale": N("resolution factor: 0.5 = draft (default), 1 = full size", minimum=0.05, maximum=4),
              "check": B("include the diagnostic report (default true)"),
              "layers": B("also write editable layers (PSD / ORA / PNG stack), one per log() step"),
              "svg": B("also write an SVG of the vector drawings"),
              "style": S("apply a style after rendering, e.g. watercolor, oil, ukiyoe, risograph"),
              "set": {"type": "array", "items": {"type": "string"},
                      "description": "override scene constants without editing the file, e.g. [\"SUN_X=0.7\"]"},
              "seed": {"type": "integer", "description": "random seed (default: the scene's)"},
              "deliver": S("also copy the final PNG (and SVG / PSD / ORA) to this file or folder")},
             ["scene"]), t_render),
    ("look", "Inspect a rendered picture. Views: guides (coordinate grid), zoom (args: u0 v0 u1 v1 as fractions), "
             "values (squint test), edges, grid (3x3 tiles), probe (args: 'u,v' points -> exact colours), "
             "diff (args: second image), sheet (args: more images), critique / report / stats (text).",
     _schema({"image": S("path of the picture, e.g. outputs/poster_x0.5.png"),
              "view": S("one of: " + ", ".join(LOOK_VIEWS), enum=list(LOOK_VIEWS)),
              "args": {"type": "array", "items": {"type": "string"},
                       "description": "extra arguments of the view, e.g. [\"0.4\", \"0.3\", \"0.6\", \"0.5\"] for zoom"}},
             ["image", "view"]), t_look),
    ("find", "Search the kit catalogue for the call that draws a subject (\"campfire\", \"bar chart\", \"oak tree\", "
             "\"isometric box\"). Returns calls with their reference page.",
     _schema({"query": S("what you want to draw or do")}, ["query"]), t_find),
    ("reference", "Read a reference page: SKILL (the workflow), api, techniques, typography, review, diagrams, "
                  "charts, solid, styles, textures, ui ... (the error lists all pages).",
     _schema({"page": S("page name, e.g. api or typography")}, ["page"]), t_reference),
    ("ask", "Ask the configured vision model (PIXELCANVAS_VLM) a question about a picture, e.g. \"does this read "
            "as a fox?\". Exact text, counts and positions come from render's report instead.",
     _schema({"image": S("path of the picture"), "question": S("the question")}, ["image", "question"]), t_ask),
    ("animate", "Render an animation by sweeping one UPPER_CASE constant from start to end: writes a GIF and "
                "returns a filmstrip of frames.",
     _schema({"scene": S("scene name or path"), "name": S("constant to sweep (default T)"),
              "start": N("first value (default 0)"), "end": N("last value (default 1)"),
              "frames": {"type": "integer", "description": "number of frames (default 24)", "minimum": 2,
                         "maximum": 240},
              "fps": N("frames per second (default 12)"), "scale": N("resolution factor (default 0.5)")},
             ["scene"]), t_animate),
    ("layers", "Render a scene as editable layers (one per log() step) and show every layer; with hide, show the "
               "picture with those layers switched off.",
     _schema({"scene": S("scene name or path, or an existing <out>_layers folder"),
              "hide": {"type": "array", "items": {"type": "string"}, "description": "layer names to hide"},
              "scale": N("resolution factor (default 0.5)")}, ["scene"]), t_layers),
]
TOOL_FN = {name: fn for name, _, _, fn in TOOLS}


def tools_list():
    return [{"name": n, "description": d, "inputSchema": s} for n, d, s, _ in TOOLS]


def call_tool(name, args):
    if name not in TOOL_FN:
        raise KeyError(name)
    if not isinstance(args, dict):
        args = {}
    try:
        return {"content": TOOL_FN[name](args), "isError": False}
    except ToolError as e:
        return {"content": [_text(str(e))], "isError": True}
    except Exception as e:                                        # noqa: BLE001 - never crash the server
        return {"content": [_text(f"{name} failed: {type(e).__name__}: {e}")], "isError": True}


def resources_list():
    out = []
    for page, p in _references().items():
        uri = "pixelcanvas://skill/SKILL.md" if page == "SKILL" else f"pixelcanvas://references/{page}.md"
        desc = "the pixelcanvas workflow (read first)" if page == "SKILL" else f"reference page: {page}"
        out.append({"uri": uri, "name": page, "title": f"pixelcanvas {page}", "description": desc,
                    "mimeType": "text/markdown"})
    return out


def resource_read(uri):
    m = re.fullmatch(r"pixelcanvas://(?:skill/SKILL|references/([\w-]+))\.md", str(uri or ""))
    pages = _references()
    key = "SKILL" if m and m.group(1) is None else (m.group(1) if m else None)
    if key not in pages:
        return None
    with open(pages[key], encoding="utf-8") as f:
        return {"contents": [{"uri": uri, "mimeType": "text/markdown", "text": f.read()}]}


# ----------------------------------------------------------------------------------------------- protocol

class Server:
    def __init__(self):
        self.legacy_version = None                                 # set by initialize

    def capabilities(self):
        return {"tools": {"listChanged": False}, "resources": {"subscribe": False, "listChanged": False}}

    def info(self):
        return {"name": SERVER_NAME, "title": "pixelcanvas", "version": _version()}

    def handle(self, msg):
        """One parsed JSON-RPC message -> response dict, or None for notifications."""
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
            if isinstance(msg, dict) and ("result" in msg or "error" in msg):
                return None                                           # a response to us: nothing to do
            return _error(msg.get("id") if isinstance(msg, dict) else None, -32600, "Invalid Request")
        mid, method = msg.get("id"), msg["method"]
        params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
        if "id" not in msg:                                           # notification
            return None
        meta = params.get("_meta") if isinstance(params.get("_meta"), dict) else {}
        version = meta.get(META_VERSION)
        modern = version is not None and method != "initialize"
        if modern and version not in MODERN:
            if version in LEGACY:                                     # legacy version label on a modern-style call
                modern = False
            else:
                return _error(mid, -32022, "Unsupported protocol version",
                              {"supported": list(SUPPORTED), "requested": version})
        try:
            result = self.dispatch(method, params, modern)
        except _RpcError as e:
            return _error(mid, e.code, e.message, e.data)
        except Exception as e:                                        # noqa: BLE001
            return _error(mid, -32603, f"Internal error: {type(e).__name__}: {e}")
        if modern:
            result = dict(result)
            result.setdefault("resultType", "complete")
            result.setdefault("_meta", {})["io.modelcontextprotocol/serverInfo"] = self.info()
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def dispatch(self, method, params, modern):
        if method == "initialize":
            want = params.get("protocolVersion")
            self.legacy_version = want if want in LEGACY else LEGACY[0]
            return {"protocolVersion": self.legacy_version, "capabilities": self.capabilities(),
                    "serverInfo": self.info(), "instructions": INSTRUCTIONS}
        if method == "server/discover":
            return {"supportedVersions": list(SUPPORTED), "capabilities": self.capabilities(),
                    "serverInfo": self.info(), "instructions": INSTRUCTIONS}
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": tools_list()}
        if method == "tools/call":
            name = params.get("name")
            if name not in TOOL_FN:
                raise _RpcError(-32602, f"Unknown tool: {name!r}", {"tools": list(TOOL_FN)})
            return call_tool(name, params.get("arguments") or {})
        if method == "resources/list":
            return {"resources": resources_list()}
        if method == "resources/templates/list":
            return {"resourceTemplates": []}
        if method == "resources/read":
            res = resource_read(params.get("uri"))
            if res is None:
                raise _RpcError(-32602 if modern else -32002, "Resource not found", {"uri": params.get("uri")})
            return res
        if method == "prompts/list":
            return {"prompts": []}
        if method == "logging/setLevel":
            return {}
        raise _RpcError(-32601, f"Method not found: {method}")


class _RpcError(Exception):
    def __init__(self, code, message, data=None):
        super().__init__(message)
        self.code, self.message, self.data = code, message, data


def _error(mid, code, message, data=None):
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": mid, "error": err}


def serve(stdin=None, stdout=None):
    """Read newline-delimited JSON-RPC from ``stdin`` (bytes), answer on ``stdout`` (bytes) until EOF."""
    stdin = stdin or sys.stdin.buffer
    stdout = stdout or sys.stdout.buffer
    server = Server()

    def send(obj):
        stdout.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
        stdout.flush()

    for raw in stdin:
        line = raw.decode("utf-8", "replace").strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            send(_error(None, -32700, "Parse error"))
            continue
        if isinstance(msg, list):                                     # JSON-RPC batch (2025-03-26)
            replies = [r for r in (server.handle(m) for m in msg) if r is not None]
            if replies:
                send(replies)
            continue
        reply = server.handle(msg)
        if reply is not None:
            send(reply)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__, file=sys.stderr)
        return 0
    proto_out = sys.stdout.buffer
    sys.stdout = sys.stderr                                           # stray prints never reach the protocol stream
    try:
        serve(sys.stdin.buffer, proto_out)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
