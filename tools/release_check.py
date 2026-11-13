#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Release check: build the sdist and the wheel the way PyPI gets them, then use the installed command for real.

  python tools/release_check.py                     build in a temp folder, check, install, smoke test, clean up
  python tools/release_check.py --keep DIR          same, and keep everything in DIR (dist/, venvs, logs)
  python tools/release_check.py --wheel W.whl       skip the build: install W.whl in a clean venv and smoke test it
                                                    (this is how the Linux check runs: uv run --python 3.9 ...)

Steps: copy the source (no gallery/, outputs/, tests/, .git), `python -m build` in its own venv, `twine check
--strict`, check what the wheel and the sdist carry, install the wheel into a second clean venv, then run the
installed `pixelcanvas` command from an empty temp folder with a temp HOME: --help, version, doctor, the kit
imports, kit find, kit list / doc, new --prompt, render --check, --deliver, compose --example + compose, styles list, install-skill,
the MCP server (initialize, tools/list, a reference page) and `<tool> --help` for every bundled tool.

One PASS / FAIL line per step; the exit code is the number of failed steps (0 = ready to publish).
Needs: python with venv + pip, and network access to PyPI for build, twine, numpy and Pillow.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
import tarfile

for _s in (sys.stdout, sys.stderr):              # Windows consoles default to a legacy code page
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIN = os.name == "nt"
REQUEST = "a corgi sitting under an oak tree at sunset"     # the docs/PYPI.md example
KITS = ("clay, pictograms, spec, drafting, diagram, chart, ui, solid, figure, creature, flora, elements, texture, "
        "stylize, vector, layout, text, backdrop, palette, form, scenery, manifest, catalog")
# left out of the source copy and never allowed in a wheel or sdist
SKIP_DIRS = {".git", ".github", ".claude", ".claude-plugin", ".pytest_cache", "__pycache__", "gallery", "outputs",
             "tests", "build", "dist", "images"}
FORBIDDEN = re.compile(r"(^|/)(gallery|outputs|tests|__pycache__)/|\.py[co]$|(^|/)ROOT\.txt$|(^|/)zz_[^/]*$")

RESULTS = []


# ------------------------------------------------------------------------------------------------ helpers

def step(name):
    """Decorator-free step runner: step(name)(fn) -> runs fn, records PASS / FAIL with fn's detail string."""
    def run(fn):
        t0 = time.time()
        try:
            detail = fn() or ""
            ok = True
        except Exception as e:                                       # noqa: BLE001
            ok, detail = False, f"{type(e).__name__}: {e}"
        dt = time.time() - t0
        RESULTS.append((name, ok))
        detail = " ".join(str(detail).split())
        if len(detail) > 600:
            detail = detail[:600] + " ..."
        print(f"{'PASS' if ok else 'FAIL'}  {name:<40} {dt:6.1f} s  {detail}", flush=True)
        return ok
    return run


def bin_path(venv, name):
    return os.path.join(venv, "Scripts", name + ".exe") if WIN else os.path.join(venv, "bin", name)


def run(args, cwd=None, env=None, timeout=900, input=None, log=None):
    r = subprocess.run([str(a) for a in args], cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout, input=input)
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"$ {' '.join(map(str, args))}\n[exit {r.returncode}]\n{r.stdout}\n{r.stderr}\n\n")
    return r


def must(r, what):
    if r.returncode != 0:
        tail = (r.stderr.strip() or r.stdout.strip())[-500:]
        raise RuntimeError(f"{what} exited {r.returncode}: {tail}")
    return r


def clean_env(home):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    for k in list(env):
        if k.startswith("PIXELCANVAS_") or k in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
            env.pop(k)
    env["HOME"] = env["USERPROFILE"] = home                          # install-skill and friends stay in the temp
    return env


def make_venv(path, log):
    must(run([sys.executable, "-m", "venv", path], log=log), "python -m venv")
    py = bin_path(path, "python")
    must(run([py, "-m", "pip", "install", "-q", "--upgrade", "pip"], log=log), "pip upgrade")
    return py


def size(path):
    n = os.path.getsize(path)
    return f"{n / 1e6:.2f} MB" if n >= 1e6 else f"{n / 1e3:.0f} kB"


# ------------------------------------------------------------------------------------------------ expected content

def expected_wheel_files(src):
    """Paths the wheel must contain, derived from the source tree (so new kits, tools and specs are covered)."""
    want = set()
    for dirpath, dirs, files in os.walk(os.path.join(src, "pixelcanvas")):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        if "__init__.py" in files:
            rel = os.path.relpath(dirpath, src).replace(os.sep, "/")
            want.add(rel + "/__init__.py")
    for folder, kinds in (("tools", (".py",)), ("scenes", (".py", ".json")), ("skills", (".md", ".py", ".json"))):
        base = os.path.join(src, folder)
        for dirpath, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                if f.endswith(kinds) and not f.startswith("zz_"):
                    rel = os.path.relpath(os.path.join(dirpath, f), base).replace(os.sep, "/")
                    want.add(f"pixelcanvas/_runtime/{folder}/{rel}")
    return want


def copy_source(dst, work):
    def ignore(d, names):
        out = {n for n in names if n in SKIP_DIRS or n.endswith((".egg-info", ".pyc")) or n.startswith("zz_")
               or os.path.normcase(os.path.abspath(os.path.join(d, n))) == os.path.normcase(work)}
        rel = os.path.relpath(d, ROOT).replace(os.sep, "/")
        if rel == "docs":
            out |= {n for n in names if n != "PYPI.md" and n != "INSTALL.md"}
        return out
    shutil.copytree(ROOT, dst, ignore=ignore)


# ------------------------------------------------------------------------------------------------ MCP

def mcp_round_trip(exe, cwd, env):
    p = subprocess.Popen([exe, "mcp"], cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    n = [0]

    def call(method, params=None):
        n[0] += 1
        msg = {"jsonrpc": "2.0", "id": n[0], "method": method}
        if params is not None:
            msg["params"] = params
        p.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
        p.stdin.flush()
        line = p.stdout.readline()
        if not line:
            raise RuntimeError("no reply: " + p.stderr.read().decode("utf-8", "replace")[-400:])
        reply = json.loads(line)
        if reply.get("id") != n[0] or "result" not in reply:
            raise RuntimeError(f"bad reply to {method}: {line[:300]!r}")
        return reply["result"]

    try:
        init = call("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                   "clientInfo": {"name": "release_check", "version": "1"}})
        if init["serverInfo"]["name"] != "pixelcanvas":
            raise RuntimeError(f"serverInfo {init['serverInfo']}")
        p.stdin.write(b'{"jsonrpc": "2.0", "method": "notifications/initialized"}\n')
        p.stdin.flush()
        tools = [t["name"] for t in call("tools/list")["tools"]]
        for need in ("render", "new_scene", "find", "reference"):
            if need not in tools:
                raise RuntimeError(f"tools/list lacks {need}: {tools}")
        ref = call("tools/call", {"name": "reference", "arguments": {"page": "SKILL"}})
        text = ref["content"][0]["text"]
        if ref.get("isError") or "pixelcanvas" not in text or len(text) < 1000:
            raise RuntimeError(f"reference SKILL: {text[:300]!r}")
        return f"{init['serverInfo'].get('version', '')} protocol {init['protocolVersion']}; {len(tools)} tools; SKILL page read"
    finally:
        try:
            p.stdin.close()
            p.wait(timeout=30)
        except Exception:                                            # noqa: BLE001
            p.kill()
        for s in (p.stdout, p.stderr):
            s.close()


# ------------------------------------------------------------------------------------------------ smoke

def smoke(wheel, work, log):
    """Install ``wheel`` into a clean venv and use the installed command from an empty folder."""
    venv = os.path.join(work, "venv-run")
    home = os.path.join(work, "home")
    proj = os.path.join(work, "project")
    for d in (home, proj):
        os.makedirs(d, exist_ok=True)
    env = clean_env(home)
    state = {}

    def install():
        py = make_venv(venv, log)
        must(run([py, "-m", "pip", "install", "-q", wheel], env=env, cwd=proj, log=log), "pip install wheel")
        state["py"], state["exe"] = py, bin_path(venv, "pixelcanvas")
        r = must(run([py, "-c", "import sys, numpy, PIL; print(sys.version.split()[0], numpy.__version__, "
                                "PIL.__version__)"], env=env, cwd=proj), "versions")
        return "python / numpy / Pillow " + r.stdout.strip()
    if not step("install wheel in clean venv")(install):
        return

    def pc(*args, cwd=proj, **kw):
        return run([state["exe"], *args], cwd=cwd, env=env, log=log, **kw)

    def help_():
        r = must(pc("--help"), "pixelcanvas --help")
        tools = json.loads(must(run([state["py"], "-c", "import json; from pixelcanvas import cli; "
                                    "print(json.dumps(sorted(cli.available_tools())))"], cwd=proj, env=env),
                                "available_tools").stdout.strip().splitlines()[-1])
        state["tools"] = tools
        missing = [t for t in tools if t not in r.stdout]
        for need in ("compose", "kit", "render", "new_scene", "install_skill", "styles", "look"):
            if need not in tools:
                missing.append(need)
        if missing:
            raise RuntimeError(f"--help does not list {missing}")
        return f"{len(tools)} tools listed: {', '.join(tools)}"
    step("pixelcanvas --help")(help_)

    def version():
        r = must(pc("version"), "version")
        root = r.stdout.split("root", 1)[-1].strip()
        if "_runtime" not in root or not os.path.normcase(os.path.realpath(root)).startswith(os.path.normcase(os.path.realpath(venv))):
            raise RuntimeError(f"runtime root is not the installed copy: {root}")
        return r.stdout.splitlines()[0] + f"; root {root}"
    step("pixelcanvas version")(version)

    def doctor():
        r = must(pc("doctor", "--json"), "doctor")
        doc = json.loads(r.stdout.strip().splitlines()[-1])
        rows = {c["name"]: c for c in doc["checks"]}
        bad = [f"{k}: {c['detail']}" for k, c in rows.items() if c["status"] == "fail"]
        if bad or not doc["ok"]:
            raise RuntimeError("; ".join(bad))
        warn = [k for k, c in rows.items() if c["status"] == "warn"]
        return rows["runtime"]["detail"] + (f"; warnings: {', '.join(warn)}" if warn else "")
    step("pixelcanvas doctor")(doctor)

    def imports():
        r = must(run([state["py"], "-c", f"from pixelcanvas import {KITS}; import pixelcanvas; "
                                         "print(pixelcanvas.__version__, pixelcanvas.__file__)"],
                     cwd=proj, env=env, log=log), "imports")
        if os.path.normcase(os.path.realpath(venv)) not in os.path.normcase(r.stdout.replace("/", os.sep)):
            raise RuntimeError("imported pixelcanvas from outside the clean venv")
        return f"{len(KITS.split(','))} modules; {r.stdout.strip()}"
    step("import every kit")(imports)

    def kit_find():
        r = must(pc("kit", "find", REQUEST), "kit find")
        if len(r.stdout.strip().splitlines()) < 3:
            raise RuntimeError(f"short answer: {r.stdout!r}")
        return r.stdout.strip().splitlines()[0][:120]
    step("kit find")(kit_find)

    def kit_pages():
        r = must(pc("kit", "list"), "kit list")
        if not re.search(r"^scenes\s+\d+", r.stdout, re.M):
            raise RuntimeError("kit list shows no scene kinds (templates not found by the catalogue)")
        r = must(pc("kit", "doc", "flora"), "kit doc flora")
        if len(r.stdout) < 1000:
            raise RuntimeError(f"kit doc flora printed {len(r.stdout)} characters")
        return f"kit list with scene kinds; kit doc flora {len(r.stdout)} characters"
    step("kit list + kit doc")(kit_pages)

    def new():
        r = must(pc("new", "corgi", "--prompt", REQUEST), "new --prompt")
        if not os.path.isfile(os.path.join(proj, "corgi.py")):
            raise RuntimeError(f"corgi.py not written in the project folder: {os.listdir(proj)}")
        kind = re.search(r"\(kind: ([\w-]+)\)", r.stdout)
        kind = kind.group(1) if kind else "?"
        if kind in ("blank", "?"):
            raise RuntimeError(f"the request did not pick a starter template (kind {kind}); the catalogue finds no "
                               f"scene kinds in the installed package: {r.stdout.strip()[:200]}")
        return f"corgi.py written from the {kind} template"
    blank = os.path.join(ROOT, "scenes", "_template_blank.py")
    if not step("new --prompt in an empty project")(new) and os.path.isfile(blank):
        shutil.copyfile(blank, os.path.join(proj, "corgi.py"))

    def render():
        r = must(pc("render", "corgi.py", "--scale", "0.25", "--check"), "render --check")
        out = os.path.join(proj, "outputs")
        got = sorted(os.listdir(out)) if os.path.isdir(out) else []
        if "corgi_x0.25.png" not in got or "corgi_x0.25_check.jpg" not in got:
            raise RuntimeError(f"outputs: {got}")
        if "WARNINGS" not in r.stdout and "OBJECTS" not in r.stdout:
            raise RuntimeError("no check report in the output")
        return "outputs/corgi_x0.25.png + _check.jpg, check report printed"
    step("render --check")(render)

    def deliver():
        must(pc("render", "corgi.py", "--scale", "0.25", "--deliver", "final/"), "render --deliver")
        if not os.path.isfile(os.path.join(proj, "final", "corgi.png")):
            raise RuntimeError(f"final/: {os.listdir(os.path.join(proj, 'final')) if os.path.isdir(os.path.join(proj, 'final')) else 'missing'}")
        return "final/corgi.png"
    step("render --deliver")(deliver)

    def compose():
        r = must(pc("compose", "--example", "poster"), "compose --example poster")
        spec = json.loads(r.stdout)
        with open(os.path.join(proj, "poster.json"), "w", encoding="utf-8") as f:
            f.write(r.stdout)
        before = set(os.listdir(os.path.join(proj, "outputs"))) if os.path.isdir(os.path.join(proj, "outputs")) else set()
        r = must(pc("compose", "poster.json", "--check"), "compose poster.json --check")
        made = sorted(f for f in os.listdir(os.path.join(proj, "outputs")) if f not in before and f.endswith(".png"))
        if not made:
            raise RuntimeError("no PNG written: " + r.stdout[-300:])
        must(pc("compose", "poster.json", "--deliver", "final/"), "compose poster.json --deliver final/")
        if not os.path.isfile(os.path.join(proj, "final", "poster.png")):
            raise RuntimeError("final/poster.png not delivered")
        return f"spec with {len(spec.get('layers', []))} layers -> outputs/{made[0]}, final/poster.png"
    step("compose --example + --check + --deliver")(compose)

    def styles():
        r = must(pc("styles", "list"), "styles list")
        n = len([x for x in r.stdout.splitlines() if x.strip()])
        if n < 10:
            raise RuntimeError(r.stdout[-300:])
        return f"{n} lines"
    step("styles list")(styles)

    def install_skill():
        must(pc("install-skill", "claude"), "install-skill claude")
        base = os.path.join(home, ".claude", "skills")
        need = [os.path.join(base, "pixelcanvas", "SKILL.md"), os.path.join(base, "pixelcanvas", "references", "api.md"),
                os.path.join(base, "pixelcanvas", "scripts", "bootstrap.py"),
                os.path.join(base, "pixelcanvas-lite", "SKILL.md"),
                os.path.join(base, "pixelcanvas-lite", "tools", "compose.py")]
        gone = [p for p in need if not os.path.isfile(p)]
        if gone:
            raise RuntimeError(f"missing {gone}")
        refs = len(os.listdir(os.path.join(base, "pixelcanvas", "references")))
        shim = os.path.join(base, "pixelcanvas-lite", "tools", "compose.py")
        must(run([state["py"], shim, "--kinds"], cwd=proj, env=env, log=log), "installed lite compose shim")
        return f"~/.claude/skills/pixelcanvas ({refs} reference pages) + pixelcanvas-lite; lite shim runs"
    step("install-skill claude (temp HOME)")(install_skill)

    step("MCP initialize + tools/list")(lambda: mcp_round_trip(state["exe"], proj, env))

    def every_tool():
        bad = []
        for t in state.get("tools", []):
            r = pc(t, "--help", timeout=300)
            if r.returncode != 0:
                bad.append(f"{t} ({r.returncode}: {(r.stderr.strip() or r.stdout.strip())[-160:]})")
        if bad:
            raise RuntimeError("; ".join(bad))
        return f"{len(state.get('tools', []))} tools answer --help"
    step("pixelcanvas <tool> --help")(every_tool)


# ------------------------------------------------------------------------------------------------ build

def build(work, log):
    src = os.path.join(work, "src")
    dist = os.path.join(work, "dist")
    state = {}

    def copy():
        copy_source(src, work)
        return src
    step("copy source")(copy)

    def build_dist():
        py = make_venv(os.path.join(work, "venv-build"), log)
        must(run([py, "-m", "pip", "install", "-q", "build", "twine"], log=log), "pip install build twine")
        state["py"] = py
        must(run([py, "-m", "build", "--outdir", dist, src], log=log), "python -m build")
        whl = glob.glob(os.path.join(dist, "*.whl"))
        sd = glob.glob(os.path.join(dist, "*.tar.gz"))
        if len(whl) != 1 or len(sd) != 1:
            raise RuntimeError(f"dist: {os.listdir(dist)}")
        state["wheel"], state["sdist"] = whl[0], sd[0]
        return f"{os.path.basename(whl[0])} ({size(whl[0])}), {os.path.basename(sd[0])} ({size(sd[0])})"
    if not step("build sdist + wheel")(build_dist):
        return None

    def twine():
        r = must(run([state["py"], "-m", "twine", "check", "--strict", state["wheel"], state["sdist"]], log=log),
                 "twine check --strict")
        return " ".join(r.stdout.split())
    step("twine check --strict")(twine)

    def wheel_content():
        with zipfile.ZipFile(state["wheel"]) as z:
            names = [n for n in z.namelist() if not n.endswith("/")]
            meta = next(z.read(n).decode("utf-8") for n in names if n.endswith(".dist-info/METADATA"))
        files = set(names)
        want = expected_wheel_files(src)
        missing = sorted(want - files)
        bad = sorted(n for n in names if FORBIDDEN.search(n))
        if missing or bad:
            raise RuntimeError(f"missing {missing[:12]}{' ...' if len(missing) > 12 else ''}; forbidden {bad[:12]}")
        if "raw.githubusercontent.com" not in meta or "](docs/" in meta or "](gallery/" in meta:
            raise RuntimeError("long description is not docs/PYPI.md with absolute links")
        pk = sorted({n.rsplit("/", 1)[0] for n in names if n.endswith("/__init__.py")})
        return (f"{len(names)} files, {size(state['wheel'])}; packages {', '.join(p.replace('/', '.') for p in pk)}; "
                f"runtime: {sum(n.startswith('pixelcanvas/_runtime/tools/') for n in names)} tools, "
                f"{sum(n.startswith('pixelcanvas/_runtime/scenes/') for n in names)} scene files, "
                f"{sum(n.startswith('pixelcanvas/_runtime/skills/') for n in names)} skill files")
    step("wheel content")(wheel_content)

    def sdist_content():
        with tarfile.open(state["sdist"]) as t:
            names = [m.name.split("/", 1)[1] for m in t.getmembers() if m.isfile() and "/" in m.name]
        bad = sorted(n for n in names if FORBIDDEN.search(n))
        need = ["pyproject.toml", "setup.py", "LICENSE", "docs/PYPI.md", "CHANGELOG.md", "tools/render.py",
                "skills/pixelcanvas-lite/SKILL.md"]
        missing = [n for n in need if n not in names]
        if bad or missing:
            raise RuntimeError(f"forbidden {bad[:12]}; missing {missing}")
        return f"{len(names)} files, {size(state['sdist'])}"
    step("sdist content")(sdist_content)
    return state["wheel"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", metavar="DIR", help="work in DIR and keep dist/, venvs and release_check.log there")
    ap.add_argument("--wheel", help="smoke test this wheel instead of building one")
    a = ap.parse_args(argv)

    work = os.path.abspath(a.keep) if a.keep else tempfile.mkdtemp(prefix="pixelcanvas-release-")
    if a.keep and os.path.isdir(work):
        for sub in ("src", "dist", "venv-build", "venv-run", "home", "project"):
            shutil.rmtree(os.path.join(work, sub), ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    log = os.path.join(work, "release_check.log")
    open(log, "w").close()
    print(f"pixelcanvas release check  python {sys.version.split()[0]}  {sys.platform}  work {work}", flush=True)
    t0 = time.time()
    try:
        wheel = os.path.abspath(a.wheel) if a.wheel else build(work, log)
        if wheel:
            smoke(wheel, work, log)
    finally:
        if not a.keep:
            shutil.rmtree(work, ignore_errors=True)
    fails = [n for n, ok in RESULTS if not ok]
    print(f"{'PASS' if not fails else 'FAIL'}  release check: {len(RESULTS) - len(fails)}/{len(RESULTS)} steps "
          f"in {time.time() - t0:.0f} s" + (f"; failed: {', '.join(fails)}" if fails else "")
          + (f"; log {log}" if a.keep else ""))
    return len(fails)


if __name__ == "__main__":
    sys.exit(main())
