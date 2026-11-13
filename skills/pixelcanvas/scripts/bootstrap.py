#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find (or download) the pixelcanvas runtime and make sure numpy + Pillow import.

An installed skill holds SKILL.md + references/; the renderer (pixelcanvas/, tools/, scenes/) is the runtime:

  python bootstrap.py              print the version, then the runtime root as the last line (download on first use)
  python bootstrap.py --update     refresh a runtime that this script downloaded
  python bootstrap.py --home DIR   use / create the runtime in DIR (default ~/.pixelcanvas)

Search order: a checkout in the current folder or its parents -> $PIXELCANVAS_HOME -> the runtime that
tools/install_skill.py recorded in scripts/ROOT.txt -> the repository this skill folder sits in (when it holds pixelcanvas/, tools/ and scenes/) -> ~/.pixelcanvas (--home) -> a pip-installed
pixelcanvas -> download. A runtime older than this skill is never picked silently: a download this script manages
is updated in place (the tag matching the skill, else the default branch); a checkout of your own is kept with a
one-line warning; without network the newest runtime found is used and its version printed. The download is a
shallow sparse git clone when git is available, else a GitHub archive. Standard library only.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request

for _s in (sys.stdout, sys.stderr):              # Windows consoles default to a legacy code page
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

VERSION = "0.4.0"                      # fallback when SKILL.md is not next to this script
MARKER = ".pixelcanvas-managed"
REPO = "treeleaves30760/pixelcanvas"
GIT_URL = f"https://github.com/{REPO}.git"
SKILLS = ("pixelcanvas", "pixelcanvas-lite")         # both skills live in the runtime's skills/ folder
SPARSE = ["pixelcanvas", "tools", "scenes"] + [f"skills/{n}" for n in SKILLS]      # cone mode adds the root files


def skill_version():
    """This skill's version: SKILL.md (next to scripts/) metadata, else VERSION."""
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "SKILL.md"), encoding="utf-8") as f:
            m = re.search(r'^\s+version:\s*"?([0-9][^"\s]*)"?\s*$', f.read(), re.M)
        if m:
            return m.group(1)
    except OSError:
        pass
    return VERSION


REFS = [f"v{skill_version()}", None]                                       # release tag first, then the default branch


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def is_root(path):
    return bool(path) and os.path.isfile(os.path.join(path, "pixelcanvas", "__init__.py")) \
        and os.path.isfile(os.path.join(path, "tools", "render.py"))


def installed_runtime():
    """tools/ + scenes/ bundled in a pip-installed pixelcanvas (pixelcanvas/_runtime), or None."""
    spec = importlib.util.find_spec("pixelcanvas")
    if spec is None or not spec.origin:
        return None
    rt = os.path.join(os.path.dirname(spec.origin), "_runtime")
    return rt if os.path.isfile(os.path.join(rt, "tools", "render.py")) else None


def run(cmd, **kw):
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def git_download(dest):
    for ref in REFS:
        shutil.rmtree(dest, ignore_errors=True)
        branch = ["--branch", ref] if ref else []
        r = run(["git", "clone", "--depth", "1", *branch, "--filter=blob:none", "--sparse", GIT_URL, dest])
        if r.returncode == 0:
            r = run(["git", "-C", dest, "sparse-checkout", "set", *SPARSE])
        if r.returncode == 0 and is_root(dest):
            return ref or "default branch"
    shutil.rmtree(dest, ignore_errors=True)
    return None


def _wanted(rel):
    top = rel.split("/", 1)[0]
    if "/" not in rel:
        return True                                                    # root files (requirements.txt, LICENSE ...)
    return top in ("pixelcanvas", "tools", "scenes") or any(rel.startswith(f"skills/{n}/") for n in SKILLS)


def archive_download(dest):
    for ref in REFS:
        url = f"https://github.com/{REPO}/archive/" + (f"refs/tags/{ref}.tar.gz" if ref else "HEAD.tar.gz")
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
        except OSError as e:
            log(f"  {url}: {e}")
            continue
        shutil.rmtree(dest, ignore_errors=True)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for m in tar.getmembers():
                rel = m.name.split("/", 1)[1] if "/" in m.name else ""
                if not rel or not m.isfile() or ".." in rel.split("/") or rel.startswith("/") or not _wanted(rel):
                    continue
                out = os.path.join(dest, *rel.split("/"))
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with tar.extractfile(m) as src, open(out, "wb") as f:
                    shutil.copyfileobj(src, f)
        if is_root(dest):
            return ref or "default branch"
    return None


class BootstrapError(Exception):
    pass


def download(dest):
    """Fresh download of the runtime into the new or empty folder ``dest``; BootstrapError when it fails."""
    if os.path.exists(dest) and os.listdir(dest):
        raise BootstrapError(f"{dest} exists but is not a pixelcanvas runtime; pass --home with an empty or new folder")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".download"
    got = git_download(tmp) if shutil.which("git") else None
    got = got or archive_download(tmp)
    if not got:
        shutil.rmtree(tmp, ignore_errors=True)
        raise BootstrapError(f"download failed (is github.com reachable?). Manual: git clone {GIT_URL} {dest}")
    with open(os.path.join(tmp, MARKER), "w", encoding="utf-8") as f:
        f.write("downloaded by skills/pixelcanvas/scripts/bootstrap.py\n")
    if os.path.isdir(dest):
        os.rmdir(dest)
    os.replace(tmp, dest)
    log(f"downloaded pixelcanvas ({got}) to {dest}")


def update(root):
    """Bring a managed runtime to this skill's version in place (git: fetch + checkout; else a fresh download
    beside it, then switch). True when it worked, False (nothing changed) when it could not."""
    if os.path.isdir(os.path.join(root, ".git")):
        for ref in REFS:
            r = run(["git", "-C", root, "fetch", "--depth", "1", "-q", "origin", f"refs/tags/{ref}" if ref else "HEAD"])
            if r.returncode == 0:
                r = run(["git", "-C", root, "checkout", "-q", "FETCH_HEAD"])
                if r.returncode == 0:
                    run(["git", "-C", root, "sparse-checkout", "set", *SPARSE])      # picks up skills added since
                    return True
        return False
    old = root + ".old"
    shutil.rmtree(old, ignore_errors=True)
    os.replace(root, old)
    try:
        download(root)
    except BootstrapError:
        shutil.rmtree(root, ignore_errors=True)
        os.replace(old, root)
        return False
    shutil.rmtree(old, ignore_errors=True)
    return True


def ensure_deps():
    missing = [pkg for mod, pkg in (("numpy", "numpy"), ("PIL", "Pillow")) if importlib.util.find_spec(mod) is None]
    if missing:
        log("installing:", ", ".join(missing))
        r = run([sys.executable, "-m", "pip", "install", "-q", *missing])
        if r.returncode != 0:
            sys.exit("pip install failed:\n" + r.stderr.strip() + f"\nRun: {sys.executable} -m pip install numpy Pillow")
    r = run([sys.executable, "-c", "import numpy, PIL; print(numpy.__version__, PIL.__version__)"])
    if r.returncode != 0:
        sys.exit("numpy / Pillow do not import:\n" + r.stderr.strip())
    return r.stdout.split()


def version_of(root):
    """The runtime's ``__version__`` read from pixelcanvas/__init__.py without importing it ("?" when unknown)."""
    try:
        with open(os.path.join(root, "pixelcanvas", "__init__.py"), encoding="utf-8") as f:
            m = re.search(r'^__version__ = "([^"]+)"', f.read(), re.M)
        return m.group(1) if m else "?"
    except OSError:
        return "?"


def vkey(v):
    """Sortable form of a version string; unknown versions sort lowest."""
    if not v or v == "?":
        return (0,)
    return tuple(int(x) for x in re.findall(r"\d+", v.split("+")[0].split("-")[0])[:3])


def is_complete(path):
    return is_root(path) and os.path.isdir(os.path.join(path, "scenes"))


def is_managed(path, home):
    """A runtime this script downloaded: the default folder, or a folder with its marker / a sparse git clone."""
    if os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(home)):
        return True
    return os.path.isfile(os.path.join(path, MARKER)) \
        or os.path.isfile(os.path.join(path, ".git", "info", "sparse-checkout"))


def candidates(home):
    """Existing runtimes in priority order: [(label, root, version)]."""
    out, seen = [], set()

    def add(label, path, version=None):
        path = os.path.abspath(os.path.expanduser(path))
        if os.path.normcase(path) not in seen:
            seen.add(os.path.normcase(path))
            out.append((label, path, version or version_of(path)))
    here = os.path.abspath(os.getcwd())
    while True:
        if is_root(here):
            add("checkout", here)
            break
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    env = os.environ.get("PIXELCANVAS_HOME")
    if env and is_root(os.path.expanduser(env)):
        add("$PIXELCANVAS_HOME", env)
    try:                                         # tools/install_skill.py writes the runtime it installed from
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ROOT.txt"), encoding="utf-8") as f:
            rec = f.read().strip()
        if is_root(rec):
            add("installer", rec)
    except OSError:
        pass
    repo = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    if is_complete(repo):
        add("repository", repo)
    if is_root(home):
        add("home", home)
    rt = installed_runtime()
    if rt:
        add("pip", rt, version_of(os.path.dirname(os.path.dirname(rt))))
    return out


def choose(home, want, found=None):
    """The runtime to use: ``(root, version)``, or None when there is none (the caller downloads)."""
    found = candidates(home) if found is None else found
    for label, path, v in found:
        if vkey(v) >= vkey(want) or v == "?":
            return path, v
        if not is_managed(path, home):
            log(f"warning: {label} {path} is pixelcanvas {v}, older than this skill ({want}); using it anyway")
            return path, v
        if update(path) and vkey(version_of(path)) >= vkey(want):
            log(f"updated {path} from {v} to {version_of(path)} (the skill is {want})")
            return path, version_of(path)
    if not found:
        return None
    best = max(found, key=lambda c: vkey(c[2]))
    log(f"no network: using pixelcanvas {best[2]} from {best[1]} (the skill is {want})")
    return best[1], best[2]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--home", default="~/.pixelcanvas", help="where to download the runtime (default ~/.pixelcanvas)")
    ap.add_argument("--update", action="store_true", help="refresh a runtime downloaded by this script")
    a = ap.parse_args(argv)

    home = os.path.abspath(os.path.expanduser(a.home))
    want = skill_version()
    got = choose(home, want)
    if got is None:
        log(f"downloading pixelcanvas {want} to {home} ...")
        try:
            download(home)
        except BootstrapError as e:
            sys.exit(str(e))
        root = home
    else:
        root = got[0]
        if a.update:
            if is_managed(root, home):
                log(f"updated {root}" if update(root) else "update skipped: could not reach github.com")
            else:
                log(f"update skipped: {root} is not a runtime downloaded by this script (update it with git or pip)")
    np_v, pil_v = ensure_deps()
    v = version_of(root) if is_root(root) else want
    print(f"pixelcanvas {v} (numpy {np_v}, Pillow {pil_v})")
    print(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
