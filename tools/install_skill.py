#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Install skills/pixelcanvas into an AI tool's skills folder and record this runtime's location in the installed
SKILL.md, so "draw me a ..." works from any project.

  python tools/install_skill.py                      -> ~/.claude/skills/pixelcanvas            (Claude Code)
  python tools/install_skill.py agents               -> ~/.agents/skills/pixelcanvas            (Codex, Gemini CLI, opencode)
  python tools/install_skill.py opencode             -> ~/.config/opencode/skills/pixelcanvas
  python tools/install_skill.py cursor               -> ~/.cursor/skills/pixelcanvas
  python tools/install_skill.py codex                -> ~/.codex/skills/pixelcanvas
  python tools/install_skill.py claude --project     -> ./.claude/skills/pixelcanvas            (this project only)
  python tools/install_skill.py --dest SOME_DIR      -> SOME_DIR/pixelcanvas
  python tools/install_skill.py --list               every target, its folder and whether it is installed
  python tools/install_skill.py --uninstall [TARGET | --dest DIR]
  python tools/install_skill.py opencode --skill pixelcanvas-lite     only the lite skill (JSON specs, small models)

--skill pixelcanvas | pixelcanvas-lite | both (default both): pixelcanvas is the full skill (Python scenes, every
kit); pixelcanvas-lite draws from a small JSON spec with tools/compose.py, made for small local models.
Only the pixelcanvas/ and pixelcanvas-lite/ skill folders are written or removed; nothing else in the skills
folder is touched.
Tools without skill support: put AGENTS.md (or skills/pixelcanvas/SKILL.md) in their rules / system prompt.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

for _s in (sys.stdout, sys.stderr):              # Windows consoles default to a legacy code page
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "skills", "pixelcanvas")
NAME = "pixelcanvas"
SKILLS = ("pixelcanvas", "pixelcanvas-lite")
# target: (user folder, project folder, who reads it)
TARGETS = {
    "claude": ("~/.claude/skills", ".claude/skills", "Claude Code"),
    "agents": ("~/.agents/skills", ".agents/skills", "Codex, Gemini CLI, opencode and other Agent Skills tools"),
    "opencode": ("~/.config/opencode/skills", ".opencode/skills", "opencode"),
    "cursor": ("~/.cursor/skills", ".cursor/skills", "Cursor"),
    "codex": ("~/.codex/skills", ".codex/skills", "Codex CLI"),
}
MARK = "## Local install location"


def skills_dir(target, project=False, dest=None):
    if dest:
        return os.path.abspath(os.path.expanduser(dest))
    user, proj, _ = TARGETS[target]
    return os.path.abspath(proj) if project else os.path.expanduser(user)


def is_our_skill(folder, name=None):
    """True for a folder this script wrote (a SKILL.md with name: pixelcanvas / pixelcanvas-lite)."""
    p = os.path.join(folder, "SKILL.md")
    if not os.path.isfile(p):
        return False
    with open(p, encoding="utf-8", errors="replace") as f:
        head = f.read(2000)
    name = name or os.path.basename(os.path.normpath(folder))
    if name not in SKILLS:
        name = NAME
    return any(line.strip() == f"name: {name}" for line in head.splitlines())


def remove(dst):
    """Delete dst only when it is one of our skill folders (never a link target or a foreign folder)."""
    if os.path.islink(dst):
        os.unlink(dst)
        return True
    if not os.path.exists(dst):
        return False
    if os.path.basename(os.path.normpath(dst)) not in SKILLS or not is_our_skill(dst):
        raise SystemExit(f"{dst} exists but is not a pixelcanvas skill folder; leaving it alone")
    shutil.rmtree(dst)
    return True


def skill_names(choice):
    """--skill value -> the skill folder names to install."""
    return list(SKILLS) if choice in (None, "both") else [choice]


def install(skills, root=ROOT, name=NAME):
    src = os.path.join(ROOT, "skills", name)
    dst = os.path.join(skills, name)
    if os.path.abspath(dst) == os.path.abspath(src):
        raise SystemExit("destination is the source folder; nothing to install")
    if not os.path.isdir(src):
        raise SystemExit(f"skill folder not found: {src}")
    remove(dst)
    os.makedirs(skills, exist_ok=True)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    p = os.path.join(dst, "SKILL.md")
    with open(p, encoding="utf-8") as f:
        text = f.read()
    head = text.split(MARK)[0].rstrip() + "\n\n"
    tool = "compose.py" if name == "pixelcanvas-lite" else "render.py"
    scripts = os.path.join(dst, "scripts")
    if os.path.isdir(scripts):                       # the skill's own bootstrap.py finds this runtime first
        with open(os.path.join(scripts, "ROOT.txt"), "w", encoding="utf-8") as f:
            f.write(root + "\n")
    if name == "pixelcanvas-lite":                   # small models copy commands: write the real path into them
        head = head.replace("ROOT/tools/", root.replace("\\", "/") + "/tools/")
        with open(os.path.join(dst, "tools", "ROOT.txt"), "w", encoding="utf-8") as f:
            f.write(root + "\n")                     # tools/compose.py in the skill folder runs the real one
    text = (f"{head}{MARK}\n\nProject root: `{root}`\n"
            f"(The tools live in this folder: `python {root}/tools/{tool} ...`; scenes and pictures stay in your "
            f"project. After moving the runtime, run `python tools/install_skill.py` again.)\n")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return dst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?", choices=sorted(TARGETS), help="where to install (default claude)")
    ap.add_argument("--tool", choices=sorted(TARGETS), help="same as TARGET (older spelling)")
    ap.add_argument("--dest", help="skills folder (pixelcanvas/ is created inside); overrides TARGET")
    ap.add_argument("--project", action="store_true",
                    help="install into the current project (./.claude/skills, ./.agents/skills ...) instead of ~")
    ap.add_argument("--list", action="store_true", help="show every target and whether it is installed")
    ap.add_argument("--uninstall", action="store_true", help="remove the pixelcanvas skill folder(s)")
    ap.add_argument("--skill", choices=list(SKILLS) + ["both"], default="both",
                    help="which skill: pixelcanvas (full), pixelcanvas-lite (JSON specs for small models) or both "
                         "(default)")
    a = ap.parse_args(argv)
    target = a.target or a.tool or "claude"
    names = skill_names(a.skill)

    if a.list:
        for t, (user, proj, who) in TARGETS.items():
            for scope, d in (("user", skills_dir(t)), ("project", skills_dir(t, project=True))):
                for name in names:
                    state = "installed" if is_our_skill(os.path.join(d, name), name) else "-"
                    print(f"{t:<9} {scope:<8} {os.path.join(d, name):<60} {state:<10} {who if scope == 'user' else ''}")
        return 0

    skills = skills_dir(target, a.project, a.dest)
    if a.uninstall:
        for name in names:
            dst = os.path.join(skills, name)
            print(("removed " if remove(dst) else "not installed: ") + dst)
        return 0
    for name in names:
        dst = install(skills, name=name)
        print("installed", dst)
    print("runtime  ", ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
