"""Install the pixelcanvas skill (SKILL.md) into an agent's skills folder.

  pixelcanvas install-skill                    -> ~/.claude/skills/pixelcanvas             (Claude Code)
  pixelcanvas install-skill --opencode         -> ~/.config/opencode/skills/pixelcanvas    (opencode)
  pixelcanvas install-skill --to .claude/skills -> .claude/skills/pixelcanvas            (any skills folder, e.g. one project)
  pixelcanvas install-skill --force            replace a pixelcanvas skill that is already there

The skill comes from the installed package (pixelcanvas/skill/) or, in a checkout, from skills/pixelcanvas/.
Only the folder `pixelcanvas` inside the skills folder is written; an existing one is replaced only with --force, and
only when it holds a pixelcanvas skill.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional

NAME = "pixelcanvas"
CLAUDE_DIR = Path("~/.claude/skills")
OPENCODE_DIR = Path("~/.config/opencode/skills")


def skill_source() -> Path:
    """The folder that holds the pixelcanvas SKILL.md: the packaged copy, else the checkout's skills/pixelcanvas."""
    here = Path(__file__).resolve().parent
    for cand in (here / "skill", here.parent / "skills" / NAME):
        if (cand / "SKILL.md").is_file():
            return cand
    raise FileNotFoundError("the pixelcanvas skill was not found (looked in pixelcanvas/skill/ and skills/pixelcanvas/)")


def _is_pixelcanvas_skill(folder: Path) -> bool:
    p = folder / "SKILL.md"
    if not p.is_file():
        return False
    head = p.read_text(encoding="utf-8", errors="replace")[:2000]
    return any(line.strip() == f"name: {NAME}" for line in head.splitlines())


def install(skills_dir: Path, force: bool = False, src: Optional[Path] = None) -> List[Path]:
    """Copy the skill to ``skills_dir/pixelcanvas``; return the files written. Raises FileExistsError when the folder is
    there and ``force`` is not set, or when it is not a pixelcanvas skill folder."""
    src = Path(src) if src is not None else skill_source()
    dst = Path(skills_dir).expanduser().resolve() / NAME
    if dst == src.resolve():
        raise FileExistsError(f"{dst} is the skill's own source folder; nothing to install")
    if dst.exists() or dst.is_symlink():
        if not force:
            raise FileExistsError(f"{dst} already exists; run again with --force to replace it")
        if dst.is_symlink() or dst.is_file():
            raise FileExistsError(f"{dst} is not a folder; remove it yourself")
        if any(dst.iterdir()) and not _is_pixelcanvas_skill(dst):
            raise FileExistsError(f"{dst} is not a pixelcanvas skill folder; leaving it alone")
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "__init__.py"))
    return sorted(p for p in dst.rglob("*") if p.is_file())


def main(argv: Optional[List[str]] = None) -> int:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
    ap = argparse.ArgumentParser(prog="pixelcanvas install-skill", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    where = ap.add_mutually_exclusive_group()
    where.add_argument("--to", metavar="DIR", help="skills folder to install into (pixelcanvas/ is created inside)")
    where.add_argument("--opencode", action="store_true", help=f"install for opencode ({OPENCODE_DIR / NAME})")
    ap.add_argument("--force", action="store_true", help="replace a pixelcanvas skill folder that is already there")
    a = ap.parse_args(argv)
    skills_dir = Path(a.to) if a.to else (OPENCODE_DIR if a.opencode else CLAUDE_DIR)
    try:
        files = install(skills_dir, force=a.force)
    except (FileExistsError, FileNotFoundError) as e:
        print(f"pixelcanvas install-skill: {e}", file=sys.stderr)
        return 1
    dst = Path(skills_dir).expanduser().resolve() / NAME
    print(f"installed the pixelcanvas skill in {dst}")
    for f in files:
        print(f"  wrote {f}")
    print("The agent finds it as the skill `pixelcanvas`; the `pixelcanvas` command must be on its PATH (pip install pixelcanvas).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
