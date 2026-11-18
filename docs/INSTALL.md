# Installing pixelcanvas

pixelcanvas runs anywhere Python 3.9+ runs. Two pieces: the `pixelcanvas` command (the Python package) and the
skill `pixelcanvas` (one `SKILL.md` your agent reads). The agent needs both: the skill tells it how to paint, the command
does the painting.

## With pip (quickest)

```bash
pip install pixelcanvas
pixelcanvas --version
pixelcanvas install-skill                 # -> ~/.claude/skills/pixelcanvas             (Claude Code)
pixelcanvas install-skill --opencode      # -> ~/.config/opencode/skills/pixelcanvas    (opencode)
pixelcanvas install-skill --to .claude/skills   # -> .claude/skills/pixelcanvas        (this project only, or any skills folder)
```

`pixelcanvas install-skill` prints every file it writes and leaves an existing skill alone; `--force` replaces it.
Every command is `pixelcanvas <command>` (`pixelcanvas help` lists them); `python -m pixelcanvas` runs the same.

With [pipx](https://pipx.pypa.io) or [uv](https://docs.astral.sh/uv/) the command gets its own environment:

```bash
pipx install pixelcanvas            # or: uv tool install pixelcanvas
```

## As a Claude Code plugin

The repository is a plugin marketplace; the plugin holds the skill. Install the command with pip as well.

```text
/plugin marketplace add treeleaves30760/pixelcanvas
/plugin install pixelcanvas@pixelcanvas
```

## As an Agent Skill

Any tool that reads Agent Skills. This installs the skill; install the command with pip as well.

```bash
npx skills add treeleaves30760/pixelcanvas
```

## From a clone

```bash
git clone https://github.com/treeleaves30760/pixelcanvas
cd pixelcanvas
pip install -e .
pixelcanvas install-skill
```

An editable install keeps `pixelcanvas` in step with the clone; `pixelcanvas install-skill --force` copies a changed
`skills/pixelcanvas/SKILL.md` again.

## Local models

A small open-weight model drives pixelcanvas through Claude Code or opencode like any other model: point the framework at
your model server, install the skill, and ask for a picture. The model needs vision (it opens the `VIEW:` pictures).
A redraw takes about 44-86 commands, and a session ends at about 55k tokens of context.

## Check the install

```bash
pixelcanvas --version
pixelcanvas help
pixelcanvas fonts            # the fonts pixelcanvas finds, Latin and CJK
```

## Uninstall

```bash
pip uninstall pixelcanvas
```

Then delete the skill folder (`~/.claude/skills/pixelcanvas` or `~/.config/opencode/skills/pixelcanvas`).
