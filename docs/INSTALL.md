# Installing easel

easel runs anywhere Python 3.9+ runs. Two pieces: the `easel` command (the Python package `pixelcanvas`) and the
skill `easel` (one `SKILL.md` your agent reads). The agent needs both: the skill tells it how to paint, the command
does the painting.

## With pip (quickest)

```bash
pip install pixelcanvas
easel --version
easel-install-skill                 # -> ~/.claude/skills/easel             (Claude Code)
easel-install-skill --opencode      # -> ~/.config/opencode/skills/easel    (opencode)
easel-install-skill --to .claude/skills   # -> .claude/skills/easel        (this project only, or any skills folder)
```

`easel-install-skill` prints every file it writes and leaves an existing skill alone; `--force` replaces it.
The package also installs `pixelcanvas`, which runs the same program as `easel`.

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
easel-install-skill
```

An editable install keeps `easel` in step with the clone; `easel-install-skill --force` copies a changed
`skills/easel/SKILL.md` again.

## Local models

A small open-weight model drives easel through Claude Code or opencode like any other model: point the framework at
your model server, install the skill, and ask for a picture. The model needs vision (it opens the `VIEW:` pictures).
A redraw takes about 46-75 commands, and a session ends at about 55k tokens of context.

## Check the install

```bash
easel --version
easel help
easel fonts            # the fonts easel finds, Latin and CJK
```

## Uninstall

```bash
pip uninstall pixelcanvas
```

Then delete the skill folder (`~/.claude/skills/easel` or `~/.config/opencode/skills/easel`).
