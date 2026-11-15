# Installing pixelcanvas

pixelcanvas runs anywhere Python 3.9+ runs, with numpy and Pillow as its only requirements. It ships two skills:
`pixelcanvas` (the studio: a picture as layers edited command by command, delivered in a real art medium such as oil, watercolour or ink, every kit) for coding agents and local
models, and `pixelcanvas-lite` (one JSON spec, drawn by `tools/compose.py`) for the smallest models. Pick the way
that fits your agent; every option is one copy-paste block.

## With pip (quickest)

```bash
pip install pixelcanvas
pixelcanvas install-skill claude          # or: agents, codex, cursor, opencode
uvx pixelcanvas doctor                    # checks Python, numpy, Pillow, fonts and write access
```

`pixelcanvas install-skill TARGET --skill pixelcanvas | pixelcanvas-lite | both` (default `both`) chooses the skills;
`--project` installs into the current project, `--dest DIR` into any folder, `--list` shows every target.

## As an Agent Skill

Any tool that reads Agent Skills (Claude Code, Codex, Gemini CLI, opencode, Cursor ...). The skill fetches the
renderer on first use with `scripts/bootstrap.py`.

```bash
npx skills add treeleaves30760/pixelcanvas
```

## As a Claude Code plugin

Add the marketplace, then install the skill.

```text
/plugin marketplace add treeleaves30760/pixelcanvas
/plugin install pixelcanvas@pixelcanvas
```

## From a clone, into any tool's skills folder

Clone once; `install_skill.py` copies the skills and records where the runtime lives.

```bash
git clone https://github.com/treeleaves30760/pixelcanvas ~/.pixelcanvas
pip install -r ~/.pixelcanvas/requirements.txt
python ~/.pixelcanvas/tools/install_skill.py claude
```

Other targets: `agents` (`~/.agents/skills`, read by Codex, Gemini CLI and opencode), `opencode`
(`~/.config/opencode/skills`), `cursor` (`~/.cursor/skills`), `codex` (`~/.codex/skills`). `--skill pixelcanvas`,
`--skill pixelcanvas-lite` or `--skill both` (the default) picks the skills.

```bash
python ~/.pixelcanvas/tools/install_skill.py agents
python ~/.pixelcanvas/tools/install_skill.py cursor
python ~/.pixelcanvas/tools/install_skill.py --list
```

Install into the current project only (`./.claude/skills`, `./.agents/skills`, ...), or into any folder.

```bash
python ~/.pixelcanvas/tools/install_skill.py claude --project
python ~/.pixelcanvas/tools/install_skill.py --dest /path/to/skills
```

## Small and local models: opencode + the lite skill

A model running on your own machine draws through the `pixelcanvas-lite` skill: it writes one JSON spec, and
`compose.py` checks it, draws it through the kits and ends with one `NEXT:` line that says what to do. No Python
to write. Point opencode at your local model, then install the lite skill for it:

```bash
pip install pixelcanvas
pixelcanvas install-skill opencode --skill pixelcanvas-lite
```

The commands the model runs (from the skill, with the runtime's path filled in):

```bash
pixelcanvas compose --example poster > ./art/poster.json
pixelcanvas compose ./art/poster.json --check
pixelcanvas compose ./art/poster.json --deliver ./art/final/
```

## Local models in opencode: the studio with vision

The full `pixelcanvas` skill runs with local models too. After every change the studio writes one image, the `look`
sheet (the picture, then each checklist item close up), and the model reads it when opencode knows the model takes images. Declare that on the model's entry in
`opencode.json`, under `provider.<id>.models.<model>`, next to the provider settings you already have:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "PROVIDER_ID": {
      "models": {
        "MODEL_ID": {
          "attachment": true,
          "modalities": {"input": ["text", "image"], "output": ["text"]}
        }
      }
    }
  }
}
```

A model without vision checks the picture with `studio look --text`: each layer's box, size and mean colour as numbers.

## The `pixelcanvas` command

The package carries the tools, every starter template, the example scenes and both skills. `scipy` is an optional
speed-up: `pip install "pixelcanvas[fast]"`.

Draw in your own project folder: the scene and its pictures stay there.

```bash
pixelcanvas new poster --kind poster --dir ./art
pixelcanvas render ./art/poster.py --scale 0.5 --check
pixelcanvas render ./art/poster.py --deliver ./site/img/poster.png
```

From a clone without installing, `tools/pc.py` is the same command.

```bash
python ~/.pixelcanvas/tools/pc.py render ./art/poster.py --scale 0.5 --check
```

## As an MCP server

The server speaks the Model Context Protocol on stdin / stdout with the standard library only. A client without a
shell or a file tool (Claude Desktop, Cursor chat) runs the whole studio loop through four tools:

| Tool | Parameters | Returns |
| --- | --- | --- |
| `studio` | `command` (one studio command, e.g. `new fox --prompt "a fox by a lantern" --size 1280x720`, `move fox --to 0.4 0.7`, `set scene medium=oil`, `look`, `note add "..."`, `deliver out.png`), `picture` (optional) | the printed report ending in `NEXT:`, plus the look sheet |
| `studio_draft` | `picture`, `code` (the whole draft, one `# %% name` section per object) | writes `draft.py`, runs `add --from`; report + look sheet |
| `studio_layer` | `picture`, `layer`, `code` (one section body) | writes `layers/<layer>.py`, renders; report + look sheet |
| `studio_read` | `picture`, `layer` (optional) | one section's source, or the layers with their descriptions and the draft |

The loop: `studio` `new NAME --prompt "..." --size WxH`, then `studio_draft` with the whole draft (each section starts
with its inventory line, `# apple: 3, green | in bowl`), then whatever each report's `NEXT:` line says: the look sheet
comes with every report, and after three looks NEXT says the picture is finished. `studio` `api sk.paint` explains any drawing call; `reference` with `page` set to `studio` returns the full
manual. A `studio` command is split like a shell command line (quotes group words, backslashes in Windows paths stay
literal) and runs without a shell. The older single-file scene workflow stays available as `kinds`, `new_scene`,
`write_scene`, `read_scene`, `render` (report + preview image), `look`, `find`, `reference`, `ask`, `animate` and
`layers`, and every reference page is a resource. Pictures live in `art/NAME/` and scenes next to them, in the folder
the host starts the server in (or `$PIXELCANVAS_WORK`). The snippets use `python -m pixelcanvas.mcp_server` after a
pip install; from a clone use `python /path/to/pixelcanvas/tools/pc.py mcp` instead.

Claude Code, for this project (writes `.mcp.json`):

```bash
claude mcp add --scope project pixelcanvas -- python -m pixelcanvas.mcp_server
```

Claude Desktop, in `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "pixelcanvas": {
      "command": "python",
      "args": ["-m", "pixelcanvas.mcp_server"],
      "env": {"PIXELCANVAS_WORK": "/Users/you/Pictures/pixelcanvas"}
    }
  }
}
```

Cursor, in `.cursor/mcp.json` (project) or `~/.cursor/mcp.json` (all projects):

```json
{
  "mcpServers": {
    "pixelcanvas": {
      "command": "python",
      "args": ["-m", "pixelcanvas.mcp_server"]
    }
  }
}
```

opencode, in `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "pixelcanvas": {
      "type": "local",
      "command": ["python", "-m", "pixelcanvas.mcp_server"],
      "enabled": true
    }
  }
}
```

Codex, from the shell:

```bash
codex mcp add pixelcanvas -- python -m pixelcanvas.mcp_server
```

Codex, the same written by hand in `~/.codex/config.toml`:

```toml
[mcp_servers.pixelcanvas]
command = "python"
args = ["-m", "pixelcanvas.mcp_server"]
```

## Settings

| Variable | Effect |
| --- | --- |
| `PIXELCANVAS_WORK` | extra folder searched for scenes by name; default folder for new scenes and for the MCP server |
| `PIXELCANVAS_OUT` | folder for every result instead of `outputs/` next to the scene |
| `PIXELCANVAS_HOME` | where `bootstrap.py` looks for (and downloads) the runtime; default `~/.pixelcanvas` |
| `PIXELCANVAS_VLM` | vision model for `look.py ask`: `ollama:<model>` or `openai:<model>@<base_url>` |
| `PIXELCANVAS_MCP_TIMEOUT` | seconds one MCP tool call may take (default 900) |

`pixelcanvas doctor` (or `python tools/pc.py doctor`) prints the Python, numpy, Pillow and scipy versions, the fonts
it found (CJK included), where results will go, the vision-model setting and a test render.
