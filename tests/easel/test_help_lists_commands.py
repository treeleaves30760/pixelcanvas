"""`easel help` is the overview of the tool: every command in the command table is named in it, except the aliases
listed here, which are deliberate."""
import re

from pixelcanvas.easel import cli

# alias -> the command it repeats (named in the overview under that name)
ALIASES = {
    "regions": "region",
    "layers": "layer",
    "changes": "change",
    "recolour": "recolor",
    "clouds": "cloud",
    "stipple": "dot",
    "view": "look",
}


def _help_text() -> str:
    code, out = cli.run(["help"])
    assert code == 0
    return out


def test_every_command_is_in_the_overview_or_a_listed_alias():
    text = _help_text()
    words = set(re.findall(r"[A-Za-z][A-Za-z-]*", text))
    missing = sorted(c for c in cli.COMMANDS if c not in words and c not in ALIASES)
    assert not missing, f"commands missing from `easel help` (add them there, or list them in ALIASES): {missing}"


def test_aliases_point_at_commands_that_are_listed():
    text = _help_text()
    words = set(re.findall(r"[A-Za-z][A-Za-z-]*", text))
    for alias, target in ALIASES.items():
        assert alias in cli.COMMANDS, f"{alias} is no longer a command: drop it from ALIASES"
        assert target in words, f"{alias} repeats {target}, which `easel help` does not name"


def test_the_changes_commands_and_polish_are_named():
    text = _help_text()
    for name in ("change", "recolor", "canvas", "graphic", "cutout", "background", "light", "polish",
                 "region refine", "palette map"):
        assert name in text, name
