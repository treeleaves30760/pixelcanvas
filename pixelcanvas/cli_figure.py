"""`pixelcanvas figure <who> ...`: one command line that poses, dresses, gives hair and a face to a 3D figure.

It expands into ``pose``, ``dress`` (one per piece), ``hair3d`` and ``face`` and runs them as one ``do`` batch, so
the report and the undo steps are those of the batch.

    pixelcanvas figure girl --pose run --yaw 30 --at 500,900 --heads 6.5 --set "head: turn 20"
        --wear "blouse:white,skirt:#2c3460:pleated,thigh-highs:black,shoes:brown"
        --hair "twintails,blunt,#f09ac0,bows:black" --face "smile,#3a78c8"
"""
import shlex
from typing import Dict, List

from .coords import UsageError

EXAMPLE = ('pixelcanvas figure girl --pose run --at 500,900 --yaw 30 --wear "blouse:white,skirt:#2c3460:pleated,'
           'shoes:brown" --hair "twintails,blunt,#f09ac0" --face "smile,#3a78c8"')

POSE_OPTS = {"preset": "value", "pose": "value", "at": "value", "height": "value", "heads": "value",
             "chibi": "flag", "yaw": "value", "set": "value", "mirror": "value", "on": "value", "hand-r": "value",
             "hand-l": "value", "foot-r": "value", "foot-l": "value", "lift": "value", "hand-shape-r": "value",
             "hand-shape-l": "value"}
DRESS_VALUES = {"length": ("short", "knee", "long"), "fit": ("tight", "loose"), "sleeve": ("none", "short", "long"),
                "style": ("pleated", "flared", "pencil", "open", "closed")}
HAIR_STYLES = ("long", "medium", "bob", "short", "spiky", "ponytail", "twintails", "braid", "bun", "buns")
ACCESSORIES = ("hairband", "hairpin", "bow", "ribbon")
GAZES = ("center", "centre", "viewer", "front", "left", "right", "up", "down", "up-left", "up-right", "down-left",
         "down-right")
BANGS = {"blunt": "blunt", "side": "side", "parted": "parted", "none": "none", "swept": "side",
         "side-swept": "side", "straight": "blunt", "hime": "blunt", "center": "parted", "middle": "parted"}
EXPRESSIONS = ("neutral", "calm", "smile", "open", "grin", "laugh", "shout", "angry", "sad", "crying", "teary",
               "surprised", "panicked", "smug", "embarrassed", "sleepy", "closed", "shut", "wink", "happy", "shy")


def _q(v: str) -> str:
    return shlex.quote(str(v))


def wear_cmds(who: str, spec: str, default_color: str = "") -> List[str]:
    """"blouse:white,skirt:#2c3460:pleated:knee,shoes" -> dress commands (one per piece)."""
    out = []
    for item in [s.strip() for s in str(spec).split(",") if s.strip()]:
        parts = [x.strip() for x in item.split(":") if x.strip()]
        piece, opts, color = parts[0], {}, default_color
        for x in parts[1:]:
            xl = x.lower()
            key = next((k for k, vals in DRESS_VALUES.items() if xl in vals), None)
            if xl in ("short", "long") and piece in ("shirt", "blouse", "hoodie") and "sleeve" not in opts:
                key = "sleeve"
            if "=" in x:
                k, v = x.split("=", 1)
                opts[k.strip().lower()] = v.strip()
            elif key and key not in opts:
                opts[key] = xl
            else:
                color = x
        cmd = f"dress {who} {piece}" + (f" --color {_q(color)}" if color else "")
        cmd += "".join(f" --{k} {_q(v)}" for k, v in opts.items())
        out.append(cmd)
    return out


def hair_cmd(who: str, spec: str) -> str:
    """"twintails,blunt,#f09ac0,bows:black,wind:left" -> one hair3d command."""
    opts: Dict[str, str] = {}
    for x in [s.strip() for s in str(spec).split(",") if s.strip()]:
        xl = x.lower()
        if ":" in x and xl.split(":", 1)[0] in ("bows", "bow", "wind", "ahoge", "color", "bangs", "style"):
            k, v = x.split(":", 1)
            k = {"bow": "bows"}.get(k.lower(), k.lower())
            opts[k] = v.strip()
        elif xl in HAIR_STYLES and "style" not in opts:
            opts["style"] = xl
        elif xl in BANGS and "bangs" not in opts:
            opts["bangs"] = BANGS[xl]
        elif xl == "ahoge":
            opts["ahoge"] = "yes"
        elif xl in ACCESSORIES:
            opts["accessory"] = (opts["accessory"] + "," + xl) if opts.get("accessory") else xl
        else:
            opts["color"] = x
    return f"hair3d {who}" + "".join(f" --{k} {_q(v)}" for k, v in opts.items())


def face_cmd(who: str, spec: str) -> str:
    """"smile,#3a78c8" -> one face command."""
    opts: Dict[str, str] = {}
    for x in [s.strip() for s in str(spec).split(",") if s.strip()]:
        xl = x.lower()
        if xl in EXPRESSIONS and "expression" not in opts:
            opts["expression"] = xl
        elif xl in GAZES or xl.startswith("gaze:"):
            opts["gaze"] = xl.split(":", 1)[-1]
        elif ":" in x and not x.startswith("#"):
            k, v = x.split(":", 1)
            opts[{"eye": "eyes"}.get(k.lower(), k.lower())] = v.strip()
        else:
            opts["eyes"] = x
    return f"face {who}" + "".join(f" --{k} {_q(v)}" for k, v in opts.items())


FIGURE_OPTS = ("wear", "dress", "clothes", "outfit", "color", "colour", "hair", "hair3d", "face", "expression",
               "eyes", "gaze")


def figure_cmds(args: List[str]) -> List[str]:
    """Split the figure's own options (--wear --color --hair --face --expression --eyes --gaze) from the rest; every
    other option goes to ``pose`` as given (--pose is --preset), so new pose options work without a change here."""
    args = list(args)
    if not args or args[0].startswith("-"):
        raise UsageError("figure needs a name first: figure <who> --pose <preset> --at x,y ...", EXAMPLE)
    who, rest = args[0], args[1:]
    own: Dict[str, str] = {}
    pose: List[str] = []
    i = 0
    while i < len(rest):
        t = rest[i]
        key, val = (t.lstrip("-").split("=", 1) + [None])[:2] if t.startswith("--") else (None, None)
        key = key.lower() if key else key
        if key in FIGURE_OPTS:
            if val is None:
                if i + 1 >= len(rest):
                    raise UsageError(f"--{key} needs a value", EXAMPLE)
                i += 1
                val = rest[i]
            own[{"dress": "wear", "clothes": "wear", "outfit": "wear", "colour": "color",
                 "hair3d": "hair"}.get(key, key)] = val
        elif key == "pose":
            pose.append("--preset" + ("=" + val if val is not None else ""))
        else:
            pose.append(t)
        i += 1
    cmds = [" ".join(["pose", _q(who)] + [_q(x) if not x.startswith("--") else x for x in pose])]
    if own.get("wear"):
        cmds += wear_cmds(who, own["wear"], own.get("color", ""))
    if own.get("hair"):
        cmds.append(hair_cmd(who, own["hair"]))
    face = own.get("face", "")
    if own.get("expression"):
        face += "," + own["expression"]
    if own.get("eyes"):
        face += ",eyes:" + own["eyes"]
    if own.get("gaze"):
        face += ",gaze:" + own["gaze"]
    if face.strip(","):
        cmds.append(face_cmd(who, face))
    return cmds


def cmd_figure(ctx, args: List[str]):
    from . import cli
    if not args:
        raise UsageError("figure needs a name: figure <who> --pose <preset> --at x,y ...", EXAMPLE)
    cmds = figure_cmds(args)
    rep = cli.COMMANDS["do"](ctx, ["; ".join(cmds)])
    rep.lines.insert(0, f"figure {cmds[0].split()[1]}: {len(cmds)} commands ({', '.join(c.split()[0] for c in cmds)})")
    if not getattr(ctx, "exit_code", 0):
        try:
            if not ctx.project.state.get("draw", {}).get("render"):
                rep.next = "pixelcanvas render   (the guide, once every figure is set)"
        except Exception:  # noqa: BLE001
            pass
    return rep
