"""The stage plan, todo list, gates and ``next`` for a project without a reference picture.

Same stage names and todo item format as ``workflow.py`` (so status, todo, stage done, export --stages keep
working); the commands are the no-reference ones (plan, sketch figure, line, flat auto, shade, hair, eye ...) and
the gates measure the drawing's own state. ``refresh(p)`` ticks items whose work is measurably done, after every
drawing command, so a model that only follows ``NEXT`` gets through.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import coords

PLANS = {
    "cel": ["plan", "sketch", "line", "flat", "shade", "detail", "finish"],
    "ink": ["plan", "sketch", "line", "flat", "detail", "finish"],
    "paint": ["plan", "sketch", "block-in", "forms", "detail", "finish"],
    "photo": ["plan", "sketch", "block-in", "forms", "detail", "finish"],
    "poster": ["plan", "sketch", "flat", "line", "detail", "finish"],
}

WHAT = {
    "plan": "say what the picture is: subjects and parts with boxes, light, palette",
    "sketch": "construction on the sketch layer: mannequin, head, big shapes, horizon",
    "line": "line art: a few points per line, the tool smooths, tapers, snaps and joins",
    "flat": "flat colours in the enclosed areas: flat auto, then fix areas by number",
    "shade": "shading from the plan light: shade, cast shadows, rim light",
    "block-in": "big shapes: sky and land bands, masses, gradients",
    "forms": "brushwork along the forms: paint <region> --dir <angle>",
    "detail": "surface on every area (what it is made of, how the light falls on it), then detail where the viewer "
              "looks: eyes, strands, folds",
    "finish": "background check, finishing touches, export",
}

GATE_WORDS = {
    "plan": ">= 1 subject with parts, light set, palette chosen",
    "sketch": "every subject has construction",
    "line": "every focus part has lines, no small gaps",
    "flat": "every area filled",
    "shade": "every coloured part shaded (or skipped with a reason)",
    "block-in": ">= 97% of the canvas painted",
    "forms": "every block-in region has brushwork",
    "detail": "every area has its surface, every focus-3 part has detail, the picture's detail measures inside the "
              "professional range",
    "finish": "picture exported",
}

HAIR_WORDS = ("hair", "bangs", "fringe", "ponytail", "braid", "fur", "mane", "tail")
CLOTH_WORDS = ("scarf", "coat", "shirt", "dress", "robe", "skirt", "jacket", "cape", "cloak", "sleeve", "hood",
               "sweater", "pants", "trousers", "kimono", "uniform", "clothes", "shirt", "blouse")


GROUND_WORDS = ("sky", "lake", "sea", "water", "ocean", "river", "shore", "ground", "field", "grass", "beach",
                "sand", "snow", "floor", "meadow", "desert", "clouds", "background")


def is_ground(name: str) -> bool:
    """A subject that is a band of the scene, placed by the horizon rather than a construction shape."""
    return any(w in name for w in GROUND_WORDS)


SCENERY_WORDS = ("cliff", "cliffs", "mountain", "mountains", "hill", "hills", "rock", "rocks", "ridge", "dune",
                 "dunes", "forest", "treeline", "horizon")


def is_scenery(state: dict, part: str) -> bool:
    """A part that is a band of the scene (sky, sea, a cliff, the land of a backdrop subject): it is laid as a
    colour band, not outlined."""
    if is_ground(part) or any(t in SCENERY_WORDS for t in re.split(r"[-_\s]+", part.lower())):
        return True
    sub = _subjects(state).get(_parts(state).get(part, {}).get("of") or "", {})
    b = sub.get("box")
    if not b:
        return False
    backdrop = (b[2] - b[0]) * (b[3] - b[1]) >= 0.8e6
    return backdrop and not any(v.get("of") == _parts(state)[part].get("of") and
                                any(w in n for w in ("head", "face", "eye", "arm", "leg", "hand", "torso"))
                                for n, v in _parts(state).items())


LOOSE_BAND_AREA = 0.02         # share of the canvas from which a band with no subject gets brushwork of its own


def band_owner(state: dict, name: str) -> Optional[str]:
    """The subject a band belongs to: a part's subject, the subject its name starts with (pines-2, cat-shadow),
    else the smallest subject box that holds its middle."""
    subs = _subjects(state)
    if name in subs:
        return name
    prt = _parts(state).get(name)
    if prt is not None:
        return prt.get("of")
    pre = [s for s in subs if name.startswith(s + "-") or name.startswith(s + "_")]
    if pre:
        return max(pre, key=len)
    b = state.get("regions", {}).get(name, {}).get("box")
    if not b:
        return None
    c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    hits = sorted(((v["box"][2] - v["box"][0]) * (v["box"][3] - v["box"][1]), s) for s, v in subs.items()
                  if v["box"][0] <= c[0] <= v["box"][2] and v["box"][1] <= c[1] <= v["box"][3])
    return hits[0][1] if hits else None


def is_draw(obj) -> bool:
    """True for a project (or its state) without a reference picture, or in hand mode (draw with a reference)."""
    st = obj if isinstance(obj, dict) else getattr(obj, "state", {})
    return st.get("mode") == "hand" or not st.get("has_ref", st.get("reference") is not None)


def plan_of(state: dict) -> List[str]:
    return PLANS[state.get("style") or "paint"]


def _parts(state: dict) -> Dict[str, dict]:
    return state.get("draw", {}).get("parts", {})


def _subjects(state: dict) -> Dict[str, dict]:
    return state.get("draw", {}).get("subjects", {})


def _ordered_parts(state: dict) -> List[str]:
    parts = _parts(state)
    return sorted(parts, key=lambda n: (-parts[n].get("focus", 2), n))


LOCK_WORDS = ("hair", "bangs", "fringe", "ponytail", "braid", "pigtail")
FIGURE_PARTS = ("face", "head", "arm", "arms", "leg", "legs", "torso", "body", "hand", "hands")


def _tokens(name: str) -> List[str]:
    return [t for t in re.split(r"[-_\s]+", name.lower()) if t]


def wear_piece(name: str) -> Optional[str]:
    """The ``pixelcanvas wear`` piece (coat, scarf, hood ...) or "hand" that a planned part name means, else None."""
    from .draw_wear import PART_WORDS, PIECE_ALIAS, PIECES
    toks = _tokens(name)
    for t in toks:
        t = PIECE_ALIAS.get(t, t)
        if t in PIECES:
            return t
    for piece, words in PART_WORDS.items():
        if any(t in words for t in toks):
            return piece
    return None


def hand_side(name: str) -> str:
    toks = _tokens(name)
    if any(t in ("l", "left") for t in toks):
        return "l"
    if any(t in ("r", "right") for t in toks):
        return "r"
    return "l|r"


def is_lock_part(name: str) -> bool:
    """A part drawn with  pixelcanvas locks  (hair; not fur, a mane or a tail)."""
    return any(w in t for t in _tokens(name) for w in LOCK_WORDS)


def detail_command(name: str, focus: int, locks: bool = False) -> Tuple[str, str]:
    base = name.lower()
    if locks:
        return f"shadow, shine and strands along the locks of {name}", f"pixelcanvas locks shine {name}"
    if any(w in base for w in HAIR_WORDS):
        return f"hair strands and shine on {name}", f'pixelcanvas hair {name} --flow "x,y x,y x,y"'
    if base.startswith("eye"):
        return f"render {name}: iris, pupil, catch-lights", f"pixelcanvas eye {name}"
    if base in ("face", "head", "cheek"):
        return f"blush and a highlight on {name}", f"pixelcanvas blush {name}"
    if any(w in base for w in CLOTH_WORDS):
        return f"folds on {name}", f'pixelcanvas fold "x,y x,y x,y" --region {name}'
    return f"detail on {name}: highlight, pattern or scatter", f"pixelcanvas highlight {name}"


def face_item(state: dict, part: str) -> Tuple[str, str]:
    """The todo item for the face features of a head part: mouth, nose and brows on the construction's guides (an
    animal: muzzle, nose and mouth)."""
    from .draw_checks import ANIMAL_WORDS
    d = state.get("draw", {})
    subj = d.get("parts", {}).get(part, {}).get("of") or ""
    req = str(state.get("request") or "").lower()
    animal = any(w in subj for w in ANIMAL_WORDS) or (len(d.get("subjects", {})) == 1 and
                                                      any(w in req.split() for w in ANIMAL_WORDS))
    hid = next((e["id"] for e in reversed(d.get("sketch", [])) if e["kind"] == "head"
                and (not e.get("subject") or e.get("subject") == subj)), None)
    if subj in rigged_subjects(state) and not animal:
        return (f"face of {subj} on the 3D head: brows, eyes, nose, mouth (it turns with the head)",
                f"pixelcanvas face {subj} --expression neutral   (smile, open, grin, angry, sad, crying, surprised, "
                f"panicked, sleepy; --eyes <colour>)")
    if animal:
        text = f"muzzle, nose and mouth on {part} (an animal face: no human nose and mouth)"
        cmd = (f"pixelcanvas face {hid} --kind animal" if hid else
               f'pixelcanvas line "x,y x,y x,y" for the muzzle round the nose and mouth in {part}')
    else:
        text = f"mouth, nose and brows on {part}, on the face guides"
        cmd = f"pixelcanvas face {hid}" if hid else f'pixelcanvas line "x,y x,y x,y" for each of mouth, nose and brows in {part}'
    return text, cmd + "   (each feature is an ordinary line you can move or redo)"


def is_animal_state(state: dict, subject: str) -> bool:
    """``draw_checks.is_animal`` on a state (an animal subject, or the one subject of a request about an animal)."""
    from .draw_checks import ANIMAL_WORDS, FIGURE_WORDS
    s = (subject or "").lower()
    if any(w in s for w in ANIMAL_WORDS):
        return True
    req = str(state.get("request") or "").lower()
    return len(_subjects(state)) == 1 and any(w in req.split() or w + "," in req or w + "s" in req.split()
                                              for w in ANIMAL_WORDS) and not any(w in s for w in FIGURE_WORDS)


def _plain_animal(state: dict, subject: str) -> bool:
    """An animal that is not dressed and holds nothing: no mannequin, no human proportions."""
    if not is_animal_state(state, subject):
        return False
    return not any(v.get("of") == subject and wear_piece(n) is not None and "paw" not in n
                   for n, v in _parts(state).items())


SCENE_BLOCK_WORDS = ("street", "bench", "wall", "stairs", "steps", "staircase", "chair", "sofa", "couch", "table",
                     "railing", "fence", "ledge", "road", "room", "building", "city", "roof", "bridge", "station",
                     "platform", "classroom", "alley", "lamp", "pole", "pillar", "desk", "bed", "crate")

TEXT_WORDS = ("title", "text", "logo", "letter", "caption", "words", "headline")


def is_text(name: str) -> bool:
    """A part that is lettering: it is set with  pixelcanvas text, not inked, flatted or shaded."""
    return any(w in name for w in TEXT_WORDS)


def rigged_subjects(state: dict) -> List[str]:
    """Subjects posed in 3D (their clothes, hair and face are rendered from the 3D figure)."""
    return [el.get("subject") for el in state.get("draw", {}).get("sketch", [])
            if el.get("kind") == "figure" and el.get("rig") and el.get("rig_joints") == el.get("joints")]


def flat_mannequins(state: dict) -> List[str]:
    return [el.get("subject") for el in state.get("draw", {}).get("sketch", [])
            if el.get("kind") == "figure" and not el.get("rig")]


def build_todo(state: dict) -> None:
    plan = plan_of(state)
    old = {(t["stage"], t.get("region"), t["kind"], t.get("key")): t for t in state.get("todo", [])}
    user = [t for t in state.get("todo", []) if t["kind"] == "user"]
    items: List[dict] = []
    parts = _parts(state)
    subjects = _subjects(state)
    painterly = (state.get("style") or "paint") in ("paint", "photo")
    # with a reference the figure is in the picture already: no mannequin, clothes on it, cast shadow or folds
    ref = state.get("mode") == "hand" or bool(state.get("has_ref"))

    def add(stage, text, cmd, region=None, key=None, kind="task"):
        prev = old.get((stage, region, kind, key))
        items.append({"stage": stage, "region": region, "kind": kind, "key": key, "text": text, "cmd": cmd,
                      "status": prev["status"] if prev else "open", "note": prev.get("note", "") if prev else ""})

    for stage in plan:
        if stage == "plan":
            add("plan", "subjects: one box per thing the request names", "pixelcanvas plan subject <name> --box x0,y0,x1,y1",
                key="subjects")
            if not painterly:
                add("plan", "parts of each subject with boxes (focus 3: face, eyes, hands)",
                    "pixelcanvas plan part <name> --box x0,y0,x1,y1", key="parts")
            add("plan", "light direction", "pixelcanvas plan light top-left", key="light")
            add("plan", "palette: make, look, pick one", 'pixelcanvas palette make --base "#rrggbb"', key="palette")
        elif stage == "sketch":
            for s in subjects:
                # body parts by word (a title's "heading" is not a head)
                figure = not is_text(s) and any(
                    parts[n]["of"] == s and any(t in FIGURE_PARTS or t.endswith(("arm", "leg", "hand"))
                                                for t in _tokens(n)) for n in parts)
                if figure and ref:
                    if any(parts[n]["of"] == s and n in ("face", "head") for n in parts):
                        add("sketch", f"head construction for {s}", f"pixelcanvas sketch head {s}", region=s, key="head")
                elif figure and _plain_animal(state, s):
                    # an animal on its own legs (not dressed, no hands): its body and head shapes, no mannequin
                    b = subjects[s]["box"]
                    add("sketch", f"big shapes of {s}: body and head", f"pixelcanvas sketch ellipse {coords.fmt_box(b)}"
                        f"   (the body; a curve for a curled or lying pose)", region=s, key="construct")
                    if any(parts[n]["of"] == s and n in ("face", "head") for n in parts):
                        add("sketch", f"head construction for {s}", f"pixelcanvas sketch head {s}", region=s, key="head")
                elif figure:
                    if not any(t.get("key") == "camera" for t in items):
                        add("sketch", "camera: horizon and eye height of the scene",
                            "pixelcanvas camera --horizon 450 --eye-height 1.5   (low angle: --horizon 750 "
                            "--eye-height 0.7; high: --eye-height 4)", key="camera")
                    req = str(state.get("request") or "").lower()
                    if not any(t.get("key") == "blocks" for t in items) and \
                            any(w in req for w in SCENE_BLOCK_WORDS):
                        add("sketch", "blocks the figures stand, sit or lean on (a bench, a wall, stairs, a building)",
                            "pixelcanvas block bench box --at x,y --size 1.5,0.45,0.45   (then pixelcanvas pose "
                            "<subject> --preset sit-chair --on bench)", key="blocks")
                    add("sketch", f"mannequin for {s}: a 3D pose",
                        f"pixelcanvas pose {s} --preset stand   (pixelcanvas pose list shows run, jump, sit-ledge ...; "
                        f"a flat mannequin: pixelcanvas sketch figure {s} --pose stand)", region=s, key="construct")
                    if s not in flat_mannequins(state):
                        add("sketch", f"clothes for {s}: garments on the 3D figure",
                            f"pixelcanvas dress {s} blouse,skirt,socks,shoes --color <name>   (pixelcanvas dress list; "
                            f"one piece at a time takes --length, --fit, --sleeve, --style)", region=s, key="dress")
                        add("sketch", f"hair for {s}: 3D locks",
                            f"pixelcanvas hair3d {s} --style long --bangs blunt --color <name>   (long, medium, bob, "
                            f"short, spiky, ponytail, twintails, braid, bun)", region=s, key="hair3d")
                    if s in flat_mannequins(state) and any(parts[n]["of"] == s and n in ("face", "head")
                                                           for n in parts):
                        # a flat mannequin's face is laid on a head construction; a 3D figure has its 3D head
                        add("sketch", f"head construction for {s}", f"pixelcanvas sketch head {s}", region=s, key="head")
                elif is_ground(s):
                    continue            # placed by the horizon line
                else:
                    b = subjects[s]["box"]
                    add("sketch", f"big shape of {s}", f'pixelcanvas sketch curve "x,y x,y x,y ..." (its silhouette in '
                        f'{coords.fmt_box(b)}) or pixelcanvas sketch ellipse {coords.fmt_box(b)}', region=s,
                        key="construct")
            if painterly or any(is_ground(s) for s in subjects):
                add("sketch", "horizon line", "pixelcanvas sketch horizon 500", key="horizon")
        elif stage == "line":
            poster = state.get("style") == "poster"
            rig3 = [s for s in rigged_subjects(state) if s]
            if rig3 and not ref:
                add("line", "render the 3D figures and blocks: line art, flat colours and shading in one step",
                    "pixelcanvas render   (then look; a re-pose, dress or hair3d renders again by itself)",
                    key="render")
            for n in _ordered_parts(state):
                if is_text(n) or poster:
                    continue        # lettering is set with pixelcanvas text; poster shapes need no outline
                if not ref and is_scenery(state, n):
                    continue        # sky, sea, a cliff: colour bands in the flat stage, not outlines
                # every part needs an outline of its own, or flat cannot give it an area (a torso left unlined
                # merges with the background)
                if True:
                    piece = None if ref else wear_piece(n)
                    subj = parts[n].get("of") or "<subject>"
                    if piece is None and not ref and _tokens(n)[:1] in (["torso"], ["body"], ["chest"]) and \
                            not any(wear_piece(m) not in (None, "hand", "scarf", "collar") for m in parts
                                    if parts[m].get("of") == subj and m != n):
                        piece = "shirt"     # a dressed torso: the clothes drawn on the mannequin close its outline
                    if piece == "hand":
                        cmd = (f"pixelcanvas hand {subj} {hand_side(n)}   (the hand drawn on the mannequin's arm; "
                               f"--grip x,y round a held thing)")
                    elif piece is not None:
                        cmd = f"pixelcanvas wear {subj} {piece}   (the {piece} drawn on the mannequin, before the hair)"
                    elif is_lock_part(n):
                        cmd = f"pixelcanvas locks {n} --cut long   (hair as locks, after the clothes)"
                    elif n.startswith("eye"):
                        eyes = ",".join(e for e in _ordered_parts(state) if e.startswith("eye"))
                        cmd = f"pixelcanvas line eye {eyes}   (outline and lash line of the eyes in their boxes)"
                    elif n in ("face", "head") and any(e["kind"] == "head" for e in state.get("draw", {})
                                                      .get("sketch", [])):
                        hid = next(e["id"] for e in state["draw"]["sketch"] if e["kind"] == "head")
                        cmd = f"pixelcanvas line trace {hid}   (the jaw and chin from the head construction)"
                    else:
                        cmd = f'pixelcanvas line "x,y x,y x,y ..."   (along the {n} outline)'
                    if parts[n].get("of") in rig3 and not ref:
                        if parts[n].get("r3d"):
                            continue            # a material of the render: it has its outline already
                        cmd = f"pixelcanvas render   (draws {n} from the 3D figure)"
                    add("line", f"line {n}", cmd, region=n, key="line")
                    if n in ("face", "head") and not ref and not poster:
                        add("line", *face_item(state, n), region=n, key="face")
            add("line", "close the small gaps", "pixelcanvas line gaps", key="gaps")
        elif stage == "flat":
            add("flat", "flat colours from the plan", "pixelcanvas flat auto", key="auto")
            add("flat", "fill the areas flat auto left open, fix wrong ones", "pixelcanvas flat", key="fix")
            if "line" in plan and not ref:
                for n in _ordered_parts(state):
                    if is_scenery(state, n) and not is_text(n):
                        side = "--above" if "sky" in n else "--closed"
                        add("flat", f"{n} as a colour band", f'pixelcanvas band {n} "x,y x,y x,y ..." {side} --color {n}',
                            region=n, key="mass")
        elif stage == "shade":
            for n in sorted(parts, key=lambda k: -parts[k].get("focus", 2)):
                if not is_text(n) and (ref or not is_scenery(state, n)):     # sky and sea bands carry their light
                    add("shade", f"shade {n}", f"pixelcanvas shade {n}", region=n, key="shade")
            hair = [n for n in parts if any(w in n for w in ("hair", "bangs", "fringe"))]
            face = [n for n in parts if n in ("face", "head", "neck")]
            if hair and face and not ref:
                add("shade", f"{hair[0]} casts a shadow on {face[0]}",
                    f"pixelcanvas shadow --from {hair[0]} --onto {face[0]}", key="cast")
        elif stage == "block-in":
            add("block-in", "sky / background and land bands", 'pixelcanvas band sky "0,450 500,430 1000,460" --above '
                '--color sky', key="bands")
            for s in subjects:
                if is_ground(s):
                    cmd = f'pixelcanvas band {s} "x,y x,y x,y ..." --below --color {s}   (its top edge, left to right)'
                else:
                    cmd = f'pixelcanvas band {s} "x,y x,y x,y ..." --closed --color {s}   (its silhouette)'
                add("block-in", f"big shape of {s}", cmd, region=s, key="mass")
        elif stage == "forms":
            # every subject, and every big band that belongs to no subject; a band of a subject (its ears, its
            # shadow, the hills in its window) is part of that subject's brushwork, and a small loose band (an
            # accent) needs none of its own
            loose = [b for b in state.get("draw", {}).get("bands", {}) if b not in subjects
                     and band_owner(state, b) is None
                     and float(state.get("regions", {}).get(b, {}).get("area", 0.0)) >= LOOSE_BAND_AREA]
            for s in list(subjects) + loose:
                if is_text(s):
                    continue        # lettering is set with pixelcanvas text, not painted
                add("forms", f"brushwork on {s}", f"pixelcanvas paint {s} --dir 0", region=s, key="forms")
        elif stage == "detail":
            add("detail", "surface on every area: materials chosen from the part names, marks at every scale",
                "pixelcanvas surface auto   (then look; change one area:  pixelcanvas surface <area> --material <kind>)",
                key="surface")
            for n in _ordered_parts(state):
                if is_text(n) or n in state.get("draw", {}).get("marks", {}):
                    continue        # lettering is set with pixelcanvas text; a mark (a muzzle patch) is detail itself
                if n in ("face", "head") and painterly and not ref:
                    add("detail", *face_item(state, n), region=n, key="face")
                f = parts[n].get("focus", 2)
                text, cmd = detail_command(n, f, locks=n in state.get("draw", {}).get("locks", {}))
                if ref and cmd.startswith("pixelcanvas fold"):
                    text, cmd = f"small shapes of {n} from the reference", f"pixelcanvas detail candidates {n}"
                if f >= 3 or (f >= 2 and "highlight" not in cmd):      # a plain highlight only where the eye goes
                    add("detail", text, cmd, region=n, key="detail")
                if f >= 3:
                    add("detail", f"look close at {n} and fix what is off", f"pixelcanvas look {n}", region=n,
                        key="closeup")
            for n in parts:
                if is_text(n):
                    b = parts[n]["box"]
                    add("detail", f"set the lettering of {n}", f'pixelcanvas text "WORDS" --box {coords.fmt_box(b)} '
                        f'--color {parts[n].get("swatch") or "line"}', region=n, key="text")
            if painterly or not parts:
                add("detail", "accents: scatter, texture, highlights where the eye goes",
                    'pixelcanvas scatter <region> --kind leaf --count 60', key="accents")
        elif stage == "finish":
            add("finish", "look at the whole picture; fix what is off", "pixelcanvas look", key="look")
            if not ref:
                add("finish", "finishing pass: one light over the picture, far things set back, coloured lines, "
                    "strays out", "pixelcanvas polish   (each part can be left out: --no-grade --no-depth ...)",
                    key="polish")
            add("finish", "export (PNG; --stages stages for the stage sheet)",
                "pixelcanvas export final.png --stages stages", key="export")
        add(stage, f"gate {stage}: {GATE_WORDS.get(stage, '')}", "pixelcanvas stage done", key="gate", kind="gate")
    items.extend(user)
    for i, t in enumerate(items, 1):
        t["id"] = i
    state["todo"] = items


def plan_lines(state: dict) -> List[str]:
    out = []
    for k, stage in enumerate(plan_of(state), 1):
        items = [t for t in state["todo"] if t["stage"] == stage and t["kind"] in ("task", "user")]
        done = stage in state.get("stages_done", [])
        mark = "done" if done else ("in progress" if stage == state["stage"] else "open")
        cmds = "; ".join(dict.fromkeys(t["cmd"].split("   (")[0] for t in items))[:220]
        out.append(f"{k}. [{mark}] {stage}: {WHAT.get(stage, '')} -> {cmds} -> gate: {GATE_WORDS.get(stage, '')}")
    return out


# ============================================================================================ measured state

def _region_names(p) -> List[str]:
    return list(p.state.get("regions", {}).keys())


def _layer_cov(p, layer: str, mask: np.ndarray, thr: float = 0.2) -> float:
    arr = p.layer(layer, create=False)
    if arr is None or not mask.any():
        return 0.0
    return float((arr[..., 3][mask] > thr).mean())


def _small_gaps(p) -> List[dict]:
    from . import draw_line
    d = p.state.get("draw", {})
    if not d.get("lines"):
        return []
    an = draw_line.analyze(p)
    return [g for g in an["gaps"] if g["dist"] < 0.025 * max(p.W, p.H)]


def part_has_lines(p, name: str) -> bool:
    d = p.state.get("draw", {})
    return any(r.get("part") == name for r in d.get("lines", []))


def refresh(p) -> List[int]:
    """Tick open items of the current stage whose work is measurably done. Returns the ticked ids."""
    from .workflow import tick
    st = p.state
    stage = st["stage"]
    d = st.get("draw", {})
    done: List[int] = []
    if is_draw(st) and st.get("todo") is not None:
        # the list is a function of the picture: the same items as a fresh build (bands drawn since the last plan
        # change join, statuses carry over)
        build_todo(st)

    def ok(item) -> bool:
        k = item.get("key")
        r = item.get("region")
        if stage == "plan":
            if k == "subjects":
                return bool(d.get("subjects"))
            if k == "parts":
                return bool(d.get("subjects")) and all(any(v["of"] == s for v in d.get("parts", {}).values())
                                                       for s in d["subjects"])
            if k == "light":
                return bool(d.get("light", {}).get("set"))
            if k == "palette":
                return bool(d.get("palette", {}).get("swatches"))
        if stage == "sketch":
            from .draw_checks import _centre_in
            if k == "construct" and r in d.get("subjects", {}) and is_ground(r) and                     any(el["kind"] == "horizon" for el in d.get("sketch", [])):
                return True             # sky, water and ground are placed by the horizon
            if k in ("construct", "head") and r in d.get("subjects", {}):
                kinds = ("figure",) if "mannequin" in item["text"] else ("head",) if k == "head" else None
                for el in d.get("sketch", []):
                    if (kinds is None or el["kind"] in kinds) and (el.get("subject") == r or
                                                                    _centre_in(el, d["subjects"][r]["box"])):
                        if kinds is None or el["kind"] in kinds:
                            return True
                return False
            if k == "horizon":
                return any(el["kind"] == "horizon" for el in d.get("sketch", []))
            if k == "camera":       # set, or a figure is placed (posing one sets the default camera)
                return any(el["kind"] == "horizon" and el.get("camera") or el["kind"] == "figure"
                           for el in d.get("sketch", []))
            if k == "blocks":       # a block, or the figures are flat mannequins (no 3D scene to stand in)
                return any(el["kind"] == "block" for el in d.get("sketch", [])) or \
                    (any(el["kind"] == "figure" for el in d.get("sketch", [])) and
                     not any(el["kind"] == "figure" and el.get("rig") for el in d.get("sketch", [])))
        if k in ("dress", "hair3d") and r:
            figs = [el for el in d.get("sketch", []) if el.get("kind") == "figure" and el.get("subject") == r]
            if figs and not figs[0].get("rig"):
                return True             # a flat mannequin: clothes and hair are drawn on it in the line stage
            lk = (figs[0].get("look") or {}) if figs else {}
            return bool(lk.get("garments")) if k == "dress" else bool(lk.get("hair"))
        if k == "render":
            from .draw_render import is_stale
            return bool(d.get("render")) and not is_stale(p)
        subj_r = d.get("parts", {}).get(r, {}).get("of") if r else None
        if k == "face" and subj_r in rigged_subjects(st):
            fig = next(el for el in d.get("sketch", []) if el.get("kind") == "figure" and el.get("subject") == subj_r)
            return bool((fig.get("look") or {}).get("face")) and bool(d.get("render"))
        if r and str(r).startswith("eye") and subj_r in rigged_subjects(st) and k in ("line", "detail", "shade"):
            fig = next(el for el in d.get("sketch", []) if el.get("kind") == "figure" and el.get("subject") == subj_r)
            return bool((fig.get("look") or {}).get("face")) and bool(d.get("render"))
        if d.get("render") and subj_r in rigged_subjects(st) and r in st.get("regions", {}) and \
                str(st["regions"][r].get("how", "")).startswith("3D render"):
            if stage == "shade" and k == "shade":
                return True             # shaded by the render (a part turned to the light has no shadow shape)
            if stage == "detail" and k == "detail" and r in ("face", "head"):
                fig = next(el for el in d.get("sketch", []) if el.get("kind") == "figure" and el.get("subject") == subj_r)
                return bool((fig.get("look") or {}).get("face"))    # blush and shine come with the 3D face
        if stage == "shade" and k == "cast" and d.get("render") and any(
                "hair" in n and v.get("of") in rigged_subjects(st) for n, v in d.get("parts", {}).items()):
            return True                 # the render casts the hair's shadow on the face from the light
        if k == "face":
            from .draw_checks import face_missing, head_elements
            els = [el for el in head_elements(p) if el.get("subject") == d.get("parts", {}).get(r, {}).get("of")]
            return bool(els) and not any(face_missing(p, el) for el in els)
        if stage == "line":
            if k == "line":
                if subj_r in rigged_subjects(st) and d.get("render"):
                    return True         # drawn by the render from the 3D figure
                return part_has_lines(p, r)
            if k == "gaps":
                others = [t for t in st.get("todo", []) if t["stage"] == "line" and t.get("key") == "line"
                          and t["status"] == "open" and t is not item]
                any_line_items = any(t["stage"] == "line" and t.get("key") == "line" for t in st.get("todo", []))
                return (bool(d.get("lines")) or not any_line_items) and not others and not _small_gaps(p)
        if stage == "flat":
            from . import draw_flat
            if k == "auto":
                return bool(d.get("fills"))
            if k == "mass":
                return r in d.get("bands", {}) or r in st.get("regions", {})
            if k == "fix":
                labels, table, _ = draw_flat.compute_areas(p)
                un = [t for t in draw_flat.unfilled(p, labels, table) if t["share"] > 0.0004]
                return bool(d.get("fills")) and not un
        if stage == "shade":
            if k == "shade" and r in st.get("regions", {}):
                return _layer_cov(p, "shade", p.region_mask(r), 0.15) >= 0.03
            if k == "cast":
                return bool(d.get("cast"))
        if stage == "block-in":
            if k == "bands":
                return bool(d.get("bands"))
            if k == "mass":
                return r in d.get("bands", {}) or r in st.get("regions", {})
        if stage == "forms" and k == "forms":
            return r in d.get("painted", [])
        if k == "text":
            return bool(st.get("texts"))
        if stage == "finish" and k == "polish":
            return bool(d.get("polished"))
        if stage == "detail" and k == "surface":
            from .draw_surface import surface_targets
            done_s = d.get("surface", {})
            return bool(done_s) and all(n in done_s for n in surface_targets(p))
        if stage == "detail":
            if k == "detail" and r in st.get("regions", {}):
                if r in d.get("surface", {}) and item["cmd"].startswith(("pixelcanvas fold", "pixelcanvas hair ", "pixelcanvas highlight")):
                    return True         # folds, strands and highlights came with the area's surface
                return r in d.get("detailed", [])
            if k == "accents":
                return bool(d.get("detailed"))
        return False

    regions = st.get("regions", {})
    flat_done = "flat" in st.get("stages_done", []) or "block-in" in st.get("stages_done", [])
    for t in st.get("todo", []):
        if t.get("key") == "face" and t["status"] == "open" and t.get("region") in d.get("parts", {}):
            t["text"], t["cmd"] = face_item(st, t["region"])     # the head construction may have come since
        if t.get("key") in ("bands", "mass") and t["status"] == "open" and "--color sky" in t["cmd"] and \
                "sky-low" in d.get("palette", {}).get("swatches", {}) and "--color2" not in t["cmd"]:
            # a sunset or night sky is a gradient from the sky colour down to the horizon colour
            t["cmd"] = t["cmd"].replace("--color sky", "--color sky --color2 sky-low", 1)
        if t.get("key") == "detail" and t["status"] == "open" and t.get("region") in d.get("locks", {}) \
                and not t["cmd"].startswith("pixelcanvas locks shine"):
            # the hair was drawn as locks since the todo was made: shine along the locks, not loose strands
            t["text"], t["cmd"] = detail_command(t["region"], 2, locks=True)
    for t in st.get("todo", []):
        if t["stage"] != stage or t["kind"] != "task" or t["status"] != "open":
            continue
        if t.get("key") in ("closeup", "look", "export"):
            continue
        r = t.get("region")
        if flat_done and stage in ("shade", "detail", "forms") and r and r in d.get("parts", {}) and r not in regions \
                and t.get("key") != "face":
            t["status"], t["note"] = "skipped", "no coloured area of its own (it is part of a neighbouring area)"
            done.append(t["id"])
            continue
        if stage == "shade" and r and (r.startswith("eye") or d.get("parts", {}).get(r, {}).get("swatch") == "eye-white"):
            t["status"], t["note"] = "skipped", "eyes are shaded by  pixelcanvas eye  in the detail stage"
            done.append(t["id"])
            continue
        try:
            if ok(t):
                t["status"] = "done"
                t["note"] = "measured"
                done.append(t["id"])
        except Exception:  # noqa: BLE001
            continue
    return done


def check_gate(p, stage: str, comparison=None) -> Tuple[bool, List[dict]]:
    st = p.state
    if stage == st.get("stage"):
        try:
            refresh(p)          # work done before the stage opened (an eye drawn while shading) counts
        except Exception:  # noqa: BLE001
            pass
    d = st.get("draw", {})
    unmet: List[dict] = []

    def short(metric, value, need, fix, region=None):
        unmet.append({"region": region, "metric": metric, "value": value, "need": need, "fix": fix})

    skipped = {(t["stage"], t.get("region"), t.get("key")) for t in st.get("todo", []) if t["status"] == "skipped"}
    if stage == "plan":
        if not d.get("subjects"):
            short("subjects", 0, 1, "pixelcanvas plan subject <name> --box x0,y0,x1,y1")
        elif (st.get("style") or "paint") not in ("paint", "photo"):
            for s in d["subjects"]:
                if not any(v["of"] == s for v in d.get("parts", {}).values()):
                    short("parts", 0, 1, f"pixelcanvas plan part <name> --of {s} --box x0,y0,x1,y1", s)
        if not d.get("light", {}).get("set"):
            short("light", "unset", "set", "pixelcanvas plan light top-left")
        if not d.get("palette", {}).get("swatches"):
            short("palette", 0, "chosen", 'pixelcanvas palette make --base "#rrggbb"')
        return not unmet, unmet
    if stage == "finish":
        _add_holes(p, stage, unmet)
        if not st.get("exported"):
            short("exported", "no", "yes", "pixelcanvas export final.png --stages stages")
        elif _export_stale(p):
            short("exported", "before the last changes", "after them", "pixelcanvas export final.png --stages stages")
        return not unmet, unmet
    for t in st.get("todo", []):
        if t["stage"] != stage or t["kind"] != "task" or t["status"] != "open":
            continue
        if t.get("key") in ("closeup", "dress", "hair3d"):
            continue                # clothes and hair in 3D are proposed; the 2D path (wear, locks) stays open
        if stage == "line" and t.get("key") == "gaps":
            gaps = _small_gaps(p)
            if gaps:
                short("small gaps", len(gaps), 0, f"pixelcanvas line close {gaps[0]['id']}")
            continue
        if stage == "flat" and t.get("key") == "fix":
            from . import draw_flat
            labels, table, _ = draw_flat.compute_areas(p)
            un = [r for r in draw_flat.unfilled(p, labels, table) if r["share"] > 0.0004]
            if un:
                short("unfilled areas", len(un), 0,
                      f"pixelcanvas flat {','.join(str(r['n']) for r in un[:6])} --color <name>")
            continue
        if stage == "block-in":
            cov = 0.0
            for lay in ("flat", "paint"):
                arr = p.layer(lay, create=False)
                if arr is not None:
                    cov = max(cov, float((arr[..., 3] > 0.5).mean()))
            if cov < 0.97:
                short("painted %", int(cov * 100), 97, 'pixelcanvas band <name> "x,y ..." --below --color <name>')
                break
            continue
        short(f"todo {t['id']} ({t['text']})", "open", "done", t["cmd"].split("   (")[0], None)
    if st.get("mode") == "hand":        # hand mode: scores against the reference
        from .draw_ref import gate
        unmet.extend(gate(p, stage))
    else:
        _add_holes(p, stage, unmet)
    return not unmet, unmet


def _export_stale(p) -> bool:
    """True when the canvas changed after the last export (the exported file no longer shows the picture)."""
    import json
    jf = p.root / "journal.jsonl"
    try:
        lines = jf.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    for ln in reversed(lines):
        try:
            e = json.loads(ln)
        except ValueError:
            continue
        cmd = str(e.get("cmd") or "")
        if "export" in cmd.split()[:2]:
            return False
        if e.get("ops") or e.get("layer") or "undo" in e:
            return True
    return False


def _add_holes(p, stage: str, unmet: List[dict]) -> None:
    """The measured holes of the stage (``draw_checks.holes``: what the FIX lines print) join the gate."""
    from .draw_checks import holes
    have = {u["metric"] for u in unmet}
    for h in holes(p, stage):
        if h.get("soft"):
            continue
        if h["metric"] in ("small gaps",) and "small gaps" in have:
            continue
        if h["metric"] == "unfilled areas" and any(m.startswith("unfilled") for m in have):
            continue
        # the shell's stage report prints "<region> <metric> <value> < <need>   raise it: <fix>": the metric
        # carries the measured sentence, so the model reads what is wrong and where
        said = h["text"].split(":  pixelcanvas")[0]
        unmet.append({"region": None, "metric": said, "value": h.get("value", ""), "need": h.get("need", ""),
                      "fix": h["fix"], "text": h["text"], "kind": h["metric"]})


def next_action(p, comparison=None) -> Tuple[str, str]:
    st = p.state
    try:
        refresh(p)
    except Exception:  # noqa: BLE001
        pass
    if st.get("finished"):
        return ("the picture is finished" + (f" and exported to {st['exported']}" if st.get("exported") else ""),
                "pixelcanvas look")
    stage = st["stage"]
    d = st.get("draw", {})
    if st.get("mode") == "hand" and stage in ("flat", "shade", "detail"):
        from .draw_ref import split_fix
        try:
            sf = split_fix(p)
        except Exception:  # noqa: BLE001
            sf = None
        if sf:
            return sf
    if stage == "plan":
        if not d.get("subjects"):
            return ("no reference: first say what the picture shows, one box per subject (0-1000, leave room "
                    "around it)", "pixelcanvas plan subject <name> --box x0,y0,x1,y1")
    if d.get("render") and stage not in ("plan", "sketch"):
        try:
            from .draw_render import is_stale
            if is_stale(p):
                return ("the 3D scene changed since the last render: draw it again (the face and touch-layer marks "
                        "stay)", "pixelcanvas render")
        except Exception:  # noqa: BLE001
            pass
        if d.get("palette", {}).get("candidates") and not d["palette"].get("swatches"):
            return "four palettes are proposed; pick the one that fits the request", "pixelcanvas palette pick <n>"
    for t in st.get("todo", []):
        if t["stage"] == stage and t["kind"] in ("task", "user") and t["status"] == "open":
            if stage == "finish" and t.get("key") == "export":
                from .draw_checks import holes
                hs = [h for h in holes(p, stage) if not h.get("soft")]
                if hs:
                    return f"finish: {hs[0]['text']}", hs[0]["fix"]
            cmd = t["cmd"]
            why = f"{WHAT.get(stage, stage)} (todo {t['id']}: {t['text']})"
            if t.get("key") == "parts" and d.get("subjects"):
                s = next(iter(d["subjects"]))
                why = (f"parts of {s}: one box each for head / face / hair / eyes / torso / arms / hands ... "
                       "(focus 3 = where the viewer looks first)")
            if stage == "line" and t.get("key") == "line" and t.get("region") in d.get("parts", {}):
                b = d["parts"][t["region"]]["box"]
                why = (f"ink {t['region']} (box {coords.fmt_box(b)}): 3-8 points per line along its outline; the "
                       "tool smooths, tapers and snaps them to the sketch")
            return why, cmd
    ok, unmet = check_gate(p, stage, comparison)
    if ok:
        return f"every {stage} item is done and the gate passes", "pixelcanvas stage done"
    u = unmet[0]
    if u.get("text"):
        return f"gate {stage}: {u['text']}", u["fix"]
    return f"gate {stage}: {u['metric']} {u['value']} (need {u['need']})", u["fix"]
