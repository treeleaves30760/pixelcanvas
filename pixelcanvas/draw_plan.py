"""Plan without a reference: subjects and their parts with boxes, light direction, a harmonised palette.

The plan is what every later stage builds on: part boxes tell ``flat auto`` which colour an enclosed area gets,
which lines belong to which part, which parts are focus parts; the light drives every shading command; the palette
gives each material a base, a shadow and a light colour.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from . import coords, view
from .coords import UsageError
from .draw_core import (dstate, from_hex, lch_to_rgb, light_of, light_vec, light_words, lum, rgb_to_lch, shadow_of,
                        to_hex)

SKIN_PARTS = ("face", "neck", "arm", "arms", "hand", "hands", "leg", "legs", "foot", "feet", "cheek", "body",
              "torso-skin", "shoulder", "shoulders", "knee", "knees", "skin")
EYE_PARTS = ("eye", "eyes")
FOCUS3_WORDS = ("face", "eye", "eyes", "hand", "hands", "mouth", "title", "lettering")


_SIDE = re.compile(r"(?:[-_](?:l|r|left|right)|[-_]?\d+)$")


def part_base(part: str) -> str:
    return _SIDE.sub("", part)


def default_swatch(part: str, figure: bool = True) -> str:
    """The colour name a part gets when none is given: body-part words of a figure share skin / eye-white; a part of
    anything else (a saxophone's neck, a needle's eye) keeps its own name."""
    base = part_base(part)
    if not figure:
        return base
    if base in SKIN_PARTS or part in SKIN_PARTS:
        return "skin"
    if base in EYE_PARTS or base.startswith("eye"):
        return "eye-white"
    return base


FIGURE_WORDS = {
    "girl", "boy", "man", "woman", "person", "character", "people", "child", "kid", "baby", "lady", "guy", "figure",
    "hero", "heroine", "knight", "mage", "wizard", "witch", "warrior", "soldier", "ninja", "samurai", "princess",
    "prince", "king", "queen", "elf", "fairy", "angel", "idol", "maid", "student", "chibi", "mascot", "portrait",
    "bust", "men", "women", "children", "human", "humanoid", "robot", "android", "cyborg", "zombie", "vampire", "pirate", "cowboy",
    "fox", "cat", "kitten", "dog", "puppy", "wolf", "bear", "rabbit", "bunny", "bird", "dragon", "owl", "lion",
    "tiger", "horse", "deer", "mouse", "monkey", "panda", "penguin", "duck", "frog", "squirrel", "raccoon", "otter",
    "animal", "creature", "monster", "beast", "pet",
}
FIGURE_PART_WORDS = {"head", "face", "eye", "eyes", "hand", "hands", "arm", "arms", "leg", "legs", "torso"}


def _word_forms(w: str) -> set:
    return {w, w.rstrip("s"), w[:-2] if w.endswith("es") else w}


def is_figure(d: dict, subject: Optional[str]) -> bool:
    """A figure (person or animal) by its name, or by already having a head / face / eye / hand / arm / leg / torso."""
    if not subject:
        return False
    tokens = [t for t in re.split(r"[-_\s.]+", str(subject).lower()) if t]
    tokens = [re.sub(r"\d+$", "", t) for t in tokens]
    if any(_word_forms(t) & FIGURE_WORDS for t in tokens):
        return True
    return any(v.get("of") == subject and part_base(n) in FIGURE_PART_WORDS for n, v in d.get("parts", {}).items())


def refresh_default_swatches(d: dict) -> None:
    """Re-decide every defaulted part colour name (never one given with --color): a neck planned before the head
    becomes skin once the head shows the subject is a figure."""
    from .draw_checks import ANIMAL_WORDS
    for n, v in d.get("parts", {}).items():
        if v.get("swatch_auto"):
            fig = is_figure(d, v.get("of"))
            v["swatch"] = default_swatch(n, figure=fig)
            subj = str(v.get("of") or "").lower()
            if fig and part_base(n) == "head" and not any(w in subj for w in ANIMAL_WORDS):
                v["swatch"] = "skin"        # a person's head part is the face: skin (an animal's head is its fur)


def default_focus(part: str) -> int:
    return 3 if part_base(part) in FOCUS3_WORDS else 2


_MODS = {"pale": (88, 0.32), "light": (80, 0.6), "soft": (72, 0.55), "dusty": (60, 0.4), "dark": (-0.58, 1.0),
         "deep": (-0.72, 1.1), "bright": (0, 1.15), "muted": (0, 0.55), "warm": (0, 1.0), "cool": (0, 1.0)}


def tame(rgb) -> Tuple[int, int, int]:
    """A named colour as a painter would mix it: no pure screen primaries, greys a little cool."""
    L, C, h = rgb_to_lch(rgb)
    if C < 3:
        return lch_to_rgb(L, 4.0, 260.0)
    return lch_to_rgb(min(max(L, 30.0), 82.0) if C > 60 else L, min(C, 62.0), h)


def colour_words(text: str) -> Tuple[int, int, int]:
    """'#c04030', 'red', 'pale blue', 'dark green', 'silver' -> rgb; words are tamed and modifiers act in LCh."""
    s = str(text).strip().strip("'\"").lower()
    if s.startswith("#") or re.fullmatch(r"[0-9a-f]{6}", s) or re.search(r"\d", s):
        return coords.parse_colour(s)
    words = re.split(r"[\s_]+|-(?=[a-z])", s)
    mods = [w for w in words if w in _MODS]
    rest = " ".join(w for w in words if w not in _MODS)
    if rest in ("silver", "grey", "gray") and not mods:
        return lch_to_rgb(80.0, 4.0, 265.0)
    base = tame(coords.parse_colour(rest or s))
    L, C, h = rgb_to_lch(base)
    for m in mods:
        a, k = _MODS[m]
        if a > 1:
            L = max(L, a)
        elif a < 0:
            L = L * -a
        elif m == "warm":
            h = (h + ((60 - h + 540) % 360 - 180) * 0.2) % 360
        elif m == "cool":
            h = (h + ((250 - h + 540) % 360 - 180) * 0.2) % 360
        C *= k
    return lch_to_rgb(L, C, h)


def clean(name: str) -> str:
    from .regions import clean_name
    return clean_name(name)


# ============================================================================================ subjects / parts

def set_subject(p, name: str, box) -> str:
    d = dstate(p)
    name = clean(name)
    if name in d["parts"]:
        raise UsageError(f"{name} is already a part; choose another subject name", f"pixelcanvas plan subject girl --box "
                         f"{coords.fmt_box(box)}")
    d["subjects"][name] = {"box": [round(v, 1) for v in box]}
    return name


def set_part(p, name: str, box, of: Optional[str], focus: Optional[int], swatch: Optional[str]) -> str:
    d = dstate(p)
    name = clean(name)
    if "." in str(name):
        name = name.split(".")[-1]
    if of is None:
        subs = list(d["subjects"])
        if not subs:
            raise UsageError("plan a subject first: the parts belong to it",
                             "pixelcanvas plan subject girl --box 250,80,750,1000")
        # the subject whose box holds the part's centre, else the last one
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        inside = [s for s in subs if _in_box((cx, cy), d["subjects"][s]["box"])]
        of = inside[-1] if inside else subs[-1]
    else:
        of = clean(of)
        if of not in d["subjects"]:
            raise UsageError(f"unknown subject {of!r}; subjects: {', '.join(d['subjects']) or 'none yet'}",
                             f"pixelcanvas plan subject {of} --box 250,80,750,1000")
    if name in d["subjects"]:
        raise UsageError(f"{name} is a subject; name the part differently (e.g. {name}-body)",
                         f"pixelcanvas plan part {name}-body --box {coords.fmt_box(box)}")
    prev = d["parts"].get(name, {})
    if swatch:                                          # given with --color: never re-decided
        sw, auto = swatch, False
    elif prev.get("swatch") and not prev.get("swatch_auto"):
        sw, auto = prev["swatch"], False
    else:
        sw, auto = default_swatch(name), True
    d["parts"][name] = {"of": of, "box": [round(v, 1) for v in box],
                        "focus": int(focus if focus is not None else prev.get("focus", default_focus(name))),
                        "swatch": sw}
    if auto:
        d["parts"][name]["swatch_auto"] = True
    poly = PENDING_OUTLINE.pop("poly", None)
    if poly:            # a part made from marks: its outline (units), the region it stands for
        d["parts"][name]["poly"] = poly
    refresh_default_swatches(d)
    return name


PENDING_OUTLINE: dict = {}      # set by  plan part --marks  just before set_part runs


def part_mask(p, name: str) -> Optional[np.ndarray]:
    """The mask of a part made from marks (its outline), or None for a part that is just a box."""
    from .draw_core import poly_mask, u2px
    polys = dstate(p)["parts"].get(name, {}).get("poly")
    if not polys:
        return None
    m = np.zeros((p.H, p.W), bool)
    for q in polys:
        m |= poly_mask(p.H, p.W, u2px(p, q))
    return m


def _in_box(pt, box) -> bool:
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def parts_at(d: dict, pt_u) -> List[str]:
    """Parts whose box holds the point, smallest box first."""
    hits = [(abs((b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1])), n) for n, b in d["parts"].items()
            if _in_box(pt_u, b["box"])]
    return [n for _, n in sorted(hits)]


def subject_at(d: dict, pt_u) -> Optional[str]:
    hits = [(abs((b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1])), n) for n, b in d["subjects"].items()
            if _in_box(pt_u, b["box"])]
    return sorted(hits)[0][1] if hits else None


# ============================================================================================ palette

SCHEMES = ("analogous", "complementary", "triad", "split")
DEFAULTS = {"line": (34, 28, 36), "blush": (240, 140, 140), "eye-white": (246, 246, 250), "iris": (70, 110, 190)}


def needed_swatches(d: dict) -> List[str]:
    names = []
    for prt in d["parts"].values():
        s = prt.get("swatch")
        if s and s not in names:
            names.append(s)
    for sname in d.get("subjects", {}):
        # a subject without parts (sky, lake, a band of trees) is painted as one mass in its own colour
        if not any(v["of"] == sname for v in d["parts"].values()) and sname not in names:
            names.append(sname)
    if "eye-white" in names and "iris" not in names:
        names.append("iris")
    for extra in ("background", "line"):
        if extra not in names:
            names.append(extra)
    return names


_COLOUR_WORD = re.compile(r"([a-z]+(?:[ -][a-z]+)?)\s+([a-z]+)")


def request_colours(request: str, names: List[str]) -> Dict[str, Tuple[int, int, int]]:
    """'long silver hair, a red scarf, pale blue background' -> {hair: silver, scarf: red, background: pale blue}."""
    out: Dict[str, Tuple[int, int, int]] = {}
    req = (request or "").lower()
    words = re.findall(r"[a-z]+", req)
    for name in names:
        key = name.split("-")[0]
        alts = {key, key + "s", key.rstrip("s")}
        if key == "background":
            alts |= {"backdrop", "sky"} if "sky" not in names else {"backdrop"}
        if key == "eye-white" or key == "line" or name == "sky-low":
            continue
        if key == "iris":
            alts = {"eye", "eyes"}
        for i, w in enumerate(words):
            if w in alts and i > 0:
                for span in (2, 1):
                    if i - span < 0:
                        continue
                    cand = " ".join(words[i - span:i])
                    try:
                        out[name] = colour_words(cand)
                        break
                    except UsageError:
                        continue
                if name in out:
                    break
    return out


def _skin(temp: float) -> Tuple[int, int, int]:
    return lch_to_rgb(91.0, 20.0 + 3 * temp, 52.0 - 6 * temp)


MATERIALS = {
    "wood": (128, 86, 52), "bark": (96, 70, 52), "stone": (140, 138, 132), "rock": (118, 112, 120),
    "metal": (150, 156, 166), "steel": (150, 156, 166), "gold": (222, 172, 60), "brass": (196, 160, 80),
    "leaf": (92, 150, 70), "leaves": (92, 150, 70), "grass": (110, 160, 70), "pine": (40, 82, 62),
    "sky": (150, 196, 236), "water": (70, 120, 170), "lake": (70, 120, 170), "sea": (60, 110, 160),
    "snow": (236, 240, 248), "sand": (226, 200, 150), "fire": (240, 120, 40), "flame": (250, 150, 50),
    "cloud": (238, 238, 244), "mountain": (110, 112, 140), "ground": (120, 100, 70), "fur": (224, 130, 60),
    "shore": (92, 80, 62), "hill": (96, 120, 80), "tree": (52, 96, 64), "forest": (40, 80, 60),
    "skin": None, "background": None, "line": None, "eye-white": None,
}


def name_colour(name: str):
    """A swatch whose name is a colour or a common material starts from that colour (cream, wood, sky ...)."""
    base = name.split("-")[0]
    for b in (base, base.rstrip("s"), base[:-2] if base.endswith("es") else base):
        if b in MATERIALS:
            return MATERIALS[b]
    try:
        return colour_words(base)
    except UsageError:
        return None


# The light and season of the scene, read from the request. Each mood: its words; the sky (and sky-low, the colour
# near the horizon) as LCh per palette; a background for scenes that are not outdoors; how every other colour moves:
# pull = (hue, share, added chroma), light = (target lightness, share), chroma = factor, foliage = (hue, share, factor)
# for leaves, trees and grass, accents = hues for the free colours.
MOODS = {
    "neon": {"words": ("neon", "cyberpunk", "synthwave", "vaporwave"),
             "sky": [(16, 14, 290), (14, 12, 275), (18, 16, 300), (15, 12, 260)],
             "low": [(40, 48, 330), (38, 40, 200), (42, 50, 320), (36, 40, 210)],
             "bg": (14, 10, 285), "light": (30, 0.35), "accents": ((335, 62), (200, 85), (305, 60), (185, 85))},
    "night": {"words": ("night", "midnight", "moonlit", "moonlight", "starry", "nocturnal", "nighttime"),
              "sky": [(22, 14, 265), (18, 12, 272), (26, 16, 258), (20, 14, 285)],
              "low": [(34, 16, 250), (30, 14, 262), (38, 18, 245), (32, 18, 300)],
              "bg": (20, 12, 265), "pull": (262, 0.4, 0.0), "light": (24, 0.45), "chroma": 0.75},
    "sunset": {"words": ("sunset", "sundown", "dusk", "dawn", "sunrise", "golden hour", "twilight", "evening glow"),
               "sky": [(60, 40, 20), (55, 38, 345), (66, 44, 40), (50, 34, 320)],
               "low": [(80, 50, 65), (78, 52, 55), (84, 46, 75), (74, 52, 45)],
               "bg": (84, 20, 50), "pull": (50, 0.2, 6.0)},
    "fog": {"words": ("fog", "foggy", "mist", "misty", "haze", "hazy"),
            "sky": [(86, 4, 240), (84, 5, 230), (88, 3, 250), (82, 5, 220)],
            "bg": (88, 4, 240), "light": (80, 0.45), "chroma": 0.45},
    "rain": {"words": ("rain", "rainy", "storm", "stormy", "overcast", "drizzle"),
             "sky": [(62, 6, 250), (58, 8, 255), (66, 5, 240), (55, 8, 265)],
             "bg": (70, 6, 250), "pull": (250, 0.15, 0.0), "light": (50, 0.2), "chroma": 0.6},
    "winter": {"words": ("winter", "snow", "snowy", "frost", "frosty", "frozen", "icy", "wintry"),
               "sky": [(84, 10, 240), (80, 12, 235), (86, 8, 250), (82, 10, 225)],
               "bg": (92, 5, 240), "pull": (240, 0.25, 0.0), "light": (85, 0.3), "chroma": 0.5},
    "autumn": {"words": ("autumn", "fall", "autumnal"), "foliage": (50, 0.9, 1.4)},
    "spring": {"words": ("spring", "blossom", "blossoms"), "foliage": (135, 0.3, 1.1)},
    "summer": {"words": ("summer", "sunny", "midday"),
               "sky": [(74, 40, 245), (70, 44, 240), (76, 36, 250), (72, 42, 235)], "chroma": 1.1},
}
FOLIAGE = ("leaf", "leaves", "tree", "trees", "forest", "grass", "foliage", "bush", "bushes", "hedge", "canopy",
           "meadow", "field", "shrub", "shrubs", "ivy", "vine", "vines", "woods")
OUTDOOR_WORDS = {"sky", "skies", "sunset", "sunrise", "dawn", "dusk", "landscape", "lake", "mountain", "mountains",
                 "sea", "ocean", "beach", "field", "fields", "meadow", "forest", "hill", "hills", "river", "outdoor",
                 "outdoors", "outside", "garden", "street", "valley", "coast", "shore", "desert", "seaside",
                 "countryside", "horizon", "park", "cliff", "cliffs", "waterfall", "island", "skyline", "scenery",
                 "tree", "trees", "pine", "pines", "woods", "grass", "cloud", "clouds", "snow", "sun", "moon"}


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z]+", (text or "").lower())


def request_moods(request: str) -> List[str]:
    """The moods the request names, in MOODS order (the first one with a sky sets the sky)."""
    joined = " " + " ".join(_words(request)) + " "
    return [m for m, spec in MOODS.items() if any(f" {w} " in joined for w in spec["words"])]


def is_outdoor(d: dict, request: str = "", names: Optional[List[str]] = None) -> bool:
    """An outdoor scene: the request, a subject, a part or a colour name says sky, lake, mountains, street ..."""
    words = set(_words(request))
    for n in list(d.get("subjects", {})) + list(d.get("parts", {})) + list(names or []):
        words |= set(re.split(r"[-_\s]+", part_base(str(n)).lower()))
    return bool(words & OUTDOOR_WORDS)


def scene_names(d: dict, request: str = "", names: Optional[List[str]] = None) -> List[str]:
    """The colour names plus what the scene needs: an outdoor scene gets a sky, a sunset / night / neon sky also a
    sky-low (the colour near the horizon:  band sky ... --color sky --color2 sky-low)."""
    names = list(names or needed_swatches(d))
    if not is_outdoor(d, request, names):
        return names
    add = ["sky"]
    if any("low" in MOODS[m] for m in request_moods(request)):
        add.append("sky-low")
    at = names.index("background") if "background" in names else len(names)
    for n in add:
        if n not in names:
            names.insert(at, n)
            at += 1
    return names


def _mood_tint(name: str, rgb, moods: List[str], accent_i: Optional[int] = None) -> Tuple[int, int, int]:
    from .draw_core import _toward
    L, C, h = rgb_to_lch(rgb)
    foliage = part_base(name).split("-")[0] in FOLIAGE
    for m in moods:
        spec = MOODS[m]
        if foliage and "foliage" in spec:
            hf, a, ck = spec["foliage"]
            h, C = _toward(h, hf, a), max(C, 20.0) * ck
        if accent_i is not None and "accents" in spec:
            h, L = spec["accents"][accent_i % len(spec["accents"])]
            C = max(C, 70.0)
            continue
        if "pull" in spec:
            hp, a, dc_ = spec["pull"]
            h, C = _toward(h, hp, a), C + dc_
        if "light" in spec:
            Lt, a = spec["light"]
            L = L + (Lt - L) * a
        C *= spec.get("chroma", 1.0)
    return lch_to_rgb(L, C, h)


def make_candidates(d: dict, base_rgb, request: str = "", names: Optional[List[str]] = None) -> List[dict]:
    """Four harmonised palettes around ``base_rgb`` (one per scheme); colours the request names stay fixed, and
    swatches named after a colour or a material start from it. Mood words in the request (sunset, night, autumn,
    winter, neon ...) set the sky and move every other colour; an outdoor scene always gets a sky (and a sky-low for
    a two-colour sky) and its background follows the sky."""
    names = scene_names(d, request, names)
    moods = request_moods(request)
    outdoor = is_outdoor(d, request, names)
    sky_mood = next((m for m in moods if "sky" in MOODS[m]), None)
    fixed = request_colours(request, names)
    material: Dict[str, tuple] = {}
    for n in names:
        if n not in fixed:
            c = name_colour(n)
            if c is not None:
                material[n] = c
    L0, C0, h0 = rgb_to_lch(base_rgb)
    # how much of the picture each swatch will cover (its parts' plan boxes): big fills are kept muted, as a
    # professional's are (their broad chroma is several times lower than a naive palette's)
    cover: Dict[str, float] = {}
    for pn, pv in d.get("parts", {}).items():
        b = pv.get("box") or [0, 0, 0, 0]
        a = max(0.0, (b[2] - b[0]) * (b[3] - b[1]) / 1e6)
        sw_n = pv.get("swatch") or pn
        cover[sw_n] = cover.get(sw_n, 0.0) + a
    for sn, sv in d.get("subjects", {}).items():
        if sn in names and sn not in cover:
            b = sv.get("box") or [0, 0, 0, 0]
            cover[sn] = max(0.0, (b[2] - b[0]) * (b[3] - b[1]) / 1e6)

    def muted(n, rgb, asked=False):
        """A colour covering a big part of the picture takes a muted version of itself (same hue, same
        lightness); the full chroma stays for the small accents. A professional picture's broad chroma is several
        times lower than a palette of full-strength fills."""
        cv_ = cover.get(n, 0.0)
        cap = 22.0 if cv_ > 0.2 else 30.0 if cv_ > 0.08 else 40.0 if cv_ > 0.03 else None
        if asked and cap is not None:
            cap = max(cap * 2.0, 52.0)      # the colour the request names stays clearly that colour
        if cap is None or n in ("skin", "line", "eye-white", "iris") or (sky_mood and n in ("sky", "sky-low", "sun")):
            return rgb                  # a sunset / night sky is the mood the request asked for
        L_, C_, h_ = rgb_to_lch(rgb)
        return lch_to_rgb(L_, min(C_, cap), h_) if C_ > cap else rgb
    out = []
    for k, scheme in enumerate(SCHEMES):
        hues = {"analogous": [0, 30, -30, 60, -60, 15], "complementary": [0, 180, 30, 210, -30, 160],
                "triad": [0, 120, 240, 60, 180, 300], "split": [0, 150, 210, 30, 180, 330]}[scheme]
        sw: Dict[str, str] = {}
        free = [n for n in names if n not in fixed and n not in material
                and n not in ("skin", "line", "eye-white", "background")]
        shift = 1 if any(abs(rgb_to_lch(c)[2] - h0) < 20 for c in fixed.values()) else 0   # base hue is taken
        for i0, n in enumerate(free):
            i = i0 + shift
            h = (h0 + hues[i % len(hues)]) % 360
            L = [L0, 62.0, 45.0, 72.0, 55.0, 80.0][i % 6] if i else L0
            C = max(18.0, C0 * [1.0, 0.75, 0.6, 0.5, 0.85, 0.4][i % 6]) if C0 > 8 else C0 + 4
            rgb = muted(n, lch_to_rgb(L, C, h))
            sw[n] = to_hex(_mood_tint(n, rgb, moods, accent_i=i0 + k) if moods else rgb)
        for n, rgb in material.items():
            # a material keeps its own colour, pulled toward the base hue (the light of the scene: sunset, night)
            t = [0.12, 0.25, 0.38, 0.5][k]
            if n in ("sky", "sky-low"):
                t *= 0.3                    # a sky without a mood word stays a sky colour, whatever the base
            elif any("pull" in MOODS[m] for m in moods):
                t *= 0.5                    # the mood already brings the scene's light
            Lm, Cm, hm = rgb_to_lch(rgb)
            dh = ((h0 - hm + 180) % 360) - 180
            rgb = lch_to_rgb(Lm, Cm * (1 - t) + C0 * t * 0.6, (hm + dh * t) % 360)
            if n == "sky-low":
                rgb = light_of(rgb)
            sw[n] = to_hex(_mood_tint(n, rgb, moods) if moods and n not in ("sky", "sky-low") else rgb)
        if sky_mood:
            spec = MOODS[sky_mood]
            low = next((MOODS[m]["low"] for m in moods if "low" in MOODS[m]), None)
            if "sky" in names and "sky" not in fixed:
                sw["sky"] = to_hex(lch_to_rgb(*spec["sky"][k]))
            if "sky-low" in names and "sky-low" not in fixed:
                sw["sky-low"] = to_hex(lch_to_rgb(*low[k]) if low else light_of(from_hex(sw["sky"])))
        for n, rgb in fixed.items():
            sw[n] = to_hex(muted(n, rgb, asked=True))      # the asked colour, muted to suit its size
        for n in list(sw):
            if n in material:
                sw[n] = to_hex(muted(n, from_hex(sw[n]), asked=True))   # its own colour (fur, wood ...)
        if "background" in names and "background" not in sw:
            bg_mood = next((m for m in moods if "bg" in MOODS[m]), None)
            if outdoor and "sky" in sw:
                sw["background"] = sw["sky"]            # an outdoor background is the sky, not the base's complement
            elif bg_mood:
                sw["background"] = to_hex(lch_to_rgb(*MOODS[bg_mood]["bg"]))
        if "skin" in names and "skin" not in sw:
            sw["skin"] = to_hex(_skin([0.0, 0.5, -0.5, 1.0][k]))
        if "eye-white" in names and "eye-white" not in sw:
            sw["eye-white"] = to_hex(lch_to_rgb(96.0, 3.0, (h0 + 180) % 360))
        if "background" in names and "background" not in sw:
            hb = (h0 + [180, 180, 120, 150][k]) % 360
            sw["background"] = to_hex(lch_to_rgb(90.0, 14.0, hb))
        if "line" in names and "line" not in sw:
            sw["line"] = to_hex(lch_to_rgb(24.0, 16.0, 20.0))
        out.append({"scheme": scheme, "swatches": {n: sw[n] for n in names if n in sw}})
    return out


def full_palette(swatches: Dict[str, str]) -> Dict[str, str]:
    """Base swatches plus a -shadow and a -light variant of each (unless given)."""
    out = dict(swatches)
    for n, hx in swatches.items():
        if n.endswith(("-shadow", "-light")) or n == "line":
            continue
        rgb = from_hex(hx)
        out.setdefault(f"{n}-shadow", to_hex(shadow_of(rgb)))
        out.setdefault(f"{n}-light", to_hex(light_of(rgb)))
    return out


def swatch(p, name: str, example: str = 'pixelcanvas palette set hair "#c0c8d8"') -> Tuple[int, int, int]:
    """A palette name (skin, hair-shadow), a hex value or a plain colour word -> rgb."""
    d = dstate(p)
    pal = full_palette(d["palette"]["swatches"])
    key = str(name).strip().strip("'\"").lower()
    if key in pal:
        return from_hex(pal[key])
    if key.startswith("pick@"):         # the reference's colour at x,y (hand mode)
        from .draw_ref import pick_colour
        return pick_colour(p, key[5:], example)
    alt = key.replace(" ", "-").replace("_", "-")
    if alt in pal:
        return from_hex(pal[alt])
    if alt in DEFAULTS:
        return DEFAULTS[alt]
    if alt.endswith(("-shadow", "-light")) and alt.rsplit("-", 1)[0] in DEFAULTS:
        b = DEFAULTS[alt.rsplit("-", 1)[0]]
        return shadow_of(b) if alt.endswith("-shadow") else light_of(b)
    try:
        return colour_words(name)
    except UsageError:
        known = ", ".join(n for n in pal if not n.endswith(("-shadow", "-light"))) or "none yet"
        raise UsageError(f"unknown colour {name!r}: use a palette name ({known}; each also -shadow and -light), "
                         "a hex value or a plain colour word", example)


def swatch_of_part(p, part: Optional[str]) -> Optional[str]:
    d = dstate(p)
    if part and part in d["parts"]:
        return d["parts"][part].get("swatch")
    if part and part in d["palette"]["swatches"]:
        return part
    return None


# ============================================================================================ views

def palette_image(cands: List[dict], current: Optional[Dict[str, str]] = None, width: int = 1024) -> np.ndarray:
    rows = []
    if current:
        rows.append(("current palette", current))
    for i, c in enumerate(cands, 1):
        rows.append((f"{i}  {c['scheme']}", c["swatches"]))
    ncol = max(len(r[1]) for r in rows)
    cw = max(80, min(150, (width - 40) // max(1, ncol)))
    rh = 108
    W = 20 + cw * ncol
    H = 16 + rh * len(rows)
    im = Image.new("RGB", (max(W, 320), H), (250, 250, 250))
    dr = ImageDraw.Draw(im)
    f, fs = view.font(18), view.font(14, bold=False)
    for r, (title, sw) in enumerate(rows):
        y = 8 + r * rh
        dr.text((12, y), title, fill=(20, 20, 20), font=f)
        for c, (n, hx) in enumerate(sw.items()):
            x = 12 + c * cw
            rgb = from_hex(hx)
            sh, li = shadow_of(rgb), light_of(rgb)
            dr.rectangle([x, y + 26, x + cw - 10, y + 70], fill=rgb, outline=(120, 120, 120))
            dr.rectangle([x, y + 58, x + (cw - 10) // 2, y + 70], fill=sh)
            dr.rectangle([x + (cw - 10) // 2, y + 58, x + cw - 10, y + 70], fill=li)
            dr.text((x, y + 74), n[:14], fill=(20, 20, 20), font=fs)
            dr.text((x, y + 90), hx, fill=(90, 90, 90), font=fs)
    arr = np.asarray(im)
    h, w = arr.shape[:2]
    H2, W2 = int(math.ceil(h / 32) * 32), int(math.ceil(w / 32) * 32)
    out = np.full((H2, W2, 3), 250, np.uint8)
    out[:h, :w] = arr
    return out


def plan_overlay(p, base: np.ndarray) -> np.ndarray:
    """The canvas with subject boxes (thick), part boxes (thin, 'name f<focus>'), the light arrow and the palette
    strip, drawn at canvas resolution."""
    d = dstate(p)
    im = Image.fromarray(base.copy())
    dr = ImageDraw.Draw(im)
    S = max(p.W, p.H)
    fz = max(13, int(S / 52))
    f = view.font(fz)
    lw = max(2, S // 300)
    for i, (n, s) in enumerate(d["subjects"].items()):
        x0, y0, x1, y1 = s["box"][0] * p.W / 1000, s["box"][1] * p.H / 1000, s["box"][2] * p.W / 1000, \
            s["box"][3] * p.H / 1000
        dr.rectangle([x0, y0, x1, y1], outline=(30, 30, 30), width=lw * 2)
        _label(dr, x0 + 3, y0 + 3, n.upper(), (30, 30, 30), (255, 255, 255), f)
    for i, (n, s) in enumerate(d["parts"].items()):
        col = view.MARK_COLOURS[i % len(view.MARK_COLOURS)]
        x0, y0, x1, y1 = s["box"][0] * p.W / 1000, s["box"][1] * p.H / 1000, s["box"][2] * p.W / 1000, \
            s["box"][3] * p.H / 1000
        dr.rectangle([x0, y0, x1, y1], outline=col, width=lw)
        _label(dr, x0 + 2, y1 - fz - 8, f"{n} f{s.get('focus', 2)}", col, (255, 255, 255), view.font(max(12, fz - 3)))
    if d["light"].get("set"):
        v = light_vec(d)
        cx, cy = p.W / 2, p.H / 2
        r = 0.42 * min(p.W, p.H)
        sx, sy = cx + v[0] / (math.hypot(v[0], v[1]) + 1e-9) * r, cy + v[1] / (math.hypot(v[0], v[1]) + 1e-9) * r
        ex, ey = sx + (cx - sx) * 0.25, sy + (cy - sy) * 0.25
        dr.line([sx, sy, ex, ey], fill=(250, 190, 0), width=lw * 3)
        ang = math.atan2(ey - sy, ex - sx)
        hl = S / 40
        for da in (2.6, -2.6):
            dr.line([ex, ey, ex + hl * math.cos(ang + da), ey + hl * math.sin(ang + da)], fill=(250, 190, 0),
                    width=lw * 3)
        dr.ellipse([sx - S / 60, sy - S / 60, sx + S / 60, sy + S / 60], fill=(255, 220, 60), outline=(120, 90, 0))
        _label(dr, min(max(4, sx - 40), p.W - 120), min(max(4, sy + S / 50), p.H - 40), "LIGHT", (250, 190, 0),
               (30, 30, 30), f)
    sw = d["palette"]["swatches"]
    if sw:
        n = len(sw)
        ch = max(18, int(p.H * 0.035))
        cw = p.W / n
        for i, (nm, hx) in enumerate(sw.items()):
            dr.rectangle([i * cw, p.H - ch, (i + 1) * cw, p.H], fill=from_hex(hx), outline=(60, 60, 60))
            rgb = from_hex(hx)
            dr.text((i * cw + 3, p.H - ch + 2), nm[:10], fill=(0, 0, 0) if lum(rgb) > 55 else (255, 255, 255),
                    font=view.font(max(10, int(ch * 0.55)), bold=False))
    return np.asarray(im)


def _label(dr, x, y, text, bg, fg, f):
    tw, th = view.text_size(dr, text, f)
    dr.rectangle([x, y, x + tw + 6, y + th + 7], fill=bg)
    dr.text((x + 3, y + 1), text, fill=fg, font=f)


def plan_lines(p) -> List[str]:
    d = dstate(p)
    out = []
    if d["subjects"]:
        out.append("subjects: " + "; ".join(f"{n} {coords.fmt_box(s['box'])}" for n, s in d["subjects"].items()))
        if len(d["subjects"]) > 1:
            from .draw_checks import depth_order
            out.append(f"depth, far to near: {', '.join(depth_order(p))} (change:  pixelcanvas plan order far,...,near)")
    if d["parts"]:
        by = {}
        for n, s in d["parts"].items():
            by.setdefault(s["of"], []).append(f"{n} f{s['focus']} [{s['swatch']}]")
        for subj, items in by.items():
            out.append(f"parts of {subj}: " + ", ".join(items))
    lt = d["light"]
    out.append("light: " + (f"from the {light_words(lt['deg'])} ({lt['deg']:.0f} deg"
                            + (", behind" if lt.get('z', 0.6) < 0 else ", front" if lt.get('z', 0.6) > 1.5 else "")
                            + ")" if lt.get("set") else "not set (default top-left)"))
    sw = d["palette"]["swatches"]
    out.append("palette: " + (", ".join(f"{n} {h}" for n, h in sw.items()) if sw else "none yet"))
    return out
