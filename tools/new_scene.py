#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a new scene from the starter template of its kind.

  python tools/new_scene.py NAME --prompt "REQUEST"    the request picks the kind: a picture -> illustration (its plan
                                                       filled in, every subject yours to draw); a chart, diagram, UI,
                                                       drawing ... -> its kit's starter with the request's data
  python tools/new_scene.py NAME --kind KIND           scenes/NAME.py from scenes/_template_KIND.py
  python tools/new_scene.py NAME --kind poster --size 1080x1920 --prompt "a poster titled \\"Blue Hour\\""
  python tools/new_scene.py --list                     every kind with its one-line description
  python tools/new_scene.py NAME                       landscape (the default kind); --poster = --kind poster
  python <root>/tools/new_scene.py poster --kind poster --dir ./art      ./art/poster.py in your own project

Where the file goes: --dir DIR, else $PIXELCANVAS_WORK, else scenes/ when run inside the pixelcanvas checkout, else
the current folder. render.py finds it by name from the same folder, or by path from anywhere.

A picture starts from the illustration template: its plan docstring is the art direction, and you draw each
subject yourself as a shaded form (Sketch, cv.draw; animals: creature; solids: form). A format template (chart,
diagram, ui, drawing ...) already draws the request's data: edit the values. Keep the structure (plan docstring,
PROMPT, UPPER_CASE tunables, one log("step") per object, post last).
Kinds are discovered from scenes/_template_<kind>.py: drop a new template file there and its kind appears here.
"""
import argparse
import difflib
import glob
import json
import os
import re
import sys

for _s in (sys.stdout, sys.stderr):              # Windows consoles default to cp950 -> mojibake
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_KIND = "landscape"
KIND_FALLBACK = {"banner": "card", "characters": "people", "pets": "animals"}


def kinds():
    """{kind: template path} from scenes/_template_<kind>.py (a bare _template.py counts as the landscape kind)."""
    table = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "scenes", "_template*.py"))):
        stem = os.path.basename(path)[:-3]
        if stem.startswith("_template_"):
            table[stem[len("_template_"):]] = path
        elif stem == "_template":
            table.setdefault(DEFAULT_KIND, path)
    return table


def _source(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def first_doc_line(path):
    """First line of the template docstring (what --list shows)."""
    m = re.search(r'"""\s*(.+?)\s*(?:\n|""")', _source(path))
    return m.group(1).strip() if m else ""


def reference_line(src):
    """The template's '# Read skills/...' comment: which reference page to read first."""
    m = re.search(r"^# (Read .+)$", src, re.M)
    return m.group(1) if m else "Read skills/pixelcanvas/SKILL.md"


def fill(src, name, size=None, prompt=None):
    """Template source -> scene source: NAME, and optionally DEFAULT_SIZE and PROMPT, replaced in place."""
    src = re.sub(r'^NAME = "[^"\n]*"', lambda m: f'NAME = "{name}"', src, count=1, flags=re.M)
    if size:
        src, n = re.subn(r"^DEFAULT_SIZE = \([^)\n]*\)", lambda m: f"DEFAULT_SIZE = ({size[0]}, {size[1]})", src,
                         count=1, flags=re.M)
        if not n:
            raise ValueError("the template has no DEFAULT_SIZE line")
    if prompt is not None:
        src, n = re.subn(r'^PROMPT = ""', lambda m: f"PROMPT = {prompt!r}", src, count=1, flags=re.M)
        if not n:
            raise ValueError("the template has no PROMPT line")
    return src


# ------------------------------------------------------------------ a starter that already draws the request
# A template marks the parts a request replaces with comment lines:
#   # >>> setting ... # <<< setting      the place (room, street, park, beach, night, plain backdrop)
#   # >>> subjects ... # <<< subjects    one call per subject of the request (header "stage" = the template
#                                        defines u0, v0, u1, v1: the area the subjects stand in, as fractions)
#   # >>> overlay ... # <<< overlay      3D templates: 2D calls painted after sc.render
MODULES = ("pictograms", "creature", "figure", "clay", "flora", "elements", "solid", "form", "texture", "backdrop",
           "stylize", "diagram", "chart", "drafting", "ui", "layout", "text", "vector", "palette", "sdf", "masks",
           "brush", "manifest", "noise", "filters")
TEXT_SLOTS = ("TITLE", "HEADLINE", "SUBTITLE", "TAGLINE", "KICKER", "BADGE", "FOOTER", "CAPTION")
BLANKS_OK = ("card", "poster")                     # templates that skip a text line set to ""
FORMAT_KINDS = ("diagram", "figure", "chart", "dashboard", "drawing", "floorplan", "ui", "infographic", "icons", "photo",
                "wallpaper", "comic", "animation")
SIZE_WORDS = [                                     # (words, (w, h)); the first match wins, explicit WxH before all
    (r"\b(?:phone|mobile|iphone)\s+(?:wallpaper|lock\s*screen)", (1170, 2532)),
    (r"\b(?:desktop|laptop|monitor)\s+wallpaper", (2560, 1440)),
    (r"\binstagram\s+stor(?:y|ies)\b|\bstory\b|\breel\b|\btiktok\b|\b9:16\b", (1080, 1920)),
    (r"\binstagram\s+post\b|\b4:5\b", (1080, 1350)),
    (r"\bsquare\b|\b1:1\b|\bprofile picture\b|\bavatar\b|\bapp icon\b", (1080, 1080)),
    (r"\byoutube\b|\bthumbnail\b", (1280, 720)),
    (r"\b(?:og|open graph|link)\s+(?:image|card|preview)\b|\bsocial card\b", (1200, 630)),
    (r"\b(?:twitter|x)\s+header\b|\blinkedin\s+banner\b", (1500, 500)),
    (r"\breadme\b|\bgithub\b", (1280, 640)),
    (r"\ba4\b", (2480, 3508)), (r"\ba3\b", (3508, 4961)), (r"\bletter size\b", (2550, 3300)),
    (r"\b4k\b|\buhd\b", (3840, 2160)), (r"\b16:9\b|\bwidescreen\b|\bslide\b", (1920, 1080)),
    (r"\b4:3\b", (1600, 1200)), (r"\b3:2\b", (1800, 1200)), (r"\b21:9\b|\bultrawide\b", (2560, 1080)),
    (r"\bportrait\s+(?:poster|format|orientation)\b", (1080, 1350)),
]
SETTINGS = {                                        # setting: the request words that name it
    "room": "room kitchen bedroom office cafe nook interior indoor indoors living library classroom bathroom "
            "restaurant bakery shop store studio apartment home lounge windowsill sofa couch fireplace desk bed",
    "street": "street road city town bus sidewalk pavement alley crosswalk downtown village station plaza",
    "park": "park garden meadow field lawn yard backyard picnic hill hills countryside playground orchard",
    "beach": "beach sea ocean shore coast seaside pier harbour harbor island",
    "night": "night midnight starry moonlit nighttime stars",
    "plain": "plain isolated sticker cutout studio packshot",
}
SETTING_2D = {
    "room": (["room = Room(cv, horizon=0.42, vanishing=0.52, back=0.6, width=5.2, height=3.0, eye=1.45)",
              'room.draw(wall="#f3dfbd", side="#efd6ae", floor="planks", window="left", seed=seed, sun=0.9, '
              "sun_dir=(1.0, -0.65, -0.35))"], {"form"}),
    "street": (['backdrop.backdrop(cv, "skyline", seed=seed)',
                'cv.paint("#8d9099", clamp01((cv.xy()[1] / cv.h - 0.8) * cv.h * 0.5))     # pavement',
                'cv.paint("#c9c6bd", clamp01((cv.xy()[1] / cv.h - 0.8) * cv.h * 0.5) * '
                'clamp01((0.83 - cv.xy()[1] / cv.h) * cv.h * 0.5))       # kerb'], {"backdrop"}),
    "park": (['elements.sky(cv, "day", horizon=0.7)',
              'cv.paint("#86b45c", clamp01((cv.xy()[1] / cv.h - 0.7) * cv.h * 0.5))      # lawn',
              'flora.plant(cv, "oak", at=(0.08, 0.74), height=0.5, seed=seed)'], {"elements", "flora"}),
    "beach": (['elements.sky(cv, "day", horizon=0.55)', "elements.ocean(cv, horizon=0.55, seed=seed)",
               'cv.paint("#e6cf9c", clamp01((cv.xy()[1] / cv.h - 0.78) * cv.h * 0.5))     # sand'], {"elements"}),
    "night": (['elements.sky(cv, "night", horizon=0.7)', "elements.stars(cv, seed=seed, horizon=0.7)",
               'elements.moon(cv, at=(0.82, 0.18), size=50)',
               'cv.paint("#1d2436", clamp01((cv.xy()[1] / cv.h - 0.7) * cv.h * 0.5))     # ground'], {"elements"}),
    "plain": (['cv.gradient([(0, "#f4efe6"), (1, "#ddd2c0")])'], set()),
}
SETTING_3D = {                                      # (lines added after the camera, GROUND colour, background stops)
    "room": (['sc.add(solid.room(3.6, 3.6), "room")             # walls behind the subjects, window on the left'],
             "#c99a68", [(0, "#efe2cf"), (1, "#d9c7ad")]),
    "street": ([], "#9a9ca3", [(0, "#a9c4dc"), (1, "#e6ebee")]),
    "park": ([], "#8fb46a", [(0, "#9cc4e8"), (1, "#e8f0f0")]),
    "beach": (['sc.add(solid.box(60, 0.02, 30).at(0, 0.01, -18.0), "sea", solid.material("glossy", "#3f7fae"))'],
              "#e6cf9c", [(0, "#8ec5e8"), (1, "#eaf4f6")]),
    "night": ([], "#3a4660", [(0, "#0d1530"), (1, "#2a3358")]),
    "plain": ([], "#e9ddcc", [(0, "#f7ecde"), (1, "#e8d6c0")]),
}


def request_size(prompt):
    """(w, h) the request names: 'WxH', or a format word (square, story, A4, YouTube thumbnail ...); else None."""
    m = re.search(r"\b(\d{3,5})\s*[xX×*]\s*(\d{3,5})\b", prompt or "")
    if m and int(m.group(1)) >= 16 and int(m.group(2)) >= 16:
        return int(m.group(1)), int(m.group(2))
    low = (prompt or "").lower()
    for rx, size in SIZE_WORDS:
        if re.search(rx, low):
            return size
    return None


def request_setting(prompt):
    """The place the request sets its subjects in: room, street, park, beach, night or plain (None: the template's)."""
    words = set(re.findall(r"[a-z]+", (prompt or "").lower()))
    words |= {w[:-1] for w in words if w.endswith("s")}
    hit = {k for k, ws in SETTINGS.items() if words & set(ws.split())}
    for k in ("room", "night", "street", "beach", "park", "plain"):
        if k in hit:
            return k
    return None


def _block(src, name):
    """(start, end, indent, header) of a marked block, or None."""
    m = re.search(r"^([ \t]*)# >>> " + name + r"\b([^\n]*)\n(.*?)^[ \t]*# <<< " + name + r"\b[^\n]*\n", src,
                  re.M | re.S)
    return (m.start(3), m.end(3), m.group(1), m.group(2)) if m else None


def _replace_block(src, name, lines):
    b = _block(src, name)
    if b is None:
        return src, False
    s, e, ind, _h = b
    return src[:s] + "".join(ind + ln + "\n" if ln else "\n" for ln in lines) + src[e:], True


def _set_constant(src, name, value):
    """NAME = <literal> at the top level, replaced by ``value`` (a Python literal string)."""
    rx = r"^(" + name + r" = )(\"(?:[^\"\\\n]|\\.)*\"|'(?:[^'\\\n]|\\.)*'|\[[^\n]*\]|\([^\n]*\))(.*)$"

    def put(m):
        tail = m.group(3)
        c = re.match(r"(\s+)(#.*)$", tail)
        if c:                                             # the comment stays in its column when it fits
            col = len(m.group(1)) + len(m.group(2)) + len(c.group(1))
            tail = " " * max(2, col - len(m.group(1)) - len(value)) + c.group(2)
        return m.group(1) + value + tail
    return re.sub(rx, put, src, count=1, flags=re.M)


def _text_values(res):
    """The text the request asks for, quotes first: [str]."""
    quotes = [c for c in res.content if re.search(r"['\"“‘]" + re.escape(c), res.query)]
    return quotes + [c for c in res.content if c not in quotes]


def fill_text(src, kind, res):
    """Quoted strings of the request become the template's TITLE / HEADLINE / SUBTITLE ... in order."""
    texts = _text_values(res)
    if not texts:
        return src, []
    present = [n for n in TEXT_SLOTS if re.search(r"^" + n + r" = [\"']", src, re.M)]
    used = []
    for i, n in enumerate(present):
        if i < len(texts):
            src = _set_constant(src, n, json.dumps(texts[i], ensure_ascii=False))
            used.append((n, texts[i]))
        elif kind in BLANKS_OK:
            src = _set_constant(src, n, '""')
    if kind in BLANKS_OK and re.search(r"^DETAILS = \[", src, re.M):
        src = _set_constant(src, "DETAILS", "[]")          # no example details next to the request's own text
    return src, used


def _calls_of(g):
    """[(call, entry)] of a subject group in the order find ranks them."""
    calls = list(getattr(g, "calls", None) or [])
    return [(c, e) for c, e in zip(calls, g.results)]


def _u_positions(k):
    return [0.5] if k == 1 else [round(0.2 + 0.6 * i / (k - 1), 2) for i in range(k)]


def _place2d(call, u, staged, k=1):
    """The call standing at horizontal fraction ``u`` (its own v kept; inside the stage when the template has one,
    sized for ``k`` subjects side by side)."""
    m = re.search(r"\bat=\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)", call)
    lead = "at="
    if not m:                                             # form.ball(cv, (u, v), ...): the position comes second
        m = re.search(r"(?<=\(cv, )\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)", call)
        lead = ""
    if not m:
        return call
    v = float(m.group(2))
    if staged:
        at = f"{lead}(u0 + {u:.2f} * (u1 - u0), v0 + {min(v, 0.96):.2f} * (v1 - v0))"
        big = {1: 0.62, 2: 0.48}.get(k, 0.34)
        call = re.sub(r"\bsize=([\d.]+)", lambda s: f"size={max(float(s.group(1)), big):.2f} * (v1 - v0)", call,
                      count=1)
    else:
        at = f"{lead}({u:.2f}, {v:.2f})"
    return call[:m.start()] + at + call[m.end():]


def _place3d(call, x, z=0.0, rot=None):
    m = re.fullmatch(r'(?:(\w+) = )?sc\.add\((.+), "([^"]+)"\)', call)
    if not m or ".place(" in m.group(2) or ".on(" in m.group(2):
        return call
    turn = (f".face({rot:g})" if m.group(2).startswith("clay.") else f".rotate_y({rot:g})") if rot else ""
    return f'{m.group(1) + " = " if m.group(1) else ""}sc.add({m.group(2)}{turn}.place(x={x:.2f}, z={z:.2f}), ' \
           f'"{m.group(3)}")'


def _mini_scene(call, u, staged=False, k=1, box=None):
    """A 3D model in a 2D picture: its own small Scene (no floor) rendered into a box over the painting."""
    half = min(0.18, 0.45 / max(k, 1))
    a, b = max(0.0, u - half), min(1.0, u + half)
    if box is not None:                                    # (u0, v0, u1, v1) fractions from the ground plan
        box = f"box=(int({box[0]:.3f} * cv.w), int({box[1]:.3f} * cv.h), int({box[2]:.3f} * cv.w), int({box[3]:.3f} * cv.h))"
    elif staged:
        box = (f"box=(int((u0 + {a:.2f} * (u1 - u0)) * cv.w), int(v0 * cv.h), int((u0 + {b:.2f} * (u1 - u0)) * cv.w), "
               "int(v1 * cv.h))")
    else:
        box = f"box=(int({a:.2f} * cv.w), int(0.45 * cv.h), int({b:.2f} * cv.w), int(0.95 * cv.h))"
    return ["sub = solid.Scene(lighting=\"studio\", ground=None)", call.replace("sc.add(", "sub.add(", 1),
            "sub.camera = solid.Camera.frame(sub, fill=0.95)", f"sub.render(cv, {box}, background=False)"]


# ------------------------------------------------------------------ placement: ground lines, real sizes, relations
# Things stand on the setting's ground line (boats on the water, birds in the sky, a lighthouse on the far shore),
# sized by their real height and their depth: on a ground plane seen from eye height E, a thing of height h whose
# base is drawn at v is h * (v - horizon) / E tall. E is chosen so the main subject is a good size; everything else
# follows. Nothing overlaps another subject unless the request relates them ("on", "in", "holding").
_REAL_H = {   # standing height in metres (the size classes: insect < bird < cat < dog < person < car < tent ...)
    "person": 1.7, "woman": 1.65, "man": 1.78, "girl": 1.3, "boy": 1.3, "child": 1.2, "kid": 1.2, "baby": 0.7,
    "friend": 1.7, "chef": 1.75, "robot": 1.4, "dog": 0.6, "corgi": 0.35, "puppy": 0.35, "cat": 0.3, "kitten": 0.2,
    "rabbit": 0.3, "bunny": 0.3, "fox": 0.45, "owl": 0.45, "bird": 0.22, "seagull": 0.3, "gull": 0.3, "parrot": 0.35,
    "duck": 0.4, "swan": 0.8, "butterfly": 0.08, "bee": 0.03, "ladybug": 0.02, "insect": 0.04, "squirrel": 0.25,
    "hedgehog": 0.15, "frog": 0.08, "horse": 1.6, "cow": 1.5, "sheep": 0.9, "deer": 1.3, "bear": 1.3, "wolf": 0.85,
    "elephant": 3.0, "giraffe": 5.0, "car": 1.5, "taxi": 1.5, "bus": 3.2, "truck": 3.2, "tram": 3.3, "train": 4.0,
    "bicycle": 1.0, "bike": 1.0, "scooter": 1.1, "motorcycle": 1.2, "boat": 3.5, "fishing_boat": 4.5, "sailboat": 7.0,
    "motorboat": 2.5, "rowing_boat": 1.2, "canoe": 1.0, "ship": 15.0, "yacht": 9.0, "tent": 2.0, "campfire": 0.6,
    "bench": 0.9, "tree": 8.0, "pine": 10.0, "oak": 9.0, "palm": 8.0, "birch": 8.0, "bush": 1.2, "shrub": 1.2,
    "flower": 0.4, "tulip": 0.45, "sunflower": 1.6, "rock": 0.8, "house": 7.0, "cottage": 6.0, "cabin": 5.0,
    "lighthouse": 25.0, "tower": 30.0, "windmill": 15.0, "castle": 25.0, "barn": 8.0, "bus_stop": 2.6,
    "bus_stop_sign": 2.6, "street_lamp": 4.0, "mailbox": 1.2, "table": 0.75, "chair": 0.9, "armchair": 1.0,
    "sofa": 0.85, "bookshelf": 1.9, "floor_lamp": 1.6, "lamp": 1.5, "desk_lamp": 0.5, "plant": 0.8,
    "potted_plant": 0.6, "teapot": 0.2, "mug": 0.1, "vase": 0.3, "lemon": 0.07, "toolbox": 0.3, "umbrella": 1.0,
    "kite": 0.9, "balloon": 0.5, "snowman": 1.6, "fountain": 2.0, "statue": 2.5, "window": 1.2,
}
_CLASS_H = {"person": 1.7, "animal": 0.6, "bird": 0.3, "sea creature": 0.4, "vehicle": 1.6, "plant": 0.8,
            "building": 8.0, "furniture": 0.9, "food": 0.12, "container": 0.25, "instrument": 0.7, "tool": 0.3,
            "toy": 0.4, "device": 0.2, "sport": 0.3, "timepiece": 0.4, "ornament": 1.2, "textile": 0.3,
            "clothing": 0.5, "audio": 0.6, "landform": 30.0}
_ASPECT = {"person": 0.38, "woman": 0.38, "man": 0.38, "girl": 0.4, "boy": 0.4, "child": 0.42, "friend": 0.38,
           "dog": 1.3, "corgi": 1.5, "cat": 1.3, "fox": 1.4, "rabbit": 1.0, "horse": 1.3, "owl": 0.7, "bird": 1.2,
           "seagull": 1.6, "butterfly": 1.4, "boat": 1.1, "fishing_boat": 1.1, "sailboat": 0.8, "car": 2.4,
           "bus": 2.6, "bicycle": 1.7, "tent": 1.5, "tree": 0.7, "pine": 0.45, "house": 1.1, "lighthouse": 0.4,
           "campfire": 1.2, "bench": 2.0, "table": 1.6, "armchair": 1.0, "sofa": 2.2, "bookshelf": 0.55,
           "lamp": 0.3, "floor_lamp": 0.3, "plant": 0.7, "bus_stop": 0.5, "bus_stop_sign": 0.4}
_WATER = {"boat", "ship", "sailboat", "yacht", "canoe", "kayak", "ferry", "raft", "swan", "duck", "motorboat",
          "fishing_boat", "rowing_boat", "gondola", "submarine", "buoy"}
_SKY = {"seagull", "gull", "plane", "airplane", "aeroplane", "kite", "balloon", "helicopter", "eagle", "hawk", "bat",
        "bird", "moon", "sun", "drone", "ufo", "dragon"}
_SHORE = {"lighthouse", "pier", "harbour", "harbor"}
_SIDE = {"tree", "pine", "oak", "palm", "house", "lighthouse", "tower", "building", "windmill", "castle", "bus_stop",
         "bus_stop_sign", "street_lamp", "cottage", "barn"}
_REL_WORDS = r"on top of|on|in|inside|atop|upon|holding|carrying|riding|beside|next to|by|near"
_RIDE = {"bicycle": 0.12, "bike": 0.12, "motorbike": 0.15, "motorcycle": 0.15, "scooter": 0.1, "horse": 0.55,
         "pony": 0.5, "camel": 0.7, "elephant": 0.75, "donkey": 0.5, "unicycle": 0.45, "skateboard": 0.06,
         "surfboard": 0.05, "boat": 0.3, "fishing_boat": 0.3}       # where the rider's feet go, of its height
_HELD_UP = {"balloon", "umbrella", "kite", "flag", "torch", "lantern", "sign", "placard", "parasol", "sparkler"}
_CONTAINERS = {"vase", "bowl", "basket", "pot", "jar", "bucket", "cup", "mug", "glass", "box", "crate"}
_PERCH = {"tree": 0.58, "pine": 0.5, "oak": 0.55, "palm": 0.75, "table": 0.97, "desk": 0.97, "armchair": 0.5,
          "sofa": 0.5, "chair": 0.52, "bed": 0.55, "bench": 0.5, "windowsill": 0.3, "window": 0.3, "rock": 0.9,
          "rug": 0.02, "boat": 0.35, "vase": 0.85, "bowl": 0.6, "basket": 0.7, "pot": 0.8, "jar": 0.85, "fishing_boat": 0.35, "shelf": 0.8, "bookshelf": 1.0, "fence": 1.0, "car": 1.0}
_PART3D = {"armchair": "seat", "sofa": "seat", "chair": "seat", "table": "top", "side_table": "top", "desk": "top",
           "bed": "top", "bookshelf": "shelf", "rug": None, "counter": "top"}
_GROUND2D = {   # setting: (horizon, ground band, water band)
    "room": (0.42, (0.64, 0.97), None), "street": (0.62, (0.83, 0.97), None), "park": (0.7, (0.74, 0.97), None),
    "beach": (0.55, (0.8, 0.97), (0.57, 0.76)), "night": (0.7, (0.73, 0.97), None),
    "plain": (0.5, (0.74, 0.95), None),
}


def _keyname(g, e=None):
    """The thing's size-class key: its entry name ("fishing_boat") or its head word."""
    names = []
    if e is not None:
        names += [e.name.replace(" ", "_").lower()]
    names += [(g.head or "").lower()]
    for n in names:
        if n in _REAL_H:
            return n
    for n in names:
        last = n.split("_")[-1]
        if last in _REAL_H:
            return last
    return names[-1] if names[-1] else (names[0] if names else "")


def _real_h(g, e, catalog):
    k = _keyname(g, e)
    if k in _REAL_H:
        return _REAL_H[k]
    cls = catalog._CLASS_OF.get(g.head or "") if g.head else None
    return _CLASS_H.get(cls, 1.0)


def _relations(prompt, heads):
    """{child head: (relation, target head)} the request states: "an owl in a pine tree", "a cat on the armchair"."""
    low = (prompt or "").lower()
    out = {}
    for a in heads:
        for b in heads:
            if a == b or a in out or not a or not b:
                continue
            m = re.search(r"\b" + re.escape(a) + r"(?:e?s)?\b([^,.;]{0,40}?)\b(" + _REL_WORDS + r")\s+(?:(?:a|an|the|"
                          r"his|her|its|their|two|three|some)\s+)?(?:[a-z\-]+\s+){0,2}?" + re.escape(b) + r"(?:e?s)?\b",
                          low)
            gap = m.group(1) if m else ""
            if m and (re.search(r"\b(?:with|and|under|at)\b", gap) or
                      any(re.search(r"\b" + re.escape(h) + r"\b", gap) for h in heads if h and h not in (a, b))):
                m = None                                      # "a tent at night ... an owl in a pine": the owl's
            if m:
                m = type("M", (), {"group": lambda self, i, _m=m: _m.group(i + 1)})()
                rel = m.group(1)
                rel = {"on top of": "on", "atop": "on", "upon": "on", "inside": "in", "next to": "by", "beside": "by",
                       "near": "by", "carrying": "holding"}.get(rel, rel)
                if rel == "holding":                       # "a woman holding an umbrella": the umbrella is held
                    if b not in out:
                        out[b] = ("held", a)
                    continue
                out[a] = (rel, b)
    for a in heads:                                       # "a vase of tulips": the tulips are in the vase
        for b in heads:
            if a in _CONTAINERS and b and b != a and b not in out and \
                    re.search(r"\b" + re.escape(a) + r"s?\s+of\s+(?:[a-z\-]+\s+)?" + re.escape(b), low):
                out[b] = ("in", a)
    return out


_SEATS = {"bench", "chair", "sofa", "couch", "armchair", "stool", "swing", "log", "rock"}


def _sits_on(t):
    """A person whose stated relation puts them on a seat ("a man on a bench"): drawn sitting, feet on the floor."""
    r = t.get("rel")
    return bool(r and r[0] == "on" and r[1] in _SEATS and t["call"].startswith("figure."))


def _ground_of(src, setting, kind, res):
    """(horizon, ground band, water band, sky band) of the picture."""
    if setting in _GROUND2D:
        vh, gb, wb = _GROUND2D[setting]
    else:
        m = re.search(r"^HORIZON = ([\d.]+)", src, re.M) or re.search(r"^EYE_LEVEL(?:, \w+)? = ([\d.]+)", src, re.M)
        vh = float(m.group(1)) if m else 0.5
        mt = re.search(r"^TABLE = dict\(top=([\d.]+), front=([\d.]+), horizon=([\d.]+)\)", src, re.M)
        if mt:                                            # a still life: things stand on the table top
            vh = float(mt.group(3))
        if mt:
            gb, wb = (float(mt.group(1)) + 0.04, float(mt.group(2)) - 0.04), None
        elif kind == "landscape":
            gb, wb = (0.93, 0.985), (vh + 0.02, 0.9)
        elif kind in ("people", "interior"):
            gb, wb = (0.64, 0.97), None
        elif kind == "blank":
            vh, gb, wb = 0.5, (0.74, 0.95), None
        else:
            gb, wb = (min(0.97, vh + 0.08), 0.97), None
    sea = any((g.head or "") in ("sea", "ocean", "lake", "harbour", "harbor", "river", "bay") for g in res.groups)
    if wb is None and sea and setting not in ("room", "street"):
        wb = (vh + 0.02, max(vh + 0.12, gb[0] - 0.04))
    return vh, gb, wb, (0.1, max(0.16, vh - 0.12))


def _resize2d(call, u, v, s, aspect_wh):
    """A 2D call standing with its base at (u, v), ``s`` canvas-heights tall."""
    m = re.search(r"\bat=\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)", call)
    lead = "at="
    if not m:
        m = re.search(r"(?<=\(cv, )\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)", call)
        lead = ""
    if not m:
        return call
    if call.startswith("form.ball("):                    # a ball rests on (u, v) with radius s / 2
        out = call[:m.start()] + f"({u:.3f}, {v - s / 2:.3f})" + call[m.end():]
        return re.sub(r"\)\s*,\s*([\d.]+)", lambda r: f"), {s / 2:.3f}", out, count=1)
    out = call[:m.start()] + f"{lead}({u:.3f}, {v:.3f})" + call[m.end():]
    if call.startswith("elements.campfire("):
        return re.sub(r"\bsize=[\d.]+", f"size={round(s * 1080 * 0.7)}", out, count=1)
    if call.startswith("figure.person("):
        out = re.sub(r"\bheight=[\d.]+", "", out)
        return re.sub(r"^figure\.person\(([^)]*)\)", lambda r: f"figure.person({r.group(1)}, height={s:.3f})", out,
                      count=1)
    if call.startswith("creature."):
        return re.sub(r"\bsize=[\d.]+", f"size={s * 0.9:.3f}", out, count=1)
    if re.search(r"\bheight=[\d.]+", out):
        return re.sub(r"\bheight=[\d.]+", f"height={s:.3f}", out, count=1)
    if re.search(r"\bsize=[\d.]+", out):
        return re.sub(r"\bsize=[\d.]+", f"size={s:.3f}", out, count=1)
    return out


def _layout2d(things, ground, prompt, aspect_wh):
    """Base point and height for every 2D thing: [(thing, [(u, v, s)])], drawn far to near."""
    vh, gb, wb, skyb = ground
    rel = _relations(prompt, [t["key"] for t in things] + [t["head"] for t in things])
    for t in things:
        r = rel.get(t["key"]) or rel.get(t["head"])
        k = t["key"]
        cover = lambda o: not re.search(r"\bat=\(|^form\.ball\(", o["call"])     # a meadow, grass: ground cover
        if r and not any(r[1] in (o["key"], o["head"]) and not cover(o) for o in things if o is not t):
            r = None                                    # "a fox in a meadow": the meadow is the ground, no perch
        t["rel"] = r
        if cover(t):
            t["zone"] = "cover"
        elif r and r[0] in ("on", "in", "riding", "held"):
            t["zone"] = "perch"
        elif r and r[0] == "by" and k not in _SKY:
            t["zone"] = "beside"
        elif k in _SHORE and wb:
            t["zone"] = "shore"
        elif k in _WATER and wb:
            t["zone"] = "water"
        elif k in _SKY and not (k == "bird" and t.get("perched")):
            t["zone"] = "sky"
        else:
            t["zone"] = "ground"
    grounded = [t for t in things if t["zone"] in ("ground", "water")]
    if not grounded:                                       # everything stands beside something: the first stands
        grounded = [t for t in things if t["zone"] == "beside"][:1]
        for t in grounded:
            t["zone"] = "ground"
    main = next((t for t in grounded if t["key"] not in _SIDE), grounded[0] if grounded else None)
    if main is None:
        return [(t, [(0.5, 0.9, 0.3)]) for t in things]
    n = max(1, sum(t["n"] for t in grounded if t["key"] not in _SIDE))
    target = {1: 0.5, 2: 0.42, 3: 0.36}.get(n, 0.3)
    band = wb if main["zone"] == "water" else gb
    v_main = band[0] + 0.72 * (band[1] - band[0])
    riders = [t for t in things if t.get("rel") and (t["rel"][0] == "riding" or _sits_on(t))]
    ref = max([t for t in grounded if t["key"] not in _SIDE] + riders or [main], key=lambda t: t["h"])
    E = ref["h"] * (v_main - vh) / target                 # the tallest subject sets the scale ...
    smin = 0.2 if main["h"] >= 0.25 else 0.09
    E = min(E, main["h"] * (v_main - vh) / smin)          # ... and the main subject stays readable
    placed, boxes = [], []

    def free_u(w, v, s, pref, side=False):
        best, cost = pref, 1e9
        for i in range(41):
            u = 0.08 + 0.84 * i / 40
            lo, hi = u - w / 2, u + w / 2
            if lo < 0.02 or hi > 0.98:
                continue
            ov = sum(max(0.0, min(hi, b[2]) - max(lo, b[0])) * max(0.0, min(v, b[3]) - max(v - s, b[1]))
                     / max(min(w * s, (b[2] - b[0]) * (b[3] - b[1])), 1e-6) for b in boxes)
            c = 4.0 * ov + abs(u - pref) * (0.6 if side else 1.0)
            if c < cost:
                best, cost = u, c
        return best

    order = sorted(things, key=lambda t: t["zone"] in ("perch", "beside"))     # targets before what rests on them
    side_pref = iter([0.84, 0.16, 0.9, 0.1] * 3)
    for t in order:
        spots = []
        if t["zone"] == "cover":                          # drawn first, as it is
            placed.append((t, spots))
            continue
        for i in range(t["n"]):
            h, z = t["h"], t["zone"]
            tgt = next((p for p in placed if t.get("rel") and t["rel"][1] in (p[0]["key"], p[0]["head"]) and p[1]),
                       None)
            if z in ("perch", "beside") and tgt is not None and t["rel"][0] in ("riding", "held", "by"):
                tu, tv, ts = tgt[1][0]
                tw = ts * _ASPECT.get(tgt[0]["key"], 1.0) / aspect_wh
                if t["rel"][0] == "riding":                     # on the saddle, at the vehicle's depth and scale
                    s = max(0.05, h * (tv - vh) / E)
                    spots.append((tu + 0.08 * tw * (1 if i == 0 else -1) * i, tv - _RIDE.get(tgt[0]["key"], 0.4) * ts, s))
                elif t["rel"][0] == "held":                     # in the holder's hand, beside the body
                    s = min(0.45 * ts, max(0.04, h * (tv - vh) / E))
                    up = t["key"] in _HELD_UP or t["head"] in _HELD_UP      # a balloon, an umbrella: above the hand
                    spots.append((tu + 0.3 * ts / aspect_wh, tv - 0.42 * ts + (0.0 if up else 0.5 * s), s))
                else:                                           # side by side at the same depth, never overlapping
                    s = min(0.72, max(0.035, h * (tv - vh) / E))
                    w = s * _ASPECT.get(t["key"], 1.0) / aspect_wh
                    side = 1 if tu < 0.5 else -1
                    u = min(0.97 - w / 2, max(0.03 + w / 2, tu + side * (tw / 2 + w / 2 + 0.02 + (w + 0.02) * i)))
                    spots.append((u, tv, s))
                    boxes.append((u - w / 2, tv - s, u + w / 2, tv))
                continue
            if z == "beside":
                z = "ground"
            if z == "perch":
                if tgt is not None and _sits_on(t):             # a person on a bench: feet on its ground line
                    tu, tv, ts = tgt[1][0]
                    s = max(0.05, h * (tv - vh) / E)
                    spots.append((tu + 0.35 * s * i / aspect_wh, tv, s))
                    continue
                if tgt is not None:
                    tu, tv, ts = tgt[1][0]
                    k = _PERCH.get(tgt[0]["key"], 0.9)
                    s = max(0.05, h * (tv - vh) / E) * (0.6 if tgt[0]["key"] in _CONTAINERS else 1.0)
                    off = (0.12 if tgt[0]["key"] in ("tree", "pine", "oak") else 0.0) * ts / aspect_wh
                    spots.append((tu + off * (1 if i % 2 == 0 else -1) + 0.6 * s * i / aspect_wh, tv - k * ts, s))
                    continue
                z = "ground"
            if z == "sky":
                s = min(0.12, max(0.035, 0.6 * target * (h / main["h"]) ** 0.5))
                v = skyb[0] + (skyb[1] - skyb[0]) * (0.35 + 0.3 * ((i * 37) % 10) / 10)
                pref = 0.62 + 0.09 * i if i else 0.62
            elif z == "shore":
                s = min(0.36, max(0.16, h * 0.012 / E))
                v = vh + 0.004
                pref = next(side_pref)
            else:
                b = wb if z == "water" else gb
                v_lo, v_hi = vh + 0.05 * E / h, vh + 0.62 * E / h       # depth range for heights 0.05 .. 0.62
                v = min(max(v_main - (0.05 * i if z == "water" else 0.0), b[0]), b[1])
                if t is not main:
                    v = min(max(v, v_lo), v_hi)
                    v = min(max(v, b[0]), b[1])
                    if i:
                        v = max(b[0], v - 0.03 * i)
                s = min(0.72, max(0.035, h * (v - vh) / E))
                pref = 0.5 if t is main and i == 0 else (next(side_pref) if t["key"] in _SIDE else 0.5)
            w = s * _ASPECT.get(t["key"], 1.0) / aspect_wh
            u = free_u(w, v, s, pref, t["key"] in _SIDE)
            boxes.append((u - w / 2, v - s, u + w / 2, v))
            spots.append((u, v, s))
        if t["zone"] == "perch":
            for (u, v, s) in spots:
                w = s * _ASPECT.get(t["key"], 1.0) / aspect_wh
                boxes.append((u - w / 2, v - s, u + w / 2, v))
        placed.append((t, spots))
    return sorted(placed, key=lambda p: (p[0]["zone"] == "perch", min(v for _u, v, _s in p[1]) if p[1] else 0))


_ROOM_WALL_BACK = {"bookshelf", "floor_lamp", "wardrobe", "potted_plant", "counter", "coffee_machine", "fridge",
                   "window_frame", "plant", "tree", "piano", "clock"}
_ROOM_WALL_LEFT = {"sofa", "armchair", "bed", "chair", "desk", "couch"}
_ROOM_CENTER = {"rug", "table", "side_table"}
_FLAT = {"rug", "picnic", "picnic_blanket", "blanket", "mat", "carpet", "towel", "beach_towel"}
_ON_TABLE = {"desk_lamp", "laptop", "mug", "cup", "vase", "book", "bottle", "teapot", "speaker", "smartphone",
             "coffee_machine", "earbuds", "charging_case", "smartwatch"}
_FOOT = {"bookshelf": (1.0, 0.32), "floor_lamp": (0.45, 0.45), "potted_plant": (0.5, 0.5), "sofa": (2.0, 0.95),
         "armchair": (0.9, 0.85), "bed": (1.4, 2.0), "table": (1.2, 0.8), "side_table": (0.5, 0.5),
         "chair": (0.5, 0.5), "counter": (1.6, 0.65), "rug": (2.0, 1.4), "tent": (2.2, 2.2), "tree": (2.5, 2.5),
         "house": (6.0, 6.0), "car": (4.2, 1.8), "truck": (6.0, 2.4), "stone_lantern": (0.6, 0.6),
         "arched_bridge": (4.0, 1.6), "desk_lamp": (0.25, 0.25), "robot": (0.6, 0.5), "character": (0.6, 0.4)}


def _solid_name(call):
    m = re.search(r"sc\.add\((?:solid|clay)\.(\w+)\(", call)
    return m.group(1) if m else ""


def _layout3d(things, room_r, prompt, water=False):
    """(x, z, rotate) for every 3D thing, and the 2D stand-ins' floor spots: furniture along the walls (tallest at
    the back), people and animals in the clear centre, big outdoor things behind and to the sides."""
    rel = _relations(prompt, [t["key"] for t in things] + [t["head"] for t in things])
    spots = {}
    back_x = -room_r + 0.6 if room_r else None
    left_z = -room_r + 1.0 if room_r else None
    centre = [(0.35, 0.45), (-0.25, 0.75), (0.85, 0.15), (0.1, 1.05), (1.0, 0.85), (-0.6, 1.2)]
    out_front = [(-0.4, 0.5), (0.45, 0.65), (-1.2, 0.9), (1.2, 0.9), (0.0, 1.3), (-1.9, 0.4)]
    flat = [t for t in things if t["key"] in _FLAT]
    rug_at = (0.5, 0.6) if room_r else (0.0, 0.5)
    out_back = [(-2.6, -2.2), (2.6, -2.4), (-4.4, -3.6), (4.4, -3.4), (0.0, -4.6)]
    ci, fi, bi = 0, 0, 0
    tables = [t for t in things if t["key"] in ("table", "side_table", "desk", "counter")]
    for t in things:
        r = rel.get(t["key"]) or rel.get(t["head"])
        if r and r[0] in ("on", "in") and any(r[1] in (o["key"], o["head"]) for o in things if o is not t and o["is3d"]):
            tgt = next(o for o in things if o is not t and r[1] in (o["key"], o["head"]) and o["is3d"])
            t["on"] = tgt
            continue
        if t["is3d"] and (t["h"] < 0.35 or t["key"] in _ON_TABLE) and tables and t["kind"] == "prop" and \
                t not in tables:
            t["on"] = tables[0]                             # a mug, a laptop, a vase: on the table
            continue
        res_ = []
        for i in range(t["n"]):
            key, fw = t["key"], _FOOT.get(t["key"], (max(0.4, 0.8 * t["h"]), max(0.4, 0.5 * t["h"])))
            if t["head"] in _ROOM_WALL_LEFT | _ROOM_WALL_BACK:
                key = t["head"]                             # "a desk" drawn by a table: against the wall
            if room_r:
                if key in _ROOM_WALL_BACK or (t["kind"] == "prop" and t["h"] >= 1.3):
                    x, z, rot = back_x + fw[0] / 2, -room_r + fw[1] / 2 + 0.05, None
                    back_x += fw[0] + 0.25
                    if back_x > room_r - 0.4:
                        back_x = -room_r + 0.6
                elif key in _ROOM_WALL_LEFT:
                    x, z, rot = -room_r + fw[1] / 2 + 0.1, left_z + fw[0] / 2, 90
                    left_z += fw[0] + 0.3
                elif key in _ROOM_CENTER or key in _FLAT:
                    x, z, rot = (rug_at[0], rug_at[1], None) if key in _FLAT else \
                        ((-0.4, -0.6, None) if flat else (0.5, -0.2, None))
                else:
                    x, z = centre[ci % len(centre)]
                    ci += 1
                    rot = -30
                x, z = max(-room_r + 0.3, min(room_r - 0.3, x)), max(-room_r + 0.3, min(room_r - 0.3, z))
            elif water and key in _WATER:                    # boats on the sea behind the beach
                x, z, rot = 2.6 + 2.8 * i, -12.0 - 1.0 * i, None    # beside the people, not behind them
            elif key in _FLAT:                               # a rug, a picnic blanket: under the subjects
                x, z, rot = rug_at[0], rug_at[1], None
            else:
                if t["h"] >= 1.8 or key in _SIDE:
                    x, z = out_back[bi % len(out_back)]
                    bi += 1
                    rot = None
                else:
                    x, z = out_front[fi % len(out_front)]
                    fi += 1
                    rot = -20 if x > 0 else 20
            res_.append((x, z, rot))
        spots[id(t)] = res_
    return spots, rel


def plan_subjects(res, is3d, staged, catalog, spread=3.0, fixed="", ctx=None):
    """[(label, [code lines], overlay?, how)] for every subject of the request, in order."""
    ctx = ctx or {}
    prompt, setting, src, kind = ctx.get("prompt", res.query), ctx.get("setting"), ctx.get("src", ""), ctx.get("kind")
    groups = [g for g in res.groups if not _skip(g, catalog, is3d) and not _in_template(g, fixed)]
    if setting == "room" and is3d:                          # the room has its own window
        groups = [g for g in groups if (g.head or "") not in ("window", "windowsill", "room", "wall", "floor")]
    if is3d:                                                # "a meadow", "a field": the ground of the 3D scene
        groups = [g for g in groups if (g.head or "") not in catalog._SETTING_KIND or
                  any(e.kit == "solid" for e in g.results)]
    effects = [g for g in groups if g.results and g.results[0].kit == "elements" and
               ("at=" not in (g.calls[0] if g.calls else "") or g.results[0].kind == "sky")]
    things_g = [g for g in groups if g not in effects]
    atmos = [g for g in effects if g.results[0].name.replace(" ", "_") in catalog._ATMOS]
    behind = [g for g in effects if g not in atmos and g.results[0].kind != "sky"]
    out = []
    for g in behind:                                        # the sea, the stars: behind the subjects
        c = g.calls[0] if g.calls else g.results[0].call
        if c.startswith("elements.ocean(") and re.search(r"^HORIZON = ", src, re.M) and not setting:
            c = re.sub(r"horizon=[\d.]+", "horizon=HORIZON", c) if "horizon=" in c else c.rstrip(")") + ", horizon=HORIZON)"
        out.append(((g.phrase or g.head or "effect")[:24], [c], is3d, g.results[0].id))
    things = []
    for g in things_g:
        label = (g.phrase or g.head or "subject").split(",")[0].strip()[:24]
        n = max(1, min(g.count, 6))
        head = (g.head or "").replace('"', "")
        t = {"g": g, "label": label, "n": n, "head": head, "comment": [], "how": ""}
        if g.fallback:
            fb = g.fallback
            if fb.get("standins"):
                c = fb["standins"][0]
                t["how"] = f"stand-in for {g.head}: no kit draws it yet; replace with your own shape ({fb.get('ref')})"
                t["comment"] = [f"# STAND-IN: {t['how']}"]
                t["call"], t["entry"], t["mark"] = c, None, "stand-in"
            else:
                t["lines"] = [f"# STAND-IN: no kit draws {g.head} yet; own shape, read {fb.get('ref')}"] + \
                    list(fb["lines"]) + [fb["register"]]
                t["how"], t["call"], t["entry"] = "own shape", None, None
        else:
            pairs = _calls_of(g)
            pick = next(((c, e) for c, e in pairs if c.startswith("sc.add(")), None) if is3d else \
                next(((c, e) for c, e in pairs if not c.startswith("sc.")), None)
            solid3 = None
            if pick is None and is3d and g.head:            # "a desk" in a 3D room: the solid prefab it is a kind of
                solid3 = next((x for x in catalog._index() if x.kit == "solid" and x.kind == "prefab" and
                               g.head in x._kw), None)
            if solid3 is not None:
                pick = (f'sc.add(solid.{solid3.name}(), "{g.head}")', solid3)
            if pick is None:
                pick = pairs[0]
            c, e = pick
            stand = getattr(g, "standin", False) and e.kit == "pictograms"
            if e.kit == "pictograms" and "style=" not in c:   # an illustration: the pictogram lit and shaded
                c = catalog._shaded_call(e, g.colors)
            t["how"] = f"{e.id}" + (" (stand-in pictogram drawn shaded)" if stand else "")
            if stand:
                t["comment"] = [f"# STAND-IN: the {e.name} pictogram drawn shaded; a dedicated kit call replaces it"]
            elif getattr(g, "standin", False) or getattr(g, "kind_of", False) or getattr(g, "kin", False) or \
                    solid3 is not None:
                t["how"] = f"{e.id} (stand-in for {g.head})"
                t["comment"] = [f"# STAND-IN: {e.id} stands in for the {g.head}; a dedicated kit call replaces it"]
                stand = True
            t["call"], t["entry"], t["mark"] = c, e, "stand-in" if stand else None
        c = t.get("call")
        if c:
            if t.get("mark") and head and c.startswith("pictograms.pictogram(") and "name=" not in c:
                c = c[:-1] + f', name="{_sing_word(head)}")'           # recorded under the request's word
            if c.startswith("form.ball(") and head and "name=" not in c:
                c = c[:-1] + f', name="{_sing_word(head)}")'
            if t.get("mark") and c.startswith("sc.add("):
                c = re.sub(r', "[^"]+"\)$', f', "{_sing_word(head)}")', c)
            c = re.sub(r"(clay\.animal\(\"[^\"]+\"), size=0\.5", r"\1", c)    # clay animals at their real size
            t["call"] = c
        e = t.get("entry")
        t["key"] = _keyname(g, e)
        t["h"] = _real_h(g, e, catalog)
        t["is3d"] = bool(c and c.startswith("sc.add("))
        cls = catalog._CLASS_OF.get(head)
        sem = catalog._sem(g) if g.results else None
        t["kind"] = "being" if sem in ("person", "animal") or cls in ("person", "animal", "bird") else "prop"
        things.append(t)
    if is3d:
        out += _plan3d(things, ctx, staged, spread)
    elif staged:
        us = _u_positions(len(things))
        for t, u in zip(things, us):
            if "lines" in t:
                out.append((t["label"], t["lines"], False, t["how"]))
                continue
            n = t["n"]
            sub_u = [u + (i - (n - 1) / 2) * 0.15 for i in range(n)]
            shift = max(0.0, 0.1 - min(sub_u)) - max(0.0, max(sub_u) - 0.9)
            sub_u = [round(x + shift, 2) for x in sub_u]
            c = t["call"]
            lines = _mini_scene(c, u, True, len(things)) if c.startswith("sc.") else \
                [_place2d(c, x, True, len(things)) for x in sub_u]
            out.append((t["label"], t["comment"] + lines, False, t["how"]))
    else:
        out += _plan2d(things, ctx, res)
    for g in atmos:                                         # rain, snow, fog: over everything
        c = g.calls[0] if g.calls else g.results[0].call
        out.append(((g.phrase or g.head or "effect")[:24], [c], is3d, g.results[0].id))
    return out


def _seated(line, riding=False):
    """A figure sitting on what the planner placed under it (a bench, a sofa, a bicycle): no seat of its own."""
    if not line.startswith("figure.person("):
        return line
    if '.pose("sit"' in line:
        return line if "seat=" in line else line.replace('.pose("sit"', '.pose("sit", seat="none"', 1)
    if riding and ".pose(" not in line:
        return line.replace(").draw(", ').pose("sit", seat="none").draw(', 1)
    return line


def _plan2d(things, ctx, res):
    src, setting, kind, prompt = ctx.get("src", ""), ctx.get("setting"), ctx.get("kind"), ctx.get("prompt", "")
    size = ctx.get("size") or (1920, 1080)
    aspect_wh = size[0] / size[1]
    ground = _ground_of(src, setting, kind, res)
    drawable = [t for t in things if "lines" not in t]
    placed = _layout2d(drawable, ground, prompt, aspect_wh)
    out = [(t["label"], t["lines"], False, t["how"]) for t in things if "lines" in t]
    vh, gb, wb, _sky = ground
    if any(t["zone"] == "shore" for t, _s in placed):       # the far shore the lighthouse stands on
        u = next(sp[0][0] for t, sp in placed if t["zone"] == "shore")
        out.append(("shore", ["X, Y = cv.xy()",
                              f"cv.paint(\"#4f5a52\", sdf.fill(sdf.sd_ellipse(X, Y, {u:.3f} * cv.w, {vh + 0.012:.3f} * "
                              f"cv.h, 0.16 * cv.w, 0.03 * cv.h)))     # headland on the horizon"], False, "shore"))
    for t, spots in placed:
        c = t["call"]
        lines = [] if spots else [c]
        for (u, v, s) in spots:
            if c.startswith("sc."):
                w = s * _ASPECT.get(t["key"], 1.0) / aspect_wh * 1.15
                lines += _mini_scene(c, u, box=(max(0.0, u - w / 2), max(0.0, v - s * 1.08), min(1.0, u + w / 2),
                                                min(1.0, v + 0.01)))
            else:
                lines.append(_resize2d(c, u, v, s, aspect_wh))
        where = {"water": "on the water", "sky": "in the sky", "shore": "on the far shore",
                 "perch": f"{t['rel'][0]} the {t['rel'][1]}" if t.get("rel") else "", "ground": "",
                 "beside": f"beside the {t['rel'][1]}" if t.get("rel") else "", "cover": ""}[t["zone"]]
        if t.get("rel") and t["zone"] == "perch" and t["rel"][0] == "held":
            how = t["how"] + f", held by the {t['rel'][1]}"
            out.append((t["label"], t["comment"] + lines, False, how))
            continue
        if t.get("rel") and t["zone"] == "perch" and t["rel"][0] in ("on", "in", "riding"):
            lines = [_seated(ln, t["rel"][0] == "riding" or _sits_on(t)) for ln in lines]    # the target is the seat
        how = t["how"] + (f", {where}" if where else "")
        out.append((t["label"], t["comment"] + lines, False, how))
    return out


def _plan3d(things, ctx, staged, spread):
    src, setting, prompt = ctx.get("src", ""), ctx.get("setting"), ctx.get("prompt", "")
    m = re.search(r"^ROOM = ([\d.]+)", src, re.M)
    room_r = float(m.group(1)) / 2 if m and "room(" in src else (1.8 if setting == "room" else None)
    spots, rel = _layout3d(things, room_r, prompt, water=setting == "beach")
    if not room_r:                                         # 2D stand-ins stand beside the models, in the frame
        xs = [x for t in things if t.get("is3d") for x, _z, _r in spots.get(id(t), [])] or [0.0]
        side = iter([max(xs) + 1.3, min(xs) - 1.3, max(xs) + 2.6, min(xs) - 2.6] * 2)
        for t in things:
            if "lines" not in t and not t["is3d"] and not (setting == "beach" and t["key"] in _WATER):
                spots[id(t)] = [(next(side), 0.2, None)]
    lo, hi = [1e9, 0.0, 1e9], [-1e9, 0.3, -1e9]            # the framing box: every subject, not the floor items
    for t in things:
        if "lines" in t or t["key"] in ("rug", "picnic_blanket", "blanket", "picnic", "mat", "carpet") or \
                (setting == "beach" and t["key"] in _WATER):
            continue
        fw = _FOOT.get(t["key"], (max(0.3, 0.8 * t["h"]), max(0.3, 0.5 * t["h"])))
        for x, z, _r in spots.get(id(t), []):
            lo = [min(lo[0], x - fw[0] / 2), 0.0, min(lo[2], z - fw[1] / 2)]
            hi = [max(hi[0], x + fw[0] / 2), max(hi[1], min(t["h"], 4.0)), max(hi[2], z + fw[1] / 2)]
    if lo[0] < 1e8 and not room_r:
        ctx["frame_box"] = (tuple(round(v, 2) for v in lo), tuple(round(v, 2) for v in hi))
    out, names = [], {}
    order = sorted(things, key=lambda t: "on" in t)          # what things rest on comes first
    for t in order:
        if "lines" in t:
            out.append((t["label"], t["lines"], True, t["how"]))
            continue
        c = t["call"]
        if not t["is3d"]:                                    # a 2D stand-in: on the floor plane, after the render
            h = min(t["h"], 2.5)
            lines = []
            for x, z, _r in (spots.get(id(t)) or [(0.8, 0.6, None)]):
                lines += [f"x0, y0 = sc.project(({x:.2f}, 0.0, {z:.2f}))          # its spot on the floor, in pixels",
                          f"_x1, y1 = sc.project(({x:.2f}, {h:.2f}, {z:.2f}))",
                          re.sub(r"at=\([^)]*\), size=[\d.]+",
                                 "at=(x0 / cv.w, y0 / cv.h), size=max(0.04, (y0 - y1) / cv.h)", c, count=1)]
            out.append((t["label"], t["comment"] + lines, True, t["how"] + ", on the floor"))
            continue
        var = re.sub(r"[^a-z0-9_]", "_", (t["head"] or "obj").lower())[:14] or "obj"
        names[id(t)] = var
        if "on" in t:
            tgt = t["on"]
            part = _PART3D.get(tgt["key"], "top")
            tv = names.get(id(tgt), "")
            m2 = re.fullmatch(r'sc\.add\((.+), "([^"]+)"\)', c)
            body = m2.group(1) if m2 else c
            on = f'.on({tv}, "{part}")' if part else f".on({tv})"
            lines = [f'sc.add({body}{on}, "{m2.group(2) if m2 else t["head"]}")'] if tv else [c]
            out.append((t["label"], t["comment"] + lines, False, t["how"] + f", on the {tgt['head']}"))
            continue
        lines = []
        for i, (x, z, rot) in enumerate(spots.get(id(t), [])):
            ln = _place3d(c, x, z, rot)
            if t["n"] > 1:
                ln = re.sub(r'"\)$', f' {i + 1}")', ln.replace("seed=1,", f"seed={i + 1},", 1))
            lines.append(ln)
        if any(o.get("on") is t for o in things) and lines:
            lines[0] = f"{var} = {lines[0]}"
        out.append((t["label"], t["comment"] + lines, False, t["how"]))
    return out


def _skip(g, catalog, is3d=False):
    if getattr(g, "format", False) or getattr(g, "text_object", False) or getattr(g, "setting_only", False):
        return True
    if g.fallback:
        return False
    if not g.results:
        return True
    e = g.results[0]
    if is3d and e.kit == "texture" and any(x.kit == "solid" for x in g.results):
        return False                                    # "a rug" in a 3D room: solid.rug
    return e.kit in ("stylize", "palette", "texture", "layout", "ui", "diagram", "chart", "drafting", "text") or \
        e.kind in ("theme", "palette", "material", "backdrop", "3D scene")


def _in_template(g, fixed):
    """The subject is already part of the template outside its subjects block (the table of a still life, the
    room of an isometric room)."""
    if not fixed or not g.head:
        return False
    names = {g.head} | ({g.results[0].name.replace(" ", "_")} if g.results else set())
    return any(re.search(r"\b" + re.escape(n) + r"\w*\(", fixed, re.I) for n in names if len(n) > 2)


def _sky_of(res):
    g = next((g for g in res.groups if g.results and g.results[0].kind == "sky" and g.calls), None)
    return g.calls[0] if g else None


def _add_imports(src, code):
    mods = sorted({m for m in re.findall(r"\b(" + "|".join(MODULES) + r")\.", code)})
    have = set()
    for m in re.finditer(r"^from pixelcanvas import (.+)$", src, re.M):
        have |= {x.strip() for x in m.group(1).split(",")}
    need = [m for m in mods if m not in have]
    if "clamp01(" in code and "clamp01" not in have:
        need.append("clamp01")
    if "np." in code and not re.search(r"^import numpy as np", src, re.M):
        src = re.sub(r"^(from pixelcanvas import )", "import numpy as np\n\n\\1", src, count=1, flags=re.M)
    if "Room(" in code and not re.search(r"^from pixelcanvas\.form import .*\bRoom\b", src, re.M):
        need_room = "from pixelcanvas.form import Room\n"
    else:
        need_room = ""
    if not need and not need_room:
        return src
    last = list(re.finditer(r"^(?:from pixelcanvas[\w.]* import .+|import .+)$", src, re.M))[-1]
    end = last.end()
    if "(" in last.group(0) and ")" not in last.group(0):  # a parenthesised import: after its closing line
        end = src.index(")", end) + 1
        end = src.index("\n", end)
    add = (f"from pixelcanvas import {', '.join(need)}\n" if need else "") + need_room
    return src[:end + 1] + add + src[end + 1:]


def compose(src, kind, prompt, res=None):
    """Template source -> a starter that already draws the request: its text, size-independent subjects (one
    log() step each, stand-ins marked), its setting. Returns (source, report lines)."""
    sys.path.insert(0, ROOT)
    from pixelcanvas import catalog
    res = res if res is not None else catalog.find(prompt)
    report = []
    if kind == "illustration":                                        # freehand: a plan to draw, no kit calls
        return compose_illustration(src, prompt, res)
    if kind in FORMAT_KINDS and _block(src, "data") is not None:     # the request's own data in the data block
        return compose_format(src, kind, prompt, res)
    src, used = fill_text(src, kind, res)
    report += [f"text     {n} = {t!r}" for n, t in used]
    is3d = bool(re.search(r"^from pixelcanvas\.solid import[^\n]*\bScene\b", src, re.M))
    if kind in FORMAT_KINDS:
        subs = [g.phrase for g in res.groups if not _skip(g, catalog)]
        if subs:
            report.append("subjects " + ", ".join(subs) + " (drawn by this kind's own calls: edit the data above)")
        return src, report
    setting = request_setting(prompt)
    blk = _block(src, "subjects")
    m = re.search(r"spread ([\d.]+)", blk[3]) if blk else None
    sb = _block(src, "setting")
    body = src.split("def render", 1)[-1]
    fixed = body.replace(src[blk[0]:blk[1]], "") if blk else body
    if setting and sb:
        fixed = fixed.replace(src[sb[0]:sb[1]], "")
    ms = re.search(r"^DEFAULT_SIZE = \((\d+), (\d+)\)", src, re.M)
    ctx = {"src": src, "setting": setting if sb else None, "kind": kind, "prompt": prompt,
           "size": (int(ms.group(1)), int(ms.group(2))) if ms else None}
    plan = plan_subjects(res, is3d, bool(blk and "stage" in blk[3]), catalog, float(m.group(1)) if m else 3.0, fixed,
                         ctx)
    if sb and blk and not plan and re.search(r"\b(\w+)\.place\(", src[blk[0]:blk[1]]) and \
            re.search(r"\b(\w+) = ", src[sb[0]:sb[1]]):
        setting = None                                    # the example subjects stand in the example setting
    if setting and sb:
        if is3d:
            lines, ground, back = SETTING_3D[setting]
            for n in ("GROUND", "GRASS"):
                src = _set_constant(src, n, json.dumps(ground))
            for n in ("SKY", "BACKDROP", "BACKGROUND"):
                if re.search(r"^" + n + r" = \[", src, re.M):
                    src = _set_constant(src, n, repr(back).replace("'", '"'))
            src, _ok = _replace_block(src, "setting", ["pass                                        # " +
                                                       ("the room is added after the camera frames the subjects"
                                                        if lines else "GROUND and the background set the place")])
            if setting == "room" and re.search(r"\bwindow", prompt or "", re.I):
                lines = [ln.replace("solid.room(3.6, 3.6)", 'solid.room(3.6, 3.6, window="back", door=None)')
                         .replace("window on the left", "the window behind them") for ln in lines]
            if lines:                                     # framed on the subjects, then the room / sea around them
                src = re.sub(r"^([ \t]*)(sc\.camera = Camera\.frame\([^\n]*\n)",
                             lambda m_: m_.group(1) + m_.group(2) + "".join(m_.group(1) + ln + "\n" for ln in lines),
                             src, count=1, flags=re.M)
        else:
            lines = list(SETTING_2D[setting][0])
            if any((g.head or "") in ("tree", "oak", "pine", "palm", "birch", "maple") for g in res.groups):
                lines = [ln for ln in lines if not ln.startswith('flora.plant(cv, "oak"')]    # the request's tree
            sky = _sky_of(res)
            if sky and setting not in ("room", "plain", "night"):
                lines[0] = sky if "horizon" in sky else sky.rstrip(")") + ", horizon=0.72)"
            src, _ok = _replace_block(src, "setting", ['log("setting")'] + lines)
        report.append(f"setting  {setting}")
    if is3d and re.search(r"\b(?:sunrise|sunset|dawn|dusk|golden hour|evening glow)\b", prompt or "", re.I):
        src = re.sub(r'Lighting\("(?:sunny|studio)"', 'Lighting("sunset"', src, count=1)
        for n in ("SKY", "BACKDROP"):
            if re.search(r"^" + n + r" = \[", src, re.M):
                src = _set_constant(src, n, '[(0, "#f2a46b"), (0.6, "#f8d3a6"), (1, "#fbe9d2")]')
        report.append("light    sunset")
    blk = _block(src, "subjects")
    if plan:
        main, over = [], []
        for label, lines, overlay, how in plan:
            step = [f'log({json.dumps(label, ensure_ascii=False)})'] + lines
            (over if overlay and _block(src, "overlay") else main).extend(step)
            report.append(f"subject  {label}: {how}")
        if blk:
            src, _ok = _replace_block(src, "subjects", main)
        else:
            ind = "    "
            code = "".join(ind + ln + "\n" for ln in main)
            if re.search(r"^from pixelcanvas\.scenery import", src, re.M):
                code = ind + "cv = sc.cv                                           # the scenery canvas\n" + code
            src = re.sub(r'^([ \t]*(?:sc\.)?log\("post"\))', lambda m: code + "\n" + m.group(1), src, count=1,
                         flags=re.M)
        if over:
            src, _ok = _replace_block(src, "overlay", over)
        if ctx.get("frame_box"):                              # framed on the subjects and the stand-ins' spots
            lo, hi = ctx["frame_box"]
            if hi[1] < 1.0 and setting != "room":             # small animals outdoors: a low camera shows the sky
                src = re.sub(r"^(AZIMUTH, ELEVATION = [-\d.]+, )([\d.]+)", r"\g<1>6", src, count=1, flags=re.M)
            src = re.sub(r"Camera\.frame\(sc,", f"Camera.frame(({lo}, {hi}),", src, count=1)
            src = re.sub(r"(sc\.camera = Camera\.frame\([^\n]*\))([ \t]*#[^\n]*)?",
                         lambda m_: m_.group(1) + "     # fits the subjects (lo, hi box in metres)", src, count=1)
    if res.style and "apply(" not in src:                 # the named medium: the last step before post
        step = f'    log("style")\n    stylize.apply(cv, "{res.style}")\n'
        if re.search(r"^from pixelcanvas\.scenery import", src, re.M) and "cv = sc.cv" not in src:
            step = "    cv = sc.cv\n" + step
        src = re.sub(r'^([ \t]*(?:sc\.)?log\("post"\))', lambda m_: step + m_.group(1), src, count=1, flags=re.M)
        report.append(f"style    stylize.apply(cv, {res.style!r}) before post")
    src = _add_imports(src, src.split("def render", 1)[1] if "def render" in src else src)
    return src, report


# ------------------------------------------------------------------ illustration: the request as a plan to draw
# The freehand starter gets the request's art direction in its docstring plan (mood, time of day, palette words,
# named style, subjects, exact text), MOOD / STYLE / TITLE, and a "# draw: <subject>" line per subject with an
# optional ingredient call as a comment. It inserts no kit call: the agent draws every subject itself.
_TIMES = ("dawn", "sunrise", "early morning", "morning", "noon", "midday", "afternoon", "golden hour", "sunset", "dusk",
          "evening", "twilight", "night", "midnight", "nighttime")
_MOOD_WORDS = ("cosy", "cozy", "calm", "peaceful", "serene", "quiet", "dreamy", "moody", "mysterious", "eerie",
               "spooky", "cheerful", "playful", "whimsical", "joyful", "happy", "melancholic", "melancholy", "sad",
               "lonely", "dramatic", "epic", "romantic", "nostalgic", "festive", "magical", "cute", "gloomy", "stormy",
               "rainy", "misty", "foggy", "snowy", "sunny", "bright", "warm", "cold", "chilly", "hazy", "lively")
_PALETTE_WORDS = ("pastel", "pastels", "muted", "earthy", "earth tones", "neon", "vibrant", "saturated", "monochrome",
                  "monochromatic", "black and white", "sepia", "duotone", "jewel tones", "candy", "autumn colours",
                  "autumn colors", "limited palette", "two colours", "two colors", "three colours", "three colors")
_GRADE = [("print", ("ukiyo", "woodblock", "risograph", "riso", "screen print", "linocut", "poster art", "paper cut",
                     "papercut", "vintage")),
          ("moody", ("moody", "mysterious", "eerie", "spooky", "gloomy", "stormy", "storm", "melancholy",
                     "melancholic", "lonely", "dramatic")),
          ("warm night", ("night", "midnight", "moon", "starry", "lantern", "evening", "twilight")),
          ("golden", ("sunset", "sunrise", "golden hour", "dusk", "dawn", "autumn", "fall")),
          ("moody", ("foggy", "misty", "rainy", "rain")),
          ("cosy", ("cosy", "cozy", "snug", "candle", "fireplace", "lamp", "warm")),
          ("cool day", ("morning", "noon", "midday", "afternoon", "sunny", "spring", "summer", "winter", "snow",
                        "day"))]


def _words_in(low, words):
    return [w for w in words if re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", low)]


def _plan_line(src, label, value):
    """The docstring plan line starting with ``* <label>`` gets ``value`` in place of its <placeholder>."""
    return re.sub(r"^(\* " + re.escape(label) + r"[^:\n]*: )<[^>\n]*>", lambda m: m.group(1) + value, src, count=1,
                  flags=re.M)


def compose_illustration(src, prompt, res):
    """The freehand starter: the request's own art direction in the plan, MOOD / STYLE / TITLE set, and one
    ``# draw: <subject>`` line per subject (with an optional ingredient call as a comment). No kit calls."""
    sys.path.insert(0, ROOT)
    from pixelcanvas import catalog
    low = " " + (prompt or "").lower() + " "
    report = []
    src, used = fill_text(src, "illustration", res)
    report += [f"text     {n} = {t!r}" for n, t in used]
    times, moods = _words_in(low, _TIMES), _words_in(low, _MOOD_WORDS)
    if times or moods:
        src = _plan_line(src, "Mood and light", ", ".join(dict.fromkeys(moods + times)) +
                         "; <where the light comes from and what glows>")
    pal = [w for w, _h in res.colors] + _words_in(low, _PALETTE_WORDS)
    if pal:
        src = _plan_line(src, "Palette", ", ".join(dict.fromkeys(pal)) + " + <the rest of 4-6 colours, one accent>")
        hexes = [h for _w, h in res.colors]
        if hexes:
            src = re.sub(r'("accent": )"#[0-9a-fA-F]{6}"', lambda m: m.group(1) + json.dumps(hexes[0]), src, count=1)
    if res.style:
        src = _plan_line(src, "Style", f"{res.style}: <its parts, from illustration.md 'Build the style'>")
        src = _set_constant(src, "STYLE", json.dumps(res.style))
        report.append(f"style    {res.style}: build it by hand (references/illustration.md, Build the style)")
    if used:
        src = _plan_line(src, "Exact text", "; ".join(json.dumps(t, ensure_ascii=False) for _n, t in used))
    mood = next((m for m, ws in _GRADE if _words_in(low, ws)), None)
    if mood:
        src = _set_constant(src, "MOOD", json.dumps(mood))
        report.append(f"mood     grade(cv, {mood!r}) as the last step")
    subs = [g for g in catalog.drawn_subjects(res) if g.results or g.fallback]
    if subs:
        names = [g.phrase + (f" x{g.count}" if g.count > 1 else "") for g in subs]
        src = _plan_line(src, "Subject and frame", f"{', '.join(names)}; <{subs[0].phrase} is the focal point: "
                                                   "where it stands and how much of the height it fills>")
    lines, bound = [], set()
    for g in res.groups:                                             # "at the harbour": the background to paint
        if getattr(g, "setting_only", False) and g.head:
            lines.append(f"# draw: {g.head} (the setting, in the background step)")
            report.append(f"setting  {g.head}: paint it in the background step")
    for g in subs:
        name = g.phrase + (f" x{g.count}" if g.count > 1 else "")
        ing = res._ingredient_lines(g, bound) if g.results or g.fallback else []
        lines.append(f"# draw: {name}")
        for call, note in ing:
            lines.append(f"#   optional {note.split(' · ')[0]}: {call}")
        report.append(f"subject  {name}: draw it" + (f" (optional: {ing[0][1].split(' · ')[0]})" if ing else ""))
    if lines:
        src, _ok = _replace_block(src, "subjects", lines)
    return src, report


# ------------------------------------------------------------------ format kinds: the request written into the data
# A format template marks its data with # >>> data ... # <<< data (and the code that draws it with # >>> build /
# # >>> screen). compose_format() reads the request's named things (components, steps, series, KPI tiles, icons,
# sizes, rooms, panels ...) and writes them there, so the untouched starter already draws the request.
_NUMW = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         "eleven": 11, "twelve": 12, "a": 1, "an": 1, "single": 1, "pair": 2, "couple": 2, "dozen": 12}
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_QUOTE_RE = re.compile(r"(?<![A-Za-z])['\"“‘«「]([^'\"“”‘’«»「」]{1,120}?)['\"”’»」](?![A-Za-z])")
_PASTELS = ["#f6d6cf", "#cfe3f2", "#c9b6e4", "#f2d58a", "#bfe3c8", "#f2b36b", "#d8c7e8", "#f4c7d9", "#c6e0e6"]
_CATS = {   # plausible category names for "by <word>" / "a donut of <word>" (the agent replaces the values)
    "country": ["USA", "China", "India", "Brazil", "Germany", "Japan"], "region": ["North", "South", "East", "West"],
    "plan": ["Free", "Pro", "Team", "Enterprise"], "channel": ["Search", "Direct", "Social", "Email", "Ads"],
    "device": ["Mobile", "Desktop", "Tablet"], "browser": ["Chrome", "Safari", "Edge", "Firefox"],
    "bean": ["Arabica", "Robusta", "Liberica", "Excelsa"], "platform": ["iOS", "Android", "Web"],
    "language": ["Python", "JavaScript", "Go", "Rust", "Java"], "team": ["Design", "Engineering", "Sales", "Support"],
    "department": ["Design", "Engineering", "Sales", "Support"], "age": ["18-24", "25-34", "35-44", "45-54", "55+"],
    "city": ["London", "Paris", "Berlin", "Madrid", "Rome"], "source": ["Solar", "Wind", "Hydro", "Gas", "Coal"],
    "energy": ["Solar", "Wind", "Hydro", "Gas", "Coal"], "fruit": ["Apples", "Bananas", "Oranges", "Grapes", "Pears"],
    "continent": ["Asia", "Europe", "Africa", "Americas", "Oceania"], "product": ["Basic", "Plus", "Pro", "Max"],
    "segment": ["Consumer", "SMB", "Enterprise"], "store": ["Downtown", "Airport", "Mall", "Online"],
    "category": ["Food", "Housing", "Transport", "Leisure", "Other"], "expense": ["Rent", "Food", "Transport", "Fun"],
}
_MONEY = {"revenue", "sales", "income", "mrr", "arr", "profit", "gmv", "bookings", "spend", "cost", "costs", "price",
          "budget", "earnings", "salary", "salaries", "donations", "funding", "aov", "ltv"}
_PERCENT = {"churn", "rate", "conversion", "retention", "margin", "ctr", "bounce", "uptime", "share", "percent",
            "percentage", "growth", "satisfaction", "nps", "utilization", "occupancy"}
_DOWN_GOOD = {"churn", "bounce", "refund", "refunds", "cost", "costs", "latency", "errors", "error", "complaints",
              "spend", "downtime", "returns", "tickets", "response"}
_PROTO = {"https": "HTTPS", "http": "HTTP", "grpc": "gRPC", "rest": "REST", "websocket": "WebSocket",
          "websockets": "WebSocket", "graphql": "GraphQL", "sql": "SQL", "tcp": "TCP", "amqp": "AMQP", "mqtt": "MQTT",
          "pubsub": "pub/sub", "pub/sub": "pub/sub", "jdbc": "JDBC", "smtp": "SMTP"}
_ROLES = (   # tier, words (a component's words decide its tier; the first tier that matches wins)
    ("clients", "client clients browser browsers frontend front-end spa mobile ios android user users customer "
                "customers desktop"),
    ("edge", "load balancer lb balancer gateway cdn proxy nginx ingress firewall dns router haproxy envoy waf"),
    ("data", "db database databases postgres postgresql mysql mariadb mongo mongodb redis cache memcached queue "
             "bus kafka rabbitmq sqs pubsub s3 storage bucket blob elasticsearch opensearch warehouse snowflake "
             "bigquery sqlite dynamodb cassandra stream topic"),
)
_TIER_LOOK = {"clients": ("Clients", "users"), "edge": ("Edge", "shield"), "services": ("Services", "layers"),
              "data": ("Data", "database")}
_NODE_ICONS = [("balancer", "network"), ("gateway", "shield"), ("proxy", "shield"), ("firewall", "shield"),
               ("cdn", "globe"), ("dns", "globe"), ("browser", "monitor"), ("web", "monitor"), ("desktop", "monitor"),
               ("mobile", "phone"), ("ios", "phone"), ("android", "phone"), ("user", "user"), ("customer", "users"),
               ("client", "users"), ("auth", "key"), ("login", "key"), ("identity", "key"), ("payment", "tag"),
               ("billing", "tag"), ("search", "search"), ("elastic", "search"), ("mail", "mail"),
               ("notification", "bell"), ("redis", "bolt"), ("cache", "bolt"), ("memcached", "bolt"),
               ("bus", "network"), ("queue", "layers"), ("kafka", "layers"), ("rabbitmq", "layers"),
               ("stream", "layers"), ("topic", "layers"), ("s3", "folder"), ("storage", "folder"),
               ("bucket", "folder"), ("blob", "folder"), ("db", "database"), ("database", "database"),
               ("postgres", "database"), ("sql", "database"), ("mongo", "database"), ("dynamo", "database"),
               ("cassandra", "database"), ("warehouse", "database"), ("worker", "cube"), ("function", "code"),
               ("lambda", "code"), ("chat", "link"), ("api", "server"), ("server", "server"),
               ("service", "server"), ("app", "server")]
_CYLINDER = ("db", "database", "postgres", "sql", "mongo", "dynamo", "cassandra", "warehouse", "sqlite")
_ASYNC = ("bus", "queue", "kafka", "rabbitmq", "stream", "topic", "pubsub", "sqs")
_ICON_SYN = {"settings": "gear", "cog": "gear", "profile": "user", "account": "user", "person": "user",
             "people": "users", "team": "users", "shop": "cart", "basket": "cart", "email": "mail",
             "chat": "message", "messages": "message", "notification": "bell", "notifications": "bell",
             "money": "dollar", "payment": "credit-card", "card": "credit-card", "timer": "clock", "time": "clock",
             "weather": "cloud-sun", "photo": "camera", "picture": "image", "call": "phone-call",
             "security": "shield", "delete": "trash", "like": "heart", "favorite": "heart", "favourite": "heart",
             "analytics": "chart", "stats": "chart", "heart rate": "heartbeat line", "sleep": "crescent moon",
             "water": "water drop", "energy": "bolt", "power": "bolt", "world": "globe", "travel": "plane",
             "flight": "plane", "document": "file", "temperature": "thermometer", "idea": "bulb", "logout": "logout"}


_BASE_ICONS = set("alert bell bolt bot bulb calendar camera cart chart check chip clock cloud code cube database download "
                  "file folder gear globe heart home image info key layers link lock mail monitor network phone pin "
                  "play refresh search server shield sparkles star tag terminal trend upload user users wifi".split())


def _py(o):
    """A Python literal with double-quoted strings (starter code reads like the templates)."""
    if isinstance(o, str):
        return json.dumps(o, ensure_ascii=False)
    if isinstance(o, float):
        return repr(round(o, 3))
    if isinstance(o, (list, tuple)):
        body = ", ".join(_py(x) for x in o)
        return f"[{body}]" if isinstance(o, list) else f"({body}{',' if len(o) == 1 else ''})"
    if isinstance(o, dict):
        return "{" + ", ".join(f"{_py(k)}: {_py(v)}" for k, v in o.items()) + "}"
    return repr(o)


def _assign(name, value, comment=""):
    """NAME = literal, one element per line when it is long."""
    one = f"{name} = {_py(value)}" + (f"  # {comment}" if comment else "")
    if len(one) <= 116 or not isinstance(value, (list, dict)) or len(value) < 2:
        return [one]
    items = list(value.items()) if isinstance(value, dict) else list(value)
    open_, close = ("{", "}") if isinstance(value, dict) else ("[", "]")
    lines = [f"{name} = {open_}" + (f"  # {comment}" if comment else "")]
    for it in items:
        lines.append("    " + (f"{_py(it[0])}: {_py(it[1])}," if isinstance(value, dict) else f"{_py(it)},"))
    return lines + [close]


def _quotes(prompt):
    return [m.group(1).strip() for m in _QUOTE_RE.finditer(prompt or "") if m.group(1).strip()]


def _unquoted(prompt):
    return _QUOTE_RE.sub(" ", prompt or "")


def _cap(s):
    s = (s or "").strip()
    return s[:1].upper() + s[1:]


def _sing_word(w):
    lw = w.lower()
    if len(w) > 3 and lw.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and re.search(r"(?:ches|shes|sses|xes)$", lw):
        return w[:-2]
    if len(w) > 3 and lw.endswith("s") and not lw.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def _items(text):
    """[(label, count, raw)] of a list: "load balancer, two API servers, a Redis bus and Postgres"."""
    text = re.split(r"(?<!\d)\.(?:\s|$)|;|\s+-\s+(?=[a-z])", text)[0]
    parts = re.split(r",\s*(?:and\s+|or\s+|then\s+)?|\s+and\s+(?:then\s+)?|\s+&\s+|\s+then\s+|\s*->\s*|\s*→\s*|\s*>\s+",
                     text)
    out = []
    for p in parts:
        raw = p.strip(" .!?:")
        words = raw.split()
        while words and words[0].lower() in ("the", "a", "an", "with", "and", "plus", "its", "their", "some"):
            words = words[1:]
        n = 1
        if words and (words[0].lower() in _NUMW or words[0].isdigit()) and len(words) > 1:
            n = int(words[0]) if words[0].isdigit() else _NUMW[words[0].lower()]
            words = words[1:]
            if n > 1:
                words[-1] = _sing_word(words[-1])
        label = " ".join(words).strip()
        if label and len(label) <= 48:
            out.append((label, max(1, min(n, 12)), raw))
    return out


def _tail(text, starts=("between", "with", "showing", "including", "for", "of")):
    """The list a request names: after a colon, else after the first of ``starts``."""
    if ":" in text:
        return text.split(":", 1)[1]
    for w in starts:
        m = re.search(r"\b" + w + r"\s+(.+)$", text, re.I)
        if m:
            return m.group(1)
    return ""


def _subject(text, heads, stop=r":|,|;|\.(?:\s|$)|\s+\d|\s+(?:with|between|showing|including|in|over|across|by|from|for|"
                                r"at|on|to|using|that|which|where)\b"):
    """The phrase after "<head> of / for / about / on": "a chat app" of "an architecture diagram of a chat app"."""
    m = re.search(r"\b(?:" + heads + r")s?\s+(?:of|for|about|on|showing)\s+(?:an?\s+|the\s+|my\s+|our\s+)?(.+)", text,
                  re.I)
    if not m:
        return ""
    s = re.split(stop, m.group(1), maxsplit=1)[0]
    return s.strip(" .")


def _seeded(text):
    import random
    return random.Random(sum(ord(c) * (i + 1) for i, c in enumerate(text or "")) % 100003)


def _slug(label, used):
    base = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:16] or "n"
    sid, i = base, 2
    while sid in used:
        sid, i = f"{base}{i}", i + 1
    used.add(sid)
    return sid


def _period(low):
    """(x labels, word) for "12 months", "monthly", "weekly", "2015 to 2025", "quarterly" ..., else (None, None)."""
    m = re.search(r"\b(?:from\s+)?((?:19|20)\d\d)\s*(?:to|-|–|through|until)\s*((?:19|20)\d\d)\b", low)
    if m and 1 <= int(m.group(2)) - int(m.group(1)) <= 40:
        return [str(y) for y in range(int(m.group(1)), int(m.group(2)) + 1)], "year"
    m = re.search(r"\b(\d+|" + "|".join(_NUMW) + r")\s+(months|weeks|days|years|quarters|hours)\b", low)
    n = None
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else _NUMW[m.group(1)]
        unit = m.group(2)
    elif re.search(r"\bmonthly|per month|each month|by month|months\b", low):
        unit = "months"
    elif re.search(r"\bweekly|per week|each week|by week\b", low):
        unit = "weeks"
    elif re.search(r"\bdaily|per day|each day|days of the week|this week\b", low):
        unit = "days"
    elif re.search(r"\bquarterly|quarters\b", low):
        unit = "quarters"
    elif re.search(r"\byearly|annual|per year|over the years|years\b", low):
        unit = "years"
    elif re.search(r"\bhourly|per hour|hours\b", low):
        unit = "hours"
    else:
        return None, None
    if unit == "months":
        n = min(n or 12, 36)
        return [_MONTHS[i % 12] + ("" if n <= 12 else f" {i // 12 + 1}") for i in range(n)], "month"
    if unit == "weeks":
        return [f"W{i + 1}" for i in range(min(n or 8, 52))], "week"
    if unit == "days":
        n = n or 7
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        return (days[:n] if n <= 7 else [f"Day {i + 1}" for i in range(min(n, 60))]), "day"
    if unit == "quarters":
        return [f"Q{i % 4 + 1}" + ("" if (n or 4) <= 4 else f" Y{i // 4 + 1}") for i in range(min(n or 4, 16))], "quarter"
    if unit == "hours":
        return [f"{i:02d}:00" for i in range(min(n or 24, 24))], "hour"
    n = min(n or 6, 40)
    return [str(2026 - n + 1 + i) for i in range(n)], "year"


def _cats_for(words, n=5):
    for w in words:
        w = _sing_word(w.lower())
        if w in _CATS:
            return list(_CATS[w])
    noun = next((_sing_word(w) for w in words if len(w) > 2), "Item")
    return [f"{_cap(noun)} {c}" for c in "ABCDE"[:n]]


def _values(x, measure, rng, kind="line"):
    """Plausible example values for categories ``x`` of ``measure`` (seasonal rain / temperature, growth ...)."""
    import math
    n, low = len(x), measure.lower()
    if kind == "donut":
        raw = sorted((rng.uniform(4, 40) for _ in range(n)), reverse=True)
        tot = sum(raw)
        vals = [round(100 * v / tot) for v in raw]
        vals[0] += 100 - sum(vals)
        return vals
    months = all(str(c)[:3] in _MONTHS for c in x)
    if months and re.search(r"rain|precipitation|snow|wet", low):
        return [round(12 + 95 * (math.cos(2 * math.pi * _MONTHS.index(str(c)[:3]) / 12) + 1) / 2
                      * rng.uniform(0.85, 1.1)) for c in x]
    if months and re.search(r"temperat|heat|sun|tourism|visitors", low):
        return [round(9 + 16 * (1 - math.cos(2 * math.pi * (_MONTHS.index(str(c)[:3]) - 0.5) / 12)) / 2
                      + rng.uniform(-1, 1), 1) for c in x]
    if re.search(r"growth|users|signups|sign-ups|downloads|revenue|sales|subscribers|customers|visits|traffic", low) \
            and kind in ("line", "area"):
        v, out = rng.uniform(30, 60), []
        for _ in x:
            out.append(round(v))
            v *= 1 + rng.uniform(0.01, 0.12)
        return out
    if kind in ("bar", "barh"):
        return sorted((round(rng.uniform(20, 95)) for _ in x), reverse=True)
    v, out = rng.uniform(40, 70), []
    for _ in x:
        out.append(round(v))
        v = max(5.0, v + rng.uniform(-8, 10))
    return out


def _unit(measure):
    low = measure.lower()
    if re.search(r"rain|precipitation|snowfall", low):
        return " mm"
    if re.search(r"temperat", low):
        return "°C"
    if set(re.findall(r"[a-z]+", low)) & _MONEY or "$" in low:
        return "$"
    if set(re.findall(r"[a-z]+", low)) & _PERCENT or "%" in low:
        return "%"
    return None


def _chart_phrases(low):
    """[(type, subject words)] of "a line chart of signups", "a donut of plans", "bar chart of cups per day"."""
    out = []
    for m in re.finditer(r"\b(line|area|bar|column|horizontal bar|stacked bar|scatter|donut|doughnut|pie|histogram)"
                         r"(?:\s+(?:chart|graph|plot))?\s+(?:of|for|showing|by|with)\s+(?:the\s+)?"
                         r"([a-z0-9][a-z0-9 \-/%]*?)(?=\s*(?:,|\.|;|$|\band\b|\bwith\b|\bfor\b(?!\s+each)|\bon\b))", low):
        t = m.group(1)
        t = {"column": "bar", "doughnut": "donut", "pie": "donut", "histogram": "bar", "horizontal bar": "barh",
             "stacked bar": "bar"}.get(t, t)
        out.append((t, m.group(2).strip()))
    return out


def _kpi(name, rng):
    words = set(re.findall(r"[a-z]+", name.lower()))
    good = "down" if words & _DOWN_GOOD else "up"
    if words & _MONEY:
        v, fmt = round(rng.uniform(120, 2400)) * 1000, "$.2k"
    elif words & _PERCENT:
        v, fmt = (round(rng.uniform(1.5, 6.5), 1) if good == "down" else round(rng.uniform(20, 80), 1)), "%"
    elif words & {"latency", "response", "load"}:
        v, fmt = round(rng.uniform(80, 400)), " ms"
    else:
        v, fmt = round(rng.uniform(1200, 48000)), None
    d = round(rng.uniform(2, 15), 1)
    return (name, v, -d if good == "down" else d, fmt, good)


def _diagram_type(low):
    if re.search(r"\bsequence\b", low):
        return "sequence"
    if re.search(r"\btimeline|roadmap|milestones?\b", low):
        return "timeline"
    if re.search(r"\borg(?:ani[sz]ation(?:al)?)? chart|org-chart|hierarchy|mind ?map|family tree|tree diagram|\btree\b",
                 low):
        return "tree"
    if re.search(r"\bflow ?chart|process|workflow|decision|\bsteps?\b|\bflow of\b|user flow|pipeline", low) and \
            not re.search(r"\barchitecture|system|infrastructure|microservice", low):
        return "flowchart"
    return "architecture"


def _role(label):
    low = label.lower()
    toks = set(re.findall(r"[a-z0-9]+", low))
    if re.search(r"\b(?:web|mobile|ios|android|desktop|client)\s+app\b", low):
        return "clients"
    for tier, ws in _ROLES:
        wl = ws.split()
        if toks & set(wl) or any(" " in w and w in low for w in ("load balancer",)):
            return tier
    return "services"


def _node_icon(label):
    low = label.lower()
    return next((ic for k, ic in _NODE_ICONS if re.search(r"\b" + k, low)), "server")


def _diagram(text, low, title, rng):
    """(data lines, build lines or None, report) for diagram requests."""
    typ = _diagram_type(low)
    items = _items(_tail(text))
    rep = [f"type     {typ}"]
    if typ == "architecture":
        tiers, explicit = {}, []
        for label, n, raw in items:
            m = re.match(r"(?:an?\s+|the\s+)?(.+?)\s+(?:tier|layer)\s*(?:with|:|of|holding|running)?\s+(.+)$", raw, re.I)
            if m:                                           # "a data tier with Postgres and Redis": a named tier
                explicit.append((_cap(m.group(1)) + " " + ("tier" if "tier" in raw.lower() else "layer"),
                                 [(_cap(lb), c, r) for lb, c, r in _items(m.group(2))]))
        if explicit:
            groups = [(g, mem) for g, mem in explicit]
        else:
            for label, n, raw in items:
                tiers.setdefault(_role(label), []).append((label, n, raw))
            groups = [(_TIER_LOOK[t][0], tiers[t]) for t in ("clients", "edge", "services", "data") if t in tiers]
        used, tier_rows, tier_ids, drawn = set(), [], [], []
        for gl, members in groups:
            nodes = []
            for label, n, raw in members:
                lab = re.sub(r"\s+(?:over|via|using)\s+\S+$", "", _cap(label))
                for i in range(n):
                    nm = lab + (f" {i + 1}" if n > 1 else "")
                    nodes.append((_slug(nm, used), nm, _node_icon(lab),
                                  "cylinder" if any(w in lab.lower() for w in _CYLINDER) else "rounded", raw))
                    drawn.append(nm)
            gid = _slug(gl, used) if len(nodes) > 1 or explicit else None
            icon = next((ic for t, (lb, ic) in _TIER_LOOK.items() if lb == gl), "layers")
            tier_rows.append((gid, gl if gid else None, icon if gid else None, [n_[:4] for n_ in nodes]))
            tier_ids.append(nodes)
        protos = [(_PROTO[p.lower()], p) for p in re.findall(r"\b(https?|grpc|rest|websockets?|graphql|sql|tcp|amqp|mqtt|"
                                                             r"pub/?sub|jdbc|smtp)\b", text, re.I)]
        free = [p for p, raw in protos if not any(raw.lower() in r_.lower() for nodes in tier_ids for *_x, r_ in nodes)]
        edges = []
        for a, b in zip(tier_ids, tier_ids[1:]):
            src_ids = [x[0] for x in a if not any(w in x[1].lower() for w in _ASYNC)] or [x[0] for x in a]
            for nid, nm, _ic, _sh, raw in b:
                bound = next((p for p, r in protos if r.lower() in raw.lower()), None)
                lab = bound or (free.pop(0) if free and not edges else None)
                asyn = any(w in nm.lower() for w in _ASYNC)
                edges.append((src_ids if len(src_ids) > 1 else src_ids[0], nid, lab, "dashed" if asyn else "solid"))
        if len(tier_ids) == 1 and len(tier_ids[0]) > 1:      # one tier: the components in the order named
            ids = [x[0] for x in tier_ids[0]]
            edges = [(a, b, None, "solid") for a, b in zip(ids, ids[1:])]
        sub = "solid = request, dashed = async event" if any(e[3] == "dashed" for e in edges) else ""
        data = [f"TITLE = {_py(title)}", f"SUBTITLE = {_py(sub)}",
                "TIERS = [  # (group id or None, group label, group icon, [(node id, label, icon, shape)])"]
        data += [f"    {_py(t)}," for t in tier_rows] + ["]", "EDGES = [  # (from, to, label, style solid / dashed / dotted)"]
        data += [f"    {_py(e)}," for e in edges] + ["]"]
        rep.append("drawn    nodes: " + (", ".join(drawn) if drawn else "none named: the request names no components"))
        return data, None, rep
    if typ == "flowchart":
        used, nodes, edges = set(), [], []
        steps = items or [("Start", 1, ""), ("Step", 1, ""), ("Done", 1, "")]
        for i, (label, n, raw) in enumerate(steps):
            dec = label.endswith("?") or bool(re.match(r"(?:is|are|does|do|can|has|if|valid|approved?|decide|whether)\b",
                                                        label, re.I))
            shape = "diamond" if dec else "pill" if i in (0, len(steps) - 1) else "rounded"
            nodes.append((_slug(label, used), _cap(label), None, shape))
        for i, (a, b) in enumerate(zip(nodes, nodes[1:])):
            edges.append((a[0], b[0], "yes" if a[3] == "diamond" else None, "solid"))
            if a[3] == "diamond" and i > 0:
                edges.append((a[0], nodes[i - 1][0], "no", "dashed"))
        data = [f"TITLE = {_py(title)}", 'SUBTITLE = ""', "TIERS = [  # one row of steps, chained in order",
                f"    {_py((None, None, None, nodes))},", "]", "EDGES = [  # (from, to, label, style)"]
        data += [f"    {_py(e)}," for e in edges] + ["]"]
        rep.append("drawn    steps: " + ", ".join(n_[1] for n_ in nodes))
        return data, None, rep
    if typ == "sequence":
        used, parts = set(), []
        for label, n, raw in (items or [("Client", 1, ""), ("Server", 1, ""), ("Database", 1, "")]):
            lw = label.lower()
            kind = "actor" if re.search(r"\b(user|customer|person|browser)\b", lw) else \
                "database" if any(w in lw for w in _CYLINDER) else "box"
            parts.append((_slug(label, used), _cap(label), kind))
        ids = [p[0] for p in parts]
        auth = bool(re.search(r"\b(login|log in|oauth|sign[ -]?in|auth\w*|sso|token)\b", low))
        fwd = ["Sign in", "Authorization request", "Verify"] if auth else ["Request", "Call", "Query", "Call"]
        back = ["Token", "Session", "OK"] if auth else ["Result", "Response", "Rows", "Result"]
        msgs = [(a, b, fwd[min(i, len(fwd) - 1)], False) for i, (a, b) in enumerate(zip(ids, ids[1:]))]
        msgs += [(b, a, back[min(i, len(back) - 1)], True) for i, (a, b) in enumerate(reversed(list(zip(ids, ids[1:]))))]
        data = [f"TITLE = {_py(title)}", 'SUBTITLE = ""', "PARTICIPANTS = [  # (id, label, kind box / actor / database)"]
        data += [f"    {_py(p)}," for p in parts] + ["]", "MESSAGES = [  # (from, to, label, reply?) in time order"]
        data += [f"    {_py(m_)}," for m_ in msgs] + ["]"]
        build = ["def build():", "    d = Sequence(theme=THEME, title=TITLE, subtitle=SUBTITLE or None, autonumber=True)",
                 "    for pid, label, kind in PARTICIPANTS:", "        d.participant(pid, label, kind=kind)",
                 "    for a, b, label, reply in MESSAGES:                 # with d.frame(\"alt\", ...): for branches",
                 "        (d.reply if reply else d.message)(a, b, label)", "    return d"]
        rep.append("drawn    participants: " + ", ".join(p[1] for p in parts) + f"; {len(msgs)} example messages")
        return data, build, rep
    if typ == "timeline":
        year = (re.findall(r"\b((?:19|20)\d\d)\b", text) or [None])[0]
        ms = []
        for i, (label, n, raw) in enumerate(items):
            m = re.match(r"(.+?)\s+(?:in|on|by|at|for|during)\s+(.+)$", label, re.I)
            m2 = re.match(r"(q[1-4]|(?:19|20)\d\d|" + "|".join(x.lower() for x in _MONTHS) + r")\w*\s*[:\-–]?\s+(.+)$",
                          label, re.I)
            what, when = (m.group(1), m.group(2)) if m else (m2.group(2), m2.group(1)) if m2 else (label, None)
            wl = (when or "").lower()
            if re.fullmatch(r"q[1-4]", wl):
                when = f"{wl.upper()} {year}" if year else wl.upper()
            elif wl[:3] in [x.lower() for x in _MONTHS]:
                when = f"{year or 2027}-{[x.lower() for x in _MONTHS].index(wl[:3]) + 1:02d}"
            elif not when:
                when = i + 1
            ms.append((when, _cap(what)))
        if not ms:
            ms = [(1, "Start"), (2, "Milestone"), (3, "Launch")]
        if any(isinstance(w, int) for w, _ in ms) and not all(isinstance(w, int) for w, _ in ms):
            ms = [(i + 1, lb) for i, (_w, lb) in enumerate(ms)]
        data = [f"TITLE = {_py(title)}", 'SUBTITLE = ""', *_assign("MILESTONES", ms, "(when, label): \"Q1 2027\", "
                                                                                     "\"2027-03\", a year or a number")]
        build = ["def build():", "    d = Timeline(theme=THEME, title=TITLE, subtitle=SUBTITLE or None)",
                 "    for when, label in MILESTONES:                     # d.phase(start, end, label) for spans",
                 "        d.milestone(when, label)", "    return d"]
        rep.append("drawn    milestones: " + ", ".join(lb for _w, lb in ms))
        return data, build, rep
    org = bool(re.search(r"\borg|hierarchy|company|team\b", low))
    mind = bool(re.search(r"mind ?map", low))
    names = []
    for label, n, raw in items:
        names += [_cap(label) + (f" {i + 1}" if n > 1 else "") for i in range(n)]
    if mind or not names:
        root, kids = title, names
    else:
        root, kids = names[0], names[1:]
    tree = {root: {k: {} for k in kids}}
    data = [f"TITLE = {_py(title)}", 'SUBTITLE = ""', f"TREE = {_py(tree)}  # nested dict: name -> children"]
    kind = "mindmap" if mind else "org" if org else "tree"
    build = ["def build():",
             f'    return Tree(TREE, kind="{kind}", theme=THEME, title=TITLE, subtitle=SUBTITLE or None)   '
             '# "Name | Role" labels']
    rep.append(f"drawn    {kind}: {root} -> " + (", ".join(kids) or "no children named"))
    return data, build, rep


def _chart(text, low, title, rng):
    typ = next((t for t in ("barh", "scatter", "donut", "area", "bar", "line") if re.search(
        {"barh": r"horizontal bar", "scatter": r"\bscatter", "donut": r"\b(?:donut|doughnut|pie)\b",
         "area": r"\barea (?:chart|graph|plot)", "bar": r"\b(?:bar|column|histogram)s?\b",
         "line": r"\bline (?:chart|graph|plot)|\btrend\b"}[t], low)), None)
    x, unit = _period(low)
    items = [(_cap(lb), n) for lb, n, _r in _items(text.split(":", 1)[1])] if ":" in text else []
    subj = _subject(text, r"chart|graph|plot|histogram|diagram") or title
    vs = re.search(r"([A-Za-z][\w\-]*(?:\s[A-Za-z][\w\-]*)?)\s+(?:vs\.?|versus|against)\s+([A-Za-z][\w\-]*(?:\s[A-Za-z]"
                   r"[\w\-]*)?)", text)
    cmp_ = re.search(r"\bcompar(?:ing|e|es)\s+(?:the\s+)?(.+?)(?:\s+(?:over|across|by|in|for|from|during)\b|[,.;:]|$)",
                     text, re.I)
    if vs:
        names = [_cap(vs.group(1).split()[-1]), _cap(vs.group(2).split()[0])]
    elif cmp_:
        names = [_cap(lb) for lb, _n, _r in _items(cmp_.group(1))][:4]
        if len(names) > 1 and len(names[-1].split()) > 1 and all(len(x.split()) == 1 for x in names[:-1]):
            names[-1] = names[-1].split()[0]                # "tea and coffee sales": Tea, Coffee (of sales)
    else:
        measure = re.sub(r"\b(?:monthly|weekly|daily|yearly|annual|quarterly|hourly)\b\s*", "", subj, flags=re.I)
        measure = re.split(r"\s+(?:in|of|over|across|for|per|by|from)\s+", measure, maxsplit=1)[0]
        names = [_cap(measure.strip()) or "Value"]
    if items and (typ in (None, "bar", "barh", "donut") or not x):
        x = [lb for lb, _n in items]
        typ = typ or "bar"
    by = re.search(r"\bby\s+([a-z]+)", low)
    if x is None:
        x = _cats_for([by.group(1)] if by else subj.split()) if (by or typ in ("bar", "barh", "donut")) else \
            [str(y) for y in range(2021, 2027)]
    typ = typ or ("line" if unit else "bar")
    series = {nm: _values(x, nm + " " + subj, rng, typ) for nm in names[:1 if typ == "donut" else 4]}
    fmt = _unit(subj + " " + " ".join(names))
    callout = None
    if typ in ("line", "area", "bar") and x:
        ys = next(iter(series.values()))
        i = max(range(len(ys)), key=lambda k: ys[k])
        callout = (f"Highest: {x[i]}", (x[i], ys[i]))
    data = [f'CHART = "{typ}"                      # line area bar barh scatter donut', f"TITLE = {_py(title)}",
            f"SUBTITLE = {_py(_cap(subj) if _cap(subj) != title else '')}", *_assign("X", x),
            *_assign("SERIES", series, "example values: replace them with the real data"),
            f"CALLOUT = {_py(callout)}     # text, (x, y) of the point it explains; None for none",
            'SOURCE = "Source: example values"']
    rep = [f"chart    {typ}: {', '.join(series)} over {len(x)} {unit or 'categories'} ({x[0]} .. {x[-1]})"]
    return data, fmt, rep


def _dashboard(text, low, title, rng):
    kp = re.search(r"\b(?:kpis?|tiles?|metrics?|stats?|numbers?)\s+(?:tiles?\s+|cards?\s+)?(?:for|of|showing|on|:|"
                   r"with)\s+(.+?)(?=,\s*(?:an?|the)\s|;|\.\s|$|\s+(?:plus|and)\s+(?:an?|the)\s)", text, re.I)
    names = [_cap(lb) for lb, _n, _r in _items(kp.group(1))][:5] if kp else []
    charts = _chart_phrases(low)
    x, unit = _period(low)
    x = x or _MONTHS[:]
    if not names:
        names = ["Revenue", "Active users", "Conversion rate"]
    kpis = [_kpi(nm, rng) for nm in names]
    trend = next(((t, s) for t, s in charts if t in ("line", "area", "bar")), None)
    share = next(((t, s) for t, s in charts if t == "donut"), None)
    rest = [c for c in charts if c not in (trend, share)]
    tname = _cap(trend[1]) if trend else names[0]
    tfmt = _unit(tname)
    tvals = _values(x, tname, rng, trend[0] if trend else "line")
    if tfmt == "$":
        tvals = [v * 1000 for v in tvals]
        tfmt = "$k"
    trend_t = (trend[0] if trend else "line", tname, tfmt, x, {tname: tvals})
    sname = _cap(share[1]) if share else "Share by channel"
    sc = _cats_for(sname.split(), 4)
    share_t = (sname, dict(zip(sc, _values(sc, sname, rng, "donut"))))
    rname = _cap(rest[0][1]) if rest else f"{names[0]} by region"
    rc = _cats_for(rname.split() if rest else ["region"], 5)
    rfmt = "$k" if _unit(rname) == "$" else None
    rv = _values(rc, rname, rng, "bar")
    rank_t = (rname, rfmt, dict(zip(rc, [v * 1000 for v in rv] if rfmt else rv)))
    period = {"month": f"Last {len(x)} months", "week": f"Last {len(x)} weeks", "day": f"Last {len(x)} days",
              "year": f"{x[0]}-{x[-1]}", "quarter": "Quarterly"}.get(unit, "Last 12 months")
    data = [f"TITLE, PERIOD = {_py(title)}, {_py(period)}",
            "KPIS = [  # name, value, delta %, fmt, good direction (example values)"]
    data += [f"    {_py(k)}," for k in kpis] + ["]", f"TREND = {_py(trend_t)}", f"SHARE = {_py(share_t)}",
                                                  f"RANK = {_py(rank_t)}"]
    rep = ["drawn    KPI tiles: " + ", ".join(names),
           f"drawn    panels: {trend_t[0]} of {tname}, donut of {sname}, bars of {rname}"]
    return data, rep


def _infographic(text, low, title, rng):
    charts = _chart_phrases(low)
    share = next(((t, s) for t, s in charts if t == "donut"), None)
    bars = next(((t, s) for t, s in charts if t in ("bar", "barh", "line", "area")), None)
    by = re.search(r"\bby\s+([a-z]+)", low)
    sname = _cap(share[1]) if share else "Share"
    sc = _cats_for(sname.split(), 3)
    bname = _cap(bars[1]) if bars else "By category"
    bc = _cats_for(([by.group(1)] if by else []) + bname.split(), 5)[:5]
    bv = _values(bc, bname, rng, "bar")
    fmt = "%" if _unit(bname) == "%" else None
    st = re.search(r"\b(?:steps?|stages?|process|how it works|journey|flow)\s*(?::|of|from)\s*(.+?)(?=;|\.\s|$)", text,
                   re.I)
    steps = [(_cap(lb), n) for lb, n, _r in _items(st.group(1))][:6] if st else []
    used = set()
    step_rows = [(_slug(lb, used), lb, _node_icon(lb) if _node_icon(lb) != "server" else "star") for lb, _n in steps] \
        or [("s1", "Step 1", "star"), ("s2", "Step 2", "layers"), ("s3", "Step 3", "check")]
    data = [f"TITLE = {_py(title)}", 'SUBTITLE = "Example values"',
            f"SHARE = {_py((sname, dict(zip(sc, _values(sc, sname, rng, 'donut')))))}",
            f"DATA = {_py((bname, fmt, dict(zip(bc, bv)), f'Highest: {bc[0]}'))}",
            f"STEPS = {_py(('How it works' if steps else 'Key steps', step_rows))}",
            'SOURCE = "Source: example values"']
    rep = [f"drawn    donut of {sname}, bars of {bname}" + (", steps: " + ", ".join(s for s, _n in steps) if steps
                                                            else "; no steps named: the steps panel shows three")]
    return data, rep


_UI_SCREENS = [("login", r"\b(?:login|log in|sign[ -]?in|sign[ -]?up|register)\b"),
               ("music_player", r"\b(?:music|player|playlist|podcast|now playing)\b"),
               ("weather", r"\b(?:weather|forecast)\b"), ("calendar", r"\b(?:calendar|schedule|agenda)\b"),
               ("settings", r"\b(?:settings|preferences)\b"), ("chat", r"\b(?:chat|messag\w*|inbox|conversation)\b"),
               ("checkout", r"\b(?:checkout|cart|payment)\b"), ("profile", r"\b(?:profile|account)\b"),
               ("dashboard", r"\b(?:dashboard|analytics)\b"), ("feed", r"\b(?:feed|social)\b"),
               ("onboarding", r"\b(?:onboarding|welcome)\b"), ("pricing", r"\bpricing\b"),
               ("landing", r"\b(?:landing|website|homepage|hero)\b"), ("kanban", r"\b(?:kanban|task board)\b")]
_UI_CONTENT = {"login": ("brand", "title"), "landing": ("brand", "title"), "pricing": ("brand", "title"),
               "dashboard": ("title",), "kanban": ("title",), "feed": ("title",), "onboarding": ("title",)}


def _app_name(text, prompt):
    m = re.search(r"\b(?:called|named)\s+['\"“]?([A-Z][\w\-]*(?:\s+[A-Z][\w\-]*)?)", prompt or "")
    if m:
        return m.group(1)
    m = re.search(r"\bfor\s+([A-Z][\w\-]+)\b", text)
    return m.group(1) if m else None


def _ui(text, low, prompt, rng):
    screen = next((n for n, rx in _UI_SCREENS if re.search(rx, low)), "home")
    device = "tablet" if re.search(r"\b(?:tablet|ipad)\b", low) else \
        "browser" if re.search(r"\b(?:desktop|web ?app|browser|website|laptop|landing|saas|web page|webpage)\b", low) \
        else "phone"
    theme = "light" if re.search(r"\blight\b", low) else "dark" if re.search(r"\bdark\b", low) else "light"
    app = _app_name(text, prompt) or _cap(screen.replace("_", " ")) if screen != "home" else \
        (_app_name(text, prompt) or "Home")
    quotes = _quotes(prompt)
    title = quotes[0] if quotes else {"login": f"Welcome to {app}", "landing": f"{app}, made simple",
                                      "home": f"Good morning"}.get(screen, app)
    items = [(lb, n) for lb, n, _r in _items(re.split(r"\bwith\b", text, maxsplit=1)[1])] \
        if re.search(r"\bwith\b", text) else []
    data = [f'DEVICE = "{device}"                        # phone, tablet, browser, desktop ...',
            f'THEME = "{theme}"                          # dark or light', 'ACCENT = "#ff7a59"',
            'STYLE = "modern"                        # modern, glass, material, wireframe', f"APP = {_py(app)}",
            f"TITLE = {_py(title)}"]
    if screen != "home":
        keys = _UI_CONTENT.get(screen, ())
        content = {k: ("APP" if k == "brand" else "TITLE") for k in keys}
        ctext = "{" + ", ".join(f'"{k}": {v}' for k, v in content.items()) + "}" if content else "None"
        body = ["def home():",
                f'    s = template("{screen}", DEVICE, theme=THEME, accent=ACCENT, style=STYLE, content={ctext})',
                "    return s                                         # add components: s.button(...), s.list_row(...)"]
        rep = [f"screen   {screen} ({device}, {theme}) from ui.template; app {app}"]
        if items:
            rep.append("named    " + ", ".join(lb for lb, _n in items) + f" (part of the {screen} screen)")
        return data, body, rep
    body = ["def home():", '    s = Screen(DEVICE, theme=THEME, accent=ACCENT, style=STYLE, name="home")',
            '    s.app_bar(APP, leading="menu", trailing=["bell"], center=False)',
            "    s.heading(TITLE, level=1)"]
    tabs, drawn = None, []
    scenes = iter(["ocean", "dusk", "forest", "lilac", "desert", "dawn"] * 3)
    for label, n in items:
        lw, lab = label.lower(), _cap(re.sub(r"\s*\b(?:cards?|buttons?|tabs?|lists?|sections?)\b", "", label).strip())
        lab = lab or _cap(label)
        if re.search(r"\btabs?\b|tab bar|navigation", lw):
            tabs = re.findall(r"[A-Z][a-z]+", label) or None
            continue
        if "card" in lw:
            body.append("    with s.hstack(gap=12, align=\"start\"):")
            for i in range(max(1, min(n, 3))):
                body.append(f'        s.image(height=170, scene="{next(scenes)}", caption={_py(lab)}, w="fill")')
        elif "button" in lw:
            body.append(f"    s.button({_py(lab)}, full=True)")
        elif "search" in lw:
            body.append(f"    s.search({_py('Search ' + (lab.lower() if lab.lower() != 'search' else ''))})")
        elif re.search(r"\blist\b", lw):
            body.append(f"    with s.list_group({_py(lab)}):")
            for i in range(3):
                body.append(f'        s.list_row("{lab} {i + 1}", "Details", leading="img:{next(scenes)}", chevron=True)')
        elif re.search(r"chart|graph", lw):
            body.append(f'    s.chart("line", {_py([round(rng.uniform(20, 90)) for _ in range(7)])}, height=140)')
        elif re.search(r"toggle|switch", lw):
            body.append(f"    s.toggle({_py(lab)}, True)")
        else:
            body.append(f"    s.list_row({_py(lab)}, leading=\"img:{next(scenes)}\", chevron=True)")
        drawn.append(lab)
    if len(body) == 4:                                  # nothing named: a search, a card row and a list
        body += ['    s.search("Search")', "    with s.hstack(gap=12, align=\"start\"):",
                 '        s.image(height=170, scene="ocean", caption="Featured", w="fill")',
                 '        s.image(height=170, scene="dusk", caption="New", w="fill")']
    tab_items = [(ic, t) for ic, t in zip(["home", "search", "heart", "user", "gear"], tabs or ["Home", "Search",
                                                                                                   "Saved", "Profile"])]
    body += [f"    s.tab_bar({_py(tab_items)}, active=0)", "    return s"]
    rep = [f"screen   home ({device}, {theme}); app {app}"]
    if drawn:
        rep.append("drawn    " + ", ".join(drawn))
    return data, body, rep


def _glyph(word, app=False):
    """A glyph name for an icon word: a vector icon, else a pictogram ("pictogram:<name>" in sheets)."""
    sys.path.insert(0, ROOT)
    from pixelcanvas import pictograms, vector
    w = word.lower().strip()
    syn = _ICON_SYN.get(w)
    icons = _BASE_ICONS
    if app:                                             # app words, pictograms and the base icons resolve anywhere
        ok = set(pictograms.names()) | set(getattr(vector, "APP_GLYPHS", {})) | _BASE_ICONS | {"weather"}
        for cand in ([syn] if syn else []) + [w, w.replace(" ", "-")] + list(pictograms.find(w))[:2]:
            if cand in ok:
                return cand
        return "star"
    if syn and syn in icons:
        return syn
    if w.replace(" ", "-") in icons:
        return w.replace(" ", "-")
    pic = ([syn] if syn and syn in pictograms.names() else []) + list(pictograms.find(w))
    return f"pictogram:{pic[0]}" if pic else "sparkles"


def _icons(text, low, prompt, res, title):
    quotes = _quotes(prompt)
    brand = _app_name(text, prompt) or (quotes[0] if quotes else None)
    if re.search(r"\blogo|logotype|wordmark|emblem\b", low) and not re.search(r"\bicons\b", low):
        sys.path.insert(0, ROOT)
        from pixelcanvas import pictograms
        subj = [w for w in re.findall(r"[a-z]+", re.split(r"\b(?:called|named)\b", low)[0])
                if w not in ("a", "an", "the", "logo", "for", "of", "my", "our", "modern", "minimal", "simple", "flat",
                             "brand", "company", "startup", "shop", "store")]
        pic = next((p for w in reversed(subj) for p in pictograms.find(w)[:1]), None)
        name = brand or _cap(subj[-1] if subj else "Brand")
        data = [f"TITLE = {_py(name)}",f"ICONS = [(TITLE, {_py('pictogram:' + pic if pic else 'logo')})]"]
        build = ["def logo(dwg, x, y, s, color):",
                 '    """The mark: a disc with the subject\'s glyph; replace with your own shapes."""',
                 "    dwg.circle(x, y, 0.5 * s, fill=color)",
                 "    glyph = ICONS[0][1]",
                 "    if glyph.startswith(\"pictogram:\"):",
                 "        dwg.embed(pictograms.drawing(glyph[10:], color=\"#ffffff\"), box=(x - 0.3 * s, y - 0.3 * s, "
                 "x + 0.3 * s, y + 0.3 * s))",
                 "", "", "def build(aspect=16 / 9):",
                 "    W = 1080 * aspect", "    dwg = Drawing(W, 1080, theme=THEME)",
                 "    with dwg.group(\"mark\", shadow=True):", "        logo(dwg, W / 2, 440, 380, dwg.theme.accent(0))",
                 "    with dwg.group(\"wordmark\"):",
                 "        dwg.text(W / 2, 800, TITLE, size=120, weight=\"bold\", anchor=\"ms\")",
                 "    return dwg", "", "",
                 "def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):",
                 "    cv = Canvas(width, height, THEMES[THEME].background)",
                 "    build(width / height).draw(cv, log=log)              # --svg exports the same mark",
                 "    return cv"]
        return data, build, [f"drawn    logo mark ({pic or 'abstract'}) + wordmark {name!r}"]
    tail = _tail(_unquoted(prompt), ("for", "of", "with", "showing"))
    names = [lb for lb, _n, _r in _items(tail)]
    names = [n_ for n_ in names if len(n_.split()) <= 3]
    m = re.search(r"\b(\d+|" + "|".join(_NUMW) + r")\s+(?:app\s+)?icons\b", low)
    count = (int(m.group(1)) if m.group(1).isdigit() else _NUMW[m.group(1)]) if m else None
    if not names:
        names = [g.head for g in res.groups if g.head and not getattr(g, "format", False)][:12] or \
            ["home", "search", "user", "bell", "gear", "heart"]
    if count and len(names) < count:
        names += [n_ for n_ in ["home", "search", "user", "bell", "gear", "heart", "mail", "chart", "cloud", "lock",
                               "star", "camera"] if n_ not in names][:count - len(names)]
    if re.search(r"\bapp icons?\b", low):
        apps = [(_cap(n_), _glyph(n_, app=True)) for n_ in names]
        data = [f"TITLE = {_py(title)}", *_assign("ICONS", apps, "(label, glyph): app words, "
                                                                                       "pictogram or icon names")]
        build = ["def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):",
                 "    cv = Canvas(width, height)",
                 "    log = log or (lambda m: None)",
                 "    log(\"app icons\")                                   # one rounded tile per app, labelled",
                 "    vector.app_icon_sheet(cv, ICONS, box=(0.05, 0.08, 0.95, 0.95))",
                 "    return cv"]
        return data, build, ["drawn    app icons: " + ", ".join(a for a, _g in apps)]
    icons = [(_cap(n_), _glyph(n_)) for n_ in names]
    data = [f"TITLE = {_py(title)}",
            *_assign("ICONS", icons, "(label, glyph): icon name, \"pictogram:<name>\" or \"logo\"")]
    return data, None, ["drawn    icons: " + ", ".join(f"{a} ({g})" for a, g in icons)]


def _mm(v, u):
    return float(v) * {"cm": 10, "m": 1000}.get((u or "mm").lower(), 1)


def _drawing(text, low, title):
    dims = re.search(r"(\d+(?:\.\d+)?)\s*(mm|cm|m)?\s*(?:x|×|by)\s*(\d+(?:\.\d+)?)\s*(mm|cm|m)?(?:\s*(?:x|×|by)\s*"
                     r"(\d+(?:\.\d+)?))?\s*(mm|cm|m)?", low)
    unit = (dims.group(6) or dims.group(4) or dims.group(2)) if dims else "mm"
    L, H, D = ((_mm(dims.group(1), unit), _mm(dims.group(3), unit), _mm(dims.group(5) or 0, unit)) if dims
               else (80.0, 50.0, 30.0))
    D = D or round(min(L, H) * 0.4)
    bore = re.search(r"(\d+(?:\.\d+)?)\s*mm\s+(?:diameter\s+)?(?:bore|centre hole|center hole|through[- ]hole|hole)(?!s)",
                     low)
    holes = re.search(r"\b(\d+|" + "|".join(_NUMW) + r")\s+(?:x\s+)?(?:m(\d+)|(\d+(?:\.\d+)?)\s*mm)\s+(?:\w+\s+)?holes",
                      low)
    cham = re.search(r"(\d+(?:\.\d+)?)\s*mm\s+(?:x\s*45\S*\s+)?chamfer|chamfer\s+(?:of\s+)?(\d+(?:\.\d+)?)", low)
    mat = re.search(r"\b(stainless steel|steel|aluminium|aluminum|brass|copper|bronze|cast iron|titanium|nylon|abs|"
                    r"pom|s355|s235|6061|7075)\b", low)
    views = [v for v in ("top", "side", "section") if re.search(r"\b" + v + r"\b", low)]
    if not re.search(r"\bviews?\b|\bsection\b", low):
        views = ["top", "section"]
    hole_n = (int(holes.group(1)) if holes.group(1).isdigit() else _NUMW[holes.group(1)]) if holes else 0
    hole_d = float(holes.group(2) or holes.group(3)) if holes else 0
    scale = next(s for s, f in (("5:1", 5), ("2:1", 2), ("1:1", 1), ("1:2", 0.5), ("1:5", 0.2), ("1:10", 0.1),
                                ("1:20", 0.05), ("1:50", 0.02)) if (L * f <= 170 and (H + D) * f <= 125) or f == 0.02)
    part = _cap(_subject(text, r"drawing|sheet|blueprint|part") or "part").upper()
    data = [f"TITLE, DRAWING_NO, MATERIAL = {_py(part)}, \"PC-2001\", {_py(mat.group(1).upper() if mat else 'S355')}",
            f"LENGTH, HEIGHT, DEPTH = {L:g}, {H:g}, {D:g}          # part size in mm",
            f"HOLE = {float(bore.group(1)) if bore else 0:g}                                   # bore diameter in mm "
            f"(0 = none)",
            f"HOLES = ({hole_n}, {hole_d:g})                             # count, diameter of the holes at the corners",
            f"CHAMFER = {float((cham.group(1) or cham.group(2))) if cham else 0:g}",
            f'SCALE = "{scale}"                               # drawing scale: the part fills the sheet']
    b = ["def build():", '    sh = Sheet("A3", scale=SCALE, title=TITLE, drawing_no=DRAWING_NO, material=MATERIAL, '
                         'style="white")',
         "    cx, cy = LENGTH / 2, HEIGHT / 2",
         "    n, d = HOLES",
         "    inset = max(d, 0.12 * min(LENGTH, HEIGHT))",
         "    corners = [(inset, inset), (LENGTH - inset, inset), (LENGTH - inset, HEIGHT - inset), (inset, HEIGHT - inset)]",
         "    corners = (corners * 2)[:n] if n else []",
         '    fr = sh.view("FRONT", origin=(60, 150))              # model mm, y up; origin = sheet mm of model (0, 0)',
         "    fr.rect(0, 0, LENGTH, HEIGHT, chamfer=CHAMFER)",
         "    if HOLE:", "        fr.hole(cx, cy, HOLE)", "        fr.dim_diameter(cx, cy, HOLE)",
         "    for x, y in corners:", "        fr.hole(x, y, d)",
         "    if corners:", '        fr.dim_diameter(*corners[0], d, text=f"{n}x Ø{d:g}", angle=225)',
         '    fr.dim_h((0, HEIGHT), (LENGTH, HEIGHT), side="top")', "    fr.dim_v((0, 0), (0, HEIGHT))"]
    if "section" in views:
        b.append('    fr.cutting_plane([(cx, 0), (cx, HEIGHT)], "A", direction="right")')
    if "top" in views:
        b += ['    top = sh.projected(fr, "top", gap=20)               # first angle: the top view goes below',
              "    top.rect(0, 0, LENGTH, DEPTH)",
              "    for x in ([cx - HOLE / 2, cx + HOLE / 2] if HOLE else []) + [c[0] + s * d / 2 for c in corners[:2] "
              "for s in (-1, 1)]:",
              '        top.line(x, 0, x, DEPTH, "hidden")              # line types: visible hidden center phantom',
              '    top.dim_v((0, 0), (0, DEPTH), side="left")']
    if "side" in views:
        b += ['    side = sh.projected(fr, "left", gap=40)', "    side.rect(0, 0, DEPTH, HEIGHT)",
              "    if HOLE:", '        for y in (cy - HOLE / 2, cy + HOLE / 2):',
              '            side.line(0, y, DEPTH, y, "hidden")',
              '    side.dim_h((0, HEIGHT), (DEPTH, HEIGHT), side="top")']
    elif "section" in views:
        b += ['    sec = sh.projected(fr, "left", gap=40).section_label("A")',
              "    cuts = [0, cy - HOLE / 2, cy + HOLE / 2, HEIGHT] if HOLE else [0, HEIGHT]",
              "    for y0, y1 in zip(cuts[::2], cuts[1::2]):",
              '        sec.hatch([(0, y0), (DEPTH, y0), (DEPTH, y1), (0, y1)], material="steel", outline=True)',
              '    sec.dim_h((0, HEIGHT), (DEPTH, HEIGHT), side="top")']
    b += ['    notes = ["Untoleranced dimensions ISO 2768-m."]',
          '    if CHAMFER:', '        notes.append(f"Chamfer {CHAMFER:g} x 45°.")',
          "    sh.notes(notes)", "    return sh"]
    feats = ([f"bore Ø{float(bore.group(1)):g}"] if bore else []) + ([f"{hole_n} holes Ø{hole_d:g}"] if holes else []) + \
        ([f"chamfer {float(cham.group(1) or cham.group(2)):g}"] if cham else [])
    rep = [f"drawn    {part} {L:g} x {H:g} x {D:g} mm" + (", " + ", ".join(feats) if feats else "") +
           f"; views: front, {', '.join(views)}; scale {scale}"]
    return data, b, rep


_ROOM_W = {"living": 3.0, "lounge": 3.0, "kitchen": 1.6, "dining": 1.5, "bedroom": 2.0, "master": 2.4, "bathroom": 1.0,
           "bath": 1.0, "shower": 0.8, "wc": 0.5, "toilet": 0.5, "study": 1.3, "office": 1.3, "hall": 0.8,
           "hallway": 0.8, "entrance": 0.7, "corridor": 0.7, "utility": 0.6, "laundry": 0.6, "storage": 0.5,
           "store": 0.5, "closet": 0.4, "pantry": 0.4, "garage": 3.0, "nursery": 1.4, "guest": 1.6, "balcony": 0.9}
_FURNITURE = {"bedroom": [("bed", 0, -1)], "master": [("bed", 0, -1)], "nursery": [("single_bed", 0, -1)],
              "guest": [("bed", 0, -1)], "living": [("sofa", 0, -1), ("table", 0, 0.1)],
              "lounge": [("sofa", 0, -1)], "kitchen": [("counter", 0, -1), ("fridge", 0.38, -1)],
              "dining": [("table", 0, 0)], "bathroom": [("toilet", -0.3, -1), ("basin", 0.05, -1), ("shower", 0.33, -1)],
              "bath": [("bathtub", 0, -1)], "wc": [("toilet", 0, -1)], "toilet": [("toilet", 0, -1)],
              "study": [("desk", 0, -1)], "office": [("desk", 0, -1)], "utility": [("sink", 0, -1)],
              "laundry": [("sink", 0, -1)]}


def _floorplan(text, low, title):
    m = re.search(r"(\d+(?:\.\d+)?)\s*(m|mm|cm)?\s*(?:x|×|by)\s*(\d+(?:\.\d+)?)\s*(m|mm|cm|metres|meters)\b", low)
    area = re.search(r"(\d+(?:\.\d+)?)\s*(?:m2|m²|sq\.?\s*m|square met(?:re|er)s)", low)
    rooms = []
    for label, n, raw in _items(_tail(text, ("with", "including", "of"))):
        key = next((k for k in _ROOM_W if re.search(r"\b" + k, label.lower())), None)
        if key:
            nm = re.sub(r"\s+room$", "", label.strip(), flags=re.I) if key in ("living", "dining") else label
            rooms += [(_cap(nm) + (f" {i + 1}" if n > 1 else ""), key) for i in range(n)]
    nb = re.search(r"\b(\d+|" + "|".join(_NUMW) + r")[- ]bed(?:room)?\b", low)
    if not rooms:
        k = (int(nb.group(1)) if nb.group(1).isdigit() else _NUMW[nb.group(1)]) if nb else 1
        rooms = [("Living", "living"), ("Kitchen", "kitchen")] + \
            [(f"Bedroom{f' {i + 1}' if k > 1 else ''}", "bedroom") for i in range(k)] + [("Bathroom", "bathroom")]
    tot = sum(_ROOM_W[k] for _n, k in rooms)
    if m:
        u = m.group(4)
        f = 1000 if u in ("m", "metres", "meters") else 10 if u == "cm" else 1
        W, D = float(m.group(1)) * f, float(m.group(3)) * f
    else:
        a = float(area.group(1)) if area else 9.0 * tot
        W, D = (a * 1.35) ** 0.5 * 1000, (a / 1.35) ** 0.5 * 1000
    W, D = round(W / 100) * 100, round(D / 100) * 100
    sc = re.search(r"\b1\s*:\s*(\d+)\b", low)
    scale = f"1:{sc.group(1)}" if sc else ("1:50" if W / 50 <= 230 and D / 50 <= 170 else "1:100")
    rows = [[], []]                                       # bottom row (entrance, living) and top row
    for nm, k in sorted(rooms, key=lambda r: (r[1] not in ("living", "lounge"), -_ROOM_W[r[1]])):
        s = [sum(_ROOM_W[x[1]] for x in r) for r in rows]
        rows[0 if (not rows[0] or (s[0] <= s[1] and rows[1])) else 1].append((nm, k))
    rows = [r for r in rows if r]
    sums = [sum(_ROOM_W[x[1]] for x in r) for r in rows]
    h, it = 125, 50
    mid = round(D * sums[0] / sum(sums) / 50) * 50 if len(rows) == 2 else D
    b = ["def build():", '    sh = Sheet("A3", scale=SCALE, title=TITLE, drawing_no=DRAWING_NO, style=STYLE)',
         '    p = sh.view("PLAN", label="FLOOR PLAN")              # placed in the free area of the sheet',
         "    h = EXT / 2",
         "    p.wall([(h, h), (WIDTH - h, h), (WIDTH - h, DEPTH - h), (h, DEPTH - h)], thickness=EXT, closed=True)"]
    if len(rows) == 2:
        b.append(f"    p.wall([(h, {mid}), (WIDTH - h, {mid})], thickness=INT)      # between the two rows of rooms")
    furn, labels, doors, wins = [], [], [], []
    living_x = None
    for ri, row in enumerate(rows):
        y0, y1 = (0, mid) if ri == 0 else (mid, D)
        x = 0.0
        for j, (nm, k) in enumerate(row):
            w = round(W * _ROOM_W[k] / sums[ri] / 50) * 50 if j < len(row) - 1 else W - x
            x0, x1 = x, x + w
            if j < len(row) - 1:
                b.append(f"    p.wall([({x1:g}, {y0 + h if ri == 0 else y0:g}), ({x1:g}, {y1 - h if ri else y1:g})], "
                         f"thickness=INT)")
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            outer = y0 + h if ri == 0 else y1 - h               # the room's outside wall
            inward = 1 if ri == 0 else -1
            if k in ("living", "lounge") and ri == 0:
                living_x = x0 + 0.72 * w
                doors.append(f"    p.door({living_x:g}, h, 1000, toward=({living_x:g}, {h + 1000:g}), hinge=\"left\")"
                             "      # entrance")
                wins.append(f"    p.window({x0 + 0.3 * w:g}, {outer:g}, {min(1800, 0.4 * w):g})")
            else:
                wins.append(f"    p.window({cx:g}, {outer:g}, {min(1500, 0.45 * w):g})")
                if len(rows) == 2:
                    dx = x0 + min(0.25 * w + 400, w - 500)
                    doors.append(f"    p.door({dx:g}, {mid}, 800, toward=({dx:g}, {mid + (600 if ri else -600)}))")
                elif j > 0:
                    doors.append(f"    p.door({x0:g}, {cy:g}, 800, toward=({x0 + 600:g}, {cy:g}))")
            for kind_, du, dv in _FURNITURE.get(k, []):
                fy = outer + inward * 700 if dv < 0 else cy + dv * (y1 - y0)
                ang = 0 if ri == 0 else 180
                furn.append(f'    p.furniture("{kind_}", {cx + du * w:g}, {fy:g}, angle={ang})')
            labels.append(f"    p.room({_py(nm.upper())}, {cx:g}, {cy + (y1 - y0) * 0.12 * (1 if ri == 0 else -1):g})")
            x = x1
    b += ["    # doors and windows find their wall from (x, y)"] + doors + wins
    b += ["    # furniture symbols: bed sofa table chair toilet basin shower bathtub counter sink fridge wardrobe desk"]
    b += furn + labels
    b += ["    p.dim_h((0, 0), (WIDTH, 0)); p.dim_v((0, 0), (0, DEPTH))",
          "    sh.north_arrow(380, 250); sh.scale_bar(300, 230, length=5); sh.room_schedule(p, x=296, y=200)",
          "    return sh"]
    name = _cap(_subject(text, r"plan|layout") or title).upper()
    data = [f"TITLE, DRAWING_NO = {_py(name[:40])}, \"A-100\"",
            f"WIDTH, DEPTH = {W:g}, {D:g}                   # outside size in mm (y up, origin = bottom-left corner)",
            "EXT, INT = 250, 100                         # wall thicknesses in mm", f'SCALE = "{scale}"',
            'STYLE = "white"                             # white | paper | blueprint']
    rep = [f"drawn    {len(rooms)} rooms: " + ", ".join(nm for nm, _k in rooms) +
           f"; {W / 1000:g} x {D / 1000:g} m = {W * D / 1e6:.1f} m²; scale {scale}"]
    return data, b, rep


_MOTIFS = [("synthwave", r"synthwave|retro|80s|outrun|neon"), ("starry", r"star|space|galaxy|night sky|cosmos"),
           ("aurora", r"aurora|northern lights"), ("mountains", r"mountain|peak|alps|summit"),
           ("dunes", r"dune|desert|sand"), ("ocean", r"ocean|sea\b|beach|sunset over"), ("forest", r"forest|woods|trees"),
           ("hills", r"\bhill|valley|meadow|countryside"), ("underwater", r"underwater|coral|reef"),
           ("skyline", r"city|skyline|urban"), ("clouds", r"cloud|sky\b"), ("waves", r"\bwaves?\b"),
           ("ribbons", r"ribbon|flow|silk"), ("low_poly", r"low[- ]poly|geometric|triangle"),
           ("topographic", r"topograph|contour"), ("marble", r"marble"), ("bokeh", r"bokeh"),
           ("fluid", r"fluid|liquid"), ("mesh", r"abstract|gradient|blur|minimal")]


def _wallpaper(text, low):
    sys.path.insert(0, ROOT)
    from pixelcanvas import catalog
    bd = next((n for n, rx in _MOTIFS if re.search(rx, low)), "mesh")
    extra = {"peach", "coral", "mint", "lavender", "rose", "sand", "plum", "olive", "salmon", "lilac", "sage", "cream"}
    cols = [w for w in re.findall(r"[a-z]+", low) if w in catalog._colors() or w in extra]
    mood = [w for w in re.findall(r"[a-z]+", low) if w in ("calm", "soft", "warm", "cool", "dark", "moody", "vibrant",
                                                            "pastel", "serene", "dreamy", "bold", "muted")]
    pal = " and ".join(dict.fromkeys(cols)) if cols else None
    if mood and pal:
        pal = f"{mood[0]} {pal}"
    pal = pal or (mood[0] if mood else {"starry": "night", "aurora": "aurora", "synthwave": "synthwave"}.get(bd, "sunset"))
    opts = {"mountains": "dict(mist=0.8 if MISTY else 0.55, sun_x=0.62)", "hills": "dict(layers=6, sun_x=0.62)",
            "ocean": "dict(sun_x=0.5)"}.get(bd, "dict()")
    data = [f'BACKDROP = "{bd}"                      # hills mountains dunes ocean synthwave starry forest | mesh aurora',
            f"PALETTE = {_py(pal)}                      # names, moods and colour words, or [\"#123\", ...]"]
    if "MISTY" in opts:
        data.append(f"MISTY = {bool(re.search(r'mist|fog|haz', low))}")
    data.append(f"OPTIONS = {opts}     # backdrop options; backdrop.describe(\"{bd}\") lists them")
    return data, [f"drawn    {bd} backdrop, palette {pal!r}"]


def _comic(text, low, prompt):
    quotes = _quotes(prompt)
    m = re.search(r"\b(\d+|" + "|".join(_NUMW) + r")[- ](?:panel|frame)s?\b|\b(\d+|" + "|".join(_NUMW) +
                  r")\s+(?:panels|frames)\b", low)
    n = None
    if m:
        g = m.group(1) or m.group(2)
        n = int(g) if g.isdigit() else _NUMW[g]
    n = max(1, min(n or max(len(quotes), 3), 6))
    lines = [(i % 2, quotes[i] if i < len(quotes) else "…") for i in range(n)]
    about = re.search(r"\babout\s+(.+?)(?=:|;|,\s*['\"“]|$)", _unquoted(prompt), re.I)
    cap = _cap(about.group(1).strip(" .,")) if about else (quotes[n] if len(quotes) > n else "")
    data = [*_assign("LINES", lines, "(speaker, text), one panel per line"), f"CAPTION = {_py(cap)}"]
    return data, [f"drawn    {n} panels, speech: " + "; ".join(t for _s, t in lines) +
                  (f"; caption {cap!r}" if cap else "")]


def _figure(text, low, title):
    items = _items(_tail(text, ("showing", "labelling", "labeling", "labelled", "labeled", "with", "including", "of")))
    parts = [_cap(lb) for lb, _n, _r in items if len(lb.split()) <= 4]
    import math
    rows = [(title, ("ellipse", 0.5, 0.55, 0.36, 0.3), "#f3ead8")]
    for i, p in enumerate(parts):
        a = -math.pi / 2 + 2 * math.pi * i / max(len(parts), 1)
        rows.append((p, ("ellipse" if i % 3 else "capsule", round(0.5 + 0.2 * math.cos(a), 3),
                         round(0.55 + 0.16 * math.sin(a), 3), 0.06, 0.045), _PASTELS[i % len(_PASTELS)]))
    data = ["# STAND-IN shapes: one block for the whole subject, one marker per named part; draw each real shape",
            f"TITLE = {_py(title)}",
            "PARTS = [  # label, (kind ellipse / capsule, centre u, v, half-width, half-height) of the square, fill"]
    data += [f"    {_py(r)}," for r in rows] + ["]"]
    return data, [f"drawn    callouts: " + (", ".join(parts) if parts else "none named") + f" around a {title} block"]


def _title_of(text, kind, quotes):
    if quotes and kind not in ("comic",):
        return quotes[0]
    heads = {"diagram": r"diagram|flowchart|flow chart|chart|timeline|roadmap|map|tree|graph",
             "chart": r"chart|graph|plot|histogram", "dashboard": r"dashboard",
             "infographic": r"infographic|poster|graphic", "figure": r"diagram|figure|illustration|drawing",
             "icons": r"icons?|set|sheet", "drawing": r"drawing", "floorplan": r"plan|layout"}.get(kind, r"\w+")
    s = _subject(text, heads)
    if kind == "dashboard" and not s:
        m = re.search(r"\b(?:an?\s+|the\s+)?(?:(?:dark|light|clean|minimal|modern|simple)\s+)*([a-z]+(?:\s+[a-z]+)?)\s+"
                      r"dashboard", text, re.I)
        s = (m.group(1) + " dashboard") if m else "Dashboard"
    if kind == "diagram" and s and re.search(r"\barchitecture\b", text, re.I) and "architecture" not in s.lower():
        s = s + " architecture"
    return _cap(re.sub(r"\s+", " ", s)) or {"icons": "Icons", "figure": "Figure"}.get(kind, "Overview")


def compose_format(src, kind, prompt, res):
    """A format template with the request's own data: (source, report lines)."""
    text = _unquoted(prompt).strip()
    low = text.lower()
    quotes = _quotes(prompt)
    title = _title_of(text, kind, quotes)
    rng = _seeded(prompt)
    data, build, rep = None, None, []
    if kind == "diagram":
        data, build, rep = _diagram(text, low, title, rng)
    elif kind == "chart":
        data, fmt, rep = _chart(text, low, title, rng)
        src = src.replace('yfmt="$", ', f"yfmt={_py(fmt)}, " if fmt else "", 1)
    elif kind == "dashboard":
        data, rep = _dashboard(text, low, title, rng)
        if re.search(r"\blight\b", low):
            src = _set_constant(src, "THEME", '"light"')
    elif kind == "infographic":
        data, rep = _infographic(text, low, title, rng)
    elif kind == "ui":
        data, build, rep = _ui(text, low, prompt, rng)
        return _write(src, data, build, "screen"), rep
    elif kind == "icons":
        data, build, rep = _icons(text, low, prompt, res, title)
        src = _write(src, data, build, "build")
        if build and "app_icon_sheet" in "\n".join(build):
            src = re.sub(r"^from pixelcanvas import Canvas, pictograms$", "from pixelcanvas import Canvas, pictograms, "
                                                                           "vector", src, count=1, flags=re.M)
        return src, rep
    elif kind == "drawing":
        data, build, rep = _drawing(text, low, title)
    elif kind == "floorplan":
        data, build, rep = _floorplan(text, low, title)
    elif kind == "wallpaper":
        data, rep = _wallpaper(text, low)
        src = _set_constant(src, "GLOW", "0.12")
    elif kind == "comic":
        data, rep = _comic(text, low, prompt)
        who = [g.head for g in res.groups if g.head and not getattr(g, "format", False) and g.results
               and g.results[0].kit in ("figure", "clay", "creature", "solid", "pictograms")
               and g.head not in ("comic", "panel", "strip")]
        if who:
            rep.append("cast     " + ", ".join(who) + ": the two figure.person characters speak; give them the "
                                                      "request's looks (or draw your own character)")
    elif kind == "figure":
        data, rep = _figure(text, low, title)
        if not re.search(r"\bcell|bacteri|microb|micro|virus|tissue\b", low):
            src = re.sub(r"^[ \t]*ann\.scale_bar\([^\n]*\n", "", src, count=1, flags=re.M)
    if data is None:
        return src, []
    return _write(src, data, build, "build"), rep


def _write(src, data, build, block):
    src, _ok = _replace_block(src, "data", data)
    if build:
        src, _ok = _replace_block(src, block, build)
    return src


def parse_size(text):
    m = re.fullmatch(r"\s*(\d+)\s*[xX*,]\s*(\d+)\s*", text or "")
    if not m or int(m.group(1)) < 16 or int(m.group(2)) < 16:
        raise ValueError(f"--size expects WxH such as 1080x1350, got {text!r}")
    return int(m.group(1)), int(m.group(2))


def print_kinds(table, out=sys.stdout):
    width = max(len(k) for k in table)
    for k, p in table.items():
        print(f"  {k:<{width}}  {first_doc_line(p)}", file=out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?", help="lower-case letters, digits and _ (e.g. night_lake)")
    ap.add_argument("--kind", default=None, help=f"template kind (see --list; default {DEFAULT_KIND})")
    ap.add_argument("--poster", action="store_true", help="same as --kind poster")
    ap.add_argument("--list", action="store_true", help="list every kind with its description")
    ap.add_argument("--size", default=None, metavar="WxH", help="set DEFAULT_SIZE, e.g. 1080x1920")
    ap.add_argument("--prompt", default=None, metavar="TEXT",
                    help="the request: stored in PROMPT; it picks the kind when --kind is not given, sets the size "
                         "and text, and fills a picture's plan (subjects to draw) or a format's data")
    ap.add_argument("--plain", action="store_true", help="with --prompt: keep the template's own example content")
    ap.add_argument("--quiet", action="store_true", help="print only the short summary, not the file it wrote")
    ap.add_argument("--dir", default=None, metavar="DIR",
                    help="folder for the new scene (created if needed; default: $PIXELCANVAS_WORK, scenes/ inside "
                         "the checkout, else the current folder)")
    a = ap.parse_args(argv)
    table = kinds()
    if a.list:
        print(f"kinds ({len(table)}):  python tools/new_scene.py NAME --kind KIND")
        print_kinds(table)
        return 0
    found = None
    if a.prompt and not a.kind and not a.poster:          # the request names the kind
        sys.path.insert(0, ROOT)
        from pixelcanvas import catalog
        found = catalog.find(a.prompt)
        a.kind = found.kind or "blank"
    kind = (a.kind or ("poster" if a.poster else DEFAULT_KIND)).strip().lower().replace("-", "_")
    if kind not in table and kind in KIND_FALLBACK and KIND_FALLBACK[kind] in table:
        kind = KIND_FALLBACK[kind]                        # a banner before its template exists: the card
    if kind not in table:
        close = [k for k in table if kind and (kind in k or k in kind)]
        close += [k for k in difflib.get_close_matches(kind, list(table), n=3, cutoff=0.5) if k not in close]
        print(f"unknown kind {kind!r}." + (f" Did you mean: {', '.join(close[:3])}?" if close else ""), file=sys.stderr)
        print(f"kinds: {', '.join(table)}  (python tools/new_scene.py --list describes each)", file=sys.stderr)
        return 1
    if not a.name or not re.fullmatch(r"[a-z][a-z0-9_]*", a.name):
        print(__doc__)
        print("name: lower-case letters, digits and _ (e.g. night_lake)")
        return 1
    report = []
    try:
        size = parse_size(a.size) if a.size else (request_size(a.prompt) if a.prompt else None)
        src = fill(_source(table[kind]), a.name, size, a.prompt)
        if a.prompt and not a.plain:
            src, report = compose(src, kind, a.prompt, found)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    folder = os.path.abspath(os.path.expanduser(a.dir)) if a.dir else default_dir()
    dst = os.path.join(folder, f"{a.name}.py")
    shown = _rel(dst)
    if os.path.exists(dst):
        print(f"already exists: {shown} (pick another name or edit that file)", file=sys.stderr)
        return 1
    os.makedirs(folder, exist_ok=True)
    src = ascii_source(src, a.prompt or "")
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    in_root = os.path.normcase(os.path.abspath(os.getcwd())) == os.path.normcase(ROOT)
    if in_root and os.path.normcase(folder) == os.path.normcase(os.path.join(ROOT, "scenes")):
        target, render = a.name, "python tools/render.py"
    else:
        target, render = _rel(dst), f"python {_rel(os.path.join(ROOT, 'tools', 'render.py'))}"
        if " " in render:
            render = f'python "{_rel(os.path.join(ROOT, "tools", "render.py"))}"'
        if os.environ.get("PIXELCANVAS_CMD"):         # started through `pixelcanvas new` / tools/pc.py
            render = os.environ["PIXELCANVAS_CMD"] + " render"
    print(f"created: {shown}  (kind: {kind}" + (f", {size[0]}x{size[1]}" if size else "") + ")")
    for ln in report:
        print(f"  {ln}")
    ref = reference_line(src)[5:]
    print(f"read:    {ref if in_root else ref.replace('skills/', ROOT.replace(os.sep, '/') + '/skills/', 1)}")
    if not a.quiet:                                  # the file itself: no second call to read it
        print("\n".join(listing(src, shown)))
    print("edit:    the plan docstring, PROMPT and the UPPER_CASE tunables first, then the steps in render()")
    print(f"next:    {render} {target} --scale 0.5 --check")
    return 0


def ascii_source(src, prompt=""):
    """The starter as pure ASCII: a character the request does not carry is written as a \\uXXXX escape (inside a
    string it draws the same; in a comment it reads as its code), so a Windows console or a cp950 read never
    breaks on it. The request's own text (a CJK title) stays as real characters."""
    keep = set(prompt or "")
    return "".join(c if ord(c) < 128 or c in keep else
                   ("\\u%04x" % ord(c) if ord(c) < 0x10000 else "\\U%08x" % ord(c)) for c in src)


LISTING_MAX = 120


def listing(src, shown):
    """The written file with line numbers: all of it up to LISTING_MAX lines; else the plan docstring, the
    UPPER_CASE tunables and the marked blocks (data, subjects, setting ...), with "..." between the parts."""
    lines = src.rstrip("\n").split("\n")
    keep = set(range(len(lines)))
    if len(lines) > LISTING_MAX:
        keep = set()
        doc = [i for i, ln in enumerate(lines) if '"""' in ln][:2]
        if len(doc) == 2:
            keep |= set(range(doc[0], doc[1] + 1))
        end = next((i for i, ln in enumerate(lines) if ln.startswith("def ")), len(lines))
        for i in range(end):                              # UPPER_CASE tunables, with their continuation lines
            if re.match(r"[A-Z][A-Z0-9_]*(?:, [A-Z][A-Z0-9_]*)* = ", lines[i]) or \
                    (i - 1 in keep and lines[i].startswith((" ", ")", "]", "}"))):
                keep.add(i)
        opened = None
        for i, ln in enumerate(lines):
            m = re.match(r"\s*# (>>>|<<<) (\w+)", ln)
            if m and m.group(1) == ">>>":
                opened = i
            if opened is not None:
                keep.add(i)
            if m and m.group(1) == "<<<":
                opened = None
    out = [f"--- {shown} ({len(lines)} lines" + (")" if len(keep) == len(lines) else
                                                 ": plan, tunables and the request's blocks)") + " ---"]
    last = -1
    for i in sorted(keep):
        if i != last + 1:
            out.append("     ...")
        out.append(f"{i + 1:4d}  {lines[i]}")
        last = i
    if last != len(lines) - 1:
        out.append("     ...")
    out.append("--- end ---")
    return out


def _rel(p):
    try:
        r = os.path.relpath(p)
    except ValueError:                                # another drive on Windows
        return p.replace("\\", "/")
    return (p if r.startswith("..") else r).replace("\\", "/")


def default_dir():
    """$PIXELCANVAS_WORK, scenes/ when working inside the pixelcanvas checkout, else ./art in the current folder
    (the current folder itself when started as `pixelcanvas new`)."""
    sys.path.insert(0, ROOT)
    try:
        from pixelcanvas.cli import default_scene_dir
    except ImportError:                               # package not importable: plain rules
        work = os.environ.get("PIXELCANVAS_WORK")
        folder = os.path.abspath(os.path.expanduser(work)) if work else os.getcwd()
    else:
        folder = default_scene_dir(ROOT)
    here = os.path.normcase(os.path.abspath(os.getcwd()))
    if os.path.normcase(folder) == here and not os.environ.get("PIXELCANVAS_WORK") and \
            not os.environ.get("PIXELCANVAS_CMD"):
        return os.path.join(os.getcwd(), "art")       # a project folder: scenes go to ./art
    return folder


if __name__ == "__main__":
    sys.exit(main())
