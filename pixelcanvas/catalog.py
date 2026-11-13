# -*- coding: utf-8 -*-
"""catalog: one searchable index of everything the kits can draw.

Every kit keeps its own registry (``flora.names()``, ``stylize.STYLES``, ``texture.TEXTURES``, ``vector.ICONS``,
``ui.TEMPLATES`` ...). This module reads them all into one list of entries, each with the exact call to paste, and
answers plain requests:

>>> from pixelcanvas import catalog
>>> res = catalog.find("a corgi sitting under an oak tree at sunset")
>>> print(res.text())                       # best calls per phrase, a scene kind, a style
>>> catalog.entries(kit="flora", kind="plant")[:3]
>>> print(catalog.show("flora.oak"))        # call, signature, docstring, reference page
>>> catalog.suggest_kind("an engineering drawing of a flange")     # -> "drawing"

An entry is ``Entry(kit, name, kind, call, summary, keywords, ref)``; ``entry.snippet()`` is a complete program
(imports + canvas + call) that runs as written. Kits are discovered from the package folder: a kit that is
missing or fails to import is skipped, a new kit with a ``names()`` registry or documented functions joins on its
own. Nothing is rendered while the index is built; it is built once per process.
"""
from __future__ import annotations

import difflib
import glob
import importlib
import inspect
import json
import math
import os
import re
import unicodedata

__all__ = ["Entry", "Found", "Group", "entries", "find", "show", "suggest_kind", "kits", "kinds", "doc",
           "reference", "build", "reset", "extract_content", "recipe", "kit_kinds"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "pixelcanvas")
# scenes/ and skills/ sit next to the package in a checkout and under pixelcanvas/_runtime in an installed wheel
RUNTIME = ROOT if os.path.isdir(os.path.join(ROOT, "scenes")) else os.path.join(PKG, "_runtime")
REFS = os.path.join(RUNTIME, "skills", "pixelcanvas", "references")

# modules that are building blocks, not kits
_CORE = {"__init__", "canvas", "color", "noise", "masks", "brush", "filters", "sdf", "vision", "layers", "manifest",
         "cli", "catalog"}
# order = tie-break between kits (subject kits first)
_KIT_ORDER = ("creature", "figure", "pictograms", "clay", "flora", "elements", "solid", "form", "texture", "backdrop", "stylize",
              "diagram", "chart", "drafting", "ui", "layout", "text", "vector", "palette", "scenery", "scenes")
_UNFINISHED = ("form", "figure", "creature", "clay")       # kits still being written (tests skip their calls)
_REF_OVERRIDE = {"stylize": "styles.md", "text": "typography.md", "layout": "typography.md",
                 "palette": "backdrops.md", "backdrop": "backdrops.md", "scenery": "api.md",
                 "scenes": "../SKILL.md"}


# --------------------------------------------------------------------------------------------- entries
class Entry:
    """One drawable thing: ``call`` is the line to paste, ``snippet()`` a complete program around it."""

    __slots__ = ("kit", "name", "kind", "call", "summary", "keywords", "ref", "aliases", "code", "func", "lang",
                 "poses", "heavy", "order", "_toks", "_alias_toks", "_kw", "_doc")

    def __init__(self, kit, name, kind, call, summary="", keywords=(), ref=None, aliases=(), code=None, func=None,
                 lang="python", poses=(), heavy=False):
        self.kit, self.name, self.kind, self.call = kit, str(name), kind, call
        self.summary = (summary or "").strip().split("\n")[0][:160]
        self.keywords = tuple(dict.fromkeys(k for k in keywords if k))
        self.ref = _ref_page(kit) if ref is None else ref
        self.aliases = tuple(dict.fromkeys(a for a in aliases if a and a != name))
        self.code = code if code is not None else call
        self.func = func
        self.lang = lang
        self.poses = tuple(poses)
        self.heavy = heavy
        self.order = 0

    @property
    def id(self):
        """``kit.name`` with spaces as underscores (what ``show`` takes)."""
        return f"{self.kit}.{self.name.replace(' ', '_')}"

    def snippet(self, width=960, height=540, call=None):
        """A complete program: imports, a Canvas ``cv`` of the given size and the call (shell entries: the command)."""
        if self.lang != "python":
            return call or self.code
        code = self.code if call is None else self.code.replace(self.call, call)
        kits = list(dict.fromkeys(_ALL_KITS + [k for k in kit_names() if k.isidentifier()]))   # new kits too
        mods = sorted(set(re.findall(r"\b(" + "|".join(kits) + r")\.", code)) - {"cv", "sc", "s", "d"})
        imp = "from pixelcanvas import " + ", ".join(["Canvas"] + mods)
        return f"{imp}\ncv = Canvas({width}, {height})\n{code}"

    def as_dict(self):
        return {"id": self.id, "kit": self.kit, "name": self.name, "kind": self.kind, "call": self.call,
                "summary": self.summary, "keywords": list(self.keywords), "aliases": list(self.aliases),
                "ref": self.ref, "snippet": self.snippet()}

    def __repr__(self):
        return f"<Entry {self.id} ({self.kind}): {self.call}>"


_ALL_KITS = list(_KIT_ORDER) + ["sdf", "noise", "filters", "masks"]


def _first_line(obj):
    doc = inspect.getdoc(obj) or ""
    line = doc.strip().split("\n")[0].strip()
    line = re.sub(r":\w+:`([^`]*)`", r"\1", line)
    return re.sub(r"``([^`]*)``", r"\1", line)


def _args(fn, keep=("kind", "at", "size", "region", "box", "height", "horizon", "v_range"), skip=1):
    """``name=default`` for the placement parameters of ``fn`` that have defaults (a readable starting call)."""
    try:
        params = list(inspect.signature(fn).parameters.values())[skip:]
    except (TypeError, ValueError):
        return ""
    out = []
    for p in params:
        if p.name in keep and p.default is not p.empty and p.default is not None:
            out.append(f"{p.name}={p.default!r}")
    return ", ".join(out)


def _needs(fn, skip=1):
    """Required positional parameters of ``fn`` after the first ``skip``."""
    try:
        params = list(inspect.signature(fn).parameters.values())[skip:]
    except (TypeError, ValueError):
        return ["?"]
    return [p.name for p in params if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]


def _words(text):
    return [w for w in re.findall(r"[a-z][a-z0-9]+", (text or "").lower().replace("_", " ")) if w not in _STOP]


# --------------------------------------------------------------------------------------------- kit handlers
def _kit_flora(m):
    out = []
    reg = getattr(m, "_REG", {})
    aliases = {}
    for a, t in getattr(m, "_ALIASES", {}).items():
        aliases.setdefault(t, []).append(a)
    heights = {"tree": 0.6, "shrub": 0.3, "ground": 0.25, "flower": 0.35, "potted": 0.4, "leaf": 0.3, "rock": 0.2}
    fns = {"tree": "tree", "shrub": "bush", "ground": "grass", "flower": "flower", "potted": "potted", "leaf": "leaf",
           "rock": "rock"}
    finder = getattr(m, "find", None)
    for word in ("tree", "flower", "plant", "bush", "shrub", "rock", "leaf", "grass", "houseplant", "potted plant"):
        try:
            hit = finder(word) if finder else None
        except Exception:
            hit = None
        hit = hit or {"tree": "oak", "flower": "daisy", "plant": "potted monstera"}.get(word)
        if hit in reg or hit in getattr(m, "_ALIASES", {}).values():
            aliases.setdefault(hit, []).append(word)
    for name in m.names():
        sub = reg.get(name, (None, "plant"))[1]
        at = "(0.5, 0.5)" if sub == "leaf" else "(0.5, 0.9)"
        call = f'flora.plant(cv, "{name}", at={at}, height={heights.get(sub, 0.4)})'
        fn = getattr(m, fns.get(sub, "plant"), None)
        summary = f"{sub}: {name}" + (f" - {_first_line(fn)}" if fn else "")
        kw = [sub, "plant"] + (["tree"] if sub == "tree" else []) + (["flower"] if sub == "flower" else [])
        if sub == "potted":
            kw += ["houseplant", "pot", "indoor"]
        out.append(Entry("flora", name, "rock" if sub == "rock" else "plant", call, summary, kw,
                         aliases=aliases.get(name, ()), func="flora.plant"))
    have = {e.name for e in out}
    skip = {"plant", "names", "find", "cellular", "strokes", "tree", "flower", "potted", "leaf", "rock", "bush",
            "grass", "palm", "bamboo", "saguaro", "banana", "hedge", "topiary", "reeds", "cattails", "fern",
            "mushroom", "pumpkin", "prickly_pear", "lily_pads", "rock_pile", "vase", "bouquet"}
    out += _fn_entries(m, "flora", "plant", skip=skip | have)
    return out


def _kit_elements(m):
    out = []
    reg = getattr(m, "REGISTRY", {})
    fix = {"wet ground": 'elements.wet_ground(cv, mask=(cv.uv()[1] > 0.6).astype("float32"))',
           "snow cap": 'elements.snow_cap(cv, mask=(cv.uv()[1] > 0.6).astype("float32"))',
           "neon": 'elements.neon(cv, text="OPEN", at=(0.5, 0.5), size=140)',
           "volumetric fog": "elements.volumetric_fog(cv, elements.lamp(cv, at=(0.5, 0.3)).lights)"}
    for name in m.names():
        words = reg.get(name, (None, ""))[1]
        fname = name.replace(" ", "_")
        fn = getattr(m, fname, None)
        if name in fix:
            call = fix[name]
        elif fn is not None and not _needs(fn) and inspect.isfunction(fn):
            a = _args(fn)
            call = f"elements.{fname}(cv{', ' + a if a else ''})"
        else:
            call = f'elements.element(cv, "{name}")'
            fn = getattr(m, "element", None)
        out.append(Entry("elements", name, "effect", call, _first_line(fn) if fn else words,
                         words.split(), func=f"elements.{fname}" if hasattr(m, fname) else "elements.element"))
    for sky in getattr(m, "SKIES", {}):
        out.append(Entry("elements", sky, "sky", f'elements.sky(cv, "{sky}")', f"{sky} sky gradient",
                         ["sky", sky, "gradient", "background", "light", "time"], func="elements.sky"))
    return out


def _kit_texture(m):
    out = []
    meta = getattr(m, "_META", {})
    for name in m.names():
        fam, words = meta.get(name, ("", ""))
        try:
            summary = m.describe(name).split("\n")[0]
        except Exception:
            summary = name
        out.append(Entry("texture", name, "material", f'cv.paint(texture.texture("{name}", cv.h, cv.w, seed=1))',
                         summary, [fam, "texture", "material", "surface", "pattern"] + words.split(),
                         code=f'cv.paint(texture.texture("{name}", cv.h, cv.w, seed=1))\n'
                              f'# into a shape: texture.fill(cv, mask, "{name}", shadow=0.5)',
                         func="texture.fill"))
    return out


def _kit_stylize(m):
    out = []
    styles = getattr(m, "STYLES", {})
    for name in m.names():
        info = styles.get(name, {})
        kw = list(info.get("keywords", ())) + ["style", "medium", "look"]
        out.append(Entry("stylize", name, "style", f'stylize.apply(cv, "{name}")', info.get("desc", ""), kw,
                         code=f'cv.gradient([(0, "#3a6ea5"), (1, "#f2c48d")])        # ... your picture ...\n'
                              f'stylize.apply(cv, "{name}")                          # last step: log("medium")',
                         func="stylize.apply"))
    return out


def _kit_palette(m):
    out = []
    fam = {n: f for f, ns in getattr(m, "FAMILIES", {}).items() for n in ns}
    zh = {}
    for k, v in getattr(m, "ZH_KEYWORDS", {}).items():
        for w in v.split():
            zh.setdefault(w, []).append(k)
    for name in m.names():
        try:
            d = m.describe(name)
        except Exception:
            d = ""
        kw = re.findall(r"\[([^\]]*)\]", d)
        kw = kw[0].split() if kw else []
        extra = [z for w in [name] + kw for z in zh.get(w, ())]
        mode = "dark" if " dark " in d else "light"
        out.append(Entry("palette", name, "palette", f'palette.palette("{name}")',
                         f"{fam.get(name, '')} palette ({mode}): {' '.join(kw[:8])}".strip(),
                         kw + extra + ["palette", "colors", "color scheme", fam.get(name, "")],
                         code=f'p = palette.palette("{name}")\ncv.gradient(p.stops())',
                         func="palette.palette"))
    return out


def _kit_backdrop(m):
    out = []
    reg = getattr(m, "BACKDROPS", {})
    for name in m.names():
        info = reg.get(name, {})
        kw = list(info.get("words", ())) + [info.get("family", ""), "backdrop", "background", "wallpaper"]
        out.append(Entry("backdrop", name, "backdrop", f'backdrop.backdrop(cv, "{name}")', info.get("desc", ""),
                         kw, func="backdrop.backdrop"))
    return out


_UI_TEMPLATE_WORDS = {
    "login": "sign in signin log auth password form account", "dashboard": "admin analytics metrics kpi panel",
    "settings": "preferences options toggles", "chat": "messages messenger conversation dm",
    "feed": "social timeline posts", "profile": "user account bio", "checkout": "cart payment shop order",
    "onboarding": "welcome intro signup", "landing": "landing page website homepage hero marketing",
    "pricing": "plans tiers subscription", "music_player": "music player audio song podcast",
    "weather": "forecast temperature", "calendar": "schedule agenda events", "kanban": "board tasks trello"}
_UI_COMPONENTS = {        # method: (arguments, extra keywords, device)
    "button": ('"Save"', "button cta action", "phone"),
    "icon_button": ('"bell"', "icon button", "phone"),
    "fab": ('"plus"', "floating action button", "phone"),
    "input": ('"Email", placeholder="you@example.com"', "text field input form", "phone"),
    "search": ("", "search bar field", "phone"),
    "select": ('"Country", value="Japan"', "dropdown select picker", "phone"),
    "checkbox": ('"Remember me", checked=True', "checkbox tick", "phone"),
    "radio": ('"Monthly", selected=True', "radio option", "phone"),
    "toggle": ('"Notifications", on=True', "switch toggle setting", "phone"),
    "slider": ('0.6, label="Volume"', "slider range", "phone"),
    "stepper": ('2, label="Guests"', "stepper counter quantity", "phone"),
    "password": ("", "password field", "phone"),
    "text_area": ('"Message"', "textarea multiline", "phone"),
    "date_field": ('"Date"', "date picker field", "phone"),
    "app_bar": ('"Inbox", leading="menu", trailing=["search"]', "top bar header navigation", "phone"),
    "tab_bar": ('[("home", "Home"), ("search", "Search"), ("user", "Profile")]', "bottom navigation tabs", "phone"),
    "sidebar": ('[("home", "Home"), ("chart", "Reports"), ("gear", "Settings")]', "side navigation menu", "browser"),
    "navbar": ('"Brand", links=["Product", "Pricing", "Docs"]', "website navigation header", "browser"),
    "tabs": ('["Overview", "Activity", "Settings"]', "tabs", "phone"),
    "segmented": ('["Day", "Week", "Month"]', "segmented control", "phone"),
    "breadcrumb": ('["Home", "Projects", "Atlas"]', "breadcrumb path", "browser"),
    "pagination": ("page=2, pages=9", "pagination pages", "browser"),
    "list_row": ('"Coffee Lab", "Today, 8:12", leading="cart", value="-$4.50", chevron=True', "list item row",
                 "phone"),
    "table": ('["Name", "Status"], [["Atlas", "Live"], ["Orion", "Draft"]]', "data table grid rows", "browser"),
    "stat": ('"Revenue", "$12.4k", delta=8.2', "stat kpi metric tile number", "phone"),
    "badge": ('"New"', "badge label pill", "phone"),
    "chip": ('"Design", selected=True', "chip tag filter", "phone"),
    "avatar": ('"Ana Silva"', "avatar profile picture user", "phone"),
    "avatar_group": ('["Ana", "Ben", "Chen", "Dee"]', "avatars team members", "phone"),
    "progress": ('0.6, label="Upload"', "progress bar loading", "phone"),
    "ring": ('0.72, label="Done"', "progress ring circular timer", "phone"),
    "skeleton": ("3, avatar=True", "loading placeholder skeleton", "phone"),
    "chart": ('"bar", {"Mon": 3, "Tue": 5, "Wed": 4}', "chart graph in app", "phone"),
    "banner": ('"Update available", "Restart to install."', "alert banner notice", "phone"),
    "toast": ('"Saved"', "toast snackbar", "phone"),
    "tooltip": ('"Copy link"', "tooltip hint", "phone"),
    "menu": ('[("edit", "Edit", "E"), ("trash", "Delete", "Del")]', "dropdown context menu", "browser"),
    "message": ('"See you at 8?", me=True', "chat bubble message", "phone"),
    "notification": ('"New comment", "Ana replied to you", time="2m"', "notification alert", "phone"),
    "calendar": ("", "calendar month date", "phone"),
    "kanban": ('{"To do": ["Design", "Copy"], "Done": ["Brief"]}', "kanban board tasks", "browser"),
    "timeline": ('[("9:00", "Kickoff", "Team call", "users")]', "timeline activity feed", "phone"),
    "hero": ('"Think in the margins", "Notes that stay out of your way"', "hero headline landing", "browser"),
    "logo_row": ('["Acme", "Globex", "Initech"]', "logos customers", "browser"),
    "feature_grid": ('[("bolt", "Fast", "Renders in seconds"), ("lock", "Private", "Runs locally")]',
                     "features grid", "browser"),
    "pricing": ('[{"name": "Free", "price": "$0", "features": ["1 project"]}, '
                '{"name": "Pro", "price": "$9", "features": ["Unlimited"]}]', "pricing plans tiers", "browser"),
    "testimonial": ('"It changed how we work.", "Mina Okafor", role="CTO"', "testimonial quote review", "browser"),
    "footer": ('"Brand"', "footer site links", "browser"),
    "image": ('height=120, scene="ocean"', "image photo placeholder", "phone"),
    "map": ("height=120", "map location", "phone"),
    "video": ('height=120, title="Intro"', "video player", "phone"),
    "waveform": ("", "waveform audio", "phone"),
    "code": ("'print(\"hello\")'", "code block snippet syntax", "browser"),
    "terminal": ('["$ npm install", "added 42 packages"]', "terminal console shell", "browser"),
    "empty_state": ("", "empty state", "phone"),
    "heading": ('"Good evening, Sam", level=1', "heading title", "phone"),
    "text": ('"Body copy that wraps inside its box."', "paragraph text", "phone"),
    "divider": ("", "divider separator line", "phone"),
    "icon_badge": ('"bolt"', "icon badge", "phone"),
    "modal": ('"Delete file?", "This cannot be undone.", actions=["Cancel", "Delete"]', "modal dialog popup",
              "phone"),
    "sheet": ('"Share"', "bottom sheet", "phone"),
    "card": ("", "card surface panel", "phone"),
}
_UI_CONTEXT = {"modal", "sheet", "card"}
_UI_NAMES = {"card": "card container", "banner": "alert banner", "menu": "dropdown menu", "sheet": "bottom sheet",
             "table": "data table", "text": "body text", "chart": "app chart", "image": "image placeholder",
             "map": "map placeholder", "code": "code block", "timeline": "activity timeline", "badge": "status badge",
             "calendar": "calendar month", "video": "video player", "heading": "screen heading"}


def _kit_ui(m):
    out = []
    tpl = getattr(m, "TEMPLATES", {})
    for name in tpl:
        dev = "browser" if name in ("dashboard", "landing", "pricing", "kanban") else "phone"
        out.append(Entry("ui", name, "ui template", f'ui.template("{name}", screen="{dev}").draw(cv)',
                         f"finished {name.replace('_', ' ')} screen mock-up",
                         ["ui", "app", "screen", "mockup", "template"] + _UI_TEMPLATE_WORDS.get(name, "").split(),
                         func="ui.template"))
    for dev in getattr(m, "DEVICES", {}):
        out.append(Entry("ui", dev, "ui device", f'ui.Screen("{dev}")', f"{dev.replace('_', ' ')} frame for a screen",
                         ["device", "frame", "screen", "mockup"],
                         code=f's = ui.Screen("{dev}")\ns.heading("Hello")\ns.draw(cv)', func="ui.Screen"))
    screen = getattr(m, "Screen", None)
    if screen is None:
        return out
    for meth, (args, words, dev) in _UI_COMPONENTS.items():
        fn = getattr(screen, meth, None)
        if fn is None:
            continue
        call = f"s.{meth}({args})"
        body = f"with {call}:\n    s.text(\"...\")" if meth in _UI_CONTEXT else call
        summary = _first_line(fn) or f"{meth.replace('_', ' ')} component"
        out.append(Entry("ui", _UI_NAMES.get(meth, meth), "ui component", call, summary,
                         words.split() + ["ui", "component"],
                         code=f's = ui.Screen("{dev}")\n{body}\ns.draw(cv)', func=f"ui.Screen.{meth}"))
    return out


_ICON_WORDS = {"gear": "settings cog", "cart": "shopping basket", "bolt": "lightning power energy fast",
               "bulb": "idea light", "pin": "location map marker", "trend": "growth up", "user": "person account",
               "users": "people team group", "home": "house", "mail": "email envelope letter",
               "heart": "love like favourite", "star": "favorite rating", "lock": "security private",
               "shield": "security safe protection", "chip": "cpu processor", "monitor": "screen desktop computer",
               "phone": "mobile smartphone", "bot": "robot ai", "cloud": "weather sky", "globe": "world web earth"}


def _kit_vector(m):
    out = []
    if hasattr(m, "icon"):
        out.append(Entry("vector", "icon", "icon",
                         'vector.icon("star", color="#3f74c0").draw(cv, box=(130, 60, 190, 120))',
                         f"{len(getattr(m, 'ICONS', {}))} line icons in one style (names: vector.icon_names())",
                         ["icon", "symbol", "pictogram", "app icon", "glyph"], aliases=("icon set", "app icon"),
                         func="vector.icon"))
    for name in getattr(m, "ICONS", {}):
        out.append(Entry("vector", name, "icon",
                         f'vector.icon("{name}", color="#3f74c0").draw(cv, box=(130, 60, 190, 120))',
                         f"{name.replace('_', ' ')} line icon (vector, SVG-ready)",
                         ["icon", "symbol", "glyph", "pictogram"] + _ICON_WORDS.get(name, "").split(),
                         func="vector.icon"))
    for name in getattr(m, "THEMES", {}):
        out.append(Entry("vector", name, "theme", f'vector.Drawing(cv.w, cv.h, theme="{name}")',
                         f"{name} theme for drawings, diagrams and charts", ["theme", name],
                         code=f'dwg = vector.Drawing(cv.w, cv.h, theme="{name}")\n'
                              f'dwg.text(cv.w / 2, cv.h / 2, "{name}", size=40, anchor="mm")\ndwg.draw(cv)',
                         func="vector.Drawing"))
    shapes = [
        ("arrow", 'dwg.line(40, 90, 280, 90, stroke_width=4, arrow="end")', "arrow line connector pointer"),
        ("dashed line", 'dwg.line(40, 90, 280, 90, stroke_width=3, dash="dash")', "dashed dotted line"),
        ("rounded rectangle", 'dwg.rect(60, 40, 200, 100, r=16, fill="#3f74c0")', "rounded rectangle box card"),
        ("circle", 'dwg.circle(160, 90, 60, fill="#e5484d")', "circle dot disc"),
        ("bezier curve", 'dwg.spline([(30, 150), (110, 40), (210, 140), (290, 30)], stroke_width=4)',
         "curve spline bezier path"),
        ("svg path", 'dwg.path("M40 140 C 100 20, 220 20, 280 140", stroke_width=4)', "svg path"),
        ("gradient fill", 'dwg.rect(40, 30, 240, 120, r=12, fill=vector.linear_gradient([(0, "#ff7a59"), '
                          '(1, "#6a5cff")]))', "gradient fill"),
        ("hatch fill", 'dwg.rect(60, 40, 200, 100, fill="#ffffff", hatch=vector.hatch(45, spacing=8), stroke="#222")',
         "hatch hatching pattern fill"),
        ("pie slice", 'dwg.pie(160, 90, 70, 0, 250, inner=35, fill="#3f74c0")', "pie donut slice arc"),
    ]
    for name, call, words in shapes:
        out.append(Entry("vector", name, "vector shape", call, f"{name} drawn with vector.Drawing (also SVG)",
                         words.split() + ["vector", "svg", "shape"],
                         code=f"dwg = vector.Drawing(cv.w, cv.h)\n{call}\ndwg.draw(cv)",
                         func=f"vector.Drawing.{call.split('(')[0].split('.')[-1]}"))
    return out


_DIAGRAMS = [
    ("flowchart", "diagram", 'diagram.Diagram(title="Sign-up")',
     'd = diagram.Diagram(title="Sign-up")\nd.node("start", "Start", shape="pill")\n'
     'd.node("ok", "Email valid?", shape="diamond")\nd.node("done", "Account created", shape="pill", color="green")\n'
     'd.chain("start", "ok")\nd.edge("ok", "done", label="yes")\nd.draw(cv)',
     "flowchart flow chart process decision workflow steps algorithm", "Flowchart: nodes, decisions, routed edges"),
    ("architecture diagram", "diagram", 'diagram.Diagram(theme="dark", title="Platform")',
     'd = diagram.Diagram(theme="dark", title="Platform")\nwith d.group("svc", "Services"):\n'
     '    d.node("api", "API", icon="server")\nd.node("web", "Web app", icon="monitor")\n'
     'd.node("db", "Postgres", shape="cylinder", icon="database")\nd.chain("web", "api", "db")\nd.draw(cv)',
     "architecture system infrastructure services cloud tier network microservice backend",
     "Architecture diagram: grouped tiers, icons, sync / async edges"),
    ("org chart", "diagram", 'diagram.Tree({"Ada | CEO": {"Grace | CTO": ["Web", "Data"]}}, kind="org")',
     'diagram.Tree({"Ada | CEO": {"Grace | CTO": ["Web", "Data"], "Alan | CFO": ["Finance"]}}, kind="org").draw(cv)',
     "org organisation organization chart hierarchy team structure", "Org chart from a nested dict"),
    ("mind map", "diagram", 'diagram.Tree({"Launch": {"Product": ["Pricing"], "Marketing": ["Blog"]}}, '
                            'kind="mindmap")',
     'diagram.Tree({"Launch": {"Product": ["Pricing", "Docs"], "Marketing": ["Blog"]}}, kind="mindmap").draw(cv)',
     "mind map mindmap brainstorm ideas branches", "Mind map from a nested dict"),
    ("sequence diagram", "diagram", 'diagram.Sequence(autonumber=True)',
     'seq = diagram.Sequence(autonumber=True)\nseq.message("app", "api", "POST /login", activate=True)\n'
     'seq.reply("api", "app", "token")\nseq.draw(cv)',
     "sequence uml messages protocol oauth login request response lifeline",
     "Sequence diagram: participants, numbered messages, notes, frames"),
    ("roadmap", "diagram", 'diagram.Timeline(title="Roadmap")',
     'road = diagram.Timeline(title="Roadmap")\nroad.phase("2026-01", "2026-04", "Build")\n'
     'road.milestone("2026-05-01", "Launch", icon="star")\nroad.draw(cv)',
     "roadmap timeline milestones gantt schedule phases history plan", "Timeline / roadmap with phases"),
    ("event bus", "diagram", 'd.bus("events", "Event bus", members=["api", "worker"])',
     'd = diagram.Diagram(title="Events")\nd.node("api", "API")\nd.node("worker", "Worker")\n'
     'd.bus("events", "Event bus", members=["api", "worker"])\nd.draw(cv)',
     "bus pub sub pubsub topic event message queue broker kafka redis stream", "Event bus: one bar with a tap to "
     "each member"),
    ("comparison table", "diagram", 'diagram.Table(["", "Free", "Pro"], [["SSO", False, True]], highlight="Pro")',
     'diagram.Table(["", "Free", "Pro"], [["SSO", False, True], ["Seats", "3", "Unlimited"]], '
     'highlight="Pro").draw(cv)',
     "comparison table matrix features versus compare checkmarks", "Comparison table with checks and crosses"),
    ("callouts", "diagram", 'diagram.callouts(cv, [((0.4, 0.4), "Nucleus"), ((0.6, 0.6), "Vacuole")])',
     None, "labels labelled labeled callout leader lines annotate annotation textbook parts",
     "Labels with leader lines beside a figure, never overlapping"),
    ("numbered markers", "diagram", 'diagram.markers(cv, [((0.4, 0.4), "Inlet"), ((0.6, 0.6), "Outlet")])',
     None, "markers numbered badges legend key", "Numbered badges with a legend card"),
    ("brace", "diagram", 'diagram.brace(cv, (0.2, 0.8), (0.8, 0.8), "60 cm")', None, "brace bracket span group",
     "Curly brace with a label"),
    ("dimension arrow", "diagram", 'diagram.dimension(cv, (0.2, 0.85), (0.8, 0.85), "60 µm")', None,
     "dimension measurement arrow length size", "Measurement arrow with its value"),
    ("infographic", "diagram", 'diagram.page(1920, 1080, title="Overview")',
     'dwg, box = diagram.page(1920, 1080, title="Overview")\ndwg.draw(cv)',
     "infographic page layout panels overview", "Blank themed page with a title band for infographics"),
]


def _kit_diagram(m):
    out = []
    for name, kind, call, code, words, summary in _DIAGRAMS:
        head = call.split("(")[0].split(".")[-1]
        if not hasattr(m, head) and not hasattr(getattr(m, "Diagram", None), head):
            continue
        out.append(Entry("diagram", name, kind, call, summary, words.split() + ["diagram"],
                         code=code or call, func=f"diagram.{head}" if hasattr(m, head) else f"diagram.Diagram.{head}"))
    aliases = {}
    for a, t in getattr(m, "_ALIASES", {}).items():
        aliases.setdefault(t, []).append(a)
    for shape in getattr(m, "SHAPES", ()):
        call = f'd.node("a", "{shape.title()}", shape="{shape}")'
        out.append(Entry("diagram", shape, "diagram shape", call, f"{shape} node in a Diagram",
                         ["node", "shape", "flowchart", "diagram"], aliases=aliases.get(shape, ()),
                         code=f"d = diagram.Diagram()\n{call}\nd.draw(cv)", func="diagram.Diagram.node"))
    return out


_CHART_DATA = {
    "line": ("[3, 5, 4, 6, 8]", "line trend time series over time growth"),
    "area": ("[3, 5, 4, 6, 8]", "area filled trend cumulative"),
    "bar": ('{"Mon": 12, "Tue": 18, "Wed": 9}', "bar column compare categories"),
    "barh": ('{"Alpha": 42, "Beta": 31, "Gamma": 18}', "horizontal bar ranking"),
    "stacked": ('{"2025": [3, 4, 5], "2026": [4, 5, 7]}, x=["Q1", "Q2", "Q3"]', "stacked bar composition"),
    "scatter": ('{"x": [1, 2, 3, 4, 5], "y": [2, 4, 3, 5, 6]}', "scatter correlation points xy plot"),
    "hist": ("[1, 2, 2, 3, 3, 3, 4, 4, 5]", "histogram distribution frequency"),
    "heatmap": ('[[1, 4, 2], [3, 1, 5]], x=["A", "B", "C"], y=["Row 1", "Row 2"]', "heatmap matrix grid correlation"),
    "pie": ('{"Search": 34, "Direct": 24, "Ads": 18}', "pie share proportion percent breakdown"),
    "donut": ('{"Basic": 50, "Pro": 35, "Team": 15}', "donut doughnut ring share mix proportion"),
    "radar": ('{"Model": [4, 3, 5, 2]}, x=["Speed", "Cost", "Quality", "Docs"]', "radar spider web profile"),
    "spark": ("[3, 5, 4, 6, 5, 7]", "sparkline small inline trend"),
}


def _kit_chart(m):
    out = []
    quick = getattr(m, "quick", None)
    kinds = []
    if quick is not None:
        mm = re.search(r"``kind``:\s*([a-z ,]+)\.", inspect.getdoc(quick) or "")
        kinds = [k.strip() for k in mm.group(1).split(",")] if mm else list(_CHART_DATA)
    for k in kinds:
        data, words = _CHART_DATA.get(k, ("[3, 5, 4, 6]", k))
        call = f'chart.quick(cv, "{k}", {data}, title="Title")'
        out.append(Entry("chart", k, "chart kind", call, f"{k} chart in one call (axes, ticks, legend automatic)",
                         words.split() + ["chart", "graph", "plot", "data"], func="chart.quick"))
    extra = [
        ("kpi tile", 'chart.kpi(cv, (0.1, 0.1, 0.9, 0.9), "Revenue", 128400, delta=12.4, spark=[3, 4, 3.6, 5, 6])',
         "kpi metric stat tile number big delta", "KPI tile: big number, delta pill, sparkline", "kpi"),
        ("dashboard", 'chart.figure(cv, theme="dark")',
         "dashboard analytics report panel kpi",
         "Dashboard figure: header, KPI tiles and charts in cards", "figure"),
        ("dashboard header", 'chart.header(cv, "Store performance", "Q3 2026")',
         "header title band dashboard", "Dashboard header band", "header"),
    ]
    for name, call, words, summary, fn in extra:
        if not hasattr(m, fn):
            continue
        code = call
        if name == "dashboard":
            code = ('fig = chart.figure(cv, theme="dark")\nfig.header("Store performance", "Q3 2026")\n'
                    'fig.kpi((0.02, 0.2, 0.32, 0.4), "Orders", 18420, delta=8.1)\n'
                    'fig.axes((0.02, 0.45, 0.98, 0.98), title="Revenue").line([3, 5, 4, 6])\nfig.draw()')
        out.append(Entry("chart", name, "chart kind", call, summary, words.split() + ["chart"], code=code,
                         func=f"chart.{fn}"))
    return out


_SHEET = 'sh = drafting.Sheet("A3", title="PART", drawing_no="PC-0001")'
_DRAFTING = [
    ("part drawing", 'drafting.Sheet("A3", title="CLAMP BLOCK", drawing_no="PC-2001")',
     _SHEET + '\nv = sh.view("FRONT")\nv.rect(0, 0, 80, 50, chamfer=3)\nv.hole(40, 25, 16)\n'
              'v.dim_h((0, 50), (80, 50), side="top")\nv.dim_v((0, 0), (0, 50))\nsh.draw(cv)',
     "engineering technical mechanical part drawing cad dimension view orthographic blueprint",
     "Engineering sheet: views, dimensions, title block (raster, SVG, DXF)"),
    ("floor plan", 'drafting.Sheet("A3", scale="1:50", title="STUDIO")',
     'sh = drafting.Sheet("A3", scale="1:50", title="STUDIO")\np = sh.view("PLAN")\n'
     'p.wall([(0, 0), (6000, 0), (6000, 4000), (0, 4000)], thickness=250, closed=True)\n'
     'p.door(3000, 0, 900, toward=(3000, 1500))\np.window(0, 2000, 1400)\np.room("STUDIO", 3000, 2000)\nsh.draw(cv)',
     "floor plan floorplan apartment house layout architectural room wall door window",
     "Floor plan: walls, doors with swings, windows, rooms with areas"),
    ("section view", 'v.hatch(points, material="steel")',
     _SHEET + '\nv = sh.view("SECTION A-A")\nv.hatch([(0, 0), (40, 0), (40, 20), (0, 20)], material="steel", '
              'outline=True)\nsh.draw(cv)',
     "section cut hatch hatching cross section", "Sectioned area with material hatching"),
    ("isometric view", 'v.iso_box(0, 0, 0, 60, 40, 20)',
     _SHEET + '\nv = sh.view("ISO", label="ISOMETRIC VIEW")\nv.iso_box(0, 0, 0, 60, 40, 20)\nsh.draw(cv)',
     "isometric pictorial 3d view exploded", "Isometric pictorial with hidden lines removed"),
    ("gd&t frame", 'v.fcf(40, -52, "position", "0.2", ["A"], diameter=True)',
     _SHEET + '\nv = sh.view("FRONT")\nv.circle(0, 0, 50)\nv.datum(0, -50, "A")\n'
              'v.fcf(40, -52, "position", "0.2", ["A"], diameter=True)\nsh.draw(cv)',
     "gdt gd&t tolerance feature control frame datum", "GD&T feature control frame and datum"),
    ("bolt circle", 'v.bolt_circle(0, 0, 76, 6, 9)',
     _SHEET + '\nv = sh.view("FRONT")\nv.circle(0, 0, 50)\nv.bolt_circle(0, 0, 76, 6, 9)\nsh.draw(cv)',
     "bolt circle holes flange pattern pitch", "Holes on a pitch circle with centre lines"),
    ("scale bar", 'sh.scale_bar(300, 230, length=5)', 'sh = drafting.Sheet("A3")\nsh.scale_bar(300, 230, length=5)\n'
                                                       'sh.draw(cv)', "scale bar plan map", "Scale bar for plans"),
    ("north arrow", 'sh.north_arrow(380, 250)', 'sh = drafting.Sheet("A3")\nsh.north_arrow(380, 250)\n'
                                               'sh.scale_bar(300, 230, length=5)\nsh.draw(cv)',
     "north arrow compass scale bar map", "North arrow and scale bar for plans"),
    ("room", 'p.room("KITCHEN", 3000, 2000)',
     'sh = drafting.Sheet("A3", scale="1:50")\np = sh.view("PLAN")\n'
     'p.wall([(0, 0), (6000, 0), (6000, 4000), (0, 4000)], thickness=250, closed=True)\n'
     'p.room("KITCHEN", 3000, 2000)\nsh.draw(cv)',
     "room name area label kitchen bedroom bathroom living balcony plan", "Room name with its measured floor area"),
    ("stairs", 'p.stairs(1000, 1000, 1000, 3000)',
     'sh = drafting.Sheet("A3", scale="1:50")\np = sh.view("PLAN")\np.stairs(1000, 1000, 1000, 3000)\nsh.draw(cv)',
     "stairs staircase steps plan", "Stair symbol for floor plans"),
]


def _kit_drafting(m):
    out = []
    if not hasattr(m, "Sheet"):
        return out
    for name, call, code, words, summary in _DRAFTING:
        out.append(Entry("drafting", name, "drawing", call, summary, words.split() + ["drafting"], code=code,
                         func="drafting.Sheet"))
    for f in getattr(m, "FURNITURE", ()):
        call = f'p.furniture("{f}", 1500, 1500)'
        out.append(Entry("drafting", f, "plan symbol", call, f"{f.replace('_', ' ')} plan symbol for floor plans",
                         ["furniture", "plan", "symbol", "floor"],
                         code=f'sh = drafting.Sheet("A4", scale="1:50")\np = sh.view("PLAN")\n{call}\nsh.draw(cv)',
                         func="drafting.View.furniture"))
    for mat in getattr(m, "MATERIALS", ()):
        out.append(Entry("drafting", f"{mat} hatch", "hatch", f'v.hatch(points, material="{mat}")',
                         f"{mat} section hatching", ["hatch", "section", mat],
                         code=_SHEET + f'\nv = sh.view("SECTION")\nv.hatch([(0, 0), (40, 0), (40, 20), (0, 20)], '
                                       f'material="{mat}")\nsh.draw(cv)', func="drafting.View.hatch"))
    return out


_SOLID_WORDS = {"quadruped": "animal dog cat bear sheep pet", "character": "person people figure human",
                "car": "vehicle automobile", "truck": "vehicle lorry", "rocket": "spaceship space",
                "robot": "bot android", "tree": "pine", "house": "home building cottage", "room": "interior",
                "laptop": "computer notebook", "speaker": "audio", "potted_plant": "plant houseplant",
                "coffee_machine": "espresso", "desk_lamp": "lamp light", "pendant_lamp": "lamp light ceiling",
                "toy_block": "toy blocks letter cube", "mug": "cup coffee", "rug": "carpet"}


def _solid_prefab_names(m):
    """The prefab registry: ``solid.names()`` (a prefab is indexed the day it is registered)."""
    try:
        names = list(m.names())
    except Exception:                                   # an older solid without a registry
        names = []
    return [n for n in dict.fromkeys(names) if callable(getattr(m, n, None)) and n != "model"]


def _solid_aliases(m, name):
    """[(alias words)] of a prefab from the registry: its own alias phrases ('reading lamp', 'coffee table')."""
    reg = getattr(m, "PREFABS", {}).get(name)
    words = reg[1] if isinstance(reg, tuple) and len(reg) > 1 else ""
    others = {t for n in getattr(m, "PREFABS", {}) if n != name for t in [n.replace("_", " ")] if " " not in t}
    return [w for w in words.split() if w not in others and w not in _SOLID_LOOSE]


_SOLID_LOOSE = {"glass", "pot", "flowers", "bar", "shop", "kitchen", "dining", "seat", "drink", "ceramic", "opening",
                "view", "glowing", "japanese", "garden", "letter", "space", "audio", "smart", "apple", "fitness", "mobile",
                "wireless", "standard", "standing", "end", "drum", "double", "bedroom", "books", "library", "interior"}


def _kit_solid(m):
    out = []
    after = "sc.camera = solid.Camera.frame(sc)\nsc.render(cv)"
    for name in _solid_prefab_names(m):
        fn = getattr(m, name)
        if _needs(fn, skip=0):
            continue
        try:
            summary = (m.describe(name).split("\n")[1:2] or [""])[0].strip() or _first_line(fn)
        except Exception:
            summary = _first_line(fn)
        call = f'sc.add(solid.{name}(), "{name}")'
        out.append(Entry("solid", name, "prefab", call, "3D " + (summary or name),
                         [name.replace("_", " "), "3d", "render", "model"] + _SOLID_WORDS.get(name, "").split()
                         + _solid_aliases(m, name),
                         code=f'sc = solid.Scene(lighting="studio")\n{call}\n{after}', func=f"solid.{name}",
                         heavy=True))
    prims = {"sphere": "sphere(0.5)", "box": "box(1, 0.6, 0.8, r=0.05)", "cylinder": "cylinder(0.4, 1.0)",
             "cone": "cone(0.5, 0.0, 1.0)", "torus": "torus(0.5, 0.15)",
             "capsule": "capsule((0, 0, 0), (0, 1, 0), 0.2)",
             "lathe": "lathe([(0.0, 0.0), (0.3, 0.0), (0.35, 0.4), (0.15, 0.8), (0.0, 0.8)])",
             "extrude_text": 'extrude_text("Hi", height=0.4, depth=0.12)'}
    for name, expr in prims.items():
        if not hasattr(m, name):
            continue
        call = f'sc.add(solid.{expr}.place(x=0, z=0), "{name}")'
        out.append(Entry("solid", name, "3D shape", call, "3D " + _first_line(getattr(m, name)),
                         [name.replace("_", " "), "3d", "shape", "solid", "primitive"],
                         code=f'sc = solid.Scene(lighting="studio")\n{call}\n{after}', func=f"solid.{name}",
                         heavy=True))
    for name in getattr(m, "PRESETS", {}):
        call = f'sc.add(solid.sphere(0.5).place(x=0, z=0), "ball", "{name}")'
        out.append(Entry("solid", name, "3D material", call, f"{name} material preset for 3D objects",
                         [name, "material", "3d", "shader", "finish"],
                         code=f'sc = solid.Scene(lighting="studio")\n{call}\n{after}', func="solid.material",
                         heavy=True))
    extra = [("isometric", "sc.camera = solid.Camera.isometric(sc)", "isometric diorama cutaway view",
              "Isometric camera fitted to the whole scene"),
             ("studio backdrop", 'sc.backdrop("#f2d4d8", distance=0.6, radius=0.35)',
              "seamless backdrop sweep studio product shot packshot", "Seamless studio sweep behind a product")]
    for name, call, words, summary in extra:
        code = (f'sc = solid.Scene(lighting="studio", ground=None)\nsc.add(solid.mug(), "mug")\n{call}\n'
                + ("sc.render(cv)" if "camera" in call else after))
        out.append(Entry("solid", name, "3D scene", call, summary, words.split(), code=code, func="solid.Scene",
                         heavy=True))
    for name in getattr(m, "LIGHTING", {}):
        call = f'solid.Scene(lighting="{name}")'
        out.append(Entry("solid", name, "lighting", call, f"{name} lighting preset for 3D scenes",
                         [name, "lighting", "light", "3d"],
                         code=f'sc = solid.Scene(lighting="{name}")\n'
                              f'sc.add(solid.sphere(0.5).place(x=0, z=0), "ball")\n'
                              f'{after}', func="solid.Scene", heavy=True))
    return out


_TEXT_RECIPES = [
    ("text on arc", 'text.text_on_arc(cv, "PIXELCANVAS TYPE", (160, 90), 60)',
     "arc circle ring circular badge seal curved",
     "text_on_arc"),
    ("text on path", 'text.text_on_path(cv, "follows any curve", [(20, 150), (160, 40), (300, 150)], 24)',
     "path curve wavy follow", "text_on_path"),
    ("vertical text", 'cv.text("縦書きの日本語", (300, 10), 22, "#222222", anchor="ra", vertical=True)',
     "vertical cjk chinese japanese column tategaki", None),
    ("fit text in box", 'cv.text("Fits any box", box=(20, 20, 300, 160), color="#ffffff", bold=True)',
     "fit box autofit wrap paragraph headline title", None),
    ("outlined text", 'cv.text("OUTLINE", (160, 90), 48, "#ffffff", anchor="mm", outline=4, outline_color="#111111")',
     "outline stroke outlined thumbnail bold", None),
    ("glowing text", 'cv.text("GLOW", (160, 90), 56, "#7af7ff", anchor="mm", glow=12)', "glow glowing", None),
    ("gradient text", 'cv.text("GRADIENT", (160, 90), 48, anchor="mm", gradient=[(0, "#ff7a59"), (1, "#6a5cff")])',
     "gradient colourful", None),
]


def _kit_text(m):
    out = []
    for eff in getattr(m, "EFFECTS", ()):
        call = f'cv.text("Hello", (cv.w / 2, cv.h / 2), 48, "#ffcc33", anchor="mm", bold=True, effect="{eff}")'
        out.append(Entry("text", eff, "text effect", call, f"{eff.replace('_', ' ')} lettering effect",
                         [eff.replace("_", " "), "text", "lettering", "title", "font", "type", "word", "effect"],
                         func="canvas.Canvas.text"))
    for name, call, words, fn in _TEXT_RECIPES:
        if fn and not hasattr(m, fn):
            continue
        out.append(Entry("text", name, "text", call, f"{name} (exact text in any language)",
                         words.split() + ["text", "lettering", "type"],
                         func=f"text.{fn}" if fn else "canvas.Canvas.text"))
    return out


_LAYOUT_CONTENT = {"title": '"Title"', "subtitle": '"Subtitle"', "kicker": '"Kicker"',
                   "name": '"Ada Lovelace"', "quote": '"Simplicity is the ultimate sophistication."',
                   "author": '"Leonardo da Vinci"', "role": '"Engineer"', "company": '"Analytical Co."',
                   "date": '"Fri 21:00"', "headline": '"The new issue"', "center": '"2026"'}
_LAYOUT_WORDS = {"poster": "poster print", "flyer": "flyer leaflet event", "book_cover": "book cover novel",
                 "magazine": "magazine cover masthead", "business_card": "business card contact visiting",
                 "quote_card": "quote card social", "thumbnail": "thumbnail youtube video social preview banner og",
                 "slide": "slide presentation deck", "certificate": "certificate award diploma",
                 "menu": "menu restaurant cafe prices food", "badge": "badge seal stamp emblem round",
                 "label": "label product packaging jar bottle"}


def _kit_layout(m):
    out = []
    tpl = getattr(m, "TEMPLATES", {})
    seen = {}
    for name, fn in tpl.items():
        seen.setdefault(id(fn), []).append(name)
    for names in seen.values():
        name, aliases = names[0], [a for a in names[1:] if a not in ("card", "book", "quote")]
        fn = tpl[name]
        need = _needs(fn, skip=1) if callable(fn) else []
        try:
            params = list(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            params = []
        args = [f"{p}={_LAYOUT_CONTENT[p]}" for p in params if p in _LAYOUT_CONTENT and
                (p in need or p in ("title", "subtitle"))]
        args += [f'{p}="..."' for p in need if p not in _LAYOUT_CONTENT]
        call = f'layout.template("{name}", cv, {", ".join(args)})'
        words = " ".join(_LAYOUT_WORDS.get(n, n.replace("_", " ")) for n in names)
        out.append(Entry("layout", name, "layout template", call,
                         f"finished {name.replace('_', ' ')} design from content alone (type, margins, image box)",
                         words.split() + ["design", "template", "layout", "typography"], aliases=aliases,
                         func="layout.template"))
    for name in getattr(m, "THEMES", {}):
        out.append(Entry("layout", name, "theme", f'layout.template("poster", cv, title="Title", theme="{name}")',
                         f"{name} type theme for layout templates", [name, "theme", "design"],
                         func="layout.template"))
    return out


def _kit_scenery(m):
    return []         # landscape recipes are reached through the "landscape" scene kind and `show scenery.<fn>`


_SPECIES_GROUP = {"quad": "animal mammal pet", "bird": "bird animal", "sea": "sea animal marine ocean fish",
                  "small": "small animal bug insect"}


def _kit_creature(m):
    out = []
    if not hasattr(m, "species"):
        return out
    aliases = {}
    for dname in ("_ALIASES", "_BIRD_ALIASES", "_SEA_ALIASES", "_SMALL_ALIASES"):
        for a, t in getattr(m, dname, {}).items():
            aliases.setdefault(t, []).append(a)
    group = getattr(m, "_KINDS", {})
    poses = tuple(getattr(m, "POSES", ()))
    for sp in m.species():
        call = f'creature.creature(cv, "{sp}", at=(0.5, 0.9), size=0.4)'
        out.append(Entry("creature", sp, "animal", call, f"{sp} - 2D illustration, posable, 5 views",
                         [sp] + _SPECIES_GROUP.get(group.get(sp, "quad"), "animal").split() + ["2d"],
                         aliases=aliases.get(sp, ()), func="creature.creature", poses=poses))
    if hasattr(m, "flock"):
        out.append(Entry("creature", "flock", "animal", 'creature.flock(cv, "crow", n=7)',
                         "a flock of birds with their own wing beats", ["flock", "birds", "flying", "group"],
                         func="creature.flock"))
    if hasattr(m, "school"):
        out.append(Entry("creature", "school", "animal", 'creature.school(cv, "fish", n=9)',
                         "a school of fish heading the same way", ["school", "fish", "shoal", "group"],
                         func="creature.school"))
    return out


def _kit_figure(m):
    out = []
    if not hasattr(m, "person"):
        return out
    poses = tuple(getattr(m, "POSES", ()))
    words = {"office_worker": "office worker coworker employee", "businessman": "business suit",
             "kid": "child kid boy girl", "grandma": "grandmother old woman elderly",
             "grandpa": "grandfather old man elderly", "teen": "teenager student", "hero": "superhero",
             "musician": "guitar music", "astronaut": "space", "doctor": "nurse medical"}
    for p in getattr(m, "PEOPLE", {}):
        call = f'figure.person("{p}").draw(cv, at=(0.5, 0.95))'
        out.append(Entry("figure", p, "person", call, f"{p.replace('_', ' ')} - posable dressed person (2D)",
                         [p.replace("_", " "), "person", "people", "character"] + words.get(p, "").split(),
                         func="figure.person", poses=poses))
    out.append(Entry("figure", "person", "person",
                     'figure.person("a woman in a red coat waving").draw(cv, at=(0.5, 0.95))',
                     "any person from a sentence: build, clothes, hair, pose, expression",
                     ["person", "people", "man", "woman", "girl", "boy", "friend", "human", "character", "figure",
                      "guy", "lady", "crowd"], func="figure.person", poses=poses))
    if hasattr(m, "crowd"):
        out.append(Entry("figure", "crowd", "person", "figure.crowd(cv, 8, (0.1, 0.7, 0.9, 0.95), seed=1)",
                         "a crowd of varied people, back to front", ["crowd", "people", "group", "team", "audience"],
                         func="figure.crowd"))
    return out


def _kit_clay(m):
    out = []
    poses = tuple(getattr(m, "ANIMAL_POSES", ()) or ())
    after = "sc.camera = solid.Camera.frame(sc)\nsc.render(cv)"
    if hasattr(m, "animal"):
        for sp in getattr(m, "SPECIES", ()):
            call = f'sc.add(clay.animal("{sp}"), "{sp}")'
            out.append(Entry("clay", sp, "animal", call, f"{sp} - 3D clay / designer-toy look",
                             [sp, "animal", "3d", "toy", "clay"], func="clay.animal", poses=poses, heavy=True,
                             code=f'sc = solid.Scene(lighting="studio")\n{call}\n{after}'))
    if hasattr(m, "cast"):
        for c in getattr(m, "CAST", ()):
            call = f'sc.add(clay.cast("{c}"), "{c}")'
            out.append(Entry("clay", c, "3D character", call, f"{c.replace('_', ' ')} - 3D clay character",
                             [c.replace("_", " "), "person", "character", "3d", "toy", "clay"], func="clay.cast",
                             poses=tuple(getattr(m, "POSES", ()) or ()), heavy=True,
                             code=f'sc = solid.Scene(lighting="studio")\n{call}\n{after}'))
    if hasattr(m, "prop"):
        for p in getattr(m, "PROPS", ()):
            call = f'sc.add(clay.prop("{p}"), "{p}")'
            out.append(Entry("clay", p, "prop", call, f"{p.replace('_', ' ')} - 3D clay prop",
                             [p.replace("_", " "), "prop", "3d", "toy", "clay"], func="clay.prop", heavy=True,
                             code=f'sc = solid.Scene(lighting="studio")\n{call}\n{after}'))
    return out


_FORM = [
    ("ball", "form.ball(cv, (0.5, 0.6), 0.15, base=\"#f07a18\", material=\"fruit\", cast=True)",
     "ball sphere orange fruit apple round"),
    ("cylinder", "form.cylinder(cv, (0.5, 0.75), 0.1, 0.25, base=\"#e9e1d3\", material=\"ceramic\", handle=True)",
     "cylinder mug can glass pillar cup"),
    ("cone", "form.cone(cv, (0.5, 0.8), 0.12, 0.3)", "cone lamp shade pot"),
    ("lathe", "form.lathe(cv, (0.5, 0.85), [(0.08, 0), (0.09, 0.03), (0.09, 0.26), (0.03, 0.4)], material=\"glass\")",
     "lathe bottle vase bowl jug round object"),
    ("cuboid", "form.cuboid(cv, (0.5, 0.8), (0.3, 0.2, 0.15), yaw=30)", "cuboid box crate book cube"),
    ("torus", "form.torus(cv, (0.5, 0.6), 0.12, 0.04)", "torus ring donut"),
    ("tabletop", "form.tabletop(cv, top=0.5, front=0.95, horizon=0.38)", "table tabletop still life surface"),
    ("room", "form.Room(cv).draw()", "room interior perspective walls floor"),
    ("ground", "form.Ground(cv, horizon=0.45)", "ground floor plane perspective"),
]


def _kit_form(m):
    out = []
    for name, call, words in _FORM:
        head = call.split("(")[0].split(".")[-1]
        fn = getattr(m, head, None)
        if fn is None:
            continue
        out.append(Entry("form", name, "2.5D object", call, _first_line(fn) or name,
                         words.split() + ["volume", "shaded", "object"], func=f"form.{head}"))
    for mat in getattr(m, "MATERIALS", {}):
        out.append(Entry("form", f"{mat} material", "form material",
                         f'form.ball(cv, (0.5, 0.6), 0.15, base="#3a8fe0", material="{mat}")',
                         f"{mat} shading for form solids", [mat, "material", "shading"], func="form.paint_form"))
    return out


def _scene_kinds():
    """{kind: one-line description} from scenes/_template_<kind>.py (what new_scene.py --kind takes)."""
    out = {}
    for path in sorted(glob.glob(os.path.join(RUNTIME, "scenes", "_template*.py"))):
        stem = os.path.basename(path)[:-3]
        kind = stem[len("_template_"):] if stem.startswith("_template_") else "landscape"
        if kind in out:
            continue
        try:
            with open(path, encoding="utf-8") as f:
                src = f.read(4000)
            mm = re.search(r'"""\s*(.+?)\s*(?:\n|""")', src)
            out[kind] = mm.group(1).strip() if mm else ""
        except OSError:
            out[kind] = ""
    return out


def _kit_scenes(_m):
    out = []
    for kind, desc in _scene_kinds().items():
        out.append(Entry("scenes", kind, "scene kind", f"python tools/new_scene.py NAME --kind {kind}", desc,
                         [kind.replace("_", " "), "scene", "template", "starter"] + _words(desc)[:12],
                         lang="shell"))
    return out


def _fn_entries(m, kit, kind, skip=()):
    """Light introspection: public functions with a docstring that run with only a canvas."""
    out = []
    for name, fn in inspect.getmembers(m, inspect.isfunction):
        if name.startswith("_") or name in skip or getattr(fn, "__module__", "") != m.__name__:
            continue
        doc = _first_line(fn)
        try:
            params = list(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            continue
        if not doc or not params or params[0] not in ("cv", "canvas") or _needs(fn):
            continue
        a = _args(fn)
        out.append(Entry(kit, name.replace("_", " "), kind, f"{kit}.{name}(cv{', ' + a if a else ''})", doc,
                         [name.replace("_", " ")] + _words(doc)[:10], func=f"{kit}.{name}"))
    return out


def _kit_generic(m):
    """A kit without a handler: its names() registry when one function takes (cv, name), else its functions."""
    kit = m.__name__.rsplit(".", 1)[-1]
    out = []
    names = getattr(m, "names", None)
    if callable(names):
        target = None
        for fname, fn in inspect.getmembers(m, inspect.isfunction):
            try:
                ps = list(inspect.signature(fn).parameters)
            except (TypeError, ValueError):
                continue
            if len(ps) >= 2 and ps[0] in ("cv", "canvas") and ps[1] in ("name", "kind", "species", "key") \
                    and not _needs(fn, skip=2):
                target = fname
                break
        if target:
            try:
                for n in names():
                    desc = ""
                    if callable(getattr(m, "describe", None)):
                        try:
                            desc = str(m.describe(n)).split("\n")[0]
                        except Exception:
                            desc = ""
                    out.append(Entry(kit, n, kit, f'{kit}.{target}(cv, "{n}")', desc, [kit] + _words(desc)[:10],
                                     func=f"{kit}.{target}"))
            except Exception:
                out = []
    return out + _fn_entries(m, kit, "function")


def _kit_pictograms(m):
    """The long tail of nameable objects: one entry per pictogram, searchable by its names, aliases (English and
    Chinese), keywords and category."""
    out = []
    for name in m.names():
        try:
            info = m.info(name)
        except Exception:
            info = {}
        aliases = [a for a in info.get("aliases", []) if a.isascii()]
        cjk = [a for a in info.get("aliases", []) if not a.isascii()]
        cat = info.get("category", "")
        try:
            desc = str(m.describe(name))
        except Exception:
            desc = name
        out.append(Entry("pictograms", name, "pictogram",
                         f'pictograms.pictogram(cv, "{name}", at=(0.5, 0.5), size=0.3, color="#1d2633")', desc,
                         [cat, "pictogram", "silhouette"] + list(info.get("keywords", [])) + cjk,
                         aliases=aliases, func="pictograms.pictogram"))
    return out


_HANDLERS = {"pictograms": _kit_pictograms, "flora": _kit_flora, "elements": _kit_elements, "texture": _kit_texture, "stylize": _kit_stylize,
             "palette": _kit_palette, "backdrop": _kit_backdrop, "ui": _kit_ui, "vector": _kit_vector,
             "diagram": _kit_diagram, "chart": _kit_chart, "drafting": _kit_drafting, "solid": _kit_solid,
             "text": _kit_text, "layout": _kit_layout, "scenery": _kit_scenery, "creature": _kit_creature,
             "figure": _kit_figure, "clay": _kit_clay, "form": _kit_form}


# --------------------------------------------------------------------------------------------- the index
def kit_names():
    """Kit modules found in the package folder (plus "scenes"), in display order."""
    found = [os.path.basename(p)[:-3] for p in glob.glob(os.path.join(PKG, "*.py"))]
    found += [os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(PKG, "*", "__init__.py"))]
    found = [k for k in found if k not in _CORE and not k.startswith("_")]
    order = {k: i for i, k in enumerate(_KIT_ORDER)}
    return sorted(set(found) | {"scenes"}, key=lambda k: (order.get(k, 50), k))


def _load(kit):
    if kit == "scenes":
        return None
    return importlib.import_module(f"pixelcanvas.{kit}")


def build(kits=None):
    """Build a fresh index (list of Entry). A kit that fails to import or to describe itself is skipped."""
    out = []
    for kit in (kits or kit_names()):
        try:
            mod = _load(kit)
            handler = _HANDLERS.get(kit, _kit_scenes if kit == "scenes" else _kit_generic)
            got = handler(mod)
        except Exception:
            continue
        for i, e in enumerate(got):
            e.order = i
        out.extend(got)
    for e in out:
        extra = _EXTRA_ALIASES.get(f"{e.kit}.{e.name}")
        if extra:
            e.aliases = tuple(dict.fromkeys(e.aliases + extra))
        _prepare(e)
    return out


_ROUND = {"orange": ("#f08a24", "citrus"), "lemon": ("#f2c14e", "citrus"), "lime": ("#9ccc3c", "citrus"),
          "tomato": ("#d64541", "fruit"), "peach": ("#f4a77a", "fruit"), "plum": ("#6e3a7a", "fruit"),
          "ball": ("#3a8fe0", "plastic"), "sphere": ("#3a8fe0", "glossy")}

_EXTRA_ALIASES = {"diagram.comparison table": ("table", "matrix", "comparison"), "ui.data table": ("table",),
                  "chart.bar": ("chart", "graph", "bar graph", "column chart"), "chart.line": ("line graph",),
                  "chart.pie": ("pie graph",), "diagram.infographic": ("infographic page",),
                  "diagram.architecture diagram": ("architecture",),
                  "diagram.flowchart": ("flow chart", "flow diagram", "process diagram"),
                  "elements.campfire": ("bonfire", "camp fire"), "elements.flame": ("fire",),
                  "elements.stars": ("starry sky", "night sky", "milky way"),
                  "form.ball": tuple(_ROUND) + ("fruit", "round fruit"),
                  "elements.breaking wave": ("great wave", "big wave", "huge wave", "crashing wave"),
                  "layout.thumbnail": ("banner", "readme banner", "social preview", "cover image", "header image"),
                  "diagram.callouts": ("labelled diagram", "labeled diagram", "textbook figure", "physics diagram",
                                       "science diagram", "scientific diagram", "anatomy diagram", "leader lines"), "elements.moon": ("planet",),
                  "drafting.part drawing": ("engineering drawing", "technical drawing", "mechanical drawing"),
                  "layout.poster": ("print",), "layout.book_cover": ("cover",),
                  "texture.film_grain": ("grain", "film grain", "noise"),
                  "solid.isometric": ("isometric camera", "isometric view", "isometric 3d"),
                  "pictograms.bus stop sign": ("bus stop",),
                  "pictograms.bridge (arch)": ("bridge", "arched bridge", "arch bridge", "stone bridge"),
                  "pictograms.bridge (suspension)": ("suspension bridge", "golden gate")}


_INDEX = None
_MIN_SCORE = 60          # a phrase needs a name / alias match or two keywords to be answered


def _index():
    global _INDEX
    if _INDEX is None:
        _INDEX = build()
    return _INDEX


def reset():
    """Forget the cached index (the next call rebuilds it)."""
    global _INDEX
    _INDEX = None
    _VOCAB.clear()
    _COLOR_CACHE.clear()
    _TYPO_BY_FIRST.clear()
    _NAME_TOKS.clear()


def entries(kit=None, kind=None):
    """Every entry, or those of one kit and / or one kind."""
    return [e for e in _index() if (kit is None or e.kit == kit) and (kind is None or e.kind == kind)]


def kits():
    """[(kit, number of entries, kinds, reference page)] for the kits that loaded."""
    rows = {}
    for e in _index():
        r = rows.setdefault(e.kit, [0, {}, e.ref])
        r[0] += 1
        r[1][e.kind] = r[1].get(e.kind, 0) + 1
    return [(k, n, list(ks), ref) for k, (n, ks, ref) in rows.items()]


def kinds():
    """{kind: [entries]} in index order."""
    out = {}
    for e in _index():
        out.setdefault(e.kind, []).append(e)
    return out


# --------------------------------------------------------------------------------------------- references
def _ref_page(kit):
    """'references/<page>.md' documenting a kit ('' when there is none yet)."""
    name = _REF_OVERRIDE.get(kit)
    cands = [name] if name else [f"{kit}.md", f"{kit}s.md"]
    for c in cands:
        if os.path.exists(os.path.normpath(os.path.join(REFS, c))):
            return "references/" + c if not c.startswith("..") else "SKILL.md"
    for p in sorted(glob.glob(os.path.join(REFS, "*.md"))):
        try:
            with open(p, encoding="utf-8") as f:
                if f"pixelcanvas.{kit}`" in f.read(3000):
                    return "references/" + os.path.basename(p)
        except OSError:
            pass
    return ""


def reference(kit):
    """Full path of a kit's reference page ('' if none)."""
    ref = _ref_page(kit)
    if not ref:
        return ""
    return os.path.join(RUNTIME, "skills", "pixelcanvas", *ref.split("/"))


def doc(kit):
    """The text of a kit's reference page."""
    path = reference(kit)
    if not path:
        from ._args import unknown_name
        raise KeyError(str(unknown_name("kit", kit, [k for k, *_ in kits() if reference(k)])))
    with open(path, encoding="utf-8") as f:
        return f.read()


def _section(ref, needles):
    """Heading of the first section of a reference page that mentions one of ``needles``."""
    if not ref:
        return ""
    path = os.path.join(RUNTIME, "skills", "pixelcanvas", *ref.split("/"))
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().split("\n")
    except OSError:
        return ""
    for needle in needles:
        head, level, fence = "", 0, False
        for ln in lines:
            if ln.startswith("```"):
                fence = not fence
            if not fence and ln.startswith("#"):
                level = len(ln) - len(ln.lstrip("#"))
                head = ln.lstrip("#").strip()
            if needle and needle in ln and level >= 2:
                return head
    return ""


def _example(ref, needle):
    """First line of a code block in the page that uses ``needle``."""
    if not ref or not needle:
        return ""
    path = os.path.join(RUNTIME, "skills", "pixelcanvas", *ref.split("/"))
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return ""
    strict = re.compile(r"(?<![\w.])" + re.escape(needle))      # plant( but not icon_plant( or s.plant(
    loose = re.compile(r"(?<!\w)" + re.escape(needle))
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    for pat in (strict, loose):
        for block in blocks:
            for ln in block.split("\n"):
                if pat.search(ln):
                    return ln.strip()
    return ""


# --------------------------------------------------------------------------------------------- language
_STOP = set("""line a an the of and or with without in on at by for to from into onto over under above below beside
behind near next between among around through across along inside outside against while as is are be being been
this that these those it its their his her them they he she we our you your my me i who which what where when
there here some any all each every one two three four five six seven eight nine ten twelve dozen few many several
lot lots very really quite just also only than then so such like please make draw create render show showing
generate paint painted give get want need want would could should can will picture image illustration scene
artwork art piece view shot style styled version kind type using use made look looking feel feeling
detailed detail beautiful nice pretty cute simple clean modern minimal minimalist realistic high quality hd 4k 8k
big small tiny huge large little giant tall short wide narrow long thin thick round full half
new old young bright dark light soft hard warm cool cold hot deep pale vivid subtle gentle strong heavy
left right top bottom center centre middle front back side corner foreground background distance distant
morning evening some about something way lots setting end edge
called titled named entitled says saying tagline sheet after before during each whole set

""".split())
_STOP_SCORE = _STOP - {"background", "light", "dark", "old", "warm", "cold", "morning", "evening", "day",
                       "afternoon", "night", "big", "tall", "long", "round"}
_SPLIT = set("""with and under over on in at beside behind near by of from into onto above below between among
around through across along inside outside against while who which that featuring plus or next to for past
toward towards beneath underneath atop upon beyond via amid amongst""".split())

_SING = {"settings": "settings", "leaves": "leaf", "wolves": "wolf", "knives": "knife", "shelves": "shelf",
         "halves": "half", "calves": "calf", "loaves": "loaf", "elves": "elf", "scarves": "scarf", "hooves": "hoof",
         "geese": "goose",
         "mice": "mouse", "people": "person", "children": "child", "men": "man", "women": "woman", "teeth": "tooth",
         "feet": "foot", "shoes": "shoe", "series": "series", "species": "species", "news": "news",
         "glasses": "glasses", "lens": "lens", "canvas": "canvas", "gas": "gas", "bus": "bus", "dice": "dice",
         "data": "data", "cactus": "cactus", "cacti": "cactus", "fungi": "mushroom", "dunes": "dune",
         "tomatoes": "tomato", "potatoes": "potato", "heroes": "hero", "volcanoes": "volcano", "dominoes": "domino",
         "chess": "chess", "moss": "moss", "grass": "grass", "glass": "glass", "dress": "dress", "class": "class",
         "fish": "fish", "sheep": "sheep", "deer": "deer", "aurora": "aurora", "koi": "koi", "physics": "physics",
         "graphics": "graphics", "analytics": "analytics", "metrics": "metric", "stairs": "stairs",
         "trousers": "trousers", "jeans": "jeans", "shorts": "shorts", "pants": "pants", "earbuds": "earbud"}


def _sing(w):
    if w in _SING:
        return _SING[w]
    if len(w) <= 3 or not w.endswith("s") or w.endswith(("ss", "us", "is", "os", "ys")) and not w.endswith("ays"):
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("sses", "xes", "zes", "ches", "shes")):
        return w[:-2]
    return w[:-1]


# synonyms: word -> canonical words (all singular); each also keeps itself
_SYN = {
    "puppy": "dog", "pup": "dog", "doggy": "dog", "hound": "dog", "canine": "dog", "kitty": "cat", "kitten": "cat",
    "feline": "cat", "pony": "horse", "stallion": "horse", "mare": "horse", "bunny": "rabbit", "birdie": "bird",
    "flames": "fire flame", "fire": "flame fire campfire", "blaze": "fire flame", "bonfire": "campfire",
    "fireplace": "fireplace", "firework": "fireworks", "candlelight": "candle",
    "skyscraper": "skyline city building", "building": "house city", "city": "skyline city",
    "downtown": "city skyline", "town": "city house", "village": "house city", "cityscape": "skyline city",
    "couch": "sofa", "settee": "sofa", "seat": "chair",
    "blossom": "cherry blossom", "sakura": "cherry blossom", "bloom": "flower",
    "floral": "flower", "flower": "flower", "bouquet": "bouquet", "houseplant": "potted",
    "sea": "ocean", "seaside": "ocean beach", "beach": "ocean sand", "wave": "ocean breaking wave",
    "surf": "breaking wave", "pond": "lake", "lagoon": "lake", "stream": "river", "creek": "river",
    "woods": "forest", "woodland": "forest", "jungle": "palm forest tropical", "hill": "hills",
    "hillside": "hills", "mountain": "mountains", "peak": "mountains", "valley": "mountains hills",
    "galaxy": "stars", "milky": "stars", "starry": "stars", "star": "stars", "space": "deep space stars rocket",
    "dusk": "dusk sunset", "sundown": "sunset", "sunrise": "dawn sunrise", "daybreak": "dawn",
    "thunderstorm": "storm lightning", "thunder": "lightning", "storm": "storm clouds lightning",
    "snowfall": "snow", "snowy": "snow", "rainy": "rain", "raining": "rain", "drizzle": "rain droplets",
    "foggy": "fog", "misty": "fog mist", "mist": "fog", "haze": "fog", "cloudy": "cloud", "sunny": "sun noon",
    "sunshine": "sun", "sunlight": "sun god rays", "moonlight": "moon", "moonlit": "moon night",
    "night": "night", "nighttime": "night", "graph": "chart", "plot": "chart", "flow": "flowchart",
    "org": "org chart", "hierarchy": "org chart tree", "wallpaper": "backdrop wallpaper",
    "background": "backdrop background", "bg": "backdrop background", "logo": "icon logo", "glyph": "icon",
    "emoji": "icon", "symbol": "icon", "font": "text", "lettering": "text", "typography": "text",
    "headline": "text title", "title": "text title", "word": "text", "colour": "color", "colors": "color",
    "scheme": "palette", "automobile": "car", "vehicle": "car truck", "lorry": "truck", "spaceship": "rocket",
    "teacup": "cup", "coffee": "mug coffee", "tea": "cup", "man": "person", "woman": "person", "girl": "person",
    "boy": "person kid", "guy": "person", "lady": "person", "friend": "person", "human": "person",
    "coworker": "person office worker", "colleague": "person office worker", "worker": "person office worker",
    "child": "kid person", "toddler": "kid", "baby": "kid", "student": "student person", "chef": "chef",
    "cook": "chef", "wooden": "wood", "timber": "wood", "boulder": "rock", "stone": "stone rock",
    "pebble": "pebbles", "brickwork": "brick", "metallic": "metal", "steel": "metal brushed metal",
    "lawn": "grass lawn", "turf": "grass", "watercolour": "watercolor", "colour pencil": "colored pencil",
    "signin": "login", "signup": "login onboarding", "kpi": "kpi", "metric": "kpi", "planet": "moon", "moon": "moon",
    "apartment": "floor plan", "floorplan": "floor plan", "blueprint": "blueprint drawing",
    "cad": "engineering drawing", "schematic": "diagram", "infographic": "infographic diagram",
    "lighthouse": "lighthouse", "beam": "spotlight god rays", "spark": "sparks", "ember": "embers",
    "smoky": "smoke", "steamy": "steam", "puddle": "puddle", "reflection": "lake puddle",
    "waterfall": "waterfall", "lantern": "lantern", "neon": "neon", "bokeh": "bokeh", "glow": "glow",
    "maple": "maple", "branch": "branch frame twig", "twig": "branch", "afternoon": "day", "autumn": "autumn",
    "fall": "autumn", "fallen": "falling", "leaf": "leaf", "steaming": "steam", "smoking": "smoke",
    "burning": "fire flame", "glowing": "glow", "sparkling": "sparks",
    "phone": "phone", "mobile": "phone app", "app": "app screen ui", "website": "landing browser",
    "webpage": "landing browser", "homepage": "landing", "screen": "screen ui",
    "pine": "pine", "conifer": "pine spruce", "evergreen": "pine spruce", "palm": "palm",
    "tulip": "tulip", "rose": "rose", "sunflower": "sunflower", "daisy": "daisy", "lotus": "lotus",
    "fruit": "apple", "lemon": "fruit citrus", "orange": "fruit citrus",
    "sticker": "sticker", "comic": "comic", "manga": "manga", "anime": "anime", "photo": "photo_finish",
    "photograph": "photo_finish", "photorealistic": "photo_finish",
    "card": "card", "banner": "banner", "thumbnail": "thumbnail", "cover": "book cover",
    "poster": "poster", "flyer": "flyer poster", "leaflet": "flyer", "menu": "menu", "certificate": "certificate",
    "slide": "slide", "deck": "slide", "presentation": "slide",
}

# Traditional Chinese / Japanese -> English words (longest match wins)
_CJK = {
    "樹": "tree", "樹木": "tree", "桜の木": "cherry tree", "櫻花樹": "cherry tree", "櫻樹": "cherry tree",
    "松の木": "pine tree", "楓の木": "maple tree", "木": "tree", "森林": "forest", "森": "forest", "貓": "cat", "猫": "cat",
    "狗": "dog", "犬": "dog", "柯基": "corgi", "コーギー": "corgi", "火": "fire", "火焰": "flame", "炎": "fire",
    "營火": "campfire", "焚き火": "campfire", "海": "ocean", "大海": "ocean", "海洋": "ocean", "山": "mountains",
    "高山": "mountains", "雲": "cloud", "雲朵": "cloud", "花": "flower", "花朵": "flower", "海報": "poster",
    "ポスター": "poster", "圖表": "chart", "グラフ": "chart", "流程圖": "flowchart", "フローチャート": "flowchart",
    "工程圖": "engineering drawing", "図面": "engineering drawing", "製圖": "engineering drawing",
    "平面圖": "floor plan", "間取り図": "floor plan", "桌布": "wallpaper", "壁紙": "wallpaper", "圖示": "icon",
    "アイコン": "icon", "圖標": "icon", "背景": "background", "櫻花": "cherry blossom", "桜": "cherry blossom",
    "楓": "maple", "楓樹": "maple", "紅葉": "autumn maple", "松": "pine", "松樹": "pine", "竹": "bamboo",
    "玫瑰": "rose", "向日葵": "sunflower", "鬱金香": "tulip", "蓮花": "lotus", "橡樹": "oak", "棕櫚": "palm",
    "椰子樹": "coconut palm", "仙人掌": "cactus", "盆栽": "bonsai", "草": "grass", "草地": "grass lawn",
    "石頭": "rock", "岩石": "rock", "石": "stone", "馬": "horse", "鳥": "bird", "魚": "fish", "兔子": "rabbit",
    "兔": "rabbit", "狐狸": "fox", "狐": "fox", "熊": "bear", "獅子": "lion", "老虎": "tiger", "虎": "tiger",
    "大象": "elephant", "企鵝": "penguin", "ペンギン": "penguin", "貓頭鷹": "owl", "鯨魚": "whale", "鯨": "whale",
    "海豚": "dolphin", "蝴蝶": "butterfly", "蝶": "butterfly", "青蛙": "frog", "烏龜": "turtle", "亀": "turtle",
    "蠟燭": "candle", "ろうそく": "candle", "煙火": "fireworks", "花火": "fireworks", "煙": "smoke",
    "霧": "fog", "雨": "rain", "雪": "snow", "閃電": "lightning", "雷": "lightning", "彩虹": "rainbow",
    "虹": "rainbow", "極光": "aurora", "オーロラ": "aurora", "星空": "stars", "星星": "stars", "星": "stars",
    "月亮": "moon", "月": "moon", "太陽": "sun", "瀑布": "waterfall", "滝": "waterfall", "湖": "lake",
    "河": "river", "川": "river", "沙漠": "desert dune", "砂漠": "desert dune", "夕陽": "sunset", "日落": "sunset",
    "夕焼け": "sunset", "黃昏": "dusk", "日出": "sunrise", "夜景": "night city", "夜": "night", "城市": "city",
    "都市": "city", "街": "city street", "建築": "building", "房子": "house", "家": "house", "汽車": "car",
    "車": "car", "機器人": "robot", "ロボット": "robot", "火箭": "rocket", "杯子": "mug", "咖啡": "coffee",
    "人": "person", "人物": "person", "女孩": "girl", "男孩": "boy", "女性": "woman", "男性": "man",
    "廚師": "chef", "醫生": "doctor", "磚": "brick", "磚牆": "brick wall", "木紋": "wood grain", "木頭": "wood",
    "大理石": "marble", "布料": "fabric", "金屬": "metal", "漸層": "gradient", "調色盤": "palette",
    "配色": "palette", "按鈕": "button", "ボタン": "button", "登入": "login", "儀表板": "dashboard",
    "ダッシュボード": "dashboard", "介面": "ui screen", "手機": "phone app", "網頁": "landing page",
    "長條圖": "bar chart", "棒グラフ": "bar chart", "折線圖": "line chart", "圓餅圖": "pie chart",
    "円グラフ": "pie chart", "散佈圖": "scatter chart", "架構圖": "architecture diagram", "組織圖": "org chart",
    "心智圖": "mind map", "時間軸": "timeline", "表格": "table", "名片": "business card", "傳單": "flyer",
    "封面": "cover", "菜單": "menu", "證書": "certificate", "縮圖": "thumbnail", "投影片": "slide",
    "文字": "text", "金色": "gold", "霓虹": "neon", "立體": "3d", "3D": "3d", "等角": "isometric",
    "插畫": "illustration", "風景": "landscape", "風景畫": "landscape", "山水": "mountains", "產品": "product",
    "静物": "still life", "靜物": "still life", "房間": "room", "部屋": "room", "客廳": "living room",
    "標誌": "logo", "ロゴ": "logo", "貼紙": "sticker", "ステッカー": "sticker",
}

_POSE_WORDS = {"sitting": ("sit",), "sit": ("sit",), "seated": ("sit",), "lying": ("lie",), "lie": ("lie",),
               "sleeping": ("sleep", "lie"), "asleep": ("sleep", "lie"), "curled": ("sleep", "lie"),
               "running": ("run", "gallop", "trot"), "galloping": ("gallop",), "trotting": ("trot",),
               "walking": ("walk",), "standing": ("stand",), "jumping": ("jump",), "leaping": ("jump",),
               "flying": ("fly", "glide"), "soaring": ("glide", "fly"), "swimming": ("swim",),
               "perched": ("perch",), "perching": ("perch",), "waving": ("wave",), "pointing": ("point",),
               "dancing": ("dance",), "kneeling": ("kneel",), "cheering": ("cheer",), "grazing": ("graze",),
               "sniffing": ("sniff",), "crouching": ("crouch",), "holding": ("hold",), "thinking": ("thinking",),
               "talking": ("talk",), "presenting": ("point", "talk"), "rearing": ("rear",), "pecking": ("peck",),
               "reaching": ("reach",)}

_CHART_WORDS = {"chart", "graph", "plot", "data", "sparkline", "histogram", "kpi", "dashboard", "trend", "statistic",
                "axis", "series"}
_NEEDS = {"theme": ({"theme"}, False), "lighting": ({"lighting", "light", "lit"}, True),
          "3D material": ({"material", "3d", "render", "shader", "finish"}, True),
          "hatch": ({"hatch", "hatching", "section"}, True), "plan symbol": ({"plan", "floor", "symbol"}, True),
          "ui device": ({"screen", "mockup", "device", "frame", "app", "ui", "mock"}, True),
          "text effect": ({"text", "lettering", "title", "font", "type", "letter", "word", "headline", "logo",
                           "typography", "wordmark"}, True),
          "chart.line": (_CHART_WORDS, True), "chart.area": (_CHART_WORDS, True), "chart.bar": (_CHART_WORDS, True),
          "chart.barh": (_CHART_WORDS, True), "chart.stacked": (_CHART_WORDS, True),
          "chart.spark": (_CHART_WORDS, True), "texture.wallpaper": ({"wall", "damask", "pattern", "texture"}, True),
          "diagram.brace": ({"brace", "bracket", "diagram"}, True),
          "diagram shape": ({"node", "shape", "flowchart", "diagram", "box"}, True),
          "diagram.comparison table": ({"comparison", "compare", "matrix", "versus", "infographic", "feature",
                                        "pricing", "diagram"}, True),
          "ui.data table": ({"ui", "app", "dashboard", "screen", "data", "admin"}, True),
          "icon": ({"icon", "logo", "symbol", "glyph", "pictogram", "emoji", "app", "sheet", "ui"}, True),
          "drawing": ({"drawing", "plan", "engineering", "technical", "cad", "floor", "blueprint", "dimension",
                       "dimensioned", "isometric", "exploded", "north", "scale", "gdt", "tolerance", "bolt",
                       "flange", "stairs", "staircase", "hatch", "drafting", "floorplan"}, True)}


_SURFACE = {"texture", "material", "surface", "pattern", "wall", "floor", "fabric", "cloth", "seamless", "tile",
            "ground", "brick", "wooden", "wood"}


def _need_ok(e, slots, prefer):
    """Some entries answer only when the phrase says what they are ("dark theme", "studio lighting", "bar chart")."""
    need = _NEEDS.get(f"{e.kit}.{e.name}") or _NEEDS.get(e.kind)
    if not need:
        return True
    words, context = need
    if context and e.kit in prefer:
        return True
    return any(t in words for s in slots for t in s.toks)


_KIND_ADJ = {"backdrop": -4, "palette": -6, "lighting": -18, "theme": -18, "function": -20, "scene kind": -6, "hatch": -10,
             "plan symbol": -4, "3D material": -12, "ui device": -10, "3D shape": -8, "diagram shape": -4,
             "vector shape": -6, "sky": -2, "material": -2, "rock": 0, "prefab": -12,
             "3D character": -4}

_SCENE_WORDS = {   # kind: "word:weight ..."
    "animation": "animation:6 animated:6 gif:6 loop:3 looping:3 moving:2",
    "banner": "banner:6 hero:3 header:4 readme:6 github:3 masthead:5",
    "card": "card:4 social:3 banner:5 thumbnail:6 youtube:6 preview:2 og:2 header:2 readme:4",
    "chart": "chart:4 graph:4 plot:4 histogram:5 scatter:4 pie:3 bar:2 line:1",
    "dashboard": "dashboard:7 kpi:5 analytics:4 metric:3",
    "diagram": "diagram:4 architecture:5 flowchart:7 sequence:5 org:3 mind:2 uml:5 network:2 pipeline:2 "
               "workflow:3 oauth:2 timeline:5 roadmap:7 milestones:4 gantt:6",
    "drawing": "engineering:5 technical:3 mechanical:4 bracket:3 bearing:4 flange:4 cad:5 dimensioned:4 "
               "exploded:3 housing:2 part:2",
    "figure": "labelled:5 labeled:5 textbook:5 anatomy:4 physics:4 biology:4 science:3 scientific:3 "
              "refraction:4 cell:2 optics:4",
    "floorplan": "floor:2 plan:2 floorplan:8 apartment:3 bedroom:2 architectural:3",
    "icons": "icon:6 logo:3 glyph:4 emoji:3 pictogram:4",
    "infographic": "infographic:9 comparison:3 comparing:3 compare:3 cycle:3 explainer:3 timeline:2 statistic:2 "
                   "versus:2",
    "interior": "room:3 interior:6 kitchen:3 bedroom:2 living:2 office:2 cafe:2 nook:2",
    "isometric": "isometric:7 diorama:5 cutaway:6",
    "landscape": "landscape:5 mountain:3 mountains:3 valley:3 lake:2 river:2 sunset:2 sunrise:2 dune:3 "
                 "desert:2 scenery:3 vista:3 ocean:2 sea:2 hills:2 canyon:3 fuji:3",
    "nature": "tree:2 forest:2 flower:2 garden:3 campfire:3 meadow:3 animal:2 plant:2 park:2 pond:2 rock:1 "
              "storm:2 rain:1 snow:2 lighthouse:2 reef:3 underwater:3 coral:3 bird:1 fox:2",
    "photo": "retouch:7 edit:3 grade:4 enhance:4 photo:1 restore:5",
    "poster": "poster:7 flyer:7 cover:4 book:2 menu:5 concert:3 event:2 typography:3 magazine:5 "
              "certificate:5 invitation:5 label:2",
    "product": "product:6 packshot:6 studio:3 earbud:3 perfume:3 sneaker:3 bottle:1 headphone:2",
    "scene3d": "3d:5 render:3 lowpoly:5 poly:3 toy:2 clay:2 car:3 vehicle:3 floating:2 island:1",
    "still_life": "still:4 life:3 jug:3 fruit:2 bowl:2 vase:2 lemon:3",
    "styled": "",
    "comic": "comic:6 manga:6 strip:3 storyboard:4",
}
_UI_SCENE = "app:3 screen:3 mockup:6 mock:3 ui:5 ux:4 landing:4 website:3 login:3 onboarding:3 wireframe:5"
_SCENE_KITS = {"nature": ("flora", "elements", "creature", "texture"), "landscape": ("elements", "flora", "backdrop"),
               "poster": ("layout", "backdrop", "palette", "text"), "banner": ("layout", "backdrop", "text", "palette"), "card": ("layout", "backdrop", "text", "palette"),
               "scene3d": ("solid", "clay"), "product": ("solid", "clay"), "isometric": ("solid", "clay"),
               "interior": ("form", "solid"), "still_life": ("form", "flora"), "chart": ("chart",),
               "dashboard": ("chart",), "diagram": ("diagram", "vector"), "infographic": ("diagram", "chart"),
               "drawing": ("drafting",), "floorplan": ("drafting",), "icons": ("vector",), "ui": ("ui",),
               "figure": ("diagram",), "styled": ("stylize",)}

_COLOR_BASIC = {"red": "#d64541", "orange": "#f08a24", "yellow": "#f2c14e", "green": "#3f9b5a",
                "blue": "#3f74c0", "purple": "#7a4fb5", "pink": "#e86a9a", "brown": "#8a5a3a", "gray": "#8a8f98",
                "grey": "#8a8f98", "black": "#16181d", "white": "#f6f4ef", "gold": "#d4a63a", "silver": "#c0c6cc",
                "teal": "#2a9d8f", "navy": "#1d3557", "cyan": "#3bc6d9", "magenta": "#d23c9a", "violet": "#7f5cd6",
                "cream": "#f6efdd", "beige": "#e8dcc4", "turquoise": "#36c0b0", "crimson": "#b3243a",
                "maroon": "#6e1f2a", "indigo": "#4b4fb8", "lime": "#9ccc3c", "ginger": "#c8692a"}


_COLOR_CACHE = {}


def _colors():
    if _COLOR_CACHE:
        return _COLOR_CACHE
    names = {t for e in _index() if e.kit not in ("palette", "stylize", "texture") for t in e._toks}
    out = dict(_COLOR_BASIC)
    try:
        from . import palette as _p
        for k, v in getattr(_p, "COLOR_WORDS", {}).items():
            if " " not in k and k not in names:
                out[k] = v
    except Exception:
        pass
    for k in ("ginger", "coral", "lemon", "salmon", "peach", "olive", "sand", "sky", "wine", "honey", "cherry",
              "rose", "lavender", "mint", "moss", "chocolate", "coffee", "pine", "ocean", "forest", "plum", "rust"):
        out.pop(k, None)                    # subjects first ("coral reef", "lemons"), not colour requests
    _COLOR_CACHE.update(out)
    return out


def _norm_tokens(text):
    """Lower-case, split on non-letters, singularise."""
    if not text.isascii():
        text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    text = text.lower().replace("_", " ").replace("-", " ").replace("’", "'")
    toks = re.findall(r"[a-z0-9&]+", re.sub(r"'s\b", "", text))
    return [_sing(t) for t in toks]


def _translate_cjk(text):
    """Replace runs of CJK / kana with English words (longest dictionary match), separated as phrases."""
    table = dict(_CJK)
    for e in _index():                       # CJK keywords from the kits' own registries (styles, palettes ...)
        for k in e.keywords:
            if k and not k.isascii() and len(k) >= 2:
                table.setdefault(k, e.name.replace("_", " "))
    try:
        from . import palette as _p
        for k, v in getattr(_p, "ZH_KEYWORDS", {}).items():
            table.setdefault(k, v)
    except Exception:
        pass
    longest = max(len(k) for k in table)
    out, i, run = [], 0, False
    while i < len(text):
        ch = text[i]
        if ord(ch) < 0x2E80 and not ("　" <= ch <= "ヿ"):
            if run:
                out.append(" , ")
                run = False
            out.append(ch)
            i += 1
            continue
        for L in range(min(longest, len(text) - i), 0, -1):
            seg = text[i:i + L]
            if seg in table:
                out.append(" , " + table[seg] + " ")
                i += L
                run = True
                break
        else:
            out.append(" , " if ch in "，、。．・；：！？「」『』（）" else " ")
            i += 1
            run = True
    return "".join(out)


class _Slot:
    __slots__ = ("raw", "toks")

    def __init__(self, raw, toks):
        self.raw, self.toks = raw, toks         # toks: {token: weight}


_VOCAB = {}
_NAME_TOKS = set()


def _name_tokens():
    """Tokens of entry names ("hot" of "hot dog", "old" of "old oak"): kept even when they are common words."""
    if not _NAME_TOKS:
        _NAME_TOKS.update(t for e in _index() if len(e._toks) > 1 for t in e._toks if t not in _FUNCTION)
    return _NAME_TOKS


def _vocab():
    if not _VOCAB:
        for e in _index():
            for t in e._toks + tuple(t for a in e._alias_toks for t in a) + tuple(e._kw):
                _VOCAB[t] = 1
    return _VOCAB


def _slots(words, colors):
    """Tokens of one phrase with synonyms (0.9) and typo fixes (0.75); colours kept at a low weight."""
    vocab = _vocab()
    slots = []
    for raw in words:
        for t in _norm_tokens(raw):
            if not t or t.isdigit():
                continue
            if t in _STOP_SCORE and t not in _SYN and t not in _name_tokens():
                continue
            toks = {t: 0.35 if t in colors else 1.0}
            for s in _SYN.get(t, "").split():
                toks.setdefault(_sing(s), 0.9)
            if t not in colors:
                fix = _typo(t, vocab)
                if fix:
                    toks.setdefault(fix, 0.75)
                    for s in _SYN.get(fix, "").split():
                        toks.setdefault(_sing(s), 0.7)
            slots.append(_Slot(raw, toks))
    return slots


def _edit_distance(a, b, cap):
    """Damerau-Levenshtein distance (adjacent swaps cost 1), or cap + 1 as soon as it exceeds ``cap``."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
            if prev2 is not None and i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > cap:
            return cap + 1
        prev2, prev = prev, cur
    return prev[-1]


_TYPO_BY_FIRST = {}


def _derived(a, b):
    """``a`` is ``b`` with a suffix (balancer / balance, painter / paint) or the other way round: a word, no typo."""
    short, long_ = sorted((a, b), key=len)
    return long_.startswith(short) and long_[len(short):] in ("r", "er", "s", "es", "d", "ed", "ing", "ly", "y",
                                                              "ist", "ism", "ness", "en", "n", "e")


def _typo(t, vocab):
    """The one vocabulary word a misspelt token stands for, or None. Typos only: same first letter, at least 5
    letters, edit distance 1 (2 from 9 letters); a word the lexicon knows is never re-spelt (baker is not beaker,
    tram is not train), and two equally close words mean no answer."""
    if len(t) < 5 or not t.isalpha() or t in vocab or _known(t):
        return None
    if not _TYPO_BY_FIRST:
        for v in vocab:
            if v.isalpha() and len(v) >= 4:
                _TYPO_BY_FIRST.setdefault(v[0], []).append(v)
    cap = 2 if len(t) >= 9 else 1
    best, hits = cap + 1, []
    for v in _TYPO_BY_FIRST.get(t[0], ()):
        if abs(len(v) - len(t)) > cap:
            continue
        d = _edit_distance(t, v, cap)
        if d < best:
            best, hits = d, [v]
        elif d == best:
            hits.append(v)
    hits = [h for h in hits if _sing(h) != _sing(t) and not _derived(t, h)]
    return hits[0] if best <= cap and len(set(_sing(h) for h in hits)) == 1 else None


_FUNCTION = set("a an the of and or with in on at by for to from into onto over under".split())


def _prepare(e):
    toks = tuple(_norm_tokens(e.name))
    core = tuple(t for t in toks if t not in _FUNCTION)
    e._toks = core or toks
    e._alias_toks = tuple(tuple(_norm_tokens(a)) for a in e.aliases)
    kw = set()
    for k in e.keywords:
        if k.isascii():
            kw.update(_norm_tokens(k))
    kw.update(_norm_tokens(e.kind))
    e._kw = frozenset(kw - _STOP - set(e._toks))
    e._doc = frozenset(set(_norm_tokens(e.summary)) - _STOP - kw - set(e._toks))


def _seq_score(seq, slots):
    """1.0 when every token of ``seq`` is in the phrase in order, 0.85 any order, partial below; times weights."""
    if not seq:
        return 0.0
    found, pos = [], []
    for t in seq:
        best, where = 0.0, -1
        for i, s in enumerate(slots):
            w = s.toks.get(t, 0.0)
            if w > best:
                best, where = w, i
        found.append(best)
        pos.append(where)
    hit = [f for f in found if f > 0]
    if not hit:
        return 0.0
    if len(hit) == len(seq):
        w = min(found)
        ordered = all(b > a for a, b in zip(pos, pos[1:])) or len(set(pos)) == 1
        return w * (1.0 if ordered else 0.85)
    if seq[-1] == "leaf" or len(seq) > 1 and found[-1] == 0:   # missing head noun: a different thing
        return 0.35 * sum(hit) / len(seq)
    return 0.5 * sum(hit) / len(seq)


def _score(e, slots):
    name = _seq_score(e._toks, slots)
    alias = max((_seq_score(a, slots) for a in e._alias_toks), default=0.0)
    base = max(100 * name + (6 * (len(e._toks) - 1) if name >= 0.85 else 0),
               90 * alias + (6 * (max(len(a) for a in e._alias_toks) - 1) if alias >= 0.85 else 0))
    kw = doc = 0.0
    used = 0
    for s in slots:
        best_kw = max((w for t, w in s.toks.items() if t in e._kw), default=0.0)
        best_doc = max((w for t, w in s.toks.items() if t in e._doc), default=0.0)
        in_name = any(t in e._toks or any(t in a for a in e._alias_toks) for t in s.toks)
        if in_name:
            used += 1
        elif best_kw:
            kw += 30 * best_kw
            used += 1
        elif best_doc:
            doc += 8 * best_doc
    if base == 0 and kw == 0:
        return 0.0
    score = base + min(kw, 75) + min(doc, 16) + _KIND_ADJ.get(e.kind, 0)
    score -= 4 * (len(slots) - used) if base else 0
    return score


# --------------------------------------------------------------------------------------------- the request
# SKILL.md's kit table is the one source for "which kind does this kit's subject belong to"; it is read at run time
# so the page and the answer cannot drift. The built-in copy answers when the page is not next to the package.
_SKILL_PAGE = os.path.join(RUNTIME, "skills", "pixelcanvas", "SKILL.md")
_TABLE_DEFAULT = {"figure": ["people", "comic"], "creature": ["animals"], "flora": ["nature"],
                  "elements": ["nature"], "form": ["still_life", "interior"],
                  "solid": ["product", "scene3d", "isometric"], "clay": ["characters", "pets", "scene3d"],
                  "scenery": ["landscape"], "text": ["poster", "card"], "layout": ["poster", "card"],
                  "diagram": ["diagram", "figure"], "chart": ["chart", "dashboard", "infographic"],
                  "ui": ["ui"], "drafting": ["drawing", "floorplan"], "vector": ["icons"],
                  "backdrop": ["wallpaper"], "palette": ["wallpaper"]}
_KIT_TABLE = {}


def kit_kinds():
    """{kit: [kinds]} from the kit table of skills/pixelcanvas/SKILL.md (a built-in copy when the page is absent)."""
    if _KIT_TABLE:
        return _KIT_TABLE
    table = {}
    try:
        with open(_SKILL_PAGE, encoding="utf-8") as f:
            sect = f.read().split("\n## Kits", 1)[1].split("\n## ", 1)[0]
        for ln in sect.splitlines():
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if not ln.lstrip().startswith("|") or len(cells) < 4:
                continue
            kinds = re.findall(r"`([a-z0-9_]+)`", cells[3])
            for kit in re.findall(r"`([a-z0-9_]+)`", cells[1]):
                for k in kinds:
                    if k not in table.setdefault(kit, []):
                        table[kit].append(k)
    except (OSError, IndexError):
        table = {}
    _KIT_TABLE.update(table or _TABLE_DEFAULT)
    return _KIT_TABLE


_KIT_KIND_DEFAULT = {"pictograms": "icons", "clay": "scene3d", "scenery": "landscape", "texture": None,
                     "stylize": None, "text": "poster", "solid": "scene3d"}
# word votes that choose between the kinds of one table row (the subject's meaning class votes too: "person", "animal")
_ROW_WORDS = {"comic": "comic:6 manga:6 cartoon:3 strip:3 panel:3", "people": "person:1",
              "characters": "person:12 character:4 people:3 human:3 figure:1",
              "pets": "animal:12 pet:4 bird:2 fish:2",
              "interior": "room:3 interior:6 kitchen:3 bedroom:2 living:2 office:2 cafe:2 nook:2 shelf:2 shelves:2 "
                          "jar:1",
              "still_life": "still:4 life:3 jug:3 fruit:2 bowl:2 vase:2 lemon:3 table:1",
              "product": "product:6 packshot:6 studio:3 earbud:3 perfume:3 sneaker:3 bottle:1 headphone:2",
              "scene3d": "3d:5 render:3 lowpoly:5 poly:3 toy:2 clay:2 car:3 vehicle:3 floating:2 island:1 cartoon:1",
              "isometric": "isometric:7 diorama:5 cutaway:6"}

_FORMAT_KINDS = ("animation", "comic", "banner", "card", "chart", "dashboard", "diagram", "drawing", "figure", "floorplan", "icons",
                 "infographic", "isometric", "photo", "poster", "product", "scene3d", "still_life", "wallpaper", "ui")
_ABSTRACT_HEADS = {"palette", "theme", "scheme", "color", "colour", "style", "background", "layout", "font"}
# words of a later phrase that name what the picture is for ("for a biology textbook", "for my GitHub README")
_PURPOSE = {"textbook", "worksheet", "readme", "github", "infographic"}
_NO_RECIPE_FORMATS = ("diagram", "chart", "dashboard", "ui", "drawing", "floorplan", "figure", "infographic")
_FORMAT_KITS = ("layout", "diagram", "chart", "drafting", "ui", "text", "vector")
_FORMAT_EXTRA = {"wallpaper": "wallpaper:6 lockscreen:4", "figure": "figure:3 textbook:2",
                 "drawing": "numbered:1 parts:2"}
_WRAPPER = set("""shot packshot closeup painting sketch print render photo sheet set collection version scene drawing illustration picture mascot
window woodblock engraving mural portrait artwork""".split())
_FORM_WORDS = {"silhouette", "outline"}
_SKIP_HEAD = {"up", "down", "out", "off", "away", "seen", "shown", "held", "made", "set", "pulled", "lit", "cut", "built", "filled", "full", "style"}
_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "eleven": 11, "twelve": 12, "dozen": 12}
_SETTING = {
    "landscape": "mountain valley lake river sea ocean beach shore coast dune desert hill canyon cliff headland island "
                 "road path alley lane highway pier jetty dock wharf boardwalk landscape scenery vista seascape "
                 "ridge plain prairie savanna tundra "
                 "village town city street skyline harbor harbour field meadow countryside horizon wave lagoon",
    "interior": "room kitchen office cafe nook bedroom bathroom library workshop classroom",
    "nature": "garden forest wood jungle park reef underwater pond swamp grove orchard lawn yard backyard"}
_SETTING_KIND = {_sing(w): k for k, ws in _SETTING.items() for w in ws.split()}
_SUBJECT_KINDS = ("people", "comic", "animals", "characters", "pets", "nature", "landscape", "still_life", "interior", "scene3d",
                  "isometric", "product", "wallpaper")
_ATMOS = set("""rain snow fog mist glow sparks embers bokeh god_rays window_light wet_ground droplets lightning storm
caustics puddle ripples lens_flare heat_haze volumetric_fog dust_motes spotlight breath splash""".split())

# meaning classes of things that have no entry of their own: class -> (words, nearest entry names, icon names, kind)
_CLASSES = {
    "instrument": ("sax saxophone trumpet trombone clarinet flute violin cello piano keyboard drum harp banjo "
                   "accordion microphone synthesizer synth ukulele tuba oboe horn harmonica zither lute sitar "
                   "mandolin lyre bagpipe xylophone marimba tambourine gong organ koto erhu guzheng shamisen "
                   "bassoon cymbal triangle", "harp guitar piano", "music mic headphones", "still_life"),
    "audio": ("gramophone phonograph jukebox cassette walkman boombox stereo hifi", "turntable vinyl_record speaker "
              "radio", "music headphones", "still_life"),
    "person": ("baker butcher barber tailor waiter waitress cashier mechanic plumber electrician carpenter sailor "
               "fisherman soldier guard king queen prince clown juggler magician acrobat referee coach judge lawyer "
               "priest monk nun vendor merchant shopkeeper florist librarian pilot driver cyclist skier surfer "
               "swimmer climber gamer singer guitarist drummer pianist violinist band trio quartet duo choir "
               "orchestra mother father mom dad son daughter brother sister twin bride groom tourist traveller "
               "traveler cowboy ninja samurai witch vampire fairy mermaid elf", "person musician chef",
              "user users", "people"),
    "timepiece": ("sundial hourglass clock watch stopwatch metronome", "clock hourglass alarm_clock", "clock",
                  "still_life"),
    "ornament": ("birdbath gnome fountain statue sculpture totem obelisk monument", "fountain statue", "star",
                 "nature"),
    "textile": ("yarn wool thread rope string twine ribbon quilt blanket pillow cushion cloth tablecloth towel "
                "napkin curtain", "rug ball", "heart",
                "still_life"),
    "vehicle": ("bus tram train truck van bike motorcycle scooter boat ship yacht canoe sailboat plane airplane "
                "helicopter submarine tractor taxi ambulance cart wagon caravan sled ferry kayak raft", "car truck bicycle "
                "rocket", "plane map pin", "scene3d"),
    "building": ("tower castle lighthouse church temple mosque bridge barn windmill cabin hut tent skyscraper palace "
                 "pagoda gate fort station cottage shed igloo stall kiosk booth shop store market bakery restaurant "
                 "diner bar pub hotel inn cafe teahouse garage", "shop house", "home pin", "landscape"),
    "furniture": ("bed desk shelf cabinet wardrobe bench couch dresser drawer stool cradle sideboard fridge oven "
                  "windowsill sill counter mantelpiece mantel "
                  "stove sink bathtub", "chair sofa table bookshelf bed", "home", "interior"),
    "food": ("pizza burger bread croissant cake sushi noodle soup egg cheese donut cookie pie pasta salad taco "
             "vegetable veggie pastry dessert snack meal "
             "pancake waffle muffin pastry bagel vegetable carrot tomato broccoli onion corn mushroom", "cake sandwich apple mug", "gift", "still_life"),
    "tool": ("hammer wrench saw axe shovel screwdriver scissors pen pencil brush key ladder pliers drill spade "
             "rake knife fork", "sword paintbrush", "gear edit key", "still_life"),
    "clothing": ("hat jacket coat dress shirt shoe boot scarf glove raincoat sneaker cap sweater skirt trouser "
                 "helmet crown", "person chef", "user", "still_life"),
    "sport": ("football soccer basketball tennis racket racquet surfboard ski skate dumbbell", "ball skateboard",
              "target flag", "still_life"),
    "device": ("earbud headphone camera watch speaker tv television monitor remote console joystick radio "
               "microphone drone", "phone laptop speaker", "phone headphones monitor", "product"),
    "container": ("jar bottle pot pan kettle teapot basket bucket barrel crate box chest vase mug bowl case", "mug vase "
                  "flask cup basket", "gift", "still_life"),
    "landform": ("island peninsula volcano iceberg cave glacier archipelago atoll", "mountains rock", "pin",
                 "landscape"),
    "animal": ("camel llama alpaca donkey moose bison buffalo yak kangaroo koala monkey gorilla hippo rhino panda "
               "otter beaver seal walrus hedgehog bat hamster raccoon", "horse deer cow", "smile", "animals"),
    "bird": ("peacock ostrich seagull gull pigeon dove hawk falcon heron crane stork pelican toucan kingfisher "
             "hummingbird bird", "bird parrot crow flamingo", "smile", "animals"),
    "sea creature": ("anemone crab lobster starfish seahorse stingray squid shrimp coral oyster eel", "fish "
                     "jellyfish octopus", "drop", "animals"),
    "plant": ("bougainvillea ivy vine cactus fern moss orchid wisteria jasmine hydrangea geranium", "daisy bush "
              "potted_monstera", "bolt", "nature"),
    "toy": ("teddy doll kite balloon puzzle lego", "ball", "gift", "product"),
}
_CLASS_OF = {_sing(w): c for c, (ws, *_r) in _CLASSES.items() for w in ws.split()}
# the closest relatives of a thing without an entry, before the rest of its class
_KIN = {"vegetable": "carrot tomato corn", "veggie": "carrot tomato corn", "fruit": "apple orange banana",
        "pastry": "croissant cake donut", "gramophone": "turntable vinyl_record", "phonograph": "turntable vinyl_record", "jukebox": "speaker radio",
        "zither": "harp guitar", "lute": "guitar harp", "sitar": "guitar", "mandolin": "guitar", "koto": "harp",
        "guzheng": "harp", "lyre": "harp", "banjo": "guitar", "ukulele": "guitar", "sundial": "clock hourglass",
        "boombox": "radio speaker", "cassette": "radio", "yarn": "ball", "baker": "chef", "trio": "musician",
        "band": "musician", "quartet": "musician", "duo": "musician", "choir": "person", "stall": "shop",
        "kiosk": "shop", "booth": "shop", "bakery": "shop", "cafe": "shop", "windowsill": "window",
        "sill": "window", "case": "suitcase briefcase", "bridge": "bridge_(arch)",
        "weather": "sun cloud umbrella", "forecast": "sun cloud", "task": "checklist clipboard",
        "todo": "checklist clipboard", "chore": "checklist", "note": "notes notepad", "reminder": "calendar bell",
        "schedule": "calendar", "armchair": "armchair sofa chair", "stool": "chair", "recliner": "armchair sofa",
        "produce": "carrot tomato corn", "greens": "lettuce broccoli cabbage", "picnic": "picnic_blanket basket",
        "toolbox": "toolbox wrench hammer", "seagull": "bird", "gull": "bird", "koi": "goldfish fish"}
_KIN["vegetable"] = _KIN["veggie"] = "carrot tomato eggplant broccoli corn pepper"

# --------------------------------------------------------------- the request's grammar
# participles and verbs that say what a subject does: they end its phrase and become its pose
_VERBS = set("""sitting standing lying sleeping running walking jumping leaping flying soaring swimming waving
pointing dancing kneeling cheering grazing sniffing crouching holding thinking talking presenting rearing pecking
reaching perched perching watching reading playing looking wearing carrying eating drinking riding chasing hugging
facing climbing singing cooking baking typing writing using feeding throwing catching kicking selling serving
waiting smiling laughing crying resting leaning sipping hiding staring gazing listening pulling pushing peeking
curled galloping trotting begging clapping bowing stretching yawning napping lounging basking strolling skating
surfing hiking cycling rising flowing floating hovering glowing burning drifting circling soaring blowing pouring
spilling hanging dangling crossing heading leaving arriving approaching guarding juggling showing saying says falling
dropping cutting shining sparkling coming going peering wandering roaming nesting
reads holds sits stands waves shows wears carries playing""".split())
_POSE_WORDS.update({"reading": ("reading", "read"), "typing": ("typing",), "playing": ("play_bow", "play"),
                    "carrying": ("carry", "hold"), "hugging": ("hug",), "clapping": ("clap",),
                    "throwing": ("throw",), "kicking": ("kick",), "bowing": ("bow",), "begging": ("beg",),
                    "resting": ("rest", "lie"), "leaning": ("lean",), "napping": ("sleep", "lie"),
                    "lounging": ("lie", "sit"), "waiting": ("stand",), "looking": ("look_up",),
                    "running": ("run", "gallop", "trot"), "sits": ("sit",), "stands": ("stand",),
                    "waves": ("wave",), "holds": ("hold",), "carries": ("carry", "hold"), "reads": ("reading", "read"),
                    "stretching": ("reach_up", "reach"), "guarding": ("stand",), "juggling": ("throw",)})
# hyphenated modifiers: "lantern-lit", "toy-like", "three-quarter" describe a subject and never are one
_COMPOUND_TAILS = set("""lit like style styled shaped colored coloured themed inspired sized made covered filled free
based eyed haired tailed legged powered painted topped lined framed faced looking toned tone sided quarter bit
time up ish""".split())
# words that steer people, animals and objects to 3D clay / solid or to the flat 2D kits
_STEER_3D = {"3d", "toy", "clay", "figurine", "plasticine", "vinyl", "render", "rendered", "isometric", "claymation",
             "cgi", "blender", "lowpoly", "poly", "polygon", "diorama", "miniature", "plush"}
_STEER_2D = {"2d", "flat", "vector", "illustration", "illustrated", "cartoon", "anime", "line", "storybook",
             "sticker", "comic", "manga", "doodle", "drawn", "editorial", "kawaii", "chibi", "picture-book"}
_STEER_2D.discard("line")
_NOT_STEER = {"clay": {"pot", "jar", "vase", "bowl", "pipe", "tile", "court", "soil", "oven", "brick", "mug", "cup",
                       "plate", "pottery", "roof", "wall", "path"},
              "vinyl": {"record", "records", "lp", "album", "disc", "floor", "flooring", "sticker", "banner"},
              "line": {"chart", "graph", "up", "of", "ray", "rays"}}
# words that make a phrase a flat mark (pictogram / icon) rather than an illustration
_FLAT_WORDS = {"silhouette", "icon", "pictogram", "symbol", "sign", "logo", "glyph", "emoji", "outline", "badge",
               "pictograph", "mark"}
# heads that are events or times of a thing: the noun before them is the subject ("kite festival" -> kite)
_EVENTS = {"festival", "party", "fair", "event", "celebration", "parade", "ceremony", "day", "season", "holiday",
           "race", "ride", "show", "contest", "competition", "tournament", "class", "lesson", "club", "night",
           "session", "meetup", "jam", "week", "weekend"}
# collectives: "a team of five coworkers" -> five coworkers
_FLOCKS = {"school", "flock", "herd", "swarm", "shoal", "pack", "pod"}
_COLLECTIVE = {"caravan": 0, "crowd": 0, "team": 0, "group": 0, "pair": 2, "couple": 2, "bunch": 0, "handful": 0, "trio": 3, "duo": 2,
               "quartet": 4, "set": 0, "family": 0, "row": 0, "line": 0, "pile": 0, "stack": 0}
_MEASURE = {"ball", "cup", "mug", "glass", "bowl", "plate", "slice", "loaf", "jar", "bottle", "pot", "spoonful",
            "bucket", "bag", "box", "basket", "can", "carton", "tin", "piece", "bar", "scoop", "sack"}
_MASS = {"yarn", "wool", "thread", "coffee", "tea", "water", "milk", "wine", "beer", "juice", "soup", "rice", "sand",
         "paper", "gold", "ice", "cheese", "flour", "sugar", "salt", "honey", "jam", "chocolate", "cream", "string",
         "light", "smoke", "fire", "flame", "noodle", "oil", "paint", "ink", "clay", "soil", "dirt", "snow", "dust"}
_GROUP_COUNT = {"duo": 2, "pair": 2, "couple": 2, "trio": 3, "quartet": 4, "quintet": 5, "twin": 2}
# nouns that name no thing to draw: measures, qualities, text, abstractions
_ABSTRACT = set("""height size colour color shape quality detail type kind theme mood distance angle age variety
genre jazz music sound style vibe feel atmosphere lighting composition perspective focus
command credit tagline name caption fact message note number price item
section column header footer list step value word letter headline subheadline greeting date location lineup
dimension area data model language scaling query verifier index law formula equation contrast tone hint
center centre order relative edge layer border shadow surface
typography font serif quote review author post blog article story podcast brand company startup business project
service suite tool app software platform website product novel movie film series episode headline tagline title
caption topic subject idea concept habit tip lesson consistent outlined shading air line lines reflection highlight outline grain noise
detail contour""".split())
# nouns after which a compound names no thing ("a key fact", "lead lines", "a sub-headline", "visible grain")
_ABSTRACT_COMPOUND = set("""fact line name number value data model idea law formula equation word letter headline
subheadline tagline title caption quote grain noise detail type kind style tone sound""".split())
_ADJECTIVES = set("""cosy cozy layered overhead japanese english chinese french italian spanish german greek
mediterranean nordic scandinavian tropical rustic quaint vintage retro futuristic friendly cheerful surprised happy sad
angry sleepy lazy curious playful gentle fluffy fuzzy shiny glossy matte translucent transparent wooden metallic
golden silver ancient medieval victorian modern elegant fancy plain simple tiny huge giant massive distant nearby
overhead underneath upstairs downstairs outdoors indoors inside outside above below beyond everywhere somewhere
diverse different various multiple several numerous colorful colourful vibrant pastel muted moody dramatic
serene peaceful calm busy crowded empty lonely quiet loud bold subtle relative early late next last first second
asleep awake alone fresh
third final open closed real fake whole entire""".split())
_SEASONS = {"autumn", "fall", "spring", "summer", "winter", "season"}
_ADJ_SUFFIX = ("ese", "ish", "ful", "ous", "less", "able", "ible", "ical", "ly")
_GENERIC = {"animal", "pet", "creature", "thing", "object", "item", "stuff", "character", "critter", "beast"}
_SYN.update({"baker": "chef", "cook": "chef", "trio": "musician", "band": "musician", "quartet": "musician",
             "duo": "musician", "kitten": "cat", "dinosaur": "tyrannosaurus dinosaur", "dino": "dinosaur", "kittie": "cat", "doggo": "dog",
             "fuji": "mountains", "streetcar": "tram", "spray": "splash", "sea spray": "splash", "shaft": "god rays", "ray": "god rays", "grandmother": "grandma", "granny": "grandma", "grandfather": "grandpa",
             "grandad": "grandpa", "children": "kid", "fisherman": "person", "mermaid": "person", "lp": "vinyl record", "record player": "turntable"})
# extra lexicon: common words that are not catalogue names (never re-spelt as a lookalike)
_LEXICON = set("""baker bakery maker taker later water wafer paper piper viper wiper diner dinner liner miner poker
joker biker mover lover cover tower power mower rower boxer mixer fixer sitter setter letter better butter batter
bitter litter ladder larder planet plane player prayer layer lawyer mayor manor minor motor rotor actor tutor sailor
tailor trailer trawler beaker baker bonnet basket blanket bucket button carpet castle cattle cradle candle sandal
saddle paddle puddle needle noodle poodle pebble marble barrel kernel tunnel funnel channel panel parcel pencil
stencil tinsel vessel tassel hostel hotel motel label cable table stable fable gable maple apple ample temple simple
tram trio train trail trial trolley stall still stile style smile skill shell spell spill steel steal stool stoop
stove store storm story stork stone stoke stake snake shake shade shape share spare spade space spice slice
splice price prize pride bride bridge ridge fridge hedge wedge ledge badge judge fudge nudge lodge dodge
sundial dial denial medal metal petal pedal portal mortal total vocal local focal coral moral mural rural
canal camel cancel tassel hazel easel weasel diesel chisel muzzle puzzle nozzle tower flower power shower
glower lower hour flour floor flood blood brood broom bloom gloom groom grove glove globe probe robe lobe
night light sight fight might right tight bright fright flight knight weight height eight freight
monster lobster hamster toaster coaster poster roster master plaster faster
garden warden harden burden border order corder murder powder ladder
string strong strand stand strap straw stray tray gray grey prey pray
jacket racket rocket pocket socket locket ticket thicket cricket wicket picket
silver sliver shiver liver river fiver giver
""".split())


def _known(t):
    """A word of the lexicon: an entry word, a class member, a synonym, a pose, a setting, a colour, a stop word."""
    return (t in _LEXICON or t in _CLASS_OF or t in _SYN or t in _STOP or t in _VERBS or t in _POSE_WORDS
            or t in _SETTING_KIND or t in _ABSTRACT or t in _COLOR_BASIC or t in _STEER_3D or t in _STEER_2D
            or t in _EVENTS or t in _FLAT_WORDS or t in _KIN)
_RECIPES = {
    "silhouette": (
        "silhouette from a curve: bezier_points + a tapering stroke (masks.line_mask for a constant width)",
        ["pts = masks.bezier_points((0.40, 0.15), (0.62, 0.28), (0.52, 0.60), (0.44, 0.86))   # control points as fractions",
         "m = brush.stroke(np.zeros((cv.h, cv.w), 'float32'), [(x * cv.w, y * cv.h) for x, y in pts], "
         "r=0.03 * cv.h, r_end=0.01 * cv.h, hardness=1.0)",
         "cv.paint('#10162b', m)"]),
    "sdf": (
        "shape from SDF primitives: sdf.sd_* + smooth_union + form.paint_form for volume",
        ["X, Y = cv.xy(); S = cv.h / 1080",
         "d = sdf.smooth_union(sdf.sd_ellipse(X, Y, 0.5 * cv.w, 0.45 * cv.h, 150 * S, 90 * S), "
         "sdf.sd_box(X, Y, 0.5 * cv.w, 0.62 * cv.h, 40 * S, 110 * S, radius=12 * S), k=30 * S)",
         "m = form.paint_form(cv, d, '#c8a24a', material='metal')"]),
    "3d": (
        "3D: solid primitives blended with smooth_union",
        ["body = solid.smooth_union(solid.ellipsoid(0.6, 0.35, 0.35), solid.capsule((0.3, 0, 0), (0.9, 0.5, 0), 0.12), "
         "solid.sphere(0.18).place(x=0.9, y=0.55), k=0.08)",
         "sc = solid.Scene(lighting='studio'); sc.add(body.place(x=0, z=0), '{name}', 'plastic')",
         "sc.camera = solid.Camera.frame(sc); sc.render(cv)"]),
}
_RECIPE_KINDS = {"scene3d": "3d", "product": "3d", "isometric": "3d"}
_REGISTER = "manifest.note(cv, 'object', '{name}', mask=m)    # register the shape for --check"
_REGISTER_3D = "# sc.add(body, '{name}') already names it for --check"


def recipe(name, kind=None, silhouette=False):
    """(title, three snippet lines, register line) for drawing ``name`` when no entry exists."""
    key = "silhouette" if silhouette else _RECIPE_KINDS.get(kind, "sdf")
    title, lines = _RECIPES[key]
    reg = _REGISTER_3D if key == "3d" else _REGISTER
    return title, [ln.replace("{name}", name) for ln in lines], reg.replace("{name}", name)


# ------------------------------------------------------------------ content: text the picture must carry
_MONTHS = "jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|" \
          "oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
_DAYS = "mon(?:day)?|tue(?:s(?:day)?)?|wed(?:nesday)?|thu(?:r(?:s(?:day)?)?)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?"
_TIME = r"(?:[01]?\d|2[0-3]):[0-5]\d(?:\s?[AaPp][Mm])?|\b\d{1,2}\s?[AaPp][Mm]\b"
_CONTENT_RE = [
    ("url", re.compile(r"(?:https?://|www\.)\S+|\b[\w.-]+@[\w-]+\.[\w.]+")),
    ("quote", re.compile(r"(?<![A-Za-z])(['\"“‘])([^'\"”’]{1,160}?)(['\"”’])(?![A-Za-z])")),
    ("price", re.compile(r"[$€£¥]\s?\d(?:[\d,]*\d)?(?:\.\d+)?|\b\d+(?:\.\d+)?\s?(?:USD|EUR|GBP|dollars?|euros?)\b")),
    ("date", re.compile(r"\b(?:(?:%s)\.?,?\s+)?(?:\d{1,2}[/.]\d{1,2}(?:[/.]\d{2,4})?(?:\s?[–-]\s?\d{1,2}[/.]\d{1,2})?|"
                        r"(?:%s)\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?|\d{1,2}(?:st|nd|rd|th)?\s+(?:%s)\b(?:,?\s+\d{4})?|"
                        r"\d{4}-\d{2}-\d{2})" % (_DAYS, _MONTHS, _MONTHS), re.I)),
    ("time", re.compile(r"(?:\b(?:%s)\b\s+)?(?:%s)(?:\s?[–-]\s?(?:%s))?" % (_DAYS, _TIME, _TIME), re.I)),
    ("day", re.compile(r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)s?\b")),
]
_NAME_RE = re.compile(r"\b[A-Z][\w’']*(?:\s+(?:[A-Z][\w’']*|&|of))*\s+[A-Z][\w’']*\b")
_KEEP_NAME = set("mount mt lake river milky great golden north south sea ocean big little new old".split())
_DET = {"a", "an", "the", "this", "that", "some", "his", "her", "their", "its"}


def extract_content(query):
    """(clean request, [text to draw]): quoted strings, dates, times, prices, URLs and capitalised multi-word names
    (``The Velvet Room``) are content of the picture, not subjects; they are cut out of the request."""
    found = []                                                       # (position, text)
    text = query

    def cut(rx, pick=0):
        nonlocal text

        def repl(m):
            found.append((m.start(), m.group(pick)))
            return " , " + " " * (len(m.group(0)) - 3) if len(m.group(0)) >= 3 else " , "
        text = rx.sub(repl, text)

    for kind, rx in _CONTENT_RE:
        if kind == "quote":
            cut(rx, 2)
        else:
            cut(rx)
    for m in list(_NAME_RE.finditer(text)):
        words = m.group(0).split()
        before = text[:m.start()].rstrip()
        sentence_start = not before or before[-1] in ".!?:;,\n·—–-" or before.endswith(",")
        if sentence_start and (words[0].lower() in _DET or len(words) == 2 or not before or before[-1] in ".!?"):
            continue
        if any(w.lower() in _KEEP_NAME for w in words):
            continue
        if before and before[-1] in ",:;" and words[0].lower() in _DET:
            continue
        found.append((m.start(), m.group(0)))
        text = text[:m.start()] + " , " + " " * max(0, len(m.group(0)) - 3) + text[m.end():]
    items, seen = [], set()
    for _pos, s in sorted(found):
        s = s.strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            items.append(s)
    return text, items


# ------------------------------------------------------------------ phrases
def _chunks2(query):
    """Noun phrases of a request in order: [(words, word that introduced the phrase)]."""
    q = _translate_cjk(query)
    q = re.sub(r"\b[Aa] (?:pair|couple) of\b", "two", q)
    q = re.sub(r"\b[Aa] dozen\b", "twelve", q)
    q = q.replace("next to", ",").replace("in front of", ",")
    q = re.sub(r"(?<=[A-Za-z])/(?=[A-Za-z])", "-", q)
    out = []
    for part in re.split(r"[,.;:!?()\[\]/|\n·]+|\s+[-–—]\s+", q):
        cur, prev = [], ""
        for raw in re.findall(r"[A-Za-z0-9&'’\-]+", part):
            low = raw.lower().strip("'’-")
            if low in _SPLIT:
                if cur:
                    out.append((cur, prev))
                cur, prev = [], low
                continue
            if low.endswith(("'s", "’s", "s'", "s’")) and len(low) > 3:        # a possessive modifies, it is no subject
                continue
            cur.append(raw)
        if cur:
            out.append((cur, prev))
    return out


def _toks(words):
    return [t for w in words for t in _norm_tokens(w) if t]


def _head(words, vocab=None):
    """The head noun of a phrase: its last word that is not a participle, number, colour, stop or form word."""
    vocab = vocab if vocab is not None else _vocab()
    toks = _toks(words)
    colors = _colors()
    if toks and toks[-1] in _ROUND:                          # "three red oranges": the oranges are fruit
        colors = {k: v for k, v in colors.items() if k != toks[-1]}
    cands = []
    for t in toks:
        if t in _STOP or t.isdigit() or t in _SKIP_HEAD or t in _NUMBERS or t in colors or t in _ABSTRACT:
            continue
        if t in _ADJECTIVES or t in _SEASONS or (t.endswith(_ADJ_SUFFIX) and t not in vocab and not _known(t)):
            continue                                       # "cosy", "Japanese", "in autumn": no thing to draw
        if t.endswith("ing") and t not in vocab and t not in _CLASS_OF and t not in _SETTING_KIND:
            continue
        if t.endswith("ed") and len(t) > 4 and t not in _CLASS_OF and t not in _SETTING_KIND and t not in _LEXICON:
            continue
        cands.append(t)
    if any(t not in _STEER_WORDS for t in cands):            # "a toy-like 3D baker": the baker; "a toy": the toy
        cands = [t for t in cands if t not in _STEER_WORDS]
    cands = [t for t in cands if len(t) > 1]
    last = [t for t in toks if t not in _STOP and not t.isdigit() and t not in _NUMBERS and t not in colors]
    if last and last[-1] in _ABSTRACT_COMPOUND and (not cands or cands[-1] != last[-1]):
        return None
    plain = [t for t in cands if t not in _FORM_WORDS and t not in _FLAT_WORDS]
    if plain and cands and cands[-1] in _FLAT_WORDS:          # "a bicycle icon": the bicycle
        cands = plain
    content = [t for t in toks if t not in _STOP and not t.isdigit() and t not in _NUMBERS and t not in colors]
    if cands and cands[-1] in _EVENTS and len(content) >= 2 and content[-1] == cands[-1] and \
            content[-2] in _ABSTRACT:
        return None                                          # "a jazz night": the event, no thing to draw
    if len(cands) >= 2 and cands[-1] in _EVENTS:                 # "a kite festival": the kites
        cands = cands[:-1]
    cands = [t for t in cands if t not in _FORM_WORDS] or cands
    flock = [t for t in cands if t in _FLOCKS]
    return flock[0] if flock else cands[-1] if cands else None


def _count_of(words):
    for i, w in enumerate(words[:3]):
        lw = w.lower()
        if lw in _NUMBERS and lw != "dozen":
            return _NUMBERS[lw]
        nxt = words[i + 1].lower() if i + 1 < len(words) else ""
        if lw.isdigit() and 1 < int(lw) <= 24 and nxt not in ("x", "mm", "cm", "m", "km", "kg", "g", "px", "%"):
            return int(lw)
    for w in words:
        if _sing(w.lower()) in _GROUP_COUNT:
            return _GROUP_COUNT[_sing(w.lower())]
    return 1


def _wrapper_chunk(words):
    """A phrase that only names the medium or the sheet ("pencil sketch", "stained glass window", "sheet")."""
    toks = _toks(words)
    if not toks:
        return True
    if all(t in _STOP or t in _WRAPPER or t in _STEER_WORDS or t in _FORMAT_HEADS or t in _ADJECTIVES
           for t in toks) and any(t in _WRAPPER or t in _STEER_WORDS or t in _FORMAT_HEADS for t in toks):
        return True                                       # "a flat vector illustration", "a cute vinyl toy"
    style = _pick_style(" ".join(words))
    return (any(t in _WRAPPER for t in toks)
            and all(t in _STOP or t in _WRAPPER or t in _colors() or t.isdigit() or t in _NUMBERS for t in toks)) or \
        bool(style and all(t in _STOP or t in _WRAPPER or t in _norm_tokens(style) or t == "style" for t in toks)
             or style and len(toks) <= 3 and not _CLASS_OF.get(toks[-1]) and toks[-1] in _WRAPPER)


def _format_kind(chunks):
    """The picture's own format named in the frame of the request ("a poster for ...", "an architecture diagram of
    ...", "... as a wide infographic"): the first phrase, the one after it when the first only names the medium,
    and any phrase introduced by "as"."""
    avail = _scene_kinds()
    frame, first = [], True
    for i, (words, prev) in enumerate(chunks):
        if i == 0 or prev == "as" or (i == 1 and _wrapper_chunk(chunks[0][0])):
            frame.append(words)
        elif prev in ("for", "in", "on") and set(_toks(words)) & _PURPOSE:
            frame.append(words)                              # "a plant cell for a biology textbook": a figure
    toks = []
    for words in frame:
        ts = _toks(words)
        toks += ts + [_sing(s) for t in ts for s in _SYN.get(t, "").split()]
    tokset = set(toks)
    low = " " + " ".join(toks) + " "
    score = {}
    specs = dict(_SCENE_WORDS)
    for k, extra in _FORMAT_EXTRA.items():
        specs[k] = (specs.get(k, "") + " " + extra).strip()
    for kind in _FORMAT_KINDS:
        for item in specs.get(kind, "").split():
            w, v = item.rsplit(":", 1)
            if _sing(w) in tokset:
                score[kind] = score.get(kind, 0) + float(v)
    raw = {t for words in frame for t in _toks(words)}
    ui = sum(float(v) for w, v in (i.rsplit(":", 1) for i in _UI_SCENE.split()) if w in raw)
    if ui:
        score["ui"] = ui
    for phrase, kind, v in ((" floor plan ", "floorplan", 6), (" still life ", "still_life", 6),
                            (" product shot ", "product", 4), (" low poly ", "scene3d", 3),
                            (" engineering drawing ", "drawing", 4), (" exploded isometric ", "drawing", 9),
                            (" mind map ", "diagram", 3), (" org chart ", "diagram", 3)):
        if phrase in low:
            score[kind] = score.get(kind, 0) + v
    if "figure" in score:
        score["figure"] += 0.5                                   # a labelled / textbook diagram is a figure
    score = {k: v for k, v in score.items() if k in avail and v >= 3}
    if not score:
        return None
    return max(score, key=lambda k: (score[k], -_FORMAT_KINDS.index(k)))


def _kind_of_entry(e, tokset):
    """The scene kind of an entry's subject: its kit's row of the SKILL.md table (the request's words and the
    subject's meaning class choose between the kinds of a row)."""
    return _row_kind(e.kit, set(tokset) | {_sem_of_entry(e)})


def _row_kind(kit, tokset):
    row = kit_kinds().get(kit)
    if not row:
        return _KIT_KIND_DEFAULT.get(kit)
    if len(row) == 1:
        return row[0]
    best, bk = 0.0, None
    for k in row:
        sc = 0.0
        for item in (_ROW_WORDS.get(k) or _SCENE_WORDS.get(k, "")).split():
            w, v = item.rsplit(":", 1)
            if _sing(w) in tokset:
                sc += float(v)
        if sc > best:
            best, bk = sc, k
    if bk:
        return bk
    return _KIT_KIND_DEFAULT.get(kit) if _KIT_KIND_DEFAULT.get(kit) in row else row[0]


class Group:
    """Results for one subject of the request: ``phrase``, ``results`` (entries), ``calls``, ``count``, ``colors``;
    ``fallback`` (a dict) when the catalogue has no entry and the answer is a way to draw it anyway."""

    def __init__(self, phrase, results, scores, calls, count=1, colors=(), head=None, fallback=None):
        self.phrase, self.results, self.scores, self.calls = phrase, results, scores, calls
        self.count, self.colors, self.head, self.fallback = count, list(colors), head, fallback

    def as_dict(self):
        out = {"phrase": self.phrase, "count": self.count,
               "colors": [{"word": w, "hex": h} for w, h in self.colors],
               "results": [dict(e.as_dict(), call=c, score=round(s, 1))
                           for e, s, c in zip(self.results, self.scores, self.calls)]}
        if self.fallback:
            out["fallback"] = self.fallback
        return out

    def __repr__(self):
        return f"<Group {self.phrase!r}: {[e.id for e in self.results] or 'recipe'}>"


def _repeat_call(call, n):
    """The call for ``n`` copies of a subject: a loop over positions from form.row / form.grid_positions."""
    if n <= 1:
        return call
    pos = f"form.row({n}, (0.25, 0.6), (0.75, 0.6))[0]" if n <= 6 else \
        f"form.grid_positions({math.ceil(n ** 0.5)}, {math.ceil(n / math.ceil(n ** 0.5))})[0][:{n}]"
    if re.search(r"\bat=\([^)]*\)", call):
        return f"for p in {pos}: " + re.sub(r"\bat=\([^)]*\)", "at=p", call, count=1)
    m = re.fullmatch(r'sc\.add\((.+), "([^"]+)"\)', call)
    if m:                                               # 3D: side by side along x, one named layer each
        return (f"for i in range({n}): sc.add({m.group(1).replace('seed=1', 'seed=i + 1')}.place(x=(i - {(n - 1) / 2}) * 0.9), "
                f"\"{m.group(2)} %d\" % (i + 1))")
    return f"{call}    # x{n}: one call per copy, each at its own place"



_FORMAT_NOISE = {"square", "portrait", "vertical", "horizontal", "wide", "tall", "format", "size", "ratio", "aspect",
                 "widescreen", "thumbnail", "4k", "hd", "resolution"}


def drawn_subjects(found):
    """The subjects of a request the picture draws (no format frame, no setting-only phrase, no size or style
    words)."""
    styles = {k for _n, k in _style_keywords()} | {"watercolour", "colour", "color"}
    return [g for g in found.groups if not getattr(g, "format", False) and not getattr(g, "setting_only", False)
            and not getattr(g, "text_object", False) and (g.head or "") not in _FORMAT_NOISE
            and (g.head or "") not in styles and g.phrase not in styles
            and not re.fullmatch(r"\d+\s*[x×*]\s*\d+", g.head or g.phrase)
            and g.phrase.split()[-1:] != ["square"]]


_INGREDIENT_2D = {"pictograms": "pictogram", "flora": "flora", "elements": "elements effect", "texture": "texture",
                  "backdrop": "backdrop", "palette": "palette"}


class Found:
    """What ``find`` returns: ``groups`` (one per subject, in the order they appear), ``colors``, ``style``,
    ``kind``, ``content`` (the text to draw), ``main`` (the main subject's group), ``extras`` (other kits to add
    inside it), ``kit_kind`` (the kind the kit table gives: the kind of ``text(kits=True)``); ``text()``,
    ``as_dict()``."""

    def __init__(self, query, groups, colors, style, kind, content=(), main=None, extras=(), format=False):
        self.format, self.kit_kind = format, kind
        self.query, self.groups, self.colors, self.style, self.kind = query, groups, colors, style, kind
        self.content, self.main, self.extras = list(content), main, list(extras)

    def best(self):
        """The top entry of every subject that has one."""
        return [g.results[0] for g in self.groups if g.results]

    def ids(self):
        return [e.id for g in self.groups for e in g.results]

    def as_dict(self):
        return {"query": self.query, "text_to_draw": self.content,
                "groups": [g.as_dict() for g in self.groups],
                "colors": [{"word": w, "hex": h} for w, h in self.colors], "style": self.style,
                "kind": self.kind, "kit_kind": self.kit_kind,
                "main": self.main.phrase if self.main else None, "add_kits": self.extras,
                "new_scene": f"python tools/new_scene.py NAME --kind {self.kind}" if self.kind else None}

    def _group_lines(self, g, alternatives, full_recipe):
        lab = max(6, min(20, max([len(x.phrase) + (4 if x.count > 1 else 0) for x in self.groups] + [6])))
        head = (g.phrase + (f" x{g.count}" if g.count > 1 else ""))[:lab].ljust(lab)
        pad = " " * (lab + 2)
        lines = []
        if g.fallback:
            fb = g.fallback
            near = fb.get("standins") or fb["nearest"]
            cls = f" ({fb['class']})" if fb["class"] else ""
            first = f"{near[0]}    # stand-in{cls}" if near else f"own shape{cls}: {fb['title']}"
            ref = f"{pad}own shape: read {fb.get('ref', _RECIPE_REF['sdf'])}"
            if not full_recipe:
                return [f"{head}  {first}", ref]
            lines.append(f"{head}  {first}")
            for c in near[1:3]:
                lines.append(f"{pad}{c}    # {'3D stand-in' if c.startswith('sc.add(') else 'stand-in'}")
            if near:
                lines.append(f"{pad}own shape: {fb['title']}")
            lines += [f"{pad}  {ln}" for ln in fb["lines"] + [fb["register"]]]
            lines.append(f"{pad}read: {fb.get('ref', _RECIPE_REF['sdf'])}")
            return lines
        calls = getattr(g, "_print_calls", None) or g.calls
        if getattr(g, "standin", False) and calls:
            note = g.results[0].kind + (" · " + g.results[0].ref if g.results[0].ref else "")
            lines.append(f"{head}  {_repeat_call(calls[0], g.count)}    # {note} · stand-in drawn shaded")
            for i in _shown(g, alternatives)[1:]:
                e, c = g.results[i], calls[i]
                lines.append(f"{' ' * lab}  {c}    # {e.kind}" + (" · " + e.ref if e.ref else ""))
            return lines
        if not g.results:
            return [f"{head}  the setting: the --kind template draws it"]
        for k, i in enumerate(_shown(g, alternatives)):
            e, c = g.results[i], calls[i]
            note = e.kind + (" · " + e.ref if e.ref else "")
            lead = head if k == 0 else " " * lab
            call = _repeat_call(c, g.count) if k == 0 else ("flat mark: " + c if e.kit == "pictograms" else c)
            lines.append(f"{lead}  {call}    # {note}" + (" · " + ", ".join(f'{w} = "{h}"' for w, h in g.colors[:2])
                                                          if k == 0 and g.colors else ""))
        return lines

    def _ingredient_lines(self, g, bound):
        """Optional ingredients for one subject of a freehand picture: a 2D part (pictogram, flora, an elements
        effect, a texture) and a 3D alternative (clay / solid), each a call that runs as printed."""
        out = []
        if g.fallback:                                              # no entry: a close relative's mark, if any
            near = (g.fallback.get("standins") or g.fallback.get("nearest") or []) if g.head in _KIN else []
            return [(c, "nearest relative" if c.startswith("pictograms.") else
                     "3D alternative" if c.startswith("sc.add(") else "nearest") for c in near[:1]]
        two = next(((e, c) for e, c in zip(g.results, g.calls) if e.kit in _INGREDIENT_2D), None)
        if two is None and g.head:                                  # the subject's pictogram as a flat part
            pic = next((e for e in _index() if e.kit == "pictograms" and (
                e.name == g.head or (g.head,) in e._alias_toks)), None) or \
                next(iter(_head_noun_entries(g.head, ("pictograms",))), None)
            if pic is not None:
                two = (pic, _shaded_call(pic, g.colors))
        three = next(((e, c) for e, c in zip(g.results, g.calls) if e.kit in ("clay", "solid")), None)
        for pair, what in ((two, None), (three, "3D alternative")):
            if pair is None:
                continue
            e, c = pair
            note = what or _INGREDIENT_2D[e.kit]
            out.append((_bind(e, c, bound), note + (" · " + e.ref if e.ref else "")))
        return out

    def _illustration_text(self, limit=30):
        """The freehand route, short: the start line, one line per subject the library has a real ingredient for
        (a pictogram, flora, an elements effect, a texture; else a clay / solid model as a 3D alternative), the
        kind last. Nothing here is needed to draw the picture: the ingredients are optional."""
        q = " ".join(self.query.split())
        quoted = '"' + q + '"' if '"' not in q else "'" + q.replace("'", "'\\''") + "'"
        lines = [f"draw it freehand (Sketch, cv.draw · references/illustration.md); starter: "
                 f"python tools/new_scene.py NAME --prompt {quoted}"]
        subs = drawn_subjects(self)
        lab = max(6, min(20, max([len(g.phrase) + (4 if g.count > 1 else 0) for g in subs] + [6])))
        bound = set()
        for g in subs:
            ing = self._ingredient_lines(g, bound) if g.results or g.fallback else []
            if ing and len(lines) < limit - 1:
                name = (g.phrase + (f" x{g.count}" if g.count > 1 else ""))[:lab].ljust(lab)
                c, note = ing[0]
                lines.append(f"  {name}  {c}    # optional {note}")
        more = f" (main subject: {self.main.phrase})" if self.main else ""
        lines.append(f"-> suggested kind: python tools/new_scene.py NAME --kind illustration{more}")
        return "\n".join(lines)

    def text(self, width=None, limit=30, kits=False):
        """Plain-text answer sized for a small context window (``limit`` lines at most): the text to draw, a call
        per subject in the order they appear (colours and counts attached), the style, the kind last. A freehand
        picture (kind illustration) starts with the command that starts it and lists the calls as ingredients;
        ``kits=True`` prints the kit answer (every call to paste, ``kit_kind``) instead."""
        if self.kind == "illustration" and not kits:
            return self._illustration_text(limit)
        kind = self.kit_kind if kits else self.kind
        head, tail = [], []
        if self.content:
            s = "text to draw: " + "; ".join(self.content)
            head.append(s if len(s) <= 150 else s[:147] + "...")
        groups = self.groups
        if not groups:
            title, lines, reg = recipe("<name>", kind)
            head += [title] + ["  " + ln for ln in lines] + [reg]
        attached = {w for g in groups for w, _ in g.colors}
        for w, h in [c for c in self.colors if c[0] not in attached][:3]:
            tail.append(f"colour  {w} -> color=\"{h}\" or palette.palette(\"{w}\")")
        if self.style:
            tail.append(f"style   {self.style} -> last step: stylize.apply(cv, \"{self.style}\")  · references/styles.md")
        if kind:
            more = f"; add {', '.join(self.extras)} calls inside it" if self.extras else ""
            what = "format" if self.format else "main subject"
            sub = f" ({what}: {self.main.phrase}{more})" if self.main else ""
            tail.append(f"-> suggested kind: python tools/new_scene.py NAME --kind {kind}{sub}")
        elif groups:
            tail.append("-> kind: the one that fits the main subject (python tools/kit.py kinds)")
        if groups and groups[0].results:
            tail.append(f"full snippet: python tools/kit.py show {groups[0].results[0].id}")
        bound = set()
        for g in groups:                                           # "d.bus(...)": with the d it needs, once
            g._print_calls = [_bind(e, c, bound if i == 0 else set(bound))
                              for i, (e, c) in enumerate(zip(g.results, g.calls))] if g.calls else []
        blocks = []
        budget = limit - len(head) - len(tail)
        first_fb = self.main if self.main is not None and self.main.fallback else \
            None if self.format else next((g for g in groups if g.fallback), None)
        for g in groups:                                           # main: alternatives + flat mark; others: + flat mark
            alt = len(g.results) if g is self.main else (2 if _flat_index(g) else 1)
            blocks.append([g, max(1, alt), g is first_fb])

        def size(b):
            return len(self._group_lines(b[0], b[1], b[2]))
        while sum(size(b) for b in blocks) > budget:
            drop = next((b for b in reversed(blocks) if b[1] > 1 and b[0] is not self.main), None) or \
                next((b for b in blocks if b[1] > 1), None)
            if drop:
                drop[1] -= 1
                continue
            drop = next((b for b in blocks if b[2] and b[0] is not self.main), None)
            if drop:
                drop[2] = False
                continue
            drop = next((b for b in reversed(blocks) if b[0] is not self.main and not b[0].fallback), None)
            if drop is None:
                drop = next((b for b in reversed(blocks) if b[0] is not self.main), None)
            if drop is None:
                break
            blocks.remove(drop)
        lines = list(head)
        for b in blocks:
            lines += self._group_lines(b[0], b[1], b[2])
        return "\n".join(lines + tail)

    def __repr__(self):
        return f"<Found {len(self.groups)} subjects, kind={self.kind}, style={self.style}>"


def _bind(e, call, bound):
    """A call on an object the kit builds first ("d.bus(...)", "v.bolt_circle(...)") with the lines that build it
    (from the entry's own snippet), once per answer; a bare constructor becomes the assignment it is in the snippet."""
    lines = [ln.strip() for ln in (e.code or "").split("\n") if ln.strip()]
    defs = {}
    for ln in lines:
        m = re.match(r"([a-z_]\w*) = (.+)$", ln)
        if m:
            defs.setdefault(m.group(1), ln)
            if m.group(2) == call and m.group(1) not in ("cv", "sc"):
                bound.add(m.group(1))
                return ln
    pre = []

    def need(var, depth=0):
        if var in bound or var in ("cv", "sc", "np") or var in _ALL_KITS or var not in defs or depth > 3:
            return
        rhs = defs[var].split(" = ", 1)[1]
        m = re.match(r"([a-z_]\w*)\.", rhs)
        if m:
            need(m.group(1), depth + 1)
        pre.append(defs[var])
        bound.add(var)

    m = re.match(r"([a-z_]\w*)\.", call)
    if m:
        need(m.group(1))
    return "; ".join(pre + [call])


def _flat_index(g):
    """Index of the pictogram alternative of a group (None when it has none or the pictogram is the answer)."""
    return next((i for i, e in enumerate(g.results) if i > 0 and e.kit == "pictograms"), None)


def _shown(g, alt):
    """Which results a group prints with ``alt`` lines: the answer, the other kits, the flat mark last."""
    fi = _flat_index(g)
    others = [i for i in range(1, len(g.results)) if i != fi]
    if alt <= 1:
        return [0]
    if fi is not None:
        return [0] + others[:alt - 2] + [fi]
    return [0] + others[:alt - 1]


def _adapt_call(e, slots, words, poses=None):
    """The entry's call, with a pose / coat / description taken from the phrase where the kit supports it."""
    call = e.call
    if poses is None:
        poses = [p for w in words for p in _POSE_WORDS.get(w.lower(), ())]
    pose = next((p for p in poses if p in e.poses), None)
    if pose and "pose=" not in call:
        if e.kit == "figure":
            call = call.replace(").draw(", f').pose("{pose}").draw(', 1)
        elif "size=0.4)" in call:
            call = call.replace("size=0.4)", f'size=0.4, pose="{pose}")', 1)
        else:
            call = call.replace(f'("{e.name}")', f'("{e.name}", pose="{pose}")', 1)
    if e.kit == "creature" and e.kind == "animal":
        try:
            from . import creature as _c
            coats = sorted(getattr(_c, "COATS", {}), key=len, reverse=True)
            text = " ".join(words).lower()
            for c in coats:
                if re.search(r"\b" + re.escape(c) + r"\b", text) and c != e.name:
                    call = call.replace("size=0.4", f'size=0.4, coat="{c}"', 1)
                    break
        except Exception:
            pass
    if e.kit == "figure" and e.name == "person":
        desc = " ".join(w for w in words if w.lower() not in ("a", "an", "the") and w.lower() not in _NUMBERS
                        and not w.isdigit())
        if desc:
            call = call.replace("a woman in a red coat waving", desc.replace('"', ""), 1)
    return call


def _legacy_kind(query, _groups=None):
    """Word-count vote for a scene kind (used to bias the kit ranking before the subjects are known)."""
    avail = _scene_kinds()
    text = _translate_cjk(query)
    toks = _norm_tokens(text)
    toks += [_sing(s) for t in toks for s in _SYN.get(t, "").split()]
    tokset = set(toks)
    score = {}
    for kind, spec in _SCENE_WORDS.items():
        if kind not in avail:
            continue
        for item in spec.split():
            w, v = item.rsplit(":", 1)
            if _sing(w) in tokset:
                score[kind] = score.get(kind, 0) + float(v)
    low = " " + " ".join(toks) + " "
    for phrase, kind, v in ((" floor plan ", "floorplan", 6), (" still life ", "still_life", 6),
                            (" product shot ", "product", 4), (" low poly ", "scene3d", 3),
                            (" engineering drawing ", "drawing", 4), (" mind map ", "diagram", 3),
                            (" org chart ", "diagram", 3), (" line chart ", "chart", 2), (" bar chart ", "chart", 2),
                            (" pie chart ", "chart", 2)):
        if phrase in low and kind in avail:
            score[kind] = score.get(kind, 0) + v
    if not score or max(score.values()) < 2.5:
        return None
    order = list(avail)
    return max(score, key=lambda k: (score[k], -order.index(k)))


def suggest_kind(query, _groups=None):
    """The ``new_scene.py --kind`` for a request: the picture's own format when the request names one (poster,
    diagram, dashboard ...), else the SKILL.md kit-table row of the main subject (the first subject noun phrase).
    None when nothing points anywhere."""
    return find(query).kind


def _pick_style(query):
    try:
        from . import stylize as _s
        return _s.pick(query)
    except Exception:
        return None


# ------------------------------------------------------------------ nothing in the catalogue: draw it anyway
def _pictograms():
    try:
        return importlib.import_module("pixelcanvas.pictograms")
    except Exception:                                              # absent or half-written: the catalogue goes on
        return None


_NEAR_KIT_ORDER = ("pictograms", "creature", "figure", "clay", "solid", "flora", "form", "vector")


def _nearest(head, cls):
    """Calls of the entries closest in meaning to a subject with no entry of its own: pictograms first, then the
    class's neighbours (instrument -> guitar), then a vector icon of a similar meaning."""
    idx = {}
    for e in _index():
        idx.setdefault(e.name.replace(" ", "_"), []).append(e)
    out = []

    def add(name, kits=_NEAR_KIT_ORDER):
        for kit in kits:
            for e in idx.get(name, []):
                if e.kit == kit and e.call not in [c for c, _ in out] and e.kind not in ("theme", "palette"):
                    out.append((e.call, kit))
                    return True
        return False

    for n in _KIN.get(head, "").split():                            # the closest relatives first (gramophone: turntable)
        add(n, ("pictograms",)) or add(n, tuple(k for k in _NEAR_KIT_ORDER if k != "vector"))
    if cls not in _CLASSES:                                         # unknown meaning: its relatives, else the recipe
        return [c for c, _ in out][:3]
    words, near, icons, _kind = _CLASSES[cls]
    for n in near.split():                                          # the class's nearest pictograms (zither: harp, guitar)
        add(n, ("pictograms",))
    if _pictograms() is not None:                                   # the class's other members that have a pictogram
        for n in sorted({_sing(w) for w in words.split()}):
            if len(out) >= 3:
                break
            if n != head and n in idx and any(e.kit == "pictograms" for e in idx[n]):
                add(n, ("pictograms",))
    for n in near.split():                                          # then the other kits' drawings of them
        if len(out) >= 3:
            break
        add(n, tuple(k for k in _NEAR_KIT_ORDER if k not in ("vector", "pictograms")))
    for n in icons.split():
        if add(n, ("pictograms", "vector")) and sum(1 for _, k in out if k == "vector") >= 1:
            break
    return [c for c, _ in out][:3]


def _fallback(head, cls, kind, silhouette=False):
    """A subject no kit names: stand-ins in order (a relative's pictogram drawn shaded at illustration size, a solid
    prefab), then a short recipe for its own shape and the reference section to read."""
    title, lines, reg = recipe(head or "<name>", kind, silhouette)
    near = _nearest(head or "", cls)
    by_name = {}
    for e in _index():
        by_name.setdefault((e.kit, e.name), e)
    standins = []
    for c in near:
        m = re.match(r'pictograms\.pictogram\(cv, "([^"]+)"', c)
        e = by_name.get(("pictograms", m.group(1))) if m else None
        standins.append(_shaded_call(e, []) if e is not None else c)
    rel = (_KIN.get(head or "", "") + " " +
           (_CLASSES[cls][1] if cls in ("furniture", "audio", "container") else "")).split()
    solid3d = next((e.call for n in rel for e in _index() if e.kit == "solid" and e.kind == "prefab"
                    and e.name == n), None)
    if solid3d and solid3d not in standins:
        standins.insert(1 if standins else 0, solid3d)
    key = "silhouette" if silhouette else _RECIPE_KINDS.get(kind, "sdf")
    return {"name": head, "class": cls, "nearest": near, "standins": standins[:3], "recipe": title,
            "title": title, "lines": lines, "register": reg, "ref": _RECIPE_REF[key]}


# ------------------------------------------------------------------ style: only a style construction names one
_MEDIUM = ("style", "styled", "look", "aesthetic", "effect", "finish", "technique", "art", "artwork", "painting",
           "drawing", "print", "sketch", "illustration", "render", "rendering", "photo", "photograph", "sheet",
           "window", "portrait", "piece", "panel", "poster", "game", "work", "image", "picture", "version")
_KEEP_MEDIUM = {"poster", "sheet", "window", "portrait", "panel", "picture", "image"}
_MEDIUM_RE = r"(?:" + "|".join(_MEDIUM) + r")s?\b"
_FORMAT_ADJ = {"engineering", "technical", "mechanical", "architectural", "cad", "patent", "assembly", "exploded",
               "construction", "shop", "floor"}
_STEER_WORDS = _STEER_3D | _STEER_2D


def _style_keywords():
    try:
        from . import stylize as _s
        return [(name, k.lower()) for name, st in getattr(_s, "STYLES", {}).items() for k in st.get("keywords", ())]
    except Exception:
        return []


def _style_of(text):
    """(style, [(start, end)]): the style a request names through a style construction, and the spans of that
    construction. "in watercolour", "as a charcoal drawing", "pixel-art", "in the style of ...", "an oil painting of
    ...", "a watercolour of ..." name a style; "in orange and charcoal", "a clay pot", "an oil lamp" and "a pencil
    case" do not. ``stylize.pick`` decides which style the construction's words mean."""
    low = text.lower()
    colors = _colors()
    found = []
    for name, kw in _style_keywords():
        if not kw:
            continue
        if not kw.isascii():
            if kw in low and kw not in _CJK:
                i = low.index(kw)
                found.append((len(kw), i, i + len(kw), kw))
            continue
        pat = r"(?<![a-z0-9])" + r"[\s-]+".join(re.escape(p) for p in re.split(r"[\s-]+", kw)) + r"(?![a-z0-9])"
        for m in re.finditer(pat, low):
            s, e = m.start(), m.end()
            before, after = low[:s], low[e:]
            prev_word = (re.findall(r"[a-z0-9]+", before) or [""])[-1]
            if prev_word in _FORMAT_ADJ:
                continue                                   # "an engineering drawing" is a format, not a pencil style
            span = None
            if re.search(r"\bin the style of\s+(?:(?:a|an|the)\s+)?(?:[a-z-]+\s+){0,2}$", before):
                span = (before.rindex("in the style of"), e)
            m_in = re.search(r"\b(in|as)\s+((?:(?:a|an|the)\s+)?(?:[a-z-]+\s+){0,2})$", before)
            if span is None and m_in:
                mid = re.findall(r"[a-z]+", m_in.group(2))
                tail_ok = re.match(r"\s*(?:$|[,.;:!?)]|(?:with|on|of|for|to|at|in|from|by)\b|[\s-]*" + _MEDIUM_RE + ")",
                                   after)
                listy = any(w in ("and", "or") or w in colors for w in mid) or \
                    re.match(r"\s*(?:and|or|,)\s+([a-z]+)", after) and \
                    re.match(r"\s*(?:and|or|,)\s+([a-z]+)", after).group(1) in colors
                if tail_ok and not listy:
                    span = (m_in.start(), e)
            m_med = re.match(r"[\s-]+(?:[a-z0-9]+[\s-]+)?(" + _MEDIUM_RE + ")", after)
            if span is None and m_med:
                word = m_med.group(1).rstrip("s")
                span = (s, e if word in _KEEP_MEDIUM else e + m_med.end())
            if span is None and re.match(r"\s+of\b", after) and len(re.findall(r"[a-z0-9]+", before)) <= 3:
                span = (s, e)                              # "a watercolour of two ducks"
            if span is None and ("-" in m.group(0) or re.match(r"-(?:style|styled|inspired|like|look)\b", after)):
                span = (s, e + (re.match(r"-[a-z]+", after).end() if after.startswith("-") else 0))
            if span:
                found.append((len(kw), span[0], span[1], kw))
    if not found:
        return None, []
    pick = _pick_style(" , ".join(f[3] for f in found))
    if pick is None:
        return None, []
    return pick, [(s, e) for _l, s, e, _k in found]


def _blank(text, spans):
    for s, e in spans:
        text = text[:s] + " , " + " " * max(0, e - s - 3) + text[e:]
    return text


def _steer_of(words):
    """'3d', '2d' or None from the words of a phrase or a request (hyphenated parts too: "toy-like", "low-poly")."""
    parts = [p.strip("'’").lower() for w in words for p in re.split(r"[-–]", w) if p.strip("'’")]
    s3 = s2 = False
    for i, p in enumerate(parts):
        nxt = parts[i + 1] if i + 1 < len(parts) else ""
        if p in _NOT_STEER and nxt in _NOT_STEER[p]:
            continue
        if p in ("lineup",):
            continue
        s3 |= p in _STEER_3D
        s2 |= p in _STEER_2D
    return "3d" if s3 and not s2 else "2d" if s2 and not s3 else None


# ------------------------------------------------------------------ phrases: subject, pose, count
class _Phrase:
    __slots__ = ("words", "prev", "poses", "steer", "start")

    def __init__(self, words, prev, poses=()):
        self.words, self.prev, self.poses = list(words), prev, list(poses)
        self.steer = None

    def __repr__(self):
        return f"<{' '.join(self.words)} prev={self.prev} poses={self.poses}>"


def _content_word(w):
    lw = w.lower().strip("'’")
    return bool(lw) and lw not in _STOP and lw not in _NUMBERS and not lw.isdigit() and lw not in _DET


def _parse(text):
    """Noun phrases of a request in order. A participle or verb after a subject ("sitting", "reading", "waving",
    "playing with") ends that subject's phrase and becomes its pose; the words after it are a new phrase. A
    participle before a noun ("a sleeping cat") is a pose too. Hyphenated modifiers ("lantern-lit", "toy-like") are
    no subjects; collectives give way to their members ("a team of five coworkers")."""
    vocab = _vocab()
    out = []
    for words, prev in _chunks2(text):
        cur, cprev, poses = [], prev, []
        for i, w in enumerate(words):
            lw = w.lower().strip("'’")
            parts = lw.split("-")
            if len(parts) > 1 and (parts[-1] in _COMPOUND_TAILS or parts[-1].rstrip("s") in _COMPOUND_TAILS):
                continue                                          # "lantern-lit", "toy-like": a modifier
            if lw in _SAYS and any(_content_word(x) for x in cur):
                out.append(_Phrase(cur, cprev, poses))              # "text that says SALE": SALE is the text
                cur, poses = [], []
                break
            if lw in _VERBS or (lw.endswith("ing") and len(lw) > 5 and lw not in _ING_NOUNS
                                and lw not in _CLASS_OF and lw not in _SETTING_KIND
                                and any(_content_word(x) for x in cur)):
                nxt = words[i + 1] if i + 1 < len(words) else ""
                content = [x for x in cur if _content_word(x)]
                noun_next = bool(nxt) and _content_word(nxt) and nxt.lower() not in _PARTICLES
                noun_before = bool(content) and any((t in vocab or t in _CLASS_OF or t in _LEXICON or t in _SYN)
                                                    and t not in _ADJECTIVES and t not in _colors()
                                                    for t in _norm_tokens(content[-1]))
                pose = list(_POSE_WORDS.get(lw, ()))
                if content and (not noun_next or noun_before):     # "a corgi sitting", "a man reading a book"
                    out.append(_Phrase(cur, cprev, poses + pose))
                    cur, cprev, poses = [], lw, []
                elif not content and not noun_next:               # "one waving at the bus": the friend waves
                    if out:
                        out[-1].poses += pose
                    cur, cprev = [], lw
                else:                                             # "a sleeping cat", "falling leaves"
                    poses += pose
                    cur.append(w)
                continue
            cur.append(w)
        if any(_content_word(x) for x in cur):
            out.append(_Phrase(cur, cprev, poses))
        elif poses and out:
            out[-1].poses += poses
    merged = []
    for p in out:
        if merged and p.prev == "of":
            last = merged[-1]
            ltoks = [t for t in _toks(last.words) if t not in _STOP and t not in _DET]
            if ltoks and all(t in _COLLECTIVE or t in _NUMBERS or t in _colors() for t in ltoks):
                n = _count_of(p.words)
                c = _COLLECTIVE.get(ltoks[-1], 0) if ltoks[-1] in _COLLECTIVE else 0
                if n == 1 and c:
                    p.words = [str(c)] + p.words
                p.prev, p.poses = last.prev, last.poses + p.poses
                merged[-1] = p                                    # "a team of five coworkers" -> five coworkers
                continue
            h, lh = _head(p.words, vocab), _head(last.words, vocab)
            if h in _MASS and lh in _MEASURE:
                continue                                          # "a ball of yarn" -> the ball (of yarn)
            if lh in _FLOCKS:
                last.words = last.words + p.words                 # "a school of yellow fish": one school
                continue
        merged.append(p)
    return merged


_SAYS = {"says", "saying", "spells", "spelling", "reads", "read", "titled", "labelled", "labeled"}
_SCENE_ENTRIES = {"isometric", "studio backdrop", "tabletop", "room"}
_BACKGROUND_HEADS = set(_SETTING_KIND) | {"background", "backdrop", "wallpaper", "sky", "cloud", "mountain",
                                          "skyline", "city", "horizon", "sunset", "sunrise", "dusk", "dawn", "night",
                                          "gradient", "pattern", "space", "galaxy", "hill", "dune", "wave", "aurora",
                                          "bokeh", "star", "underwater", "landscape", "scenery", "vista"}
_PALETTE_WORDS = {"palette", "color", "colour", "scheme", "tone", "hue", "swatch"}
_ING_NOUNS = set("""building painting drawing ceiling string king ring wing thing spring swing evening morning
lightning pudding clothing railing lighting earring wedding dumpling sapling seedling icing bedding landing parking
housing bearing engineering offering packaging frosting awning duckling gosling herring stuffing setting heading
opening ending listing pricing onboarding training meeting sibling darling viking topping filling seasoning
dressing stocking legging leggings kingfisher wellington""".split())
_FORMAT_HEADS = {"product", "shot", "packshot", "render", "rendering", "scene", "3d", "wallpaper", "lockscreen",
                 "diorama", "cutaway", "view", "still", "life", "illustration", "picture", "image", "photo",
                 "isometric", "lowpoly", "poly", "closeup", "artwork", "comic", "strip", "manga", "panel", "toy",
                 "figurine", "sticker", "sheet"}
_MOD_EFFECTS = {"neon", "fire", "flame", "smoke", "steam", "fireworks", "stars", "rainbow", "aurora", "snow", "rain"}
_PARTICLES = {"past", "away", "up", "down", "out", "off", "overhead", "gently", "around", "about", "back", "together",
              "slowly", "quietly", "happily", "high", "low", "alone", "outside", "inside", "home"}
_TEXT_OBJECTS = {"sign", "signboard", "banner", "placard", "plaque", "billboard", "noticeboard", "marquee", "board",
                 "chalkboard", "blackboard", "sandwich board", "scroll", "pennant", "flag", "label", "tag", "sticker"}
_ATTRIBUTE = set("""hair eye eyes face hand hands arm arms leg legs tail ear ears beard moustache mustache smile nose
mouth fur feather feathers wing wings paw paws whisker whiskers skin cheek cheeks headlight headlights wheel wheels
tyre tyres tire tires bumper handle lid""".split())


# ------------------------------------------------------------------ which kit answers a subject
_D2 = {"figure", "creature", "flora", "elements", "texture", "backdrop", "scenery", "form"}
_D3 = {"clay", "solid"}
_DESIGN_KINDS = {"poster", "banner", "card", "icons", "diagram", "chart", "dashboard", "infographic", "ui", "figure", "drawing",
                 "floorplan"}
_ANIMAL_HEADS = {"dog", "cat", "puppy", "kitten", "bird", "fish", "horse", "pony", "bear", "rabbit", "bunny"}
_HYPERNYMS = {"bird", "fish", "animal", "insect", "bug", "tree", "flower", "plant", "bush", "shrub", "houseplant",
              "person", "pet", "mammal", "dinosaur"}
_PICTO_PLANTS = {"flower", "tulip", "rose", "sunflower", "tree", "pine", "palm", "cactus", "clover", "leaf",
                 "maple leaf", "toadstool"}


def _tier(e, mode, steer, app=False, head=None):
    """Rank of an entry's kit for one subject (lower first): drawing kits before pictograms in illustrations, the
    pictogram first in designs and flat marks; 3D words put clay / solid first, flat words the 2D kits."""
    k = e.kit
    if k in ("layout", "diagram", "chart", "ui", "drafting", "text", "palette", "stylize"):
        return 0 if mode == "flat" else 1
    if k == "backdrop" and head is not None and head not in _BACKGROUND_HEADS:
        return 4                                        # "a glass": a glass, not a glass backdrop
    if mode == "flat" and steer is None:
        order = {"pictograms": 1, "vector": 0} if app else {"pictograms": 0, "vector": 1}
        return order.get(k, 2 if k in _D2 else 3)
    if steer == "3d":
        return 0 if k in _D3 else 1 if k in _D2 else 2 if k == "pictograms" else 3
    if steer == "2d":
        return 0 if k in _D2 else 1 if k == "pictograms" else 2 if k in _D3 else 3
    return 0 if k in _D2 else 1 if k in _D3 else 2 if k == "pictograms" else 3


def _sem_of_entry(e):
    """The meaning class of what an entry draws: person, animal, plant, effect, setting, material, object, format."""
    k = e.kit
    if k == "figure" or (k == "clay" and e.kind == "3D character"):
        return "person"
    if k == "creature" or (k == "clay" and e.kind == "animal"):
        return "animal"
    if k == "pictograms":
        cat = e.keywords[0] if e.keywords else ""
        return {"animals": "animal", "people": "person"}.get(cat, "plant" if cat == "nature" and e.name in
                                                             _PICTO_PLANTS else "object")
    if k == "flora":
        return "plant"
    if k == "elements":
        return "effect"
    if k in ("backdrop", "scenery"):
        return "setting"
    if k in ("texture", "palette", "stylize"):
        return "material"
    if k in ("layout", "diagram", "chart", "ui", "drafting", "text", "vector", "scenes"):
        return "format"
    return "object"


_CLASS_SEM = {"person": "person", "animal": "animal", "bird": "animal", "sea creature": "animal", "plant": "plant"}


def _sem(g):
    if g.fallback:
        return _CLASS_SEM.get(g.fallback.get("class"), "object")
    if not g.results:
        return "setting"
    return _sem_of_entry(g.results[0])


def _scored(slots, ctx):
    """[(score, entry)] for a phrase, best first (the catalogue's word ranking)."""
    index, prefer = ctx["index"], ctx["prefer"]
    chunk_style = _pick_style(" ".join(s.raw for s in slots))
    surface = any(t in _SURFACE for s in slots for t in s.toks)
    out = []
    paint = any(t in _PALETTE_WORDS for s in slots for t in s.toks)
    for e in index:
        if e.kit == "scenes" or not _need_ok(e, slots, prefer):
            continue
        sc = _score(e, slots)
        if sc <= 0:
            continue
        if e.kit in prefer:
            sc += 10
        elif prefer:
            sc -= 6
        if e.kind == "material" and not surface:
            sc -= 10
        if chunk_style and e.kit == "stylize" and e.name == chunk_style:
            sc += 20
        if e.kit == "palette" and not paint:
            sc -= 1000                                  # a palette answers "palette" / "colours", else it trails
        out.append((sc, e))
    out.sort(key=lambda p: (-p[0], ctx["kit_rank"].get(p[1].kit, 60), p[1].order))
    return [(sc + 1000, e) if sc < -500 else (sc, e) for sc, e in out]


def _full(seq, slots, exact=False):
    """Every token of ``seq`` is in the phrase (by the word, a synonym or a typo fix; ``exact``: by the word)."""
    if not seq:
        return False
    for t in seq:
        w = max((s.toks.get(t, 0.0) for s in slots), default=0.0)
        if w < (1.0 if exact else 0.7):
            return False
    return True


def _other_compound(e, head, mod):
    """The phrase is a compound ("pencil case") and the entry is a different thing that only shares its head word
    ("charging case", "suitcase", a keyword "case"): no match through the head alone."""
    if not mod or not head or head in _HYPERNYMS:
        return False
    toks = set(e._toks) | {t for a in e._alias_toks for t in a} | set(e._kw)
    if mod in toks:
        return False
    return not (len(e._toks) == 1 and e._toks[0] == head)          # "a kitchen table": the table itself is fine


def _modifier(words, head, vocab=None):
    """The noun right before the head that makes it a compound ("pencil" of "pencil case"), else None."""
    toks = [_sing(w.lower().strip("'’")) for w in words]
    if not head or head not in toks:
        return None
    i = len(toks) - 1 - toks[::-1].index(head)
    if i == 0:
        return None
    m = toks[i - 1]
    if not m or m in _STOP or m in _DET or m in _NUMBERS or m.isdigit() or m in _colors() or m in _ADJECTIVES \
            or "-" in m or m.endswith(_ADJ_SUFFIX) or m.endswith("ed") or m in _SEASONS:
        return None
    if m in _STEER_WORDS and m not in _NOUN_STEER:
        return None
    return m if (m in _name_tokens() or m in _LEXICON or m in _CLASS_OF or m in _NOUN_STEER) else None


_NOUN_STEER = {"pencil", "crayon", "chalk", "ink", "paper", "marker", "pen", "brush", "pixel", "wax"}


def _level(e, slots, head_slot, exact=False, mod=None):
    """3: the entry's name or an alias names the phrase's head; 1: a keyword of the entry is the head; 0.5: the
    entry names another word of the phrase; 0: no match by name (also: another compound of the same head word)."""
    named = _full(e._toks, slots, exact) or any(_full(a, slots, exact) for a in e._alias_toks)
    if head_slot is None:
        return 3 if named else 0
    ht = set(head_slot.toks)
    if mod and _other_compound(e, _sing(head_slot.raw.lower()), mod) and not _full(e._toks, slots, True):
        return 0
    if named and e.kind == "animal" and ht & _ANIMAL_HEADS:
        return 3                                        # "a golden retriever puppy", "a tabby cat"
    names = set(e._toks) | {t for a in e._alias_toks for t in a}
    if named and ht & names:
        return 3
    if ht & set(e._kw):
        if ht & _HYPERNYMS and e.kit in _D2 | _D3:
            return 3                                    # "a bird": a sparrow drawn by creature names it
        return 3 if named else 1
    if e._toks and e._toks[-1] in ht:
        return 1                                        # "a seamless backdrop": the studio backdrop
    return 0.5 if named else 0


def _coverage(e, slots):
    """How many words of the phrase the entry's name or best alias covers ("bus stop" > "stop")."""
    return max([len(e._toks) if _full(e._toks, slots) else 0] +
               [len(a) for a in e._alias_toks if _full(a, slots)])


def _alias_only(e, slots, head):
    """The entry names the phrase only through an alias that is part of its own longer name, or an alias of
    another thing (pictogram "document" for "paper")."""
    if _full(e._toks, slots):
        return False
    return any(_full(a, slots) for a in e._alias_toks)


def _choose(slots, head, ctx, mode, steer, n, exact=False):
    """(results, scores, modifier entry) for one phrase: the entries that name its head, ordered by kit (``_tier``)
    then score, one per kit, with the pictogram kept as the flat-mark alternative."""
    scored = _scored(slots, ctx)
    if not scored:
        return [], [], None
    head_slot = next((s for s in slots if head and head in _norm_tokens(s.raw)), None)
    if head_slot is None and head:
        head_slot = next((s for s in slots if head in s.toks), None)
    mod = ctx.get("mod")
    lv = [(_level(e, slots, head_slot, exact, mod), sc, e) for sc, e in scored]
    fk = ctx.get("fmt_kits", ())
    lv = [(3 if (lev == 1 and e.kit in fk) else lev, sc, e) for lev, sc, e in lv]
    best = max((x[0] for x in lv), default=0)
    if best < 1:
        best = 0
    pool = [(sc, e) for l, sc, e in lv if l >= best and best >= 1 and
            sc >= (_MIN_SCORE - 20 if best >= 3 else 30 if e.kit == "form" else 50)]
    if best < 3:
        pool = [(sc, e) for sc, e in pool if e.kit not in ("pictograms", "vector", "palette")]
    modifier = None
    mods = [(sc, e) for l, sc, e in lv if l == 0.5 and e.kit in ("elements", "creature", "flora") and sc >= _MIN_SCORE
            and e.kind not in ("sky",) and _full(e._toks, slots, True)
            and (e.kit != "elements" or e.name.replace(" ", "_") in _ATMOS or e.name in _MOD_EFFECTS)]
    if mods:
        modifier = mods[0][1]
    if not pool:
        return [], [], modifier
    if modifier is not None and any(e.kit == modifier.kit for l, sc, e in lv if l >= best):
        modifier = None                                 # "a cherry blossom tree": one cherry tree
    top = max(sc for sc, _ in pool)
    pool = [(sc, e) for sc, e in pool if sc >= min(top * 0.55, _MIN_SCORE)]
    if head in _ABSTRACT_HEADS:
        pool = [(sc, e) for sc, e in pool if e.kit != "pictograms"]             # "a gold palette": colours
        if not pool:
            return [], [], None
    content = [s_ for s_ in slots if s_.raw.lower() not in _STOP]
    pool = [(sc, e) for sc, e in pool if e.kit != "pictograms" or _full(e._toks, slots) or
            any(_full(a, slots) and (len(a) > 1 or len(content) <= 1) for a in e._alias_toks)]
    if not pool:
        return [], [], modifier
    exact = [e for _sc, e in pool if head and head in e._toks and _full(e._toks, slots, True)]
    if exact:                                           # "an arrow": the arrow, not "bow and arrow" by its alias
        most = max(_coverage(e, slots) for e in exact)
        pool = [(sc, e) for sc, e in pool if e in exact or e.kit != "pictograms" or not _alias_only(e, slots, head)
                or _coverage(e, slots) > most]
    paint = any(t in _PALETTE_WORDS for s in slots for t in s.toks)
    app = ctx.get("app", False)
    def _names_mod(e):                                  # "a stone lantern", "a reading lamp": the compound itself
        return bool(mod) and (mod in e._toks or any(mod in a for a in e._alias_toks) or
                              (e.kit == "solid" and mod in e._kw))
    pool.sort(key=lambda p: (p[1].kit == "palette" and not paint, not _names_mod(p[1]),
                             _tier(p[1], mode, steer, app, head),
                             -_coverage(p[1], slots), -p[0],
                             ctx["kit_rank"].get(p[1].kit, 60), p[1].order))
    picked, kits = [], set()
    for sc, e in pool:
        if e.kit in kits:
            continue
        if len(picked) < n:
            picked.append((sc, e))
            kits.add(e.kit)
    if picked and not paint and picked[0][1].kit not in ("elements", "backdrop"):
        picked = [x for x in picked if x[1].kit != "palette"]      # "oranges" are fruit, not a palette
    flat = next(((sc, e) for sc, e in pool if e.kit == "pictograms"), None)
    if flat and all(e.kit != "pictograms" for _, e in picked):
        picked.append(flat)
    return [e for _, e in picked], [sc for sc, _ in picked], modifier


def _quote_after(query, head, used):
    """The quoted text that follows a word of the request ("a sign reading 'Open Late'"), else None."""
    if not head:
        return None
    m = re.search(r"\b" + re.escape(head) + r"\w*\b[^'\"“‘.;]{0,40}?(?<![A-Za-z])['\"“‘]([^'\"”’]{1,80})['\"”’]",
                  query, re.I)
    if m and m.group(1) not in used:
        return m.group(1)
    return None


def _with_text(call, title=None, subtitle=None):
    """A text-bearing call with the request's own words in place of the placeholders."""
    if title:
        lit = json.dumps(title, ensure_ascii=False)
        if re.search(r'\btitle="[^"]*"', call):
            call = re.sub(r'\btitle="[^"]*"', lambda m: "title=" + lit, call, count=1)
        elif re.search(r'\btext="[^"]*"', call):
            call = re.sub(r'\btext="[^"]*"', lambda m: "text=" + lit, call, count=1)
        elif call.startswith("cv.text(") or call.startswith("text.text_on"):
            call = re.sub(r'\("[^"]*"', lambda m: "(" + lit, call, count=1) if call.startswith("cv.text(") else \
                re.sub(r'\(cv, "[^"]*"', lambda m: "(cv, " + lit, call, count=1)
        elif call.startswith("chart.quick(") and "title=" not in call:
            call = call[:-1] + f", title={lit})"
    if subtitle:
        lit = json.dumps(subtitle, ensure_ascii=False)
        call = re.sub(r'\bsubtitle="[^"]*"', lambda m: "subtitle=" + lit, call, count=1)
    elif title:
        call = call.replace(', subtitle="Subtitle"', "")          # no placeholder beside the request's own title
    return call


def _figure_person():
    for e in _index():
        if e.kit == "figure" and e.name == "person":
            return e
    return None


def _text_entry():
    for e in _index():
        if e.kit == "text" and e.name == "fit text in box":
            return e
    return None


# ------------------------------------------------------------------ head nouns, stand-ins, 3D people and animals
# words that keep people and animals in the flat 2D kits (figure / creature); without one, an illustration of a
# person or an animal is drawn by the 3D clay kit, and these words send it there explicitly
_CHAR_FLAT = {"flat", "vector", "anime", "comic", "manga", "textbook", "2d", "cartoon", "chibi", "kawaii", "doodle",
              "lineart", "sticker", "icon", "icons", "pictogram", "silhouette", "editorial", "pixel", "infographic",
              "diagram", "labelled", "labeled", "emoji", "logo", "outline"}
_CHAR_3D = {"storybook", "soft", "cute", "toy", "clay", "3d", "mascot", "plush", "figurine", "claymation", "vinyl",
            "render", "rendered", "isometric", "diorama", "miniature", "lowpoly", "plasticine", "cgi", "blender",
            "adorable"}
_STYLE_3D = {"storybook", "clay", "low_poly"}
# a body colour for a pictogram drawn as a shaded stand-in, by its category (the request's colour wins)
_PICTO_TINT = {"food": "#e0703a", "transport": "#3f74c0", "home": "#b98a5a", "nature": "#4f9a55",
               "animals": "#b07a4a", "buildings": "#c9a27a", "tech": "#5a6474", "music": "#9a6a3a",
               "clothing": "#c8553d", "sport": "#e0a030", "holidays": "#d64541", "people": "#e0a070",
               "science": "#4a90b8", "tools": "#7a8088", "office": "#4a7bb7"}
_RECIPE_REF = {"sdf": "references/techniques.md sections 9 (shapes and strokes) and 3 (normals -> lighting)",
               "silhouette": "references/techniques.md sections 1 (silhouettes) and 9 (shapes and strokes)",
               "3d": "references/solid.md (primitives, smooth_union, materials)"}
_BUILDS = {"kid": "kid", "child": "kid", "boy": "kid", "girl": "kid", "toddler": "kid", "baby": "baby",
           "teen": "teen", "teenager": "teen", "student": "teen", "grandma": "elder", "grandpa": "elder",
           "elderly": "elder", "old": "elder", "hero": "hero", "superhero": "hero", "chibi": "chibi"}


def _char_steer(query, gmode, style):
    """'3d' (clay), '2d' (figure / creature) or None for the people and animals of a request: flat, vector, anime,
    comic, line-art and textbook keep them 2D; storybook, soft shading, cute, toy, clay, 3D and mascot make them
    clay, and so does an illustration that names no flat style."""
    low = query.lower().replace("line art", "lineart").replace("line-art", "lineart").replace("low poly", "lowpoly")
    low = low.replace("low-poly", "lowpoly")
    ws = {_sing(w) for w in re.findall(r"[a-z0-9]+", low)} | set(re.findall(r"[a-z0-9]+", low))
    if ws & _CHAR_FLAT:
        return "2d"
    if style and style not in _STYLE_3D:
        return "2d"                                     # a watercolour, an ukiyo-e print, pixel art: flat kits
    if ws & _CHAR_3D or style in _STYLE_3D:
        return "3d"
    return None if gmode == "flat" else "3d"


def _head_noun_entries(head, kits=None):
    """Entries that are a kind of ``head``: a multi-word name ending in it ("rowing boat", "desk lamp") or one word
    ending in it after a known word ("sailboat", "motorboat"); name or alias."""
    if not head or len(head) < 3:
        return []
    vocab = _vocab()
    out = []
    for e in _index():
        if (kits and e.kit not in kits) or e.kit in ("scenes", "palette", "stylize", "texture", "backdrop") or \
                e.kind in ("theme", "palette", "material", "3D material", "lighting", "function"):
            continue
        for t in (e._toks,) + tuple(e._alias_toks):
            if not t:
                continue
            if len(t) > 1 and t[-1] == head:
                out.append(e)
                break
            if len(t) == 1 and t[0] != head and len(head) >= 4 and t[0].endswith(head):
                pre = t[0][:-len(head)]
                if len(pre) >= 3 and (pre in vocab or pre in _LEXICON):
                    out.append(e)
                    break
    return out


def _order_kits(es, mode, steer):
    """Entries ordered for one subject: one per drawing kit by ``_tier``, then up to two pictograms."""
    ranked = sorted(es, key=lambda e: (_tier(e, mode, steer), len(e._toks), e.order))
    out, kits, pics = [], set(), 0
    for e in ranked:
        if e.kit == "pictograms":
            if pics < 2:
                out.append(e)
                pics += 1
        elif e.kit not in kits:
            out.append(e)
            kits.add(e.kit)
    return out


def _clay_for(g):
    """The clay entry for a person or animal subject (same species / role), else a clay Character for a person."""
    names = [g.head or ""] + [e.name for e in g.results] + _SYN.get(g.head or "", "").split()
    names = [n.replace("_", " ").lower() for n in names if n]
    for n in names:
        for e in _index():
            if e.kit == "clay" and e.kind in ("animal", "3D character") and e.name.replace("_", " ").lower() == n:
                return e
    if _sem(g) != "person":
        return None
    words = [w.lower() for w in getattr(g, "words", [])] + [g.head or ""]
    build = next((_BUILDS[w] for w in words if w in _BUILDS), "adult")
    name = (g.head or "person").replace('"', "")
    e = Entry("clay", "character", "3D character",
              f'sc.add(clay.random_person(seed=1, build="{build}"), "{name}")',
              "any person as a 3D clay character (a new seed is a new face, hair and outfit)",
              ["person", "character", "3d", "clay"], func="clay.random_person")
    _prepare(e)
    return e


def _refine(groups, gmode, char_steer):
    """Second pass over the subjects: kinds of a head noun for a subject with no entry, a 3D entry first in a 3D
    request, clay for people and animals that are not flat, and stand-in marks for subjects only a pictogram draws."""
    for g in groups:
        if getattr(g, "format", False) or getattr(g, "text_object", False) or getattr(g, "setting_only", False):
            continue
        mode, steer = getattr(g, "mode", gmode), getattr(g, "psteer", None)
        mod = getattr(g, "mod", None)
        if g.fallback and g.head:                                   # "a boat": the sailboat, the rowing boat
            hn = _order_kits([e for e in _head_noun_entries(g.head) if not _other_compound(e, g.head, mod)],
                             mode, steer)
            if hn:
                g.results, g.scores, g.calls, g.fallback = hn[:4], [float(_MIN_SCORE)] * len(hn[:4]), [], None
                g.kind_of = True
        if steer == "3d" and g.results and not any(e.kit in _D3 for e in g.results):
            hn = [e for e in _head_noun_entries(g.head, _D3)      # "a lamp" in a 3D room: a 3D lamp
                  if not _other_compound(e, g.head, mod)]
            if hn:
                g.results = hn[:1] + g.results
                g.scores = [float(_MIN_SCORE)] + list(g.scores)
        sem = _sem(g) if g.results else None
        cs = getattr(g, "own_steer", None) or char_steer
        if sem in ("person", "animal") and g.results:
            if cs == "3d":
                ce = next((e for e in g.results if e.kit == "clay"), None) or _clay_for(g)
                if ce is not None:
                    rest = [(e, s) for e, s in zip(g.results, g.scores) if e is not ce]
                    g.results = [ce] + [e for e, _ in rest]
                    g.scores = [float(_MIN_SCORE)] + [s for _, s in rest]
                    g.char_steer = "3d"
            elif cs == "2d":
                keep = [(e, s) for e, s in zip(g.results, g.scores) if e.kit != "clay"]
                if keep and g.results[0].kit == "clay":
                    clay_ = [(e, s) for e, s in zip(g.results, g.scores) if e.kit == "clay"]
                    g.results, g.scores = [e for e, _ in keep + clay_], [s for _, s in keep + clay_]
                g.char_steer = "2d"
        if mode == "flat" and g.results and g.head and all(e.kit in _D3 for e in g.results):
            pm = _pictograms()                              # "a smartphone app icon": the flat mark first
            names = list(pm.find(g.head))[:1] if pm is not None and hasattr(pm, "find") else []
            pe = [e for n_ in names for e in _index() if e.kit == "pictograms" and e.name == n_]
            if pe:
                g.results = pe[:1] + list(g.results)
                g.scores = [float(_MIN_SCORE)] + list(g.scores)
        g.results, g.scores = g.results[:4], list(g.scores)[:4]
        if g.results and mode != "flat" and g.results[0].kit == "pictograms":
            g.standin = True                                        # only a pictogram: drawn shaded, full size
            rel = _KIN.get(g.head or "", "").split()               # its own relatives only: no car for a boat
            s3 = next((e for n in rel for e in _index() if e.kit == "solid" and e.kind == "prefab" and e.name == n
                       and e not in g.results), None) if not any(e.kit == "solid" for e in g.results) else None
            if s3 is not None:                                      # then a solid prefab of a relative
                g.results = g.results[:1] + [s3] + g.results[1:3]
                g.scores = list(g.scores[:1]) + [float(_MIN_SCORE)] + list(g.scores[1:3])


def _tint(e, colors):
    if colors:
        return colors[0][1]
    return _PICTO_TINT.get(e.keywords[0] if e.keywords else "", "#4a7bb7")


def _shaded_call(e, colors, at="(0.5, 0.85)", size=0.32):
    """A pictogram as an illustration: lit and shaded, standing on ``at``, with a soft shadow."""
    return (f'pictograms.pictogram(cv, "{e.name}", at={at}, size={size}, color="{_tint(e, colors)}", '
            f'style="shaded", shadow=True, anchor="bm")')


def _with_colour(e, call, colors):
    """The request's colour inside the call when the kit takes one (pictogram colour, coat, clay colour)."""
    if not colors:
        return call
    hexc = colors[0][1]
    if e.kit == "pictograms":
        return call.replace('color="#1d2633"', f'color="{hexc}"', 1)
    if e.kit == "creature" and e.kind == "animal" and "coat=" not in call:
        return call.replace("size=0.4", f'size=0.4, coat="{hexc}"', 1)
    if e.kit == "clay" and call.startswith("sc.add(clay.animal(") and "color=" not in call:
        return re.sub(r'clay\.animal\("([^"]+)"\)', lambda m: f'clay.animal("{m.group(1)}", color="{hexc}")', call,
                      count=1)
    return call


# ------------------------------------------------------------------ find
def _skip_group(g):
    """Groups that never name the picture's subject: a medium, a palette, a material, the light, the weather."""
    if g.fallback:
        return False
    if not g.results:
        return True
    e = g.results[0]
    return (e.kit in ("stylize", "palette", "texture") or e.kind in ("sky", "theme", "palette", "material")
            or e.name.replace(" ", "_") in _ATMOS or (e.kit == "backdrop" and e.name in ("paper", "glass", "ink"))
            or getattr(g, "text_object", False))


def find(query, n=3):
    """The answer to a whole request in one call -> ``Found``.

    ``text_to_draw`` (quoted strings, dates, times, prices, names) is cut out first and the style is read from its
    construction ("in watercolour", "a pencil sketch of"); the rest is split into noun phrases in the order they
    appear, each with its count, colours and pose. Every phrase's head noun is a subject: the kit that draws it as an
    illustration answers it (clay / solid first when the request asks for 3D or toys, the pictogram first in designs
    and flat marks, and always as the "flat mark" alternative); a subject with no entry gets the nearest entries of
    its meaning class and a recipe. ``kind`` is the picture's format or the SKILL.md row of the main subject."""
    index = _index()
    colors_tbl = _colors()
    clean, content = extract_content(query)
    qtext = dict(_CONTENT_RE)["quote"].sub(lambda m: " " * len(m.group(0)), query)
    style, spans = _style_of(qtext)                  # "in the style of Van Gogh": the name is the style's
    content = [c for c in content if not any(c in qtext[s0:e0] for s0, e0 in spans)]
    text = _blank(clean, spans)
    phrases = _parse(text)
    chunks = [(p.words, p.prev) for p in phrases]
    fmt = _format_kind(chunks) if chunks else None
    base_kind = fmt or _legacy_kind(text)
    gmode = "flat" if fmt in _DESIGN_KINDS else "illus"
    gsteer = _steer_of(re.findall(r"[A-Za-z0-9&'’\-]+", clean))
    prefer = set(_SCENE_KITS.get(base_kind, ()))
    kit_rank = {k: i for i, k in enumerate(_KIT_ORDER)}
    ctx = {"index": index, "prefer": prefer, "kit_rank": kit_rank,
           "fmt_kits": set(_SCENE_KITS.get(fmt, ())) & set(_FORMAT_KITS) if fmt in _NO_RECIPE_FORMATS else set(),
           "app": bool(set(_norm_tokens(text)) & {"app", "ui", "toolbar", "menu", "button", "line", "favicon"})}
    vocab = _vocab()
    quotes = [c for c in content if re.search(r"(?<![A-Za-z])['\"“‘]" + re.escape(c), query)]
    used_text = set()
    groups, colors, poses_of = [], [], {}
    frame_done = False
    for ci, p in enumerate(phrases):
        words = p.words
        head = _head(words, vocab)
        frame = fmt is not None and not frame_done and (ci == 0 or p.prev == "as" or not groups and all(
            _wrapper_chunk(x.words) or _head(x.words, vocab) is None for x in phrases[:ci]))
        last = _sing(words[-1].lower()) if words else ""
        ctbl = {k: v for k, v in colors_tbl.items() if k != last} if last in _ROUND else colors_tbl
        mod = _modifier(words, head, vocab) if not frame else None
        drop = _FORM_WORDS | (_STEER_WORDS if not frame and head not in _STEER_WORDS else set())
        drop = drop - {mod}                                             # "a pencil case": pencil is part of it
        keep = [w for w in words if _norm_tokens(w) and not set(_norm_tokens(w)) <= drop]
        slots = _slots(keep or words, ctbl)
        chunk_colors = []
        for w in words:
            t = _sing(w.lower())
            if t in colors_tbl and all(t != c for c, _ in chunk_colors):
                chunk_colors.append((t, colors_tbl[t]))
        toks = [t for t in _toks(words) if t not in _STOP and t not in _DET]
        if toks and all(t in colors_tbl or t in ("and", "or") for t in toks) and (ci > 0 or p.prev == "in"):
            colors += [c for c in chunk_colors if c not in colors]      # "in orange and charcoal": colours
            continue
        if head in _ATTRIBUTE and groups:
            continue                                                    # "long blue hair": the girl's
        if head in (_FORM_WORDS | _FLAT_WORDS) and p.prev in ("in", "as") and len(toks) <= 1 and groups:
            continue                                                    # "camels in silhouette"
        if head in _GENERIC and ci < len(phrases) - 1:
            continue                                                    # "nine animals - panda, fox ...": the list
        count = _count_of(words) if fmt not in ("drawing", "floorplan", "chart", "dashboard") else 1
        flat = any(t in _FLAT_WORDS for t in toks) and head not in _FLAT_WORDS
        steer = _steer_of(words) or gsteer
        mode = "flat" if (flat or gmode == "flat") else "illus"
        psteer = _steer_of(words) if mode == "flat" else steer
        wrapper = _wrapper_chunk(words)
        gs = []
        if frame and slots:                                             # the picture's own format comes first
            fw = {_sing(i.rsplit(":", 1)[0]) for i in (_SCENE_WORDS.get(fmt, "") + " " + _FORMAT_EXTRA.get(fmt, "")
                                                         + (" " + _UI_SCENE if fmt == "ui" else "")).split()
                  if float(i.rsplit(":", 1)[1]) >= 2}
            fres = []
            fkits = _FORMAT_KITS + (("solid",) if fmt in ("isometric", "product", "scene3d") else ()) +                 (("form",) if fmt in ("still_life", "interior") else ())
            for cand in ([sl for sl in slots if fw & set(sl.toks)], slots):
                if cand and not fres:
                    fr, fsc, _m = _choose(cand, None, ctx, "flat", None, n)
                    fres = [(e, s) for e, s in zip(fr, fsc) if e.kit in _FORMAT_KITS or
                            (e.kit in fkits and e.kind in ("3D scene", "2.5D object") and e.name in _SCENE_ENTRIES)]
            if not fres and fmt in _SUBJECT_KINDS and (head is None or head in _FORMAT_HEADS or head in _WRAPPER
                                                       or all(t in _FORMAT_HEADS or t in _WRAPPER or t in _STOP or
                                                              t in _STEER_WORDS for t in toks)):
                frame_done = True                                       # "a phone wallpaper of", "a product shot of"
                continue
            if fres:
                fe = fres[0][0]
                known = set(fe._toks) | {t for a in fe._alias_toks for t in a} | \
                    (set(fe._kw) if fe.kit != "vector" else set())      # "a tasks icon": the tasks are the subject
                fs = [sl for sl in slots if known & set(sl.toks) or fw & set(sl.toks)]
                frame_done = True
                title = quotes[0] if quotes else (content[0] if content else None)
                rest = [c for c in content if c != title]
                sub = rest[0] if rest else None
                if title:
                    used_text.add(title)
                g = Group(" ".join(w.lower() for w in words if _norm_tokens(w) and fw & set(
                    t for s in _slots([w], ctbl) for t in s.toks))[:24] or fres[0][0].name,
                          [e for e, _ in fres], [s for _, s in fres],
                          [_with_text(e.call, title, sub) for e, _ in fres], 1, [], head)
                g.format = True
                gs.append(g)
                rest_slots = [sl for sl in slots if sl not in fs]
                if rest_slots:                                          # "a fox poster": the fox, named exactly
                    rhead = _head([sl.raw for sl in rest_slots], vocab)
                    rr, rs, _m = _choose(rest_slots, rhead, ctx, mode, psteer, n, exact=True)
                    keep_r = [(e, x) for e, x in zip(rr, rs) if fmt == "ui" or
                              e.kind not in ("layout template", "ui template")]
                    rr, rs = [e for e, _ in keep_r], [x for _, x in keep_r]     # "a weather icon": no weather app
                    if not (rr and _full(rr[0]._toks, rest_slots, True)) and rhead in _KIN:   # "a weather icon"
                        rel = [e for nm in _KIN[rhead].split() for e in index
                               if e.kit == "pictograms" and e.name.replace(" ", "_") == nm]
                        if rel:
                            gk = Group(rhead, rel[:3], [float(_MIN_SCORE)] * len(rel[:3]),
                                       [e.call for e in rel[:3]], count, chunk_colors, rhead)
                            gk.kin = True
                            gs.append(gk)
                    if rr and _full(rr[0]._toks, rest_slots, True):
                        gs.append(Group(rhead or rr[0].name, rr, rs, [e.call for e in rr], count, chunk_colors,
                                        rhead))
                groups += gs
                continue
        if wrapper or not slots:
            colors += [c for c in chunk_colors if c not in colors]
            continue
        if head is None and not frame:
            colors += [c for c in chunk_colors if c not in colors]      # "a jazz night", "big outlined letters"
            continue
        ctx["mod"] = mod
        res, scs, modifier = _choose(slots, head, ctx, mode, psteer, n,
                                     exact=gmode == "flat" and fmt in _NO_RECIPE_FORMATS)
        ctx["mod"] = None
        if groups and res and res[0].kind in ("layout template", "ui template"):
            keep_r = [(e, x) for e, x in zip(res, scs) if e.kind not in ("layout template", "ui template")]
            res, scs = [e for e, _ in keep_r], [x for _, x in keep_r]       # a whole-page template is no object
        cls = _CLASS_OF.get(head) if head else None
        if head in _SETTING_KIND and res and res[0].kit == "pictograms" and not _full(res[0]._toks, slots, True):
            res, scs = [], []
        if cls == "person" or (res and _sem_of_entry(res[0]) == "person"):
            if not any(e.kit == "figure" for e in res):                 # any person: figure.person("a knight")
                fp = next((e for e in index if e.kit == "figure" and head and e.kind == "person" and
                           (head in e._kw or head in e._toks) and e.name != "person"), None) or _figure_person()
                if fp is not None:
                    at = 0 if (psteer != "3d" and not (mode == "flat" and res and res[0].kit == "pictograms"))                         else min(1, len(res))
                    res, scs = res[:at] + [fp] + res[at:], scs[:at] + [float(_MIN_SCORE)] + scs[at:]
                    res, scs = res[:n + 1], scs[:n + 1]
        if head in _TEXT_OBJECTS:                                       # a sign that carries the request's text
            q = _quote_after(query, head, used_text) or next((c for c in quotes if c not in used_text), None)
            te = _text_entry()
            if q and te is not None:
                used_text.add(q)
                rest = [(e, s) for e, s in zip(res, scs) if e.kit != "text"]
                g = Group(head, [te] + [e for e, _ in rest], [100.0] + [s for _, s in rest],
                          [_with_text(te.call, q)] + [e.call for e, _ in rest], count, chunk_colors, head)
                g.text_object = True
                gs.append(g)
                res = []
        if head in _TEXT_OBJECTS and not gs and not res:
            continue
        if res:
            if cls == "clothing" or (res[0].kit == "pictograms" and res[0].keywords[:1] == ("clothing",)):
                if any(_sem(x) == "person" for x in groups):
                    continue                                            # "a chef in a white jacket": the chef's
            label = " ".join(dict.fromkeys(w.lower() for w in words if _content_word(w) and
                                           not (set(_norm_tokens(w)) & (_STEER_WORDS | set(ctbl) | _EVENTS))
                                           and "-" not in w.strip("-")))[:24]
            g = Group(label or head or res[0].name, res, scs, [], count, chunk_colors, head)
            g.poses = list(p.poses)
            g.words = words
            g.slots = slots
            gs.append(g)
        elif not gs and head and head not in colors_tbl and len(head) > 2 and head not in _STOP \
                and head not in _ABSTRACT and fmt not in _NO_RECIPE_FORMATS and \
                not (gmode == "flat" and cls is None and head not in _KIN):
            if cls == "clothing" and any(_sem(x) == "person" for x in groups):
                continue
            near = [e for _sc, e in _scored(slots, ctx) if e.kit in ("elements", "backdrop") and
                    _full(e._toks, slots, True)] if cls is None and head in _SETTING_KIND else []
            if near:                                                    # "an underwater coral reef": underwater
                g = Group(" ".join(dict.fromkeys(w.lower() for w in words if _content_word(w)))[:24], near[:2],
                          [float(_MIN_SCORE)] * len(near[:2]), [e.call for e in near[:2]], count, chunk_colors, head)
                gs.append(g)
            elif cls is None and head in _SETTING_KIND:
                g = Group(head, [], [], [], count, chunk_colors, head, None)
                g.setting_only = True                                  # "a street": the setting, drawn by the kind
                gs.append(g)
            else:
                g = Group(head, [], [], [], count, chunk_colors, head,
                          _fallback(head, cls, None, any(t in _FORM_WORDS for t in toks)))
                g.poses = list(p.poses)
                gs.append(g)
        if modifier is not None and all(modifier is not x.results[0] for x in groups + gs if x.results):
            gs.append(Group(modifier.name, [modifier], [float(_MIN_SCORE)], [modifier.call], 1, [], None))
        for g in gs:
            if g.results and not g.calls:
                g.calls = [e.call for e in g.results]
        merged = []
        for g in gs:                                                    # "a kite festival ... three kites": one kite
            twin = next((x for x in groups if g.results and x.results and x.results[0].id == g.results[0].id
                         and x.head == g.head), None)
            if twin is None and (g.fallback or getattr(g, "setting_only", False)):
                twin = next((x for x in groups if not x.results and x.head == g.head), None)
            if twin is None and g.results:                              # "a living room, two bedrooms": rooms
                twin = next((x for x in groups if x.results and x.results[0].id == g.results[0].id), None)
                if twin is not None and g.results[0].kit in _FORMAT_KITS:
                    continue                                            # the format named twice: once
                if twin is not None:
                    twin.count += g.count
                    if g.head and g.head not in twin.phrase:
                        twin.phrase = (twin.phrase + ", " + g.head)[:24]
                    continue
            if twin is not None:
                twin.count = max(twin.count, g.count)
                twin.poses = list(dict.fromkeys(getattr(twin, "poses", []) + getattr(g, "poses", [])))
                continue
            merged.append(g)
        if merged:
            merged[0].colors = chunk_colors[:2]
        for g in merged:
            g.mode, g.psteer, g.own_steer = mode, psteer, _steer_of(words)
            g.mod = mod if g.head == head else None
        for w, h in chunk_colors:
            if all(w != c for c, _ in colors):
                colors.append((w, h))
        groups += merged
    groups = [g for g in groups if not getattr(g, "setting_only", False) or len(groups) == 1 or True]
    # poses: a phrase's pose goes to its subject, or to the nearest subject before it that can take one
    for i, g in enumerate(groups):
        ps = getattr(g, "poses", [])
        if not ps:
            continue
        target = g if g.results and any(any(p in e.poses for p in ps) for e in g.results) else next(
            (x for x in reversed(groups[:i]) if x.results and any(any(p in e.poses for p in ps) for e in x.results)
             and not getattr(x, "posed", False)), None)
        if target is not None and target is not g:
            target.poses = list(dict.fromkeys(getattr(target, "poses", []) + ps))
            g.poses = []
        if target is not None:
            target.posed = True
    char_steer = _char_steer(query, gmode, style)
    _refine(groups, gmode, char_steer)
    for g in groups:
        if not g.results or getattr(g, "format", False) or getattr(g, "text_object", False):
            continue
        g.calls = [_with_colour(e, _adapt_call(e, getattr(g, "slots", []), getattr(g, "words", []),
                                               getattr(g, "poses", [])), g.colors)
                   for e in g.results]
        if getattr(g, "standin", False):
            g.calls[0] = _shaded_call(g.results[0], g.colors)
        nm = (g.head or "").replace('"', "")
        for i, e in enumerate(g.results):                           # "a woman": recorded as the woman
            if e.kit == "figure" and nm and nm not in ("person", e.name) and g.calls[i].startswith("figure.person(") \
                    and "name=" not in g.calls[i]:
                g.calls[i] = re.sub(r'^figure\.person\(("[^"]*")\)', lambda m: f'figure.person({m.group(1)}, name="{nm}")',
                                    g.calls[i], count=1)
        g0 = g.results[0]
        if g0.id == "form.ball" and g.head in _ROUND:
            base, mat = _ROUND[g.head]
            g.calls[0] = f'form.ball(cv, (0.5, 0.6), 0.08, base="{base}", material="{mat}")'
            g.colors = [c for c in g.colors if c[0] != g.head]
        if g0.kit == "chart" and quotes:
            q = _quote_after(query, g.head or "chart", used_text)
            if q:
                used_text.add(q)
                g.calls[0] = _with_text(g.calls[0], q)
    groups = groups[:12]
    avail = _scene_kinds()
    allt = set(_toks([w for ws, _ in chunks for w in ws]))
    allt |= {_sing(s) for t in list(allt) for s in _SYN.get(t, "").split()}
    real = [g for g in groups if not _skip_group(g) and not getattr(g, "format", False)]
    main = next((g for g in real if not getattr(g, "setting_only", False)), None)
    main = main or next((g for g in groups if not _skip_group(g)), None) or         next((g for g in groups if getattr(g, "setting_only", False)), None)
    if fmt is not None and fmt not in _SUBJECT_KINDS:
        main = next((g for g in groups if getattr(g, "format", False)), main)
    if main is None and groups:
        main = groups[0]
    settings = [g.head for g in groups if g is not main and g.head in _SETTING_KIND]
    kind = fmt
    if kind == "scene3d" and main is not None and _sem(main) in ("person", "animal"):
        kind = None                                                     # "a toy-like 3D baker": characters
    if kind is None and main is not None:
        first = next((g for g in groups if not _skip_group(g) or getattr(g, "setting_only", False)), None)
        if first is not None and getattr(first, "setting_only", False) and _sem(main) not in ("person", "animal"):
            kind = _SETTING_KIND.get(first.head)                       # "a Japanese garden: a maple, a lantern"
        else:
            kind = _subject_kind(main, settings, groups, allt, gsteer)
    if kind not in avail:
        kind = None
    if kind is None and main is None:
        kind = _legacy_kind(text)
        if kind not in avail:
            kind = None
    extras = []
    if main is not None and kind in _SUBJECT_KINDS + ("characters", "pets"):
        mkit = main.results[0].kit if main.results else ""
        for g in groups:
            if g is main or g.fallback or _skip_group(g) or not g.results:
                continue
            e = g.results[0]
            if e.kit in ("creature", "figure", "flora", "clay") and e.kit != mkit and \
                    _sem_of_entry(e) in ("person", "animal", "plant"):
                if _kind_of_entry(e, allt) != kind and e.kit not in extras:
                    extras.append(e.kit)
    found = Found(query, groups, colors, style, _route(kind, query, style, main, groups, avail), content, main,
                  extras[:2], fmt is not None and fmt not in _SUBJECT_KINDS)
    found.kit_kind = kind                                   # the kind a kit-first picture would take
    return found


# ------------------------------------------------------------------ the route: freehand unless structure is hard
# A capable agent draws pictures freehand (kind "illustration": illustrate.Sketch + Canvas.draw). Kits lead where
# structure is the hard part (charts, diagrams, drawings, UI, text layout) and where the request asks for 3D.
_STRUCTURE_KINDS = {"diagram", "chart", "dashboard", "infographic", "ui", "icons", "drawing", "floorplan", "banner",
                    "poster", "card", "comic", "figure", "animation", "photo", "wallpaper"}
_3D_KINDS = {"characters", "pets", "scene3d", "isometric", "product"}
_3D_WORDS = re.compile(r"\b(?:product\s+(?:shot|render|photo|image)|packshot|studio\s+(?:shot|lighting|light|backdrop)|3-d)\b")
_PANEL_WORDS = re.compile(r"\b(?:strip|panels?|\w+-panel|page|speech|bubbles?|dialogue|says|saying|storyboard)\b")
_ART_WORDS = re.compile(r"\b(?:poster art|art poster|art print|artwork|illustrated poster|painting|illustration)\b")
_STICKER_WORDS = re.compile(r"\bstickers?\b")


def _asks_3d(query):
    """True when the request asks for 3D: 3D, clay, toy, isometric, a render, a product shot ("a clay pot" does not)."""
    low = query.lower()
    if _3D_WORDS.search(low):
        return True
    parts = [p.strip("'’") for w in re.findall(r"[a-z0-9'’\-]+", low) for p in re.split(r"-", w) if p.strip("'’")]
    for i, w in enumerate(parts):
        nxt = parts[i + 1] if i + 1 < len(parts) else ""
        if w in _STEER_3D and not (w in _NOT_STEER and nxt in _NOT_STEER[w]):
            return True
    return False


def _route(kind, query, style, main, groups, avail):
    """The kind after the freehand rule: pictures (people, animals, scenes, a named style, poster art, stickers) get
    ``illustration``; a 3D request keeps its 3D kind; structure kinds stay."""
    if "illustration" not in avail:
        return kind
    low = query.lower()
    pictorial = [g for g in groups if not getattr(g, "format", False) and not _skip_group(g) and
                 (g.fallback or not g.results or _sem(g) in ("person", "animal", "plant", "object", "setting",
                                                             "effect"))]
    if kind in _STRUCTURE_KINDS:
        if kind == "comic" and not _PANEL_WORDS.search(low) and not _asks_3d(query):
            return "illustration"                                   # "a comic-style fox": a style, not a strip
        if kind == "poster" and (_ART_WORDS.search(low) or style and pictorial):
            return "illustration"                                   # poster art: the picture is the poster
        if kind in ("icons", "card", "poster") and _STICKER_WORDS.search(low) and pictorial:
            return "illustration"                                   # a sticker sheet
        return kind
    if _asks_3d(query):
        if kind in _3D_KINDS:
            return kind
        if main is not None and _sem(main) == "person":
            return "characters" if "characters" in avail else kind
        if main is not None and _sem(main) == "animal":
            return "pets" if "pets" in avail else kind
        return "scene3d" if kind is None and "scene3d" in avail and pictorial else kind
    if kind is None:
        return "illustration" if (pictorial or style or _STICKER_WORDS.search(low)) else None
    return "illustration"


_PICTO_KIND = {"home": "still_life", "transport": "scene3d", "buildings": "landscape", "nature": "nature",
               "animals": "animals", "people": "people", "tech": "product", "holidays": "still_life"}


def _subject_kind(main, settings, groups, allt, gsteer):
    """The kind of a picture whose main subject is ``main``: the subject's row in SKILL.md's kit table (people and
    animals: figure / creature, clay when the request asks for 3D); an object in a named setting takes the
    setting's kind."""
    sem = _sem(main)
    steer = getattr(main, "char_steer", None) if sem in ("person", "animal") else None
    steer = steer or _steer_of(getattr(main, "words", [])) or gsteer
    tok = set(allt) | {sem}
    if steer == "3d" and sem in ("object", "setting", "effect"):
        return _row_kind("solid", tok)
    if main.head in _SETTING_KIND and sem not in ("person", "animal", "plant"):
        return _SETTING_KIND[main.head]
    if sem in ("person", "animal"):
        kit = "clay" if steer == "3d" else "figure" if sem == "person" else "creature"
        return _row_kind(kit, {sem} if kit == "clay" else tok)       # a cat with a girl: the cat's row
    if sem == "plant":
        return _row_kind("flora", tok)
    if settings and sem in ("object", "setting", "material"):
        return _SETTING_KIND[settings[0]]
    if main.fallback:
        cls = main.fallback.get("class")
        if steer == "3d":
            return _row_kind("solid", tok)
        return _CLASSES[cls][3] if cls in _CLASSES else "still_life"
    e0 = main.results[0]
    if steer == "3d" and sem == "object":
        return _row_kind("solid", tok)
    if e0.kit == "pictograms":                        # "icons, or the scene's kind"
        for g in groups:
            if g is not main and g.results and g.results[0].kit != "pictograms" and _sem(g) in ("person", "animal",
                                                                                                "plant"):
                return _kind_of_entry(g.results[0], allt)
        if main.head in _CLASS_OF and _CLASS_OF[main.head] in _CLASSES:
            return _CLASSES[_CLASS_OF[main.head]][3]
        cat = e0.keywords[0] if e0.keywords else ""
        return _PICTO_KIND.get(cat, "still_life")
    return _kind_of_entry(e0, allt)


# --------------------------------------------------------------------------------------------- show
def _resolve_func(dotted):
    """'flora.plant' / 'ui.Screen.button' / 'canvas.Canvas.text' -> object (or None)."""
    parts = dotted.split(".")
    try:
        obj = importlib.import_module(f"pixelcanvas.{parts[0]}")
    except Exception:
        return None
    for p in parts[1:]:
        obj = getattr(obj, p, None)
        if obj is None:
            return None
    return obj


def _lookup(name):
    """Entries matching 'kit.name', 'name', or 'kit.name with spaces' (underscores and spaces are the same)."""
    key = name.strip().lower().replace(" ", "_")
    idx = _index()
    hits = [e for e in idx if e.id.lower() == key]
    if hits:
        return hits
    return [e for e in idx if e.name.lower().replace(" ", "_") == key]


def _func_text(dotted, obj, ref):
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
        sig = "(...)"
    doc = (inspect.getdoc(obj) or "").strip().split("\n")
    doc = [ln for ln in doc][:14]
    head = dotted.rsplit(".", 1)[-1]
    lines = [f"{dotted}{sig}"[:400], ""] + ["  " + ln for ln in doc]
    ex = _example(ref, head + "(")
    if ex:
        lines += ["", "example:  " + ex]
    return lines


def show(name):
    """Text about one entry (``"flora.oak"``) or function (``"plant"``, ``"flora.plant"``, ``"ui.Screen.button"``)."""
    hits = _lookup(name)
    lines = []
    if hits:
        e = hits[0]
        lines += [f"{e.id}  ({e.kind})  {e.summary}", ""]
        lines += ["snippet:"] + ["  " + ln for ln in e.snippet().split("\n")]
        if e.aliases:
            lines += ["", "aliases:  " + ", ".join(e.aliases[:12])]
        kws = [k for k in e.keywords if k][:16]
        if kws:
            lines += ["keywords: " + ", ".join(kws)]
        obj = _resolve_func(e.func) if e.func else None
        if obj is not None:
            lines += [""] + _func_text(e.func, obj, e.ref)
        if e.ref:
            sec = _section(e.ref, [f"`{e.name}`", e.func.rsplit(".", 1)[-1] + "(" if e.func else "", e.name])
            lines += ["", f"read more: skills/pixelcanvas/{e.ref}" + (f"  § {sec}" if sec else "") if e.ref !=
                      "SKILL.md" else "read more: skills/pixelcanvas/SKILL.md"]
        if len(hits) > 1:
            lines += ["", "also: " + ", ".join(h.id for h in hits[1:8])]
        return "\n".join(lines)
    # a function: kit.func or bare func
    key = name.strip()
    cands = []
    if "." in key:
        obj = _resolve_func(key)
        if obj is not None:
            cands.append((key, obj))
    else:
        for kit in kit_names():
            if kit == "scenes":
                continue
            try:
                mod = importlib.import_module(f"pixelcanvas.{kit}")
            except Exception:
                continue
            obj = getattr(mod, key, None)
            if obj is not None and (inspect.isfunction(obj) or inspect.isclass(obj)) and \
                    getattr(obj, "__module__", "") == mod.__name__:
                cands.append((f"{kit}.{key}", obj))
    if not cands:
        near = difflib.get_close_matches(key.lower(), [e.id for e in _index()] + [e.name for e in _index()], n=5,
                                         cutoff=0.6)
        return f"nothing called {name!r}" + (f"; did you mean: {', '.join(near)}" if near else
                                             f"; try: python tools/kit.py find \"{name}\"")
    dotted, obj = cands[0]
    kit = dotted.split(".")[0]
    ref = _ref_page(kit)
    lines += _func_text(dotted, obj, ref)
    if ref:
        sec = _section(ref, [dotted.rsplit(".", 1)[-1] + "("])
        lines += ["", f"read more: skills/pixelcanvas/{ref}" + (f"  § {sec}" if sec else "")]
    if len(cands) > 1:
        lines += ["", "also: " + ", ".join(d for d, _ in cands[1:])]
    return "\n".join(lines)


def to_json(obj):
    return json.dumps(obj, ensure_ascii=False, indent=1)
