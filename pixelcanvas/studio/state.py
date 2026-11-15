# -*- coding: utf-8 -*-
"""studio state: the picture folder ``<dir>/NAME/`` = picture.json + layers/<cell>.py + cache/ + history/ +
preview.png. Every write is atomic (a temporary file, then a rename); every change is a revision whose
picture.json and cell sources (content-addressed) are kept in history/, so undo restores code too."""
from __future__ import annotations

import json
import os
import re
import time

from .engine import sha
from .compositor import LAYER_DEFAULTS, SCENE_DEFAULTS

FORMAT = 1
PICTURE = "picture.json"
CURRENT = ".current"
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]{0,47}$")


class StudioError(Exception):
    """A mistake the agent can fix: printed as one ERROR line plus the command to run instead."""

    def __init__(self, message, fix=None):
        super().__init__(message)
        self.fix = fix


def atomic_write(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    for i in range(20):                                  # Windows: a reader may hold the file for a moment
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 * (i + 1))
    os.replace(tmp, path)


def replace_file(tmp, path):
    """os.replace that waits for a reader on Windows (an image viewer holding preview.png); True when done."""
    for i in range(20):
        try:
            os.replace(tmp, path)
            return True
        except PermissionError:
            time.sleep(0.05 * (i + 1))
    try:
        os.remove(tmp)
    except OSError:
        pass
    return False


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def check_name(name, what="layer"):
    if not NAME_RE.match(str(name or "")):
        raise StudioError(f"{what} name {name!r}: use letters, digits, '_' or '-', starting with a letter "
                          "(e.g. fox, lantern, title_2)")
    return name


def layer_key(text):
    """A section or layer name as written ('sled dog', 'Sled-Dog') -> the layer name (sled_dog): spaces and
    hyphens to '_', other punctuation dropped."""
    s = re.sub(r"[\s\-]+", "_", str(text or "").strip())
    return re.sub(r"[^A-Za-z0-9_]", "", s).strip("_")


def first_comment(src):
    """The cell's description: its first line when that is a comment (``# fox: orange, walking right``)."""
    for ln in (src or "").splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#") and not re.match(r"#\s*role\s*[:=]", s) and not s.startswith("#!"):
            return s.lstrip("#").strip()[:240]
        return ""
    return ""


SCENE_KEY = "@scene:"                                # history key of a scene file a cell reads with scene("...")
SCENE_RE = re.compile(r"""\bscene\(\s*(?:r?)(['"])([^'"]+)\1""")


# ============================================================================ the checklist
# The contract comes from the draft: the first comment line of a section is that thing's inventory line,
#   # bus: 1, yellow, overturned on its side, wheels up | on road
# NAME: count, colours, 'quoted text', states seen by eye | places. From the request itself only its quoted
# strings become items (each must be written somewhere).

_NUMBERS = {"a": 1, "an": 1, "one": 1, "single": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "a couple of": 2,
            "a pair of": 2, "pair of": 2, "a couple": 2, "a pair": 2, "a dozen": 12, "dozen": 12, "several": 3}
# relation words, longest first ("in front of" is never read as "in"); -> the word the checker measures
_RELS = (("to the left of", "left of"), ("to the right of", "right of"), ("in front of", "in front of"),
         ("on top of", "on"), ("next to", "next to"), ("left of", "left of"), ("right of", "right of"),
         ("behind", "behind"), ("beside", "next to"), ("underneath", "below"), ("beneath", "below"),
         ("under", "below"), ("below", "below"), ("above", "above"), ("over", "above"), ("inside", "in"),
         ("on", "on"), ("in", "in"), ("at", "at"))
FRAME = ("top left", "top right", "bottom left", "bottom right", "left", "right", "top", "bottom", "centre",
         "center", "middle")
_Q = (r"""(?:(?<=^)|(?<=[\s,(:|]))'([^']+?)'(?=$|[\s,.;:|)!?])""",
      r'"([^"]+?)"', "“([^”]+?)”", "‘([^’]+?)’(?=$|[\\s,.;:|)!?])")


def item_name(cell):
    """The thing a section names when it has no inventory line: 'sled_dog' -> 'sled dog'."""
    return re.sub(r"[_\-]+", " ", str(cell)).strip()


def _quotes(text):
    """Quoted strings of an inventory line ('OPEN', "OPEN", curly quotes; an apostrophe is no quote)."""
    out = []
    for p in _Q:
        for m in re.finditer(p, text):
            out.append((m.start(), m.end(), m.group(1).strip()))
    out.sort()
    keep, last = [], -1
    for a, b, s in out:
        if a >= last and s:
            keep.append((a, b, s))
            last = b
    return keep


def _colour(bit):
    """The colour words of a bit ('yellow', 'dark green', 'red and white', '#c84') or None."""
    from .. import manifest as M
    b = bit.strip().lower()
    if re.fullmatch(r"#?[0-9a-f]{6}|#[0-9a-f]{3}", b):
        try:
            from ..color import rgb
            return [M.colour_name(rgb(b if b.startswith("#") else "#" + b)[:3])]
        except Exception:                                # noqa: BLE001
            return None
    out = []
    for part in re.split(r"\s+and\s+|\s*/\s*", b):
        words = part.split()
        if not words:
            return None
        c = M._colour_of("-".join(words))
        if c is None:
            return None
        out.append(c["word"].replace("-", " "))
    return out or None


def _place(bit):
    """(relation, target) when a bit starts with a relation word ('in front of bus', 'at top left'), else None."""
    b = " ".join(bit.lower().split())
    for word, rel in _RELS:
        if b.startswith(word + " ") or (rel == "at" and b == word):
            target = b[len(word):].strip()
            target = re.sub(r"^(?:the|a|an)\s+", "", target)
            target = re.sub(r"\s+of\s+the\s+(?:frame|picture|image|scene|canvas)$", "", target)
            if rel == "at":
                target = re.sub(r"\s+(?:side|corner|edge|third)$", "", target)
                if target not in FRAME:
                    return None
            if not target:
                return None
            return rel, target
    return None


def _split_bits(text):
    """Split at commas outside quotes."""
    marks = {}
    t = text
    for a, b, s in reversed(_quotes(text)):
        k = f"\x00{len(marks)}\x00"
        marks[k] = t[a:b]
        t = t[:a] + k + t[b:]
    out = []
    for part in re.split(r"[,;]", t):
        for k, v in marks.items():
            part = part.replace(k, v)
        part = part.strip()
        if part:
            out.append(part)
    return out


def inventory(line, cell):
    """An inventory line -> {name, count, colours, states, places, texts}; None when the line is no inventory
    line (a plain comment). ``cell`` names the thing when the line starts at its bits ('3, green | in bowl')."""
    s = " ".join(str(line or "").split())
    if not s:
        return None
    m = re.match(r"^([^:|'\"]{1,48}?)\s*:\s*(.*)$", s)
    if m and len(m.group(1).split()) <= 5 and not re.search(r"[(){}\[\]=#]", m.group(1)):
        name, rest = m.group(1).strip(), m.group(2).strip()
    elif "|" in s:
        name, rest = item_name(cell), s
    else:
        return None
    head, _, tail = rest.partition("|")
    out = {"name": name, "count": None, "colours": [], "states": [], "places": [], "texts": []}
    bits = _split_bits(head)
    for i, bit in enumerate(bits):
        low = bit.lower()
        q = _quotes(bit)
        if q:
            out["texts"] += [x[2] for x in q]
            rest_ = bit
            for a, b, t in reversed(q):
                rest_ = rest_[:a] + rest_[b:]
            rest_ = re.sub(r"^\s*(?:says|reads|reading|saying|text|the text|written|labelled|labeled|with)\b", "",
                           rest_.strip(), flags=re.I).strip(" :-")
            if rest_ and not re.fullmatch(r"(?:on it|in it|the words?|words?)", rest_.lower()):
                out["states"].append(bit)
            continue
        if i == 0:                                       # a leading count: 3, three, a pair of
            nums = "|".join(sorted((re.escape(k) for k in _NUMBERS), key=len, reverse=True))
            mm = re.fullmatch(r"(\d{1,3}|" + nums + r")(?:\s+(.*))?", low)
            if mm and not (mm.group(1) in ("a", "an") and mm.group(2)):
                word, more = mm.group(1), mm.group(2)
                out["count"] = int(word) if word.isdigit() else _NUMBERS[word]
                if not more:
                    continue
                bit = bit[len(bit) - len(more):]
        col = _colour(bit)
        if col:
            out["colours"] += [c for c in col if c not in out["colours"]]
            continue
        pl = _place(bit)
        if pl:
            out["places"].append(list(pl))
            continue
        out["states"].append(bit)
    for bit in _split_bits(tail):
        pl = _place(bit)
        if pl:
            out["places"].append(list(pl))
        else:                                            # 'centre', 'left' alone: a place in the frame
            low = " ".join(bit.lower().split())
            low = re.sub(r"^(?:the\s+)", "", low)
            out["places"].append(["at", low] if low in FRAME else ["near", low])
    return out


def inventory_line(src):
    """The section's inventory line: its first comment line, when it reads as one."""
    for ln in (src or "").splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#") and not re.match(r"#\s*role\s*[:=]", s) and not s.startswith(("#!", "# %%", "#%%")):
            text = s.lstrip("#").strip()
            return text if inventory(text, "x") is not None else None
        if s.startswith("#"):
            continue
        return None
    return None


def thing_item(cell, line):
    """The checklist item of one section (its inventory line, or the plain name)."""
    inv = inventory(line, cell) if line else None
    if inv is None:
        inv = {"name": item_name(cell), "count": None, "colours": [], "states": [], "places": [], "texts": []}
        text = inv["name"]
    else:
        text = " ".join(line.split())
        if not re.match(r"^[^:|]{1,48}:", text):
            text = f"{inv['name']}: {text}"
    return {"kind": "thing", "key": inv["name"].lower(), "name": inv["name"], "text": text,
            "count": inv["count"], "colours": inv["colours"], "states": inv["states"], "places": inv["places"],
            "texts": inv["texts"], "cell": cell, "line": line or "", "src": "draft", "layer": None, "ok": None,
            "why": None, "on": None}


def request_needs(prompt):
    """The items the request itself gives: each quoted string (it must be written in the picture)."""
    from .. import manifest as M
    out = []
    for q in M.quoted(prompt or ""):
        out.append({"kind": "text", "key": q, "text": f"text '{q}'", "on": None, "src": "request", "layer": None,
                    "ok": None, "why": None})
    return out


def _same_text(a, b):
    return " ".join(str(a).split()).casefold() == " ".join(str(b).split()).casefold()


def has_code(src):
    """True when a section holds code, not only comments (a remark under ``# %% finish`` is no item, no error)."""
    import ast
    try:
        body = ast.parse(src or "").body
    except (SyntaxError, ValueError):
        return True                                      # its run reports the syntax error
    return any(not (isinstance(b, ast.Expr) and isinstance(getattr(b, "value", None), ast.Constant)
                    and isinstance(b.value.value, str)) for b in body)


def remark_only(src):
    """A section of comments only and no ``NAME: ...`` inventory line (a remark under ``# %% finish``): skipped."""
    if has_code(src):
        return False
    line = inventory_line(src)
    return not (line and re.match(r"^[^:|]{1,48}:", line))


def rebuild_needs(data, cells, role_of):
    """The checklist from the sections (program order) and the request's quoted strings, keeping the items whose
    inventory line did not change (with their ticks and numbers) and every item added by hand. ``role_of(cell)``
    -> 'object' | 'backdrop' | 'pass'."""
    if data.get("whole"):
        return data["needs"]
    old = data.get("needs") or []
    prev = {n.get("cell"): n for n in old if n.get("src") == "draft"}
    dismissed = data.get("dismissed") or {}
    used = []
    for L in data["layers"]:
        c = L.get("cell") or L["name"]
        if c in cells and c not in used:
            used.append(c)
    order = [c for c in data.get("program") or [] if c in used] + [c for c in used
                                                                   if c not in (data.get("program") or [])]
    draft = []
    for c in order:
        if remark_only(cells[c]):                        # only a remark: skipped, said once in the layer list
            continue
        line = inventory_line(cells[c])
        if line is None and role_of(c) in ("backdrop", "pass"):
            continue
        if role_of(c) == "pass":
            continue
        if c in dismissed and dismissed[c] == (line or ""):
            continue
        p = prev.get(c)
        if p is not None and p.get("line", "") == (line or ""):
            draft.append(p)
            continue
        n = thing_item(c, line)
        if p is not None:
            n["id"] = p["id"]                            # the same section keeps its number
        draft.append(n)
    texts = []
    olds = {n["key"]: n for n in old if n.get("src") == "request" and n.get("kind") == "text"}
    for t in request_needs(data.get("prompt")):
        if any(_same_text(t["key"], q) for n in draft for q in n.get("texts") or []):
            continue                                     # the section that writes it carries it: one item
        o = olds.get(t["key"])
        if o is not None:
            t.update(id=o.get("id"), ok=o.get("ok"), why=o.get("why"))
        texts.append(t)
    agent = [n for n in old if n.get("src") not in ("draft", "request")]
    need = draft + texts + agent
    frozen = any(n.get("ok") for n in old) or any(nt.get("item") for nt in data.get("notes") or []) or \
        data.get("reviews") or data.get("dismissed")
    if not frozen:                                       # nothing refers to a number yet: 1, 2, 3 in list order
        for i, n in enumerate(need, 1):
            n["id"] = i
        data["max_id"] = len(need)
        data["needs"] = need
        return need
    nxt = max([int(n.get("id") or 0) for n in old + need] + [int(data.get("max_id") or 0)])
    seen = set()
    for n in need:
        if not n.get("id") or n["id"] in seen:
            nxt += 1
            n["id"] = nxt
        seen.add(n["id"])
    data["max_id"] = max([nxt] + [n["id"] for n in need])
    data["needs"] = need
    return need


# ============================================================================ the picture

class Picture:
    """A studio picture on disk. ``data`` is picture.json; ``cells`` maps cell names to their source."""

    def __init__(self, folder):
        self.folder = os.path.abspath(folder)
        self.data = None

    # ---- paths
    @property
    def path(self):
        return os.path.join(self.folder, PICTURE)

    @property
    def layers_dir(self):
        return os.path.join(self.folder, "layers")

    @property
    def cache_dir(self):
        return os.path.join(self.folder, "cache")

    @property
    def history_dir(self):
        return os.path.join(self.folder, "history")

    @property
    def preview(self):
        return os.path.join(self.folder, "preview.png")

    @property
    def name(self):
        return self.data["name"]

    def cell_path(self, cell):
        return os.path.join(self.layers_dir, f"{cell}.py")

    # ---- create / load / save
    @classmethod
    def create(cls, art_dir, name, prompt, size=(1920, 1080), kind="painting", seed=1):
        check_name(name, "picture")
        folder = os.path.join(art_dir, name)
        if os.path.isfile(os.path.join(folder, PICTURE)):
            raise StudioError(f"a picture called {name} already exists in {folder}",
                              fix=f"-p {name} render   (or pick another NAME)")
        pic = cls(folder)
        os.makedirs(pic.layers_dir, exist_ok=True)
        os.makedirs(pic.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(pic.history_dir, "src"), exist_ok=True)
        pic.data = {"pcstudio": FORMAT, "name": name, "prompt": prompt or "", "size": [int(size[0]), int(size[1])],
                    "kind": kind or "painting", "seed": int(seed), "scene": dict(SCENE_DEFAULTS),
                    "layers": [], "needs": request_needs(prompt), "notes": [], "rev": 0, "program": []}
        for i, n in enumerate(pic.data["needs"], 1):
            n["id"] = i
        return pic

    def load(self):
        d = read_json(self.path)
        if not isinstance(d, dict) or "pcstudio" not in d:
            raise StudioError(f"{self.path} is not a studio picture (picture.json missing or unreadable)")
        sc = dict(SCENE_DEFAULTS)
        sc.update(d.get("scene") or {})
        d["scene"] = sc
        d.setdefault("layers", [])
        d.setdefault("needs", [])
        d.setdefault("notes", [])
        d.setdefault("rev", 0)
        self.data = d
        return self

    def save(self):
        atomic_write(self.path, json.dumps(self.data, ensure_ascii=False, indent=1))

    # ---- cells on disk
    def cell_files(self):
        out = {}
        if os.path.isdir(self.layers_dir):
            for f in os.listdir(self.layers_dir):
                if f.endswith(".py") and not f.startswith(("_", ".")) and NAME_RE.match(f[:-3]):
                    out[f[:-3]] = os.path.join(self.layers_dir, f)
        return out

    def read_cells(self):
        cells = {}
        for stem, p in self.cell_files().items():
            try:
                with open(p, encoding="utf-8-sig") as f:
                    cells[stem] = f.read()
            except (OSError, UnicodeDecodeError):
                continue
        return cells

    def write_cell(self, cell, src):
        if not src.endswith("\n"):
            src += "\n"
        atomic_write(self.cell_path(cell), src)

    # ---- layers
    def names(self):
        return [L["name"] for L in self.data["layers"]]

    def layer(self, name):
        for L in self.data["layers"]:
            if L["name"] == name:
                return L
        alt = layer_key(name)                            # "sled dog" names layer sled_dog
        for L in self.data["layers"]:
            if L["name"] == alt or L["name"].lower() == alt.lower():
                return L
        from .._args import unknown_name
        names = self.names()
        if not names:
            raise StudioError(f"unknown layer {name!r}: the picture has no layers yet")
        raise StudioError(str(unknown_name("layer", name, names, hint="Layers: " + ", ".join(names))))

    def index(self, name):
        return self.names().index(self.layer(name)["name"])

    def new_layer(self, name, cell=None, at=None, **params):
        check_name(name)
        if name in self.names():
            raise StudioError(f"a layer called {name} already exists")
        L = {"name": name, "cell": cell or name}
        L.update(params)
        if at is None:
            self.data["layers"].append(L)
        else:
            self.data["layers"].insert(at, L)
        return L

    def sync(self, cells=None, order=None):
        """Pick up new / changed / deleted cell files -> (added, changed, removed) layer names. New cells go on
        top: in ``order`` when given (a draft's section order), else by file time, then name."""
        cells = self.read_cells() if cells is None else cells
        used = {L.get("cell") or L["name"] for L in self.data["layers"]}
        added, changed, removed = [], [], []
        keep = []
        for L in self.data["layers"]:
            cell = L.get("cell") or L["name"]
            if cell not in cells:
                removed.append(L["name"])
                continue
            h = sha(cells[cell])
            if L.get("src") != h:
                if L.get("src") is not None:
                    changed.append(L["name"])
                L["src"] = h
                L["desc"] = first_comment(cells[cell])
            keep.append(L)
        self.data["layers"] = keep
        new = [c for c in cells if c not in used]
        if order:
            rank = {n: i for i, n in enumerate(order)}
            new.sort(key=lambda c: (rank.get(c, len(rank)), c))
        else:
            def mtime(c):
                try:
                    return os.path.getmtime(self.cell_path(c))
                except OSError:
                    return 0.0
            new.sort(key=lambda c: (mtime(c), c))
        for c in new:
            if c in self.names():
                continue
            at = self._slot(c, order) if order else None
            L = self.new_layer(c, at=at)
            L["src"] = sha(cells[c])
            L["desc"] = first_comment(cells[c])
            added.append(c)
        prog = [c for c in (self.data.get("program") or []) if c in cells]
        if order:                                        # a draft's order is the program order; other cells after
            prog = [c for c in order if c in cells] + [c for c in prog if c not in order]
        for L in self.data["layers"]:
            c = L.get("cell") or L["name"]
            if c in cells and c not in prog:
                prog.append(c)
        self.data["program"] = prog
        return added, changed, removed

    def _slot(self, cell, order):
        """Where a new section goes in the stack: above the layers of the section before it in the draft, else
        below those of the section after it, else on top."""
        if cell not in order:
            return None
        i = order.index(cell)
        names = self.names()
        for c in reversed(order[:i]):
            fam = [k for k, L in enumerate(self.data["layers"]) if (L.get("cell") or L["name"]) == c]
            if fam:
                return fam[-1] + 1
        for c in order[i + 1:]:
            fam = [k for k, L in enumerate(self.data["layers"]) if (L.get("cell") or L["name"]) == c]
            if fam:
                return fam[0]
        return len(names)

    # ---- history
    def _index(self):
        return read_json(os.path.join(self.history_dir, "index.json"), {"revs": [], "head": 0})

    def _save_index(self, idx):
        atomic_write(os.path.join(self.history_dir, "index.json"), json.dumps(idx, ensure_ascii=False, indent=0))

    def revs(self):
        return self._index()["revs"]

    def snapshot(self, cmd, cells=None):
        """Record the current state as a new revision (child of the current one); returns its number."""
        cells = self.read_cells() if cells is None else cells
        idx = self._index()
        rev = int(idx.get("head", 0)) + 1
        os.makedirs(os.path.join(self.history_dir, "src"), exist_ok=True)
        hashes = {}
        for c, src in cells.items():
            h = sha(src)
            p = os.path.join(self.history_dir, "src", f"{h}.py")
            if not os.path.isfile(p):
                atomic_write(p, src)
            hashes[c] = h
        parent = int(self.data.get("rev") or 0)
        self.data["rev"] = rev
        snap = {"picture": self.data, "cells": hashes}
        atomic_write(os.path.join(self.history_dir, f"r{rev:04d}.json"), json.dumps(snap, ensure_ascii=False))
        idx["revs"].append({"rev": rev, "parent": parent, "cmd": cmd[:120], "score": None,
                            "t": time.strftime("%H:%M:%S")})
        idx["head"] = rev
        self._save_index(idx)
        return rev

    def set_score(self, rev, ok, total, errors=0, ok_ids=None):
        idx = self._index()
        for r in idx["revs"]:
            if r["rev"] == rev:
                r["score"] = [int(ok), int(total)]
                r["errors"] = int(errors)
                if ok_ids is not None:                   # which items held, so a later loss names the right one
                    r["ok_ids"] = list(ok_ids)
        self._save_index(idx)
        snap_p = os.path.join(self.history_dir, f"r{rev:04d}.json")
        snap = read_json(snap_p)
        if snap:                                         # keep the snapshot's layer state (good renders) current
            snap["picture"] = self.data
            atomic_write(snap_p, json.dumps(snap, ensure_ascii=False))

    def rev_info(self, rev):
        for r in self.revs():
            if r["rev"] == rev:
                return r
        return None

    def load_rev(self, rev):
        """(picture dict, {cell: source}) of a revision."""
        snap = read_json(os.path.join(self.history_dir, f"r{int(rev):04d}.json"))
        if not snap:
            have = [str(r["rev"]) for r in self.revs()]
            raise StudioError(f"no revision {rev} (revisions: {', '.join(have[-15:]) or 'none'})")
        cells = {}
        for c, h in snap["cells"].items():
            try:
                with open(os.path.join(self.history_dir, "src", f"{h}.py"), encoding="utf-8") as f:
                    cells[c] = f.read()
            except OSError:
                pass
        return snap["picture"], cells

    def restore(self, rev):
        """Make revision ``rev`` current: picture.json and every cell file as they were."""
        pic, cells = self.load_rev(rev)
        os.makedirs(self.layers_dir, exist_ok=True)
        for stem, p in self.cell_files().items():
            if stem not in cells:
                os.remove(p)
        on_disk = self.read_cells()
        for c, src in cells.items():
            if c.startswith(SCENE_KEY):                  # the scene file behind `add NAME --scene`
                p = os.path.join(self.folder, c[len(SCENE_KEY):])
                try:
                    with open(p, encoding="utf-8-sig") as f:
                        same = f.read() == src
                except OSError:
                    same = False
                if not same:
                    atomic_write(p, src)
                continue
            if on_disk.get(c) != src:
                atomic_write(self.cell_path(c), src)
        pic = dict(pic)
        pic["rev"] = int(rev)
        self.data = pic
        self.load_defaults()
        return {c: v for c, v in cells.items() if not c.startswith(SCENE_KEY)}

    def load_defaults(self):
        sc = dict(SCENE_DEFAULTS)
        sc.update(self.data.get("scene") or {})
        self.data["scene"] = sc

    def children(self, rev):
        return [r["rev"] for r in self.revs() if r.get("parent") == rev]

    def parent(self, rev):
        r = self.rev_info(rev)
        return r.get("parent") if r else None


# ============================================================================ finding the picture

def art_dir(opt=None):
    return os.path.abspath(opt or os.environ.get("PIXELCANVAS_STUDIO_DIR") or os.path.join(".", "art"))


def set_current(adir, name):
    atomic_write(os.path.join(adir, CURRENT), name + "\n")


def current_name(adir):
    try:
        with open(os.path.join(adir, CURRENT), encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


def list_pictures(adir):
    if not os.path.isdir(adir):
        return []
    return sorted(d for d in os.listdir(adir) if os.path.isfile(os.path.join(adir, d, PICTURE)))


def open_picture(adir, name=None):
    """The picture called ``name`` (or the current one) in ``adir``; a folder path works too."""
    if name and os.path.isfile(os.path.join(name, PICTURE)):
        pic = Picture(name).load()
        set_current(os.path.dirname(os.path.abspath(name)), pic.name)
        return pic
    if not name:
        name = current_name(adir)
        if not name:
            have = list_pictures(adir)
            if len(have) == 1:
                name = have[0]
            else:
                raise StudioError(f"no current picture in {adir}" + (f" (pictures: {', '.join(have)})" if have else ""),
                                  fix='new NAME --prompt "the request"' if not have else f"-p {have[0]} render")
    folder = os.path.join(adir, name)
    if not os.path.isfile(os.path.join(folder, PICTURE)):
        from .._args import unknown_name
        have = list_pictures(adir)
        if have:
            raise StudioError(str(unknown_name("picture", name, have, hint="Pictures: " + ", ".join(have))))
        raise StudioError(f"no picture {name!r} in {adir}", fix=f'new {name} --prompt "the request"')
    pic = Picture(folder).load()
    set_current(adir, pic.name)
    return pic


__all__ = ["Picture", "StudioError", "open_picture", "art_dir", "request_needs", "LAYER_DEFAULTS", "atomic_write"]
