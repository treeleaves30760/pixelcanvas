# -*- coding: utf-8 -*-
"""studio checks: run render.py's report on the composite (one manifest entry per layer) and turn its outcome
into the picture's checklist (which requested items are confirmed) and advice worded as studio commands."""
from __future__ import annotations

import math
import re
import types

from .. import manifest as M
from .compositor import base_name

# ASCII for the report (a cp950 / cp1252 console prints it as it is)
_ASCII = {"\u2713": "ok", "\u2717": "x", "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u2192": "->",
          "\u21e2": "->", "\u00d7": "x", "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u00a0": " ",
          "\u2264": "<=", "\u2265": ">=", "\u00b0": " deg"}

# a colour name of the request -> a paint that reads as that name (manifest.colour_name gives it back)
COLOUR_HEX = {"red": "d63c2f", "orange": "f08a24", "yellow": "f5d142", "green": "4caf50", "teal": "1fa39a",
              "blue": "3a6fd8", "purple": "8e44ad", "pink": "f28cb1", "brown": "8b5a2b", "black": "1e1e1e",
              "white": "f4f4f4", "grey": "8a8a8a", "gray": "8a8a8a", "gold": "e0b040", "silver": "c0c4c8",
              "cream": "f3e6c4", "navy": "1f2f6b", "violet": "8e44ad", "beige": "e3d3b0", "golden": "e0b040",
              "tan": "c8a070", "cyan": "35c4d8", "magenta": "d63c9c", "lime": "9bd43a", "olive": "7a7a2a",
              "maroon": "7a1f24", "crimson": "c0203a", "turquoise": "2ab3a8", "lavender": "b49ce0"}

ITEM_RULES = {"prompt_text_missing", "prompt_unrecorded", "prompt_small", "prompt_cut", "prompt_count",
              "prompt_colour", "prompt_relation", "prompt_covered", "prompt_text_place", "prompt_text_small",
              "prompt_hidden"}
_REL_VERB = {"on": "on", "in": "in", "left of": "left-of", "right of": "right-of", "above": "above", "below": "below",
             "left": "left-of", "right": "right-of", "next to": "next-to", "next": "next-to", "under": "below",
             "beside": "next-to", "over": "above", "inside": "in", "behind": "below", "in front of": "above"}


def ascii(s):
    s = str(s)
    for a, b in _ASCII.items():
        s = s.replace(a, b)
    return s


def forms(word):
    """Lower-case forms a thing is named by: the word, its singular(s) and a plural."""
    w = str(word or "").lower().strip()
    if not w:
        return set()
    out = {w}
    last = w.split()[-1]
    try:
        out.update(M._singular(last))
    except Exception:                                    # noqa: BLE001
        pass
    out.add(last)
    out.add(last + "s")
    if last.endswith("s"):
        out.add(last[:-1])
    return {f for f in out if f}


def _said(w, fs):
    names = [str(n).split(" #")[0].lower() for n in (w.get("names") or [])]
    if any(forms(n) & fs for n in names):
        return True
    item = str(w.get("item") or "").lower()
    if item and forms(item) & fs:
        return True
    val = w.get("value")
    if isinstance(val, str) and forms(val) & fs:
        return True
    txt = " ".join(str(w.get(k) or "") for k in ("short", "message")).lower()
    return any(re.search(r"(?<![\w-])" + re.escape(f) + r"(?![\w-])", txt) for f in fs)


def colour_hex(word):
    w = str(word or "").lower().split()[-1:] or [""]
    return COLOUR_HEX.get(w[0]) or COLOUR_HEX.get(str(word).lower())


class Outcome:
    """Checked picture: ``needs`` (each with ok / reason / cmd), ``extra`` (report items not on the list),
    ``rows`` (manifest summary), ``data`` (the full render.py report dict), ``score`` (ok, total)."""

    def __init__(self):
        self.needs, self.extra, self.rows, self.data, self.text = [], [], [], {}, ""
        self.layer_rows = {}

    @property
    def score(self):
        return sum(1 for n in self.needs if n["ok"]), len(self.needs)


def run(pic, comp, cv, owner, image=None):
    """Check the composite ``cv`` against the picture's request; -> Outcome."""
    from .engine import _render_module
    from .. import vision
    R = _render_module()
    d = pic.data
    prompt = d.get("prompt") or None
    mod = types.SimpleNamespace(PROMPT=prompt, KIND=d.get("kind") or "painting")
    warns = vision.critique(image if image is not None else cv.to_pil())
    data, text = R.build_report(cv, mod, warns)
    out = Outcome()
    out.comp = comp
    out.data, out.text = data, text
    rows = data.get("objects") or []
    out.rows = rows
    by_order = {r.get("order"): r for r in rows}
    for name, orders in owner.items():
        rs = [by_order[o] for o in orders if o in by_order]
        if rs:
            out.layer_rows[name] = rs
    _tick(pic, comp, cv, data, out)
    return out


# ============================================================================ the checklist

def _layer_index(pic, comp):
    """{form: [layer names]}: the words each visible layer answers to (label, name, the cell's own names)."""
    idx = {}
    for L in pic.data["layers"]:
        name = L["name"]
        words = {L.get("label") or base_name(name), name}
        al = L.get("alias")
        if al:
            words |= {al} if isinstance(al, str) else set(al)
        cr = comp.results.get(name) if comp is not None else None
        for e in (cr.entries if cr is not None else []):
            words.add(e["name"].split(" #")[0])
            words |= set(e.get("aliases") or [])
        for w in words:
            for f in forms(w):
                idx.setdefault(f, [])
                if name not in idx[f]:
                    idx[f].append(name)
    return idx


def layers_for(pic, comp, key):
    idx = _layer_index(pic, comp)
    out = []
    for f in forms(key):
        for n in idx.get(f, []):
            if n not in out:
                out.append(n)
    return out


def _text_layer(pic, comp, q):
    """The layer whose cell draws the quoted text ``q``."""
    for L in pic.data["layers"]:
        cr = comp.results.get(L["name"]) if comp is not None else None
        if cr is None:
            continue
        for t in cr.texts:
            if " ".join(str(t.get("text", "")).split()).casefold() == " ".join(q.split()).casefold():
                return L["name"]
    for L in pic.data["layers"]:
        if L["name"] in ("title", "caption", "text", "heading"):
            return L["name"]
    return None


def _free_name(pic, want):
    names = set(pic.names())
    if want not in names:
        return want
    i = 2
    while f"{want}{i}" in names:
        i += 1
    return f"{want}{i}"


def _box_of(out, layer):
    rs = out.layer_rows.get(layer) or []
    if not rs:
        return None
    b = [r["bbox"] for r in rs]
    return [min(x[0] for x in b), min(x[1] for x in b), max(x[2] for x in b), max(x[3] for x in b)]


def fam(pic, layer):
    """'goldfish*' when the layer's cell has copies, else the layer."""
    try:
        L = pic.layer(layer)
    except Exception:                                    # noqa: BLE001
        return layer
    cell = L.get("cell") or layer
    sib = [x["name"] for x in pic.data["layers"] if (x.get("cell") or x["name"]) == cell]
    return f"{cell}*" if len(sib) > 1 and all(x.startswith(cell) for x in sib) else layer


def advise(w, pic, comp, out, need=None):
    """(reason, studio command) for one report issue."""
    rule = w.get("rule", "")
    short = ascii(w.get("short") or w.get("message") or rule).strip()
    key = need["key"] if need and need["kind"] == "thing" else None
    names = [str(n).split(" #")[0] for n in (w.get("names") or [])]
    layer = None
    for n in names + ([key] if key else []):
        ls = layers_for(pic, comp, n)
        if ls:
            layer = ls[0]
            break
    reason = re.split(r" - (?=[a-z])", short, maxsplit=1)[0]
    if rule == "prompt_text_missing":
        q = str(w.get("value") or (need or {}).get("key") or "")
        on = (need or {}).get("on")
        if on:                                           # written on a thing: that thing's cell writes it
            ls = layers_for(pic, comp, on)
            nm = ls[0] if ls else re.sub(r"[^a-z0-9_]", "_", M._one(on.split()[-1]).lower())
            return f"no layer writes it on the {on}", f"write layers/{nm}.py (draws the {on} with '{q}' on it), then render"
        tl = _text_layer(pic, comp, q)
        nm = tl or _free_name(pic, "title")
        return "no layer draws this text", f'add {nm} --text "{q}"'
    if rule == "prompt_unrecorded" or (need and need.get("kind") == "thing" and not layer
                                       and rule != "prompt_relation" and not rule.startswith("text_")):
        thing = key or str(w.get("item") or "").split()[-1:] or ["thing"]
        thing = thing if isinstance(thing, str) else thing[0]
        nm = re.sub(r"[^a-z0-9_]", "_", M._one(thing.split()[-1]).lower()) or "thing"
        return "no layer draws it", f"write layers/{nm}.py (draws the {thing}), then render"
    if rule == "prompt_small" and layer:
        v = w.get("value")
        f = 1.6
        if isinstance(v, (int, float)) and v > 0:
            f = min(3.0, max(1.3, math.sqrt(0.03 / float(v))))
        return reason, f"scale {fam(pic, layer)} {f:.1f}"
    if rule == "prompt_text_small":
        tl = layer or _text_layer(pic, comp, str((need or {}).get("key") or ""))
        return reason, f"scale {tl or 'title'} 1.4"
    if rule == "prompt_cut" and layer:
        b = _box_of(out, layer) or [0.2, 0.2, 0.8, 0.8]
        x0, y0, x1, y1 = b
        w_, h_ = min(x1 - x0, 0.9), min(y1 - y0, 0.9)
        cx = min(max((x0 + x1) / 2, 0.05 + w_ / 2), 0.95 - w_ / 2)
        cy = min(max((y0 + y1) / 2, 0.05 + h_ / 2), 0.95 - h_ / 2)
        return reason, (f"fit {layer} {cx - w_ / 2:.2f} {cy - h_ / 2:.2f} {cx + w_ / 2:.2f} {cy + h_ / 2:.2f}")
    if rule in ("prompt_covered", "prompt_hidden") and layer:
        hb = None
        for r in out.layer_rows.get(layer, []):
            if r.get("hidden_by"):
                hb = r["hidden_by"].split(" #")[0]
        hl = layers_for(pic, comp, hb)[0] if hb and layers_for(pic, comp, hb) else None
        if hl and hl != layer:
            return reason, f"order {layer} above {hl}"
        return reason, f"order {layer} top"
    if rule == "prompt_colour" and layer:
        want = None
        if need and need.get("colours"):
            want = need["colours"][0]
        if not want:
            m = re.search(r"request says (\w+)", short) or re.search(r"requested (\w+)", short)
            want = m.group(1) if m else None
        hx = colour_hex(want)
        return reason, f"recolor {fam(pic, layer)} {hx}" if hx else f"recolor {fam(pic, layer)} COLOUR"
    if rule == "prompt_count":
        n = (need or {}).get("count")
        have = w.get("value")
        if layer and isinstance(have, int):
            n = n if isinstance(n, int) else 1
            if have < n:
                return reason, f"copy {layer} --total {n}"
            extra = [x for x in layers_for(pic, comp, key or layer) if x != layer]
            if extra:
                return reason, f"rm {extra[-1]}"
            return reason, f"write layers/{layer}.py (draws the {key or layer}: {n} in all), then render"
        return reason, (f"copy {layer} --total {n}" if layer and isinstance(n, int) else
                        f"look {layer}" if layer else "")
    if rule == "prompt_relation":
        if need and need.get("kind") == "place":
            a, rel_, b = (list(need.get("key") or []) + [None, None, None])[:3]
            la, lb = layers_for(pic, comp, a), (layers_for(pic, comp, b) if b else [])
            if la and lb and rel_ in _REL_VERB:
                return reason, f"put {fam(pic, la[0])} {_REL_VERB[rel_]} {lb[0]}"
        m = re.search(r"puts the (.+?) (left of|right of|above|below|next to|beside|under|over|on|in|"
                      r"at the [\w ]+?) the (.+?)[,.]", ascii(w.get("message", "")))
        if m:
            a, rel, b = m.group(1), m.group(2), m.group(3)
            la, lb = layers_for(pic, comp, a), layers_for(pic, comp, b)
            if la and lb and rel in _REL_VERB:
                return reason, f"put {fam(pic, la[0])} {_REL_VERB[rel]} {lb[0]}"
        if layer:
            return reason, f"move {layer} --to X Y"
        return reason, ""
    if rule == "prompt_text_place":
        tl = layer or _text_layer(pic, comp, str((need or {}).get("key") or "")) or "title"
        where = str(w.get("threshold") or "")
        x = 0.25 if "left" in where else 0.75 if "right" in where else 0.5
        y = 0.12 if "top" in where else 0.88 if "bottom" in where else 0.5
        return reason, f"move {tl} --to {x:.2f} {y:.2f}"
    if rule.startswith("text_"):
        tl = _text_layer(pic, comp, str(w.get("value") or "")) or layer
        if rule == "text_contrast" and tl:
            return reason, f"set {tl} outline=1d1830 outline_width=0.006"
        if rule == "text_clipped" and tl:
            return "runs off the frame", f"fit {tl} 0.1 0.03 0.9 0.2"
        if rule == "text_small" and tl:
            return reason, f"scale {tl} 1.5"
    return reason, ""


def _tick(pic, comp, cv, data, out):
    """Each checklist item -> ok / reason / cmd. A section's item is measured on the picture (drawn, count, apart,
    place, visible, inside the frame, colours, its quoted text); its states are for the eye (the review)."""
    issues = [w for w in (data.get("prompt_issues") or []) if w.get("rule") in ITEM_RULES
              and w.get("severity") in ("warn", "fail")]
    texts = [w for w in (data.get("text_issues") or []) if w.get("severity") in ("warn", "fail")]
    quoted = {q["text"]: q for q in (data.get("quoted_text") or [])}
    used, tused = set(), set()
    needs = pic.data["needs"]
    open_notes = {}
    for nt in pic.data.get("notes") or []:
        if not nt.get("done") and nt.get("item"):
            open_notes.setdefault(nt["item"], nt)
    for n in needs:
        r = dict(n)
        r.update(ok=False, reason="", cmd="", by=None, misses=[])
        kind = n.get("kind")
        if n.get("ok"):
            r.update(ok=True, by="eye")
            used.update(_settled(n, issues))             # what the eye settled is not reported again
        elif kind == "thing":
            ls = family(pic, n["cell"]) if n.get("cell") else \
                list(dict.fromkeys(sum((layers_for(pic, comp, k) for k in [n["key"]] + list(n.get("alias") or [])),
                                       [])))
            misses = judge_thing(pic, comp, out, n, ls, texts, tused)
            r["misses"] = misses
            if misses:
                r["reason"], r["cmd"] = misses[0]
            else:
                r["ok"], r["by"] = True, (ls[0] if ls else None)
        elif kind == "text":
            q = n["key"]
            st = quoted.get(q)
            mine = [i for i, w in enumerate(issues) if i not in used and w.get("rule", "").startswith("prompt_text")
                    and (str(w.get("value") or "") == q or q.lower() in (w.get("message") or "").lower())]
            tmine = [w for w in texts if q.lower() in str(w.get("value") or w.get("message") or "").lower()
                     and w.get("rule") in ("text_clipped", "text_contrast", "text_small")]
            if mine:
                used.update(mine)
                r["reason"], r["cmd"] = advise(issues[mine[0]], pic, comp, out, r)
            elif st is not None and not st.get("drawn"):
                r["reason"], r["cmd"] = advise({"rule": "prompt_text_missing", "value": q}, pic, comp, out, r)
                if st.get("closest"):
                    r["reason"] = f"drawn as {ascii(st['closest'])[:30]!r}, not {q!r}"
            elif tmine and tmine[0]["severity"] == "fail":
                r["reason"], r["cmd"] = advise(tmine[0], pic, comp, out, r)
                tused.add(id(tmine[0]))
            elif _hidden(comp, [_text_layer(pic, comp, q)]):
                r["reason"], r["cmd"] = _hidden(comp, [_text_layer(pic, comp, q)])
            else:
                r["ok"] = True
                r["by"] = _text_layer(pic, comp, q)
        elif kind == "place":
            a, rel, b = (list(n.get("key") or []) + [None, None, None])[:3]
            judged = place_by_boxes(pic, comp, a, rel, b)
            if judged is not None:                       # the studio's placed boxes decide
                ok, reason, cmd = judged
                r["ok"] = ok
                r["by"] = "boxes" if ok else None
                r["reason"], r["cmd"] = ("", "") if ok else (reason, cmd)
            else:
                la, lb = layers_for(pic, comp, a), layers_for(pic, comp, b) if b else []
                if str(b or "").lower() in _THIRDS:      # a place in the frame needs only A
                    lb, b = ["frame"], None
                if la and (lb or not b):
                    r["reason"] = "the check cannot measure this place: look, then tick it"
                    r["cmd"] = f"need ok {n['id']}"
                else:
                    miss = a if not la else b
                    r["reason"] = f"{miss} has no layer yet"
                    nm = re.sub(r'[^a-z0-9_]', '_', M._one(str(miss).split()[-1]).lower())
                    r["cmd"] = f"write layers/{nm}.py (draws the {miss}), then render"
        else:                                            # an item the agent added: by eye, or bound to a layer
            ln = n.get("layer")
            if ln:
                st = layer_state(pic, out, ln)
                r["ok"] = st is None
                r["reason"] = st or ""
                r["cmd"] = "" if st is None else (_edit(ln, "draws something visible") if "draws nothing" in st
                                                  or "fails" in st else f"look {ln}")
            else:
                r["reason"] = "ticked by eye"
                r["cmd"] = f"need ok {n['id']}"
        nt = open_notes.get(n.get("id"))
        if nt is not None:                               # a review note on this line: open until it is done
            r.update(ok=False, by=None, reason=f"note {nt['id']}: {nt['text'][:80]}", cmd=f"note:{nt['id']}")
        out.needs.append(r)
    for i, w in enumerate(issues):                       # the request's prose is no contract: only its text
        if i in used or not w.get("rule", "").startswith("prompt_text"):
            continue
        reason, cmd = advise(w, pic, comp, out)
        out.extra.append({"reason": ascii(w.get("short") or w.get("message") or w.get("rule")), "cmd": cmd,
                          "rule": w.get("rule")})
    for w in texts:
        if id(w) in tused or (w.get("severity") != "fail" and w.get("rule") not in ("text_contrast",)):
            continue
        reason, cmd = advise(w, pic, comp, out)
        out.extra.append({"reason": ascii(w.get("message") or w.get("rule")), "cmd": cmd, "rule": w.get("rule")})


# ============================================================================ a section's item, measured

_PLACE_REL = {"left of": "left", "right of": "right", "above": "above", "below": "below", "on": "on", "in": "in",
              "next to": "next", "behind": "behind", "in front of": "front"}


def family(pic, cell):
    """The layers drawn by one cell (the cell's layer and its copies), in stack order."""
    return [L["name"] for L in pic.data["layers"] if (L.get("cell") or L["name"]) == cell]


def target_layers(pic, comp, target):
    """The layers a place's other thing names: a section (its name or its item's name), else a layer's words."""
    from .state import layer_key, item_name
    t = " ".join(str(target or "").lower().split())
    t = re.sub(r"^(?:the|a|an)\s+", "", t)
    cands = [t, re.sub(r"'s?\b.*$", "", t).strip(), t.split()[-1] if t else ""]
    for c in [x for x in dict.fromkeys(cands) if x]:
        for n in pic.data.get("needs") or []:
            if n.get("src") == "draft" and n.get("cell") and (n["key"] == c or item_name(n["cell"]).lower() == c):
                got = family(pic, n["cell"])
                if got:
                    return got
        k = layer_key(c).lower()
        for L in pic.data["layers"]:
            if L["name"].lower() == k or (L.get("cell") or "").lower() == k:
                return family(pic, L.get("cell") or L["name"])
    for c in [x for x in dict.fromkeys(cands) if x]:
        got = layers_for(pic, comp, c)
        if got:
            return got
    return []


def _drawn(pic, comp, names):
    return [x for x in names if _visible(pic, x) and comp is not None and comp.placed.get(x) is not None
            and comp.placed[x].box is not None]


def _frac_box(comp, name):
    x0, y0, x1, y1 = comp.placed[name].box
    return (x0 / comp.W, y0 / comp.H, x1 / comp.W, y1 / comp.H)


def own_count(comp, name):
    """How many of its thing one layer draws: its kit records of one name (three apples in one cell), else its
    separate solid parts of a similar size (twelve shards drawn in one cell)."""
    cr = comp.results.get(name)
    if cr is not None and cr.entries:
        names = [e["name"].split(" #")[0].lower() for e in cr.entries]
        if len(set(names)) == 1:
            return max(len(names), sum(int((e.get("meta") or {}).get("count", 1) or 1) for e in cr.entries))
    pl = comp.placed.get(name)
    if pl is None or pl.A.size == 0:
        return 1
    return blob_count(pl.A > 0.5)


def _repeats(comp, name):
    """True when the layer's cell draws in a loop (several things in one cell, which may run together)."""
    import ast
    cr = comp.results.get(name)
    src = (comp.cells.get(getattr(cr, "cell", None) or name) or comp.cells.get(name) or "")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    loops = (ast.For, ast.While, ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp)
    return any(isinstance(node, loops) for node in ast.walk(tree))


def _label(m):
    """Connected parts (4-neighbours) of a boolean mask -> (labels, count)."""
    try:
        from scipy import ndimage
        lab, n = ndimage.label(m)
        return lab, int(n)
    except ImportError:                                  # the same labels with numpy alone
        import numpy as np
        h, w = m.shape
        lab = np.where(m, np.arange(1, h * w + 1).reshape(h, w), 0)
        big = h * w + 1
        while True:
            cur = np.where(m, lab, big)
            nb = cur.copy()
            nb[1:] = np.minimum(nb[1:], cur[:-1])
            nb[:-1] = np.minimum(nb[:-1], cur[1:])
            nb[:, 1:] = np.minimum(nb[:, 1:], cur[:, :-1])
            nb[:, :-1] = np.minimum(nb[:, :-1], cur[:, 1:])
            new = np.where(m, nb, 0)
            if np.array_equal(new, lab):
                break
            lab = new
        u = np.unique(lab[lab > 0])
        remap = np.zeros(big + 1, np.int64)
        remap[u] = np.arange(1, len(u) + 1)
        return remap[lab], int(len(u))


def blob_count(mask):
    """Separate solid parts of a mask at least a quarter the size of the largest (a handle that touches its
    mug is one part; a detail far smaller than the thing is no copy)."""
    import numpy as np
    h, w = mask.shape
    f = max(1, int(math.ceil(max(h, w) / 240.0)))
    if f > 1:
        hh, ww = h // f * f, w // f * f
        mask = mask[:hh, :ww].reshape(hh // f, f, ww // f, f).any((1, 3))
    if not mask.any():
        return 0
    lab, n = _label(mask)
    if n <= 1:
        return n
    sizes = np.bincount(lab.ravel())[1:]
    return int((sizes >= 0.25 * sizes.max()).sum())


def touching(comp, names, gap=0.004):
    """Pairs of these layers whose solid parts touch or overlap (a viewer counts them as one)."""
    import numpy as np
    out = []
    boxes = {n: _frac_box(comp, n) for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A, B = boxes[a], boxes[b]
            if A[0] - gap > B[2] or B[0] - gap > A[2] or A[1] - gap > B[3] or B[1] - gap > A[3]:
                continue
            pa, pb = comp.placed[a], comp.placed[b]
            x0, y0 = max(pa.x0, pb.x0) - 2, max(pa.y0, pb.y0) - 2
            x1 = min(pa.x0 + pa.A.shape[1], pb.x0 + pb.A.shape[1]) + 2
            y1 = min(pa.y0 + pa.A.shape[0], pb.y0 + pb.A.shape[0]) + 2
            if x1 <= x0 or y1 <= y0:
                continue

            def cut(p):
                m = np.zeros((y1 - y0, x1 - x0), bool)
                sy0, sx0 = max(0, y0 - p.y0), max(0, x0 - p.x0)
                sy1, sx1 = min(p.A.shape[0], y1 - p.y0), min(p.A.shape[1], x1 - p.x0)
                if sy1 > sy0 and sx1 > sx0:
                    m[sy0 + p.y0 - y0:sy1 + p.y0 - y0, sx0 + p.x0 - x0:sx1 + p.x0 - x0] = p.A[sy0:sy1, sx0:sx1] > 0.5
                return m
            ma, mb = cut(pa), cut(pb)
            grow = ma.copy()                             # one pixel apart counts as touching
            grow[1:] |= ma[:-1]
            grow[:-1] |= ma[1:]
            grow[:, 1:] |= grow[:, :-1].copy()
            grow[:, :-1] |= grow[:, 1:].copy()
            if (grow & mb).any():
                out.append((a, b))
    return out


def row_for(comp, names, n):
    """A row region where n copies of this size stand apart (a gap of at least 15 % of one's width)."""
    bs = [_frac_box(comp, x) for x in names]
    w = max(b[2] - b[0] for b in bs)
    y0, y1 = min(b[1] for b in bs), max(b[3] for b in bs)
    total = n * max(1.25 * w, w + 0.01)                  # copy --row gives each a cell of 1.2 times its width
    cx = sum((b[0] + b[2]) / 2 for b in bs) / len(bs)
    total = min(total, 0.96)
    cx = min(max(cx, 0.02 + total / 2), 0.98 - total / 2)
    return [round(cx - total / 2, 2), round(y0, 2), round(cx + total / 2, 2), round(y1, 2)]


def _edit(cell, what):
    """A code change, said by NEXT as the draft section to edit (or the cell file when no draft holds it)."""
    return f"edit:{cell}:{what}"


def _painted(comp, name):
    """The colours of a layer's own solid paint (its recolor applied), named like the report names them."""
    import numpy as np
    from .engine import body
    pl = comp.placed.get(name)
    if pl is None or pl.A.size == 0:
        return None
    try:
        P, A, _, _ = body(pl, comp.params(comp.by_name[name]), comp.H)
    except Exception:                                    # noqa: BLE001
        return None
    solid = A > 0.5
    for _ in range(2):                                   # leave the outline band out
        core = solid.copy()
        core[1:, :] &= solid[:-1, :]
        core[:-1, :] &= solid[1:, :]
        core[:, 1:] &= solid[:, :-1]
        core[:, :-1] &= solid[:, 1:]
        if int(core.sum()) < 12:
            break
        solid = core
    if int(solid.sum()) < 3:
        return None
    rgb = np.clip(P[solid] / np.maximum(A[solid][:, None], 1e-6), 0, 1)
    L, C, h = M._oklch(rgb)
    sel = np.ones(len(L), bool)
    if len(L) >= 30:                                     # the darkest and lightest tenth: shade and highlight
        lo, hi = np.percentile(L, [10, 90])
        sel = (L >= lo) & (L <= hi)
        if int(sel.sum()) < 12:
            sel = np.ones(len(L), bool)
    return M._Paint(L[sel], C[sel], h[sel], rgb[sel])


def colour_ok(p, words):
    """(ok, what it is painted) for the colour words of a line ('yellow', 'red and white')."""
    cs = [M._colour_of(w.replace(" ", "-")) for w in words]
    cs = [c for c in cs if c]
    if not cs:
        return True, ""
    shares = [p.share(c) for c in cs]
    if len(cs) > 1:
        ok = all(s >= 0.08 for s in shares)
    else:
        ok = shares[0] >= 0.20 or p.top in M._accepted(cs[0])
    return ok, p.mostly()


def judge_thing(pic, comp, out, n, ls, texts=(), tused=None):
    """[(reason, command)] of what the picture does not show of one item, in fixing order: drawn, count, apart,
    place, visible, inside the frame, colours, its text. [] = every measured bit holds."""
    cell = n.get("cell")
    who_cell = cell or (ls[0] if ls else re.sub(r"[^a-z0-9_]", "_", str(n["key"]).split()[-1].lower()))
    if not ls:
        return [("no layer draws it", _edit(who_cell, f"draws the {n['key']}"))]
    vis = [x for x in ls if _visible(pic, x)]
    if not vis:
        return [(f"{ls[0]} is hidden", f"show {fam(pic, ls[0])}")]
    drawn = _drawn(pic, comp, vis)
    if not drawn:
        err = (comp.info.get(vis[0]) or {}).get("error") if comp is not None else None
        if err:
            return [(f"{vis[0]} fails ({err.get('type') or 'error'})", _edit(cell or vis[0], "fails: see NEXT"))]
        src = (comp.cells.get(cell or vis[0]) or "") if comp is not None else ""
        if not [ln for ln in src.splitlines() if ln.strip() and not ln.strip().startswith("#")]:
            return [(f"{vis[0]} has no code yet", _edit(cell or vis[0], f"draws the {n['key']}"))]
        return [(f"{vis[0]} draws nothing the check can see",               # look would fail on it: edit it
                 _edit(cell or vis[0], f"draws the {n['key']} large and opaque enough to see"))]
    misses = []
    who = fam(pic, drawn[0]) if len(drawn) > 1 else drawn[0]
    want = n.get("count")
    have = len(drawn) if len(drawn) > 1 else own_count(comp, drawn[0])
    cr0 = comp.results.get(drawn[0])
    by_parts = len(drawn) == 1 and not (cr0 is not None and cr0.entries)
    if by_parts and want and not 2 <= want <= 6:
        want = None                                      # one thing, or many shards: counted by eye
    if want and have != want and not (by_parts and have > want):
        if have < want:
            if len(drawn) == 1 and have > 1:
                misses.append((f"{have} drawn apart, the line says {want}",
                               _edit(cell or drawn[0], f"draws {want}, apart")))
            elif len(drawn) == 1 and _repeats(comp, drawn[0]):     # its copies run together: copying the
                misses.append((f"{have} seen (its copies run together), the line says {want}",   # layer doubles them
                               _edit(cell or drawn[0], f"draws {want} with a gap between them")))
            else:
                r = row_for(comp, drawn, want)
                misses.append((f"{have} drawn, the line says {want}",
                               f"copy {drawn[0]} --total {want} --row {r[0]:.2f} {r[1]:.2f} {r[2]:.2f} {r[3]:.2f}"))
        elif len(drawn) > want:
            misses.append((f"{have} drawn, the line says {want}", f"rm {drawn[-1]}"))
        else:
            misses.append((f"{have} drawn, the line says {want}", _edit(cell or drawn[0], f"draws {want}")))
    elif len(drawn) > 1:
        pairs = touching(comp, drawn)
        if pairs:
            names = list(dict.fromkeys(x for p in pairs for x in p))
            r = row_for(comp, drawn, len(drawn))
            misses.append((f"{', '.join(names)} touch: a viewer counts fewer",
                           f"copy {drawn[0]} --total {len(drawn)} --row {r[0]:.2f} {r[1]:.2f} {r[2]:.2f} {r[3]:.2f}"))
    for x in drawn:                                      # a 3D scene in one section: its own objects, measured
        cr_ = comp.results.get(x)
        gone = [e["name"] for e in (getattr(cr_, "entries", None) or []) if e.get("unseen")]
        if gone:
            misses.append((f"{', '.join(gone)} (3D object{'s' if len(gone) > 1 else ''} of {x}) "
                           f"{'are' if len(gone) > 1 else 'is'} outside the frame",
                           _edit(cell or x, f"frames {gone[0]} too: Camera.frame(sc, ...) after the last sc.add")))
        for nm, b in solid_objects(comp, x):
            if b[0] <= 0.002 or b[1] <= 0.002 or b[2] >= 0.998 or b[3] >= 0.998:
                misses.append((f"{nm} (a 3D object of {x}) is cut by the frame (x {b[0]:.2f}-{b[2]:.2f} y "
                               f"{b[1]:.2f}-{b[3]:.2f})", _edit(cell or x, f"frames the whole {nm}: Camera.frame(sc, "
                                                                       "...) after the last sc.add, or move it in")))
                break
    behind = set()
    for rel, target in n.get("places") or []:
        if rel == "at":
            res = _frame_place(pic, comp, None, target, drawn) if target in _THIRDS else None
        else:
            r_ = _PLACE_REL.get(rel)
            lb = target_layers(pic, comp, target) if r_ else []
            lb = [x for x in lb if x not in drawn]
            if r_ == "behind":
                behind.update(lb)
            res = place_layers(pic, comp, drawn, r_, _drawn(pic, comp, lb), rel) if r_ and lb else None
        if res is not None and not res[0]:
            misses.append((res[1], res[2]))
    for x in drawn:                                      # seen in the picture, not only drawn
        v, cover = comp.visibility().get(x, (1.0, None))
        pl = comp.placed.get(x)
        if v >= 0.5 or not cover or (pl is not None and float(pl.A.sum()) >= 0.25 * comp.W * comp.H):
            continue
        if cover in behind:                              # it should stand behind: move it out, never above
            b = _frac_box(comp, x)
            c = _frac_box(comp, cover)
            dx = (b[2] - c[0] + 0.02) * -1 if (b[0] + b[2]) / 2 < (c[0] + c[2]) / 2 else (c[2] - b[0] + 0.02)
            misses.append((f"{x} is {v * 100:.0f}% visible: covered by {cover}", f"move {x} --by {dx:.2f} 0"))
        else:
            misses.append((f"{x} is {v * 100:.0f}% visible: covered by {cover}", f"order {x} above {cover}"))
        break
    for x in drawn:
        pl = comp.placed[x]
        if comp.role(comp.by_name[x]) == "backdrop" or float(pl.A.sum()) >= 0.25 * comp.W * comp.H:
            continue                                     # a backdrop or a surface runs past the frame by design
        solid = float((pl.A > 0.5).sum())
        inside = float((pl.full_alpha(comp.W, comp.H) > 0.5).sum())
        if solid > 0 and inside < 0.9 * solid:
            b = _frac_box(comp, x)
            w_, h_ = min(b[2] - b[0], 0.9), min(b[3] - b[1], 0.9)
            cx = min(max((b[0] + b[2]) / 2, 0.05 + w_ / 2), 0.95 - w_ / 2)
            cy = min(max((b[1] + b[3]) / 2, 0.05 + h_ / 2), 0.95 - h_ / 2)
            misses.append((f"{x} is cut by the frame ({(1 - inside / solid) * 100:.0f}% outside)",
                           f"fit {x} {cx - w_ / 2:.2f} {cy - h_ / 2:.2f} {cx + w_ / 2:.2f} {cy + h_ / 2:.2f}"))
            break
    if n.get("colours"):
        for x in drawn:
            p = _painted(comp, x)
            if p is None:
                continue
            ok, mostly = colour_ok(p, n["colours"])
            if not ok:
                want_c = " and ".join(n["colours"])
                hx = colour_hex(n["colours"][0])
                misses.append((f"{x} is painted mostly {mostly}, the line says {want_c}",
                               f"recolor {who} {hx}" if hx else f"recolor {who} COLOUR"))
                break
    for q in n.get("texts") or []:
        writer = None
        for x in drawn:
            cr = comp.results.get(x)
            if cr is not None and any(" ".join(str(t.get("text", "")).split()).casefold() ==
                                      " ".join(q.split()).casefold() for t in cr.texts):
                writer = x
                break
        if writer is None:
            misses.append((f"'{q}' is not written in it", _edit(cell or drawn[0], f"writes '{q}' on the {n['key']}: "
                                                            f"cv.text({q!r}, box=(X0, Y0, X1, Y1), color=...)")))
            continue
        bad = [w for w in texts if q.lower() in str(w.get("value") or w.get("message") or "").lower()
               and w.get("rule") in ("text_clipped", "text_contrast", "text_small") and w.get("severity") == "fail"]
        if bad:
            if tused is not None:
                tused.add(id(bad[0]))
            reason, cmd = advise(bad[0], pic, comp, out)
            misses.append((reason, cmd))
    return misses


def room_to_grow(comp, names, margin=0.02, others=()):
    """The largest scale factor that keeps every one of these layers inside the frame and clear of the boxes in
    others it does not overlap yet, each growing about its own pivot (the landmark it hangs from, else its bottom
    centre)."""
    best = 9.0
    for x in names:
        pl = comp.placed.get(x)
        if pl is None or pl.box is None:
            continue
        b = _frac_box(comp, x)
        pv = getattr(pl, "pivot", None)
        px, py = (pv[0] / comp.W, pv[1] / comp.H) if pv is not None else ((b[0] + b[2]) / 2, b[3])
        for lo, hi, p in ((b[0], b[2], px), (b[1], b[3], py)):
            if lo < p - 1e-6:
                best = min(best, max(0.0, p - margin) / (p - lo))
            if hi > p + 1e-6:
                best = min(best, max(0.0, 1 - margin - p) / (hi - p))
        cr = comp.results.get(x)                         # hung from a landmark: it never reaches below its owner
        owner = next((d[1][0] for d in (getattr(cr, "deps", None) or []) if d and d[0] == "anchor"
                      and isinstance(d[1], (list, tuple)) and d[1]), None)
        if owner in comp.placed and owner != x and comp.placed[owner].box is not None and b[3] > py + 1e-6:
            ob = _frac_box(comp, owner)[3]
            if b[3] < ob:
                best = min(best, (ob - py) / (b[3] - py))
    if (others or len(names) > 1) and best > 1:                            # nor into a neighbour it stands clear of now
        lo, hi = 1.0, best
        if _bumps(comp, names, others, hi):
            for _ in range(20):
                mid = (lo + hi) / 2
                lo, hi = (lo, mid) if _bumps(comp, names, others, mid) else (mid, hi)
            best = lo
    return best


def _item_boxes(pic, comp, skip):
    """Frame-fraction boxes of every other requested thing and text (not backdrops, not surfaces)."""
    out = []
    for n in pic.data.get("needs") or []:
        if not n.get("cell"):
            continue
        for y in _drawn(pic, comp, family(pic, n["cell"])):
            pl = comp.placed.get(y)
            if (y in skip or pl is None or pl.box is None or comp.role(comp.by_name[y]) in ("backdrop", "pass")
                    or float(pl.A.sum()) >= 0.25 * comp.W * comp.H):
                continue
            out.append(_frac_box(comp, y))
    return out


def _spread_row(comp, names, f, others):
    """A row where these copies, grown by f, would stand apart, inside the frame and clear of others; None when
    there is none (copy --row keeps their size, so the next report can offer the scale)."""
    n = len(names)
    bs = [_frac_box(comp, x) for x in names]
    w = max(b[2] - b[0] for b in bs)
    y0, y1 = min(b[1] for b in bs), max(b[3] for b in bs)
    hb = y1 - y0
    for g in (f, (1 + f) / 2, 1.3):
        total = n * max(1.4 * w * g, w * g + 0.01)       # room for each copy to grow by g with a gap
        if total > 0.94:
            continue
        cx = sum((b[0] + b[2]) / 2 for b in bs) / n
        cx = min(max(cx, 0.03 + total / 2), 0.97 - total / 2)
        r = (cx - total / 2, y0, cx + total / 2, y1)
        grown = (r[0], y1 - hb * g, r[2], y1)            # the row's band once each copy grows from its foot
        if not any(grown[0] < o[2] and o[0] < grown[2] and grown[1] < o[3] and o[1] < grown[3] for o in others):
            return [round(v, 2) for v in r]
    return None


def _grown(comp, x, f):
    pl = comp.placed[x]
    b = _frac_box(comp, x)
    pv = getattr(pl, "pivot", None)
    px, py = (pv[0] / comp.W, pv[1] / comp.H) if pv is not None else ((b[0] + b[2]) / 2, b[3])
    return (px + (b[0] - px) * f, py + (b[1] - py) * f, px + (b[2] - px) * f, py + (b[3] - py) * f)


def _bumps(comp, names, others, f, gap=0.01):
    """True when growing these layers by f makes one of them reach a box in others it stands clear of now (a
    box within the gap already, such as a tag on its bag, belongs with it)."""
    def meet(a, b, g):
        return a[0] - g < b[2] and b[0] - g < a[2] and a[1] - g < b[3] and b[1] - g < a[3]
    live = [x for x in names if comp.placed.get(x) is not None and comp.placed[x].box is not None]
    now = {x: _frac_box(comp, x) for x in live}
    big = {x: _grown(comp, x, f) for x in live}
    for i, x in enumerate(live):
        if any(not meet(now[x], o, gap) and meet(big[x], o, gap) for o in others):
            return True
        for y in live[i + 1:]:                           # copies grow together: nor into each other
            if not meet(now[x], now[y], gap) and meet(big[x], big[y], gap):
                return True
    return False


BLEND_RATIO = 1.25                                  # luminance contrast ratio under which a thing may blend
BLEND_DE = 0.03                                      # ... when its colour is also this close (OKLab distance)


def _lab(rgb):
    import numpy as np
    L, C, h = M._oklch(np.asarray(rgb, float)[None])
    r = np.radians(h[0])
    return np.array([L[0], C[0] * np.cos(r), C[0] * np.sin(r)])


def _lum(rgb):
    import numpy as np
    c = np.clip(np.asarray(rgb, float), 0, 1)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])


def blends(pic, comp, name):
    """A doubt when an object layer melts into what lies right around it in the picture (its solid paint against
    the composite in a thin ring around its silhouette): luminance contrast under 1.25:1 and a close colour."""
    import numpy as np
    from .engine import body
    from .. import filters as _f
    L = pic.layer(name)
    if comp.role(L) in ("backdrop", "pass"):
        return None
    cr = comp.results.get(name)
    if cr is not None and cr.texts and comp._text_only(cr):
        return None
    pl = comp.placed.get(name)
    if pl is None or pl.box is None or float(pl.A.sum()) >= 0.25 * comp.W * comp.H:
        return None
    try:
        P, A, x0, y0 = body(pl, comp.params(L), comp.H)
    except Exception:                                    # noqa: BLE001
        return None
    solid = A > 0.5
    if int(solid.sum()) < 30:
        return None
    fg = np.median(P[solid] / np.maximum(A[solid][:, None], 1e-6), axis=0)
    W, H = comp.W, comp.H
    bx0, by0, bx1, by1 = pl.box
    r = max(2.0, 0.04 * max(bx1 - bx0, by1 - by0))
    pad = int(math.ceil(3 * r)) + 2
    cx0, cy0 = max(0, int(bx0) - pad), max(0, int(by0) - pad)
    cx1, cy1 = min(W, int(bx1) + pad), min(H, int(by1) + pad)
    if cx1 - cx0 < 4 or cy1 - cy0 < 4:
        return None
    fa = pl.full_alpha(W, H)[cy0:cy1, cx0:cx1]
    near = _f.gaussian_blur(fa.astype(np.float32), r) > 0.03
    ring = near & (fa < 0.05)
    if int(ring.sum()) < 20:
        return None
    buf = np.clip(np.asarray(comp.canvas.buf, np.float32)[cy0:cy1, cx0:cx1], 0, 1)
    bg = np.median(buf[ring], axis=0)
    lf, lb = _lum(fg), _lum(bg)
    ratio = (max(lf, lb) + 0.05) / (min(lf, lb) + 0.05)
    de = float(np.linalg.norm(_lab(fg) - _lab(bg)))
    if ratio >= BLEND_RATIO or de >= BLEND_DE:
        return None
    behind = None
    names = pic.names()
    best = 0.0
    for other in names[:names.index(name)][::-1]:        # the layer under it that fills most of the ring
        po = comp.placed.get(other)
        if po is None or po.box is None or not pic.layer(other).get("visible", True) or \
                comp.role(pic.layer(other)) == "pass":
            continue
        share = float((po.full_alpha(W, H)[cy0:cy1, cx0:cx1][ring] > 0.5).mean())
        if share > best + 0.1:
            behind, best = other, share
    tone = fg * 0.55 if lb > 0.35 else fg + (1 - fg) * 0.6
    hx = "".join(f"{int(round(min(max(v, 0), 1) * 255)):02x}" for v in tone)
    where = f"the {behind} behind it" if behind else "what is behind it"
    return (f"{name} blends into {where} (contrast {ratio:.1f}:1): recolor {fam(pic, name)} {hx}")


def doubts(pic, comp, out):
    """'worth checking' lines (suggestions, never blocking): [(item id or None, key, text)]. A requested thing
    under 1.5 % of the frame; an outline on a photo request."""
    res = []
    W, H = comp.W, comp.H
    for n in pic.data.get("needs") or []:
        if n.get("kind") != "thing" or not n.get("cell"):
            continue
        drawn = _drawn(pic, comp, family(pic, n["cell"]))
        for x in drawn:
            cr = comp.results.get(x)
            if cr is not None and cr.texts and comp._text_only(cr):
                continue                                 # a title: its size is the text's to judge
            pl = comp.placed[x]
            a = float(pl.A.sum()) / (W * H)
            b = _frac_box(comp, x)                       # a thin thing (a spoon) reads by its extent
            seen = max(a, 0.5 * (b[2] - b[0]) * (b[3] - b[1]))
            if seen < 0.015 and comp.role(comp.by_name[x]) not in ("backdrop", "pass"):
                f_want = min(3.0, max(1.3, math.sqrt(0.03 / max(seen, 1e-4))))
                f = math.floor(10 * min(f_want, room_to_grow(comp, drawn, others=_item_boxes(pic, comp, drawn)))
                               + 1e-6) / 10                # printed with one decimal: never rounded past the room
                if f >= 1.2:
                    res.append((n["id"], f"small:{n['cell']}", f"{x} covers {a * 100:.1f}% of the frame: scale "
                                f"{fam(pic, x)} {f:.1f}"))
                else:
                    r = _spread_row(comp, drawn, f_want, _item_boxes(pic, comp, drawn)) if len(drawn) > 1 else None
                    if r:                                # copies packed too close: lay them out wider first
                        res.append((n["id"], f"small:{n['cell']}", f"{x} covers {a * 100:.1f}% of the frame and its "
                                    f"copies stand too close to grow: copy {drawn[0]} --total {len(drawn)} --row "
                                    f"{r[0]:.2f} {r[1]:.2f} {r[2]:.2f} {r[3]:.2f}"))
                    elif float(pic.layer(x).get("scale", 1.0) or 1.0) < 1.05:   # already grown as far as it can
                        res.append((n["id"], f"small:{n['cell']}", f"{x} covers {a * 100:.1f}% of the frame and has "
                                    "no room to grow where it is: draw it larger in its section"))
                break
    for n in pic.data.get("needs") or []:
        if n.get("kind") != "thing" or not n.get("cell"):
            continue
        for x in _drawn(pic, comp, family(pic, n["cell"])):
            b = blends(pic, comp, x)
            if b:
                res.append((n["id"], f"blend:{x}", b))
                break
    prompt = str(pic.data.get("prompt") or "")
    if re.search(r"\b(photo\w*|realistic|photorealistic|photo-realistic|3d render\w*|3-d render\w*)\b", prompt, re.I):
        for L in pic.data["layers"]:
            if not L.get("visible", True):
                continue
            src = comp.cells.get(L.get("cell") or L["name"]) or ""
            item = next((n["id"] for n in pic.data.get("needs") or [] if n.get("cell") == (L.get("cell") or
                                                                                         L["name"])), None)
            if re.search(r"\bline\s*=\s*(?:True|\"ink\"|'ink')", src):
                res.append((item, f"photo:{L['name']}", f"the request is a photo: {L['name']} is drawn with line=True "
                            "(an outline reads as a drawing): remove it"))
            elif L.get("outline"):
                res.append((item, f"photo:{L['name']}", f"the request is a photo: {L['name']} has an outline (it reads "
                            f"as a drawing): set {L['name']} outline=off"))
    return res


_REL_NORM = {"left of": "left", "left": "left", "to the left of": "left", "right of": "right", "right": "right",
             "to the right of": "right", "above": "above", "over": "above", "below": "below", "under": "below",
             "beneath": "below", "underneath": "below", "on": "on", "on top of": "on", "atop": "on", "in": "in",
             "inside": "in", "within": "in", "next to": "next", "next": "next", "beside": "next", "by": "next",
             "behind": "behind", "in front of": "front"}
_PUT_VERB = {"left": "left-of", "right": "right-of", "above": "above", "below": "below", "on": "on", "in": "in",
             "next": "next-to"}


def solid(comp, name):
    """(box, mass centre) in frame fractions of the SOLID part of a layer: its painted bulk without faint glow,
    smoke, sparks, embers or a thin shadow. Solid pixels (alpha >= 0.6) in parts at least 1.5 % of the solid
    area, trimmed to the 2nd-98th percentile of that mass on each axis. Falls back to the full box."""
    import numpy as np
    cache = comp.__dict__.setdefault("_solid", {})
    pl = comp.placed.get(name)
    if pl is None or pl.box is None:
        return None
    key = (name, pl.x0, pl.y0, pl.A.shape)
    if key in cache:
        return cache[key]
    full = _frac_box(comp, name)
    res = (full, ((full[0] + full[2]) / 2, (full[1] + full[3]) / 2))
    A = np.asarray(pl.A, np.float32)
    if A.size:
        h, w = A.shape
        f = max(1, int(math.ceil(max(h, w) / 320.0)))
        hh, ww = h // f * f, w // f * f
        a = A[:hh, :ww].reshape(hh // f, f, ww // f, f).mean((1, 3)) if f > 1 else A
        m = a >= 0.6
        if int(m.sum()) < 12:
            m = a >= 0.3
        if int(m.sum()) >= 12:
            lab, n = _label(m)
            if n > 1:
                sizes = np.bincount(lab.ravel())[1:]
                keep = np.zeros(n + 1, bool)
                keep[1:] = sizes >= 0.015 * sizes.sum()
                m = keep[lab]
            wgt = np.where(m, a, 0.0)
            tot = float(wgt.sum())
            if tot > 0:
                def span(prof):
                    c = np.cumsum(prof) / tot
                    lo = int(np.searchsorted(c, 0.02))
                    hi = int(np.searchsorted(c, 0.98)) + 1
                    return lo, max(hi, lo + 1)
                y0, y1 = span(wgt.sum(1))
                x0, x1 = span(wgt.sum(0))
                ys, xs = np.mgrid[0:a.shape[0], 0:a.shape[1]]
                my, mx = float((wgt * ys).sum() / tot), float((wgt * xs).sum() / tot)
                W, H = comp.W, comp.H
                box = ((pl.x0 + x0 * f) / W, (pl.y0 + y0 * f) / H, (pl.x0 + x1 * f) / W, (pl.y0 + y1 * f) / H)
                box = (max(box[0], full[0]), max(box[1], full[1]), min(box[2], full[2]), min(box[3], full[3]))
                if box[2] > box[0] and box[3] > box[1]:
                    res = (box, ((pl.x0 + (mx + 0.5) * f) / W, (pl.y0 + (my + 0.5) * f) / H))
    cache[key] = res
    return res


def solid_objects(comp, name):
    """[(name, frame box)] of the named objects a solid.Scene rendered in this layer (from its own records),
    carried through the layer's move / scale / crop."""
    cr = comp.results.get(name)
    pl = comp.placed.get(name)
    if cr is None or pl is None or not getattr(cr, "entries", None):
        return []
    out = []
    W, H = cr.size
    for e in cr.entries:
        b = e.get("bbox")
        if not e.get("solid") or not b or len(b) != 4:
            continue
        pts = [pl.point(b[0] * W, b[1] * H), pl.point(b[2] * W, b[3] * H)]
        xs, ys = [p[0] / comp.W for p in pts], [p[1] / comp.H for p in pts]
        out.append((e["name"], (min(xs), min(ys), max(xs), max(ys))))
    return out


def solid_box(comp, name):
    s = solid(comp, name)
    return s[0] if s else None


def _boxes_of(pic, comp, word, names=None):
    """Solid boxes (fractions, see ``solid``) of the visible layers a word names (or of ``names``), with their
    stack index: places are judged on the bulk of a thing, not its glow or sparks."""
    out = []
    order = [L["name"] for L in pic.data["layers"]]
    for n in (names if names is not None else layers_for(pic, comp, word)):
        pl = comp.placed.get(n)
        if not _visible(pic, n) or pl is None or pl.box is None:
            continue
        out.append((n, solid_box(comp, n), order.index(n)))
    return out


def _holds(rel, a, b, ia, ib, surface, big=(False, False), bm=None):
    """True / False for one pair of (solid) boxes, None when boxes cannot decide. big = (B spans over a third of
    the frame's width, of its height) or B is a backdrop: then above / below / left / right mean beyond B's mass
    (its mass centre bm and its near third), not clear of its whole box (a moon above a forest of tall firs)."""
    aw, ah, bw, bh = a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]
    ox = min(a[2], b[2]) - max(a[0], b[0])
    oy = min(a[3], b[3]) - max(a[1], b[1])
    acx, acy, bcx, bcy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    mx, my = bm if bm else (bcx, bcy)
    if rel in ("above", "below") and big[1]:
        return acy < min(my, b[1] + bh / 3) if rel == "above" else acy > max(my, b[3] - bh / 3)
    if rel in ("left", "right") and big[0]:
        return acx < min(mx, b[0] + bw / 3) if rel == "left" else acx > max(mx, b[2] - bw / 3)
    if rel in ("left", "right"):
        side = acx < bcx if rel == "left" else acx > bcx
        return side and ox < min(aw, bw) / 3
    if rel in ("above", "below"):
        side = acy < bcy if rel == "above" else acy > bcy
        return side and oy < min(ah, bh) / 3
    if rel == "next":
        return ox < min(aw, bw) / 3 and -ox < max(aw, bw) and oy > 0
    if rel == "on":
        if surface:
            return b[1] - 0.02 <= a[3] <= b[3] + 0.005 and ox > 0
        return abs(a[3] - b[1]) <= 0.03 + 0.1 * bh and ox > 0
    if rel == "in":                                     # its centre column inside, its base inside (a heap may
        t = 0.03                                         # rise above the rim)
        return b[0] - t <= acx <= b[2] + t and b[1] - t <= a[3] <= b[3] + t
    if rel in ("behind", "front"):
        if ox > 0 and oy > 0:                            # overlapping: the one drawn later is in front
            return ia < ib if rel == "behind" else ia > ib
        if ox <= -max(aw, bw) / 2:
            return None                                  # far apart sideways: left to the eye
        return a[3] < b[3] if rel == "behind" else a[3] > b[3]     # apart: nearer the viewer stands lower
    return None


_THIRDS = {"left": ((0, 1 / 3), None), "right": ((2 / 3, 1), None), "centre": ((1 / 3, 2 / 3), None),
           "center": ((1 / 3, 2 / 3), None), "middle": ((1 / 3, 2 / 3), None), "top": (None, (0, 1 / 3)),
           "bottom": (None, (2 / 3, 1)), "top left": ((0, 1 / 3), (0, 1 / 3)), "top right": ((2 / 3, 1), (0, 1 / 3)),
           "bottom left": ((0, 1 / 3), (2 / 3, 1)), "bottom right": ((2 / 3, 1), (2 / 3, 1))}


def _frame_place(pic, comp, a, where, names=None):
    """A at left / right / centre / top / bottom ... of the frame: its box centre in that third."""
    rx, ry = _THIRDS[where]
    A = _boxes_of(pic, comp, a, names)
    if not A:
        return None
    for na, bx, _ in A:
        cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
        if (rx and not rx[0] <= cx <= rx[1]) or (ry and not ry[0] <= cy <= ry[1]):
            tx = (rx[0] + rx[1]) / 2 if rx else cx
            ty = (ry[0] + ry[1]) / 2 if ry else cy
            who = fam(pic, na) if len(A) > 1 else na
            return False, (f"{na} is not at the {where} (its centre x {cx:.2f} y {cy:.2f}; the {where} third: "
                           + (f"x {rx[0]:.2f}-{rx[1]:.2f}" if rx else "") + (" " if rx and ry else "")
                           + (f"y {ry[0]:.2f}-{ry[1]:.2f}" if ry else "") + ")"), f"move {who} --to {tx:.2f} {ty:.2f}"
    return True, "", ""


def place_by_boxes(pic, comp, a, rel, b):
    """(ok, reason, command) for a place item judged on the layers' placed boxes; None when they cannot decide
    (no layer yet, a relation boxes do not measure)."""
    if comp is not None and a and str(b or "").lower() in _THIRDS:
        return _frame_place(pic, comp, a, str(b).lower())
    r = _REL_NORM.get(str(rel or "").lower().strip())
    if comp is None or not r or not a or not b:
        return None
    return place_layers(pic, comp, layers_for(pic, comp, a), r, layers_for(pic, comp, b), rel)


def place_layers(pic, comp, la, r, lb, rel=None):
    """(ok, reason, command) for layers ``la`` standing ``r`` (left right above below on in next behind front) of
    layers ``lb``, judged on their placed boxes; None when the boxes cannot decide."""
    rel = rel or r
    A, B = _boxes_of(pic, comp, None, la), _boxes_of(pic, comp, None, lb)
    if not A or not B:
        return None
    res = []
    for na, ba, ia in A:                                 # every instance of A, against one of B
        best = None
        for nb, bb, ib in B:
            L = pic.layer(nb)
            back = comp.role(L) == "backdrop"
            surface = back or (bb[2] - bb[0]) >= 0.45
            big = (back or bb[2] - bb[0] > 1 / 3, back or bb[3] - bb[1] > 1 / 3)
            h = _holds(r, ba, bb, ia, ib, surface, big, solid(comp, nb)[1])
            if h is None:
                continue
            best = h if best is None else (best or h)
        if best is None:
            return None
        res.append((na, best))
    bad = [na for na, h in res if not h]
    if not bad:
        return True, "", ""
    nb = B[0][0]
    x = next(bx for n, bx, _ in A if n == bad[0])
    y = B[0][1]
    reason = (f"{bad[0]} is not {rel} {nb} (solid parts: {bad[0]} x {x[0]:.2f}-{x[2]:.2f} y {x[1]:.2f}-{x[3]:.2f}, "
              f"{nb} x {y[0]:.2f}-{y[2]:.2f} y {y[1]:.2f}-{y[3]:.2f})")
    who = fam(pic, bad[0]) if len(A) > 1 else bad[0]
    apart = min(x[2], y[2]) - max(x[0], y[0]) <= 0 or min(x[3], y[3]) - max(x[1], y[1]) <= 0
    if r in ("behind", "front") and apart:               # move it to stand nearer / farther than B's base
        ah = x[3] - x[1]
        ty = y[3] + 0.1 * ah if r == "front" else y[3] - 0.25 * (y[3] - y[1]) - ah / 2
        return False, reason, f"move {bad[0]} --to {(y[0] + y[2]) / 2:.2f} {ty:.2f}"
    if r == "behind":
        return False, reason, f"order {who} below {nb}"
    if r == "front":
        return False, reason, f"order {who} above {nb}"
    return False, reason, f"put {who} {_PUT_VERB[r]} {nb}"


def _settled(n, issues):
    """Indexes of the report issues an item ticked by eye answers."""
    kind = n.get("kind")
    out = []
    for i, w in enumerate(issues):
        rule = w.get("rule", "")
        if kind == "place":
            a, _, b = (list(n.get("key") or []) + [None, None, None])[:3]
            if rule == "prompt_relation" and a and _said(w, forms(a)) and (not b or _said(w, forms(b))):
                out.append(i)
        elif kind == "thing":
            fs = forms(n["key"])
            for al in n.get("alias") or []:
                fs |= forms(al)
            if rule != "prompt_relation" and not rule.startswith("prompt_text") and _said(w, fs):
                out.append(i)
        elif kind == "text":
            if rule.startswith("prompt_text") and str(w.get("value") or "") == n["key"]:
                out.append(i)
    return out


def _hidden(comp, names):
    """(reason, command) when the best of these layers is mostly covered by a layer above it, else None."""
    if comp is None:
        return None
    vis = comp.visibility()
    got = [(vis[n][0], n, vis[n][1]) for n in names if n and n in vis]
    if not got:
        return None
    v, n, cover = max(got, key=lambda t: t[0])
    if v >= 0.5 or not cover:
        return None
    pl = comp.placed.get(n)
    if pl is not None and float(pl.A.sum()) >= 0.25 * comp.W * comp.H:
        return None                                      # a surface: what stands on it is meant to cover it
    return f"{n} is {v * 100:.0f}% visible: covered by {cover}", f"order {n} above {cover}"


def _visible(pic, name):
    for L in pic.data["layers"]:
        if L["name"] == name:
            return L.get("visible", True)
    return False


def layer_state(pic, out, name):
    """None when the layer is there, visible, whole and big enough; else what is wrong with it."""
    try:
        L = pic.layer(name)
    except Exception:                                    # noqa: BLE001
        return f"no layer {name}"
    if not L.get("visible", True):
        return f"{name} is hidden"
    rs = out.layer_rows.get(name)
    if not rs:
        return f"{name} draws nothing"
    r = rs[0]
    comp_vis = None
    if getattr(out, "comp", None) is not None:
        comp_vis = out.comp.visibility().get(name)
    if comp_vis and comp_vis[0] < 0.5:
        return f"{name} is mostly covered ({comp_vis[0] * 100:.0f}% visible, by {comp_vis[1]})"
    if r.get("clipped_pct", 0) > 10:
        return f"{name} is cut by the frame ({r['clipped_pct']:.0f} % outside)"
    if r.get("visible_pct", 100) < 50:
        return f"{name} is mostly covered ({r['visible_pct']:.0f} % visible)"
    if r.get("area_pct", 1) < 0.2:
        return f"{name} is tiny ({r['area_pct']:.2f} % of the frame)"
    return None
