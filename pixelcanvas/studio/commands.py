# -*- coding: utf-8 -*-
"""studio commands: every verb of ``pixelcanvas studio`` and the compact report it prints."""
from __future__ import annotations

import difflib
import json
import math
import os
import re
import shlex
import sys
import time

import numpy as np

from .state import (Picture, StudioError, open_picture, art_dir, set_current, check_name, request_needs,
                    atomic_write, first_comment, list_pictures, replace_file, SCENE_KEY, SCENE_RE, sha, layer_key,
                    rebuild_needs, inventory, thing_item, read_json, PICTURE)
from .compositor import (Compositor, LAYER_DEFAULTS, SCENE_DEFAULTS, guess_role, base_name, layer_params,
                         crop_of, text_like)
from .engine import Cache, NAMESPACE
from . import check as C

PREVIEW_MAX = 1280
REPORT_NEEDS = 14                                    # checklist lines shown before "+N more"
REPORT_LAYERS = 16                                   # layer lines shown before "+N more"
SHEET_TILES = 9                                      # layers shown alone on the look sheet
ITEM_TILES = 12                                      # items shown close up on the look sheet
LOOK_LINES = 12                                      # look list lines before "+N more lines"


# ============================================================================ output helpers

def me():
    """How to run the studio again (printed in NEXT lines)."""
    cmd = os.environ.get("PIXELCANVAS_STUDIO_CMD")
    if cmd:
        return cmd
    base = os.environ.get("PIXELCANVAS_CMD")
    if base:
        return f"{base} studio"
    from .. import cli
    root = cli.runtime_root()
    tool = os.path.join(root, "tools", "studio.py") if root else None
    if tool and os.path.isfile(tool):
        p = os.path.abspath(tool).replace("\\", "/")
        return f'python "{p}"' if " " in p else f"python {p}"
    return "pixelcanvas studio"


def ab(p):
    """An absolute path with forward slashes: for every file the agent is told to write or open."""
    return os.path.abspath(p).replace("\\", "/")


def rel(p):
    try:
        r = os.path.relpath(p)
    except ValueError:
        r = str(p)
    return r.replace("\\", "/")


def say(*lines):
    for ln in lines:
        text = C.ascii(ln)
        try:
            print(text)
        except (OSError, ValueError):                    # a Windows console or pipe that refuses a write
            try:
                sys.stdout.flush()
            except (OSError, ValueError):
                pass
            try:
                data = (text + "\n").encode("ascii", "replace")
                buf = getattr(sys.stdout, "buffer", None)
                if buf is not None:
                    buf.write(data)
                    buf.flush()
                else:
                    sys.stdout.write(data.decode("ascii"))
            except (OSError, ValueError):
                pass


def _num(tok, what):
    try:
        return float(str(tok).rstrip("%")) / (100.0 if str(tok).endswith("%") else 1.0)
    except ValueError:
        raise StudioError(f"{what}: {tok!r} is not a number") from None


# ============================================================================ argument parsing

class Args:
    """Positionals and options of one verb; numbers may be written '0.4 0.6', '0.4,0.6' or '(0.4, 0.6)'."""

    def __init__(self, verb, argv, options):
        self.verb, self.pos, self.opt = verb, [], {}
        spec = dict(options)
        i = 0
        while i < len(argv):
            a = argv[i]
            if a.startswith("--") and len(a) > 2 and not re.match(r"^--?\d", a):
                key, eq, val = a.partition("=")
                if key not in spec:
                    near = difflib.get_close_matches(key, list(spec), n=1, cutoff=0.5)
                    raise StudioError(f"{verb}: unknown option {key}" + (f" - did you mean {near[0]}?" if near else "")
                                      + (f" (options: {', '.join(spec)})" if spec else " (it takes no options)"))
                n = spec[key]
                if n == 0:
                    self.opt[key] = True
                    i += 1
                    continue
                vals = [val] if eq else []
                i += 1
                while len(_split(vals)) < n and i < len(argv) and not (argv[i].startswith("--") and
                                                                      not re.match(r"^--?\d", argv[i])):
                    vals.append(argv[i])
                    i += 1
                got = _split(vals) if n > 1 else [" ".join(vals)]
                if n > 1 and len(got) < n:
                    raise StudioError(f"{verb} {key} needs {n} values, got {len(got)}: {' '.join(vals) or '(none)'}")
                self.opt[key] = got[:n] if n > 1 else got[0]
                if n > 1 and len(got) > n:
                    self.pos += got[n:]
                continue
            self.pos.append(a)
            i += 1

    def nums(self, key, what=None):
        v = self.opt.get(key)
        return None if v is None else [_num(x, what or f"{self.verb} {key}") for x in v]


def _split(vals):
    out = []
    for v in vals:
        out += [x for x in re.split(r"[\s,()\[\]]+", str(v)) if x]
    return out


def _numbers(pos, n, usage):
    vals = _split(pos)
    if len(vals) != n:
        raise StudioError(f"expected {n} numbers: {usage}")
    return [_num(v, usage) for v in vals]


# ============================================================================ the session: one command

def write_out(pic, comp):
    """The picture file (new --out / set scene out=): the full-size picture after every render, replaced
    atomically; nothing is written while no layer draws anything."""
    path = pic.data.get("out")
    if not path or not any(v is not None for v in comp.results.values()):
        return
    base, ext = os.path.splitext(path)
    tmp = f"{base}.tmp{ext or '.png'}"
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        shown_pil(comp).save(tmp)
        if not replace_file(tmp, path):
            say(f"note: {path} is locked by another program; the picture is in {tmp}")
    except OSError as e:
        say(f"note: the picture file {path} was not written ({e.strerror or e})")


def shown_pil(comp):
    """The picture the user gets: the medium picture when the medium stage ran, else the composite."""
    img = getattr(comp, "medium_img", None)
    if img is None:
        return comp.canvas.to_pil().convert("RGB")
    from PIL import Image
    return Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))


def medium_lines(sess):
    """The report lines of the medium stage."""
    m = getattr(sess, "medium", None)
    if not m:
        return []
    from ..medium import LABELS
    lab = LABELS.get(m["medium"], m["medium"])
    if m.get("error"):
        return [f"medium: {lab} failed ({m['error']}); the picture shows the drawing"]
    dl = ", ".join(f"{k} {v}" for k, v in sorted((m.get("dials") or {}).items()))
    t = m["tb"] + m["tm"]
    out = [f"medium: {lab}" + (f" ({dl})" if dl else "") + f" - the picture, preview and look sheet show it; the "
           f"checks measure the drawn layers ({t:.1f} s: buffers {m['tb']:.1f}, {m['medium']} {m['tm']:.1f})"]
    if m.get("open"):
        out[0] += f"; painted over the drawing as it stands ({m['open']})"
    if m.get("estimated"):
        out.append("  " + "; ".join(f"{x}: volume estimated" for x in m["estimated"][:8]) +
                   " (nothing recorded: a Sketch.paint volume= or a form solid gives the exact shape)")
    for name, ok, what, fix in m.get("gates") or []:
        out.append(f"  [{'x' if ok else ' '}] medium {name}: {what}" + ("" if ok else f" -> {me()} {fix}"))
    return out


def medium_ready(pic, comp, out, errors):
    """'' when the drawing is clean enough for the medium stage, else what it waits for."""
    if errors:
        return "a section fails"
    if pic.data.get("hazard"):
        return "the last edit hid a requested thing"
    open_ = [n for n in out.needs if not n["ok"]]
    if open_:
        return f"item {open_[0]['id']} is open"
    if [n for n in pic.data.get("notes", []) if not n.get("done")]:
        return "a note is open"
    return ""


def run_medium(sess, comp, out, errors):
    """The medium stage after a clean render: comp.medium_img, and sess.medium for the report and NEXT."""
    from . import medium_stage as MS
    pic = sess.pic
    sess.medium = None
    med, dials = MS.setting(pic.data["scene"])
    if not med:
        return
    why = medium_ready(pic, comp, out, errors)       # what is still open: said, never a reason to leave it out
    if not any(v is not None for v in comp.results.values()):
        sess.medium = {"medium": med, "error": "no layer draws anything yet"}
        return
    try:
        img, b, tb, tm = MS.apply(comp, pic, med, dials)
    except Exception as e:                               # noqa: BLE001 - the drawn picture stands
        sess.medium = {"medium": med, "error": f"{type(e).__name__}: {str(e)[:160]}"}
        return
    comp.medium_img = img
    try:
        g = MS.gates(comp, pic, out, img, med, dials)
    except Exception as e:                               # noqa: BLE001
        g = [("gates", True, f"not measured ({type(e).__name__})", None)]
    sess.medium = {"medium": med, "dials": dials, "tb": tb, "tm": tm, "estimated": list(b.get("estimated") or []),
                   "recorded": list(b.get("recorded") or []), "gates": g, "open": why}


def _picture_changed(pic, comp):
    """True when the composite differs from the one the last report saw (its hash is kept in picture.json)."""
    import hashlib
    try:
        buf = getattr(comp, "medium_img", None)
        buf = np.asarray(comp.canvas.buf if buf is None else buf, np.float32)
        step = max(1, int(max(buf.shape[:2]) // 320))
        h = hashlib.sha1(np.round(np.clip(buf[::step, ::step], 0, 1) * 255).astype(np.uint8).tobytes()).hexdigest()
    except Exception:                                    # noqa: BLE001
        return True
    old = pic.data.get("picture_hash")
    pic.data["picture_hash"] = h
    return old != h


def picture_file_line(pic):
    """Where the picture the user gets is, said in every report whose picture changed."""
    if pic.data.get("out"):
        return f"picture file: {ab(pic.data['out'])} ({_size_text(pic)}, written now)"
    W, H = pic.data["size"]
    return f"picture file: {ab(pic.preview)} (preview of {W}x{H}; {me()} deliver PATH.png writes the full size)"


def out_path(pic):
    return pic.data.get("out") or f"./{pic.name}.png"


class Session:
    def __init__(self, pic):
        self.pic = pic
        self.cache = Cache(pic.cache_dir)
        self.cells = pic.read_cells()
        self.synced = ([], [], [])
        self.comp = None
        self.mute = False                                # batch: one report at the end
        self.medium = None                               # the medium stage of the last render (report, NEXT)
        self.scene_src = {}
        self.notices = []                                # lines for the next report (add --from)
        self.verb = None
        self.report_failed = False

    def sync(self, order=None):
        self.cells = self.pic.read_cells()
        self.synced = self.pic.sync(self.cells, order)
        self._scene_files()
        return self.synced

    def _scene_files(self):
        """Scene files read by scene("...") cells: their source is part of every revision; an edited one is a
        changed layer."""
        d = self.pic.data
        known = d.setdefault("scene_files", {})
        self.scene_src = {}
        changed = self.synced[1]
        for L in d["layers"]:
            src = self.cells.get(L.get("cell") or L["name"]) or ""
            for m in SCENE_RE.finditer(src):
                relp = m.group(2)
                p = relp if os.path.isabs(relp) else os.path.join(self.pic.folder, relp)
                try:
                    with open(p, encoding="utf-8-sig") as f:
                        text = f.read()
                except OSError:
                    continue
                self.scene_src[SCENE_KEY + relp] = text
                h = sha(text)
                if known.get(relp) not in (None, h) and L["name"] not in changed + self.synced[0]:
                    changed.append(L["name"])
                known[relp] = h

    def geometry(self):
        """A compositor with every layer placed (no composite): boxes and anchors for parametric edits."""
        comp = Compositor(self.pic.data, self.cells, cache=self.cache, paths=self._paths(), folder=self.pic.folder)
        for L in self.pic.data["layers"]:
            if comp.role(L) != "pass":
                try:
                    comp.place_layer(L["name"])
                except ValueError:
                    pass
        return comp

    def box(self, comp, name):
        L = self.pic.layer(name)
        if comp.role(L) == "pass":
            raise StudioError(f"{name} is a pass (it changes the whole picture): it has no place to move")
        pl = comp.place_layer(name)
        if pl is None or pl.box is None:
            err = (comp.info.get(name) or {}).get("error")
            raise StudioError(f"{name} draws nothing yet" + (f" ({err['short']})" if err else "")
                              + f": fix layers/{L.get('cell') or name}.py first")
        x0, y0, x1, y1 = pl.box
        return [x0 / comp.W, y0 / comp.H, x1 / comp.W, y1 / comp.H]

    def _paths(self):
        return {c: rel(self.pic.cell_path(c)) for c in self.cells}

    # ---- commit + render + report
    def commit(self, cmd, render=True, force_rev=False, content=True, full=False):
        """Record a revision (``content``: the picture itself changed, so NEXT asks for another look)."""
        added, changed, removed = self.synced
        if added or changed or removed:
            content = True
            bits = []
            for lab, lst in (("new", added), ("changed", changed), ("removed", removed)):
                if lst:
                    bits.append(f"{lab} {', '.join(lst)}")
            cmd = f"{cmd} ({'; '.join(bits)})" if cmd != "render" else f"cells: {'; '.join(bits)}"
            force_rev = True
        if force_rev and content:
            self.pic.data["edit"] = int(self.pic.data.get("edit") or 0) + 1
        if force_rev:
            self.pic.snapshot(cmd, dict(self.cells, **self.scene_src))
        self.pic.save()
        if render:
            return self.render(cmd, full=full)
        return None

    def render(self, cmd, size=None, quiet=False, full=False):
        pic = self.pic
        t0 = time.time()
        comp = Compositor(pic.data, self.cells, cache=self.cache, paths=self._paths(), folder=self.pic.folder)
        comp.run()
        self.comp = comp
        cv = comp.canvas
        im = cv.to_pil()
        if max(im.size) > PREVIEW_MAX:
            from PIL import Image
            f = PREVIEW_MAX / float(max(im.size))
            im = im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))), Image.LANCZOS)
        owner = comp.annotate(cv)

        def role_of(c):
            for L in pic.data["layers"]:
                if (L.get("cell") or L["name"]) == c:
                    return L.get("role") or comp.role(L)
            return "object"
        rebuild_needs(pic.data, self.cells, role_of)
        out = C.run(pic, comp, cv, owner, image=im)
        if size is None:
            _watch(pic, comp, cmd)
            run_medium(self, comp, out, sum(1 for v in comp.info.values() if v.get("error")))
            if getattr(comp, "medium_img", None) is not None:
                im = shown_pil(comp)
                if max(im.size) > PREVIEW_MAX:
                    from PIL import Image
                    f = PREVIEW_MAX / float(max(im.size))
                    im = im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))), Image.LANCZOS)
        tmp = pic.preview + ".tmp.png"
        im.save(tmp)
        if not replace_file(tmp, pic.preview):
            print(f"note: {rel(pic.preview)} is locked by another program; the picture is in {rel(tmp)}")
        for L in pic.data["layers"]:                         # remember auto roles the render found
            if not L.get("role"):
                r = comp.role(L)
                if r in ("backdrop", "pass"):
                    L["auto_role"] = r
        ok, total = out.score
        errors = sum(1 for v in comp.info.values() if v.get("error"))
        rev = int(pic.data.get("rev") or 0)
        if rev:
            pic.set_score(rev, ok, total, errors, [n["id"] for n in out.needs if n["ok"]])
        pic.data["last"] = {"score": [ok, total], "errors": errors, "rev": rev}
        pic.data["open_items"] = [f"{n['id']} {n['text']}" for n in out.needs if not n["ok"]]
        if size is None:
            write_out(pic, comp)
            self.pic_changed = _picture_changed(pic, comp)
        pic.save()
        self.look_written = None
        if not quiet and not self.mute and size is None:
            try:                                         # the sheet to open: picture + every item close up
                self.look_written = write_sheet(self, comp, out=out)[0]
            except Exception:                            # noqa: BLE001 - the report names the preview instead
                self.look_written = None
        if not quiet and not self.mute:
            try:
                print_report(self, comp, out, cmd, time.time() - t0, im.size, full=full)
            except Exception as e:                       # noqa: BLE001 - the picture is rendered and saved
                self.report_failed = True
                say(f"rendered rev {int(pic.data.get('rev') or 0)}: {ab(pic.preview)}"
                    + (f"; picture file: {pic.data['out']}" if pic.data.get("out") else ""))
                say(f"report failed: {type(e).__name__}: {str(e)[:200]}")
                say(f"NEXT: {me()} status")
            pic.save()
        return out


def _requested_layers(pic):
    """The layers the checklist asks for (each section's layers, layers bound to an item)."""
    out = []
    for n in pic.data.get("needs") or []:
        if n.get("cell"):
            out += C.family(pic, n["cell"])
        if n.get("layer"):
            out.append(n["layer"])
    return [x for x in dict.fromkeys(out) if x in pic.names()]


def _watch(pic, comp, cmd):
    """A requested layer that this command covered (visible share under 60 %) or pushed out of the frame:
    remembered for the report and for NEXT (undo first)."""
    d = pic.data
    before = d.get("seen_vis") or {}
    now, hit = {}, None
    vis = comp.visibility()
    for x in _requested_layers(pic):
        pl = comp.placed.get(x)
        L = pic.layer(x)
        if pl is None or pl.box is None or not L.get("visible", True) or comp.role(L) in ("backdrop", "pass"):
            continue
        if float(pl.A.sum()) >= 0.25 * comp.W * comp.H:
            continue                                     # a surface is meant to be covered
        v, cover = vis.get(x, (1.0, None))
        solid = float((pl.A > 0.5).sum())
        inside = float((pl.full_alpha(comp.W, comp.H) > 0.5).sum()) / solid if solid else 1.0
        now[x] = [round(v, 3), round(inside, 3)]
        was = before.get(x)
        if hit is None and was and str(cmd) not in ("render", "look", "need", "x") and not cmd.startswith("undo"):
            if was[0] >= 0.6 and v < 0.6:
                hit = {"cmd": cmd, "layer": x, "what": f"covered {x} (visible {v * 100:.0f}%, was "
                                                      f"{was[0] * 100:.0f}%" + (f", under {cover})" if cover else ")")}
            elif was[1] >= 0.9 and inside < 0.9:
                hit = {"cmd": cmd, "layer": x, "what": f"pushed {x} out of the frame ({(1 - inside) * 100:.0f}% "
                                                      "outside)"}
    d["seen_vis"] = now
    if hit:
        d["hazard"] = hit
    else:
        d.pop("hazard", None)


# ============================================================================ the report

def _layer_line(sess, comp, out, L, w):
    """(line, needs a look) for one layer."""
    name = L["name"]
    info = comp.info.get(name) or {}
    role = comp.role(L)
    err = info.get("error")
    tags = []
    p = layer_params(L)
    if abs(p["scale"] - 1) > 1e-3:
        tags.append(f"x{p['scale']:.2f}".rstrip("0").rstrip("."))
    if p["rotate"]:
        tags.append(f"rot {p['rotate']:g}")
    if p["flip"]:
        tags.append("flipped")
    for k in ("outline", "glow"):
        if p[k]:
            tags.append(k)
    if p["shadow"]:
        tags.append(f"shadow {p['shadow']}")
    if p["tint"]:
        tags.append(f"tint {p['tint']}")
    if p["blur"]:
        tags.append("blur")
    if p["opacity"] < 0.999:
        tags.append(f"opacity {p['opacity'] * 100:.0f}%")
    if info.get("grows"):
        tags.append(f"drawn at 1x: its sizes grow {info['grows']:.2f}x faster than the canvas; a kit size already "
                    "in 1080p px needs no * S")
    tag = ("  [" + ", ".join(tags) + "]") if tags else ""
    head = f"  {name:<{w}} "
    if err:
        where = f"line {err['line']}: " if err.get("line") else ""
        kept = "  (last good render kept)" if info.get("stale") else "  (nothing drawn)"
        return head + f"ERROR {where}{err.get('short', '')}"[:190] + kept, True
    if not L.get("visible", True):
        return head + "hidden" + tag, False
    if role == "pass":
        return head + "pass (whole picture)" + tag, False
    if role == "backdrop":
        objs = C.solid_objects(comp, name)               # a 3D scene in one section: its objects by name
        cr_ = comp.results.get(name)
        gone = [e["name"] for e in (getattr(cr_, "entries", None) or []) if e.get("unseen")]
        if objs or gone:
            tag += "  3D objects: " + ", ".join([f"{n} x {b[0]:.2f}-{b[2]:.2f} y {b[1]:.2f}-{b[3]:.2f}"
                                                 for n, b in objs[:6]] + [f"{n} (outside the frame)" for n in gone])
        return head + "backdrop" + tag, False
    pl = comp.placed.get(name)
    if pl is None or pl.box is None:
        from .state import remark_only
        if remark_only(comp.cells.get(L.get("cell") or name) or ""):
            return f"  {name}: no code, skipped", False
        return head + "draws nothing" + tag, True
    x0, y0, x1, y1 = pl.box
    W, H = comp.W, comp.H
    tot = float(pl.A.sum())
    area = tot / (W * H) * 100
    bits, problem = [], False
    if area < 25:                                        # surfaces are meant to be covered by what stands on them
        v, cover = comp.visibility().get(name, (1.0, None))
        vis = f"visible {v * 100:.0f}%"
        if cover and v < 0.70:
            vis += f": covered by {cover}"
            problem = True
        bits.append(vis)
    solid = float((pl.A > 0.5).sum())
    inside = float((pl.full_alpha(W, H) > 0.5).sum())
    tot = solid
    if tot > 0 and inside < 0.98 * tot:
        bits.append(f"{(1 - inside / tot) * 100:.0f}% off the frame")
        problem = problem or inside < 0.9 * tot
    return (head + f"x {x0 / W:.2f}-{x1 / W:.2f}  y {y0 / H:.2f}-{y1 / H:.2f}  {area:5.1f}% of frame"
            + ("  " + ", ".join(bits) if bits else "") + tag), problem


def _norm(line):
    return " ".join(line.split())


def draft_where(pic, cell, line=None):
    """(draft path, section header, draft line) when this cell came from a draft file that still holds its
    section, else None. ``line``: a line of the cell file, mapped to the draft's line."""
    dr = pic.data.get("draft") or {}
    info = (dr.get("cells") or {}).get(cell)
    path = dr.get("path")
    if not info or not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8-sig") as f:
            text = f.read()
        with open(pic.cell_path(cell), encoding="utf-8-sig") as f:
            if sha(f.read()) != info.get("sha"):
                return None                              # the cell file was edited after the draft: it is the code
    except OSError:
        return None
    head = None
    for ln in text.splitlines():
        h = section_head(ln)
        if h and layer_key(h[0]).lower() == cell.lower():
            head = ln.strip()
            break
    if head is None:
        return None
    dl = None
    if line:
        mp = info.get("lines") or []
        if 0 < int(line) <= len(mp):
            dl = mp[int(line) - 1]
    return ab(path), head, dl


def _cell_of(pic, name):
    try:
        return pic.layer(name).get("cell") or name
    except StudioError:
        return name


def _cmd_text(pic, cmd, M):
    """A checklist command as NEXT says it (code changes name the draft section to edit)."""
    m = re.match(r"edit:([^:]+):(.*)$", cmd, re.S)
    if m:
        cell, what = m.group(1), m.group(2)
        w = draft_where(pic, cell)
        if w:
            return f'edit section "{w[1]}" in {w[0]} (it {what}), then: {M} add --from {w[0]}'
        return f"rewrite {ab(pic.cell_path(cell))} (it {what}), then: {M} render"
    m = re.match(r"write layers/(\w+)\.py \(draws the (.+?)\), then render", cmd)
    if m:
        w = draft_where(pic, m.group(1))
        if w:
            return f'edit section "{w[1]}" in {w[0]} (it draws the {m.group(2)}), then: {M} add --from {w[0]}'
        dr = (pic.data.get("draft") or {}).get("path")
        if dr and os.path.isfile(dr):
            return (f'add a section "# %% {m.group(1)}" to {ab(dr)} (it draws the {m.group(2)}), then: {M} add '
                    f"--from {ab(dr)}")
        return f"write {ab(pic.cell_path(m.group(1)))} (one cell that draws the {m.group(2)}), then: {M} render"
    return f"{M} {cmd}"


def _cmd_short(pic, cmd):
    """A checklist command as the item line shows it."""
    m = re.match(r"edit:([^:]+):(.*)$", cmd or "", re.S)
    if m:
        w = draft_where(pic, m.group(1))
        return (f"edit section {m.group(1)} of the draft (it {m.group(2)})" if w else
                f"rewrite layers/{m.group(1)}.py (it {m.group(2)})")
    m = re.match(r"note:(\d+)$", cmd or "")
    if m:
        return f"note {m.group(1)} is open"
    return cmd


def print_report(sess, comp, out, cmd, secs, psize, full=False):
    pic = sess.pic
    d = pic.data
    rev = int(d.get("rev") or 0)
    W, H = d["size"]
    look = getattr(sess, "look_written", None)
    if look:
        say(f"rev {rev}: {cmd[:60]}    look: {ab(look)} (1280 px: picture + each item close up)")
    else:
        say(f"rev {rev}: {cmd[:60]}    picture: {ab(pic.preview)} ({psize[0]}x{psize[1]} of {W}x{H})")
    if getattr(sess, "verb", None) == "status" and d.get("prompt"):
        say(f'the request: "{d["prompt"][:300]}"')
    ran = [f"{n} {v['seconds']:.1f}s" for n, v in comp.info.items() if v.get("ran")]
    cached = [n for n, v in comp.info.items() if v.get("cached")]
    failed = [n for n, v in comp.info.items() if v.get("error")]
    say((f"cells run: {', '.join(ran) or 'none'}; from cache: {len(cached)}"
         + (f"; failed: {', '.join(failed)}" if failed else "") + f"; {secs:.1f}s")[:200])
    for ln in getattr(sess, "notices", None) or []:
        say(ln)
    layers = d["layers"]
    if not layers:
        say("layers: none yet")
    else:
        w = max(len(L["name"]) for L in layers)
        lines = {L["name"]: _layer_line(sess, comp, out, L, w) for L in layers}
        before = d.get("shown")
        if full or not isinstance(before, dict):
            shown = list(layers)
            if not full and len(shown) > REPORT_LAYERS:  # errors and problems always, then bottom to top
                bad = [L for L in shown if lines[L["name"]][1]]
                keep = bad + [L for L in shown if L not in bad][:max(0, REPORT_LAYERS - len(bad))]
                shown = [L for L in shown if L in keep]
            say("layers, back to front:")
        else:
            shown = [L for L in layers if lines[L["name"]][1] or before.get(L["name"]) != _norm(lines[L["name"]][0])]
            say(f"layers: {len(layers)}; changed or to check: {len(shown)} (status --all lists every layer)")
        for L in shown:
            say(lines[L["name"]][0])
        if (full or not isinstance(before, dict)) and len(shown) < len(layers):
            say(f"  ... +{len(layers) - len(shown)} more layers (status --all lists every layer)")
        d["shown"] = {n: _norm(v[0]) for n, v in lines.items()}
    errs = [(n, v["error"]) for n, v in comp.info.items() if v.get("error")]
    for n, e in errs[:3]:
        cell = _cell_of(pic, n)
        w_ = draft_where(pic, cell, e.get("line"))
        if e.get("source"):
            if w_ and w_[2]:
                say(f"  {n}: {w_[0]} line {w_[2]} (section {cell}): {e['source'][:110]}")
            else:
                say(f"  {n}: {ab(pic.cell_path(cell))} line {e.get('line')}: {e['source'][:110]}")
        for c in (e.get("causes") or [])[:6 if e.get("signatures") else 2]:
            say(f"    likely cause: {c[:200]}")
        for sg in (e.get("signatures") or ([e["signature"]] if e.get("signature") else []))[:3]:
            say(f"    the call: {sg[:320]}")
    ok, total = out.score
    prev = _prev_score(pic, rev)
    ptxt = f" (rev {prev[0]}: {prev[1][0]} of {prev[1][1]})" if prev else ""
    nxt = next_step(sess, comp, out, prev, errs)
    due = bool(d.get("review_due"))                      # the review list below prints every line once
    if total:
        say(f"request: {ok} of {total} confirmed{ptxt}")
    shown_n = 0
    for n in out.needs:
        if n["ok"] and (due or shown_n >= REPORT_NEEDS):
            continue
        shown_n += 1
        if n["ok"]:
            say((f"  [x] {n['id']} {n['text']}" + (" (by eye)" if n.get("by") == "eye" else ""))[:240])
            continue
        line = f"  [ ] {n['id']} {n['text']}"
        line += f" -> {n['reason']}" if n.get("reason") else ""
        line += f": {_cmd_short(pic, n['cmd'])}" if n.get("cmd") else ""
        say(line[:300])
    if shown_n < len(out.needs) and not due:
        say(f"  ... +{len(out.needs) - shown_n} more confirmed (need lists them)")
    for x in out.extra[:2]:
        say(f"  report: {x['reason'][:110]}" + (f" ({x['cmd']})" if x.get("cmd") else ""))
    notes = [n for n in d.get("notes", []) if not n.get("done")]
    more_notes = max(0, len(notes) - 12)
    if notes:
        say(f"notes: {len(notes)} open")
        for n in notes[:12]:
            say(f"  ( ) {n['id']} {n['text'][:110]}" + (f"  [{n['layer']}]" if n.get("layer") else ""))
    if more_notes:
        say(f"  ... +{more_notes} more (note lists them all)")
    hz = d.get("hazard")
    if hz:
        say(f"this command ({hz['cmd'][:50]}) {hz['what']}: undo brings it back")
    for ln in medium_lines(sess):
        say(ln)
    dbt = C.doubts(pic, comp, out)
    pfile = picture_file_line(pic) if getattr(sess, "pic_changed", False) else None
    if d.get("review_due"):
        if not getattr(sess, "look_written", None):
            try:
                write_sheet(sess, comp, out=out)
            except Exception as e:                       # noqa: BLE001 - the look still works from the preview
                say(f"  (look sheet not written: {type(e).__name__}: {e})")
        if pfile:
            say(pfile)
        say("look list ([x] = what can be measured holds; by eye: does the picture show each line at a glance?):")
        by_item = {}
        for iid, _, text in dbt:
            if iid is not None:
                by_item.setdefault(iid, []).append(text)
        rows = [((f"  [x] {n['id']} {n['text']}" + "".join(f"  <- worth checking: {t}"
                                                          for t in by_item.get(n["id"], [])))[:300],
                 bool(by_item.get(n["id"]))) for n in out.needs if n["ok"]]
        if len(rows) > LOOK_LINES:                       # lines with a doubt first, the rest capped
            rows = [r for r in rows if r[1]] + [r for r in rows if not r[1]]
            more = len(rows) - LOOK_LINES
            rows = rows[:LOOK_LINES]
        else:
            more = 0
        for r, _ in rows:
            say(r)
        if more:
            say(f"  ... +{more} more lines ({me()} need)")
        hints = [t for t in review_hints(sess, comp, out)] + [t for iid, _, t in dbt if iid is None]
        if hints:
            say("worth checking (suggestions, never blocking):")
            for h in hints:
                say(f"  - {h}")
    else:
        told = d.setdefault("doubted", {})
        fresh = [(k, t) for _, k, t in dbt if told.get(k) != t]
        if fresh:
            say("worth checking (suggestions, never blocking):")
            for k, t in fresh[:3]:
                say(f"  - {t}")
                told[k] = t
        if pfile:
            say(pfile)
    say("NEXT: " + nxt)


def _prev_score(pic, rev):
    r = pic.rev_info(rev)
    seen = set()
    while r and r.get("parent") and r["parent"] not in seen:
        seen.add(r["parent"])
        p = pic.rev_info(r["parent"])
        if p and p.get("score"):
            return p["rev"], p["score"], p.get("ok_ids")
        r = p
    return None


def review_limit(d):
    try:
        return max(0, int(d["scene"].get("reviews", 3)))
    except (TypeError, ValueError):
        return 3


def next_step(sess, comp, out, prev, errs):
    """The one next command: a failing cell, an edit that hid a requested thing, an unconfirmed request item, an
    open note, a look at the sheet, finished."""
    pic = sess.pic
    M = me()
    d = pic.data
    d["review_due"] = None
    if not d["layers"]:
        if d.get("whole"):
            return whole_next(pic)
        draft = os.path.join(pic.folder, "draft.py")
        return f"write {ab(draft)} (one '# %% name' section per requested thing), then: {M} add --from {ab(draft)}"
    med = getattr(sess, "medium", None) or {}
    if med.get("error"):                                 # a requested medium missing is said first
        from ..medium import LABELS
        return (f"the {LABELS.get(med['medium'], med['medium'])} medium is not painted ({med['error'][:80]}); "
                + _next_core(sess, comp, out, prev, errs))
    return _next_core(sess, comp, out, prev, errs)


def _next_core(sess, comp, out, prev, errs):
    pic = sess.pic
    M = me()
    d = pic.data
    if not d["layers"]:
        if d.get("whole"):
            return whole_next(pic)
        draft = os.path.join(pic.folder, "draft.py")
        return f"write {ab(draft)} (one '# %% name' section per requested thing), then: {M} add --from {ab(draft)}"
    if errs:
        n, e = errs[0]
        cell = _cell_of(pic, n)
        short = e.get("short", "").split(" -> studio api")[0]
        api = f"; the call's arguments: {M} api {e['api']}" if e.get("api") else ""
        if e.get("provider"):
            pw = draft_where(pic, e["provider"])
            if pw:
                return (f'edit section "{pw[1]}" in {pw[0]} ({short[:110]}), then: {M} add --from {pw[0]}{api}')
            return f"fix {ab(pic.cell_path(e['provider']))} ({short[:110]}), then: {M} render{api}"
        w = draft_where(pic, cell, e.get("line"))
        if w:
            where = f"line {w[2]}: " if w[2] else ""
            return f'edit section "{w[1]}" in {w[0]} ({where}{short[:110]}), then: {M} add --from {w[0]}{api}'
        where = f" line {e['line']}" if e.get("line") else ""
        return f"fix {ab(pic.cell_path(cell))}{where} ({short[:110]}), then: {M} render{api}"
    hz = d.get("hazard")
    if hz:
        return (f"{M} undo - this command ({hz['cmd'][:40]}) {hz['what']}; or, if that is what you want, go on "
                "with the next command")
    ok, total = out.score
    noted = sum(1 for n in out.needs if not n["ok"] and str(n.get("cmd") or "").startswith("note:"))
    if prev and ok + noted < prev[1][0] and total == prev[1][1]:
        lost = [n for n in out.needs if not n["ok"] and not str(n.get("cmd") or "").startswith("note:")]
        held = prev[2] if len(prev) > 2 and prev[2] is not None else None
        if held is not None:                             # the item that held before, not one open all along
            lost = [n for n in lost if n["id"] in held] or lost
        what = lost[0]["text"] if lost else "a confirmed item"
        return f"this edit lost a confirmed item ({what[:80]}): {M} undo"
    open_ = [n for n in out.needs if not n["ok"]]
    if open_:
        n = open_[0]
        seen = d.get("next_item") or {}
        times = int(seen.get("times", 0)) + 1 if seen.get("id") == n["id"] and seen.get("edit") != d.get("edit") \
            else (int(seen.get("times", 1)) if seen.get("id") == n["id"] else 1)
        d["next_item"] = {"id": n["id"], "times": times, "edit": d.get("edit")}
        m = re.match(r"note:(\d+)$", n.get("cmd") or "")
        if m:
            nt = next((x for x in d.get("notes", []) if x["id"] == int(m.group(1))), None)
            if nt is not None:
                return _note_step(pic, nt, M)
        settle = (f"; when the item is wrong: {M} need rm {n['id']} (or need set {n['id']} \"...\"), when your eyes "
                  f"confirm it: {M} need ok {n['id']}")
        if n.get("cmd"):
            return _cmd_text(pic, n["cmd"], M) + (settle if times >= 2 else "")
        return (f"item {n['id']} ({n['text'][:80]}): {n.get('reason') or 'not confirmed'} - change it ({M} look "
                f"shows where things are), or when your eyes confirm it: {M} need ok {n['id']} --why \"...\"; when "
                f"the item is wrong: {M} need rm {n['id']}")
    d.pop("next_item", None)
    notes = [n for n in d.get("notes", []) if not n.get("done")]
    if notes:
        return _note_step(pic, notes[0], M)
    mg = [g for g in ((getattr(sess, "medium", None) or {}).get("gates") or []) if not g[1] and g[3]]
    if mg:
        tries = d.get("medium_fix") or {}
        key = f"{mg[0][0]}|{d.get('edit')}|{json.dumps(d['scene'].get('medium'))}"
        if tries.get("key") != key:
            tries = {"key": key, "n": int(tries.get("n", 0)) + 1}
            d["medium_fix"] = tries
        if tries["n"] <= 4:
            return f"{M} {mg[0][3]} - medium {mg[0][0]}: {mg[0][2]}"
    else:
        d.pop("medium_fix", None)
    return look_step(pic, M)


def _size_text(pic):
    w, h = (list(pic.data.get("size") or [0, 0]) + [0, 0])[:2]
    return f"{int(w)}x{int(h)}"


def finish_text(pic, M):
    """Where the finished picture is: the --out file, or the deliver command that writes one."""
    if pic.data.get("out"):
        return f"the picture is {pic.data['out']}"
    return f"{M} deliver {out_path(pic)}"


def look_step(pic, M):
    """Nothing measured is open: look at the sheet (up to ``reviews`` times, once per change), then finished."""
    d = pic.data
    limit = review_limit(d)
    looks = d.get("looks") or {"n": 0, "edit": None}
    edit = int(d.get("edit") or 0)
    changed = looks.get("edit") != edit
    if looks["n"] >= limit or limit == 0:              # after the last look the picture is finished for good
        if pic.data.get("out"):
            return (f"finished: the picture is {ab(pic.data['out'])} ({_size_text(pic)}, up to date; open it before "
                    "you tell the user)")
        return f"finished: {finish_text(pic, M)}, then open it before you tell the user"
    if changed or looks["n"] == 0:
        looks = {"n": looks["n"] + 1, "edit": edit}
        d["looks"] = looks
    k = looks["n"]
    d["review_due"] = k
    look = ab(os.path.join(pic.folder, "look.png"))
    label = f"look {k} of {limit}" if k <= limit else "look again"
    again = " once more" if k > 1 else ""
    draft = (d.get("draft") or {}).get("path")
    if draft and os.path.isfile(draft):
        fix = f"edit its section in {ab(draft)}, then {M} add --from {ab(draft)}"
    else:
        fix = f"edit its cell in {ab(pic.layers_dir)}, then {M} render"
    return (f"{label}: open {look}{again} (the picture, then every item close up); a line above that does not show "
            f"at a glance: {fix} (or one command: move, scale, order, crop ...); every line shows: finished - "
            f"{finish_text(pic, M)}")


def _note_step(pic, n, M):
    txt = n["text"][:90]
    lay = n.get("layer")
    if lay and lay in pic.names():
        cell = pic.layer(lay).get("cell") or lay
        w = draft_where(pic, cell)
        code = (f'edit section "{w[1]}" in {w[0]} (then {M} add --from {w[0]})' if w else
                f"rewrite {ab(pic.cell_path(cell))}")
        return (f"fix note {n['id']}: {txt} - {M} look {lay} shows it; {code} or move / scale / put / set {lay}; "
                f"then: {M} note done {n['id']}")
    if re.search(r"\b(empty|space|small|tiny|frame|crop|bare|lost|wall|margin|room)\b", n["text"].lower()):
        r = crop_suggestion(pic, asked=True)
        if r:
            return (f"fix note {n['id']}: {txt} - {M} crop {r[0]:.2f} {r[1]:.2f} {r[2]:.2f} {r[3]:.2f} (the region "
                    f"that holds the drawn things, enlarged to the frame); then: {M} note done {n['id']}")
    return (f"fix note {n['id']}: {txt} - edit the section of the thing it is about, or move / scale / put / set "
            f"it; then: {M} note done {n['id']}")


_HINT_RULES = ("object_tiny", "sparse_composition", "low_focal_contrast", "blends_into_background", "object_hidden",
               "subject_cropped", "narrow_tonal_range", "blown_highlights")


def review_hints(sess, comp, out, limit=3):
    """Up to three things a reviewer should look at: things drawn in a small part of the frame (crop), the strongest
    report warnings, objects that seem to float, requested things drawn very small. A suggestion is made once per
    layer unless it got worse."""
    pic = sess.pic
    d = pic.data
    W, H = comp.W, comp.H
    hints = []                                           # (key, metric (bigger = worse), text)
    ws = [w for w in (out.data.get("warnings") or []) + (out.data.get("object_issues") or [])
          if w.get("severity") in ("warn", "fail") and w.get("rule") in _HINT_RULES]
    ws.sort(key=lambda w: (_HINT_RULES.index(w["rule"]), w.get("severity") != "fail"))
    for w in ws[:2]:
        hints.append((f"rule:{w['rule']}", 0.0, C.ascii(w.get("short") or w.get("message") or w["rule"])
                       .split(" - ")[0][:140]))
    for x in out.extra[:2]:
        if x.get("cmd"):
            hints.append((f"report:{x.get('rule')}", 0.0, f"{x['reason'][:100]} - {x['cmd']}"))
    order = [L["name"] for L in d["layers"]]
    boxes = {}
    for L in d["layers"]:
        n = L["name"]
        pl = comp.placed.get(n)
        if (not L.get("visible", True) or comp.role(L) in ("backdrop", "pass") or pl is None or pl.box is None):
            continue
        cr = comp.results.get(n)
        if cr is not None and cr.texts and not cr.entries:
            continue                                     # a text layer does not stand on anything
        x0, y0, x1, y1 = pl.box
        boxes[n] = (x0 / W, y0 / H, x1 / W, y1 / H, float(pl.A.sum()) / (W * H))
    surfaces = [n for n, b in boxes.items() if b[2] - b[0] >= 0.45]
    for n, (x0, y0, x1, y1, area) in boxes.items():
        if n in surfaces or y1 < 0.45:                   # wide surfaces and things in the sky
            continue
        i = order.index(n)
        cx = (x0 + x1) / 2
        under = [m for m in order[:i] if m in boxes and boxes[m][0] <= cx <= boxes[m][2]
                 and boxes[m][3] >= y1 - 0.02]           # only what reaches below its base can carry it
        tops = [boxes[m][1] for m in surfaces if order.index(m) < i]
        if not under and not tops:
            continue                                     # only the backdrop under it: its ground is drawn there
        gaps = [0.0 if boxes[m][1] <= y1 <= boxes[m][3] + 0.01 else min(abs(y1 - boxes[m][1]), abs(y1 - boxes[m][3]))
                for m in under]
        cr = comp.results.get(n)
        if (cr is not None and any(dep[0] in ("box", "anchor") for dep in (cr.deps or []))) or \
                (gaps and min(gaps) <= 0.05):
            continue                                     # it follows another layer, or rests on one
        if (tops and y1 < min(tops) - 0.03) or (gaps and min(gaps) > 0.05):
            on = next((m for m in surfaces if order.index(m) < i), None) or (under[0] if under else None)
            if on and _placed_beside(pic, comp, n, on):
                continue                                 # the request puts it beside / in front of that layer
            gap = max([min(tops) - y1] if tops else [0.0]) if not gaps else min(gaps)
            hints.append((f"float:{n}", gap, f"{n} may float: its base (y {y1:.2f}) rests on nothing under it"
                          + (f" - put {n} on {on}" if on else "")))
    sc = safe_crop(pic, comp)
    d["things_box"] = sc[0] if sc else None
    sa = sc or safe_crop(pic, comp, place_whole=False)   # for a note that asks for less empty space
    d["things_box_asked"] = sa[0] if sa else None
    if sc:                                               # the requested things sit in a small part of the frame
        r, cover = sc
        hints.insert(0, ("crop", -cover, f"the requested things fill only {cover * 100:.0f}% of the frame: crop "
                         f"{r[0]:.2f} {r[1]:.2f} {r[2]:.2f} {r[3]:.2f}"))
    told = d.setdefault("suggested", {})
    out_, seen = [], set()
    for key, metric, text in hints:
        if text in seen or key in seen:
            continue
        seen.update((text, key))
        if key in told and metric <= float(told[key]) + 0.01:
            continue                                     # suggested before and not worse since
        out_.append((key, metric, text))
    for key, metric, text in out_[:limit]:
        told[key] = metric
    return [t for _, _, t in out_[:limit]]


def crop_suggestion(pic, asked=False):
    """The safe crop the last report found (see ``safe_crop``), or None; asked=True when a note asks for less empty
    space (then a backdrop item may be cropped too)."""
    return (pic.data.get("things_box_asked") if asked else None) or pic.data.get("things_box") or None


def safe_crop(pic, comp, place_whole=True):
    """(region, share of the frame the items fill) for a crop that only enlarges: the union of every item's
    layers (objects and text) plus a margin, grown to the frame's shape; None when a backdrop is itself a
    requested item (the place must stay whole), when the region is half the frame or more, or when any item's
    layer would be cut by it."""
    W, H = comp.W, comp.H
    names = []
    for n in pic.data.get("needs") or []:
        if n.get("cell"):
            fam = C.family(pic, n["cell"])
            if fam and all(comp.role(pic.layer(x)) == "backdrop" for x in fam):
                if n.get("line") and place_whole:
                    return None
                continue
            names += fam
        elif n.get("layer"):
            names.append(n["layer"])
        elif n.get("kind") == "text":
            t = C._text_layer(pic, comp, n["key"])
            if t:
                names.append(t)
    boxes = []
    for x in dict.fromkeys(names):
        if x not in pic.names():
            continue
        L = pic.layer(x)
        pl = comp.placed.get(x)
        if not L.get("visible", True) or pl is None or pl.box is None or comp.role(L) in ("backdrop", "pass"):
            continue
        boxes.append((pl.box[0] / W, pl.box[1] / H, pl.box[2] / W, pl.box[3] / H))
    if not boxes:
        return None
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    if x0 < 0 or y0 < 0 or x1 > 1 or y1 > 1:
        return None                                      # something already leaves the frame
    m = 0.06
    r = crop_region(x0 - m, y0 - m, x1 + m, y1 + m)
    if (r[2] - r[0]) * (r[3] - r[1]) >= 0.5:
        return None
    if any(b[0] < r[0] - 1e-3 or b[1] < r[1] - 1e-3 or b[2] > r[2] + 1e-3 or b[3] > r[3] + 1e-3 for b in boxes):
        return None
    return r, (x1 - x0) * (y1 - y0)


def _placed_beside(pic, comp, a, b):
    """The request places layer a beside / in front of / behind layer b (so 'put a on b' would be wrong)."""
    for n in pic.data.get("needs", []):
        if n.get("kind") != "place":
            continue
        x, rel_, y = (list(n.get("key") or []) + [None, None, None])[:3]
        if rel_ in ("on", "in", "on top of") or not x or not y:
            continue
        lx, ly = C.layers_for(pic, comp, x), C.layers_for(pic, comp, y)
        if (a in lx and b in ly) or (b in lx and a in ly):
            return True
    return False


def crop_region(x0, y0, x1, y1, aspect=1.0):
    """A region of the frame (fractions) grown to the frame's shape and kept inside it."""
    w, h = max(0.02, x1 - x0), max(0.02, y1 - y0)
    side = min(1.0, max(w, h))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    cx = min(max(cx, side / 2), 1 - side / 2)
    cy = min(max(cy, side / 2), 1 - side / 2)
    return [round(cx - side / 2, 4), round(cy - side / 2, 4), round(cx + side / 2, 4), round(cy + side / 2, 4)]


# ============================================================================ verbs

def cmd_new(a, adir):
    args = Args("new", a, {"--prompt": 1, "--size": 1, "--kind": 1, "--seed": 1, "--out": 1, "--from": 1})
    if not args.pos:
        raise StudioError('new needs a NAME: new NAME --prompt "the request" [--from DRAFT.py]')
    raw = args.pos[0]
    name = re.sub(r"[^a-z0-9_]", "", re.sub(r"[\s\-.]+", "_", raw.strip().lower())).strip("_")
    if not name:
        raise StudioError(f"picture name {raw!r}: use letters, digits and _ (e.g. bakery_sign)")
    if not name[0].isalpha():
        name = "p_" + name
    if name != raw:
        say(f"name: {name} (picture names use lower-case letters, digits and _)")
    prompt = args.opt.get("--prompt") or " ".join(args.pos[1:]) or ""
    if not prompt:
        raise StudioError("new needs the request word for word", fix=f'new {name} --prompt "the request"')
    src_draft = args.opt.get("--from")
    if src_draft and not os.path.isfile(src_draft):
        raise StudioError(f"no file {src_draft}: write the draft there first (one '# %% name' section per "
                          f"requested thing), then run this again")
    size = (1920, 1080)
    if args.opt.get("--size"):
        m = re.match(r"^\s*(\d+)\s*[x*X, ]\s*(\d+)\s*$", args.opt["--size"])
        if not m:
            raise StudioError(f"--size {args.opt['--size']!r}: write WxH, e.g. --size 1920x1080")
        size = (int(m.group(1)), int(m.group(2)))
    folder = os.path.join(adir, name)
    if src_draft and os.path.isfile(os.path.join(folder, PICTURE)):
        pic = open_picture(adir, name)                   # run again: the draft goes into the same picture
        say(f"picture {name} exists: the draft goes into it")
        if prompt != pic.data.get("prompt"):
            pic.data["prompt"] = prompt
        if args.opt.get("--out"):
            pic.data["out"] = _out_arg(args.opt["--out"], name)
        sess = Session(pic)
        sess.sync()
        return _add_from(sess, src_draft)
    kind = args.opt.get("--kind")
    whole = kind if kind in WHOLE_KINDS else None
    if not kind:
        try:
            from .. import catalog
            found = catalog.find(prompt).kind
        except Exception:                                # noqa: BLE001
            found = None
        whole = found if found in WHOLE_KINDS and not src_draft else None
        kind = whole or "painting"
    try:
        R = __import__("pixelcanvas.studio.engine", fromlist=["_render_module"])._render_module()
        R.resolve_kind(kind)
    except Exception as e:                               # noqa: BLE001
        if "UserError" in type(e).__name__:
            raise StudioError(str(e).splitlines()[0]) from None
    seed = int(_num(args.opt.get("--seed", 1), "--seed"))
    pic = Picture.create(adir, name, prompt, size, kind, seed)
    if args.opt.get("--out"):
        pic.data["out"] = _out_arg(args.opt["--out"], name)
    draft = os.path.join(pic.folder, "draft.py")
    from ..medium import detect as _detect, LABELS as _MLAB
    med = None if whole else _detect(prompt)
    if med:
        pic.data["scene"]["medium"] = med
    if whole:
        pic.data["whole"] = whole
        texts = [n for n in pic.data["needs"] if n["kind"] == "text"]
        what = {"floorplan": "floor plan", "ui": "screen"}.get(whole, whole)
        pic.data["needs"] = texts + [{"kind": "eye", "key": f"the {what} shows the data of the request",
                                      "text": f"the {what} shows the data of the request"}]
        for i, n in enumerate(pic.data["needs"], 1):
            n.update(id=i, src="request", layer=None, ok=None, why=None, on=None)
    pic.snapshot(f"new {name}", {})
    pic.save()
    set_current(adir, name)
    fileline = (f"; picture file: {pic.data['out']} (kept up to date after every command)"
                if pic.data.get("out") else "")
    if med:
        fileline += (f"; medium: {_MLAB[med]} from the request (painted over the drawing after every render; "
                     f"set scene medium=off draws without it)")
    if whole:
        needs = pic.data["needs"]
        say(f"new picture: {ab(pic.folder)} ({size[0]}x{size[1]}, {kind}){fileline}; the request has {len(needs)} "
            f"item{'s' if len(needs) != 1 else ''}:")
        for n in needs:
            say(f"  [ ] {n['id']} {n['text']}")
        say(f"a {kind} is one whole picture: its scene file draws it, marks and stamps go on top as layers")
        say("NEXT: " + whole_next(pic))
        return 0
    if src_draft:
        sess = Session(pic)
        sess.sync()
        return _add_from(sess, src_draft)
    say(f"new picture: {ab(pic.folder)} ({size[0]}x{size[1]}, {kind}){fileline}")
    say(f"write the draft {ab(draft)}: one '# %% name' section per requested thing, back to front")
    say("each section's first line is its inventory line: # apple: 3, green, one bitten | in bowl "
        "(count, colours, 'text', states | places)")
    say("a thing's parts and the text written on it stay in its section; names defined in a section are seen by "
        "the sections below it")
    say("NEXT: " + f"write {ab(draft)}, then: {me()} add --from {ab(draft)}")
    return 0


def _out_arg(v, name):
    """The picture file: absolute (relative to where the command runs), a .png added to a folder."""
    v = str(v).strip().strip("'\"")
    if not os.path.splitext(v)[1]:
        v = os.path.join(v, f"{name}.png")
    return os.path.abspath(v).replace("\\", "/")


WHOLE_KINDS = ("chart", "dashboard", "diagram", "flowchart", "floorplan", "drawing", "ui")


def whole_next(pic):
    """The two commands that start a chart / diagram / UI picture: its scene file, then the base layer."""
    from .. import cli
    root = cli.runtime_root()
    tool = os.path.abspath(os.path.join(root, "tools", "new_scene.py")).replace("\\", "/") if root else "new_scene.py"
    base = f"{pic.name}_base"
    folder = rel(pic.folder)
    q = pic.data["prompt"].replace('"', '\\"')
    return (f'python {tool} {base} --prompt "{q}" --dir {folder}, then: {me()} add base --scene '
            f"{folder}/{base}.py")


_HEAD_RE = re.compile(r"^\s*#\s*%%")


def section_head(line):
    """(name as written, role or None) of a '# %% name [role]' line; None for any other line. A '# %%' line that
    names no section raises (it is never code of the section before it)."""
    if not _HEAD_RE.match(line):
        return None
    rest = re.sub(r"^\s*#\s*%+", "", line).strip()
    role = None
    m = re.search(r"[\[(]\s*(pass|backdrop|object)\s*[\])]", rest, re.I)
    if m:
        role = m.group(1).lower()
        rest = (rest[:m.start()] + " " + rest[m.end():]).strip()
    rest = re.split(r"\s+#|\s+-\s+|\s*:\s", rest + " ", maxsplit=1)[0].strip().rstrip(":")
    key = layer_key(rest)
    if not rest or not key or not key[0].isalpha():
        raise StudioError(f"draft line {line.strip()!r}: write the section's name after '# %%' (e.g. '# %% fox', "
                          "'# %% sled dog', '# %% finish [pass]')")
    return rest, role


def split_sections(text):
    """'# %% name [role]' sections of a draft -> [{name (layer), item, role, src, lines (draft line of each cell
    line), head}]. Code before the first section (imports, helpers) goes in every cell, after its leading
    comment lines, so each section's first comment line stays its inventory line."""
    sections, pre, cur = [], [], None
    for i, ln in enumerate(text.splitlines(), 1):
        h = section_head(ln)
        if h is not None:
            cur = {"item": h[0], "name": layer_key(h[0]), "role": h[1], "body": [], "head": ln.strip(), "at": i}
            sections.append(cur)
            continue
        (cur["body"] if cur else pre).append((ln, i))
    if not sections:
        raise StudioError("the draft has no '# %% name' lines: put one before each thing's code "
                          "(e.g. '# %% sky', '# %% fox', '# %% title')")
    prelude = [(ln, i) for ln, i in pre if ln.strip() and not ln.strip().startswith("#")]
    seen = set()
    for s in sections:
        check_name(s["name"])
        if s["name"].lower() in seen:
            raise StudioError(f"the draft has two sections called {s['item']}: give each thing its own name")
        seen.add(s["name"].lower())
        body = s.pop("body")
        k = 0
        while k < len(body) and (not body[k][0].strip() or body[k][0].strip().startswith("#")):
            k += 1
        lead = [x for x in body[:k] if x[0].strip()]
        rows = lead + prelude + body[k:]
        while rows and not rows[-1][0].strip():
            rows.pop()
        s["src"] = "\n".join(r[0] for r in rows) + "\n"
        s["lines"] = [r[1] for r in rows]
    return sections


def split_draft(text):
    """'# %% name [role]' sections of a draft -> [(layer name, role or None, source)]."""
    return [(s["name"], s["role"], s["src"]) for s in split_sections(text)]


def _code_lines(src):
    return [ln for ln in src.splitlines() if ln.strip() and not ln.strip().startswith("#")]


def _add_from(sess, path):
    """add --from DRAFT.py: every section becomes (or updates) its layer; unchanged sections come from the
    cache, moves and effects of sections that stay are kept, a section that left the draft is removed."""
    pic = sess.pic
    if not os.path.isfile(path):
        raise StudioError(f"no file {path}")
    with open(path, encoding="utf-8-sig") as f:
        secs = split_sections(f.read())
    if not any(any(not ln.strip().startswith("cv.text(") for ln in _code_lines(s["src"])) for s in secs):
        say(f"{os.path.basename(path)} has no drawing code yet: write it to {ab(path)} (that exact path), then "
            "run this again")
        return 2
    apath = os.path.abspath(path)
    d = pic.data
    dr = d.setdefault("draft", {})
    rec = dr.setdefault("cells", {})
    notices = []
    names = [s["name"] for s in secs]
    on_disk = pic.read_cells()
    for cell, info in list(rec.items()):                 # sections that left this draft
        if info.get("path") == apath and cell not in names:
            rec.pop(cell, None)
            if cell in on_disk:
                try:
                    os.remove(pic.cell_path(cell))
                except OSError:
                    pass
                notices.append(f"removed: {cell} (no longer in the draft)")
    for s in secs:
        name, src = s["name"], s["src"]
        old = rec.get(name)
        cur = on_disk.get(name)
        if cur is not None and old and sha(cur) != old.get("sha") and cur != src:
            if sha(src) == old.get("sha"):               # the section did not change: the file edit stays
                notices.append(f"{name}: layers/{name}.py was edited after the draft; it is kept (its draft "
                               "section is unchanged)")
                rec[name] = dict(old, path=apath, lines=s["lines"])
                continue
            notices.append(f"{name}: layers/{name}.py was edited after the draft; the draft section replaces it "
                           "(undo brings the edit back)")
        if cur != src:
            pic.write_cell(name, src)
        rec[name] = {"path": apath, "sha": sha(src), "lines": s["lines"]}
    dr["path"] = apath
    sess.sync(order=names)
    for s in secs:
        if s["role"] and s["name"] in pic.names():
            pic.layer(s["name"])["role"] = s["role"]
    sess.notices = notices
    sess.commit(f"add --from {os.path.basename(path)}", force_rev=True)
    return 1 if getattr(sess, "report_failed", False) else 0


def cmd_add(sess, a):
    args = Args("add", a, {"--from": 1, "--text": 1, "--box": 4, "--color": 1, "--font": 1, "--bold": 0,
                           "--kit": 1, "--role": 1, "--below": 1, "--above": 1, "--scene": 1})
    pic = sess.pic
    if args.opt.get("--from"):
        return _add_from(sess, args.opt["--from"])
    if not args.pos:
        raise StudioError("add needs a layer NAME: add NAME --text \"...\" | add NAME --kit \"creature('fox')\" | "
                          "add NAME --scene chart.py | add --from DRAFT.py")
    if args.opt.get("--scene"):
        return _add_scene(sess, args)
    name = args.pos[0]
    check_name(name)
    if args.opt.get("--text") is not None or (len(args.pos) > 1 and not args.opt.get("--kit")):
        text = args.opt.get("--text") if args.opt.get("--text") is not None else " ".join(args.pos[1:])
        bx = args.nums("--box") or [0.2, 0.04, 0.8, 0.17]
        color = _colour_arg(args.opt.get("--color") or "#fff4d6", "--color")
        font = args.opt.get("--font") or "serif"
        bold = ", bold=True" if args.opt.get("--bold") else ""
        src = (f"# {name}: the text {text!r}\n"
               f"cv.text({text!r}, box=({bx[0]:g}, {bx[1]:g}, {bx[2]:g}, {bx[3]:g}), color={color!r}, "
               f"font={font!r}{bold})\n")
    elif args.opt.get("--kit"):
        src = kit_cell(name, args.opt["--kit"], args.pos[1:])
    else:
        raise StudioError(f"add {name}: say what it draws: --text \"...\", --kit \"creature('fox', pose='walk')\", "
                          f"or write {rel(pic.cell_path(name))} and run render")
    if name in pic.names() or os.path.isfile(pic.cell_path(name)):
        raise StudioError(f"a layer called {name} already exists: edit {rel(pic.cell_path(name))} or pick another name")
    pic.write_cell(name, src)
    sess.sync()
    L = pic.layer(name)
    if args.opt.get("--role"):
        L["role"] = _choice("role", args.opt["--role"], ("object", "backdrop", "pass"))
    if args.opt.get("--below") or args.opt.get("--above"):
        _reorder(pic, name, "below" if args.opt.get("--below") else "above",
                 args.opt.get("--below") or args.opt.get("--above"))
    sess.commit(f"add {name}", force_rev=True)
    return 0


def _add_scene(sess, args):
    """add NAME --scene FILE.py: a layer that is the whole picture a scene file draws (chart, diagram, UI)."""
    pic = sess.pic
    name = args.pos[0]
    check_name(name)
    path = args.opt["--scene"]
    if not os.path.isfile(path):
        raise StudioError(f"no file {path} (a scene file with render(width, height, ...): tools/new_scene.py makes one)")
    try:
        relp = os.path.relpath(os.path.abspath(path), pic.folder)
    except ValueError:                                   # another drive
        relp = os.path.abspath(path)
    relp = relp.replace("\\", "/")
    if name in pic.names() or os.path.isfile(pic.cell_path(name)):
        raise StudioError(f"a layer called {name} already exists: edit {rel(pic.cell_path(name))} or pick another name")
    pic.write_cell(name, f"# role: backdrop\n# {name}: the whole picture {os.path.basename(path)} draws\n"
                         f"cv = scene({relp!r})\n")
    sess.sync()
    L = pic.layer(name)
    L["role"] = "backdrop"
    if args.opt.get("--below") or args.opt.get("--above"):
        _reorder(pic, name, "below" if args.opt.get("--below") else "above",
                 args.opt.get("--below") or args.opt.get("--above"))
    elif len(pic.data["layers"]) > 1:
        _reorder(pic, name, "bottom")
    sess.commit(f"add {name} --scene {os.path.basename(path)}", force_rev=True)
    return 0


_KIT_FIRST_CV = {"creature", "tabletop", "Ground", "ball", "cylinder", "cuboid", "lathe"}


def kit_cell(name, spec, extra=()):
    """--kit "creature('fox', pose='walk')" or --kit creature fox pose=walk -> a cell source."""
    import ast
    spec = spec.strip()
    if "(" not in spec:
        toks = shlex.split(spec, posix=True) + list(extra)
        if not toks:
            raise StudioError("--kit needs a call, e.g. --kit \"creature('fox', pose='walk')\"")
        fn, rest = toks[0], toks[1:]
        pos, kw = [], []
        for t in rest:
            if "=" in t:
                k, v = t.split("=", 1)
                kw.append(f"{k}={_lit(v)}")
            else:
                pos.append(_lit(t))
        spec = f"{fn}({', '.join(pos + kw)})"
    try:
        tree = ast.parse(spec, mode="eval")
    except SyntaxError as e:
        raise StudioError(f"--kit {spec!r} is not a call ({e.msg}); write it like creature('fox', pose='walk')") from None
    call = tree.body
    if not isinstance(call, ast.Call):
        raise StudioError(f"--kit {spec!r} is not a call; write it like creature('fox', pose='walk')")
    first = call.args[0] if call.args else None
    has_cv = isinstance(first, ast.Name) and first.id == "cv"
    if not has_cv:
        call.args.insert(0, ast.Name(id="cv", ctx=ast.Load()))
    code = ast.unparse(call) if hasattr(ast, "unparse") else spec
    var = name if re.match(r"^[A-Za-z_]\w*$", name) else "obj"
    return f"# {name}\n{var} = {code}\n"


def _lit(v):
    v = v.strip()
    if re.match(r"^-?\d+(\.\d+)?$", v):
        return v
    if re.match(r"^-?[\d.]+(,-?[\d.]+)+$", v):
        return "(" + ", ".join(v.split(",")) + ")"
    if v in ("True", "False", "None", "true", "false", "none"):
        return v.capitalize()
    return repr(v)


def _colour_arg(v, what):
    from ..color import rgb
    s = str(v).strip().strip("'\"")
    if re.match(r"^[0-9a-fA-F]{3}$|^[0-9a-fA-F]{6}$", s):
        s = "#" + s
    try:
        rgb(s)
    except Exception:                                    # noqa: BLE001
        hx = C.colour_hex(s)
        if hx:
            return "#" + hx
        raise StudioError(f"{what}: {v!r} is not a colour (write ffcf70, #ffcf70 or a name such as gold)") from None
    return s


def _choice(what, v, choices):
    if v in choices:
        return v
    near = difflib.get_close_matches(str(v), list(choices), n=1, cutoff=0.5)
    raise StudioError(f"{what} {v!r}" + (f" - did you mean {near[0]}?" if near else "") +
                      f" (choices: {', '.join(choices)})")


def _layers_arg(pic, names, verb):
    if not names:
        raise StudioError(f"{verb} needs a layer name (layers: {', '.join(pic.names()) or 'none'})")
    return _expand(pic, names, verb)


def _expand(pic, tokens, verb="", strict=True):
    """Layer names from tokens: a name, or NAME* (that cell and its copies, and every name starting so).
    -> names in order of first mention, no repeats."""
    out = []
    names = pic.names()
    for t in tokens:
        t = str(t).strip().strip("'\"")
        if t not in names and layer_key(t.rstrip("*")) in names:      # "sled dog" is layer sled_dog
            t = layer_key(t.rstrip("*")) + ("*" if t.endswith("*") else "")
        if t in names:
            hits = [t]
        else:
            pre = t.rstrip("*")
            fam = [L["name"] for L in pic.data["layers"]
                   if L["name"].startswith(pre) or (L.get("cell") or L["name"]) == pre] if pre else []
            hits = fam if t.endswith("*") else []
            if not hits:
                if not strict:
                    return None
                pic.layer(pre or t)                      # raises with a did-you-mean
        for h in hits:
            if h not in out:
                out.append(h)
    return out


def _layers_then(pic, tokens, verb):
    """[layers ...] VALUE: the longest run of leading tokens that are layers, and the rest."""
    for i in range(len(tokens), 0, -1):
        got = _expand(pic, tokens[:i], verb, strict=False)
        if got:
            return got, tokens[i:]
    pic.layer(str(tokens[0]).rstrip("*"))
    return [], tokens


def _cropf(sess, L):
    c = crop_of(sess.pic.data, L, sess.cells.get(L.get("cell") or L["name"]))
    return c[2] if c else 1.0


def _shift(sess, L, ddx, ddy):
    """Move a layer by ddx, ddy as seen in the picture (a cropped picture shows layers enlarged)."""
    f = _cropf(sess, L)
    L["dx"] = round(float(L.get("dx", 0.0)) + ddx / f, 4)
    L["dy"] = round(float(L.get("dy", 0.0)) + ddy / f, 4)


def _no_pass(sess, comp, name, verb):
    if comp.role(sess.pic.layer(name)) == "pass":
        raise StudioError(f"{verb} {name}: {name} is a pass (it changes the whole picture); it cannot be moved, "
                          "scaled or rotated")


def cmd_move(sess, a):
    args = Args("move", a, {"--to": 2, "--by": 2})
    pic = sess.pic
    names = _layers_arg(pic, args.pos, "move")
    if ("--to" in args.opt) == ("--by" in args.opt):
        raise StudioError("move needs --to X Y (where the centre goes) or --by DX DY (fractions of the picture)")
    comp = sess.geometry()
    for n in names:
        _no_pass(sess, comp, n, "move")
    if "--to" in args.opt:
        X, Y = args.nums("--to")
        bs = [sess.box(comp, n) for n in names]
        cx = (min(b[0] for b in bs) + max(b[2] for b in bs)) / 2
        cy = (min(b[1] for b in bs) + max(b[3] for b in bs)) / 2
        dx, dy = X - cx, Y - cy
    else:
        dx, dy = args.nums("--by")
    for n in names:
        _shift(sess, pic.layer(n), dx, dy)
    sess.commit(f"move {' '.join(names)} " + (f"--to {args.opt['--to'][0]} {args.opt['--to'][1]}" if "--to" in args.opt
                                              else f"--by {dx:g} {dy:g}"), force_rev=True)
    return 0


def cmd_scale(sess, a):
    args = Args("scale", a, {})
    pic = sess.pic
    if len(args.pos) < 2:
        raise StudioError("scale needs a layer and a factor: scale fox 1.3 (1.3 x its size now; 0.8 makes it smaller)")
    *names, f = args.pos
    f = _num(f, "scale factor")
    if f <= 0:
        raise StudioError("scale factor must be above 0 (1.3 = 30% larger, 0.8 = 20% smaller)")
    names = _layers_arg(pic, names, "scale")
    comp = sess.geometry()
    for n in names:
        _no_pass(sess, comp, n, "scale")
        L = pic.layer(n)
        L["scale"] = round(float(L.get("scale", 1.0)) * f, 4)
    sess.commit(f"scale {' '.join(names)} {f:g}", force_rev=True)
    return 0


def cmd_rotate(sess, a):
    args = Args("rotate", a, {})
    pic = sess.pic
    if len(args.pos) < 2:
        raise StudioError("rotate needs a layer and degrees: rotate kite 15 (clockwise; -15 turns it back)")
    *names, deg = args.pos
    deg = _num(deg, "degrees")
    names = _layers_arg(pic, names, "rotate")
    comp = sess.geometry()
    for n in names:
        _no_pass(sess, comp, n, "rotate")
        L = pic.layer(n)
        L["rotate"] = round((float(L.get("rotate", 0.0)) + deg + 180) % 360 - 180, 3)
    sess.commit(f"rotate {' '.join(names)} {deg:g}", force_rev=True)
    return 0


def cmd_flip(sess, a):
    args = Args("flip", a, {})
    pic = sess.pic
    names = _layers_arg(pic, args.pos, "flip")
    comp = sess.geometry()
    for n in names:
        _no_pass(sess, comp, n, "flip")
        L = pic.layer(n)
        L["flip"] = not bool(L.get("flip"))
    sess.commit(f"flip {' '.join(names)}", force_rev=True)
    return 0


def _fit(sess, comp, name, x0, y0, x1, y1, align="center"):
    """Scale and move a layer so what it draws fits inside the box (kept in proportion)."""
    L = sess.pic.layer(name)
    b = sess.box(comp, name)
    bw, bh = b[2] - b[0], b[3] - b[1]
    if x1 <= x0 or y1 <= y0:
        raise StudioError(f"box {x0:g} {y0:g} {x1:g} {y1:g}: x1 must be right of x0 and y1 below y0")
    f = min((x1 - x0) / max(bw, 1e-6), (y1 - y0) / max(bh, 1e-6))
    s0 = float(L.get("scale", 1.0))
    L["scale"] = round(s0 * f, 4)
    nw, nh = bw * f, bh * f
    px, py = _pivot(comp, name, b)                        # the point that stays put when scaling
    ncx, ny1 = px + ((b[0] + b[2]) / 2 - px) * f, py + (b[3] - py) * f
    tx = (x0 + x1) / 2
    ty1 = y1 if align == "bottom" else (y0 + y1) / 2 + nh / 2
    _shift(sess, L, tx - ncx, ty1 - ny1)


def _pivot(comp, name, b):
    """The point a layer scales about, in fractions: the landmark it hangs from, else its bottom centre."""
    pl = comp.placed.get(name)
    pv = getattr(pl, "pivot", None) if pl is not None else None
    if pv is None:
        return (b[0] + b[2]) / 2, b[3]
    return pv[0] / comp.W, pv[1] / comp.H


def cmd_fit(sess, a):
    args = Args("fit", a, {"--bottom": 0})
    pic = sess.pic
    if len(args.pos) < 2:
        raise StudioError("fit needs a layer and a box: fit fox 0.2 0.5 0.6 0.95 (x0 y0 x1 y1); several layers "
                          "share the box in a row: fit fish* 0.2 0.4 0.8 0.6")
    names, rest = _layers_then(pic, args.pos, "fit")
    x0, y0, x1, y1 = _numbers(rest, 4, "fit LAYER [LAYER ..] X0 Y0 X1 Y1")
    comp = sess.geometry()
    cw = (x1 - x0) / len(names)
    for i, name in enumerate(names):
        _no_pass(sess, comp, name, "fit")
        _fit(sess, comp, name, x0 + i * cw, y0, x0 + (i + 1) * cw, y1,
             "bottom" if args.opt.get("--bottom") else "center")
    sess.commit(f"fit {' '.join(names)} {x0:g} {y0:g} {x1:g} {y1:g}", force_rev=True)
    return 0


def _reorder(pic, name, how, other=None):
    lay = pic.data["layers"]
    i = pic.index(name)
    L = lay.pop(i)
    if how == "top":
        lay.append(L)
    elif how == "bottom":
        lay.insert(0, L)
    elif how == "up":
        lay.insert(min(i + 1, len(lay)), L)
    elif how == "down":
        lay.insert(max(i - 1, 0), L)
    else:
        if other == name:
            lay.insert(i, L)
            raise StudioError(f"order {name} {how} {other}: a layer cannot go {how} itself")
        try:
            j = pic.index(other)
        except StudioError:
            lay.insert(i, L)
            raise
        lay.insert(j + 1 if how == "above" else j, L)


def cmd_order(sess, a):
    args = Args("order", a, {})
    pic = sess.pic
    if len(args.pos) < 2:
        raise StudioError("order needs a layer and where: order fox top | bottom | up | down | above tree | below tree")
    hows = ("top", "bottom", "above", "below", "up", "down")
    k = next((i for i, t in enumerate(args.pos) if t in hows), None)
    if k is None:
        k = 1
    names = _layers_arg(pic, args.pos[:k], "order")
    how = _choice("order", args.pos[k] if k < len(args.pos) else "", hows)
    other = None
    if how in ("above", "below"):
        if len(args.pos) < k + 2:
            raise StudioError(f"order {names[0]} {how} needs the other layer: order {names[0]} {how} LAYER")
        other = args.pos[k + 1]
        pic.layer(other)
    seq = names if how in ("top", "below", "down") else list(reversed(names))
    if how == "up":
        seq = sorted(names, key=pic.index, reverse=True)
    elif how == "down":
        seq = sorted(names, key=pic.index)
    for name in seq:
        _reorder(pic, name, how, other)
    sess.commit(f"order {' '.join(names)} {how}" + (f" {other}" if other else ""), force_rev=True)
    return 0


def cmd_visible(sess, a, verb):
    args = Args(verb, a, {})
    pic = sess.pic
    names = _layers_arg(pic, args.pos, verb)
    for n in names:
        pic.layer(n)["visible"] = verb == "show"
    sess.commit(f"{verb} {' '.join(names)}", force_rev=True)
    return 0


def cmd_rm(sess, a):
    args = Args("rm", a, {})
    pic = sess.pic
    names = _layers_arg(pic, args.pos, "rm")
    for n in names:
        L = pic.layer(n)
        cell = L.get("cell") or n
        pic.data["layers"] = [x for x in pic.data["layers"] if x["name"] != n]
        if not any((x.get("cell") or x["name"]) == cell for x in pic.data["layers"]):
            try:
                os.remove(pic.cell_path(cell))
            except OSError:
                pass
            sess.cells.pop(cell, None)
        for nd in pic.data["needs"]:
            if nd.get("layer") == n:
                nd["layer"] = None
    sess.commit(f"rm {' '.join(names)}", force_rev=True)
    return 0


def cmd_copy(sess, a):
    args = Args("copy", a, {"--total": 1, "--n": 1, "--row": 4, "--grid": 4, "--cols": 1, "--by": 2})
    pic = sess.pic
    if not args.pos:
        raise StudioError("copy needs a layer: copy cup --total 3 --row 0.2 0.6 0.8 0.75 (makes cup2, cup3), "
                          "or copy cup saucer (one copy with that name)")
    src_name = args.pos[0]
    S = pic.layer(src_name)
    new = args.pos[1] if len(args.pos) > 1 else None
    if new:
        check_name(base_name(new).replace(" ", "_") if new[-1:].isdigit() else new)
    cell = S.get("cell") or src_name
    family = [L["name"] for L in pic.data["layers"] if (L.get("cell") or L["name"]) == cell]
    if args.opt.get("--total") is not None:
        if "--n" in args.opt:
            raise StudioError("copy takes --total N (how many in all) or --n K (how many more), not both")
        total = int(_num(args.opt["--total"], "--total"))
        if total < 1 or total > 60:
            raise StudioError("--total must be between 1 and 60")
        n = total - len(family)
        if n < 0 or (n == 0 and not (args.opt.get("--row") or args.opt.get("--grid"))):
            raise StudioError(f"there are already {len(family)} ({', '.join(family)}): rm one to have fewer, or "
                              f"lay them out again: copy {src_name} --total {len(family)} --row X0 Y0 X1 Y1",
                              fix=f"rm {family[-1]}")
    else:
        n = int(_num(args.opt.get("--n", 1), "--n"))
        if n < 1 or n > 60:
            raise StudioError("--n must be between 1 and 60")
    comp = sess.geometry()
    _no_pass(sess, comp, src_name, "copy")
    stem = new or (re.sub(r"[0-9]+$", "", src_name) or src_name)
    names = []
    taken = set(pic.names())
    for i in range(n):
        nm = stem
        k = 2
        while nm in taken:
            nm = f"{re.sub(r'[0-9]+$', '', stem)}{k}"
            k += 1
        taken.add(nm)
        names.append(nm)
    at = pic.index(family[-1] if args.opt.get("--total") is not None else src_name) + 1
    for i, nm in enumerate(names):
        L = {k: v for k, v in S.items() if k not in ("name", "good", "desc")}
        L["name"] = nm
        L["cell"] = cell
        pic.data["layers"].insert(at + i, L)
    b = sess.box(comp, src_name)
    group = ([x for x in family] if args.opt.get("--total") is not None else [src_name]) + names
    if args.opt.get("--row") or args.opt.get("--grid"):
        box = args.nums("--row") or args.nums("--grid")
        x0, y0, x1, y1 = box
        cols = len(group) if args.opt.get("--row") else int(_num(args.opt.get("--cols") or
                                                                 math.ceil(math.sqrt(len(group))), "--cols"))
        rows = int(math.ceil(len(group) / cols))
        cw, ch = (x1 - x0) / cols, (y1 - y0) / rows
        comp2 = sess.geometry()
        wide = max(sess.box(comp2, nm)[2] - sess.box(comp2, nm)[0] for nm in group)
        if wide * 1.2 > cw:                              # apart, never touching: a gap of a fifth of one's width
            f = cw / (wide * 1.2)
            for nm in group:
                L = pic.layer(nm)
                L["scale"] = round(float(L.get("scale", 1.0)) * f, 4)
            need = min(1.0, wide * cols * 1.2)
            mid = min(max((x0 + x1) / 2, need / 2), 1 - need / 2)
            say(f"copy: {len(group)} of {wide:.2f} wide do not fit apart in {x1 - x0:.2f}: each scaled {f:.2f} to "
                f"leave a gap; to keep their size use a wider {'row' if args.opt.get('--row') else 'grid'}: "
                f"{mid - need / 2:.2f} {y0:.2f} {mid + need / 2:.2f} {y1:.2f}")
            comp2 = sess.geometry()
        for i, nm in enumerate(group):                   # its bottom centre goes to the bottom centre of its cell
            r, c = divmod(i, cols)
            bb = sess.box(comp2, nm)
            _shift(sess, pic.layer(nm), x0 + (c + 0.5) * cw - (bb[0] + bb[2]) / 2, y0 + (r + 1) * ch - bb[3])
    else:
        step = args.nums("--by") or [min((b[2] - b[0]) * 1.25, 0.3), 0.0]
        base = len(group) - len(names)
        for i, nm in enumerate(names, base):
            L = pic.layer(nm)
            dx = step[0] * i
            if b[2] + dx > 1.0 and step[0] > 0:
                dx = -step[0] * i
            L["dx"], L["dy"] = float(S.get("dx", 0.0)), float(S.get("dy", 0.0))
            _shift(sess, L, dx, step[1] * i)
    fam = [L["name"] for L in pic.data["layers"] if (L.get("cell") or L["name"]) == cell]
    if all(x.startswith(cell) for x in fam):
        say(f"{cell}* names all {len(fam)} in one command (e.g. recolor {cell}* COLOUR, scale {cell}* 1.2)")
    sess.commit(f"copy {src_name} -> {' '.join(names)}" if names else
                f"copy {src_name} --total {len(group)} (laid out again)", force_rev=True)
    return 0


# ---- set

LAYER_KEYS = ("visible", "opacity", "dx", "dy", "scale", "rotate", "flip", "outline", "outline_width", "shadow",
              "shadow_strength", "shadow_size", "glow", "glow_size", "glow_strength", "tint", "tint_amount", "blur",
              "role", "label", "alias")
SCENE_KEYS = ("light", "medium", "medium_strength", "medium_mark", "medium_palette", "medium_paper", "grade", "grade_amount", "paper", "background", "shadow_color", "kind", "seed",
              "size", "prompt", "reviews", "cell_timeout", "out")
_OFF = ("none", "off", "no", "false", "0", "")
_ON = ("on", "yes", "true", "1")
_LIGHTS = {"top-left": [-0.6, -0.8], "top-right": [0.6, -0.8], "left": [-1.0, -0.2], "right": [1.0, -0.2],
           "top": [0.0, -1.0], "front": [0.0, -0.3], "bottom-left": [-0.6, 0.6], "bottom-right": [0.6, 0.6]}


def _key(owner, k, keys):
    if k in keys:
        return k
    from .._args import _resolve
    r = _resolve(k, set(keys), {"color": "tint", "colour": "tint", "alpha": "opacity", "x": "dx", "y": "dy",
                               "size": "scale", "angle": "rotate", "stroke": "outline", "border": "outline",
                               "light_dir": "light", "mood": "grade", "bg": "background"})
    if r:
        return r
    near = difflib.get_close_matches(k, keys, n=2, cutoff=0.55)
    raise StudioError(f"{owner}: unknown key {k!r}" + (f" - did you mean {' or '.join(near)}?" if near else "")
                      + f" (keys: {', '.join(keys)})")


def layer_value(k, v):
    s = str(v).strip().strip("'\"")
    low = s.lower()
    if k in ("visible", "flip"):
        if low in _ON:
            return True
        if low in _OFF:
            return False
        raise StudioError(f"{k}={s}: write {k}=on or {k}=off")
    if k in ("opacity", "tint_amount", "shadow_strength", "glow_strength"):
        x = _num(s, k)
        return round(x / 100.0 if x > 1.0 and k == "opacity" else x, 4)
    if k in ("dx", "dy", "scale", "rotate", "outline_width", "glow_size", "blur", "shadow_size"):
        x = _num(s, k)
        if k == "scale" and x <= 0:
            raise StudioError("scale must be above 0")
        return round(x, 5)
    if k in ("outline", "glow", "tint"):
        if low in _OFF:
            return None
        if low in _ON:
            return {"outline": "#1d1830", "glow": "#ffd27a", "tint": None}[k]
        return _colour_arg(s, k)
    if k == "shadow":
        if low in _OFF:
            return None
        if low in _ON:
            return "ground"
        return _choice("shadow", low, ("ground", "drop"))
    if k == "role":
        if low in ("auto", "none"):
            return None
        return _choice("role", low, ("object", "backdrop", "pass"))
    if k == "label":
        return s or None
    if k == "alias":
        return [x.strip() for x in s.split(",") if x.strip()] or None
    return s


def scene_value(pic, k, v):
    s = str(v).strip().strip("'\"")
    low = s.lower()
    if k == "light":
        if low in _LIGHTS:
            return _LIGHTS[low]
        nums = _split([s])
        if len(nums) != 2:
            raise StudioError(f"light={s}: two numbers towards the light (light=0.6,-0.4) or one of "
                              f"{', '.join(_LIGHTS)}")
        return [_num(x, "light") for x in nums]
    if k == "grade":
        if low in _OFF:
            return None
        from ..illustrate import MOODS
        return _choice("grade", low.replace("_", " ").replace("-", " "), tuple(MOODS))
    if k == "grade_amount":
        return round(_num(s, k), 3)
    if k == "paper":
        if low in _OFF:
            return False
        if low in _ON:
            return True
        return _colour_arg(s, "paper")
    if k in ("background", "shadow_color"):
        return _colour_arg(s, k)
    if k == "medium":
        if low in _OFF:
            return None
        from ..medium import canonical, MEDIA
        return _choice("medium", canonical(low), MEDIA + ("off",)) if canonical(low) != "off" else None
    if k == "medium_strength":
        if low in _OFF:
            return None
        x = _num(s, k)
        if not 0.0 <= x <= 2.0:
            raise StudioError(f"medium_strength={s}: 0 (the lit picture) to 2 (a strong medium); 1 is the default")
        return round(x, 3)
    if k == "medium_mark":
        if low in _OFF:
            return None
        x = _num(s, k)
        if not 0.2 <= x <= 4.0:
            raise StudioError(f"medium_mark={s}: the stroke / wash / pixel size, 0.5 (fine) to 3 (broad); 1 is the "
                              f"default")
        return round(x, 3)
    if k == "medium_palette":
        if low in _OFF:
            return None
        x = int(_num(s, k))
        if not 2 <= x <= 64:
            raise StudioError(f"medium_palette={s}: the number of colours, 2 to 64")
        return x
    if k == "medium_paper":
        if low in _OFF:
            return None
        return _colour_arg(s, k)
    if k == "kind":
        R = __import__("pixelcanvas.studio.engine", fromlist=["_render_module"])._render_module()
        try:
            R.resolve_kind(low)
        except Exception as e:                           # noqa: BLE001
            raise StudioError(str(e).splitlines()[0]) from None
        return low
    if k == "seed":
        return int(_num(s, "seed"))
    if k == "reviews":
        x = int(_num(s, "reviews"))
        if x < 0 or x > 20:
            raise StudioError("reviews=N: how many looks before the picture is finished, 0 to 20")
        return x
    if k == "cell_timeout":
        x = _num(s, "cell_timeout")
        if x < 0:
            raise StudioError("cell_timeout=N: seconds a cell may run (0 = no limit)")
        return round(x, 2)
    if k == "size":
        m = re.match(r"^\s*(\d+)\s*[x*X, ]\s*(\d+)\s*$", s)
        if not m:
            raise StudioError(f"size={s}: write size=1920x1080")
        return [int(m.group(1)), int(m.group(2))]
    return s


def _pairs(tokens, what):
    out = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if "=" not in t:
            if i + 1 < len(tokens) and tokens[i + 1].startswith("="):          # "key =value"
                t = t + tokens[i + 1]
                i += 1
            elif out:                                    # a value runs up to the next key=: grade=warm night
                k0, v0 = out[-1]
                out[-1] = (k0, f"{v0} {t}".strip())
                i += 1
                continue
            else:
                raise StudioError(f"{what}: {t!r} needs a value: key=value (e.g. outline=on shadow=ground)")
        k, v = t.split("=", 1)
        if v == "" and i + 1 < len(tokens) and "=" not in tokens[i + 1]:    # "key= value"
            v = tokens[i + 1]
            i += 1
        out.append((k.strip(), v))
        i += 1
    return out


def cmd_set(sess, a):
    pic = sess.pic
    if not a:
        raise StudioError("set needs a layer (or scene) and key=value: set fox outline=on shadow=ground | "
                          "set scene light=top-right grade=golden")
    k = next((i for i, t in enumerate(a) if "=" in t), len(a))
    targets, rest = a[:k], a[k:]
    target = targets[0] if targets else a[0]
    if not rest:
        raise StudioError(f"set {target} needs key=value pairs (layer keys: {', '.join(LAYER_KEYS)})")
    changes = []
    if targets != ["scene"] and len(targets) > 1 or (target != "scene" and "*" in target):
        names = _layers_arg(sess.pic, targets, "set")
        for n in names:
            cmd_set_one(sess, n, rest, changes)
        sess.commit(f"set {' '.join(names)} {' '.join(dict.fromkeys(changes))}", force_rev=True)
        return 0
    if target == "scene":
        for k, v in _pairs(rest, "set scene"):
            k = _key("set scene", k, SCENE_KEYS)
            val = scene_value(pic, k, v)
            if k == "out":
                if str(v).strip().lower() in _OFF:
                    pic.data.pop("out", None)
                else:
                    pic.data["out"] = _out_arg(v, pic.name)
                    say(f"picture file: {pic.data['out']} (kept up to date after every command)")
                changes.append(f"{k}={v}")
                continue
            if k in ("kind", "seed", "size", "prompt"):
                pic.data[k] = val                        # a new prompt: its quoted texts are the next render's
            else:
                pic.data["scene"][k] = val
            changes.append(f"{k}={v}")
    else:
        cmd_set_one(sess, target, rest, changes)
    sess.commit(f"set {target} {' '.join(changes)}", force_rev=True)
    return 0


def cmd_set_one(sess, target, rest, changes):
    L = sess.pic.layer(target)
    comp = None
    for k, v in _pairs(rest, f"set {target}"):
        k = _key(f"set {target}", k, LAYER_KEYS)
        val = layer_value(k, v)
        if k in ("dx", "dy", "scale", "rotate", "flip"):
            comp = comp or sess.geometry()
            _no_pass(sess, comp, target, "set")
        if val is None and k in L:
            L.pop(k, None)
        elif val is not None:
            L[k] = val
        changes.append(f"{k}={v}")


def cmd_recolor(sess, a):
    args = Args("recolor", a, {"--amount": 1})
    pic = sess.pic
    if len(args.pos) < 2:
        raise StudioError("recolor needs a layer and a colour: recolor lantern ffcf70 (or a name: gold; several "
                          "layers: recolor duck duck2 f5d142, recolor duck* f5d142)")
    names, rest = _layers_then(pic, args.pos[:-1], "recolor")
    col = " ".join(rest + [args.pos[-1]])
    tint = None if col.lower() in _OFF else _colour_arg(col, "recolor")
    for name in names:
        L = pic.layer(name)
        if tint is None:
            L.pop("tint", None)
        else:
            L["tint"] = tint
            L["tint_amount"] = round(_num(args.opt.get("--amount", 1.0), "--amount"), 3)
    amt = args.opt.get("--amount")
    sess.commit(f"recolor {' '.join(names)} {col}" + (f" --amount {amt}" if amt else ""), force_rev=True)
    return 0


_PUT = ("on", "above", "below", "left-of", "right-of", "in", "next-to", "under", "over", "beside")


def cmd_put(sess, a):
    args = Args("put", a, {"--gap": 1})
    pic = sess.pic
    pos = [p for p in args.pos if p.lower() not in ("of", "to", "the")]
    words = ("on", "above", "below", "left", "right", "next", "in", "under", "over", "beside", "left-of", "right-of",
             "next-to", "inside")
    k = next((i for i, t in enumerate(pos) if t.lower() in words), None)
    if k is None or k == 0 or k >= len(pos) - 1:
        raise StudioError("put needs: put A on|above|below|left-of|right-of|in|next-to B (e.g. put cup on table, "
                          "put fish* in bowl)")
    rel_ = pos[k].lower()
    rel_ = {"left": "left-of", "right": "right-of", "next": "next-to", "under": "below", "over": "above",
            "beside": "next-to", "inside": "in"}.get(rel_, rel_)
    rel_ = _choice("put", rel_, _PUT[:7])
    names = _expand(pic, pos[:k], "put")
    B = pos[-1]
    pic.layer(B)
    names = [n for n in names if n != B]
    if not names:
        raise StudioError("put needs two different layers")
    gap = _num(args.opt.get("--gap", 0.02), "--gap")
    for i, A in enumerate(names):
        _put_one(sess, A, rel_, B, gap, (i, len(names)))
    sess.commit(f"put {' '.join(names)} {rel_} {B}", force_rev=True)
    return 0


def _put_one(sess, A, rel_, B, gap, slot=(0, 1), depth=0):
    """Place one layer A against B; slot (i, n): the i-th of n layers put together (spread side by side)."""
    pic = sess.pic
    comp = sess.geometry()
    _no_pass(sess, comp, A, "put")
    _no_pass(sess, comp, B, "put")
    a_ = C.solid_box(comp, A) or sess.box(comp, A)      # the solid parts: glow, sparks, a thin shadow left out
    b_ = C.solid_box(comp, B) or sess.box(comp, B)
    r_ = {"left-of": "left", "right-of": "right", "next-to": "next", "above": "above", "below": "below"}.get(rel_)
    if r_ and slot[1] == 1 and depth == 0:
        LB0 = pic.layer(B)
        back = comp.role(LB0) == "backdrop"
        big = (back or b_[2] - b_[0] > 1 / 3, back or b_[3] - b_[1] > 1 / 3)
        if C._holds(r_, a_, b_, pic.index(A), pic.index(B), back, big, C.solid(comp, B)[1]):
            say(f"put: {A} is already {rel_.replace('-', ' ')} {B}; nothing changed")
            return
    aw, ah = a_[2] - a_[0], a_[3] - a_[1]
    acx = (a_[0] + a_[2]) / 2
    bcx, bcy = (b_[0] + b_[2]) / 2, (b_[1] + b_[3]) / 2
    LA = pic.layer(A)
    tx, ty1 = acx, a_[3]                                 # target bottom centre of A
    landed, tall = None, False
    if rel_ == "on":
        bh = b_[3] - b_[1]
        surface = comp.role(pic.layer(B)) == "backdrop" or (b_[2] - b_[0]) >= 0.45
        tx = acx if b_[0] + 0.5 * aw <= acx <= b_[2] - 0.5 * aw else bcx
        if surface:                                      # a table top, a floor, a field: stand inside it
            inside = b_[1] + 0.02 * bh <= a_[3] <= b_[3]
            ty1 = a_[3] if inside else b_[1] + 0.25 * bh
            landed = f"on the surface of {B}" + (" (its height kept)" if inside else " (a quarter down its top)")
        else:                                            # an object: sit on its top edge, overlapping it a little
            ty1 = b_[1] + 0.06 * bh
            landed = f"on the top edge of {B}"
        if not LA.get("shadow"):
            LA["shadow"] = "ground"
    elif rel_ in ("above", "below"):
        tall = comp.role(pic.layer(B)) == "backdrop" or b_[3] - b_[1] > 1 / 3
        bh, my = b_[3] - b_[1], C.solid(comp, B)[1][1]
        if tall and rel_ == "above":                     # a forest, a wall: clear of its solid top when that fits,
            cy_t = min(b_[1] + bh / 3, my) - gap         # else higher than its mass and inside its top third
            if b_[1] - gap - ah >= 0.01:
                cy_t = b_[1] - gap - ah / 2
            tx, ty1 = acx, cy_t + ah / 2
        elif tall:
            cy_t = max(b_[3] - bh / 3, my) + gap
            if b_[3] + gap + ah <= 0.99:
                cy_t = b_[3] + gap + ah / 2
            tx, ty1 = acx, cy_t + ah / 2
        elif rel_ == "above":
            tx, ty1 = bcx, b_[1] - gap
        else:
            tx, ty1 = bcx, b_[3] + gap + ah
    elif rel_ in ("left-of", "right-of", "next-to"):
        side = -1 if rel_ == "left-of" else 1
        tx = (b_[0] - gap - aw / 2) if side < 0 else (b_[2] + gap + aw / 2)
        ty1 = b_[3]
    elif rel_ == "in":
        bw, bh = b_[2] - b_[0], b_[3] - b_[1]
        if aw > 0.8 * bw or ah > 0.8 * bh:
            _fit(sess, comp, A, b_[0] + 0.15 * bw, b_[1] + 0.15 * bh, b_[2] - 0.15 * bw, b_[3] - 0.15 * bh)
            comp = sess.geometry()
            a_ = C.solid_box(comp, A) or sess.box(comp, A)
            aw, ah = a_[2] - a_[0], a_[3] - a_[1]
            acx = (a_[0] + a_[2]) / 2
        tx, ty1 = bcx, b_[1] + 0.4 * bh                 # seated in the opening, rising above the rim
    i, n = slot
    if n > 1:                                            # several layers: side by side across B
        bw, bh = b_[2] - b_[0], b_[3] - b_[1]
        if rel_ == "in":                                 # side by side across the opening; a heap when they
            span = bw * 0.75                             # do not fit: the next row sits on and between them
            per = max(1, min(n, int(span / max(aw * 0.85, 1e-6))))
            r, c = divmod(i, per)
            k = min(per, n - r * per)
            step = span / per
            tx = bcx + (c - (k - 1) / 2) * step
            ty1 -= r * 0.55 * ah
        elif rel_ == "on":
            span = bw * 0.7
            tx = bcx - span / 2 + span * (i + 0.5) / n
        else:
            tx += (i - (n - 1) / 2) * aw * 1.1
    if rel_ in ("left-of", "right-of", "next-to", "above", "below") and depth < 2:
        fb = sess.box(comp, A)                           # the whole layer (its shadow, its glow) stays in the frame
        ex = (max(0.0, a_[0] - fb[0]), max(0.0, fb[2] - a_[2]), max(0.0, a_[1] - fb[1]), max(0.0, fb[3] - a_[3]))
        nx0, nx1, ny0, ny1 = tx - aw / 2 - ex[0], tx + aw / 2 + ex[1], ty1 - ah - ex[2], ty1 + ex[3]
        m = 0.01
        if nx0 < m or nx1 > 1 - m or ny0 < m or ny1 > 1 - m:     # no room on that side: never off the frame
            s = (m - nx0) if nx0 < m else ((1 - m) - nx1) if nx1 > 1 - m else 0.0
            LB = pic.layer(B)
            movable = comp.role(LB) not in ("backdrop", "pass") and (b_[2] - b_[0]) < 0.45 and \
                b_[0] + s >= 0 and b_[2] + s <= 1 and ny0 >= m and ny1 <= 1 - m
            if s and movable:
                _shift(sess, LB, s, 0.0)
                tx += s
                say(f"put: no room {rel_.replace('-', ' ')} {B} inside the frame: {A} and {B} both moved by "
                    f"{s:+.2f}")
            else:
                if rel_ == "left-of":
                    room = b_[0] - gap - m - ex[0]
                elif rel_ in ("right-of", "next-to"):
                    room = (1 - m) - b_[2] - gap - ex[1]
                else:
                    room = None
                if rel_ in ("above", "below") and tall:        # the centre stays put; the half height must fit
                    f = (min(ty1 - ah / 2 - m, (1 - m) - (ty1 - ah / 2))) / max(ah / 2, 1e-6)
                elif rel_ == "above":
                    f = (b_[1] - gap - m - ex[2]) / max(ah, 1e-6)
                elif rel_ == "below":
                    f = ((1 - m) - b_[3] - gap - ex[3]) / max(ah, 1e-6)
                else:
                    f = room / max(aw, 1e-6) if room is not None else 1.0
                if 0.8 <= f < 1.0:
                    LA["scale"] = round(float(LA.get("scale", 1.0)) * f, 4)
                    say(f"put: no room {rel_.replace('-', ' ')} {B} inside the frame: {A} scaled {f:.2f} to fit")
                    return _put_one(sess, A, rel_, B, gap, slot, depth + 1)
                cx_ = min(max(tx, m + aw / 2), 1 - m - aw / 2)          # never worse: nothing changes
                cy_ = min(max(ty1 - ah / 2, m + ah / 2), 1 - m - ah / 2)
                why = (f"only shrunk to {max(f, 0):.2f} of its size" if f < 1.0 else "only partly outside the frame")
                say(f"put: {A} fits {rel_.replace('-', ' ')} {B} {why}; nothing changed - draw it smaller or place it "
                    f"in its section, or move {A} --to {cx_:.2f} {cy_:.2f}")
                return
    _shift(sess, LA, tx - ((a_[0] + a_[2]) / 2), ty1 - a_[3])
    if rel_ in ("on", "in") and pic.index(A) < pic.index(B):
        _reorder(pic, A, "above", B)
    if landed:
        say(f"put: {A} stands {landed}, its base at x {tx:.2f} y {ty1:.2f}")


# ---- crop

def cmd_crop(sess, a):
    args = Args("crop", a, {"--with": 1})
    pic = sess.pic
    sc = pic.data["scene"]
    if args.pos[:1] == ["off"]:
        sc.pop("crop", None)
        sess.commit("crop off", force_rev=True)
        return 0
    if not args.pos:
        c = sc.get("crop")
        raise StudioError("crop needs a region: crop X0 Y0 X1 Y1 [--with title] (or crop off)"
                          + (f"; now: {' '.join(f'{v:.2f}' for v in c['box'])}" if c else ""))
    x0, y0, x1, y1 = _numbers(args.pos, 4, "crop X0 Y0 X1 Y1")
    if not (x1 > x0 and y1 > y0):
        raise StudioError("crop X0 Y0 X1 Y1: x1 right of x0, y1 below y0 (fractions of the picture as it is now)")
    old = sc.get("crop")
    if old:                                              # a region of the cropped view: back to the full frame
        ox, oy, ox1, _ = old["box"]
        f = ox1 - ox
        x0, y0, x1, y1 = ox + x0 * f, oy + y0 * f, ox + x1 * f, oy + y1 * f
    W, H = pic.data["size"]
    box = crop_region(x0, y0, x1, y1)
    asked = (x0, y0, x1, y1)
    fmt = lambda b: " ".join(f"{v:.2f}" for v in b)      # noqa: E731
    if max(abs(u - v) for u, v in zip(box, asked)) > 0.005:
        say(f"crop: widened to the frame's shape: {fmt(box)} (asked {fmt(asked)})")
    if box[2] - box[0] >= 0.9:                           # the crop would show nearly everything: name one that bites
        side = max(0.02, min(x1 - x0, y1 - y0))
        tight = crop_region((x0 + x1) / 2 - side / 2, (y0 + y1) / 2 - side / 2, (x0 + x1) / 2 + side / 2,
                            (y0 + y1) / 2 + side / 2)
        say(f"crop: {fmt(box)} is nearly the whole frame, so this crop changes almost nothing; one that does: "
            f"crop {fmt(tight)} (the frame's shape around the same centre, as tight as the asked box's smaller side)")
    with_ = _expand(pic, args.opt["--with"].replace(",", " ").split(), "crop") if args.opt.get("--with") else []
    sc["crop"] = {"box": box, "with": with_}
    f = 1.0 / (box[2] - box[0])
    kept = [L["name"] for L in pic.data["layers"] if text_like(L, sess.cells.get(L.get("cell") or L["name"]))
            and L["name"] not in with_]
    say(f"crop: the picture shows x {box[0]:.2f}-{box[2]:.2f}, y {box[1]:.2f}-{box[3]:.2f} of the full frame, "
        f"{f:.2f}x, every layer drawn again at that size" + (f"; {', '.join(kept)} keep their place (--with "
                                                              f"{kept[0]} crops it too)" if kept else ""))
    sess.commit(f"crop {box[0]:.2f} {box[1]:.2f} {box[2]:.2f} {box[3]:.2f}" + (f" --with {','.join(with_)}"
                                                                              if with_ else ""), force_rev=True)
    return 0


# ---- needs and notes

def cmd_need(sess, a):
    args = Args("need", a, {"--layer": 1, "--why": 1, "--count": 1, "--color": 1})
    pic = sess.pic
    needs = pic.data["needs"]
    sub = args.pos[0] if args.pos else "list"
    sub = _choice("need", sub, ("list", "add", "ok", "reopen", "rm", "set"))
    if sub == "list":
        out = sess.render("need", quiet=True)
        ok, total = out.score
        say(f"request: {ok} of {total} confirmed (rev {pic.data.get('rev')})")
        for n in out.needs:
            say(f"  {'[x]' if n['ok'] else '[ ]'} {n['id']} {n['text']}" + ("" if n["ok"] else
                (f" -> {n['reason']}" if n.get('reason') else "") + (f": {_cmd_short(pic, n['cmd'])}" if n.get("cmd")
                                                                      else ""))
                + ("  (added)" if n.get("src") == "agent" else ""))
        first = next((n for n in out.needs if not n["ok"] and n.get("cmd")), None)
        if first is not None:
            say("NEXT: " + _cmd_text(pic, first["cmd"], me()))
        elif ok < total:
            say(f"NEXT: {me()} need ok N --why \"what you saw\" for an item you checked by eye")
        else:
            say(f"NEXT: every item confirmed: {finish_text(pic, me())}")
        return 0
    ln = args.opt.get("--layer")
    if ln:
        _expand(pic, [ln], "need")
    if sub == "add":
        text = " ".join(args.pos[1:]).strip()
        if not text:
            raise StudioError('need add needs the item: need add "the lantern glows" [--layer lantern]')
        nid = max([n["id"] for n in needs] + [int(pic.data.get("max_id") or 0), 0]) + 1
        pic.data["max_id"] = nid
        thing = not ln and re.fullmatch(r"[A-Za-z][A-Za-z\- ]{0,30}", text) and len(text.split()) <= 3
        n = {"id": nid, "kind": "thing" if thing else "eye", "key": text.lower() if thing else text, "text": text,
             "src": "agent", "layer": ln, "ok": None, "why": None, "count": None, "colours": []}
        m = re.fullmatch(r"(?:the |a )?([a-z][a-z ]*?) (in front of|to the left of|to the right of|on top of|next to|left of|right of|beside|behind|above|below|under|on|in|at) "
                         r"(?:the |a )?([a-z][a-z ]*)", text.lower())
        if m and not ln:                                 # "mug right of bowl": a place the check measures
            n.update(kind="place", key=[m.group(1), m.group(2), m.group(3)],
                     text=f"{m.group(1)} {m.group(2)} {m.group(3)}")
        _need_opts(n, args)
        needs.append(n)
        label = f"need add {nid}"
    else:
        if len(args.pos) < 2:
            raise StudioError(f"need {sub} needs the item number: need {sub} 2")
        rest = args.pos[1:]
        if sub == "set":
            ids, rest = [int(_num(rest[0], "item number"))], rest[1:]
        else:
            ids = [int(_num(x, "item number")) for x in _split(rest)]
        for nid in ids:
            if not any(x["id"] == nid for x in needs):
                raise StudioError(f"no item {nid} (items: {', '.join(str(x['id']) for x in needs) or 'none'})")
        for n in [x for x in needs if x["id"] in ids]:
            if sub == "ok":
                n["ok"] = True
                n["why"] = args.opt.get("--why") or (" ".join(rest) if len(ids) == 1 else None) or None
            elif sub == "reopen":
                n["ok"] = None
                n["why"] = None
            elif sub == "set":
                text = " ".join(rest).strip()
                if not (text or ln or args.opt.get("--count") or args.opt.get("--color")):
                    raise StudioError(f'need set {n["id"]} needs the new text or an option: need set {n["id"]} '
                                      '"soap: green" [--layer L] [--count K] [--color C]')
                if text:
                    if n["kind"] == "text":
                        n["key"] = text.strip("'\"")
                        n["text"] = f"text '{n['key']}'" + (f" on the {n['on']}" if n.get("on") else "")
                    elif n["kind"] == "thing" and n.get("cell"):      # a section's line, written anew
                        new = thing_item(n["cell"], text if inventory(text, n["cell"]) else f"{text}:")
                        n.update({k: v for k, v in new.items() if k not in ("id", "line", "src", "cell")})
                    elif n["kind"] == "thing":
                        n["key"] = text.split(":")[0].strip().lower()
                        n["text"] = text
                        n.pop("alias", None)
                    else:
                        n["kind"], n["key"], n["text"] = "eye", text, text
                if ln:
                    n["layer"] = ln
                _need_opts(n, args)
                n["ok"] = None
        if sub == "rm":
            for x in needs:
                if x["id"] in ids and x.get("src") == "draft" and x.get("cell"):
                    pic.data.setdefault("dismissed", {})[x["cell"]] = x.get("line", "")
            pic.data["needs"] = [x for x in needs if x["id"] not in ids]
        label = f"need {sub} {' '.join(map(str, ids))}"
    sess.commit(label, force_rev=True, content=False)
    return 0


def _need_opts(n, args):
    """--count K / --color C on a thing item: its text says them too."""
    if args.opt.get("--count") is None and not args.opt.get("--color"):
        return
    if n["kind"] != "thing":
        raise StudioError("--count and --color belong to a thing item (a word or two: need set N \"soap\")")
    if args.opt.get("--count") is not None:
        k = int(_num(args.opt["--count"], "--count"))
        n["count"] = k if k > 1 else None
    if args.opt.get("--color"):
        n["colours"] = [str(args.opt["--color"]).strip().lower()]
    bits = ([str(n["count"])] if n.get("count") else []) + list(n.get("colours") or [])
    n["text"] = n["key"] + (": " + ", ".join(bits) if bits else "")


def _note_item(pic, text):
    """A note that starts with an item number ('3 glass fragments too small') is about that item: (id, its
    first layer) or (None, None)."""
    m = re.match(r"^\s*(?:item\s*)?#?(\d+)\b", text, re.I)
    if not m:
        return None, None
    nid = int(m.group(1))
    n = next((x for x in pic.data.get("needs") or [] if x.get("id") == nid), None)
    if n is None:
        return None, None
    lay = None
    if n.get("cell"):
        fam = C.family(pic, n["cell"])
        lay = fam[0] if fam else None
    elif n.get("layer"):
        lay = n["layer"]
    elif n.get("kind") == "thing":
        got = C.layers_for(pic, None, n["key"])
        lay = got[0] if got else None
    return nid, lay


def _note_layer(pic, text):
    """The layer a note is about: the first layer name or label written in it, else None."""
    low = text.lower().replace("_", " ")
    best = None
    for L in pic.data["layers"]:
        for w in {L["name"], L.get("label") or "", base_name(L["name"])}:
            w = str(w).lower().replace("_", " ").strip()
            m = re.search(r"(?<![a-z0-9])" + re.escape(w) + r"s?(?![a-z0-9])", low) if w else None
            if m and (best is None or (m.start(), -len(w)) < best[0]):
                best = ((m.start(), -len(w)), L["name"])
    return best[1] if best else None


def cmd_note(sess, a):
    args = Args("note", a, {"--layer": 1})
    pic = sess.pic
    d = pic.data
    notes = d["notes"]
    sub = args.pos[0] if args.pos else "list"
    sub = _choice("note", sub, ("list", "add", "none", "done", "reopen"))
    if sub == "list":
        for n in notes:
            say(f"  {'(x)' if n.get('done') else '( )'} {n['id']} {n['text']}"
                + (f"  [{n['layer']}]" if n.get("layer") else "") + (f"  (review {n['review']})" if n.get("review")
                                                                       else ""))
        if not notes:
            say("no notes (note add \"3 the spoon is lost on the table\" \"the title touches the moon\")")
        return 0
    if sub in ("add", "none"):
        texts = [t.strip() for t in args.pos[1:] if t.strip()]
        ids = []
        if sub == "add":
            if not texts:
                raise StudioError('note add needs the text: note add "the fox needs a rim light" "the cup floats"')
            if len(texts) > 1 and all(" " not in t for t in texts):     # unquoted words: one note
                texts = [" ".join(texts)]
            lay = args.opt.get("--layer")
            if lay:
                pic.layer(lay)
            for t in texts:
                nid = max([n["id"] for n in notes] or [0]) + 1
                item, ilay = _note_item(pic, t)
                notes.append({"id": nid, "text": t, "done": False, "layer": lay or ilay or _note_layer(pic, t),
                              "item": item})
                ids.append(nid)
        elif texts:
            raise StudioError("note none takes no text: it says the picture shows every line")
        else:                                            # harmless: the report says what is next
            sess.render("note none")
            return 0
        label = f"note {sub}" + (" " + " ".join(map(str, ids)) if ids else "")
        sess.commit(label, force_rev=True, content=False)
        return 0
    if len(args.pos) < 2:
        raise StudioError(f"note {sub} needs the note numbers: note {sub} 1 2")
    ids = [int(_num(x, "note number")) for x in _split(args.pos[1:])]
    for nid in ids:
        if not any(x["id"] == nid for x in notes):
            raise StudioError(f"no note {nid} (notes: {', '.join(str(x['id']) for x in notes) or 'none'})")
    for n in notes:
        if n["id"] in ids:
            n["done"] = sub == "done"
    sess.commit(f"note {sub} {' '.join(map(str, ids))}", force_rev=True, content=False)
    return 0


# ---- history

def _goto(sess, rev, label):
    pic = sess.pic
    sess.cells = pic.restore(rev)
    pic.save()
    sess.synced = ([], [], [])
    sess.render(label)
    return 0


def _pending(sess):
    """Cell files edited since the last revision become a revision first (undo never loses them)."""
    added, changed, removed = sess.synced
    if added or changed or removed:
        sess.commit("render", render=False)


def cmd_undo(sess, a):
    args = Args("undo", a, {})
    n = int(_num(args.pos[0], "undo steps")) if args.pos else 1
    _pending(sess)
    pic = sess.pic
    rev = int(pic.data.get("rev") or 0)
    target = rev
    for _ in range(max(1, n)):
        p = pic.parent(target)
        if not p:
            break
        target = p
    if target == rev or target == 0:
        raise StudioError(f"nothing to undo (rev {rev} is the first)")
    return _goto(sess, target, f"undo -> rev {target} (from {rev})")


def cmd_redo(sess, a):
    pic = sess.pic
    _pending(sess)
    rev = int(pic.data.get("rev") or 0)
    kids = pic.children(rev)
    if not kids:
        raise StudioError(f"nothing to redo after rev {rev}")
    return _goto(sess, max(kids), f"redo -> rev {max(kids)}")


def cmd_goto(sess, a):
    if not a:
        raise StudioError("goto needs a revision number (history lists them)")
    _pending(sess)
    rev = int(_num(a[0], "revision"))
    sess.pic.load_rev(rev)
    return _goto(sess, rev, f"goto rev {rev}")


def cmd_history(sess, a):
    args = Args("history", a, {"--all": 0})
    pic = sess.pic
    cur = int(pic.data.get("rev") or 0)
    revs = pic.revs()
    shown = revs if args.opt.get("--all") else revs[-25:]
    cut = f" (last 25 of {len(revs)}; history --all)" if len(shown) < len(revs) else ""
    say(f"history of {pic.name} (* = now; score = confirmed of requested){cut}:")
    for r in shown:
        sc = f"{r['score'][0]}/{r['score'][1]}" if r.get("score") else "-"
        err = f" {r['errors']} error" if r.get("errors") else ""
        say(f" {'*' if r['rev'] == cur else ' '} rev {r['rev']:>3}  from {r.get('parent', 0):>3}  {sc:>5}{err}  {r['cmd']}")
    say(f"NEXT: {me()} goto REV | undo | redo | compare REV")
    return 0


# ---- look / compare

def _font(px):
    from PIL import ImageFont
    try:
        from ..text import find_font
        path = str(find_font(None) or "").partition("#")[0]
        if path:
            return ImageFont.truetype(path, max(8, int(px)))
    except Exception:                                    # noqa: BLE001
        pass
    return ImageFont.load_default()


def _label(d, xy, text, fnt):
    """A grid number on a dark backing: readable on light and dark pictures."""
    tb = d.textbbox(xy, text, font=fnt)
    d.rectangle([tb[0] - 2, tb[1] - 1, tb[2] + 2, tb[3] + 1], fill=(0, 0, 0, 170))
    d.text(xy, text, fill=(255, 235, 0, 255), font=fnt)


def _grid(im, x0=0.0, y0=0.0, x1=1.0, y1=1.0, step=None, labels=True):
    """Draw a fraction grid (of the whole picture) on an image showing the region x0..x1, y0..y1."""
    from PIL import ImageDraw
    d = ImageDraw.Draw(im, "RGBA")
    fnt = _font(max(11, im.height // 50))
    W, H = im.size
    span = max(x1 - x0, y1 - y0)
    step = step or (0.1 if span > 0.5 else 0.05 if span > 0.2 else 0.02)
    k = math.ceil(x0 / step - 1e-9)
    while k * step <= x1 + 1e-9:
        v = k * step
        px = (v - x0) / (x1 - x0) * W
        d.line([(px, 0), (px, H)], fill=(255, 255, 255, 70), width=1)
        if labels:
            _label(d, (px + 2, 2), f"{v:.2f}".lstrip("0") if v < 1 else "1", fnt)
        k += 1
    k = math.ceil(y0 / step - 1e-9)
    while k * step <= y1 + 1e-9:
        v = k * step
        py = (v - y0) / (y1 - y0) * H
        d.line([(0, py), (W, py)], fill=(255, 255, 255, 70), width=1)
        if labels:
            _label(d, (2, py + 2), f"{v:.2f}".lstrip("0") if v < 1 else "1", fnt)
        k += 1
    return im


def _alone(pl, cx0, cy0, cx1, cy1, H):
    """One layer alone on a neutral checker: the canvas region cx0..cx1, cy0..cy1 (pixels, may leave the frame)."""
    from PIL import Image
    h, w = max(1, cy1 - cy0), max(1, cx1 - cx0)
    yy, xx = np.mgrid[cy0:cy0 + h, cx0:cx0 + w]
    q = max(8, H // 90)
    chk = (((yy // q) + (xx // q)) % 2).astype(np.float32)
    ground = np.repeat((0.62 + 0.08 * chk)[..., None], 3, axis=2)
    P = np.zeros((h, w, 3), np.float32)
    A = np.zeros((h, w), np.float32)
    hh, ww = pl.A.shape
    ys0, xs0 = max(0, cy0 - pl.y0), max(0, cx0 - pl.x0)
    ys1, xs1 = min(hh, cy0 + h - pl.y0), min(ww, cx0 + w - pl.x0)
    if ys1 > ys0 and xs1 > xs0:
        oy, ox = pl.y0 - cy0, pl.x0 - cx0
        P[ys0 + oy:ys1 + oy, xs0 + ox:xs1 + ox] = pl.P[ys0:ys1, xs0:xs1]
        A[ys0 + oy:ys1 + oy, xs0 + ox:xs1 + ox] = pl.A[ys0:ys1, xs0:xs1]
    img = P + ground * (1.0 - A[..., None])
    return Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))


def _tinted(comp, name, pl):
    """The placed layer with its recolor / opacity applied (what the picture shows)."""
    import types
    from .engine import body
    try:
        P, A, x0, y0 = body(pl, comp.params(comp.by_name[name]), comp.H)
    except Exception:                                    # noqa: BLE001
        return pl
    return types.SimpleNamespace(P=P, A=A, x0=x0, y0=y0)


def _boxes_on(img, pic, comp, f):
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    fnt = _font(max(12, img.height // 40))
    for L in pic.data["layers"]:
        pl = comp.placed.get(L["name"])
        if pl is None or pl.box is None or comp.role(L) in ("backdrop", "pass") or not L.get("visible", True):
            continue
        x0, y0, x1, y1 = (v * f for v in pl.box)
        d.rectangle([x0, y0, x1, y1], outline=(0, 230, 255), width=2)
        tb = d.textbbox((x0 + 3, y0 + 2), L["name"], font=fnt)
        d.rectangle([tb[0] - 2, tb[1] - 1, tb[2] + 2, tb[3] + 1], fill=(0, 0, 0))
        d.text((x0 + 3, y0 + 2), L["name"], fill=(0, 230, 255), font=fnt)
    return img


def sheet_layers(pic, comp):
    """Object layers for the look sheet: requested things first, then the rest front to back."""
    objs = [L["name"] for L in pic.data["layers"] if L.get("visible", True) and comp.role(L) not in ("backdrop", "pass")
            and comp.placed.get(L["name"]) is not None and comp.placed[L["name"]].box is not None]
    want = []
    for n in pic.data.get("needs", []):
        if n.get("kind") == "thing":
            for x in C.layers_for(pic, comp, n["key"]):
                if x in objs and x not in want:
                    want.append(x)
    return want + [x for x in reversed(objs) if x not in want]


def item_tiles(pic, comp, needs=None):
    """[(label, pixel box or None)]: each checklist item where the picture shows it (its layers' boxes)."""
    out = []
    for n in (needs if needs is not None else pic.data.get("needs") or []):
        if n.get("cell"):
            ls = C.family(pic, n["cell"])
        elif n.get("layer"):
            ls = [n["layer"]]
        elif n.get("kind") == "text":
            t = C._text_layer(pic, comp, n["key"])
            ls = [t] if t else []
        elif n.get("kind") in ("thing", "place"):
            key = n["key"][0] if isinstance(n["key"], list) else n["key"]
            ls = C.layers_for(pic, comp, key)
        else:
            ls = []
        bs = [comp.placed[x].box for x in ls if x in pic.names() and pic.layer(x).get("visible", True)
              and comp.placed.get(x) is not None and comp.placed[x].box is not None]
        label = f"{n['id']} {n.get('name') or n.get('key') if n.get('kind') == 'thing' else n['text']}"
        if not bs:
            out.append((label[:40], None))
            continue
        out.append((label[:40], (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs),
                                 max(b[3] for b in bs))))
    return out


def write_sheet(sess, comp, names=None, picture=True, path=None, out=None, items=None):
    """look.png: the picture with a 0.1 grid and each layer's box, above a sheet of tiles: for a review (``out``
    or ``items``) every checklist item close up IN the picture, labelled with its number, in the list's order;
    with ``names`` those layers alone on a neutral ground. -> (path, tile labels, left out)."""
    from PIL import Image, ImageDraw
    pic = sess.pic
    W, H = comp.W, comp.H
    path = path or os.path.join(pic.folder, "look.png")
    if out is not None and items is None:
        items = item_tiles(pic, comp, out.needs)
    if items is not None:
        tiles, more = items[:ITEM_TILES], items[ITEM_TILES:]
        return _item_sheet(pic, comp, tiles, more, path, picture)
    order = sheet_layers(pic, comp) if names is None else list(names)
    tiles, more = order[:SHEET_TILES], order[SHEET_TILES:]
    SW, gap, lab = PREVIEW_MAX, 8, 20
    cols = max(1, min(len(tiles), 5))
    rows = math.ceil(len(tiles) / cols) if tiles else 0
    tw = (SW - gap * (cols + 1)) // cols
    th = min(tw, 230)
    tiles_h = rows * (th + lab + gap) + (gap if rows else 0)
    img = None
    if picture:
        f = min(SW / W, (PREVIEW_MAX - tiles_h) / H)
        img = shown_pil(comp)
        img = img.resize((max(1, round(W * f)), max(1, round(H * f))), Image.LANCZOS)
        img = _boxes_on(_grid(img), pic, comp, f)
    if not tiles:
        sheet = img if img is not None else Image.new("RGB", (SW, 64), (34, 34, 38))
        sheet.save(path)
        return path, tiles, more
    top = img.height if img is not None else 0
    sheet = Image.new("RGB", (SW, top + tiles_h), (34, 34, 38))
    if img is not None:
        sheet.paste(img, ((SW - img.width) // 2, 0))
    d = ImageDraw.Draw(sheet)
    fnt = _font(13)
    for i, n in enumerate(tiles):
        r, c = divmod(i, cols)
        tx, ty = gap + c * (tw + gap), top + gap + r * (th + lab + gap)
        d.rectangle([tx, ty, tx + tw - 1, ty + th - 1], fill=(52, 52, 58))
        pl = comp.placed.get(n)
        if pl is None or pl.box is None:
            d.text((tx + 6, ty + th // 2), f"{n}: draws nothing", fill=(255, 120, 120), font=fnt)
            continue
        x0, y0, x1, y1 = (int(round(v)) for v in pl.box)
        pad = int(0.08 * max(x1 - x0, y1 - y0)) + 4
        al = _alone(_tinted(comp, n, pl), x0 - pad, y0 - pad, x1 + pad, y1 + pad, H)
        k = min(tw / al.width, th / al.height, 6.0)
        al = al.resize((max(1, int(al.width * k)), max(1, int(al.height * k))), Image.LANCZOS)
        sheet.paste(al, (tx + (tw - al.width) // 2, ty + (th - al.height) // 2))
        d.text((tx + 2, ty + th + 3), f"{n}  x {x0 / W:.2f}-{x1 / W:.2f} y {y0 / H:.2f}-{y1 / H:.2f}",
               fill=(0, 230, 255), font=fnt)
    sheet.save(path)
    return path, tiles, more


def _item_sheet(pic, comp, tiles, more, path, picture=True):
    """The review sheet: the picture, then one crop of the picture per item (its boxes and a margin, enlarged)."""
    from PIL import Image, ImageDraw
    W, H = comp.W, comp.H
    SW, gap, lab = PREVIEW_MAX, 8, 20
    cols = max(1, min(len(tiles), 4))
    rows = math.ceil(len(tiles) / cols) if tiles else 0
    tw = (SW - gap * (cols + 1)) // cols
    th = min(tw, 220)
    tiles_h = rows * (th + lab + gap) + (gap if rows else 0)
    full = shown_pil(comp)
    img = None
    if picture:
        f = min(SW / W, max(200, PREVIEW_MAX - tiles_h) / H)
        img = full.resize((max(1, round(W * f)), max(1, round(H * f))), Image.LANCZOS)
        img = _boxes_on(_grid(img), pic, comp, f)
    top = img.height if img is not None else 0
    sheet = Image.new("RGB", (SW, top + tiles_h), (34, 34, 38))
    if img is not None:
        sheet.paste(img, ((SW - img.width) // 2, 0))
    d = ImageDraw.Draw(sheet)
    fnt = _font(14)
    for i, (label, box) in enumerate(tiles):
        r, c = divmod(i, cols)
        tx, ty = gap + c * (tw + gap), top + gap + r * (th + lab + gap)
        d.rectangle([tx, ty, tx + tw - 1, ty + th - 1], fill=(52, 52, 58))
        if box is None:
            d.text((tx + 6, ty + th // 2), f"{label}: not in the picture", fill=(255, 120, 120), font=fnt)
        else:
            x0, y0, x1, y1 = box
            bw, bh = x1 - x0, y1 - y0
            px, py = 0.15 * bw + 0.03 * H, 0.25 * bh + 0.03 * H
            cx0, cy0 = max(0, int(x0 - px)), max(0, int(y0 - py))
            cx1, cy1 = min(W, int(x1 + px)), min(H, int(y1 + py))
            if cx1 - cx0 < 4 or cy1 - cy0 < 4:
                d.text((tx + 6, ty + th // 2), f"{label}: off the frame", fill=(255, 120, 120), font=fnt)
            else:
                crop = full.crop((cx0, cy0, cx1, cy1))
                k = min(tw / crop.width, th / crop.height, 8.0)
                crop = crop.resize((max(1, int(crop.width * k)), max(1, int(crop.height * k))),
                                   Image.LANCZOS if k < 2 else Image.BICUBIC)
                sheet.paste(crop, (tx + (tw - crop.width) // 2, ty + (th - crop.height) // 2))
        d.text((tx + 2, ty + th + 3), label, fill=(255, 235, 0), font=fnt)
    sheet.save(path)
    return path, [t[0] for t in tiles], [t[0] for t in more]


def cmd_look(sess, a):
    args = Args("look", a, {"--zoom": 4, "--text": 0, "--out": 1})
    from PIL import Image, ImageDraw
    pic = sess.pic
    out = sess.render("look", quiet=True)
    comp = sess.comp
    cv = comp.canvas
    W, H = comp.W, comp.H
    full = cv.to_pil()
    path = args.opt.get("--out") or os.path.join(pic.folder, "look.png")
    if args.pos:
        args.pos = _expand(pic, args.pos, "look")
    name = args.pos[0] if len(args.pos) == 1 else None
    if args.opt.get("--text"):
        names = list(args.pos) or pic.names()
        if pic.data.get("prompt") and not args.pos:
            say(f'the request: "{pic.data["prompt"][:300]}"')
        for n in names:
            L = pic.layer(n)
            say(_layer_line(sess, comp, out, L, max(len(x) for x in names))[0])
            pl = comp.placed.get(n)
            if pl is not None and pl.box is not None and comp.role(L) not in ("pass",):
                from .engine import body
                P_, A_, _, _ = body(pl, comp.params(L), comp.H)
                a8 = A_ > 0.5
                if a8.any():
                    col = (P_[a8] / np.maximum(A_[a8][:, None], 1e-6)).mean(0)
                    from .. import manifest as M
                    hx = "".join(f"{int(round(min(max(c, 0), 1) * 255)):02x}" for c in col)
                    say(f"    mean colour #{hx} ({M.colour_name(col)}); params: "
                        + ", ".join(f"{k}={v}" for k, v in L.items() if k in LAYER_DEFAULTS and v not in (None, False)
                                    and v != LAYER_DEFAULTS.get(k)))
                an = comp.anchors_of(n)
                own = [k for k in an if k not in ("center", "top", "bottom", "left", "right", "top_left", "top_right",
                                                  "bottom_left", "bottom_right")]
                if own:
                    say("    anchors: " + ", ".join(f"{k} ({an[k][0]:.2f}, {an[k][1]:.2f})" for k in own[:10]))
            if L.get("desc"):
                say(f"    cell: {L['desc']}")
        return 0
    if args.opt.get("--zoom"):
        x0, y0, x1, y1 = args.nums("--zoom")
        if not (x1 > x0 and y1 > y0):
            raise StudioError("--zoom X0 Y0 X1 Y1: x1 right of x0, y1 below y0 (fractions)")
        crop = full.crop((int(x0 * W), int(y0 * H), max(int(x0 * W) + 1, int(x1 * W)), max(int(y0 * H) + 1, int(y1 * H))))
        f = min(PREVIEW_MAX / crop.width, PREVIEW_MAX / crop.height)
        crop = crop.resize((max(1, int(crop.width * f)), max(1, int(crop.height * f))),
                           Image.NEAREST if f > 2 else Image.LANCZOS)
        img = _grid(crop.convert("RGB"), x0, y0, x1, y1)
        img.save(path)
        say(f"look: {ab(path)} = region x {x0:g}-{x1:g}, y {y0:g}-{y1:g} at {f:.1f}x, grid in picture fractions")
        return 0
    if name:
        pl = comp.placed.get(name)
        if comp.role(pic.layer(name)) == "pass":
            raise StudioError(f"{name} is a pass: look at the whole picture (look) or hide it to compare")
        if pl is None or pl.box is None:
            raise StudioError(f"{name} draws nothing" + (f": {comp.info[name]['error']['short']}"
                                                          if (comp.info.get(name) or {}).get("error") else ""))
        x0, y0, x1, y1 = pl.box
        pad = 0.15 * max(x1 - x0, y1 - y0) + 0.02 * H
        cx0, cy0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
        cx1, cy1 = min(W, int(x1 + pad)), min(H, int(y1 + pad))
        ctx = full.crop((cx0, cy0, cx1, cy1))
        a_im = _alone(_tinted(comp, name, pl), cx0, cy0, cx1, cy1, H)
        dctx = ImageDraw.Draw(ctx)
        dctx.rectangle([x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0], outline=(255, 220, 0), width=max(1, H // 540))
        side = Image.new("RGB", (a_im.width * 2 + 12, a_im.height), (30, 30, 30))
        side.paste(a_im, (0, 0))
        side.paste(ctx, (a_im.width + 12, 0))
        f = min(1.0, PREVIEW_MAX / side.width, PREVIEW_MAX / side.height)
        if f < 1:
            side = side.resize((max(1, int(side.width * f)), max(1, int(side.height * f))), Image.LANCZOS)
        if f > 0 and side.width < 640:
            g = 640 / side.width
            side = side.resize((int(side.width * g), int(side.height * g)), Image.LANCZOS)
        side.save(path)
        say(f"look: {ab(path)} = {name} alone (left) and in the picture (right, yellow box x {x0 / W:.2f}-{x1 / W:.2f}"
            f" y {y0 / H:.2f}-{y1 / H:.2f})")
        return 0
    if len(args.pos) > 1:
        for n in args.pos:
            if comp.role(pic.layer(n)) == "pass":
                raise StudioError(f"{n} is a pass (it changes the whole picture): look shows it in the picture")
        path, tiles, more = write_sheet(sess, comp, names=args.pos, picture=False, path=path)
        say(f"look: {ab(path)} = {len(tiles)} layers alone, each with its box: {', '.join(tiles)}")
        return 0
    if out.needs:
        path, tiles, more = write_sheet(sess, comp, path=path, out=out)
        say(f"look: {ab(path)} = the picture (0.1 grid, each layer's box) above {len(tiles)} items close up in "
            f"the picture: {', '.join(tiles)}")
        if more:
            say(f"  more items: {', '.join(more)} (look NAME shows one layer alone)")
        return 0
    path, tiles, more = write_sheet(sess, comp, path=path)
    say(f"look: {ab(path)} = the picture (0.1 grid, each layer's box) above {len(tiles)} layers alone"
        + (f": {', '.join(tiles)}" if tiles else ""))
    if more:
        say(f"  more layers: {', '.join(more)} (look NAME shows one)")
    return 0


def _render_state(sess, data, cells, size=None):
    comp = Compositor(data, cells, size=size, cache=sess.cache, paths=sess._paths(), folder=sess.pic.folder)
    comp.run()
    return comp


def cmd_compare(sess, a):
    from PIL import Image, ImageDraw
    args = Args("compare", a, {"--out": 1})
    pic = sess.pic
    if not args.pos:
        raise StudioError("compare needs a revision: compare 4 (with now) or compare 4 7")
    revs = [int(_num(x, "revision")) for x in args.pos[:2]]
    ims = []
    for r in revs:
        data, cells = pic.load_rev(r)
        ims.append((f"rev {r}", _render_state(sess, data, cells).canvas.to_pil()))
    if len(revs) == 1:
        ims.append((f"now (rev {pic.data.get('rev')})", _render_state(sess, pic.data, sess.cells).canvas.to_pil()))
    w = PREVIEW_MAX // 2
    tiles = []
    for lab, im in ims:
        im = im.resize((w, max(1, round(im.height * w / im.width))), Image.LANCZOS)
        dd = ImageDraw.Draw(im)
        fnt = _font(max(12, im.height // 22))
        tb = dd.textbbox((6, 4), lab, font=fnt)
        dd.rectangle([0, 0, tb[2] + 6, tb[3] + 4], fill=(0, 0, 0))
        dd.text((6, 4), lab, fill=(255, 255, 0), font=fnt)
        tiles.append(im)
    side = Image.new("RGB", (w * 2 + 8, max(t.height for t in tiles)), (30, 30, 30))
    side.paste(tiles[0], (0, 0))
    side.paste(tiles[1], (w + 8, 0))
    path = args.opt.get("--out") or os.path.join(pic.folder, "compare.png")
    side.save(path)
    say(f"compare: {rel(path)} = {ims[0][0]} (left) | {ims[1][0]} (right)")
    return 0


# ---- stages

def cmd_stages(sess, a):
    """One strip of the medium stage: flat silhouettes -> relief clay -> lit colour -> the medium."""
    args = Args("stages", a, {"--out": 1})
    from PIL import Image, ImageDraw
    from . import medium_stage as MS
    from ..medium import stages, canonical, MEDIA
    from ..medium.media import font as _font
    pic = sess.pic
    sess.render("stages", quiet=True)
    comp = sess.comp
    med, dials = MS.setting(pic.data["scene"])
    if args.pos:
        med = _choice("stages", canonical(args.pos[0]), MEDIA)
    med = med or "render"
    t0 = time.time()
    b, tb = MS.buffers(comp, pic)
    tiles = stages(b, med, **dials)
    tm = time.time() - t0 - tb
    H, W = tiles[0][1].shape[:2]
    th = 360 if W >= H else 480
    tw = max(1, round(W * th / H))
    bar, gap = 24, 8
    strip = Image.new("RGB", (len(tiles) * (tw + gap) + gap, th + bar + 2 * gap), (34, 34, 38))
    dr = ImageDraw.Draw(strip)
    for i, (lab, img) in enumerate(tiles):
        im = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).resize((tw, th), Image.LANCZOS)
        x = gap + i * (tw + gap)
        strip.paste(im, (x, gap + bar))
        dr.text((x + 4, gap + 2), f"{i + 1}. {lab}", fill=(235, 235, 235), font=_font(16))
    path = args.opt.get("--out") or os.path.join(pic.folder, "stages.png")
    strip.save(path)
    say(f"stages: {ab(path)} (flat silhouettes -> relief clay -> lit colour -> {med}; buffers {tb:.1f} s, "
        f"stages {tm:.1f} s)")
    if b.get("estimated"):
        say("  " + "; ".join(f"{x}: volume estimated" for x in b["estimated"][:8]))
    if b.get("recorded"):
        say(f"  recorded: {', '.join(b['recorded'][:12])}")
    say(f"NEXT: open {ab(path)}; the relief clay shows the shapes the medium paints: a flat or wrong shape is fixed "
        f"in its section (Sketch.paint volume=, or a form solid)")
    return 0


# ---- deliver / export

def cmd_deliver(sess, a):
    args = Args("deliver", a, {"--scale": 1, "--layers": 0})
    pic = sess.pic
    if not args.pos:
        if not pic.data.get("out"):
            raise StudioError(f"deliver needs the file to write: deliver ./{pic.name}.png")
    path = args.pos[0] if args.pos else pic.data["out"]
    if not os.path.splitext(path)[1]:
        path = os.path.join(path, f"{pic.name}.png")
    scale = _num(args.opt.get("--scale", 1.0), "--scale")
    W, H = (int(round(v * scale)) for v in pic.data["size"])
    _pending(sess)
    from .. import layers as _layers
    stack = None
    if args.opt.get("--layers"):
        sc = pic.data["scene"]
        finish = [k for k in ("grade", "paper") if sc.get(k)]
        data = json.loads(json.dumps(pic.data))
        for k in finish:                                 # object layers stay exact: the finish goes on top
            data["scene"][k] = None if k == "grade" else False
        _layers.enable(target_size=(W, H))
        comp = Compositor(data, sess.cells, size=(W, H), cache=sess.cache, paths=sess._paths(),
                          folder=pic.folder, log=_layers.mark)
        try:
            comp.run()
        finally:
            stack = _layers.take(comp.canvas)
        if finish:
            raw = np.clip(np.asarray(comp.canvas.buf, np.float32), 0, 1)
            comp = Compositor(pic.data, sess.cells, size=(W, H), cache=sess.cache, paths=sess._paths(),
                              folder=pic.folder)
            comp.run()
            a_, col = _layers._extract(raw, np.clip(np.asarray(comp.canvas.buf, np.float32), 0, 1))
            label = " + ".join(f"grade {sc['grade']}" if k == "grade" else "paper" for k in finish)
            stack.append(_layers._to_layer(f"finish ({label})", a_, col, (W, H), "adjust"))
    else:
        comp = Compositor(pic.data, sess.cells, size=(W, H), cache=sess.cache, paths=sess._paths(), folder=sess.pic.folder)
        comp.run()
    errs = [(n, v["error"], bool(v.get("stale"))) for n, v in comp.info.items() if v.get("error")]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    from . import medium_stage as MS
    med, dials = MS.setting(pic.data["scene"])
    if med and not errs and not args.opt.get("--layers"):
        comp.medium_img = MS.apply(comp, pic, med, dials, max_side=min(max(W, H), 1600))[0]
    shown_pil(comp).save(path)
    said = [rel(path)]
    if args.opt.get("--layers"):
        stack, _ = _layers.reconcile(stack, comp.canvas)
        base = os.path.splitext(path)[0]
        written = _layers.save_all(stack, base, comp.canvas.buf, formats=("psd", "ora"))
        said += [rel(p) for p in written]
    pic.data.setdefault("delivered", []).append({"path": os.path.abspath(path), "rev": pic.data.get("rev"),
                                                 "size": [W, H]})
    pic.save()
    say(f"delivered: {', '.join(said)} ({W}x{H}, rev {pic.data.get('rev')})")
    for ln in still_open(pic, errs):
        say(ln)
    say(f"NEXT: open {ab(path)} (this is what the user gets), then tell the user that path")
    return 0


def still_open(pic, errs=()):
    """Two or three lines: what the delivered picture still leaves open (errors, items, notes)."""
    d = pic.data
    out = []
    if errs:
        out.append("still open: " + "; ".join(
            f"{n} fails ({e.get('short', '').split(' -> studio api')[0][:60]}), "
            + ("its last good render is in the picture" if stale else "left out") for n, e, stale in errs[:2]))
    items = d.get("open_items") or []
    notes = [n for n in d.get("notes", []) if not n.get("done")]
    bits = []
    if items:
        bits.append("request items " + "; ".join(items[:4]))
    if notes:
        bits.append("notes " + "; ".join(f"{n['id']} {n['text'][:50]}" for n in notes[:3]))
    if bits:
        out.append(("still open: " + " | ".join(bits))[:220])
    if not out:
        out.append("nothing open")
    return out[:3]


def export_source(pic, cells):
    d = json.loads(json.dumps(pic.data))
    keep = {"name", "prompt", "size", "kind", "seed", "scene", "layers", "program"}
    d = {k: v for k, v in d.items() if k in keep}
    for L in d["layers"]:
        for k in ("good", "src", "desc", "auto_role", "shown"):
            L.pop(k, None)
    used = {L.get("cell") or L["name"] for L in d["layers"]}
    cells = {c: s for c, s in cells.items() if c in used}
    W, H = d["size"]
    lines = ['"""' + (d["prompt"] or d["name"]).replace('"""', "'''") + f"\n\nExported from pixelcanvas studio "
             f"({d['name']}, rev {pic.data.get('rev')}): render with tools/render.py like any scene."
             + '\n"""',
             "from pixelcanvas.studio import render_picture", "",
             f"PROMPT = {d['prompt']!r}", f"KIND = {d['kind']!r}", f"NAME = {d['name']!r}",
             f"DEFAULT_SIZE = ({W}, {H})", f"PICTURE = {d!r}", "CELLS = {"]
    for c, s in cells.items():
        lines.append(f"    {c!r}: {s!r},")
    lines += ["}", "", "",
              f"def render(width={W}, height={H}, seed={int(d.get('seed', 1))}, log=lambda step: None):",
              "    pic = dict(PICTURE, seed=seed)",
              "    return render_picture(pic, CELLS, width, height, log=log)", ""]
    return "\n".join(lines)


def cmd_export(sess, a):
    pic = sess.pic
    if not a:
        raise StudioError(f"export needs the scene file to write: export ./{pic.name}_scene.py")
    path = a[0] if a[0].endswith(".py") else a[0] + ".py"
    _pending(sess)
    atomic_write(path, export_source(pic, sess.cells))
    say(f"exported: {rel(path)} (a scene file: python <root>/tools/render.py {rel(path)} --check)")
    return 0


# ============================================================================ main

VERBS = ("new", "render", "status", "add", "move", "scale", "rotate", "flip", "fit", "order", "hide", "show", "rm",
         "copy", "set", "recolor", "put", "look", "need", "note", "undo", "redo", "history", "goto", "compare",
         "deliver", "export", "list", "help", "names", "api", "crop", "batch", "stages")

USAGE = """pixelcanvas studio: a picture as layers you edit step by step (./art/NAME/)

  new NAME --prompt "REQUEST" [--size WxH] [--out PATH.png] --from DRAFT.py   the picture and its first draft
                                   in one command (without --from: where to write the draft, and its format)
  add --from DRAFT.py              the draft again: changed sections are drawn again, the rest from the cache
  add NAME --text "Night Walk" [--box X0 Y0 X1 Y1 --color fff4d6 --font serif]
  add NAME --kit "creature('fox', pose='walk')"   (or: --kit creature fox pose=walk)
  add NAME --scene FILE.py         a whole chart / diagram / UI scene as one backdrop layer
  render | status [--all] [--fresh]   pick up edited layers/<name>.py, render what changed, report (--all: every
                                   layer line; --fresh: run every cell again)
  move L [L2..] --to X Y | --by DX DY      scale L F      rotate L DEG      flip L      fit L X0 Y0 X1 Y1
  order L top|bottom|up|down|above X|below X      hide L / show L / rm L
  copy L --total N [--row X0 Y0 X1 Y1 | --grid X0 Y0 X1 Y1 --cols C]   (L2, L3 ...)   copy L NEW
  set L key=value ...   (opacity outline shadow=ground|drop glow tint blur outline_width role label ...)
  set scene key=value   (light=top-right grade=golden paper=on background=0e1433 medium=oil kind size prompt
                         reviews=3 cell_timeout=60)
  recolor L COLOUR                 put A on|above|below|left-of|right-of|in|next-to B
  stages [MEDIUM] [--out PATH.png]   one strip: flat silhouettes, relief clay, lit colour, the medium
  look | look L | look L1 L2 .. | look --zoom X0 Y0 X1 Y1 | look --text [L]   one image (or numbers) to check
  crop X0 Y0 X1 Y1 [--with L] | crop off       show that region enlarged to the frame, every layer drawn sharp
  batch "CMD" "CMD" ..             several commands, one report at the end
  need | need add "TEXT" [--layer L] | need ok N [--why ".."] | need reopen N | need rm N [N ..]
  need set N "TEXT" [--layer L] [--count K] [--color C]     the checklist is yours to correct
  note add ".." [".." ..] [--layer L] | note done N [N ..] | note reopen N | note   what a look found
  undo [N] | redo | history | goto REV | compare REV [REV]
  deliver PATH.png [--scale 2] [--layers]          export NAME.py (a scene file for render.py)
  names                            the names a cell sees
  api [NAME ..]                    the arguments of a call (sk.paint, creature, elements.steam); a word lists calls
Several layers: recolor a b c COLOUR, scale fish* 1.2 (NAME* = that cell and its copies).
Options anywhere: -p NAME (picture; default: the current one), --dir DIR (default ./art)."""


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
    argv = list(sys.argv[1:] if argv is None else argv)
    name, adir_opt, rest = None, None, []
    i = 0
    while i < len(argv):
        t = argv[i]
        if t in ("-p", "--picture") and i + 1 < len(argv):
            name = argv[i + 1]
            i += 2
            continue
        if t.startswith("--picture="):
            name = t.split("=", 1)[1]
        elif t == "--dir" and i + 1 < len(argv):
            adir_opt = argv[i + 1]
            i += 2
            continue
        elif t.startswith("--dir="):
            adir_opt = t.split("=", 1)[1]
        else:
            rest.append(t)
        i += 1
    if not rest or rest[0] in ("help", "-h", "--help"):
        print(USAGE)
        return 0
    verb, args = rest[0], rest[1:]
    adir = art_dir(adir_opt)
    try:
        if verb not in VERBS:
            near = difflib.get_close_matches(verb, VERBS, n=2, cutoff=0.5)
            raise StudioError(f"unknown command {verb!r}" + (f" - did you mean {' or '.join(near)}?" if near else "")
                              + f" (commands: {', '.join(v for v in VERBS if v != 'status')})")
        if verb == "new":
            return cmd_new(args, adir)
        if verb == "names":
            say("a cell sees: " + ", ".join(NAMESPACE) + " and everything of `from pixelcanvas import *`")
            say('box("fox") -> (x0, y0, x1, y1) and anchor("fox", "mouth") -> (x, y): fractions, where that layer '
                'is now; scene("chart.py") -> the whole picture of a scene file')
            say("the arguments of any call: api NAME (e.g. api sk.paint creature)")
            return 0
        if verb == "api":
            from . import api as _api
            return _api.run(args)
        if verb == "list":
            pics = list_pictures(adir)
            say(f"pictures in {rel(adir)}: {', '.join(pics) or 'none'}")
            return 0
        pic = open_picture(adir, name)
        sess = Session(pic)
        sess.verb = verb
        sess.sync()
        fn = {"render": lambda s, a: _render_cmd(s, a), "status": lambda s, a: _render_cmd(s, a, "status"),
              "add": cmd_add, "move": cmd_move, "scale": cmd_scale, "rotate": cmd_rotate, "flip": cmd_flip,
              "fit": cmd_fit, "order": cmd_order, "hide": lambda s, a: cmd_visible(s, a, "hide"),
              "show": lambda s, a: cmd_visible(s, a, "show"), "rm": cmd_rm, "copy": cmd_copy, "set": cmd_set,
              "recolor": cmd_recolor, "put": cmd_put, "look": cmd_look, "need": cmd_need, "note": cmd_note,
              "undo": cmd_undo, "redo": cmd_redo, "history": cmd_history, "goto": cmd_goto,
              "compare": cmd_compare, "deliver": cmd_deliver, "export": cmd_export, "crop": cmd_crop,
              "stages": cmd_stages}
        if verb == "batch":
            code = cmd_batch(sess, args, fn)
        else:
            code = fn[verb](sess, args) or 0
        return 1 if getattr(sess, "report_failed", False) else code
    except StudioError as e:
        say(f"ERROR {e}")
        if e.fix:
            say(f"NEXT: {me()} {e.fix}")
        else:
            say(f"NEXT: fix the command and run it again ({me()} help lists every command)")
        return 2
    except Exception as e:                               # noqa: BLE001 - never a bare traceback
        import traceback
        tb = traceback.extract_tb(e.__traceback__)
        where = f" ({os.path.basename(tb[-1].filename)} line {tb[-1].lineno})" if tb else ""
        say(f"command failed: {type(e).__name__}: {str(e)[:200]}{where}")
        say(f"NEXT: {me()} status" + (f" -p {name}" if name else ""))
        return 1


def cmd_batch(sess, a, fns):
    """batch "CMD" "CMD" ...: each command makes its revision; one report at the end."""
    if not a:
        raise StudioError('batch needs commands, each in quotes: batch "recolor fish* f5a142" "scale fish* 1.2"')
    cmds = []
    for c in a:
        toks = shlex.split(c, posix=True)
        while toks and toks[0] not in VERBS and len(toks) > 1:     # "studio ...", "python .../studio.py ..."
            toks = toks[1:]
        if not toks or toks[0] not in fns:
            near = difflib.get_close_matches(toks[0] if toks else "", list(fns), n=1, cutoff=0.5)
            raise StudioError(f"batch: {c!r} is not a studio command" + (f" - did you mean {near[0]}?" if near else
                                                                          ""))
        cmds.append((c, toks))
    for i, (c, toks) in enumerate(cmds):
        sess.mute = i < len(cmds) - 1
        sess.sync()
        try:
            fns[toks[0]](sess, toks[1:])
        except StudioError as e:
            sess.mute = False
            say(f"batch: {i} of {len(cmds)} done; command {i + 1} failed: {c}")
            say(f"ERROR {e}")
            if i:
                sess.sync()
                sess.render(f"batch: {i} of {len(cmds)}")
            else:
                say(f"NEXT: fix that command and run the batch again ({me()} help lists every command)")
            return 2
    sess.mute = False
    return 0


def _render_cmd(sess, a, verb="render"):
    args = Args(verb, a, {"--all": 0, "--fresh": 0})
    if args.opt.get("--fresh"):
        sess.cache.mem.clear()
        for L in sess.pic.data["layers"]:
            L.pop("good", None)
        import shutil
        shutil.rmtree(sess.pic.cache_dir, ignore_errors=True)
        sess.cache = Cache(sess.pic.cache_dir)
    sess.commit("render", full=bool(args.opt.get("--all")))
    return 0
