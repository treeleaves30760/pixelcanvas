#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw a picture from a JSON spec (pcscene): names from the catalogue, placed by words; no Python to write.

  python tools/compose.py --example poster > ./art/poster.json      a complete spec to start from (--kinds lists kinds)
  python tools/compose.py ./art/poster.json --check                  check, draw (draft), report, NEXT edits
  python tools/compose.py ./art/poster.json --check --deliver ./art/final/      full size, copied to ./art/final/
  python tools/compose.py ./art/poster.json --set layers.1.size=0.5 --set style=gouache --check
  python tools/compose.py ./art/poster.json --add '{"draw": "moon", "at": "top right"}' --check
  python tools/compose.py ./art/poster.json --remove moon --check
  python tools/compose.py --names bird                               what can I draw (names for "draw")
  python tools/compose.py --schema                                   the format on one screen

The spec is read forgivingly (comments, trailing commas, single quotes, a Markdown fence, True / None) and every
problem is reported at once with its fix. Results go to outputs/ next to the spec: NAME_x0.5.png and
NAME_x0.5_preview.jpg (draft) or NAME.png (with --deliver or --scale 1). --set / --add / --remove rewrite the
spec file before drawing. The report is the one render.py --check prints, followed by a NEXT block of at most
five spec edits and one line that says whether to deliver.
"""
from __future__ import annotations

import argparse
import contextlib
import difflib
import io
import json
import os
import re
import sys
import time

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from pixelcanvas import spec as S  # noqa: E402
from pixelcanvas import cli as workspace  # noqa: E402


def _rel(p):
    try:
        return os.path.relpath(p).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def _me():
    return f"python {os.path.abspath(__file__)}".replace("\\", "/")


class _Parser(argparse.ArgumentParser):
    """argparse that answers a wrong command line with one line and the command to run instead."""

    def error(self, message):
        opts = [o for a in self._actions for o in a.option_strings if o.startswith("--")]
        hint = ""
        for b in re.findall(r"(--?[\w-]+)", message.split(":", 1)[-1]):
            near = difflib.get_close_matches(b, opts, n=1, cutoff=0.5)
            if near:
                hint = f" (did you mean {near[0]}?)"
                break
        print(f"ERROR compose.py: {message}{hint}")
        _next(f"run: {_me()} ./art/NAME.json --check   (the only options you need are --check and "
              f"--deliver ./art/final/)")
        sys.exit(2)


def _next(action):
    """The last line of every output: exactly one thing to do next."""
    print(f"NEXT: {action}")


def build_parser():
    ap = _Parser(description=__doc__.split("\n")[0], epilog=__doc__,
                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", nargs="?", help="the spec file, e.g. ./art/poster.json")
    ap.add_argument("--scale", type=float, default=None, help="0.5 = draft (default), 1 = full size (default with --deliver)")
    ap.add_argument("--check", action="store_true", help="print the report and the NEXT block (default on)")
    ap.add_argument("--deliver", default=None, metavar="PATH", help="render full size and copy the results to PATH")
    ap.add_argument("--layers", action="store_true", help="also write PSD / ORA / PNG layers")
    ap.add_argument("--svg", action="store_true", help="also write SVG (diagrams, charts, UI, icons)")
    ap.add_argument("--json", action="store_true", help="print one JSON object as the last line")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="PATH=VALUE",
                    help="change one value in the spec file, e.g. layers.1.size=0.5 or style=gouache (repeatable)")
    ap.add_argument("--add", action="append", default=[], metavar="JSON", help="append a layer to the spec file")
    ap.add_argument("--remove", action="append", default=[], metavar="NAME", help="remove a layer (index or name)")
    ap.add_argument("--example", metavar="KIND", help="print a complete example spec for KIND")
    ap.add_argument("--kinds", action="store_true", help="list the kinds")
    ap.add_argument("--schema", action="store_true", help="print the format")
    ap.add_argument("--names", metavar="QUERY", nargs="?", const="", help="names you can draw (matching QUERY)")
    ap.add_argument("--quiet", action="store_true", help="only the summary, problems and NEXT (no report body)")
    return ap


def cmd_info(a):
    if a.kinds:
        for k, v in S.KINDS.items():
            print(f"{k:<11} {v['desc']}")
        _next(f"{_me()} --example KIND > ./art/NAME.json")
        return 0
    if a.schema:
        print(S.schema_text())
        _next(f"{_me()} --example KIND > ./art/NAME.json   (a complete file to fill in)")
        return 0
    if a.example is not None:
        try:
            text = S.dumps(S.example(a.example))
        except KeyError:
            near = difflib.get_close_matches(str(a.example).lower(), list(S.EXAMPLES), n=1, cutoff=0.4)
            print(f"ERROR no kind {a.example!r}" + (f" (did you mean {near[0]!r}?)" if near else "")
                  + f"; kinds: {', '.join(S.EXAMPLES)}", file=sys.stderr)
            print(f"NEXT: {_me()} --example {near[0] if near else 'poster'} > ./art/NAME.json", file=sys.stderr)
            return 2
        print(text, end="")
        # stdout usually goes into the spec file, so the next step goes to stderr
        print(f"NEXT: read the file you saved, write it again with the request's words and things (keep the "
              f"structure), then run: {_me()} ./art/NAME.json --check", file=sys.stderr)
        return 0
    if a.names is not None:
        q = a.names.strip()
        words = [w for w in re.split(r"[^\w' -]+|\s+or\s+", q) if w.strip()] if q else []
        if q.upper() == q and q.lower() in ("word", "name", "words"):      # copied the placeholder
            words = []
            q = ""
        rows = []
        for w in (words if len(words) > 1 else [q] if q else []):
            for r in S.names(w.strip(), n=40 if len(words) <= 1 else 12):
                if r not in rows:
                    rows.append(r)
        if not q:
            rows = S.names()
        if not rows:
            print(f"no name matches {q!r}.")
            _next(f"try a simpler word: {_me()} --names bird   (or tree, cup, cat, person, star)")
            return 1
        if not q:
            by = {}
            for n, summ in rows:
                kit = S._registry()[0][n].kit
                by.setdefault(kit, []).append(n)
            for kit, ns in by.items():
                print(f"{kit} ({len(ns)}): {', '.join(ns)}")
            _next('put one of these names in a layer of your JSON file: {"draw": "NAME"}')
            return 0
        for n, summ in rows:
            print(f"{n:<26} {summ}")
        _next(f'put {{"draw": "{rows[0][0]}"}} (or another name above) in your JSON file, then run --check again')
        return 0
    return None


def _write_spec(path, data):
    text = S.dumps(data)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def apply_edits(path, sets, adds, removes):
    """--set / --add / --remove on the spec file. Returns (lines, errors)."""
    data, repairs = S.load(path)
    out, errs = [], []
    for s in sets:
        k, sep, v = s.partition("=")
        if not sep:
            errs.append(f"--set {s!r}: use PATH=VALUE, e.g. layers.1.size=0.5")
            continue
        try:
            out.append(S.set_value(data, k.strip(), v))
        except (KeyError, ValueError) as e:
            errs.append(f"--set {s!r}: {e.args[0] if e.args else e}")
    for a in adds:
        try:
            out.append(S.add_layer(data, a))
        except (KeyError, ValueError, S.SpecError) as e:
            errs.append(f"--add {a!r}: {e.args[0] if e.args else e}")
    for r in removes:
        try:
            out.append(S.remove_layer(data, r))
        except KeyError as e:
            errs.append(f"--remove {r!r}: {e.args[0] if e.args else e}")
    if out:
        _write_spec(path, data)
        out.append(f"saved {_rel(path)}")
    return out, errs


_PY_HINT = re.compile(r"manifest\.note|cv\.\w+|look\.py|render\.py|new_scene|stylize\.apply|\.py\b|import ")


def _condense(text):
    """Keep the report and the saved paths; drop the report's trailing FIX block (compose prints its own
    PROBLEMS and EDITS from the structured report); drop the per-step timing lines and any advice that is about Python
    scene files (a spec author edits JSON only)."""
    keep = []
    for line in text.splitlines():
        m = re.match(r"\[\s*[\d.]+s\]\s+(.*)$", line)
        if m:
            body = m.group(1)
            if body.startswith(("saved", "preview", "layers", "svg", "        ")) or "psd" in body or "ora" in body:
                keep.append(body.strip())
            continue
        if line.startswith("FIX"):                    # the FIX block (always last): compose prints PROBLEMS / EDITS instead
            break
        line = line.replace("; exact, from cv.manifest", "")
        if line.startswith("preview is ") and "zoom" in line:
            continue
        if "Fix:" in line and _PY_HINT.search(line.split("Fix:", 1)[1]):
            line = line.split("Fix:", 1)[0].rstrip()
        elif _PY_HINT.search(line) and not line.lstrip().startswith(("saved", "preview", "deliver")):
            line = _PY_HINT.sub("...", line)
        keep.append(line)
    return "\n".join(keep)


def _rounds(outdir, name, bump):
    """How many checked drafts of this spec were drawn (kept in outputs/.spec/NAME.rounds)."""
    f = os.path.join(outdir, ".spec", f"{name}.rounds")
    try:
        with open(f, encoding="utf-8") as fh:
            n = int(fh.read().strip() or 0)
    except (OSError, ValueError):
        n = 0
    if bump:
        n += 1
        try:
            with open(f, "w", encoding="utf-8") as fh:
                fh.write(str(n))
        except OSError:
            pass
    return n


def _looks_like_python(path):
    try:
        with open(path, encoding="utf-8-sig", errors="replace") as fh:
            head = fh.read(4000)
    except OSError:
        return False
    return bool(re.search(r"^\s*(import |from \w+ import |def \w+\(|class \w+)", head, re.M)) and "{" not in head[:200]


def run(a):
    path = os.path.abspath(a.spec)
    rec = {"spec": _rel(path)}
    if not os.path.exists(path):
        alt = path + ".json"
        if os.path.exists(alt):
            path = alt
        else:
            print(f"ERROR spec file not found: {a.spec}")
            _next(f"{_me()} --example KIND > {a.spec}   (KIND from: {', '.join(S.EXAMPLES)})")
            return 2, rec
    rel = _rel(path)
    check_cmd = f"{_me()} {rel} --check"
    if os.path.isdir(path):
        print(f"ERROR {a.spec} is a folder; give the JSON file inside it")
        _next(f"{_me()} {a.spec.rstrip('/')}/NAME.json --check")
        return 2, rec
    if path.endswith(".py") or _looks_like_python(path):
        print(f"ERROR {a.spec} is Python. This tool reads one JSON file and draws it; you never write Python.")
        _next(f"{_me()} --example KIND > ./art/NAME.json   then fill that JSON file in and run --check on it")
        return 2, rec
    if a.set or a.add or a.remove:
        try:
            lines, errs = apply_edits(path, a.set, a.add, a.remove)
        except S.SpecError as e:
            print(f"ERROR {e}")
            _next(f"write {rel} again as one JSON object, then run: {check_cmd}")
            return 2, rec
        for ln in lines:
            print(f"edit: {ln}")
        for e in errs:
            print(f"ERROR {e}")
        if errs:
            _next(f"write the change into {rel} yourself (the whole file), then run: {check_cmd}")
            return 2, rec
    try:
        data, repairs = S.load(path)
    except S.SpecError as e:
        print(f"ERROR {e}")
        rec["error"] = str(e)
        _next(f"write {rel} again as one JSON object (copy the shape of: {_me()} --example KIND), "
              f"then run: {check_cmd}")
        return 2, rec
    except (OSError, UnicodeDecodeError) as e:
        print(f"ERROR cannot read {a.spec}: {e}")
        _next(f"write {rel} again with the write tool (UTF-8 text), then run: {check_cmd}")
        return 2, rec
    try:
        norm, problems = S.validate(data, base_dir=os.path.dirname(path))
    except Exception as e:                                    # noqa: BLE001  (a reader bug must not stop the loop)
        print(f"ERROR the file could not be checked ({type(e).__name__}: {str(e)[:160]})")
        _next(f"write {rel} again in the shape of {_me()} --example KIND (same keys, your words), "
              f"then run: {check_cmd}")
        rec["error"] = str(e)
        return 2, rec
    errors = [p for p in problems if p.severity == "error"]
    warns = [p for p in problems if p.severity == "warn"]
    notes = [p for p in problems if p.severity == "note"]
    read = list(repairs) + list((norm or {}).get("notes") or [])
    rec["repairs"] = read
    rec["problems"] = [p.as_dict() for p in problems]
    if read:
        print("READ (repaired or interpreted; write it this way next time):")
        for r in read[:8]:
            print(f"  {r}")
        if len(read) > 8:
            print(f"  ... {len(read) - 8} more")
    if errors:
        print(f"PROBLEMS ({len(errors)}; nothing was drawn):")
        for p in errors:
            print(f"  {p}")
        for p in warns + notes:
            print(f"  ({p.severity}) {p}")
        _next(f"fix every PROBLEMS line in {rel} (write the whole file again), then run: {check_cmd}")
        return 2, rec
    for p in warns + notes:
        print(f"{p.severity}: {p}")
    # render through render.py (the same outputs, check report, layers, svg, deliver)
    import render as R
    name = os.path.splitext(os.path.basename(path))[0]
    outdir = workspace.output_dir(path, ROOT)
    shim_dir = os.path.join(outdir, ".spec")
    os.makedirs(shim_dir, exist_ok=True)
    shim = os.path.join(shim_dir, f"{name}.py")
    with open(shim, "w", encoding="utf-8") as f:
        f.write(S.scene_module_source(path, ROOT))
    scale = a.scale if a.scale is not None else (1.0 if a.deliver else 0.5)
    argv = [shim, "--outdir", outdir, "--scale", str(scale), "--check"]
    if a.deliver:
        argv += ["--deliver", a.deliver]
    if a.layers:
        argv.append("--layers")
    if a.svg or (a.deliver and norm["group"] in ("diagram", "chart", "dashboard", "ui")):
        argv.append("--svg")
    if a.seed is not None:
        argv += ["--seed", str(a.seed)]
    args = R.build_parser().parse_args(argv)
    _build = R.build_report

    def _spec_report(*aa, **kw):                              # advice in the words of a JSON spec, never Python
        kw.setdefault("mode", "spec")
        return _build(*aa, **kw)
    R.build_report = _spec_report
    rrec = {"scene": name, "size": None, "scale": scale, "seed": a.seed, "seconds": None, "outputs": {},
            "steps": [], "warnings": [], "stats": {}}
    buf = io.StringIO()
    t0 = time.time()
    code = 0
    try:
        with contextlib.redirect_stdout(buf):
            try:
                code = R.run(args, rrec)
            finally:
                R.build_report = _build
    except R.UserError as e:
        print(_condense(buf.getvalue()))
        print(f"ERROR {e}")
        _next(f"change {rel} as the ERROR line says (write the whole file), then run: {check_cmd}")
        return 2, rec
    except Exception as e:                                    # noqa: BLE001
        print(_condense(buf.getvalue()))
        info, _text = R.failure_report(e, e.__traceback__, name)
        steps = rrec.get("steps") or []
        last = steps[-1] if steps else None
        where = (last.get("name") or last.get("step")) if isinstance(last, dict) else last
        rec["error"] = info
        png = rrec["outputs"].get("png")
        if png and os.path.exists(png):                      # the picture is there; only the report failed
            print(f"ERROR the picture was drawn ({_rel(png)}) but its check report stopped "
                  f"({type(e).__name__}: {str(e)[:120]})")
            if a.deliver:
                import shutil
                os.makedirs(a.deliver, exist_ok=True)
                dst = os.path.abspath(os.path.join(a.deliver, f"{name}.png")).replace("\\", "/")
                shutil.copyfile(png, dst)
                print(f"DELIVERED: {dst}")
                _next(f"tell the user the picture is at {dst} - the task is done; stop here.")
                return 0, rec
            _next(f"deliver: {_me()} {rel} --deliver ./art/final/")
            return 1, rec
        print(f"ERROR drawing stopped inside pixelcanvas ({type(e).__name__}: {str(e)[:160]})"
              + (f" at the step {where!r}" if where else ""))
        _next(f"remove or simplify that layer in {rel} (or pick another \"draw\" name), then run: {check_cmd}")
        return 1, rec
    secs = time.time() - t0
    out_text = _condense(buf.getvalue())
    rerrs = list(S.LAST.get("errors") or [])
    if a.quiet:
        lines = [ln for ln in out_text.splitlines() if ln.startswith(("saved", "preview", "deliver", "check:"))]
        print("\n".join(lines))
    else:
        print(out_text)
    print(f"drawn in {secs:.1f}s: {_rel(rrec['outputs'].get('preview', ''))}")
    if rerrs:
        print(f"LAYER ERRORS ({len(rerrs)}; these layers were not drawn):")
        for p in rerrs:
            print(f"  {p}")
    for w in S.LAST.get("warnings") or []:
        print(f"note: {w}")
    nxt, n_act = S.next_steps(norm, rrec, rerrs)
    counts = rrec.get("counts") or {}
    n_fail = counts.get("fail", 0) + len(rerrs)
    fixes = [f for f in (rrec.get("fix") or []) if f.startswith(("[fail]", "[warn]")) and not _PY_HINT.search(f)]
    if fixes:
        print(f"PROBLEMS ({counts.get('fail', 0)} fail, {counts.get('warn', 0)} warn):")
        for f in fixes:
            print(f"  {f}")
    else:
        print("PROBLEMS: none")
    rounds = _rounds(outdir, name, bump=not a.deliver)
    deliver_cmd = f"{_me()} {rel} --deliver ./art/final/"
    if nxt:
        print(f"EDITS ({len(nxt)} of {n_act}; make them all in {rel}):")
        for i, t in enumerate(nxt, 1):
            print(f"  {i}. {t}")
    else:
        print("EDITS: none")
    rec.update({"outputs": rrec.get("outputs"), "counts": counts, "next": nxt,
                "layer_errors": [p.as_dict() for p in rerrs], "seconds": round(secs, 2), "kind": norm["kind"],
                "rounds": rounds})
    if a.deliver:
        got = [os.path.abspath(g).replace("\\", "/") for g in (rrec["outputs"].get("delivered") or [])]
        print(f"DELIVERED: {', '.join(got) if got else os.path.abspath(a.deliver)}")
        pic = next((g for g in got if g.lower().endswith(".png")), got[0] if got else os.path.abspath(a.deliver))
        _next(f"tell the user the picture is at {pic} - the task is done; stop here.")
    elif not nxt and n_fail == 0:
        print("STOP: 0 fail and no edit left.")
        _next(f"deliver: {deliver_cmd}")
    elif rounds >= 4:
        print(f"STOP: {rounds} drafts checked; the remaining edits are optional.")
        _next(f"deliver: {deliver_cmd}")
    elif nxt:
        _next(f"make the {len(nxt)} EDITS in {rel} (write the whole file again), then run: {check_cmd}")
    else:
        _next(f"deliver: {deliver_cmd}")
    return code, rec


def main(argv=None):
    a = build_parser().parse_args(argv)
    info = cmd_info(a)
    if info is not None:
        return info
    if not a.spec:
        print('compose.py draws a picture from one JSON file.')
        _next(f"{_me()} --example KIND > ./art/NAME.json   (KIND from: {', '.join(S.EXAMPLES)})")
        return 2
    code, rec = run(a)
    if a.json:
        print(json.dumps(rec, ensure_ascii=False, default=str, separators=(",", ":")))
    return code


if __name__ == "__main__":
    sys.exit(main())
