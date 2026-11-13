#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Search everything the pixelcanvas kits can draw and get the call to paste.

  python tools/kit.py find "a corgi sitting under an oak tree at sunset"     the whole request: a picture starts with the
                                                                             command that starts it freehand, then its
                                                                             subjects with optional ingredients; a chart,
                                                                             diagram, UI ... gets the calls that build it
                                                                             (+ --json, --n 3, --kits: every kit call)
  python tools/kit.py list                                                   kits, one line each
  python tools/kit.py list flora                                             every name in a kit, grouped by kind
  python tools/kit.py show flora.oak                                         snippet, signature, docstring, reference
  python tools/kit.py show plant                                             a function by name (or kit.function)
  python tools/kit.py kinds                                                  kinds of things, with examples
  python tools/kit.py doc flora                                              print the kit's reference page
"""
import argparse
import json
import os
import sys
import textwrap

for _s in (sys.stdout, sys.stderr):              # Windows consoles default to a legacy code page
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pixelcanvas import catalog  # noqa: E402


def _wrap(items, indent="    ", width=100):
    return textwrap.fill(", ".join(items), width=width, initial_indent=indent, subsequent_indent=indent,
                         break_on_hyphens=False)


def cmd_find(a):
    res = catalog.find(" ".join(a.query), n=a.n)
    if a.json:
        print(catalog.to_json(res.as_dict()))
    else:
        print(res.text(kits=a.kits))


def cmd_list(a):
    if a.kit:
        es = catalog.entries(kit=a.kit)
        if not es:
            from pixelcanvas._args import unknown_name
            print(unknown_name("kit", a.kit, [k for k, *_ in catalog.kits()]), file=sys.stderr)
            return 1
        if a.json:
            print(catalog.to_json([e.as_dict() for e in es]))
            return 0
        by = {}
        for e in es:
            by.setdefault(e.kind, []).append(e.name)
        ref = es[0].ref
        print(f"{a.kit}: {len(es)} entries" + (f" · {ref}" if ref else ""))
        for kind, names in by.items():
            print(f"  {kind} ({len(names)})")
            print(_wrap(names))
        print(f"details: python tools/kit.py show {es[0].id}")
        return 0
    rows = catalog.kits()
    if a.json:
        print(catalog.to_json([{"kit": k, "entries": n, "kinds": ks, "ref": ref} for k, n, ks, ref in rows]))
        return 0
    for k, n, ks, ref in rows:
        print(f"{k:<9} {n:>4}  {', '.join(ks)[:62]:<62}  {ref}")
    print("names in a kit: python tools/kit.py list KIT · search: python tools/kit.py find \"what you want\"")
    return 0


def cmd_show(a):
    print(catalog.show(" ".join(a.name)))


def cmd_kinds(a):
    ks = catalog.kinds()
    if a.json:
        print(catalog.to_json({k: [e.id for e in v] for k, v in ks.items()}))
        return
    for kind, es in ks.items():
        kits = sorted({e.kit for e in es})
        ex = ", ".join(e.name for e in es[:5])
        print(f"{kind:<16} {len(es):>4}  {'/'.join(kits):<18} e.g. {ex}"[:118])


def cmd_doc(a):
    try:
        print(catalog.doc(a.kit))
    except KeyError as exc:
        print(exc.args[0])
        return 1
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("find", help="the calls, style and kind for a whole request (one call, <= 30 lines)")
    p.add_argument("query", nargs="+")
    p.add_argument("--n", type=int, default=3, help="alternatives for the main subject (default 3)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--kits", action="store_true", help="the kit answer for a picture too: a call per subject to paste")
    p.set_defaults(fn=cmd_find)
    p = sub.add_parser("list", help="kits, or every name in one kit")
    p.add_argument("kit", nargs="?")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_list)
    p = sub.add_parser("show", help="one entry (kit.name) or function")
    p.add_argument("name", nargs="+")
    p.set_defaults(fn=cmd_show)
    p = sub.add_parser("kinds", help="kinds of things with examples")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_kinds)
    p = sub.add_parser("doc", help="print a kit's reference page")
    p.add_argument("kit")
    p.set_defaults(fn=cmd_doc)
    a = ap.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
