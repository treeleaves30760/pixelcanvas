#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find fonts, look at them, and install open-licence fonts.

  python tools/fonts.py                       show the fonts pixelcanvas picks for each style word
  python tools/fonts.py georgia               list installed fonts whose file name contains "georgia"
  python tools/fonts.py noto --sheet          + render a specimen sheet → outputs/fonts_noto.jpg (look at it)
  python tools/fonts.py --sheet --text "霓虹之夜 Neon Nights 2026"   specimen of the style defaults with your text
  python tools/fonts.py --catalog             fonts and packs available to install (with licences)
  python tools/fonts.py --install starter     install the starter pack into ~/.pixelcanvas/fonts
  python tools/fonts.py --install oswald playfair-display noto-sans-tc     individual families
  python tools/fonts.py --install all         every family in the catalogue (incl. CJK, ~200 MB)
  python tools/fonts.py --missing "→ ∑ ★ 中"  which characters no installed font can draw

Installed fonts are found automatically (``font="oswald"``, ``font="condensed bold"``, ``font="handwriting"``).
Use a printed stem, path or style words as ``font=`` in ``cv.text(...)`` / ``text_mask(...)`` / ``draw_text(...)``.
Network access happens only in ``--install`` / ``--check-urls``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pixelcanvas import Canvas  # noqa: E402
from pixelcanvas import text as T  # noqa: E402
from pixelcanvas.text import find_font, list_fonts  # noqa: E402

GF = "https://raw.githubusercontent.com/google/fonts/main/"

# name: (style, licence, google-fonts directory, [files]); files are saved as <name>-VF.ttf for variable fonts
CATALOG = {
    # sans
    "inter":            ("sans", "OFL-1.1", "ofl/inter", ["Inter[opsz,wght].ttf", "Inter-Italic[opsz,wght].ttf"]),
    "montserrat":       ("sans geometric", "OFL-1.1", "ofl/montserrat", ["Montserrat[wght].ttf", "Montserrat-Italic[wght].ttf"]),
    "space-grotesk":    ("sans display", "OFL-1.1", "ofl/spacegrotesk", ["SpaceGrotesk[wght].ttf"]),
    # serif
    "eb-garamond":      ("serif (book)", "OFL-1.1", "ofl/ebgaramond", ["EBGaramond[wght].ttf", "EBGaramond-Italic[wght].ttf"]),
    "lora":             ("serif", "OFL-1.1", "ofl/lora", ["Lora[wght].ttf", "Lora-Italic[wght].ttf"]),
    "playfair-display": ("serif display", "OFL-1.1", "ofl/playfairdisplay",
                         ["PlayfairDisplay[wght].ttf", "PlayfairDisplay-Italic[wght].ttf"]),
    "dm-serif-display": ("serif display", "OFL-1.1", "ofl/dmserifdisplay",
                         ["DMSerifDisplay-Regular.ttf", "DMSerifDisplay-Italic.ttf"]),
    "cinzel":           ("serif capitals", "OFL-1.1", "ofl/cinzel", ["Cinzel[wght].ttf"]),
    # slab
    "roboto-slab":      ("slab", "Apache-2.0", "apache/robotoslab", ["RobotoSlab[wght].ttf"]),
    "zilla-slab":       ("slab", "OFL-1.1", "ofl/zillaslab",
                         ["ZillaSlab-Regular.ttf", "ZillaSlab-Bold.ttf", "ZillaSlab-Italic.ttf", "ZillaSlab-Light.ttf"]),
    "arvo":             ("slab", "OFL-1.1", "ofl/arvo", ["Arvo-Regular.ttf", "Arvo-Bold.ttf", "Arvo-Italic.ttf"]),
    # display / condensed
    "bebas-neue":       ("display condensed", "OFL-1.1", "ofl/bebasneue", ["BebasNeue-Regular.ttf"]),
    "anton":            ("display condensed", "OFL-1.1", "ofl/anton", ["Anton-Regular.ttf"]),
    "oswald":           ("condensed", "OFL-1.1", "ofl/oswald", ["Oswald[wght].ttf"]),
    "barlow-condensed": ("condensed", "OFL-1.1", "ofl/barlowcondensed",
                         ["BarlowCondensed-Regular.ttf", "BarlowCondensed-Bold.ttf", "BarlowCondensed-Black.ttf",
                          "BarlowCondensed-Light.ttf", "BarlowCondensed-Italic.ttf"]),
    "archivo-black":    ("display heavy", "OFL-1.1", "ofl/archivoblack", ["ArchivoBlack-Regular.ttf"]),
    "abril-fatface":    ("display serif", "OFL-1.1", "ofl/abrilfatface", ["AbrilFatface-Regular.ttf"]),
    "bungee":           ("display signage", "OFL-1.1", "ofl/bungee", ["Bungee-Regular.ttf"]),
    "monoton":          ("display neon lines", "OFL-1.1", "ofl/monoton", ["Monoton-Regular.ttf"]),
    # rounded
    "nunito":           ("rounded", "OFL-1.1", "ofl/nunito", ["Nunito[wght].ttf", "Nunito-Italic[wght].ttf"]),
    "fredoka":          ("rounded display", "OFL-1.1", "ofl/fredoka", ["Fredoka[wdth,wght].ttf"]),
    # handwriting / script
    "caveat":           ("handwriting", "OFL-1.1", "ofl/caveat", ["Caveat[wght].ttf"]),
    "permanent-marker": ("handwriting marker", "Apache-2.0", "apache/permanentmarker", ["PermanentMarker-Regular.ttf"]),
    "indie-flower":     ("handwriting", "OFL-1.1", "ofl/indieflower", ["IndieFlower-Regular.ttf"]),
    "great-vibes":      ("script", "OFL-1.1", "ofl/greatvibes", ["GreatVibes-Regular.ttf"]),
    "dancing-script":   ("script", "OFL-1.1", "ofl/dancingscript", ["DancingScript[wght].ttf"]),
    "pacifico":         ("script display", "OFL-1.1", "ofl/pacifico", ["Pacifico-Regular.ttf"]),
    "lobster":          ("script display", "OFL-1.1", "ofl/lobster", ["Lobster-Regular.ttf"]),
    # mono
    "jetbrains-mono":   ("mono", "OFL-1.1", "ofl/jetbrainsmono", ["JetBrainsMono[wght].ttf", "JetBrainsMono-Italic[wght].ttf"]),
    "space-mono":       ("mono", "OFL-1.1", "ofl/spacemono", ["SpaceMono-Regular.ttf", "SpaceMono-Bold.ttf", "SpaceMono-Italic.ttf"]),
    # coverage
    "noto-sans":        ("sans, Latin / Greek / Cyrillic coverage", "OFL-1.1", "ofl/notosans",
                         ["NotoSans[wdth,wght].ttf", "NotoSans-Italic[wdth,wght].ttf"]),
    "noto-symbols":     ("symbols, arrows, dingbats", "OFL-1.1", "ofl/notosanssymbols2", ["NotoSansSymbols2-Regular.ttf"]),
    "noto-symbols-1":   ("symbols", "OFL-1.1", "ofl/notosanssymbols", ["NotoSansSymbols[wght].ttf"]),
    "noto-math":        ("maths", "OFL-1.1", "ofl/notosansmath", ["NotoSansMath-Regular.ttf"]),
    "noto-emoji":       ("emoji (monochrome)", "OFL-1.1", "ofl/notoemoji", ["NotoEmoji[wght].ttf"]),
    # CJK
    "noto-sans-tc":     ("CJK sans, Traditional Chinese", "OFL-1.1", "ofl/notosanstc", ["NotoSansTC[wght].ttf"]),
    "noto-sans-sc":     ("CJK sans, Simplified Chinese", "OFL-1.1", "ofl/notosanssc", ["NotoSansSC[wght].ttf"]),
    "noto-sans-jp":     ("CJK sans, Japanese", "OFL-1.1", "ofl/notosansjp", ["NotoSansJP[wght].ttf"]),
    "noto-sans-kr":     ("CJK sans, Korean", "OFL-1.1", "ofl/notosanskr", ["NotoSansKR[wght].ttf"]),
    "noto-serif-tc":    ("CJK serif, Traditional Chinese", "OFL-1.1", "ofl/notoseriftc", ["NotoSerifTC[wght].ttf"]),
    "noto-serif-sc":    ("CJK serif, Simplified Chinese", "OFL-1.1", "ofl/notoserifsc", ["NotoSerifSC[wght].ttf"]),
    "noto-serif-jp":    ("CJK serif, Japanese", "OFL-1.1", "ofl/notoserifjp", ["NotoSerifJP[wght].ttf"]),
    "noto-serif-kr":    ("CJK serif, Korean", "OFL-1.1", "ofl/notoserifkr", ["NotoSerifKR[wght].ttf"]),
    "m-plus-rounded":   ("rounded, Japanese + Latin", "OFL-1.1", "ofl/mplusrounded1c",
                         ["MPLUSRounded1c-Regular.ttf", "MPLUSRounded1c-Bold.ttf"]),
}

PACKS = {
    "starter": ["inter", "playfair-display", "oswald", "bebas-neue", "roboto-slab", "nunito", "caveat", "great-vibes",
                "jetbrains-mono", "noto-symbols", "noto-math"],
    "display": ["bebas-neue", "anton", "archivo-black", "abril-fatface", "bungee", "monoton", "playfair-display",
                "dm-serif-display"],
    "symbols": ["noto-symbols", "noto-symbols-1", "noto-math", "noto-emoji"],
    "noto-cjk": ["noto-sans-tc", "noto-sans-sc", "noto-sans-jp", "noto-sans-kr"],
    "noto-serif-cjk": ["noto-serif-tc", "noto-serif-sc", "noto-serif-jp", "noto-serif-kr"],
    "cjk": ["noto-sans-tc", "noto-sans-sc", "noto-sans-jp", "noto-sans-kr", "noto-serif-tc", "noto-serif-sc",
            "noto-serif-jp", "noto-serif-kr"],
    "latin": [k for k in CATALOG if not k.startswith(("noto-sans-", "noto-serif-")) and k != "m-plus-rounded"],
}
PACKS["all"] = list(CATALOG)
ALIASES = {"playfair": "playfair-display", "bebas": "bebas-neue", "garamond": "eb-garamond", "jetbrains": "jetbrains-mono",
           "symbols2": "noto-symbols", "math": "noto-math", "emoji": "noto-emoji", "noto-cjk-tc": "noto-sans-tc",
           "tc": "noto-sans-tc", "sc": "noto-sans-sc", "jp": "noto-sans-jp", "kr": "noto-sans-kr"}


def local_name(file):
    """'Oswald[wght].ttf' → 'Oswald-VF.ttf' (the name pixelcanvas looks for)."""
    stem, ext = os.path.splitext(file)
    if "[" in stem:
        stem = stem.split("[")[0] + "-VF"
    return stem + ext


def url_for(name, file):
    return GF + CATALOG[name][2] + "/" + urllib.parse.quote(file)


def _expand(names):
    out = []
    for n in names:
        n = ALIASES.get(n.lower(), n.lower())
        if n in PACKS:
            out += PACKS[n]
        elif n in CATALOG:
            out.append(n)
        else:
            raise SystemExit(f"unknown font or pack {n!r}; see `python tools/fonts.py --catalog`")
    return list(dict.fromkeys(out))


def _fetch(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "pixelcanvas-fonts"})
    with urllib.request.urlopen(req, timeout=120) as r, tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(dest)) as f:
        shutil.copyfileobj(r, f)
        tmp = f.name
    os.replace(tmp, dest)
    return os.path.getsize(dest)


def install(names, dest=None, force=False):
    """Download catalogue families into ``dest`` (default ~/.pixelcanvas/fonts). Returns installed paths."""
    dest = dest or T.USER_FONT_DIR
    os.makedirs(dest, exist_ok=True)
    manifest_p = os.path.join(dest, "fonts.json")
    try:
        with open(manifest_p, encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        manifest = {}
    done = []
    for name in _expand(names):
        style, lic, folder, files = CATALOG[name]
        for file in files:
            out = os.path.join(dest, local_name(file))
            if os.path.exists(out) and not force:
                print(f"  ok      {local_name(file)}")
                done.append(out)
                continue
            try:
                n = _fetch(url_for(name, file), out)
                print(f"  fetched {local_name(file):34s} {n / 1e6:6.1f} MB")
                done.append(out)
            except Exception as e:                       # keep going: one failed family should not stop the rest
                print(f"  failed  {file}: {e}")
        lic_file = "LICENSE.txt" if lic.startswith("Apache") else "OFL.txt"
        lic_out = os.path.join(dest, f"{name}-{lic_file}")
        if not os.path.exists(lic_out):
            try:
                _fetch(GF + folder + "/" + lic_file, lic_out)
            except Exception:
                pass
        manifest[name] = dict(style=style, licence=lic, source=GF + folder, files=[local_name(f) for f in files])
    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    T.refresh_fonts()
    print(f"installed into {dest}  ({len(done)} files)")
    return done


def check_urls():
    """HEAD every catalogue URL; returns the list of failures."""
    bad = []
    for name, (_, _, folder, files) in CATALOG.items():
        for file in files:
            u = url_for(name, file)
            try:
                req = urllib.request.Request(u, method="HEAD", headers={"User-Agent": "pixelcanvas-fonts"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    ok = r.status == 200
            except Exception:
                ok = False
            print(("  ok   " if ok else "  FAIL ") + u)
            if not ok:
                bad.append(u)
    return bad


STYLE_ROWS = [("sans", "sans"), ("sans bold", "sans bold"), ("sans black", "sans black"), ("serif", "serif"),
              ("serif italic", "serif italic"), ("slab", "slab"), ("display", "display"), ("condensed", "condensed bold"),
              ("rounded", "rounded"), ("handwriting", "handwriting"), ("script", "script"), ("mono", "mono")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pattern", nargs="?", default=None)
    ap.add_argument("--sheet", action="store_true", help="render a specimen image")
    ap.add_argument("--text", default=None, help="sample text for the sheet")
    ap.add_argument("--max", type=int, default=24)
    ap.add_argument("--install", nargs="+", metavar="NAME", help="font / pack names (see --catalog)")
    ap.add_argument("--dir", default=None, help="install directory (default ~/.pixelcanvas/fonts)")
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    ap.add_argument("--catalog", action="store_true", help="list installable fonts and packs")
    ap.add_argument("--check-urls", action="store_true", help="verify every download URL (network)")
    ap.add_argument("--missing", default=None, help="print the characters of this text no font can draw")
    a = ap.parse_args()

    if a.catalog:
        for name, (style, lic, _, files) in CATALOG.items():
            have = all(os.path.exists(os.path.join(a.dir or T.USER_FONT_DIR, local_name(f))) for f in files)
            print(f"  {'*' if have else ' '} {name:18s} {style:38s} {lic}")
        print("\npacks: " + ", ".join(f"{k} ({len(v)})" for k, v in PACKS.items()))
        print("* = installed.  python tools/fonts.py --install starter")
        return 0
    if a.check_urls:
        return 1 if check_urls() else 0
    if a.install:
        install(a.install, a.dir, a.force)
        return 0
    if a.missing is not None:
        miss = T.missing_glyphs(a.missing)
        print("all characters are covered" if not miss else
              "missing: " + " ".join(f"{c} (U+{ord(c):04X})" for c in miss))
        return 0

    if a.pattern:
        fonts = list_fonts(a.pattern)
        if not fonts:
            print(f"no installed font matches {a.pattern!r}")
            return 1
        for stem, path in fonts[: a.max * 4]:
            print(f"  {stem:32s} {path}")
        rows = [(stem, path) for stem, path in fonts[: a.max]]
    else:
        rows = []
        for label, words in STYLE_ROWS:
            p = find_font(words)
            print(f"  {label:12s} {p or '(none found → Pillow built-in font, Latin only)'}")
            if p:
                rows.append((label, words))
        for label, kw in [("cjk", dict(style="cjk")), ("cjk bold", dict(style="cjk", bold=True)),
                          ("cjk serif", dict(style="cjk-serif"))]:
            p = find_font(**kw)
            print(f"  {label:12s} {p or '(none found: python tools/fonts.py --install noto-sans-tc)'}")
            if p:
                rows.append((label, p))
    if not a.sheet:
        return 0

    sample = a.text or "Sphinx of black quartz 0123 → ★  天光雲影 霓虹之夜"
    row_h, W = 64, 1400
    cv = Canvas(W, row_h * len(rows) + 16, "#f4f1ea")
    for i, (label, font) in enumerate(rows):
        y = 8 + i * row_h
        cv.text(f"{label}", (16, y + row_h / 2), 15, color="#7a6f60", anchor="lm", audit=False)
        bold = "bold" in label or "black" in label
        cv.text(sample, (300, y + row_h / 2), 34, color="#1d1b1a", anchor="lm", font=font,
                bold="black" if "black" in label else bold, italic="italic" in label, clip_ok=True, audit=False)
    from pixelcanvas import cli as workspace
    out = os.path.join(workspace.output_dir(None, ROOT), f"fonts_{a.pattern or 'defaults'}.jpg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv.save(out)
    print("saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
