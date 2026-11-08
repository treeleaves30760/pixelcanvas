#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect and re-composite editable layer stacks written by ``render.py SCENE --layers``.

  python tools/layers.py info  outputs/my_scene_layers            list layers (bbox, coverage, kind)
  python tools/layers.py sheet outputs/my_scene_layers            contact sheet: every layer on a checkerboard
                                                                  → outputs/my_scene_layers_sheet.jpg
  python tools/layers.py compose outputs/my_scene_layers --hide post --out no_post.png
                                                                  re-composite after editing / replacing PNGs
                                                                  (also honours "visible"/"opacity"/"x"/"y" in
                                                                  layers.json), --only to keep a subset
  python tools/layers.py export outputs/my_scene_layers --psd edited.psd --ora edited.ora
                                                                  rebuild PSD / ORA from an edited PNG stack
"""
from __future__ import annotations

import argparse
import os
import sys

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from pixelcanvas import layers, vision  # noqa: E402


def _checker(w, h, s=12):
    y, x = np.mgrid[0:h, 0:w]
    c = (((x // s) + (y // s)) % 2).astype(np.float32)
    return (0.55 + 0.2 * c)[..., None].repeat(3, -1)


def cmd_info(a):
    st = layers.load_pngs(a.folder)
    W, H = st[0].size
    print(f"{W}x{H}, {len(st)} layers (bottom → top)")
    for i, L in enumerate(st):
        cov = float((L.rgba[..., 3] > 127).sum()) / (W * H) if L.rgba.size else 0.0
        print(f"{i:2d}. {L.name:<28} kind={L.kind:<6} bbox={tuple(L.bbox)}  opaque area={cov:6.1%}"
              f"{'' if L.visible else '  (hidden)'}")


def cmd_sheet(a):
    st = layers.load_pngs(a.folder)
    W, H = st[0].size
    ims, labels = [], []
    for i, L in enumerate(st):
        bg = _checker(W, H)
        f = L.full()
        img = bg * (1 - f[..., 3:]) + f[..., :3] * f[..., 3:]
        ims.append(vision.to_pil(img))
        labels.append(f"{i}. {L.name} [{L.kind}]")
    ims.append(vision.to_pil(layers.composite(st)))
    labels.append("composite")
    out = a.out or a.folder.rstrip("/\\") + "_sheet.jpg"
    vision.contact_sheet(ims, labels, tile_width=min(480, W), title=f"{os.path.basename(a.folder)}: layers").save(out, quality=88)
    print(out)


def _filtered(a):
    st = layers.load_pngs(a.folder)
    hide = set(a.hide or [])
    only = set(a.only or [])
    for L in st:
        if L.name in hide or (only and L.name not in only):
            L.visible = False
    return st


def cmd_compose(a):
    st = _filtered(a)
    out = a.out or os.path.join(a.folder, "recomposite.png")
    Image.fromarray(np.round(np.clip(layers.composite(st), 0, 1) * 255).astype(np.uint8), "RGB").save(out)
    print(out)


def cmd_export(a):
    st = _filtered(a)
    if a.psd:
        print(layers.save_psd(st, a.psd))
    if a.ora:
        print(layers.save_ora(st, a.ora))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info"); p.add_argument("folder"); p.set_defaults(fn=cmd_info)
    p = sub.add_parser("sheet"); p.add_argument("folder"); p.add_argument("--out"); p.set_defaults(fn=cmd_sheet)
    for name, fn in (("compose", cmd_compose), ("export", cmd_export)):
        p = sub.add_parser(name)
        p.add_argument("folder")
        p.add_argument("--hide", nargs="+", metavar="NAME")
        p.add_argument("--only", nargs="+", metavar="NAME")
        if name == "compose":
            p.add_argument("--out")
        else:
            p.add_argument("--psd"); p.add_argument("--ora")
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
