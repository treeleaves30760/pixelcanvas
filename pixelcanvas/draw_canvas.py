"""The working canvas of a free-mode project: marks are laid on a canvas 3x the delivered picture.

A 2 px mark (the finest a brush makes cleanly) keeps 8 % of the finest octave (features 1-2 px) at 1x, 68 % at 2x
and 99 % at 3x (measured on professional pictures): professional pictures are drawn 2-4x larger
than they are delivered. Coordinates are 0-1000 on both axes, so no command changes; every size in the draw tools is
a fraction of the canvas or a delivered-pixel size times ``draw_core.work_scale``.

- ``ensure(p)``: the first drawing command of a free project (no reference, nothing painted yet) turns the canvas
  into the working canvas: ``state["work_scale"]`` = 3 (2 when 3x would pass ~13 Mpx), ``state["deliver"]`` = the
  size asked for with ``pixelcanvas new --size``, width / height = scale x that. Part of the setup (no undo step).
- ``WRAP["export"]``: after the export writes its files at working size, the picture and the stage pictures are
  reduced to the delivered size with an area filter (the anti-aliased downsample the finest band needs).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image

from .draw_core import work_scale

SCALE = 3
MAX_WORK_PX = 13.0e6
MIN_SIDE = 640          # a thumbnail-size canvas (tests, quick trials) stays at 1x


def wanted_scale(W: int, H: int) -> int:
    if max(W, H) < MIN_SIDE:
        return 1
    for s in (SCALE, 2):
        if W * H * s * s <= MAX_WORK_PX:
            return s
    return 1


def fresh(p) -> bool:
    """A free project with nothing drawn yet (only the paper layer)."""
    st = p.state
    if st.get("has_ref") or st.get("reference") or st.get("mode") == "hand" or st.get("work_scale"):
        return False
    if st.get("style") == "poster":
        return False            # flat graphic shapes and lettering: no fine octave to keep
    for e in st.get("layers", []):
        if e["name"] != "paper" and not e.get("empty", True):
            return False
    d = st.get("draw", {})
    return not d.get("lines") and not d.get("sketch") and not d.get("fills") and not d.get("bands")


def ensure(p) -> List[str]:
    """Make a fresh free project's canvas the working canvas (see module doc). Returns report lines."""
    if not fresh(p):
        return []
    W, H = p.W, p.H
    s = wanted_scale(W, H)
    if s <= 1:
        p.state["work_scale"] = 1
        p.save_state()
        return []
    # part of the project's setup, like  pixelcanvas new  (no undo step): the first step's snapshot already holds the
    # working canvas, so undo and the stage pictures / time-lapse replay see one canvas size throughout
    p.state["deliver"] = [W, H]
    p.state["work_scale"] = s
    p.state["width"], p.state["height"] = W * s, H * s
    for e in p.state["layers"]:
        arr = p.layer(e["name"], create=False) if e["name"] == "paper" else None
        if arr is not None:
            arr = cv2.resize(arr, (W * s, H * s), interpolation=cv2.INTER_NEAREST)
        p._write_layer_file(e["name"], arr)
    for n in list(p.state.get("regions", {})):
        m = p.region_mask(n)
        p.save_region_mask(n, cv2.resize(m.astype(np.uint8), (W * s, H * s), interpolation=cv2.INTER_NEAREST) > 0)
    p.save_state()
    return [f"working canvas {W * s}x{H * s}: {s}x the {W}x{H} picture you export, so the finest strokes survive; "
            f"coordinates stay 0-1000"]


def deliver_size(p):
    s = work_scale(p)
    dv = p.state.get("deliver")
    if dv:
        return int(dv[0]), int(dv[1])
    return int(round(p.W / s)), int(round(p.H / s))


def reduce_file(path: Path, size) -> bool:
    """Reduce a written picture to ``size`` (w, h) with an area filter; False when it is already that size."""
    try:
        with Image.open(path) as im:
            im.load()
            if im.size == tuple(size) or im.size[0] < size[0]:
                return False
            arr = np.asarray(im.convert("RGB"))
            fmt = im.format
    except OSError:
        return False
    small = cv2.resize(arr, tuple(int(v) for v in size), interpolation=cv2.INTER_AREA)
    if fmt == "JPEG":
        Image.fromarray(small).save(path, quality=92)
    else:
        Image.fromarray(small).save(path)
    return True


def _wrap_export(orig):
    def run(ctx, args):
        rep = orig(ctx, args)
        try:
            p = ctx.project
            p.reload()
            if work_scale(p) <= 1 or getattr(ctx, "exit_code", 0):
                return rep
            dw, dh = deliver_size(p)
            out = p.state.get("exported")
            done = []
            if out and reduce_file(Path(out), (dw, dh)):
                done.append(Path(out).name)
            # stage pictures written by --stages: same size as the canvas
            a = list(args)
            if "--stages" in a and a.index("--stages") + 1 < len(a):
                folder = Path(a[a.index("--stages") + 1])
                if folder.is_dir():
                    for f in sorted(folder.glob("*.png")):
                        if reduce_file(f, (dw, dh)):
                            done.append(f.name)
            if done:
                lines = getattr(rep, "lines", None)
                if isinstance(lines, list):
                    for i, ln in enumerate(lines):
                        if ln and f"{p.W}x{p.H}" in ln:
                            lines[i] = ln.replace(f"{p.W}x{p.H}", f"{dw}x{dh} (drawn at {p.W}x{p.H}, reduced "
                                                                 f"with an area filter)")
                            break
        except Exception:  # noqa: BLE001 - the export itself ran; reducing is a finishing touch
            pass
        return rep
    run.__name__ = "export"
    return run


COMMANDS: dict = {}
USAGE: dict = {}
WRAP = {"export": _wrap_export}
