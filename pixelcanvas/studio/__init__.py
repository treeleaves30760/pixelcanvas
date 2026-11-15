# -*- coding: utf-8 -*-
"""
studio — a picture as a persistent stack of layers that an agent edits step by step.

The picture lives in a folder ``./art/NAME/``: ``picture.json`` (the request, the size, the scene look, the
layers back to front with their place and effects, the request checklist, notes, the revision),
``layers/<name>.py`` (one short cell per layer that draws ONE thing), ``cache/`` (each cell's rendered RGBA),
``history/`` (every revision, cell sources included) and ``preview.png``.

A cell runs alone on its own canvas, so a layer can be moved, scaled, rotated, flipped, re-ordered, outlined,
shadowed, recoloured and hidden without drawing it again; only a cell whose source changed runs again. Every
command applies its change atomically, renders what changed, checks the picture against the request (the
render.py report, one manifest entry per layer) and prints a short report ending in one NEXT line.

  python tools/studio.py new fox --prompt "a fox carrying a lantern through a snowy wood at night"
  python tools/studio.py add --from ./art/fox/draft.py
  python tools/studio.py scale fox 1.2
  python tools/studio.py deliver ./fox.png

``render_picture(picture, cells, width, height)`` renders a picture dict outside the studio (exported scenes).
See skills/pixelcanvas/references/studio.md.
"""
from .compositor import Compositor, render_picture, LAYER_DEFAULTS, SCENE_DEFAULTS
from .engine import Cache, CellResult, NAMESPACE, run_cell, place
from .state import Picture, StudioError, open_picture, request_needs


def main(argv=None):
    """The ``pixelcanvas studio`` command (tools/studio.py)."""
    from .commands import main as _main
    return _main(argv)


__all__ = ["Compositor", "render_picture", "Picture", "StudioError", "open_picture", "request_needs", "Cache",
           "CellResult", "NAMESPACE", "run_cell", "place", "main", "LAYER_DEFAULTS", "SCENE_DEFAULTS"]
