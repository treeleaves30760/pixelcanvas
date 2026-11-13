# -*- coding: utf-8 -*-
"""
pixelcanvas — draw and retouch pictures with pixel-level computation (numpy + Pillow)
=====================================================================================

Three layers of API:

* Low level (``from pixelcanvas import *``): Canvas, noise, masks, brushes, blending, filters, colour, text.
  Anything can be drawn from this layer.
* Freehand (``from pixelcanvas import Sketch, paper, grade``, module ``illustrate``): shapes in canvas fractions,
  one-call shading / outline / shadow / glow, local frames for characters; ``cv.draw()`` is a supersampled Pillow
  ImageDraw layer on the same canvas.
* Kits, loaded on first use (``from pixelcanvas import flora, elements, solid, diagram, chart, drafting, texture,
  vector, layout, stylize, form, figure, creature, clay, pictograms, palette, backdrop, ui, spec, catalog``): one
  call per subject — a tree, a campfire, a 3D mug, a flowchart, a chart, an engineering sheet, a material, a style.
* Landscape recipes (``from pixelcanvas.scenery import *``): Scene + sky / sun / clouds / mountains / mist / river /
  trees / light rays / post-processing.

Every kit records what it drew in ``cv.manifest`` (see ``pixelcanvas.manifest``), so ``render.py --check`` can say
in plain text which objects are in the picture, where, and whether the request was met.

Conventions
-----------
* Coordinates: x right, y down; arrays ``arr[y, x]``; ``cv.xy()`` gives the (X, Y) pixel grids, ``cv.uv()`` 0~1 (U, V).
* Masks: float32 (H, W), 0 = untouched, 1 = fully painted.
* Colours: '#rrggbb' / (r, g, b) 0~1 / (255, 128, 0) / a full (H, W, 3) map; HDR values > 1 are allowed.
* Order: background (gradient) → far to near, each through its mask → post (bloom / soft_clip / contrast /
  vignette) → save.

Minimal example
---------------
>>> from pixelcanvas import *
>>> cv = Canvas(640, 360)
>>> cv.gradient([(0, '#0b1a3a'), (1, '#f6c28f')])          # sky gradient
>>> X, Y = cv.xy()
>>> cv.glow(400, 200, 60, '#fff0c0', strength=1.2)          # glow (HDR)
>>> ridge = 220 - 60 * fbm1d(640, 120, seed=1)              # ridge line: one y per column
>>> cv.paint('#223344', clamp01(Y - ridge[None, :] + 0.5))  # below the ridge = mountain
>>> cv.put(10, 10, '#ff0000')                               # one pixel
>>> cv.soft_clip(0.8).save('demo.png', preview='demo_small.jpg')
"""
import importlib as _importlib

from . import color, noise, masks, brush, filters, text, sdf, vision, layers, manifest
from .color import *       # noqa: F401,F403
from .noise import *       # noqa: F401,F403
from .masks import *       # noqa: F401,F403
from .brush import *       # noqa: F401,F403
from .filters import *     # noqa: F401,F403
from .text import *        # noqa: F401,F403
from .canvas import Canvas
from . import illustrate
from .illustrate import Sketch, paper, grade, limit_palette, MOODS, grow, edge, signed_distance

__version__ = "0.4.0"

# heavy kits: imported on first attribute access (``pixelcanvas.flora`` / ``from pixelcanvas import flora``) and
# never star-imported, so names such as box / union / texture never shadow the 2D helpers above
_LAZY = ("stylize", "scenery", "vector", "diagram", "chart", "drafting", "solid", "texture", "elements", "flora",
         "form", "figure", "creature", "layout", "clay", "pictograms", "palette", "backdrop", "ui", "spec", "catalog",
         "cli", "mcp_server")

__all__ = ["Canvas", "__version__"] + color.__all__ + noise.__all__ + masks.__all__ + brush.__all__ + filters.__all__ \
    + text.__all__ + ["sdf", "vision", "layers", "stylize", "manifest", "illustrate", "Sketch", "paper", "grade",
                      "limit_palette", "MOODS", "grow", "edge", "signed_distance"]


def __getattr__(name):
    if name in _LAZY:
        mod = _importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'pixelcanvas' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
