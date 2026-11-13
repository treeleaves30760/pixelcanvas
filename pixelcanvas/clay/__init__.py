# -*- coding: utf-8 -*-
"""clay: posable 3D characters, animals and props with a clay / designer-toy look (built on pixelcanvas.solid).

One call places a finished figure in a solid Scene:

>>> from pixelcanvas import Canvas
>>> from pixelcanvas.solid import Scene, Camera
>>> from pixelcanvas.clay import cast, animal, Character
>>> sc = Scene(lighting="studio", ground="#e9e2d6")
>>> sc.add(cast("chef", pose="wave").place(x=-0.5), "chef")
>>> sc.add(animal("corgi", pose="sit").place(x=0.6), "corgi")
>>> sc.camera = Camera.frame(sc, azimuth=15, elevation=10)
>>> res = sc.render(Canvas(960, 540))

Characters have a real skeleton (pelvis, spine, chest, neck, head, clavicles, shoulders, elbows, wrists, hips,
knees, ankles, toes) with forward kinematics and joint limits. Poses are joint-angle dictionaries that blend; the
feet are planted on y = 0 after posing. Faces are painted onto the 3D head as thin raised "decal" solids, so the
eyes, brows, mouth and blush stay crisp and anti-aliased at any size. Every figure is a solid ``Model`` with named
parts ("skin", "hair", "top", "bottom", "shoes", "eyes", ...), so recolouring, materials, layers and the scene
manifest work unchanged.

Units are metres, y is up, figures stand on y = 0 and face +z (towards the default camera).
"""

from . import _manifest
from .shapes import round_cone
from .tables import (BUILDS, EXPRESSIONS, HAIR_STYLES, HATS, TOPS, BOTTOMS, SHOES, EXTRAS, OUTFITS, SKIN_TONES,
                     HAIR_COLORS)
from .rig import POSES
from .character import Character, cast, random_person, CAST
from .species import SPECIES
from .animal_rig import ANIMAL_POSES
from .animal import Animal, animal
from .props import *          # noqa: F401,F403  (balloon, book, cup, ... the prop functions)
from .props import PROPS, prop
from .registry import print_on, decal_frame, names, find, describe

__all__ = [
    "Character", "cast", "random_person", "CAST", "BUILDS", "POSES", "EXPRESSIONS", "HAIR_STYLES", "HATS",
    "TOPS", "BOTTOMS", "SHOES", "EXTRAS", "OUTFITS", "SKIN_TONES", "HAIR_COLORS",
    "Animal", "animal", "SPECIES", "ANIMAL_POSES",
    "prop", "PROPS", "print_on", "decal_frame", "round_cone", "names", "find", "describe",
]

from .._args import module_getattr as _module_getattr   # noqa: E402

_HINTS = {n: 'use clay.animal("%s", pose="sit")' % n for n in SPECIES}
_HINTS.update({n: 'use clay.cast("%s", pose="wave")' % n for n in CAST})
_HINTS.update({n: 'use clay.cast("chef") for a ready-made character or clay.Character(build="toy", ...)'
               for n in ("person", "people", "human", "man", "woman", "boy", "girl", "figure", "character")})
_HINTS.update({n: 'use clay.animal("corgi") (clay.names("animal") lists the species)'
               for n in ("pet", "animals", "creature", "species")})
_HINTS.update({"props": 'use clay.prop("balloon") (clay.names("prop") lists them)',
               "pose": "poses are names: clay.POSES, clay.ANIMAL_POSES; apply with .pose(name)",
               "hold": "hold a prop with Character.hold(clay.prop(name))"})
__getattr__ = _module_getattr(__name__, _HINTS)

_manifest.install()
