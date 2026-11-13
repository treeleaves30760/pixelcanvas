# -*- coding: utf-8 -*-
"""Scene-manifest labels for clay figures.

``solid.Scene.render`` records one manifest entry per object. Clay characters and animals add what a person would
say about them: a character is a ``person`` ("people", "character", its cast role such as "chef"); an animal is a
``creature`` ("animal", its species and the family word: corgi -> dog). So ``render.py --check`` can answer prompts
such as "three people" or "two dogs".
"""
from __future__ import annotations

from .. import manifest as _manifest

# species -> the everyday words for it, besides the species name itself
FAMILY = {
    "dog": "dog", "labrador": "dog", "corgi": "dog", "poodle": "dog", "husky": "dog", "dachshund": "dog",
    "shiba": "dog", "cat": "cat", "horse": "horse", "pony": "horse", "unicorn": "horse", "bear": "bear",
    "polar_bear": "bear", "panda": "bear", "mouse": "mouse", "rabbit": "rabbit", "chick": "bird",
    "chicken": "bird", "duck": "bird", "penguin": "bird", "owl": "bird", "parrot": "bird", "flamingo": "bird",
    "eagle": "bird", "sparrow": "bird", "fish": "fish", "shark": "fish", "whale": "fish", "dolphin": "fish",
    "t_rex": "dinosaur", "tyrannosaurus": "dinosaur", "triceratops": "dinosaur", "brontosaurus": "dinosaur",
    "stegosaurus": "dinosaur", "dragon": "dragon", "ladybird": "insect", "ladybug": "insect", "bee": "insect",
    "butterfly": "insect", "caterpillar": "insect", "snail": "snail", "crab": "crab", "turtle": "turtle",
    "frog": "frog", "pig": "pig", "sheep": "sheep", "goat": "goat", "cow": "cow", "monkey": "monkey",
}


def person_info(role):
    """Manifest kind and aliases for a character with cast role ``role`` ('chef', 'kid', 'character')."""
    words = [] if role in (None, "", "character") else [str(role).replace("_", " ")]
    return {"kind": "person", "alias": words + ["person", "people", "character"]}


def animal_info(species):
    """Manifest kind and aliases for an animal of ``species`` ('corgi' -> dog, animal)."""
    sp = str(species)
    words = [sp.replace("_", " ")]
    if FAMILY.get(sp):
        words.append(FAMILY[sp])
    words.append("animal")
    return {"kind": "creature", "alias": words}


def _info(model):
    f = getattr(model, "_manifest_info", None)
    return f() if callable(f) else f


def label_entries(cv, result, start):
    """Re-label the manifest entries added since ``start`` that belong to clay figures in ``result``'s scene."""
    models = {m.name: m for m in getattr(getattr(result, "scene", None), "models", [])}
    for e in cv.manifest[start:]:
        info = _info(models[e["name"]]) if e.get("name") in models else None
        if not info:
            continue
        e["kind"] = info["kind"]
        e["aliases"] = list(dict.fromkeys(list(e.get("aliases") or []) + info["alias"]))


def install():
    """Wrap ``manifest.note_render3d`` once so clay figures are recorded as persons / creatures."""
    orig = _manifest.note_render3d
    if getattr(orig, "_clay_labels", False):
        return

    def note_render3d(cv, result):
        start = len(cv.manifest)
        orig(cv, result)
        try:
            label_entries(cv, result, start)
        except Exception:                                 # noqa: BLE001  (labels are advisory)
            pass

    note_render3d._clay_labels = True
    note_render3d.__doc__ = orig.__doc__
    _manifest.note_render3d = note_render3d
