# -*- coding: utf-8 -*-
"""pixelcanvas.pictograms: data integrity, rasterising, styles, search, SVG export, strict options, docs."""
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pixelcanvas import Canvas  # noqa: E402
from pixelcanvas import pictograms as P  # noqa: E402
from pixelcanvas.vector import Path  # noqa: E402

DOC = os.path.join(ROOT, "skills", "pixelcanvas", "references", "pictograms.md")
NAMES = P.names()


def test_library_size_and_categories():
    assert len(NAMES) >= 90
    assert len(set(NAMES)) == len(NAMES)
    assert {"music", "arts", "transport", "buildings"} <= set(P.categories())
    for must in ("saxophone", "lighthouse", "bus stop sign", "torii gate", "ferris wheel", "cable car"):
        assert must in NAMES
    for c in P.categories():
        assert P.names(c) and all(P.info(n)["category"] == c for n in P.names(c))


def test_every_record_is_well_formed():
    for n in NAMES:
        r = P.info(n)
        assert r["name"] == n and n == n.lower() and r["body"], n
        assert r["aliases"] or r["keywords"], n
        for role in P.ROLES:
            for d in r[role]:
                letters = set(re.findall(r"[A-Za-z]", d.lstrip("-")))
                assert letters <= set("MLCQAZe"), (n, role, letters)


@pytest.mark.parametrize("name", NAMES)
def test_every_pictogram_parses_rasterises_and_stays_in_its_box(name):
    r = P.info(name)
    for role in P.ROLES:
        for d in r[role]:
            if d.startswith("-"):
                continue
            x0, y0, x1, y1 = Path(d).bbox()
            assert x0 >= -0.01 and y0 >= -0.01 and x1 <= 100.01 and y1 <= 100.01, (name, role, (x0, y0, x1, y1))
    m = P.mask(name, 64, 64)
    assert m.dtype == np.float32 and m.shape == (64, 64)
    assert 0.02 < float(m.mean()) < 0.85, (name, float(m.mean()))
    assert float(m.max()) <= 1.0 and float(m.min()) >= 0.0
    edge = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]])
    assert float(edge.max()) < 0.5, name                      # nothing touches the box frame


def test_styles_run_and_return_masks():
    cv = Canvas(600, 120, "#f4efe6")
    for i, st in enumerate(P.STYLES):
        before = cv.buf.copy()
        p = P.pictogram(cv, "lighthouse", at=((i + 0.5) / 6, 0.5), size=0.8, style=st, color="#3f7fbf")
        assert isinstance(p, P.Pictogram)
        assert p.mask.shape == (120, 600) and float(p.mask.max()) > 0.9, st
        changed = np.abs(cv.buf - before).max(-1) > 0.02
        assert changed.any(), st
        cx, cy = p.anchors["center"]
        assert abs(cx - (i + 0.5) * 100) < 6 and abs(cy - 60) < 8, st
        assert "light" in p.anchors and len(p.uv("light")) == 2


def test_silhouette_cuts_details_and_duotone_paints_them():
    cv = Canvas(200, 200, "#ffffff")
    P.pictogram(cv, "keyboard", at=(0.5, 0.5), size=0.9)
    d = P.mask("keyboard", 200, 200, box=(10, 10, 190, 190), role="detail")
    assert float(d.max()) > 0.9
    # a white key's middle is background colour in a silhouette
    x, y = 10 + 0.14 * 180, 10 + 0.64 * 180
    assert cv.buf[int(y), int(x)].mean() > 0.9
    cv2 = Canvas(200, 200, "#ffffff")
    P.pictogram(cv2, "keyboard", at=(0.5, 0.5), size=0.9, style="duotone", detail_color="#ff0000")
    assert cv2.buf[int(y), int(x), 0] > 0.9 and cv2.buf[int(y), int(x), 1] < 0.2


def test_placement_anchor_rotate_flip():
    cv = Canvas(400, 400)
    p = P.pictogram(cv, "house", at=(0.5, 0.8), size=0.5, anchor="bm")
    assert abs(p.bbox[3] - 320) < 1.5
    a = P.mask("car", 100, 100)
    b = P.mask("car", 100, 100, flip=True)
    assert np.abs(a - b[:, ::-1]).mean() < 0.01
    c = P.mask("car", 100, 100, rotate=90)
    assert abs(float(c.sum()) - float(a.sum())) / float(a.sum()) < 0.03
    small = P.mask("anchor", 50, 50).mean()
    big = P.mask("anchor", 400, 400).mean()
    assert abs(small - big) < 0.01                              # resolution independent


def test_eraser_strings_remove_coverage():
    full = P.mask("car", 100, 100, role="body")
    rec = P.info("car")
    assert any(d.startswith("-") for d in rec["body"])
    M = P._v._scale(1.0)
    without = P._raster_role([d for d in rec["body"] if not d.startswith("-")], M, 100, 100)
    assert float(without.sum()) > float(full.sum())


def test_find_describe_and_aliases():
    assert P.find("sax")[0] == "saxophone"
    assert P.find("薩克斯風")[0] == "saxophone"
    assert P.find("saxofone")[0] == "saxophone"
    assert P.find("bus stop")[0] == "bus stop sign"
    assert P.find("fire truck")[0] == "fire engine"
    assert P.find("subway")[0] == "metro"
    assert "lighthouse" in P.find("light house")
    assert P.find("zzzz qqq") == []
    assert len(P.find("boat", n=3)) == 3
    assert "saxophone" in P.describe("sax") and "music" in P.describe("saxophone")
    cv = Canvas(100, 100)
    p = P.pictogram(cv, "sax")                                  # an exact alias works as a name
    assert p.name == "saxophone"


def test_strict_options_and_names():
    cv = Canvas(100, 100)
    with pytest.raises(TypeError, match="did you mean 'size'"):
        P.pictogram(cv, "car", sise=0.3)
    with pytest.raises(ValueError, match="saxophone"):
        P.pictogram(cv, "saxofone")
    with pytest.raises(ValueError, match="line"):
        P.pictogram(cv, "car", style="outline")
    with pytest.raises(ValueError):
        P.names("vehicles")
    P.pictogram(cv, "car", colour="#ff0000", position=(0.5, 0.5))     # accepted spellings


def test_manifest_entry_per_pictogram():
    cv = Canvas(300, 150)
    P.pictogram(cv, "fire truck", at=(0.3, 0.5), size=0.5, colour="#c00")
    P.pictogram(cv, "lighthouse", at=(0.7, 0.5), size=0.5, style="sticker")
    ents = [e for e in cv.manifest if e["kind"] == "object"]
    assert [e["name"] for e in ents] == ["fire engine", "lighthouse"]
    assert "fire truck" in ents[0]["aliases"]


def test_svg_export_parses_for_every_pictogram():
    for i, n in enumerate(NAMES):
        style = ("silhouette", "duotone", "line")[i % 3]
        svg = P.drawing(n, size=64, style=style).to_svg()
        root = ET.fromstring(svg)
        assert root.tag.endswith("svg") and root.get("width") == "64"
        assert root.findall(".//{http://www.w3.org/2000/svg}path"), n


def test_svg_silhouette_matches_the_raster():
    for n in ("keyboard", "turntable", "car", "theatre masks"):
        ras = P.drawing(n, size=120).render()
        m = P.mask(n, 120, 120)
        ink = np.abs(ras.buf - ras.buf[0, 0]).max(-1) > 0.3
        assert np.mean(ink == (m > 0.5)) > 0.97, n


def test_sheet_and_helpers():
    cv = Canvas(640, 200, "#ffffff")
    out = P.sheet(cv, P.names("arts")[:6], cols=6, tile_color="#eef2f7")
    assert len(out) == 6 and all(isinstance(p, P.Pictogram) for p in out)
    for d in (P.circle(50, 50, 10), P.ellipse(50, 50, 20, 10, 30), P.ring(50, 50, 20, 5), P.rect(10, 10, 30, 20, 4),
              P.poly([(10, 10), (90, 10, 6), (50, 90)], 3), P.bar(10, 10, 90, 90, 6), P.arc(50, 50, 30, 0, 120, 6),
              P.smooth([(20, 20), (80, 20), (80, 80), (20, 80)]), P.tube([(10, 50), (50, 40), (90, 50)], [4, 8, 4]),
              P.frame(10, 10, 80, 80, 8, 6), P.reverse(P.rect(0, 0, 10, 10))):
        assert set(re.findall(r"[A-Za-z]", d)) <= set("MLCQAZe")
        Path(d)
    ring = P._raster_role([P.ring(50, 50, 40, 10)], P._v._scale(1.0), 100, 100)
    assert ring[50, 50] < 0.01 and ring[50, 15] > 0.99            # the hole of a ring stays open


def test_docs_snippets_run(tmp_path, monkeypatch):
    text = open(DOC, encoding="utf-8").read()
    assert len(text.splitlines()) <= 250
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    assert len(blocks) >= 4
    monkeypatch.chdir(tmp_path)
    for code in blocks:
        code = code.replace("Canvas(1080, 1350", "Canvas(216, 270")
        exec(compile(code, "pictograms.md", "exec"), {"__name__": "__doc__"})


def test_example_scene_renders_small():
    import importlib
    sys.path.insert(0, os.path.join(ROOT, "scenes"))
    mod = importlib.import_module("pictogram_sheet")
    cv = mod.render(384, 240, 1, None)
    assert cv.buf.shape == (240, 384, 3)
    assert float(np.abs(cv.buf - cv.buf[2, 2]).max()) > 0.3


# ============================================================================================ second batch
SECOND = ("everyday", "living", "symbols", "produce", "kitchen_apps", "outdoor_home", "tech_bio_story")


def test_second_batch_size_and_categories():
    new = [n for n in NAMES if P.info(n)["module"] in SECOND]
    assert len(new) >= 380
    assert len(NAMES) >= 500
    assert {"home", "objects", "clothing", "kitchen", "tech", "tools", "food", "animals", "people", "nature",
            "sports", "science", "medical", "work", "money", "holidays", "symbols"} <= set(P.categories())
    for must in ("cat", "dog", "bird", "branch", "window", "database", "server", "speech bubble", "robot head",
                 "table lamp", "umbrella", "closed umbrella", "takeaway coffee", "bubble tea", "dinosaur",
                 "long-neck dinosaur", "wheelchair user", "thumbs up", "sun", "rain cloud", "toadstool",
                 "football", "trophy", "atom", "first-aid cross", "anatomical heart", "graduation cap",
                 "christmas tree", "heart", "wifi", "check mark", "u-turn arrow"):
        assert must in NAMES, must


def test_second_batch_aliases_never_shadow_another_pictogram():
    owner = {}
    for n in NAMES:
        for t in [n] + [a.lower() for a in P.info(n)["aliases"]]:
            owner.setdefault(t, set()).add(n)
    for n in NAMES:
        if P.info(n)["module"] not in SECOND:
            continue
        for a in P.info(n)["aliases"]:
            others = owner[a.lower()] - {n}
            assert not others, (n, a, others)          # an exact alias names exactly one pictogram


def test_second_batch_search():
    assert P.find("貓")[0] == "cat"
    assert P.find("kitten")[0] == "cat"
    assert P.find("bird")[0] == "bird"
    assert P.find("dino")[:2] == ["dinosaur", "long-neck dinosaur"]
    assert P.find("umbrela")[0] == "umbrella"
    assert P.find("咖啡")[0] == "takeaway coffee"
    assert P.find("warning")[0] == "warning triangle"
    assert P.find("settings")[0] == "gear"
    assert P.find("鍵盤")[0] == "keyboard"                     # the first batch keeps its aliases
    assert P.find("耳機")[0] == "headphones"
    assert P.find("trolley")[0] == "tram"
    assert P.pictogram(Canvas(80, 80), "sax").name == "saxophone"
    assert "animals" in P.describe("cat") and "sitting" in P.describe("cat")


def test_second_batch_silhouettes_read_at_small_size():
    # every new pictogram keeps a real silhouette at icon size: enough ink, and its details stay inside the body
    for n in NAMES:
        if P.info(n)["module"] not in SECOND:
            continue
        sil = P.mask(n, 48, 48)
        assert float(sil.mean()) > 0.04, n
        solid = P.mask(n, 96, 96, role="solid")
        det = P.mask(n, 96, 96, role="detail")
        outside = float((det * (1 - solid)).sum()) / max(float(det.sum()), 1.0)
        assert outside < 0.08, (n, outside)


def test_catalogue_scene_lists_every_category():
    import importlib
    sys.path.insert(0, os.path.join(ROOT, "scenes"))
    mod = importlib.import_module("pictogram_sheet")
    assert set(mod._order()) == set(P.categories())
    u, shelves, total = mod._layout(mod.DEFAULT_SIZE[0], mod._order())
    assert 100 * u >= 96                                       # tiles of at least 96 px at full size
    assert total * u <= mod.DEFAULT_SIZE[1] + 1
    placed = [c for s in shelves for c, _, _ in s["items"]]
    assert sorted(placed) == sorted(P.categories())


# ============================================================================================ third batch
THIRD = ("produce", "kitchen_apps", "outdoor_home", "tech_bio_story")
NEW = [n for n in NAMES if P.info(n)["module"] in THIRD]
REQUESTED = (
    "eggplant", "broccoli", "onion", "lettuce", "cabbage", "radish", "garlic", "potato", "cucumber", "peas", "pumpkin",
    "jack-o-lantern", "notes", "checklist", "chat bubble", "sun behind cloud", "calculator", "gear", "folder", "map",
    "camera", "music note", "envelope", "calendar", "jar", "spice jar", "pot", "frying pan", "shelf", "ladle",
    "chef hat", "umbrella", "bus front", "bus side", "bus stop shelter", "bus stop sign", "street lamp", "bench",
    "traffic light", "tent", "stone lantern", "arched bridge", "koi", "torii gate", "bonsai", "watering can",
    "armchair", "floor lamp", "bookshelf", "round rug", "desk lamp", "side table", "window with sill", "earbuds",
    "charging case", "phone", "smartwatch", "terminal window", "server", "cell", "nucleus", "mitochondrion",
    "chloroplast", "dna helix", "leaf cross-section", "wizard tower", "castle", "dragon", "fishing boat", "lighthouse")


def test_third_batch_adds_at_least_120_pictograms():
    assert len(NEW) >= 120
    for n in REQUESTED:
        assert n in NAMES, n
    assert P.info("pumpkin")["category"] == "food" and "carved" not in P.describe("pumpkin")
    assert "carved" in P.describe("jack-o-lantern")
    assert P.find("carved pumpkin")[0] == "jack-o-lantern"


def test_third_batch_reads_at_icon_size():
    for n in NEW:
        sil = P.mask(n, 64, 64)
        assert 0.05 < float(sil.mean()) < 0.75, n
        solid = P.mask(n, 64, 64, role="solid")
        assert float(sil.sum()) > 0.4 * float(solid.sum()), n           # the details do not eat the glyph
    cv = Canvas(640, 100, "#f4efe6")
    for i, st in enumerate(P.STYLES):
        p = P.pictogram(cv, NEW[i * 7], at=((i + 0.5) / 6, 0.5), size=0.8, style=st)
        assert float(p.mask.max()) > 0.9, st


def test_find_group_words_and_no_unrelated_names():
    assert P.find("weather")[0] == "sun behind cloud"
    assert set(P.find("weather", n=40)) <= {"sun behind cloud", "moon behind cloud", "rain cloud", "storm cloud",
                                            "snow cloud", "windy", "fog", "sun", "cloud", "lightning", "rainbow",
                                            "snowflake", "umbrella", "thermometer"}
    assert P.find("tasks")[0] == "checklist" and "shoe" not in P.find("tasks", n=40)
    veg = P.find("veg", n=40)
    for must in ("eggplant", "broccoli", "carrot", "tomato", "potato", "onion", "lettuce", "cabbage"):
        assert must in veg
    assert P.find("vegetables")[0] in veg
    boats = P.find("boat", n=20)
    for must in ("sailboat", "rowing boat", "canoe", "motorboat", "fishing boat"):
        assert must in boats
    assert "robot head" not in boats and "table tennis bat" not in boats
    assert P.find("eggplant") == ["eggplant"]
    assert "bus stop shelter" in P.find("bus stop") and P.find("bus stop")[0] == "bus stop sign"
    assert "magnet" not in P.find("shoe", n=40)


def test_find_matches_head_nouns_substrings_and_plurals():
    assert P.find("tomatoes")[0] == "tomato" and P.find("cherry")[0] == "cherries"
    assert P.find("sail")[0] == "sailboat"
    assert "wine glass" in P.find("red wine glass", n=3)
    assert P.find("light house")[0] == "lighthouse"
    assert P.find("zzzz qqq") == [] and P.find("") == []
    assert P.find("saxofone")[0] == "saxophone"
    for trig, members in P._GROUPS:
        assert all(m in NAMES for m in members), trig


def test_detail_color_colours_the_detail_role_in_every_style():
    red = np.array([0.886, 0.341, 0.298])
    box = (15, 15, 285, 285)
    for name in ("mug", "cup and saucer", "teacup"):
        acc = P.mask(name, 300, 300, box=box, role="accent")
        sel = (acc > 0.95) & (P.mask(name, 300, 300, box=box, role="body") < 0.05)
        assert sel.sum() > 20, name
        for style in ("duotone", "sticker", "shaded"):
            cv = Canvas(300, 300, "#ffffff")
            P.pictogram(cv, name, at=(0.5, 0.5), size=0.9, color="#3f7fbf", style=style, detail_color="#e2574c")
            got = cv.buf[sel].mean(0)
            if style == "shaded":
                assert got[0] > got[2] + 0.2, (name, style, got)             # still the red family when lit
            else:
                assert np.abs(got - red).max() < 0.12, (name, style, got)
    cv = Canvas(300, 300, "#ffffff")                          # the default is a tint of the body, never pale grey
    P.pictogram(cv, "mug", at=(0.5, 0.5), size=0.9, color="#3f7fbf", style="duotone")
    acc = P.mask("mug", 300, 300, box=box, role="accent") > 0.95
    assert np.abs(cv.buf[acc].mean(0) - np.array([0.79, 0.82, 0.85])).max() > 0.05
    for style in P.STYLES:                                     # detail_color is accepted by every style
        P.pictogram(Canvas(120, 120), "lighthouse", style=style, detail_color="#ff00aa", size=0.8)


def test_shaded_style_is_a_lit_volume_with_soft_shadow():
    cv = Canvas(600, 600, "#f4efe6")
    p = P.pictogram(cv, "teapot", at=(0.5, 0.5), size=0.9, color="#3f7fbf", style="shaded")
    m = p.mask > 0.99
    lum = cv.buf[m].mean(-1)
    assert lum.max() - lum.min() > 0.3                         # light and dark parts, not a flat fill
    below = cv.buf[int(p.bbox[3]) + 3, int((p.bbox[0] + p.bbox[2]) / 2)]
    assert below.mean() < 0.95                                 # the soft shadow under the object


def test_fit_box_sizes_to_a_box():
    cv = Canvas(400, 200, "#ffffff")
    a = P.pictogram(cv, "bus", at=(0.25, 0.5), size=(0.4, 0.8), fit="box")
    b = P.pictogram(cv, "water bottle", at=(0.75, 0.5), size=(0.4, 0.8), fit="box")
    for p, (x0, x1) in ((a, (20, 180)), (b, (220, 380))):
        w, h = p.bbox[2] - p.bbox[0], p.bbox[3] - p.bbox[1]
        assert w <= 160.5 and h <= 160.5 and (abs(w - 160) < 1.5 or abs(h - 160) < 1.5)
        assert x0 - 1 <= p.bbox[0] and p.bbox[2] <= x1 + 1
        assert abs((p.bbox[1] + p.bbox[3]) / 2 - 100) < 1.5
    assert (a.bbox[2] - a.bbox[0]) > 1.4 * (b.bbox[2] - b.bbox[0])        # the wide bus takes the box width
    c = P.pictogram(Canvas(400, 200), "bus", size=0.5, fit="box")        # one number: a square box
    assert max(c.bbox[2] - c.bbox[0], c.bbox[3] - c.bbox[1]) < 100.5
    h = P.pictogram(Canvas(400, 200), "bus", size=0.5)                   # size= is a fraction of the canvas height
    bb = P._role_bbox(P._get("bus"))
    assert abs((h.bbox[3] - h.bbox[1]) - 0.5 * 200 * (bb[3] - bb[1]) / 100) < 1.5
    with pytest.raises(ValueError, match="did you mean 'box'"):
        P.pictogram(Canvas(50, 50), "bus", fit="boxx")
    with pytest.raises(ValueError, match="size must be"):
        P.pictogram(Canvas(50, 50), "bus", size="big", fit="box")


def test_pictogram_attributes_and_manifest_registration():
    cv = Canvas(200, 200)
    p = P.pictogram(cv, "fishing boat", at=(0.5, 0.5), size=0.5)
    assert isinstance(p.mask, np.ndarray) and p.mask.shape == (200, 200)
    assert [e["name"] for e in cv.manifest if e["kind"] == "object"] == ["fishing boat"]
    with pytest.raises(TypeError):
        p.mask()


def test_reference_page_states_roles_mask_manifest_and_size():
    text = open(DOC, encoding="utf-8").read()
    for phrase in ('role="silhouette"', 'role="solid"', "array attribute", 'fit="box"',
                   "fraction of the canvas height", "records itself"):
        assert phrase in text, phrase
    assert len(text.splitlines()) <= 250


def test_pictogram_name_and_alias_record_under_the_requests_word():
    from pixelcanvas import Canvas, manifest
    cv = Canvas(320, 240, "#f4f0e8")
    P.pictogram(cv, "sailboat", at=(0.3, 0.5), size=0.3, name="boat", alias="dinghy")
    P.pictogram(cv, "sailboat", at=(0.7, 0.5), size=0.3)
    P.pictogram(cv, name="anchor", at=(0.5, 0.2), size=0.2)                 # old spelling: name is the shape
    E = {e["name"]: e for e in manifest.entries(cv)}
    assert "boat" in E and "dinghy" in E["boat"]["aliases"] and "sailboat" in E["boat"]["aliases"]
    assert "sailboat" in E and "anchor" in E
    with pytest.raises(TypeError, match="unknown option"):
        P.pictogram(cv, "sailboat", colour_x=1)
