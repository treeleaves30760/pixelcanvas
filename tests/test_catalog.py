# -*- coding: utf-8 -*-
"""Tests for pixelcanvas.catalog and tools/kit.py: the index, every call snippet, search quality, CJK, the CLI."""
import importlib
import json
import os
import re
import subprocess
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pixelcanvas import catalog  # noqa: E402

KIT = os.path.join(ROOT, "tools", "kit.py")
LOADED = {k for k, *_ in catalog.kits()}


def _runnable():
    return [e for e in catalog.entries() if e.lang == "python" and e.kit not in catalog._UNFINISHED]


def _ids(res):
    return set(res.ids())


def _tops(res):
    return [e.id for e in res.best()]


# ------------------------------------------------------------------------------------------------ the index
def test_index_builds_fast_and_covers_the_kits():
    code = ("import time; t = time.time(); from pixelcanvas import catalog; n = len(catalog.entries()); "
            "print(n, time.time() - t)")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    n, dt = out.stdout.split()
    assert int(n) > 500
    assert float(dt) < 3.0, f"cold build took {dt} s"          # about 0.6 s on a laptop
    for kit in ("flora", "elements", "texture", "stylize", "palette", "backdrop", "ui", "vector", "diagram",
                "chart", "drafting", "solid", "text", "layout", "scenes"):
        assert kit in LOADED, kit


def test_entry_fields():
    for e in catalog.entries():
        assert e.kit and e.name and e.kind and e.call, e
        assert e.id.startswith(e.kit + ".")
        if e.lang == "python":
            assert "import" in e.snippet()


def test_index_survives_a_broken_kit(monkeypatch):
    def broken(_mod):
        raise RuntimeError("half-written kit")

    monkeypatch.setitem(catalog._HANDLERS, "palette", broken)
    es = catalog.build(kits=["flora", "palette", "definitely_not_a_kit", "stylize"])
    kits = {e.kit for e in es}
    assert kits == {"flora", "stylize"}
    assert catalog.build(kits=["definitely_not_a_kit"]) == []


def test_generic_kit_discovery(monkeypatch):
    """A kit without a handler joins through its names() registry and its documented functions."""
    import types
    mod = types.ModuleType("pixelcanvas.fake_kit")

    def names():
        return ["alpha", "beta"]

    def paint(cv, name, at=(0.5, 0.5)):
        """Paint one fake thing."""
        return name

    def splash(cv, size=3):
        """A documented canvas function."""
        return size

    for f in (names, paint, splash):
        f.__module__ = "pixelcanvas.fake_kit"
        setattr(mod, f.__name__, f)
    monkeypatch.setitem(sys.modules, "pixelcanvas.fake_kit", mod)
    es = catalog.build(kits=["fake_kit"])
    calls = {e.name: e.call for e in es}
    assert calls["alpha"] == 'fake_kit.paint(cv, "alpha")'
    assert "splash" in calls and calls["splash"].startswith("fake_kit.splash(cv")


def test_entries_filter_and_kinds():
    plants = catalog.entries(kit="flora", kind="plant")
    assert plants and all(e.kit == "flora" and e.kind == "plant" for e in plants)
    ks = catalog.kinds()
    for kind in ("plant", "effect", "material", "style", "backdrop", "palette", "icon", "chart kind",
                 "diagram", "diagram shape", "ui template", "ui component", "prefab", "text effect",
                 "layout template", "scene kind"):
        assert kind in ks, kind


# ------------------------------------------------------------------------------------------------ every snippet
_HEAVY = [e.id for e in _runnable() if e.heavy]
_RENDERED = set(_HEAVY[::4])            # 3D snippets: every model is built, every fourth one is ray-marched too


@pytest.mark.parametrize("entry", _runnable(), ids=lambda e: e.id)
def test_every_snippet_runs(entry):
    code = entry.snippet(320, 180)
    if entry.heavy and entry.id not in _RENDERED:
        code = code.replace("sc.render(cv)", "sc.camera")
    ns = {}
    exec(compile(code, entry.id, "exec"), ns)
    assert ns["cv"].buf.shape[:2] == (180, 320)


# ------------------------------------------------------------------------------------------------ search
def test_corgi_example():
    res = catalog.find("a corgi sitting under an oak tree at sunset")
    tops = _tops(res)
    if "creature" in LOADED or "clay" in LOADED:
        assert tops[0] in ("creature.corgi", "clay.corgi")
        assert 'pose="sit"' in res.groups[0].calls[0]
    assert "flora.oak" in tops
    assert "elements.sunset" in tops
    assert "palette.sunset" in _ids(res)
    assert res.kind == "illustration"                  # a picture: drawn freehand
    assert res.kit_kind == "pets"                      # the kit route: the clay kit's row in SKILL.md
    assert res.extras == ["flora"] and "add flora calls inside it" in res.text(kits=True)


@pytest.mark.parametrize("query, expect", [
    ("a puppy on a couch", {"solid.sofa"}),
    ("flames and smoke from a burning log", {"elements.flame", "elements.smoke"}),
    ("a red brick wall with ivy", {"texture.brick", "flora.ivy"}),
    ("gold text that says SALE", {"text.gold"}),
    ("pie chart of market share", {"chart.pie"}),
    ("a login screen for a banking app", {"ui.login"}),
    ("soft pastel mesh gradient background", {"backdrop.mesh"}),
    ("a sunflwer in a pot", {"flora.sunflower"}),                       # typo
    ("an org chart for a startup", {"diagram.org_chart"}),
    ("watercolor painting of a fox in a forest", {"flora.forest"}),
    ("a floor plan of an apartment", {"drafting.floor_plan"}),
    ("an engineering drawing of a flange", {"drafting.part_drawing"}),
    ("a 3d mug on a wooden table", {"solid.mug"}),
    ("skyscrapers at night with fireworks", {"elements.night", "elements.fireworks"}),
    ("app icons for mail and camera", {"vector.mail", "vector.camera"}),
])
def test_find_top_results(query, expect):
    res = catalog.find(query)
    assert expect <= set(_tops(res)), (query, _tops(res))


def test_skyline_is_offered_for_skyscrapers():
    assert "backdrop.skyline" in _ids(catalog.find("skyscrapers at night with fireworks"))


def test_find_alternatives_come_from_different_kits():
    res = catalog.find("sunset", n=3)
    g = res.groups[0]
    assert 2 <= len(g.results) <= 3
    assert len({e.kit for e in g.results}) == len(g.results)
    assert len(catalog.find("sunset", n=1).groups[0].results) == 1


def test_find_hints_colour_style_and_kind():
    res = catalog.find("a watercolour painting of a blue boat on the sea")
    assert res.style == "watercolor"
    assert any(w == "blue" and h.startswith("#") for w, h in res.colors)
    assert res.kind == "illustration" and res.kit_kind in ("styled", "landscape")
    assert catalog.suggest_kind("an engineering drawing of a bearing housing") == "drawing"
    assert catalog.suggest_kind("a poster for a jazz night") == "poster"
    assert catalog.suggest_kind("an analytics dashboard with KPI tiles") == "dashboard"
    assert catalog.suggest_kind("a flowchart of the sign-up process") == "diagram"


def test_synonyms_and_noise():
    assert {"creature.dog", "clay.dog"} & set(_tops(catalog.find("a puppy"))) or "creature" not in LOADED
    assert "flora.cherry_blossom" in _tops(catalog.find("blossom"))
    assert "elements.ocean" in _tops(catalog.find("the sea"))
    assert catalog.find("the of and with a").groups == []
    assert catalog.find("").groups == []


def test_quoted_text_is_content_not_subject():
    res = catalog.find("a poster titled 'Fire and Ice'")
    assert _tops(res) == ["layout.poster"]


# ------------------------------------------------------------------------------------------------ CJK
@pytest.mark.parametrize("query, expect", [
    ("流程圖", {"diagram.flowchart"}),
    ("海報", {"layout.poster"}),
    ("工程圖", {"drafting.part_drawing"}),
    ("平面圖", {"drafting.floor_plan"}),
    ("圖表", {"chart.bar"}),
    ("圖示", {"vector.icon"}),
    ("一棵樹和一朵花", {"flora.oak", "flora.daisy"}),
    ("火與海", {"elements.flame", "elements.ocean"}),
    ("山和雲", {"backdrop.mountains"}),
    ("桜の木", {"flora.cherry"}),
    ("夕焼けの海", {"elements.sunset", "elements.ocean"}),
])
def test_cjk_queries(query, expect):
    res = catalog.find(query)
    assert expect <= set(_tops(res)), (query, _tops(res))


def test_cjk_animals_and_kinds():
    if "creature" in LOADED:
        assert {"clay.cat", "clay.dog"} <= set(_tops(catalog.find("貓和狗")))
        assert {"clay.cat", "clay.dog"} <= set(_tops(catalog.find("猫と犬")))
        assert {"creature.cat", "creature.dog"} <= set(_tops(catalog.find("扁平風格的貓和狗 flat")))
    assert catalog.suggest_kind("流程圖") == "diagram"
    assert catalog.suggest_kind("平面圖") == "floorplan"
    assert catalog.find("水彩畫的山").style == "watercolor"


# ------------------------------------------------------------------------------------------------ show / docs
def test_show_entry_and_function():
    txt = catalog.show("flora.oak")
    assert 'flora.plant(cv, "oak"' in txt and "references/flora.md" in txt and "flora.plant(cv, name" in txt
    assert "Canvas(" in txt
    txt = catalog.show("plant")
    assert txt.startswith("flora.plant(") and "read more" in txt
    txt = catalog.show("ui.Screen.button")
    assert "variant" in txt
    assert "did you mean" in catalog.show("flora.oakk") or "nothing called" in catalog.show("flora.oakk")
    assert catalog.show("flora.potted monstera") == catalog.show("flora.potted_monstera")


def test_reference_pages_exist():
    for kit, _n, _kinds, ref in catalog.kits():
        if ref:
            assert os.path.exists(os.path.join(ROOT, "skills", "pixelcanvas", *ref.split("/"))), (kit, ref)
    assert catalog.doc("flora").startswith("#")
    with pytest.raises(KeyError):
        catalog.doc("definitely_not_a_kit")


def test_catalog_doc_page_is_short_and_runs():
    page = os.path.join(ROOT, "skills", "pixelcanvas", "references", "catalog.md")
    with open(page, encoding="utf-8") as f:
        text = f.read()
    assert len(text.split("\n")) < 90
    for cmd in ("kit.py find", "kit.py list", "kit.py show"):
        assert cmd in text


# ------------------------------------------------------------------------------------------------ CLI
def _cli(*args):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    out = subprocess.run([sys.executable, KIT, *args], cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8", env=env)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_cli_find_text_and_json():
    txt = _cli("find", "a corgi sitting under an oak tree at sunset")
    lines = txt.strip().split("\n")
    assert 3 <= len(lines) <= 40
    assert any("flora.plant(cv, \"oak\"" in ln for ln in lines)
    assert lines[0].endswith('python tools/new_scene.py NAME --prompt "a corgi sitting under an oak tree at sunset"')
    assert lines[0].startswith("draw it freehand")
    assert lines[-1].startswith("-> suggested kind: python tools/new_scene.py NAME --kind illustration")
    kits = _cli("find", "--kits", "a corgi sitting under an oak tree at sunset").strip().split("\n")
    assert any("--kind pets" in ln and "add flora calls inside it" in ln for ln in kits)
    long_query = ("a fox, a cat, a dog, an owl, a whale, a rabbit, a frog, a penguin, a horse, a bear, an oak, a rose, "
                  "a campfire, rain, snow, fog, a brick wall, a poster, a bar chart, a flowchart, a mug, a car")
    assert len(_cli("find", long_query).strip().split("\n")) <= 40
    data = json.loads(_cli("find", "pie chart of market share", "--json", "--n", "2"))
    assert data["kind"] == "chart" and data["groups"][0]["results"][0]["id"] == "chart.pie"
    assert len(data["groups"][0]["results"]) <= 2
    assert "snippet" in data["groups"][0]["results"][0]


def test_cli_list_show_kinds_doc():
    txt = _cli("list")
    assert "flora" in txt and "references/flora.md" in txt
    txt = _cli("list", "flora")
    assert "oak" in txt and "plant (" in txt
    assert max(len(ln) for ln in txt.split("\n")) <= 110
    assert json.loads(_cli("list", "stylize", "--json"))[0]["kit"] == "stylize"
    assert "flora.plant" in _cli("show", "flora.oak")
    assert "scene kind" in _cli("kinds")
    assert _cli("doc", "flora").startswith("#")


def test_find_is_quick():
    catalog.entries()
    t = time.time()
    for q in ("a corgi sitting under an oak tree at sunset", "流程圖", "an analytics dashboard with a donut"):
        catalog.find(q)
    assert time.time() - t < 1.5


# ------------------------------------------------------------------------------------------------ the whole request
FIXTURE = os.path.join(os.path.dirname(ROOT), "plans", "build", "dogfood-prompts.json")
RECIPE = "recipe"      # no entry draws the subject: the answer is the nearest entries plus a generic recipe

# What a knowledgeable person would choose for each request, decided by reading the request: (kind, the main
# subject's top call). The main subject is the picture's own format when it is a design ("a poster", "an
# architecture diagram"), else the first subject noun phrase; a setting that comes first ("an underwater reef", "a
# Japanese garden") gives the kind. A tuple lists equally good answers; RECIPE: no kit draws the subject, so the
# answer is its pictogram or, without one, the nearest entries of its meaning class and a recipe.
DOGFOOD = {
    "dev_architecture": ("diagram", "diagram.architecture_diagram"),
    "dev_readme_banner": ("banner", "layout.thumbnail"),
    "dev_social_card": ("card", "layout.thumbnail"),
    "dev_app_icons": ("icons", "vector.icon"),
    "dev_dashboard": ("dashboard", "chart.dashboard"),
    "dev_sequence": ("diagram", "diagram.sequence_diagram"),
    "dev_comparison": ("infographic", "diagram.infographic"),
    "eng_part": ("drawing", "drafting.part_drawing"),
    "eng_floorplan": ("floorplan", "drafting.floor_plan"),
    "eng_exploded": ("drawing", "drafting.isometric_view"),
    "poster_concert": ("poster", "layout.poster"),
    "poster_film": ("poster", "layout.poster"),
    "poster_bilingual": ("poster", "layout.flyer"),
    "poster_book": ("poster", "layout.book_cover"),
    "poster_thumbnail": ("card", "layout.thumbnail"),
    "poster_menu": ("poster", "layout.menu"),
    "people_busstop": ("illustration", "figure.person"),            # three friends: any person from a sentence
    "people_chef": ("illustration", "clay.chef"),            # storybook, soft shading: clay
    "people_anime": ("illustration", "figure.person"),
    "people_team": ("illustration", "figure.office_worker"),        # a team of five coworkers: five office workers
    "animal_pets": ("illustration", "clay.cat"),                       # no flat style named: clay
    "animal_horse": ("illustration", "clay.horse"),
    "animal_fox": ("illustration", "clay.fox"),
    "animal_stickers": ("illustration", "pictograms.panda"),       # flat stickers; creature has no panda
    "animal_reef": ("illustration", "elements.underwater"),
    "3d_product": ("product", "solid.earbuds"),               # earbuds: the solid prefab
    "3d_room": ("isometric", ("solid.armchair", "pictograms.armchair", "solid.isometric")),
    "3d_car": ("illustration", "solid.car"),                  # names no 3D word: drawn freehand
    "3d_island": ("scene3d", RECIPE),                         # a floating island
    "nature_campfire": ("illustration", "elements.campfire"),
    "nature_storm": ("illustration", RECIPE),                    # a lighthouse
    "nature_garden": ("illustration", "flora.maple"),
    "nature_desert": ("illustration", "backdrop.dunes"),
    "nature_city_rain": ("wallpaper", "elements.neon"),
    "style_ukiyoe": ("illustration", "elements.breaking_wave"),
    "style_pixel": ("illustration", RECIPE),                     # a wizard's tower
    "style_watercolour": ("illustration", ("solid.house", "pictograms.house")),
    "style_stainedglass": ("illustration", RECIPE),                # a peacock
    "style_riso": ("illustration", "layout.poster"),
    "style_pencil": ("illustration", RECIPE),                    # a rowing boat
    "style_oil": ("illustration", ("form.lathe", "form.tabletop")),
    "style_papercut": ("illustration", "flora.forest"),
    "sci_cell": ("figure", "diagram.callouts"),
    "sci_watercycle": ("infographic", "diagram.infographic"),
    "sci_solar": ("infographic", "diagram.infographic"),
    "sci_optics": ("figure", "diagram.callouts"),
    "ui_mobile": ("ui", "ui.phone"),
    "ui_landing": ("ui", "ui.landing"),
}
RECIPE_HEAD = {"3d_product": "earbud", "3d_island": "island", "nature_storm": "lighthouse", "style_pixel": "tower",
               "style_stainedglass": "peacock", "style_pencil": "boat"}
DOGFOOD_STYLE = {"style_ukiyoe": "ukiyoe", "style_pixel": "pixel_art", "style_watercolour": "watercolor",
                 "style_stainedglass": "stained_glass", "style_riso": "risograph", "style_pencil": "pencil",
                 "style_oil": "oil", "style_papercut": "paper_cut", "animal_stickers": "sticker",
                 "people_anime": "cel", "3d_island": "low_poly", "people_chef": "storybook"}
ANY = object()

# The ten reviewer cases and the requests written for this fixture: every kit, people / animals / plants / objects
# with and without entries, designs, diagrams, mixed scenes, CJK, colours versus styles, 2D versus 3D versus pictogram.
# (request, kind, main top call(s) or RECIPE, style)
REQUESTS = [
    ("A toy-like 3D baker waving next to a pink cake", "characters", "clay.chef", None),
    ("a corgi sitting under an oak tree at sunset", "illustration", "clay.corgi", None),
    ("a kite festival on a beach: three kites, a lighthouse and the banner 'Wind Days 2027'", "illustration",
     "clay.kite", None),
    ("a zither", "illustration", RECIPE, None),
    ("a gramophone", "illustration", RECIPE, None),
    ("a jazz trio at 'The Copper Room'", "illustration", "clay.musician", None),
    ("a tram", ANY, "pictograms.tram", None),
    ("a sundial on a lawn", "illustration", RECIPE, None),
    ("a lantern-lit sign reading 'Open Late' above a ramen stall", "illustration", RECIPE, None),
    ("a fox in orange and charcoal", "illustration", "clay.fox", None),
    ("Two clay toy kittens playing with a ball of yarn on a rug", "pets", "clay.cat", None),
    # people
    ("a doctor talking to a kid in a hospital hallway", "illustration", "clay.doctor", None),
    ("a 3D clay astronaut waving on the moon", "characters", "clay.astronaut", None),
    ("a flat vector illustration of a runner jogging in a park", "illustration", "figure.runner", None),
    ("a cute vinyl toy of a pirate holding a sword", "characters", "clay.pirate", None),
    ("an old fisherman mending nets on a pier", "illustration", "clay.character", None),
    ("a grandmother knitting in a rocking chair", "illustration", "clay.grandma", None),
    ("a comic strip of a superhero saving a cat", "comic", "figure.hero", None),
    ("a girl reading a book under a tree", "illustration", "clay.character", None),
    ("a man walking a dog in the rain", "illustration", "clay.character", None),
    ("two children flying a kite on a hill", "illustration", "clay.kid", None),
    ("a flat vector illustration of a man walking a dog", "illustration", "figure.person", None),
    ("a textbook figure of a doctor", "figure", ANY, None),
    ("a portrait of a woman in the style of Van Gogh", "illustration", "figure.person", "oil"),
    ("a pixel-art knight", "illustration", "figure.person", "pixel_art"),
    # animals
    ("a golden retriever puppy chasing a ball on the grass", "illustration", "clay.dog", None),
    ("a flat illustration of a golden retriever puppy", "illustration", "creature.golden", None),
    ("a cartoon penguin sliding on ice", "illustration", "creature.penguin", None),
    ("a plasticine dinosaur roaring", "pets", "clay.tyrannosaurus", None),
    ("an owl perched on a branch at night", "illustration", "clay.owl", None),
    ("an anime owl perched on a branch", "illustration", "creature.owl", None),
    ("a camel in the desert", "illustration", "pictograms.camel", None),
    ("a sea turtle swimming over a coral reef", "illustration", "clay.turtle", None),
    ("a cat sleeping on a sofa by the fireplace", "illustration", "clay.cat", None),
    ("a dragon flying past a castle", "illustration", "clay.dragon", None),
    ("a fox, as a charcoal drawing", "illustration", "creature.fox", "charcoal"),
    # plants
    ("a potted monstera on a windowsill", "illustration", "flora.potted_monstera", None),
    ("a cherry blossom tree by a river", "illustration", "flora.cherry", None),
    ("a bouquet of tulips in a vase", "illustration", "flora.bouquet", None),
    ("a sunflower field under a blue sky", "illustration", "flora.sunflower", None),
    # objects with and without entries
    ("a 3D render of a coffee mug on a desk", "scene3d", "solid.mug", None),
    ("a vintage alarm clock", "illustration", "pictograms.alarm_clock", None),
    ("a teapot and two cups on a wooden table", "illustration", "pictograms.teapot", None),
    ("a sports car", "illustration", "solid.car", None),
    ("a laptop on a desk, isometric", "isometric", "solid.laptop", None),
    ("a wooden toy train", "scene3d", "pictograms.train", None),
    ("a samovar", "illustration", RECIPE, None),
    ("a ukulele on a beach towel", "illustration", RECIPE, None),
    ("a pencil case on a desk", "illustration", RECIPE, None),
    ("an oil lamp on a table", ANY, ANY, None),
    ("a clay pot with a cactus", ANY, "pictograms.pot", None),
    ("a marble texture", ANY, "texture.marble", None),
    # effects, settings, landscapes
    ("a campfire with sparks under the stars", "illustration", "elements.campfire", None),
    ("a sunset over the sea", "illustration", "elements.ocean", None),
    ("a snowy mountain landscape", "illustration", "backdrop.mountains", None),
    ("a lighthouse at sunset with seagulls", "illustration", "pictograms.lighthouse", None),
    # designs, diagrams, UI
    ("a logo for a coffee shop called 'Bean There'", "icons", "vector.icon", None),
    ("an icon of a bicycle", "icons", "vector.icon", None),
    ("a poster titled 'Save the Bees' with a bee silhouette", "poster", "layout.poster", None),
    ("a bar chart of monthly sales titled 'Q3 Sales'", "chart", "chart.bar", None),
    ("a pie chart of browser market share", "chart", "chart.pie", None),
    ("a flowchart of the checkout process", "diagram", "diagram.flowchart", None),
    ("a mind map about healthy habits", "diagram", "diagram.mind_map", None),
    ("a landing page for a bakery app", "ui", "ui.landing", None),
    ("a settings screen for a mobile app", "ui", "ui.settings", None),
    ("a floor plan of a small studio apartment", "floorplan", "drafting.floor_plan", None),
    # CJK
    ("一隻貓坐在窗邊", "illustration", "clay.cat", None),
    ("柯基和櫻花樹", "illustration", "clay.corgi", None),
    ("水彩畫的山", "illustration", "backdrop.mountains", "watercolor"),
    ("一張海報", "poster", "layout.poster", None),
]


def _check(res, kind, top, style, label):
    if kind is not ANY:
        assert res.kind == kind, (label, res.kind, res.main)
    assert res.main is not None, label
    if top is RECIPE:
        assert res.main.fallback or res.main.results[0].kit == "pictograms", (label, res.main)
    elif top is not ANY:
        tops = (top,) if isinstance(top, str) else top
        assert res.main.results and res.main.results[0].id in tops, (label, res.main)
    assert res.style == style, (label, res.style)
    text = res.text()
    assert len(text.split("\n")) <= 30, (label, text)
    assert "no match" not in text
    if kind not in (ANY, None):
        assert f"--kind {kind}" in text


@pytest.fixture
def bare(monkeypatch):
    """The index without the pictograms package: the subjects it does not draw fall to the generic recipes."""
    monkeypatch.setattr(catalog, "_INDEX", [e for e in catalog.entries() if e.kit != "pictograms"])
    monkeypatch.setattr(catalog, "_VOCAB", {})
    monkeypatch.setattr(catalog, "_TYPO_BY_FIRST", {})
    monkeypatch.setattr(catalog, "_NAME_TOKS", set())
    monkeypatch.setattr(catalog, "_pictograms", lambda: None)


def _prompts():
    try:
        with open(FIXTURE, encoding="utf-8") as f:
            return json.load(f)
    except OSError:
        pytest.skip("dogfood-prompts.json is not next to the repository")


def test_dogfood_fixture_covers_every_prompt():
    assert {p["id"] for p in _prompts()} == set(DOGFOOD)
    assert len(DOGFOOD) == 48
    assert len(REQUESTS) >= 50


@pytest.mark.parametrize("pid", sorted(DOGFOOD))
def test_dogfood_request_kind_and_main_call(pid):
    prompt = next(p["prompt"] for p in _prompts() if p["id"] == pid)
    res = catalog.find(prompt)
    kind, top = DOGFOOD[pid]
    _check(res, kind, top, DOGFOOD_STYLE.get(pid), pid)
    if top is RECIPE:
        assert res.main.head == RECIPE_HEAD[pid], (pid, res.main)
        if res.main.fallback:
            assert res.main.fallback["register"].startswith("manifest.note(cv, 'object', '")
    tops = [g.results[0].id for g in res.groups if g.results]
    assert len(tops) == len(set(tops)), "each subject once"


@pytest.mark.parametrize("query, kind, top, style", REQUESTS, ids=[r[0][:40] for r in REQUESTS])
def test_requests_get_a_knowledgeable_answer(query, kind, top, style):
    _check(catalog.find(query), kind, top, style, query)


def test_dogfood_kind_is_a_template():
    kinds = {k for k, _ in DOGFOOD.values()} | {r[1] for r in REQUESTS if r[1] is not ANY}
    assert kinds <= set(catalog._scene_kinds())


# ------------------------------------------------------------------------------------------------ the reviewer cases
def _group(res, head):
    return next(g for g in res.groups if g.head == head)


def test_toy_baker_is_a_clay_chef_not_a_beaker():
    res = catalog.find("A toy-like 3D baker waving next to a pink cake")
    assert "beaker" not in res.text() and "clay.animal" not in res.text()
    assert res.groups[0].calls[0] == 'sc.add(clay.cast("chef", pose="wave"), "chef")'
    assert res.groups[0].results[1].id == "figure.chef"                 # the 2D alternative
    assert res.kind == "characters" and res.style is None
    cake = _group(res, "cake")
    assert cake.results[0].id == "clay.cake" and cake.colors[0][0] == "pink"


def test_participles_are_poses_and_named_trees_win():
    res = catalog.find("a corgi sitting under an oak tree at sunset")
    assert [g.head for g in res.groups] == ["corgi", "tree", "sunset"]
    assert "pictograms.sitting" not in res.ids()
    assert _group(res, "tree").results[0].id == "flora.oak"
    assert 'pose="sit"' in res.groups[0].calls[0] and 'pose="sit"' in res.groups[0].calls[1]


def test_counts_merge_into_the_subject_and_the_banner_carries_its_text():
    res = catalog.find("a kite festival on a beach: three kites, a lighthouse and the banner 'Wind Days 2027'")
    kites = [g for g in res.groups if g.head == "kite"]
    assert len(kites) == 1 and kites[0].count == 3 and not kites[0].fallback
    assert "no catalogue entry" not in res.text() and '"cat"' not in res.text()
    assert res.kind == "illustration" and res.kit_kind == "landscape"
    banner = _group(res, "banner")
    assert banner.calls[0].startswith('cv.text("Wind Days 2027"')
    assert "flat mark: pictograms.pictogram(cv, \"kite\"" in res.text(kits=True)


@pytest.mark.parametrize("query, near", [
    ("a zither", ['"harp"', '"guitar"']),
    ("a gramophone", ['"turntable"', '"vinyl record"']),
])
def test_no_entry_nearest_by_meaning(query, near):
    fb = catalog.find(query).groups[0].fallback
    assert fb and len(fb["nearest"]) >= 2
    for want, call in zip(near, fb["nearest"]):
        assert want in call and call.startswith("pictograms.pictogram(")


def test_lookalike_strings_never_answer():
    assert catalog._typo("baker", catalog._vocab()) is None
    assert catalog._typo("sunflwer", catalog._vocab()) == "sunflower"
    assert catalog._typo("zither", catalog._vocab()) is None
    assert catalog.find("a tram").groups[0].results[0].id == "pictograms.tram"
    jazz = catalog.find("a jazz trio at 'The Copper Room'")
    assert jazz.groups[0].count == 3 and "train" not in jazz.text() and "rhino" not in jazz.text()
    sam = catalog.find("a samovar").groups[0].fallback
    assert sam["class"] is None and sam["nearest"] == []                  # unknown meaning: the recipe alone


def test_every_named_head_noun_is_a_subject():
    res = catalog.find("a sundial on a lawn")
    assert [g.head for g in res.groups] == ["sundial", "lawn"]
    assert res.groups[0].fallback["class"] == "timepiece"
    res = catalog.find("a lantern-lit sign reading 'Open Late' above a ramen stall")
    assert [g.head for g in res.groups] == ["sign", "stall"]
    assert res.groups[0].calls[0].startswith('cv.text("Open Late"')
    assert "pictograms.reading" not in res.ids() and "pictograms.lantern" not in res.ids()


def test_colour_words_are_not_styles():
    res = catalog.find("a fox in orange and charcoal")
    assert res.style is None and [w for w, _ in res.colors] == ["orange", "charcoal"]
    assert [g.head for g in res.groups] == ["fox"]
    for q in ("a clay pot with a cactus", "an oil lamp on a table", "a pencil case on a desk"):
        assert catalog.find(q).style is None, q
    for q, st in (("a fox in watercolour", "watercolor"), ("a fox, as a charcoal drawing", "charcoal"),
                  ("a pixel-art fox", "pixel_art"), ("an oil painting of a fox", "oil"),
                  ("a watercolour of two ducks", "watercolor"), ("a fox in the style of a ukiyo-e print", "ukiyoe")):
        assert catalog.find(q).style == st, q


def test_poster_and_banner_snippets_carry_the_request_text():
    res = catalog.find("A poster for a jazz night called 'Blue Hour' on Friday 21:00 at The Velvet Room")
    call = res.groups[0].calls[0]
    assert 'title="Blue Hour"' in call and "Night Market" not in call
    res = catalog.find("A README banner for a tool called 'driftwood' with the tagline 'your shell, everywhere'")
    assert 'title="driftwood"' in res.groups[0].calls[0] and 'subtitle="your shell, everywhere"' in res.groups[0].calls[0]
    res = catalog.find("a bar chart of monthly sales titled 'Q3 Sales'")
    assert 'title="Q3 Sales"' in res.groups[0].calls[0]
    assert all("Night Market" not in e.call for e in catalog.entries(kit="layout"))


def test_clay_kinds_come_from_the_kit_table():
    res = catalog.find("Two clay toy kittens playing with a ball of yarn on a rug")
    assert res.kind == "pets" and res.style is None
    g = res.groups[0]
    assert g.count == 2 and g.results[0].id == "clay.cat" and 'pose="play_bow"' in g.calls[0]
    assert [x.head for x in res.groups] == ["kitten", "ball", "rug"]
    assert catalog.kit_kinds()["clay"] == ["characters", "pets", "scene3d"]
    assert not hasattr(catalog, "_CLAY_KIND")


def test_diagram_snippets_leave_the_direction_to_auto():
    for e in catalog.entries(kit="diagram"):
        assert "direction=" not in e.code, e.id


# ------------------------------------------------------------------------------------------------ 2D / 3D / pictogram
def test_illustrations_take_the_drawing_kit_and_designs_the_pictogram():
    ill = catalog.find("a flat illustration of a cat on a windowsill")
    assert ill.groups[0].results[0].id == "creature.cat"
    assert any(e.id == "pictograms.cat" for e in ill.groups[0].results)          # the flat mark alternative
    poster = catalog.find("a poster for a cat cafe with a cat silhouette")
    cat = next(g for g in poster.groups if g.head == "cat")
    assert cat.results[0].id == "pictograms.cat"
    assert catalog.find("a cat icon").groups[0].results[0].kit in ("pictograms", "vector")
    toy = catalog.find("a toy cat")
    assert toy.groups[0].results[0].id == "clay.cat" and toy.kind == "pets"
    flat = catalog.find("a flat cartoon cat")
    assert flat.groups[0].results[0].id == "creature.cat" and flat.kit_kind == "animals"
    plain = catalog.find("a cat")                                   # no flat style named: clay, then creature
    assert [e.kit for e in plain.groups[0].results[:2]] == ["clay", "creature"] and plain.kit_kind == "pets"
    assert flat.kind == plain.kind == "illustration"                # both drawn freehand


def test_a_person_without_an_entry_is_figure_person():
    res = catalog.find("a flat vector illustration of an old fisherman mending nets on a pier")
    assert res.groups[0].calls[0].startswith('figure.person("old fisherman", name="fisherman")')
    res = catalog.find("an old fisherman mending nets on a pier")          # unstyled: a clay person, varied by seed
    assert res.groups[0].calls[0] == 'sc.add(clay.random_person(seed=1, build="elder"), "fisherman")'


# ------------------------------------------------------------------------------------------------ no dead ends
@pytest.mark.parametrize("query, head, cls", [
    ("saxophone silhouette", "saxophone", "instrument"),
    ("a lighthouse", "lighthouse", "building"),
    ("a camel", "camel", "animal"),
    ("a pizza", "pizza", "food"),
    ("a kayak", "kayak", "vehicle"),
])
def test_unknown_subject_is_answered_with_a_recipe(bare, query, head, cls):
    res = catalog.find(query)
    assert res.groups and res.groups[0].fallback
    fb = res.groups[0].fallback
    assert fb["class"] == cls and res.groups[0].head == head
    text = res.text(kits=True)
    assert "no match" not in text
    assert any(ln.strip().startswith(f"manifest.note(cv, 'object', '{head}', mask=m)")
               for ln in text.splitlines())
    assert fb["nearest"], "the nearest entries by meaning class"
    assert "no catalogue entry" not in text and "stand-in" in text


def test_saxophone_silhouette_gets_the_curve_recipe_and_a_nearby_icon(bare):
    fb = catalog.find("saxophone silhouette").groups[0].fallback
    assert fb["title"].startswith("silhouette from a curve: bezier_points")
    assert any("vector.icon" in c for c in fb["nearest"])
    assert len(fb["lines"]) == 3


def test_recipe_follows_the_kind():
    assert catalog.recipe("x", "scene3d")[0].startswith("3D: solid primitives blended with smooth_union")
    assert catalog.recipe("x", "product")[0].startswith("3D:")
    assert catalog.recipe("x", "landscape")[0].startswith("shape from SDF primitives")
    assert catalog.recipe("x", "poster", silhouette=True)[0].startswith("silhouette from a curve")
    for kind, sil in (("scene3d", False), ("poster", True), ("nature", False)):
        title, lines, reg = catalog.recipe("thing", kind, sil)
        assert len(lines) == 3
        assert reg == ("# sc.add(body, 'thing') already names it for --check" if kind == "scene3d" else
                       "manifest.note(cv, 'object', 'thing', mask=m)    # register the shape for --check")


@pytest.mark.parametrize("kind, sil", [("scene3d", False), ("poster", True), ("nature", False)])
def test_recipes_run_as_pasted(kind, sil):
    import numpy as np
    from pixelcanvas import Canvas, brush, form, masks, sdf, solid
    cv = Canvas(256, 144)
    _title, lines, _reg = catalog.recipe("thing", kind, sil)
    ns = dict(np=np, cv=cv, brush=brush, form=form, masks=masks, sdf=sdf, solid=solid)
    exec(compile("\n".join(lines), "recipe", "exec"), ns)
    if kind != "scene3d":
        assert float(cv.buf.max()) > 0 and ns["m"].max() > 0.5


def test_no_answer_is_never_no_match():
    for q in ("", "the of and with a", "qwzx", "a thing"):
        assert "no match" not in catalog.find(q).text()
    assert "manifest.note(cv, 'object', '<name>', mask=m)" in catalog.find("").text()


def test_the_pictograms_package_is_indexed_when_it_exists():
    pytest.importorskip("pixelcanvas.pictograms")
    es = catalog.entries(kit="pictograms")
    assert es and all(e.call.startswith("pictograms.pictogram(cv, ") for e in es)
    res = catalog.find("saxophone silhouette")
    assert res.groups[0].results[0].id == "pictograms.saxophone" and not res.groups[0].fallback
    assert any("薩克斯風" in e.keywords for e in es if e.name == "saxophone")
    assert catalog.find("薩克斯風").groups[0].results[0].id == "pictograms.saxophone"


def test_pictograms_are_first_class_when_importable(monkeypatch):
    import types
    mod = types.ModuleType("pixelcanvas.pictograms")

    def names():
        return ["saxophone", "piano"]

    def find(q):
        return [n for n in names() if q.split()[0] in n]

    def draw(cv, name, at=(0.5, 0.5), size=0.3):
        """Draw one pictogram."""
        return name

    for f in (names, find, draw):
        f.__module__ = "pixelcanvas.pictograms"
        setattr(mod, f.__name__, f)
    monkeypatch.setitem(sys.modules, "pixelcanvas.pictograms", mod)
    entries = catalog.build(kits=["pictograms", "vector", "clay"])
    assert {e.id for e in entries if e.kit == "pictograms"} >= {"pictograms.saxophone", "pictograms.piano"}
    monkeypatch.setattr(catalog, "_INDEX", entries)
    monkeypatch.setattr(catalog, "_VOCAB", {})
    monkeypatch.setattr(catalog, "_TYPO_BY_FIRST", {})
    monkeypatch.setattr(catalog, "_NAME_TOKS", set())
    res = catalog.find("saxophone silhouette")
    assert res.groups[0].results[0].id == "pictograms.saxophone" and not res.groups[0].fallback
    fb = catalog.find("a trumpet").groups[0].fallback
    assert fb is not None and fb["nearest"][0].startswith("pictograms.pictogram(cv, \"piano\"")   # same class: the piano
    monkeypatch.undo()
    catalog.reset()


# ------------------------------------------------------------------------------------------------ names and text
def test_names_quotes_dates_prices_and_urls_are_content():
    q = ("A poster for a jazz night called 'Blue Hour' on Friday 21:00 at The Velvet Room, with the line-up "
         "'Ana Reyes Trio · Kofi & The Lanterns', tickets $15, 21 Nov 2026, see https://example.org/jazz")
    clean, items = catalog.extract_content(q)
    for want in ("Blue Hour", "Friday 21:00", "The Velvet Room", "Ana Reyes Trio · Kofi & The Lanterns", "$15",
                 "https://example.org/jazz"):
        assert want in items, (want, items)
    assert any("21 Nov 2026" in i for i in items)
    for gone in ("Velvet", "Lanterns", "Friday", "example"):
        assert gone not in clean
    res = catalog.find(q)
    assert res.text().splitlines()[0].startswith("text to draw: ")
    assert "texture.velvet" not in res.ids() and "form.room" not in res.ids()
    assert _tops(res)[0] == "layout.poster"


def test_single_capitalised_nouns_and_sentence_starts_stay_subjects():
    clean, items = catalog.extract_content("A Japanese garden with Mount Fuji behind it")
    assert items == [] and "Mount Fuji" in clean and "Japanese" in clean
    clean, items = catalog.extract_content("A cat in Taipei")
    assert items == []
    assert "clay.cat" in _tops(catalog.find("Cat and dog"))


def test_the_velvet_room_is_not_a_room():
    res = catalog.find("a jazz night at The Velvet Room")
    assert "form.room" not in _ids(res) and "texture.velvet" not in _ids(res)
    assert res.content == ["The Velvet Room"]


# ------------------------------------------------------------------------------------------------ kind = SKILL.md
def test_kit_kinds_are_read_from_skill_md():
    table = catalog.kit_kinds()
    assert table["creature"] == ["animals"] and table["figure"][0] == "people"
    assert table["solid"] == ["product", "scene3d", "isometric"]
    assert table["ui"] == ["ui"] and table["drafting"] == ["drawing", "floorplan"]


def _with_table(tmp_path, monkeypatch, row, new_row):
    with open(os.path.join(ROOT, "skills", "pixelcanvas", "SKILL.md"), encoding="utf-8") as f:
        text = f.read()
    assert row in text
    page = tmp_path / "SKILL.md"
    page.write_text(text.replace(row, new_row), encoding="utf-8")
    monkeypatch.setattr(catalog, "_SKILL_PAGE", str(page))
    monkeypatch.setattr(catalog, "_KIT_TABLE", {})


def test_kind_follows_the_skill_table_not_a_copy(tmp_path, monkeypatch):
    row = "| animals | `creature` | [creatures.md](references/creatures.md) | `animals` |"
    _with_table(tmp_path, monkeypatch, row, row.replace("`animals` |", "`people` |"))
    assert catalog.find("a flat cartoon cat on a windowsill").kit_kind == "people"


def test_clay_kind_follows_the_skill_table_too(tmp_path, monkeypatch):
    row = "| `clay` | [clay.md](references/clay.md) | `characters`, `pets`, `scene3d` |"
    with open(os.path.join(ROOT, "skills", "pixelcanvas", "SKILL.md"), encoding="utf-8") as f:
        line = next(ln for ln in f.read().splitlines() if row in ln)
    _with_table(tmp_path, monkeypatch, line, line.replace("`characters`, `pets`, `scene3d`",
                                                          "`toys`, `scene3d`"))
    assert catalog.kit_kinds()["clay"] == ["toys", "scene3d"]
    assert catalog.find("a toy cat").kind == "scene3d"           # no "pets" row any more: the row's 3D kind


@pytest.mark.parametrize("query, kind", [
    ("a cat watching a bird on a branch", "pets"),
    ("people waiting at a bus stop", "characters"),
    ("flat vector people waiting at a bus stop", "people"),
    ("a product on a table", "product"),
    ("a cat on a windowsill", "pets"),
    ("a flat cat on a windowsill", "animals"),
    ("a horse on a beach at sunrise", "pets"),
    ("a beach with a horse", "landscape"),
    ("a chef in a kitchen", "characters"),
    ("a corgi under an oak tree", "pets"),
    ("an oak tree with a corgi", "nature"),
])
def test_kind_is_the_main_subjects_row(query, kind):
    assert catalog.find(query).kit_kind == kind


def test_two_prominent_kinds_name_the_primary_and_add_the_other():
    res = catalog.find("a flat illustration of a ginger cat sitting under a birch tree with a girl")
    assert res.kit_kind == "animals"
    assert set(res.extras) == {"flora", "figure"}
    line = [ln for ln in res.text(kits=True).splitlines() if "suggested kind" in ln][0]
    assert "--kind animals" in line and "calls inside it" in line
    res = catalog.find("a ginger cat sitting under a birch tree with a girl")      # clay: the cat's row
    assert res.kit_kind == "pets" and res.extras == ["flora"]


# ------------------------------------------------------------------------------------------------ one request, one call
COLD = [
    "Make a poster for a jazz night called 'Blue Hour' on Friday 21:00 at The Velvet Room, with a saxophone "
    "silhouette and a deep blue and gold palette.",
    "An architecture diagram for a real-time chat product: web and mobile clients, a load balancer, an API gateway, "
    "auth, messaging and presence services, a Redis pub/sub bus, Postgres and object storage. Dark theme, grouped "
    "into tiers.",
    "A ginger cat sitting on a windowsill watching a blue bird on a branch outside, afternoon light, in watercolour.",
]


def test_cold_read_requests():
    jazz, arch, cat = (catalog.find(q) for q in COLD)
    assert jazz.kind == "poster" and jazz.content[:3] == ["Blue Hour", "Friday 21:00", "The Velvet Room"]
    assert jazz.groups[0].results[0].id == "layout.poster"
    sax = next(g for g in jazz.groups if g.head == "saxophone")
    assert sax.results[0].id == "pictograms.saxophone"                    # a design: the pictogram first
    assert arch.kind == "diagram" and arch.main.results[0].id == "diagram.architecture_diagram"
    assert not any(g.fallback for g in arch.groups), "a pub/sub bus is not a vehicle"
    assert cat.kind == "illustration" and cat.kit_kind == "animals" and cat.style == "watercolor"
    assert [t for t in _tops(cat) if not t.startswith("pictograms.")][:3] == ["creature.cat", "creature.sparrow",
                                                                               "flora.branch_frame"]
    assert 'pose="sit"' in cat.groups[0].calls[0]
    assert cat.extras == ["flora"]
    for r in (jazz, arch, cat):
        assert len(r.text().splitlines()) <= 30


def test_counts_and_colours_attach_to_their_subject():
    res = catalog.find("three red oranges on a plate and a blue jug")
    g = res.groups[0]
    assert g.count == 3 and g.results[0].id.startswith("form.")
    line = res.text(kits=True).splitlines()[0]
    assert line.startswith("oranges x3") and "# x3: one call per copy" in line
    assert "oranges x3" in res.text()                              # freehand: the count stays with the subject
    text = catalog.find("three flat vector friends and a dog").text(kits=True)
    assert "for p in form.row(3, " in text and "at=p" in text
    assert "creature.creature(cv, \"dog\"" in text
    text = catalog.find("three friends and a dog").text(kits=True)    # clay: three varied people side by side
    assert "for i in range(3): sc.add(clay.random_person(seed=i + 1" in text and 'clay.animal("dog")' in text


def test_subjects_in_order_each_once():
    res = catalog.find("a fox and a cat, then a cat and a dog and an oak")
    tops = _tops(res)
    assert tops.index("clay.fox") < tops.index("clay.cat") < tops.index("clay.dog")
    assert len(tops) == len(set(tops))


def test_a_forty_word_request_is_one_tidy_answer():
    q = ("A cosy kitchen in the morning with a ginger cat on the table beside a blue bowl of oranges, a window with a "
         "potted fern, a wooden shelf of jars, a red kettle, a bouquet of tulips, steam rising, soft light, in "
         "watercolour, 1920x1080, text 'Good Morning' at the top")
    assert len(q.split()) >= 40
    res = catalog.find(q)
    lines = res.text().splitlines()
    assert len(lines) <= 30
    assert "Good Morning" in res.content and lines[0].startswith("draw it freehand")
    assert any(ln.startswith("-> suggested kind") for ln in lines)


def test_find_is_under_a_tenth_of_a_second_per_request():
    catalog.find("warm up")
    qs = [r[0] for r in REQUESTS]
    t = time.time()
    for q in qs:
        catalog.find(q)
    assert (time.time() - t) / len(qs) < 0.1


# ------------------------------------------------------------------------------------------------ dogfood round 1
def test_a_head_noun_finds_its_kinds():
    """'a boat' has no entry of its own: the sailboat, rowing boat, motorboat pictograms are kinds of boat."""
    res = catalog.find("two fishing boats on a great wave")
    boat = res.groups[0]
    assert boat.results and boat.results[0].kit == "pictograms" and "boat" in boat.results[0].name
    assert 'style="shaded"' in boat.calls[0] and "no catalogue entry" not in res.text()
    kinds = {e.name for e in catalog.find("a boat").groups[0].results}
    assert kinds & {"sailboat", "rowing boat", "motorboat", "fishing boat", "canoe"}
    assert catalog._head_noun_entries("corn") == [] or all("unicorn" != e.name for e in catalog._head_noun_entries("corn"))


@pytest.mark.parametrize("query, head, want", [
    ("a weather app icon", "weather", {"pictograms.sun", "pictograms.cloud", "pictograms.moon_behind_cloud"}),
    ("a tasks app icon", "task", {"pictograms.checklist", "pictograms.clipboard"}),
    ("a todo list app icon", "todo", {"pictograms.checklist", "pictograms.clipboard"}),
])
def test_an_icon_subject_without_an_icon_takes_its_relatives(query, head, want):
    res = catalog.find(query)
    g = next(g for g in res.groups if g.head == head and not getattr(g, "format", False))
    assert {e.id for e in g.results} & want, g
    assert not any(e.kind == "ui template" for e in g.results)            # a weather icon is no weather app


def test_vegetables_get_vegetable_stand_ins():
    res = catalog.find("a farmers market flyer with illustrated vegetables")
    veg = next(g for g in res.groups if g.head == "vegetable")
    calls = veg.fallback["standins"] if veg.fallback else veg.calls
    assert any('"carrot"' in c or '"eggplant"' in c or '"tomato"' in c for c in calls)
    assert all(e not in res.ids() for e in ("pictograms.factory", "pictograms.ant", "pictograms.taxi"))


def test_the_kind_follows_the_format_words():
    assert catalog.find("a labelled diagram of a plant cell").kind == "figure"
    assert catalog.find("a plant cell for a biology textbook").kind == "figure"
    kinds = catalog._scene_kinds()
    want = "banner" if "banner" in kinds else "card"
    for q in ("A README banner for a tool called 'driftwood'", "a GitHub header image for my project"):
        assert catalog.find(q).kind == want, q
    assert catalog.find("a landing page with a hero section for a bakery").kind == "ui"
    assert catalog.find("a YouTube thumbnail titled 'I quit'").kind == "card"


def test_banner_falls_back_to_card_without_its_template(monkeypatch):
    kinds = dict(catalog._scene_kinds())
    kinds.pop("banner", None)
    monkeypatch.setattr(catalog, "_scene_kinds", lambda: kinds)
    assert catalog.find("A README banner for a tool called 'driftwood'").kind == "card"


def test_a_3d_room_gets_3d_furniture():
    res = catalog.find("an isometric cosy room with an armchair, a floor lamp and a round rug")
    lamp = next(g for g in res.groups if g.head == "lamp")
    assert lamp.results[0].kit == "solid", lamp                             # not the 2D lamp effect
    arm = next(g for g in res.groups if g.head == "armchair")
    assert arm.results[0].id != "solid.chair"                              # not a dining chair
    assert arm.results[0].id in ("solid.armchair", "pictograms.armchair")
    if arm.results[0].kit == "pictograms":
        assert 'style="shaded"' in arm.calls[0] and any(e.kit == "solid" for e in arm.results[1:])


@pytest.mark.parametrize("query, kit", [
    ("a chef holding a pot", "clay"),
    ("a storybook illustration of a fox", "clay"),
    ("a cute penguin with an umbrella", "clay"),
    ("a mascot of a smiling dog", "clay"),
    ("a girl reading under a tree, soft shading", "clay"),
    ("a 3D astronaut waving", "clay"),
    ("a flat vector illustration of a chef", "figure"),
    ("an anime girl with long blue hair", "figure"),
    ("a comic strip of a superhero", "figure"),
    ("a line-art portrait of a woman", "figure"),
    ("a textbook figure of a horse skeleton", None),
    ("a flat illustration of a fox", "creature"),
    ("a watercolour painting of a cat", "creature"),
])
def test_people_and_animals_go_to_clay_unless_flat(query, kit):
    res = catalog.find(query)
    if kit is None:
        assert res.kind == "figure"
        return
    main = next(g for g in res.groups if g.results and catalog._sem(g) in ("person", "animal"))
    assert main.results[0].kit == kit, (query, main)
    assert res.kit_kind in (("characters", "pets") if kit == "clay" else ("people", "animals", "comic", "styled")), \
        res.kit_kind
    assert res.kind == ("characters" if "3D" in query else "comic" if "strip" in query else "illustration"), res.kind


SNIPPET_REQUESTS = [
    "two fishing boats on a great wave under Mount Fuji", "a blue bird on a cherry branch",
    "a storybook illustration of a fox reading a book under a tree",
    "a flat vector illustration of a woman riding a bicycle", "a cute clay mascot of a penguin holding an umbrella",
    "an isometric cosy room with an armchair, a floor lamp and a round rug",
    "a product shot of wireless earbuds in a charging case",
    "a poster for a jazz night titled 'Blue Notes' with a saxophone",
    "a README banner for a command-line tool called 'driftwood'",
    "a labelled diagram of a plant cell for a biology textbook",
    "an architecture diagram of a chat app with a load balancer, two API servers and a Redis bus",
    "a line chart of monthly revenue for 12 months", "an analytics dashboard with four KPI tiles and a bar chart",
    "a sheet of twelve app icons for notes, tasks, weather and calendar",
    "a farmers market flyer with illustrated vegetables", "a campfire beside a tent at a mountain lake at night",
    "a Japanese garden with a stone lantern, an arched bridge and koi", "a gramophone on a wooden table",
    "a chef holding a steaming pot in a kitchen", "three friends waiting at a bus stop in the rain",
    "a ginger cat sleeping on a windowsill", "a watercolour painting of a village street with a cat",
    "a pixel-art wizard's tower on a cliff with a dragon", "a still life of three lemons in a ceramic bowl",
    "an anime girl with long blue hair waving", "a toy robot and a teddy bear on a shelf",
    "a lighthouse on a cliff at sunset with seagulls", "a mobile app login screen for a meditation app",
    "an engineering drawing of a flange with four bolt holes", "a phone wallpaper with a starry sky and a whale",
]
_CODE = re.compile(r"\s{2,}(?:flat mark: )?((?:for |sc\.|cv\.|X, Y|d = |m = |pts = |body = |manifest\.|"
                   r"[a-z_]+\.[A-Za-z_]+[.(]).*)$")


def _printed_code(text):
    out = []
    for ln in text.splitlines():
        if ln.startswith(("text to draw:", "-> ", "full snippet:", "colour ", "style ")):
            continue
        m = _CODE.search(" " + ln)
        if m:
            out.append(m.group(1).split("    # ")[0])
    return out


def test_every_printed_snippet_runs():
    """What find prints runs as printed, in order, with a canvas cv and a solid Scene sc (30 varied requests)."""
    import numpy as np
    from pixelcanvas import Canvas
    assert len(SNIPPET_REQUESTS) == 30
    bad, ran = [], 0
    for q in SNIPPET_REQUESTS:
        ns = {"np": np, "Canvas": Canvas}
        for k in catalog._ALL_KITS + ["manifest", "masks", "brush"]:
            try:
                ns[k] = importlib.import_module("pixelcanvas." + k)
            except Exception:
                pass
        ns["cv"] = Canvas(160, 96)
        ns["sc"] = ns["solid"].Scene()
        for code in _printed_code(catalog.find(q).text()):
            ran += 1
            try:
                exec(code, ns)
            except Exception as exc:                                    # noqa: BLE001 - collected and reported
                bad.append((q, code[:120], f"{type(exc).__name__}: {exc}"[:160]))
    assert ran >= 60 and not bad, bad


FORMAT_SNIPPET_REQUESTS = [
    "A flowchart of a password reset: request link, check email, verify token, done",
    "A sequence diagram of an OAuth login between browser, app server and identity provider",
    "A timeline of the product roadmap for 2027: beta in Q1, launch in Q2, mobile app in Q3",
    "An org chart of a bakery: owner, head baker, two apprentices",
    "A bar chart of monthly rainfall in Lisbon over 12 months, titled 'Rainy winters'",
    "A dark sales dashboard with KPI tiles for revenue, active users and churn and a donut of plans",
    "An infographic on coffee consumption by country with a donut of bean types",
    "A light weather app home screen on a phone for an app called Skyline",
    "Eight app icons for a fitness brand: run, swim, bike, yoga, timer, heart rate, water, sleep",
    "A logo for a bakery called Crumb",
    "A technical drawing of a flange 120 x 80 x 40 mm with a 30 mm bore and four 8 mm holes",
    "A floor plan of a two-bedroom apartment, 9 x 7 m, with a kitchen and a bathroom, at 1:50",
    "A calm phone wallpaper of misty mountains in teal and peach",
    "A four-panel comic about a robot learning to cook: 'What is salt?', 'Too much salt.'",
    "A labelled diagram of a flower for a biology textbook showing the petal, sepal, stamen and stem",
]


def test_every_printed_snippet_runs_for_format_requests():
    """Diagrams, charts, dashboards, UI, icons, drawings, plans, wallpapers, comics, figures: what find prints runs."""
    import numpy as np
    from pixelcanvas import Canvas
    bad, ran = [], 0
    for q in FORMAT_SNIPPET_REQUESTS:
        res = catalog.find(q)
        assert res.kind in ("diagram", "chart", "dashboard", "infographic", "ui", "icons", "drawing", "floorplan",
                            "wallpaper", "comic", "figure"), (q, res.kind)
        ns = {"np": np, "Canvas": Canvas}
        for k in catalog._ALL_KITS + ["manifest", "masks", "brush"]:
            try:
                ns[k] = importlib.import_module("pixelcanvas." + k)
            except Exception:
                pass
        ns["cv"] = Canvas(160, 96)
        ns["sc"] = ns["solid"].Scene()
        for code in _printed_code(res.text()):
            ran += 1
            try:
                exec(code, ns)
            except Exception as exc:                                    # noqa: BLE001 - collected and reported
                bad.append((q, code[:120], f"{type(exc).__name__}: {exc}"[:160]))
    assert ran >= 15 and not bad, bad


@pytest.mark.parametrize("query, not_top", [
    ("a pencil case on a desk", "solid.charging_case"), ("a pillow case", "solid.charging_case"),
    ("a paper lantern", "solid.stone_lantern"), ("a watch tower", "solid.smartwatch"),
    ("a pocket watch", "solid.smartwatch"), ("a phone booth", "solid.smartphone"),
    ("a fishing tent", None), ("a bus stop", "solid.tent"),
])
def test_a_compound_does_not_match_another_compound_by_its_head(query, not_top):
    res = catalog.find(query)
    g = res.main
    ids = [e.id for e in g.results] if g and g.results else []
    assert not ids or ids[0] != not_top, (query, ids)
    if query in ("a pencil case on a desk", "a pillow case"):
        assert g.fallback or ids[0].startswith("pictograms."), (query, ids)


@pytest.mark.parametrize("query, top", [
    ("wireless earbuds on a table", "solid.earbuds"), ("a 3D smartwatch", "solid.smartwatch"),
    ("a tent in a forest clearing", "solid.tent"), ("a stone lantern in a Japanese garden", "solid.stone_lantern"),
    ("an arched bridge over a pond", "solid.arched_bridge"), ("an isometric room with a floor lamp", "solid.floor_lamp"),
    ("a side table", "solid.side_table"), ("a reading nook with an armchair", "solid.armchair"),
])
def test_new_prefabs_are_found(query, top):
    res = catalog.find(query)
    tops = [g.results[0].id for g in res.groups if g.results]
    assert top in tops, (query, tops)


@pytest.mark.parametrize("query, kit", [
    ("a flat icon of a tent", "pictograms"), ("a tent sticker", "pictograms"), ("a smartphone app icon", "pictograms"),
])
def test_flat_requests_keep_the_flat_mark(query, kit):
    res = catalog.find(query)
    g = next(g for g in res.groups if g.results and not getattr(g, "format", False))
    assert g.results[0].kit == kit, (query, [e.id for e in g.results])


def test_every_solid_prefab_is_indexed_from_the_registry(monkeypatch):
    from pixelcanvas import solid
    names = {e.name for e in catalog.entries("solid") if e.kind == "prefab"}
    for n in solid.names():
        if not catalog._needs(getattr(solid, n), skip=0):
            assert n in names, n
    fake = lambda height=1.0: solid.sphere(0.5)                         # noqa: E731 - a prefab added today
    fake.__doc__ = "Hot-air balloon with a woven basket."
    monkeypatch.setattr(solid, "hot_air_balloon", fake, raising=False)
    monkeypatch.setitem(solid.PREFABS, "hot_air_balloon", ("vehicle", "balloon aerostat"))
    monkeypatch.setattr(catalog, "_INDEX", None)
    try:
        assert "hot_air_balloon" in {e.name for e in catalog.entries("solid")}
        assert any(e.id == "solid.hot_air_balloon" for g in catalog.find("a 3D hot air balloon").groups
                   for e in g.results)
    finally:
        monkeypatch.undo()
        catalog.reset()


def test_colours_go_inside_the_call_in_a_form_the_kit_takes():
    res = catalog.find("a blue bird on a branch")
    bird = res.groups[0]
    creature_call = next(c for e, c in zip(bird.results, bird.calls) if e.kit == "creature")
    assert 'coat="#' in creature_call
    clay_call = next(c for e, c in zip(bird.results, bird.calls) if e.kit == "clay")
    assert 'color="#' in clay_call


def test_stand_ins_come_in_order_with_a_recipe_and_its_section():
    res = catalog.find("a gramophone on a table")
    fb = res.groups[0].fallback
    assert fb["standins"][0].startswith('pictograms.pictogram(cv, "turntable"') and 'style="shaded"' in fb["standins"][0]
    assert any(c.startswith("sc.add(solid.") for c in fb["standins"])
    text = res.text(kits=True)
    assert "own shape:" in text and "read: references/techniques.md" in text and "no catalogue entry" not in text
    assert '"turntable"' in res.text()                             # freehand: the nearest relative as an ingredient


def test_find_prints_on_a_legacy_windows_console():
    env = dict(os.environ, PYTHONIOENCODING="cp950")
    out = subprocess.run([sys.executable, KIT, "find", "海報 for a café ★ with a boat"], cwd=ROOT,
                         capture_output=True, env=env)
    assert out.returncode == 0, out.stderr.decode("utf-8", "replace")
    assert "pictograms.pictogram" in out.stdout.decode("utf-8")


def test_wrong_kit_names_get_a_did_you_mean():
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for args in (["list", "flroa"], ["doc", "flroa"]):
        out = subprocess.run([sys.executable, KIT, *args], cwd=ROOT, capture_output=True, text=True,
                             encoding="utf-8", env=env)
        assert out.returncode == 1 and "did you mean 'flora'" in (out.stdout + out.stderr), args
