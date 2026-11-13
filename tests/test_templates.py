# -*- coding: utf-8 -*-
"""Starter templates (scenes/_template_<kind>.py) and tools/new_scene.py."""
import glob
import json
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)
import new_scene  # noqa: E402
from pixelcanvas import catalog  # noqa: E402

KINDS = sorted(os.path.basename(p)[len("_template_"):-3]
               for p in glob.glob(os.path.join(ROOT, "scenes", "_template_*.py")))
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")


def _run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=ENV, timeout=600)


def _tmp_name(kind):
    return f"zz_tmpl_{kind}_{os.getpid()}"


@pytest.fixture
def scene_name(request):
    created = []

    def make(kind):
        name = _tmp_name(kind)
        created.append(name)
        return name

    yield make
    for name in created:
        for p in (os.path.join(ROOT, "scenes", f"{name}.py"),):
            if os.path.exists(p):
                os.remove(p)
        for p in glob.glob(os.path.join(ROOT, "outputs", f"{name}*")):
            if os.path.isfile(p):
                os.remove(p)


def test_every_requested_kind_exists():
    want = {"landscape", "nature", "still_life", "interior", "product", "isometric", "scene3d", "poster", "card",
            "diagram", "infographic", "chart", "dashboard", "figure", "drawing", "floorplan", "icons", "styled",
            "photo", "animation", "blank", "people", "animals", "ui", "wallpaper", "comic"}
    assert want <= set(KINDS)
    assert set(new_scene.kinds()) == set(KINDS)


@pytest.mark.parametrize("kind", KINDS)
def test_template_conventions(kind):
    path = os.path.join(ROOT, "scenes", f"_template_{kind}.py")
    src = open(path, encoding="utf-8").read()
    lines = src.rstrip("\n").split("\n")
    assert 35 <= len(lines) <= 70, f"{kind}: {len(lines)} lines"
    assert not re.search("[⺀-鿿]", src), f"{kind}: CJK characters in a template"
    doc = re.search(r'"""(.*?)"""', src, re.S).group(1).strip().split("\n")
    assert doc[0] and 6 <= len([l for l in doc if l.strip()]) <= 8, f"{kind}: plan docstring"
    assert re.search(r"^# Read skills/pixelcanvas/references/\w+\.md", src, re.M), f"{kind}: reference comment"
    assert re.search(r'^PROMPT = "" +# paste the request here, verbatim', src, re.M)
    assert re.search(r"^NAME = \"my_\w+\"", src, re.M) and re.search(r"^DEFAULT_SIZE = \(\d+, \d+\)", src, re.M)
    assert "def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):" in src
    m = re.search(r'^KIND = "(\w+)"', src, re.M)
    assert m is None or m.group(1) in ("design", "diagram", "drawing")
    names = [n for group in re.findall(r"^([A-Z][A-Z0-9_]*(?:, [A-Z][A-Z0-9_]*)*) = ", src, re.M)
             for n in group.split(", ")]
    tunables = [n for n in names if n not in ("PROMPT", "KIND", "NAME", "DEFAULT_SIZE", "DEFAULT_SEED")]
    assert 4 <= len(tunables) <= 8, f"{kind}: {tunables}"


@pytest.mark.parametrize("kind", KINDS)
def test_kind_creates_and_renders(kind, scene_name):
    name = scene_name(kind)
    r = _run("tools/new_scene.py", name, "--kind", kind)
    assert r.returncode == 0, r.stderr
    assert f"python tools/render.py {name} --scale 0.5 --check" in r.stdout
    assert "skills/pixelcanvas/references/" in r.stdout
    src = open(os.path.join(ROOT, "scenes", f"{name}.py"), encoding="utf-8").read()
    assert f'NAME = "{name}"' in src
    r = _run("tools/render.py", name, "--scale", "0.25", "--check", "--json")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    data = json.loads(r.stdout.strip().splitlines()[-1])
    issues = [w for key in ("warnings", "text_issues", "object_issues", "prompt_issues")
              for w in (data.get(key) or [])]
    fails = [w for w in issues if w.get("severity") == "fail"]
    assert not fails, fails


def test_list_shows_every_kind_with_description():
    r = _run("tools/new_scene.py", "--list")
    assert r.returncode == 0, r.stderr
    for kind in KINDS:
        line = next((l for l in r.stdout.splitlines() if l.split()[:1] == [kind]), None)
        assert line, kind
        assert new_scene.first_doc_line(os.path.join(ROOT, "scenes", f"_template_{kind}.py")) in line
    assert r.stdout.isascii()


def test_unknown_kind_suggests_close_ones():
    r = _run("tools/new_scene.py", "zz_never_created", "--kind", "postr")
    assert r.returncode != 0 and "poster" in r.stderr
    r = _run("tools/new_scene.py", "zz_never_created", "--kind", "3d")
    assert r.returncode != 0 and "scene3d" in r.stderr
    assert not os.path.exists(os.path.join(ROOT, "scenes", "zz_never_created.py"))


@pytest.mark.parametrize("prompt", ['A poster titled "Blue Hour" with three lanterns',
                                    "it's a 'quoted' \\ backslash \"mix\"\nsecond line",
                                    "海報：「藍色時刻」，水彩風格 — café ★"])
def test_prompt_round_trip(prompt, scene_name):
    name = scene_name("poster")
    r = _run("tools/new_scene.py", name, "--kind", "poster", "--prompt", prompt, "--size", "1080x1920")
    assert r.returncode == 0, r.stderr
    path = os.path.join(ROOT, "scenes", f"{name}.py")
    ns = {}
    src = open(path, encoding="utf-8").read()
    exec(compile(src.split("\ndef render")[0], path, "exec"), ns)
    assert ns["PROMPT"] == prompt
    assert ns["DEFAULT_SIZE"] == (1080, 1920)
    assert ns["NAME"] == name


def test_poster_flag_and_existing_name(scene_name):
    name = scene_name("poster")
    r = _run("tools/new_scene.py", name, "--poster")
    assert r.returncode == 0 and "(kind: poster)" in r.stdout
    assert _run("tools/new_scene.py", name, "--kind", "blank").returncode != 0      # already exists
    assert _run("tools/new_scene.py", "zz_bad", "--size", "big").returncode != 0


# ------------------------------------------------------------------------------------------------ request-aware starters
@pytest.mark.parametrize("prompt, size", [
    ("a poster, 1080x1920, titled 'Hi'", (1080, 1920)), ("an Instagram story about tea", (1080, 1920)),
    ("a square sticker of a cat", (1080, 1080)), ("a YouTube thumbnail for my vlog", (1280, 720)),
    ("an A4 flyer for a bake sale", (2480, 3508)), ("a README banner for driftwood", (1280, 640)),
    ("a phone wallpaper of a whale", (1170, 2532)), ("a cat on a sofa", None),
])
def test_request_size(prompt, size):
    assert new_scene.request_size(prompt) == size


@pytest.mark.parametrize("prompt, setting", [
    ("three friends at a bus stop in the rain", "street"), ("a chef in a kitchen", "room"),
    ("a corgi having a picnic in a park", "park"), ("a crab on the beach", "beach"),
    ("a campfire at night under the stars", "night"), ("a cat sleeping on a windowsill", "room"),
    ("a robot", None),
])
def test_request_setting(prompt, setting):
    assert new_scene.request_setting(prompt) == setting


def test_every_marked_block_is_closed():
    for kind in KINDS:
        src = open(os.path.join(ROOT, "scenes", f"_template_{kind}.py"), encoding="utf-8").read()
        for name in ("setting", "subjects", "overlay"):
            assert src.count(f"# >>> {name}") == src.count(f"# <<< {name}"), (kind, name)
            assert src.count(f"# >>> {name}") <= 1, (kind, name)


STARTERS = [   # (request, kind of the kit route, words that must be drawn by a call of the starter)
    ("A YouTube thumbnail titled 'Sailing Solo' with a sailboat and a lighthouse", "card",
     ['"sailboat"', '"lighthouse"', "HEADLINE = \"Sailing Solo\""]),
    ("A flat vector illustration of a woman riding a bicycle through a park", "people",
     ['figure.person("woman", name="woman", height=', '"bicycle"', 'elements.sky(cv, "day"']),
    ("A cute corgi and a rabbit having a picnic in a park", "pets",
     ['clay.animal("corgi"', 'clay.animal("rabbit"', '"picnic_blanket"']),
    ("An isometric reading room with an armchair, a floor lamp, a bookshelf and a cat", "isometric",
     ["solid.bookshelf(", "solid.armchair(", "solid.floor_lamp(", 'clay.animal("cat"']),
    ("A still life of three lemons and a teapot on a wooden table", "still_life",
     ['form.ball(cv, (0.', '"teapot"']),
    ("Three friends waiting at a bus stop in the rain, storybook style", "characters",
     ["clay.random_person(seed=1", "clay.random_person(seed=3", "elements.rain(cv)", 'stylize.apply(cv, "storybook")']),
]


@pytest.mark.parametrize("prompt, kind, want", STARTERS, ids=[s[1] for s in STARTERS])
def test_prompt_starter_draws_the_request(prompt, kind, want, tmp_path):
    """The kit route: with --kind (the kit table's kind) the starter composes the kits' calls for the request."""
    assert catalog.find(prompt).kit_kind == kind
    r = _run("tools/new_scene.py", "starter", "--kind", kind, "--prompt", prompt, "--dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert f"(kind: {kind}" in r.stdout and "subject " in r.stdout
    src = (tmp_path / "starter.py").read_text(encoding="utf-8")
    for w in want:
        assert w in src, (w, src)
    body = src.split("def render", 1)[1]
    assert "<" not in re.sub(r"<<< \w+|[<>]=|\s<\s", "", body), "no placeholders in the code"
    assert "Chasing the Last Light" not in src and 'log("woman")\n    a = person' not in src
    r = _run("tools/render.py", str(tmp_path / "starter.py"), "--scale", "0.25", "--check", "--json")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    data = json.loads(r.stdout.strip().splitlines()[-1])
    fails = [w for key in ("warnings", "text_issues", "object_issues", "prompt_issues")
             for w in (data.get(key) or []) if w.get("severity") == "fail"]
    assert not fails, fails


# ------------------------------------------------------------------------------- format kinds write the request's data
FORMAT_STARTERS = [   # (request, kind, strings in the starter source, strings in the printed report)
    ("A system diagram for a photo-sharing service: an nginx proxy, three image workers, an S3 bucket and MySQL",
     "diagram", ['"Nginx proxy"', '"Image worker 3"', '"S3 bucket"', '"cylinder"'], ["nodes:", "MySQL"]),
    ("A flowchart for ordering coffee: choose a drink, pick a size, pay, collect the cup", "diagram",
     ['"Choose a drink"', '"Collect the cup"', '"pill"'], ["steps:", "Pay"]),
    ("A sequence diagram of a checkout between shopper, shop backend and payment gateway", "diagram",
     ["Sequence(", '"Payment gateway"', "MESSAGES"], ["participants:", "Shopper"]),
    ("A roadmap timeline for 2028: alpha in Q2, public beta in Q3, general release in Q4", "diagram",
     ["Timeline(", '"Q2 2028"', '"General release"'], ["milestones:"]),
    ("An org chart of a bakery: owner, head baker, two apprentices, cashier", "diagram",
     ["Tree(", '"Owner"', '"Apprentice 2"'], ["org:"]),
    ("A line chart comparing tea and coffee sales over 8 quarters", "chart",
     ['CHART = "line"', '"Tea"', '"Coffee"', '"Q1 Y1"'], ["chart    line"]),
    ("A donut chart of browser share", "chart", ['CHART = "donut"', '"Chrome"'], ["chart    donut"]),
    ("A light ops dashboard with KPI tiles for uptime, latency and error rate, a bar chart of deploys and a donut of "
     "regions", "dashboard", ['"Uptime"', '"Latency"', '"Error rate"', '"bar"', '"North"'], ["KPI tiles:"]),
    ("An infographic about recycling with a donut of materials and a bar chart of tonnes by city, steps: collect, "
     "sort, melt, reuse", "infographic", ['"Materials"', '"London"', '"Melt"'], ["steps: Collect"]),
    ("A dark chat screen for a messaging app called Parley on a phone", "ui",
     ['template("chat"', 'APP = "Parley"', 'THEME = "dark"'], ["screen   chat"]),
    ("A tablet home screen for a garden planner with plant cards, a watering button and a tasks list", "ui",
     ['DEVICE = "tablet"', 's.button("Watering"', 's.list_group("Tasks")'], ["drawn    Plant"]),
    ("App icons for a travel app: map, camera, plane, hotel, wallet, weather", "icons",
     ["app_icon_sheet", '"Hotel"', '"Wallet"'], ["app icons:"]),
    ("A set of icons for a recipe site: whisk, oven, timer, shopping list", "icons",
     ['"Whisk"', '"pictogram:', '"Timer"'], ["icons:"]),
    ("A logo for a cycling club called Spoke", "icons", ['TITLE = "Spoke"', "pictogram:"], ["logo mark"]),
    ("An engineering drawing of a mounting plate 150 x 100 x 12 mm with four 10 mm holes and a 3 mm chamfer",
     "drawing", ["LENGTH, HEIGHT, DEPTH = 150, 100, 12", "HOLES = (4, 10)", "CHAMFER = 3"], ["4 holes", "MOUNTING PLATE"]),
    ("A floor plan of a studio flat 6 x 5 m with a kitchen, a bathroom and a living room", "floorplan",
     ["WIDTH, DEPTH = 6000, 5000", '"KITCHEN"', '"BATHROOM"'], ["3 rooms", "30.0 m²"]),
    ("A dark desktop wallpaper of a starry sky in navy and gold", "wallpaper",
     ['BACKDROP = "starry"', "navy and gold"], ["starry backdrop"]),
    ("A three-panel comic about a cat and a cucumber: 'What is that?', 'It moved!', 'Never again.'", "comic",
     ['"What is that?"', '"Never again."', 'CAPTION = "A cat and a cucumber"'], ["3 panels"]),
    ("A labelled diagram of a volcano for a geography textbook showing the crater, vent, magma chamber and ash cloud",
     "figure", ['"Crater"', '"Magma chamber"', '"Ash cloud"'], ["callouts:"]),
]


@pytest.mark.parametrize("prompt, kind, src_has, report_has", FORMAT_STARTERS,
                         ids=[f"{s[1]}-{i}" for i, s in enumerate(FORMAT_STARTERS)])
def test_format_starter_writes_the_request(prompt, kind, src_has, report_has, tmp_path):
    r = _run("tools/new_scene.py", "starter", "--prompt", prompt, "--dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert f"(kind: {kind}" in r.stdout, r.stdout
    src = (tmp_path / "starter.py").read_text(encoding="utf-8")
    for w in src_has:
        assert w in src, (w, src)
    for w in report_has:
        assert w in r.stdout, (w, r.stdout)
    r = _run("tools/render.py", str(tmp_path / "starter.py"), "--scale", "0.25", "--check", "--json")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    data = json.loads(r.stdout.strip().splitlines()[-1])
    fails = [w for key in ("warnings", "text_issues", "object_issues", "prompt_issues")
             for w in (data.get(key) or []) if w.get("severity") == "fail"]
    assert not fails, fails


# ------------------------------------------------------------------------------- placement of picture starters
def _starter(prompt, tmp_path, name="s"):
    """The kit-route starter of a request (its kit table kind passed with --kind; pictures are freehand by default)."""
    kind = catalog.find(prompt).kit_kind
    r = _run("tools/new_scene.py", name, "--prompt", prompt, "--dir", str(tmp_path),
             *(("--kind", kind) if kind else ()))
    assert r.returncode == 0, r.stderr
    return (tmp_path / f"{name}.py").read_text(encoding="utf-8"), r.stdout


def _spots(src, word):
    """[(u, v, size)] of the 2D calls drawing ``word``."""
    out = []
    for ln in src.splitlines():
        if word in ln:
            m = re.search(r"at=\(([\d.]+), ([\d.]+)\), (?:size|height)=([\d.]+)", ln)
            if m:
                out.append(tuple(float(x) for x in m.groups()))
    return out


def test_boats_float_birds_fly_and_the_lighthouse_stands_on_the_far_shore(tmp_path):
    src, out = _starter("Two fishing boats on a calm sea at dusk with a lighthouse and gulls", tmp_path)
    horizon = float(re.search(r"^HORIZON = ([\d.]+)", src, re.M).group(1))
    boats, light, birds = _spots(src, '"fishing boat"'), _spots(src, '"lighthouse"'), _spots(src, '"bird"')
    assert len(boats) == 2 and all(horizon < v < 0.92 for _u, v, _s in boats), boats
    assert abs(boats[0][1] - boats[1][1]) > 0.02 and abs(boats[0][0] - boats[1][0]) > 0.1   # two depths, apart
    assert light and abs(light[0][1] - horizon) < 0.03 and light[0][2] < max(s for *_x, s in boats)
    assert birds and all(v < horizon - 0.05 for _u, v, _s in birds)
    assert "on the water" in out and "on the far shore" in out and "in the sky" in out


def test_sizes_follow_real_sizes(tmp_path):
    src, _out = _starter("A flat vector picture of a man, a dog and a kitten in a garden", tmp_path)
    man = re.search(r"figure\.person\([^\n]*height=([\d.]+)", src)
    dog, kitten = _spots(src, '"dog"'), _spots(src, '"kitten"') or _spots(src, '"cat"')
    assert man and dog and kitten
    assert kitten[0][2] < dog[0][2] < float(man.group(1)), (kitten, dog, man.group(1))


def test_an_owl_in_a_tree_sits_in_the_tree(tmp_path):
    src, out = _starter("A flat illustration of an owl in an oak tree and a fox in a meadow", tmp_path)
    tree, owl = _spots(src, '"oak"'), _spots(src, '"owl"')
    assert tree and owl and "in the oak" in out
    tu, tv, ts = tree[0]
    ou, ov, os_ = owl[0]
    assert tv - ts < ov < tv and abs(ou - tu) < 0.15 and os_ < 0.3 * ts


def test_room_furniture_stands_along_the_walls_and_the_pet_in_the_middle(tmp_path):
    src, out = _starter("An isometric living room with a sofa, a bookshelf, a floor lamp and a dog", tmp_path)
    room = float(re.search(r"^ROOM = ([\d.]+)", src, re.M).group(1)) / 2

    def at(name):
        m = re.search(r"solid\." + name + r"\(\)(\.rotate_y\(([-\d.]+)\))?\.place\(x=([-\d.]+), z=([-\d.]+)\)", src)
        assert m, (name, src)
        return float(m.group(3)), float(m.group(4)), m.group(2)
    sx, sz, srot = at("sofa")
    bx, bz, _ = at("bookshelf")
    lx, lz, _ = at("floor_lamp")
    assert sx < -room + 0.8 and srot == "90"                     # the sofa against the left wall, facing the room
    assert bz < -room + 0.5 and lz < -room + 0.5                  # the tall things at the back
    dog = re.search(r'clay\.animal\("dog"[^\n]*\.place\(x=([-\d.]+), z=([-\d.]+)\)', src)
    assert dog and float(dog.group(1)) > sx + 0.8 and float(dog.group(2)) > bz + 0.8   # in the clear centre


def test_requested_relations_use_on(tmp_path):
    src, out = _starter("An isometric study with a desk, a laptop on the desk, an armchair and a cat on the armchair",
                        tmp_path)
    assert re.search(r'clay\.animal\("cat"[^\n]*\.on\(armchair, "seat"\)', src), src
    assert re.search(r'solid\.laptop\([^\n]*\.on\(desk, "top"\)', src), src
    assert "on the armchair" in out


def test_a_2d_stand_in_in_a_3d_scene_stands_on_the_floor(tmp_path):
    src, out = _starter("A storybook picture of a boy and his dog by two fishing boats at the harbour", tmp_path)
    assert "sc.project((" in src and "at=(x0 / cv.w, y0 / cv.h)" in src
    boats = re.findall(r"sc\.project\(\(([-\d.]+), 0\.0, ([-\d.]+)\)\)", src)
    assert boats and all(float(z) <= -10 for _x, z in boats)       # out on the sea, not under the people's feet
    assert 'solid.box(60, 0.02, 30)' in src                         # the sea behind the beach


def test_stand_ins_record_under_the_request_word(tmp_path):
    for prompt, words in (("Two fishing boats on the sea at sunset", ['name="boat"']),
                          ("A still life of three lemons and a teapot on a table", ['name="lemon"', 'name="teapot"'])):
        src, _out = _starter(prompt, tmp_path, name=f"r{len(words)}")
        for w in words:
            assert w in src, (w, src)
    r = _run("tools/render.py", str(tmp_path / "r1.py"), "--scale", "0.25", "--check")
    cov = r.stdout.split("COVERAGE", 1)[-1]
    assert "COVERAGE" in r.stdout and not [ln for ln in cov.splitlines() if "boat" in ln and "not recorded" in ln], cov


def test_prompt_with_kind_keeps_the_kind_and_plain_keeps_the_example(tmp_path):
    r = _run("tools/new_scene.py", "a", "--kind", "card", "--prompt", "a card titled 'Hello' with a fox",
             "--dir", str(tmp_path))
    assert r.returncode == 0 and "(kind: card" in r.stdout
    src = (tmp_path / "a.py").read_text(encoding="utf-8")
    assert 'HEADLINE = "Hello"' in src and 'BADGE = ""' in src and '"mountain"' not in src
    r = _run("tools/new_scene.py", "b", "--kind", "card", "--prompt", "a card titled 'Hello'", "--plain",
             "--dir", str(tmp_path))
    assert r.returncode == 0
    assert '"mountain"' in (tmp_path / "b.py").read_text(encoding="utf-8")


def test_banner_kind_falls_back_to_card(monkeypatch, tmp_path, capsys):
    table = {k: v for k, v in new_scene.kinds().items() if k != "banner"}
    monkeypatch.setattr(new_scene, "kinds", lambda: table)
    assert new_scene.main(["c", "--kind", "banner", "--dir", str(tmp_path)]) == 0
    assert "(kind: card" in capsys.readouterr().out


def test_tools_print_on_a_legacy_windows_console(tmp_path):
    env = dict(os.environ, PYTHONIOENCODING="cp950")
    for args in (["tools/new_scene.py", "--list"], ["tools/pc.py", "version"],
                 ["tools/new_scene.py", "zz", "--prompt", "一隻貓 on a café sofa ★ 'Bonjour ☕'", "--dir", str(tmp_path)],
                 ["tools/kit.py", "find", "海報 ★ café"]):
        r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, env=env, timeout=600)
        assert r.returncode == 0, (args, r.stderr.decode("utf-8", "replace")[-800:])


# ------------------------------------------------------------------------------- the freehand route
ROUTES = [   # (request, kind new_scene and kit.py find pick)
    ("a hedgehog knitting a scarf by the window", "illustration"),                 # an animal
    ("my grandparents dancing in their kitchen", "illustration"),                  # people
    ("a lighthouse keeper's cottage on a stormy evening", "illustration"),         # a scene
    ("a woodblock print of a heron in the reeds", "illustration"),                 # a named style
    ("a heron in the reeds, pixel art", "illustration"),
    ("a paper cut scene of a fox den under the snow", "illustration"),
    ("a sticker sheet of six cheerful vegetables", "illustration"),
    ("poster art for a mountain film festival with a lone climber", "illustration"),
    ("an anime-style portrait of a girl with a red umbrella", "illustration"),
    ("a clay figurine of a hedgehog knitting", "pets"),                            # 3D asked for
    ("a 3D render of a robot barista", "characters"),
    ("an isometric bakery with an oven and a counter", "isometric"),
    ("a product shot of a ceramic teapot on a plinth", "product"),
    ("a bar chart of rainfall per month", "chart"),                                # structure stays
    ("a flowchart of a library book loan", "diagram"),
    ("a settings screen for a podcast app", "ui"),
    ("a four-panel comic strip about a cat and a laser pointer", "comic"),
    ("a poster titled 'Night Market' for Friday at 8 pm", "poster"),
]


@pytest.mark.parametrize("prompt, kind", ROUTES, ids=[r[0][:32] for r in ROUTES])
def test_requests_route_freehand_unless_3d_or_structure(prompt, kind):
    res = catalog.find(prompt)
    assert res.kind == kind, (prompt, res.kind, res.kit_kind)
    text = res.text()
    if kind == "illustration":
        first = text.splitlines()[0]
        assert first.startswith("draw it freehand") and "python tools/new_scene.py NAME --prompt " in first
        assert prompt in first and len(text.splitlines()) <= 2 + len(res.groups)          # short: skippable
        assert "no ingredient" not in text and "text to draw" not in text
        assert text.splitlines()[-1].startswith("-> suggested kind: python tools/new_scene.py NAME --kind illustration")
        assert "stylize.apply" not in text and "instead" not in text
        assert all("# " in ln for ln in text.splitlines() if "pictograms." in ln or "sc.add(" in ln)
    else:
        assert "draw it freehand" not in text


def test_freehand_find_lists_subjects_with_optional_ingredients_that_run():
    import importlib
    import numpy as np
    from pixelcanvas import Canvas
    res = catalog.find("a fox under a mushroom in the rain, in watercolour")
    lines = res.text().splitlines()
    assert lines[0].startswith("draw it freehand") and lines[-1].startswith("-> suggested kind:")
    assert any(ln.lstrip().startswith("fox") for ln in lines) and any("elements.rain(cv)" in ln for ln in lines)
    assert all("# optional " in ln for ln in lines[1:-1])                  # one optional ingredient per subject
    bare = catalog.find("a hedgehog knitting a scarf").text().splitlines()   # nothing in the library: two lines
    assert len(bare) == 2 and bare[-1].startswith("-> suggested kind:")
    ns = {"np": np, "Canvas": Canvas, "cv": Canvas(160, 96)}
    for k in catalog._ALL_KITS:
        try:
            ns[k] = importlib.import_module("pixelcanvas." + k)
        except Exception:
            pass
    ns["sc"] = ns["solid"].Scene()
    ran = 0
    for ln in lines[1:-1]:
        code = ln.split("    # ")[0].strip()
        if "(" in code and not code.endswith(":"):
            code = code.split("  ", 1)[-1].strip()
            exec(code, ns)
            ran += 1
    assert ran >= 2


def test_freehand_starter_is_a_plan_with_no_kit_calls(tmp_path):
    prompt = ("A watercolour of two otters holding hands in a misty river at dawn, soft pastel colours, "
              "with the caption 'Never let go', 1080x1350")
    r = _run("tools/new_scene.py", "otters", "--prompt", prompt, "--dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "(kind: illustration, 1080x1350)" in r.stdout and "subject  otter x2: draw it" in r.stdout
    src = (tmp_path / "otters.py").read_text(encoding="utf-8")
    doc = src.split('"""')[1]
    assert "* Mood and light: misty, dawn;" in doc
    assert "* Palette, 4-6 colours + one accent: pastel" in doc
    assert "* Style, built not filtered: watercolor:" in doc and '* Exact text: "Never let go"' in doc
    assert "* Subject and frame: otter x2, misty river, dawn;" in doc
    assert 'TITLE = "Never let go"' in src and 'STYLE = "watercolor"' in src and 'MOOD = "golden"' in src
    assert "DEFAULT_SIZE = (1080, 1350)" in src and f"PROMPT = {prompt!r}" in src
    block = src.split("# >>> subjects", 1)[1].split("# <<< subjects", 1)[0]
    assert "# draw: otter x2" in block and "# draw: misty river" in block
    assert all(ln.strip().startswith("#") for ln in block.splitlines()[1:] if ln.strip())
    body = src.split("def render", 1)[1]
    assert "stylize" not in src and "clay." not in body.replace(block, "") and "sc.add" not in body.replace(block, "")
    r = _run("tools/render.py", str(tmp_path / "otters.py"), "--scale", "0.25", "--check", "--json")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    data = json.loads(r.stdout.strip().splitlines()[-1])
    fails = [w for key in ("warnings", "text_issues", "object_issues", "prompt_issues")
             for w in (data.get(key) or []) if w.get("severity") == "fail"]
    assert not fails, fails


@pytest.mark.parametrize("prompt", ["a girl reading a book under a tree",            # sc.add(clay.prop("book")) once
                                    "a pixel-art wizard's tower on a cliff with a dragon at night",
                                    "a cosy cafe with a sign reading 'Open Late'"])
def test_freehand_starters_render_untouched(prompt, tmp_path):
    r = _run("tools/new_scene.py", "s", "--prompt", prompt, "--dir", str(tmp_path))
    assert r.returncode == 0 and "(kind: illustration" in r.stdout, r.stdout + r.stderr
    r = _run("tools/render.py", str(tmp_path / "s.py"), "--scale", "0.25", "--check", "--json")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    data = json.loads(r.stdout.strip().splitlines()[-1])
    fails = [w for key in ("warnings", "text_issues", "object_issues", "prompt_issues")
             for w in (data.get(key) or []) if w.get("severity") == "fail"]
    assert not fails, fails


# ------------------------------------------------------------------------------- relations: riding, holding, beside
def test_a_rider_sits_on_the_bicycle_and_a_held_thing_is_in_the_hand(tmp_path):
    src, out = _starter("a flat vector picture of a boy riding a bicycle on a quiet street", tmp_path, "ride")
    bike = _spots(src, '"bicycle"')
    boy = re.search(r'figure\.person\("\w+", name="boy"[^\n]*height=([\d.]+)\)[^\n]*at=\(([\d.]+), ([\d.]+)\)', src)
    assert bike and boy and "riding the bicycle" in out
    assert '.pose("sit", seat="none")' in src                                # the bicycle is the seat
    src, _out = _starter("a flat vector picture of an old man sitting on a bench in a park", tmp_path, "bench")
    man = re.search(r'figure\.person\([^\n]*\.pose\("sit", seat="none"\)[^\n]*at=\(([\d.]+), ([\d.]+)\)', src)
    bench = _spots(src, '"bench"')
    assert man and bench and abs(float(man.group(2)) - bench[0][1]) < 0.01     # feet on the bench's ground line
    bu, bv, bs = bike[0]
    assert abs(float(boy.group(2)) - bu) < 0.05 and bv - bs < float(boy.group(3)) < bv      # feet on the bike
    src, out = _starter("a flat vector picture of a girl holding a kite on a hill", tmp_path, "hold")
    girl = re.search(r'figure\.person\("girl"[^\n]*height=([\d.]+)\)[^\n]*at=\(([\d.]+), ([\d.]+)\)', src)
    kite = _spots(src, '"kite"')
    assert girl and kite and "held by the girl" in out
    gh, gu, gv = (float(x) for x in girl.groups())
    ku, kv, ks = kite[0]
    assert abs(ku - gu) < 0.2 and gv - gh < kv < gv and ks < gh                # at the hand, smaller than her


def test_beside_places_things_side_by_side(tmp_path):
    src, out = _starter("a flat illustration of a campfire beside a tent in a meadow at night", tmp_path)
    tent, fire = _spots(src, '"tent"'), re.search(r"elements\.campfire\(cv, at=\(([\d.]+), ([\d.]+)\)", src)
    assert tent and fire and "beside the tent" in out
    tu, tv, ts = tent[0]
    fu, fv = float(fire.group(1)), float(fire.group(2))
    assert abs(fv - tv) < 0.01 and abs(fu - tu) > 0.6 * ts * 1080 / 1920        # same depth, not in front of it


# ------------------------------------------------------------------------------- ASCII files, one call, any console
ASCII_REQUESTS = {   # kind: an ASCII request (the generators add units, symbols and placeholders of their own)
    "drawing": "An engineering drawing of a flange 120 x 80 x 40 mm with a 30 mm bore, four 8 mm holes, 2 mm chamfer",
    "floorplan": "A floor plan of a two-bedroom flat, 9 x 7 m, with a kitchen and a bathroom",
    "comic": "A four-panel comic about a robot learning to cook: 'What is salt?'",
    "figure": "A labelled diagram of a plant cell for a biology textbook: nucleus, wall, chloroplast",
    "chart": "A line chart of the average temperature in Rome over 12 months",
    "dashboard": "A dark dashboard with KPI tiles for revenue, latency and churn",
}


def test_every_template_is_ascii():
    for kind in KINDS:
        raw = open(os.path.join(ROOT, "scenes", f"_template_{kind}.py"), "rb").read()
        assert raw.isascii(), (kind, sorted({c for c in raw.decode("utf-8") if ord(c) > 127}))
        assert raw.startswith(b"# -*- coding: utf-8 -*-\n"), kind


@pytest.mark.parametrize("kind", KINDS)
def test_written_starters_are_ascii_for_ascii_requests(kind, tmp_path):
    prompt = ASCII_REQUESTS.get(kind, "A fox reading a book under a lamp, titled 'Quiet Night', 1200x800")
    assert new_scene.main(["s", "--kind", kind, "--prompt", prompt, "--dir", str(tmp_path), "--quiet"]) == 0
    raw = (tmp_path / "s.py").read_bytes()
    assert raw.isascii(), (kind, sorted({c for c in raw.decode("utf-8") if ord(c) > 127}))
    open(tmp_path / "s.py", encoding="cp950").read()                   # what a patch script on a cp950 console does


def test_a_cjk_title_comes_through_as_real_characters_and_renders(tmp_path):
    prompt = "A poster titled '\u85cd\u8272\u6642\u523b' for a jazz night"
    r = _run("tools/new_scene.py", "cjk", "--kind", "poster", "--prompt", prompt, "--dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    src = (tmp_path / "cjk.py").read_text(encoding="utf-8")
    assert '"\u85cd\u8272\u6642\u523b"' in src and "\\u85cd" not in src
    assert all(ord(c) < 128 or c in prompt for c in src)               # only the request's own characters
    r = _run("tools/render.py", str(tmp_path / "cjk.py"), "--scale", "0.25", "--check")
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    assert "\u85cd\u8272\u6642\u523b" in r.stdout


@pytest.mark.parametrize("args", [
    ["tools/kit.py", "find", "a corgi under an oak tree"], ["tools/kit.py", "find", "\u4e00\u96bb\u8c93\u5728\u7a97\u908a"],
    ["tools/kit.py", "find", "a bar chart titled '\u96e8\u5b63'"], ["tools/pc.py", "version"],
    ["tools/new_scene.py", "a1", "--prompt", "a flowchart of ordering coffee: choose, pay, collect"],
    ["tools/new_scene.py", "a2", "--prompt", "\u4e00\u5f35\u6d77\u5831 'Bonjour \u2615' \u2605"],
    ["tools/pc.py", "new", "a3", "--prompt", "a cat on a sofa, titled '\u8c93'"],
])
def test_tools_run_on_a_cp950_console(args, tmp_path):
    if "new" in args or "tools/new_scene.py" in args:
        args = args + ["--dir", str(tmp_path)]
    env = dict(os.environ, PYTHONIOENCODING="cp950")
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, env=env, timeout=600)
    err = r.stderr.decode("utf-8", "replace")
    assert r.returncode == 0 and "UnicodeEncodeError" not in err and "UnicodeDecodeError" not in err, err[-800:]


def test_new_scene_prints_the_file_it_wrote_from_an_empty_project(tmp_path):
    """python <root>/tools/new_scene.py NAME --prompt ... --size WxH --dir ./art from a project folder."""
    tool = os.path.join(ROOT, "tools", "new_scene.py")
    env = {k: v for k, v in ENV.items() if k not in ("PIXELCANVAS_WORK", "PIXELCANVAS_CMD")}
    r = subprocess.run([sys.executable, tool, "sales", "--prompt", "A bar chart of quarterly sales titled 'Up and right'",
                        "--size", "1600x900", "--dir", "./art"], cwd=tmp_path, capture_output=True, text=True,
                       encoding="utf-8", env=env, timeout=600)
    assert r.returncode == 0, r.stderr
    path = tmp_path / "art" / "sales.py"
    src = path.read_text(encoding="utf-8")
    assert "(kind: chart, 1600x900)" in r.stdout and "DEFAULT_SIZE = (1600, 1900)" not in src
    assert "--- art/sales.py (" in r.stdout and "--- end ---" in r.stdout
    for i, ln in enumerate(src.rstrip("\n").split("\n"), 1):          # the whole file, numbered
        assert f"{i:4d}  {ln}" in r.stdout
    assert 'TITLE = "Up and right"' in r.stdout and "DEFAULT_SIZE = (1600, 900)" in r.stdout
    r2 = subprocess.run([sys.executable, tool, "quiet", "--kind", "chart", "--quiet"], cwd=tmp_path,
                        capture_output=True, text=True, encoding="utf-8", env=env, timeout=600)
    assert r2.returncode == 0 and "--- " not in r2.stdout and (tmp_path / "art" / "quiet.py").exists()   # ./art default
    render = os.path.join(ROOT, "tools", "render.py")
    r3 = subprocess.run([sys.executable, render, "art/sales.py", "--scale", "0.25", "--check"], cwd=tmp_path,
                        capture_output=True, text=True, encoding="utf-8", env=env, timeout=600)
    assert r3.returncode == 0 and _fix_block_clean(r3.stdout), r3.stdout[-1500:] + r3.stderr[-1500:]


def _fix_block_clean(report):
    """The check report ends in a FIX block (a line starting with FIX) and no [fail] line follows it."""
    lines = [ln.strip() for ln in report.splitlines()]
    at = next((i for i, ln in enumerate(lines) if ln.startswith("FIX")), None)
    return at is not None and not any(ln.startswith("[fail]") for ln in lines[at:])


def test_long_starters_print_plan_tunables_and_blocks():
    body = "\n".join(f"    x{i} = {i}" for i in range(150))
    src = ('# -*- coding: utf-8 -*-\n"""Plan\n\n* a\n"""\nimport os\n\nTITLE = "T"\nDATA = [\n    1,\n]\n'
           f"\n\ndef render():\n    # >>> subjects: here\n    # draw: fox\n    # <<< subjects\n{body}\n")
    out = new_scene.listing(src, "x.py")
    text = "\n".join(out)
    assert "plan, tunables and the request's blocks" in out[0] and len(out) < 30
    assert '   8  TITLE = "T"' in text and "  10      1," in text and "# draw: fox" in text and "     ..." in text
    assert "x100 = 100" not in text
