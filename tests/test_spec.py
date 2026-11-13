# -*- coding: utf-8 -*-
"""pixelcanvas.spec and tools/compose.py: the JSON spec format for small models (references/spec.md)."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pixelcanvas import spec as S  # noqa: E402

COMPOSE = os.path.join(ROOT, "tools", "compose.py")
PAGE = os.path.join(ROOT, "skills", "pixelcanvas", "references", "spec.md")


def _compose(args, cwd):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, COMPOSE] + [str(a) for a in args], cwd=str(cwd), capture_output=True,
                          text=True, encoding="utf-8", env=env, timeout=300)


def _validate(spec):
    norm, problems = S.validate(spec)
    return norm, [p for p in problems if p.severity == "error"]


# ------------------------------------------------------------------------------------------- sloppy JSON
def test_repairs_sloppy_json():
    text = """Here is the file:
```json
{
  // comment
  'pcscene': 1, kind: poster,
  "layers": [
    {"draw": "moon", "at": 'top right', "visible": True,},
    {"text": "Blue Hour", "role": "title"}
    {"text": "It's late", "role": "details", "x": None},
  ],
  /* block */ "style": None,
}
```"""
    data, notes = S.loads(text)
    assert data["kind"] == "poster" and data["style"] is None
    assert data["layers"][0] == {"draw": "moon", "at": "top right", "visible": True}
    assert data["layers"][2]["text"] == "It's late" and data["layers"][2]["x"] is None
    joined = " ".join(notes)
    for word in ("code fence", "comma", "comments", "quotes", "True", "bare"):
        assert word in joined, (word, notes)


def test_strict_json_has_no_repairs_and_unreadable_raises():
    data, notes = S.loads('{"kind": "scene", "layers": []}')
    assert notes == [] and data["kind"] == "scene"
    with pytest.raises(S.SpecError):
        S.loads("")
    data, notes = S.loads('{"kind": "scene", "layers": [{"draw": "oak"}')       # unclosed brackets
    assert data["layers"][0]["draw"] == "oak" and any("unclosed" in n for n in notes)


# ------------------------------------------------------------------------------------------- validation
def test_all_problems_at_once_with_fixes():
    spec = {"pcscene": 1, "kind": "scene", "colour": "x",
            "layers": [{"draw": "saxaphone"}, {"draw": "cat", "at": "centre-left-ish of the moon thing"},
                       {"draw": "cat", "pose": "sittting"}, {"text": "Hi", "role": "titel"},
                       {"draw": "oak", "size": "enormous-ish"}]}
    norm, errors = _validate(spec)
    paths = [p.path for p in errors]
    assert "layers[0].draw" in paths and "layers[2].pose" in paths and "layers[3].role" in paths
    assert "layers[4].size" in paths
    sax = next(p for p in errors if p.path == "layers[0].draw")
    assert "saxophone" in (sax.fix or "") or "saxophone" in sax.message
    pose = next(p for p in errors if p.path == "layers[2].pose")
    assert "'sit'" in pose.fix
    role = next(p for p in errors if p.path == "layers[3].role")
    assert "'title'" in role.fix


def test_unknown_names_suggest_and_never_draw_something_else():
    t, extras, sugg, note = S.lookup("saxaphone")
    assert t is None and "saxophone" in sugg
    t, extras, sugg, note = S.lookup("zzqx")
    assert t is None
    t, extras, _, note = S.lookup("ginger cat")
    assert t.name == "cat" and extras == {"coat": "ginger"} and "ginger" in note
    t, _, _, note = S.lookup("big coffee cup")
    assert t is not None and note                                 # what was read is reported


def test_aliases_number_words_and_people():
    norm, errors = _validate({"kind": "animals", "objects": [{"name": "two birds", "position": "in the sky"},
                                                             {"draw": "woman in a red coat"}]})
    assert not errors
    birds, woman = norm["layers"]
    assert birds["count"] == 2 and birds["thing"].cat == "bird"
    assert woman["thing"].kit == "figure" and woman["opts"]["describe"] == "woman in a red coat"
    assert any("objects" in n for n in norm["notes"])


def test_top_level_text_fields_become_layers():
    norm, errors = _validate({"kind": "card", "title": "Designing for Calm", "byline": "R. Okonkwo"})
    assert not errors
    roles = {L["role"]: L["text"] for L in norm["layers"]}
    assert roles == {"title": "Designing for Calm", "byline": "R. Okonkwo"}


# ------------------------------------------------------------------------------------------- placement
@pytest.mark.parametrize("words, want", [
    ("centre", {"h": 0.5, "v": 0.5}),
    ("top left", {"h": 0.24, "v": 0.16}),
    ("bottom third", {"v": 0.72}),
    ("far right", {"h": 0.9}),
    ("in the sky", {"sky": 1}),
    ("on the ground", {"ground": 1}),
    ("foreground", {"depth": 0.88}),
])
def test_placement_words(words, want):
    at, err = S.parse_at(words)
    assert err is None
    for k, v in want.items():
        assert at[k] == pytest.approx(v)


def test_placement_relations_numbers_and_errors():
    at, _ = S.parse_at("left of the cat")
    assert at["rel"] == "left of" and at["target"] == "cat"
    at, _ = S.parse_at("under oak")
    assert at["rel"] == "under" and at["target"] == "oak"
    at, _ = S.parse_at("on the ground")
    assert "rel" not in at and at["ground"] == 1
    assert S.parse_at([0.3, 0.7])[0] == {"mode": "point", "values": [0.3, 0.7]}
    assert S.parse_at([0.1, 0.2, 0.5, 0.6])[0]["mode"] == "box"
    at, _ = S.parse_at([30, 70])
    assert at["values"] == [0.3, 0.7] and "percent" in at["note"]
    at, err = S.parse_at("centre-left-ish of the moon thing")
    assert err is None or "not understood" in err
    at, err = S.parse_at("wobbly")
    assert at is None and "wobbly" in err


def test_relations_in_a_render_honour_target():
    spec = {"kind": "animals", "background": {"sky": "day"},
            "layers": [{"draw": "oak", "at": "right third", "size": "large"},
                       {"draw": "cat", "at": "under oak"}, {"draw": "dog", "at": "left of cat"}]}
    norm, errors = _validate(spec)
    assert not errors
    S.render(norm, 320, 180)
    placed = S.LAST["placed"]
    oak, cat, dog = placed["oak"]["bbox"], placed["cat"]["bbox"], placed["dog"]["bbox"]
    assert oak[0] < (cat[0] + cat[2]) / 2 < oak[2] + 0.1           # the cat is at the tree
    assert (dog[0] + dog[2]) / 2 < (cat[0] + cat[2]) / 2           # the dog is left of the cat
    assert not S.LAST["errors"]


def test_count_arrangements():
    for arr in ("row", "cluster", "scatter"):
        spec = {"kind": "still_life", "layers": [{"draw": "orange", "count": 3, "arrange": arr}]}
        norm, errors = _validate(spec)
        assert not errors
        cv = S.render(norm, 320, 180)
        names = [e["name"] for e in cv.manifest if isinstance(e, dict) and e.get("kind") != "text"]
        assert sum(1 for n in names if n.split(" #")[0] == "orange") == 3, (arr, names)
    spec = {"kind": "scene", "layers": [{"draw": "bird", "count": 4, "at": "in the sky"}]}
    norm, _ = _validate(spec)
    S.render(norm, 320, 180)
    assert S.LAST["placed"]["bird"]["count"] == 4


# ------------------------------------------------------------------------------------------- examples
@pytest.mark.parametrize("kind", list(S.EXAMPLES))
def test_every_example_validates_and_renders(kind):
    ex = S.example(kind)
    norm, errors = _validate(ex)
    assert not errors, [str(e) for e in errors]
    cv = S.render(norm, 192, 108 if norm["size"][0] >= norm["size"][1] else 240)
    assert not S.LAST["errors"], [str(e) for e in S.LAST["errors"]]
    assert np.isfinite(cv.buf).all() and cv.buf.std() > 0.01


def _last_next(out):
    lines = [l for l in out.splitlines() if l.strip()]
    assert lines and lines[-1].startswith("NEXT: "), out[-600:]
    assert sum(l.startswith("NEXT:") for l in lines) == 1, lines
    return lines


_PY_WORDS = ("manifest.note", "look.py", "render.py", "new_scene", "cv.", "stylize.apply")


@pytest.mark.parametrize("kind", list(S.EXAMPLES))
def test_examples_render_with_zero_fail_through_compose(tmp_path, kind):
    path = tmp_path / f"{kind}.json"
    r = _compose(["--example", kind], tmp_path)
    assert r.returncode == 0
    path.write_text(r.stdout, encoding="utf-8")
    r = _compose([path, "--check", "--scale", "0.2"], tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    lines = _last_next(r.stdout)
    assert not any(l.startswith("PROBLEMS (") and not l.startswith("PROBLEMS (0 fail") for l in lines),         (kind, r.stdout[-900:])
    assert not any(w in r.stdout for w in _PY_WORDS), (kind, r.stdout[-900:])


def test_compose_edits_name_colour_text_and_size_in_spec_terms(tmp_path):
    spec = S.example("poster")
    spec["prompt"] = "A poster with a blue cactus and a tiny fern called 'Big Sale'"
    spec["layers"] = [{"draw": "potted cactus", "at": "left", "size": "large"},
                      {"draw": "potted fern", "at": "far right", "size": 0.05},
                      {"text": "Green Days", "role": "title"}]
    path = tmp_path / "w.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    r = _compose([path, "--check", "--scale", "0.35"], tmp_path)
    out = r.stdout
    lines = _last_next(out)
    edits = [l for l in lines if l.startswith("  ") and l.strip()[:2] in ("1.", "2.", "3.", "4.", "5.")]
    assert any('set "color": "blue"' in l and "potted cactus" in l for l in edits), out[-1200:]
    assert any("'Big Sale'" in l and '"text": "Big Sale"' in l for l in edits), out[-1200:]
    assert any("potted fern" in l and '"size"' in l for l in edits), out[-1200:]
    assert any(l.startswith("PROBLEMS (") for l in lines)
    assert not any(w in out for w in _PY_WORDS), out[-1200:]


def test_next_steps_map_report_issues_to_spec_edits():
    norm, errs = _validate(S.example("poster"))
    layer = next(L for L in norm["layers"] if L["type"] == "thing")
    name = layer["label"]
    rep = {"objects": [], "text_issues": [], "warnings": [], "prompt_issues": [
        {"rule": "prompt_colour", "severity": "warn", "threshold": "blue",
         "message": f"The request says a blue {name}, the {name} is painted mostly green (blue covers 0% of it)."},
        {"rule": "prompt_small", "severity": "warn",
         "message": f"The request names the {name}, drawn at 0.10 % of the canvas: too small to be seen."},
        {"rule": "prompt_text_small", "severity": "warn",
         "message": "The quoted text 'Green Days' is drawn with letters 1.0 % of the canvas height: too small."}]}
    out, _ = S.next_steps(norm, rep)
    text = " ".join(out)
    assert f'"color": "blue" in {layer["path"]}' in text
    assert f'"size" to "large" in {layer["path"]}' in text
    assert '"role" to "title"' in text or '"role" to "subtitle"' in text
    assert not any(w in text for w in _PY_WORDS)


# ------------------------------------------------------------------------------------------- edits
def test_set_add_remove(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps(S.example("animals")), encoding="utf-8")
    data, _ = S.load(path)
    assert S.set_value(data, "layers.1.size", "0.5").startswith("set")
    assert data["layers"][1]["size"] == 0.5
    S.set_value(data, "style", "gouache")
    S.set_value(data, "background.sky", "night")
    assert data["style"] == "gouache" and data["background"]["sky"] == "night"
    S.set_value(data, "layers[2].at", "left of cherry")
    assert data["layers"][2]["at"] == "left of cherry"
    S.add_layer(data, '{"draw": "moon", "at": "top right"}')
    assert data["layers"][-1]["draw"] == "moon"
    S.remove_layer(data, "moon")
    assert all(L.get("draw") != "moon" for L in data["layers"])
    S.remove_layer(data, "0")
    assert data["layers"][0]["draw"] == "rabbit"
    with pytest.raises(KeyError):
        S.remove_layer(data, "unicorn")
    room = S.example("room")
    S.add_layer(room, '{"draw": "sofa"}')
    assert room["scene3d"]["objects"][-1]["draw"] == "sofa"


def test_compose_edits_rewrite_the_file(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps(S.example("chart")), encoding="utf-8")
    r = _compose([path, "--set", "chart.title=Revenue 2026", "--add", '{"text": "Q3 2026", "role": "caption"}',
                  "--scale", "0.25"], tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["chart"]["title"] == "Revenue 2026" and data["layers"][-1]["text"] == "Q3 2026"
    assert "edit: set chart.title" in r.stdout


def test_compose_reports_problems_and_draws_nothing(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"kind": "scene", "layers": [{"draw": "saxaphone"}, {"draw": "cat", "at": "wobbly"}]}',
                    encoding="utf-8")
    r = _compose([path, "--check"], tmp_path)
    assert r.returncode == 2
    assert "PROBLEMS (2" in r.stdout and "saxophone" in r.stdout and "wobbly" in r.stdout
    assert not (tmp_path / "outputs" / "bad_x0.5.png").exists()


def test_compose_info_commands(tmp_path):
    r = _compose(["--kinds"], tmp_path)
    assert r.returncode == 0 and "poster" in r.stdout and "scene3d" in r.stdout
    r = _compose(["--schema"], tmp_path)
    assert r.returncode == 0 and '"layers"' in r.stdout
    r = _compose(["--names", "bird"], tmp_path)
    assert r.returncode == 0 and "sparrow" in r.stdout
    r = _compose(["--example", "nosuchkind"], tmp_path)
    assert r.returncode == 2


# ------------------------------------------------------------------------------------------- NEXT block
def _report(objects=(), texts=(), prompt=()):
    return {"objects": list(objects), "text_issues": list(texts), "prompt_issues": list(prompt), "warnings": []}


def test_next_block_missing_quoted_text_and_count():
    norm, _ = _validate({"kind": "poster", "prompt": "a poster called 'Blue Hour' with three moons",
                         "layers": [{"draw": "moon", "count": 2}]})
    rep = _report(prompt=[{"rule": "prompt_text_missing", "severity": "fail", "value": "Blue Hour"},
                          {"rule": "prompt_count", "severity": "warn", "value": 2, "threshold": 3,
                           "message": "The request says three moons but 2 are drawn (moon)."}])
    lines, n = S.next_steps(norm, rep)
    joined = "\n".join(lines)
    assert '{"text": "Blue Hour", "role": "title"}' in joined
    assert '"count": 3' in joined and "layers[0]" in joined


def test_next_block_hidden_object_and_contrast():
    norm, _ = _validate({"kind": "animals", "layers": [{"draw": "oak"}, {"draw": "bird", "at": "behind oak"},
                                                        {"text": "Hello", "role": "title"}]})
    rows = [{"name": "oak", "kind": "plant", "bbox": [0.4, 0.2, 0.7, 0.8], "visible_pct": 100, "area_pct": 12,
             "clipped_pct": 0, "hidden_by": None},
            {"name": "bird", "kind": "creature", "bbox": [0.5, 0.3, 0.56, 0.36], "visible_pct": 20, "area_pct": 0.3,
             "clipped_pct": 0, "hidden_by": "oak"}]
    texts = [{"rule": "text_contrast", "severity": "warn", "value": "Hello", "message": "contrast 2.1:1"}]
    lines, n = S.next_steps(norm, _report(rows, texts))
    joined = "\n".join(lines)
    assert "layers[1] 'bird' is 80% hidden by 'oak'" in joined and 'set "at" to [' in joined
    assert "layers[2] 'Hello': low contrast" in joined and '"color"' in joined


def test_next_block_names_missing_things_and_stops():
    norm, _ = _validate({"kind": "animals", "prompt": "a cat under an oak with two birds",
                         "layers": [{"draw": "cat"}, {"draw": "oak"}]})
    lines, n = S.next_steps(norm, _report())
    assert any("'bird'" in l for l in lines)
    norm, _ = _validate({"kind": "animals", "prompt": "a cat under an oak",
                         "layers": [{"draw": "cat"}, {"draw": "oak"}]})
    assert norm["layers"][0]["at"]["rel"] == "under"            # the request's relation is applied
    lines, n = S.next_steps(norm, _report())
    assert lines == [] and n == 0
    norm, _ = _validate({"kind": "animals", "prompt": "a cat under an oak",
                         "layers": [{"draw": "cat", "at": "left of oak"}, {"draw": "oak"}]})
    lines, n = S.next_steps(norm, _report())
    assert lines == ["the request puts 'cat' under 'oak' -> set \"at\": \"under oak\" in layers[0]"]
    norm, _ = _validate({"kind": "animals", "prompt": "a cat under an oak",
                         "layers": [{"draw": "cat", "at": "under oak"}, {"draw": "oak"}]})
    lines, n = S.next_steps(norm, _report())
    assert lines == [] and n == 0
    norm, _ = _validate(S.example("chart"))
    norm["prompt"] = "bars: Jan 320, Feb 999"
    lines, _ = S.next_steps(norm, _report())
    assert any("Feb" in l and "999" in l for l in lines)


# ------------------------------------------------------------------------------------------- escape hatch
def test_python_escape_hatch(tmp_path):
    (tmp_path / "my_shapes.py").write_text(
        "def draw_logo(cv, box, layer):\n"
        "    import numpy as np\n"
        "    from pixelcanvas.masks import rect_mask\n"
        "    cv.paint('#ff0000', rect_mask(cv.h, cv.w, *box))\n", encoding="utf-8")
    spec = {"kind": "poster", "layers": [{"python": "my_shapes.py:draw_logo", "at": [0.1, 0.1, 0.5, 0.5]}]}
    norm, errors = S.validate(spec, base_dir=str(tmp_path))
    assert not [e for e in errors if e.severity == "error"]
    cv = S.render(norm, 200, 200)
    assert not S.LAST["errors"], S.LAST["errors"]
    assert cv.buf[60, 60, 0] > 0.9 and cv.buf[60, 60, 1] < 0.1
    norm["layers"][0]["ref"] = "my_shapes.py:nope"
    S.render(norm, 100, 100)
    assert any("nope" in p.message for p in S.LAST["errors"])


# ------------------------------------------------------------------------------------------- the reference page
def test_reference_page_examples_validate():
    import re
    text = open(PAGE, encoding="utf-8").read()
    assert len(text.splitlines()) <= 250
    blocks = re.findall(r"```json\n(.*?)```", text, re.S)
    assert len(blocks) >= 3
    for b in blocks:
        data, _ = S.loads(b)
        norm, errors = _validate(data)
        assert not errors, [str(e) for e in errors]


# ------------------------------------------------------------------------------------------- small-model guards
def test_empty_design_is_a_problem():
    _, errs = _validate({"kind": "card", "prompt": "a card that says Calm", "palette": "sage"})
    assert errs and errs[0].path == "layers"
    _, errs = _validate({"kind": "poster", "layers": [{"draw": "saxophone"}]})
    assert any("text" in e.message for e in errs)


def test_scene_follows_the_request_and_keeps_trees_on_the_ground():
    norm, errs = _validate({"kind": "animals", "prompt": "A ginger cat under an oak tree at sunset",
                            "background": {"sky": "sunset", "ground": "grass"},
                            "layers": [{"draw": "oak", "at": [0.64, 0.37]}, {"draw": "cat", "at": [0.3, 0.8]}]})
    assert not errs and norm["layers"][1]["at"]["target_layer"] == "oak"
    S.render(norm, 320, 180, seed=1)
    oak, cat = S.LAST["placed"]["oak"], S.LAST["placed"]["cat"]
    assert oak["bbox"][3] > 0.6                        # its base moved down onto the ground (horizon 0.62)
    assert abs((cat["bbox"][0] + cat["bbox"][2]) / 2 - (oak["bbox"][0] + oak["bbox"][2]) / 2) < 0.25


def test_backdrop_sky_word_and_prompt_pose():
    norm, errs = _validate({"kind": "scene", "prompt": "A ginger cat sitting under an oak, two birds in the sky",
                            "background": {"backdrop": "sunset"},
                            "layers": [{"draw": "oak"}, {"draw": "ginger cat"}, {"draw": "bird", "count": 2, "at": "sky"}]})
    assert not errs and norm["background"]["sky"] == "sunset" and norm["background"].get("ground")
    assert norm["layers"][1]["opts"]["pose"] == "sit"
    S.render(norm, 320, 180, seed=2)
    oak = S.LAST["placed"]["oak"]["bbox"]
    birds = [v["bbox"] for k, v in S.LAST["placed"].items() if k.startswith("bird")]
    assert birds and S.LAST["order"].index("bird") > S.LAST["order"].index("oak")
    for b in birds:                                     # flying in the open sky, not inside the crown
        cu, cv_ = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        assert not (oak[0] < cu < oak[2] and oak[1] < cv_ < oak[3])



# ------------------------------------------------------------------------------------------- malformed specs
# What a small local model writes when it gets the format slightly wrong. Every one is either repaired (with a
# READ note) or answered with one PROBLEMS line; none raises anything but SpecError, and compose.py never prints
# a traceback and always ends with one NEXT: line.
MALFORMED = [
    ("empty", ""),
    ("whitespace", "   \n\n  "),
    ("fence", '```json\n{"kind": "poster", "prompt": "A poster \'Hi\'", "layers": [{"text": "Hi", "role": "title"}]}\n```'),
    ("fence_no_lang", '```\n{"kind": "card", "layers": [{"text": "Hello", "role": "title"}]}\n```'),
    ("prose_around", 'Sure! Here is the spec:\n{"kind": "poster", "layers": [{"text": "Jazz", "role": "title"}]}\nLet me know.'),
    ("trailing_commas", '{"kind": "poster", "layers": [{"text": "Jazz", "role": "title",},],}'),
    ("comments", '{\n // the kind\n "kind": "poster", /* layers */ "layers": [{"text": "Jazz", "role": "title"}] # end\n}'),
    ("single_quotes", "{'kind': 'poster', 'layers': [{'text': 'Jazz', 'role': 'title'}]}"),
    ("python_literals", '{"kind": "poster", "style": None, "post": False, "layers": [{"text": "Jazz", "role": "title", "shadow": True}]}'),
    ("unknown_top_key", '{"kind": "poster", "colour_scheme": "blue", "layers": [{"text": "Jazz", "role": "title"}]}'),
    ("unknown_layer_key", '{"kind": "poster", "layers": [{"text": "Jazz", "role": "title", "fontsize": 90}]}'),
    ("misspelt_draw", '{"kind": "scene", "layers": [{"draw": "saxaphone"}, {"draw": "oak tre"}]}'),
    ("misspelt_kind", '{"kind": "postr", "layers": [{"text": "Jazz", "role": "title"}]}'),
    ("kind_phrase", '{"kind": "film poster", "layers": [{"text": "Jazz", "role": "title"}]}'),
    ("missing_layers", '{"kind": "poster", "prompt": "A poster"}'),
    ("layers_is_string", '{"kind": "scene", "layers": "a cat under a tree"}'),
    ("layers_is_dict", '{"kind": "scene", "layers": {"draw": "cat"}}'),
    ("layer_is_string", '{"kind": "scene", "layers": ["cat", "oak"]}'),
    ("extra_nesting", '{"spec": {"kind": "poster", "layers": [{"text": "Jazz", "role": "title"}]}}'),
    ("list_top", '[{"kind": "poster", "layers": [{"text": "Jazz", "role": "title"}]}]'),
    ("numbers_as_strings", '{"kind": "scene", "seed": "3", "layers": [{"draw": "bird", "count": "3", "size": "0.2"}]}'),
    ("size_string_x", '{"kind": "poster", "size": "1920x1080", "layers": [{"text": "Jazz", "role": "title"}]}'),
    ("size_string_by", '{"kind": "poster", "size": "1080 by 1350", "layers": [{"text": "Jazz", "role": "title"}]}'),
    ("size_list_strings", '{"kind": "poster", "size": ["1080", "1350"], "layers": [{"text": "Jazz", "role": "title"}]}'),
    ("colour_names", '{"kind": "poster", "palette": ["navy", "gold"], "background": {"color": "dark navy"}, "layers": [{"text": "Jazz", "role": "title", "color": "golden yellow"}]}'),
    ("bad_colour", '{"kind": "poster", "layers": [{"text": "Jazz", "role": "title", "color": "sparkly"}]}'),
    ("example_text_left", '{"kind": "poster", "prompt": "A jazz poster called \'Blue Hour\'", "layers": [{"text": "<the title from the request>", "role": "title"}]}'),
    ("placeholder_draw", '{"kind": "scene", "layers": [{"draw": "NAME", "at": "WHERE"}]}'),
    ("count_word", '{"kind": "scene", "layers": [{"draw": "two birds", "at": "in the sky"}]}'),
    ("at_number_list_strings", '{"kind": "scene", "layers": [{"draw": "cat", "at": ["0.3", "0.8"]}]}'),
    ("at_nonsense", '{"kind": "scene", "layers": [{"draw": "cat", "at": "wobbly"}]}'),
    ("role_unknown", '{"kind": "poster", "layers": [{"text": "Jazz", "role": "headline"}, {"text": "x", "role": "fineprint"}]}'),
    ("chart_data_list", '{"kind": "chart", "chart": {"type": "bar", "data": [3, 5, 4]}}'),
    ("chart_data_strings", '{"kind": "chart", "chart": {"type": "line", "data": {"Jan": "4", "Feb": "5,5", "Mar": "$6"}}}'),
    ("chart_type_typo", '{"kind": "chart", "chart": {"type": "lines", "data": {"A": 1, "B": 2}}}'),
    ("chart_wrong_shape", '{"kind": "chart", "chart": "bar chart of sales"}'),
    ("diagram_edges_strings", '{"kind": "diagram", "diagram": {"nodes": ["A", "B", "C"], "edges": ["A -> B", "B->C"]}}'),
    ("diagram_unknown_node", '{"kind": "diagram", "diagram": {"nodes": ["A", "B"], "edges": [["A", "Z"]]}}'),
    ("ui_bad_template", '{"kind": "ui", "ui": {"template": "loginn", "content": {"title": "Hi"}}}'),
    ("room_objects_strings", '{"kind": "room", "scene3d": {"objects": ["bed", "desk", "lamp"]}}'),
    ("section_wrong_kind", '{"kind": "poster", "chart": {"type": "bar", "data": {"A": 1}}, "layers": [{"text": "Jazz", "role": "title"}]}'),
    ("unclosed", '{"kind": "poster", "layers": [{"text": "Jazz", "role": "title"}'),
    ("garbage", "draw me a cat please"),
    ("python_code", "import numpy as np\nfrom pixelcanvas import Canvas\ndef render(w, h, seed, log):\n    pass\n"),
    ("yaml_like", "kind: poster\nlayers:\n  - text: Jazz\n    role: title\n"),
    ("equals_keys", '{kind = "poster", layers = [{text = "Jazz", role = "title"}]}'),
    ("curly_quotes", '{“kind”: “poster”, “layers”: [{“text”: “Jazz”, “role”: “title”}]}'),
    ("bom", '﻿{"kind": "poster", "layers": [{"text": "Jazz", "role": "title"}]}'),
    ("null_values", '{"kind": "scene", "background": null, "layers": [{"draw": "cat", "at": null, "size": null, "color": null}]}'),
    ("negative_count", '{"kind": "scene", "layers": [{"draw": "bird", "count": -2}]}'),
    ("huge_count", '{"kind": "scene", "layers": [{"draw": "bird", "count": 5000}]}'),
    ("style_unknown", '{"kind": "scene", "style": "vaporwave dreams", "layers": [{"draw": "cat"}]}'),
    ("prompt_list", '{"kind": "poster", "prompt": ["a", "poster"], "layers": [{"text": "Jazz", "role": "title"}]}'),
    ("text_number", '{"kind": "poster", "layers": [{"text": 2026, "role": "title"}]}'),
    ("two_objects", '{"kind": "poster", "layers": [{"text": "A", "role": "title"}]}\n{"kind": "card"}'),
]


def _read_and_check(text):
    """loads + validate like compose.py does: (norm, problems) or SpecError."""
    data, notes = S.loads(text)
    return S.validate(data)


@pytest.mark.parametrize("name,text", MALFORMED, ids=[m[0] for m in MALFORMED])
def test_malformed_specs_are_repaired_or_explained(name, text):
    try:
        norm, problems = _read_and_check(text)
    except S.SpecError as e:
        assert str(e) and "\n" not in str(e)
        return
    for p in problems:
        assert isinstance(p, S.Problem) and str(p) and len(str(p)) < 400
    assert norm is not None


def test_malformed_repairs_read_what_was_meant():
    norm, errs = _validate(S.loads('{"kind": "poster", "size": "1080 by 1350", "seed": "3", '
                                   '"layers": [{"text": "Jazz", "role": "title"}]}')[0])
    assert not errs and norm["size"] == [1080, 1350] and norm["seed"] == 3
    norm, errs = _validate(S.loads('{"kind": "poster", "size": ["1920", "1080"], "colour_scheme": "navy and gold", '
                                   '"layers": [{"text": "Jazz", "role": "title"}]}')[0])
    assert not errs and norm["size"] == [1920, 1080] and norm["palette_spec"] == "navy and gold"
    data, notes = S.loads('{"spec": {"kind": "card", "layers": [{"text": "Hi", "role": "title"}]}}')
    assert data["kind"] == "card" and any("unwrapped" in n for n in notes)
    data, notes = S.loads('{"kind": "card", "layers": [{"text": "A", "role": "title"}]}\n{"kind": "poster"}')
    assert data["kind"] == "card" and any("first JSON object" in n for n in notes)
    with pytest.raises(S.SpecError, match="YAML"):
        S.loads("kind: poster\nlayers:\n  - text: Jazz\n")
    _, errs = _validate({"kind": "scene", "layers": [{"draw": "NAME", "at": "WHERE"}]})
    assert errs and "placeholder" in errs[0].message
    _, errs = _validate({"kind": "poster", "layers": [{"text": "<the title from the request>", "role": "title"}]})
    assert len(errs) == 1 and "placeholder" in errs[0].message
    _, errs = _validate({"kind": "poster", "layers": [{"text": "A", "role": "title"}]})   # a one-letter title is fine
    assert not errs
    _, errs = _validate({"kind": "scene", "layers": [{"draw": "saxaphone"}, {"draw": "oak tre"}]})
    assert len(errs) == 2 and "saxophone" in str(errs[0]) and "oak" in str(errs[1])   # no extra "nothing to draw"
    assert S.example("film poster")["kind"] == "poster" and S.example("line chart")["kind"] == "chart"
    assert S.example("mobile app screen")["kind"] == "ui"


_COMPOSE_CASES = ["empty", "fence", "misspelt_draw", "missing_layers", "garbage", "python_code", "yaml_like",
                  "chart_wrong_shape", "size_string_x", "example_text_left", "unknown_layer_key", "two_objects"]


def test_compose_never_prints_a_traceback_and_ends_with_one_next(tmp_path):
    cases = dict(MALFORMED)
    for name in _COMPOSE_CASES:
        path = tmp_path / f"{name}.json"
        path.write_text(cases[name], encoding="utf-8")
        r = _compose([path, "--check", "--scale", "0.15"], tmp_path)
        out = r.stdout + r.stderr
        assert "Traceback" not in out, (name, out[-800:])
        lines = [l for l in r.stdout.splitlines() if l.strip()]
        assert lines and lines[-1].startswith("NEXT: "), (name, out[-800:])
        assert sum(l.startswith("NEXT:") for l in lines) == 1, (name, lines)
    for args in (["--chek", "x.json"], [], ["--names", "zzzzqqq"], ["missing.json", "--check"]):
        r = _compose(args, tmp_path)
        lines = [l for l in r.stdout.splitlines() if l.strip()]
        assert "Traceback" not in r.stdout + r.stderr and lines[-1].startswith("NEXT: "), (args, r.stdout, r.stderr)
    r = _compose(["--chek", "x.json"], tmp_path)
    assert "did you mean --check" in r.stdout
    r = _compose(["--example", "postr"], tmp_path)
    assert r.returncode == 2 and "poster" in r.stderr and r.stderr.strip().splitlines()[-1].startswith("NEXT: ")
    r = _compose(["--example", "card"], tmp_path)
    assert json.loads(r.stdout)["kind"] == "card" and r.stderr.strip().startswith("NEXT: ")


def test_compose_next_line_walks_the_loop(tmp_path):
    path = tmp_path / "card.json"
    path.write_text(json.dumps(S.example("card")), encoding="utf-8")
    r = _compose([path, "--check", "--scale", "0.25"], tmp_path)
    last = r.stdout.strip().splitlines()[-1]
    assert last.startswith("NEXT: "), r.stdout[-600:]
    assert "look.py" not in r.stdout and "manifest.note" not in r.stdout and "render.py" not in r.stdout
    r = _compose([path, "--deliver", str(tmp_path / "final"), "--scale", "0.25"], tmp_path)
    last = r.stdout.strip().splitlines()[-1]
    assert last.startswith("NEXT: tell the user the picture is at ") and "card" in last, r.stdout[-600:]
    assert list((tmp_path / "final").glob("card*.png"))
    bad = tmp_path / "scene.py"
    bad.write_text("import numpy as np\ndef render(w, h, seed, log):\n    pass\n", encoding="utf-8")
    r = _compose([bad, "--check"], tmp_path)
    assert "never write Python" in r.stdout and r.stdout.strip().splitlines()[-1].startswith("NEXT: ")


def test_compose_stops_editing_after_four_drafts(tmp_path):
    path = tmp_path / "p.json"
    spec = S.example("poster")
    spec["prompt"] = "A poster with a saxophone called 'Blue Hour'"      # edits that are never made
    path.write_text(json.dumps(spec), encoding="utf-8")
    last = ""
    for i in range(4):
        r = _compose([path, "--check", "--scale", "0.2"], tmp_path)
        last = r.stdout.strip().splitlines()[-1]
        if i == 0:
            assert last.startswith("NEXT: make the "), r.stdout[-800:]
    assert last.startswith("NEXT: deliver: "), r.stdout[-800:]


# ------------------------------------------------------------------------------------------- composed layouts
def test_card_has_panel_rule_and_author_line():
    norm, errs = _validate(S.example("card"))
    assert not errs
    cv = S.render(norm, 600, 315)
    assert not S.LAST["errors"], S.LAST["errors"]
    cup = S.LAST["placed"]["coffee cup"]["bbox"]
    assert cup[0] > 0.55                                   # the illustration sits on the right-hand panel
    buf = np.asarray(cv.buf)
    panel = buf[int(0.2 * 315), int(0.64 * 600)]
    page = buf[int(0.2 * 315), int(0.3 * 600)]
    assert np.abs(panel - page).sum() > 0.05               # the panel is a different tone from the page
    norm, errs = _validate({"kind": "card", "prompt": "A banner 'Launch Week'", "background": {"color": "#1b2a41"},
                            "layers": [{"text": "Launch Week", "role": "title"}, {"text": "May 4-8", "role": "date"}]})
    assert not errs
    S.render(norm, 600, 315)
    assert not S.LAST["errors"]


def test_poster_focal_subject_fills_the_picture_area():
    norm, errs = _validate(S.example("poster"))
    assert not errs
    S.render(norm, 540, 675)
    pl = S.LAST["placed"]
    focal = pl["potted monstera"]["bbox"]
    sides = [pl["potted cactus"]["bbox"], pl["potted fern"]["bbox"]]
    assert focal[3] - focal[1] > 0.25                      # a strong focal subject
    for b in sides:
        assert b[3] - b[1] < focal[3] - focal[1]
        assert abs(b[3] - focal[3]) < 0.04                 # all stand on one baseline
        cu, fw = (b[0] + b[2]) / 2, focal[2] - focal[0]
        assert not focal[0] + 0.2 * fw < cu < focal[2] - 0.2 * fw     # flanking, not on top of it


def test_chart_has_header_summary_and_highlight():
    norm, errs = _validate(S.example("chart"))
    assert not errs
    S.render(norm, 800, 450)
    assert not S.LAST["errors"], S.LAST["errors"]
    assert S._chart_summary("bar", {"Jan": 320, "Apr": 520}) == "total 840 · highest Apr 520"
    assert "+100%" in S._chart_summary("line", {"Jan": 10, "Dec": 20})
    kw = {}
    S._chart_page(type("C", (), {"cv": __import__("pixelcanvas").Canvas(320, 180), "log": lambda *a: None})(),
                  {"title": "T"}, "bar", {"A": 1, "B": 3, "C": 2}, "light", kw)
    assert kw["highlight"] == "B" and kw["card"] is True


def test_under_a_tree_stands_beside_the_trunk():
    norm, errs = _validate({"kind": "scene", "prompt": "A ginger cat sitting under an oak tree at sunset",
                            "background": {"sky": "sunset", "ground": "grass"},
                            "layers": [{"draw": "oak", "at": "center"},
                                       {"draw": "cat", "coat": "ginger", "pose": "sit", "at": "under oak"}]})
    assert not errs
    S.render(norm, 480, 270, seed=1)
    oak, cat = S.LAST["placed"]["oak"], S.LAST["placed"]["cat"]
    trunk_u = (oak["bbox"][0] + oak["bbox"][2]) / 2
    assert cat["bbox"][0] > trunk_u + 0.02 or cat["bbox"][2] < trunk_u - 0.02      # clear of the trunk
    assert cat["bbox"][3] >= oak["bbox"][3] - 0.01                                  # on the ground, not the roots
    assert oak["bbox"][0] - 0.05 < (cat["bbox"][0] + cat["bbox"][2]) / 2 < oak["bbox"][2] + 0.05   # under the crown


# ------------------------------------------------------------------------------------------- fixes from the r12 trials
def test_asleep_in_winter_lies_down_under_a_bare_tree():
    norm, errs = _validate({"kind": "scene", "prompt": "A red fox asleep in the snow under a birch tree on a winter evening",
                            "background": {"sky": "dusk", "ground": "snow", "far": "forest"},
                            "layers": [{"draw": "birch", "at": "center right"}, {"draw": "fox", "at": "under birch"}]})
    assert not errs and norm["layers"][1]["opts"]["pose"] == "sleep" and S._wintry(norm)
    _, errs = _validate({"kind": "scene", "layers": [{"draw": "fox", "pose": "relaxed"}]})
    assert not errs                                       # relaxed -> rest / lie, not an error
    S.render(norm, 480, 270, seed=1)
    birch, fox = S.LAST["placed"]["birch"]["bbox"], S.LAST["placed"]["fox"]["bbox"]
    trunk = (birch[0] + birch[2]) / 2
    assert fox[2] < trunk or fox[0] > trunk               # beside the trunk, not through it


def test_kind_that_does_not_fit_the_request_is_a_problem():
    _, errs = _validate({"kind": "icons", "prompt": "A mobile app screen that greets the user with 'Hi, Sam'",
                         "layers": [{"text": "Hi, Sam", "role": "title"}]})
    assert errs and errs[0].path == "kind" and '"ui"' in errs[0].fix
    _, errs = _validate({"kind": "poster", "prompt": "A poster for a mobile app launch",
                         "layers": [{"text": "Launch", "role": "title"}]})
    assert not errs                                       # a poster about an app is still a poster


def test_ui_home_screen_has_greeting_and_cards():
    spec = S.example("ui")
    assert spec["ui"]["template"] == "home"
    norm, errs = _validate(spec)
    assert not errs
    cv = S.render(norm, 480, 270)
    assert not S.LAST["errors"], S.LAST["errors"]
    texts = " ".join(str(e.get("name", "")) + " " + str(e.get("text", "")) for e in (getattr(cv, "manifest", None) or [])
                     if isinstance(e, dict))
    assert cv.buf.std() > 0.02
    _, errs = _validate({"kind": "ui", "ui": {"template": "loginn"}})
    assert errs and "login" in str(errs[0]) and "home" in str(errs[0])


def test_things_next_to_one_target_line_up():
    norm, errs = _validate({"kind": "still_life", "prompt": "two apples, a pear and grapes beside a bowl on a table",
                            "background": {"table": "wood"},
                            "layers": [{"draw": "bowl"}, {"draw": "pear"}, {"draw": "grapes"}, {"draw": "apple"}]})
    assert not errs
    S.render(norm, 480, 270)
    pl = S.LAST["placed"]
    cs = sorted((pl[k]["bbox"][0] + pl[k]["bbox"][2]) / 2 for k in ("bowl", "pear", "grapes", "apple"))
    assert min(b - a for a, b in zip(cs, cs[1:])) > 0.05, cs     # no two fruits on the same spot
    assert S._natural_colour(S.lookup("pear", "still_life")[0]) is not None


def test_colour_words_are_not_things_and_skies_go_behind():
    words = [w for w, _ in S._prompt_things("A night market poster in warm orange and deep blue")]
    assert "orange" not in words
    words = [w for w, _ in S._prompt_things("Three oranges in a bowl")]
    assert "oranges" in words
    norm, errs = _validate({"kind": "poster", "prompt": "A film poster 'Last Light'",
                            "layers": [{"draw": "film reel", "at": "bottom right"}, {"draw": "sky", "size": "huge"},
                                       {"text": "Last Light", "role": "title"},
                                       {"text": "The sun is leaving.", "role": "subtitle"},
                                       {"text": "So are we.", "role": "subtitle"}]})
    assert not errs
    S.render(norm, 270, 338)
    assert "film reel" in S.LAST["placed"]
    reel = S.LAST["placed"]["film reel"]["bbox"]
    assert abs((reel[0] + reel[2]) / 2 - 0.5) < 0.08          # the only picture is the centred focal subject
    assert S._poster_content([L for L in norm["layers"] if L["type"] == "text"])["subtitle"] == \
        "The sun is leaving. So are we."


# ------------------------------------------------------------------------------------------- example quality (int_r1)
SPECS = os.path.join(ROOT, "scenes", "specs")


@pytest.mark.parametrize("kind", list(S.EXAMPLES))
def test_spec_files_are_the_examples_and_render(kind):
    path = os.path.join(SPECS, f"{kind}.json")
    assert os.path.exists(path), f"scenes/specs/{kind}.json is missing"
    data, notes = S.load(path)
    assert data == S.example(kind) and not notes            # refreshed from the current example
    norm, errors = _validate(data)
    assert not errors, [str(e) for e in errors]
    cv = S.render(norm, 192, 108 if norm["size"][0] >= norm["size"][1] else 240)
    assert not S.LAST["errors"], [str(e) for e in S.LAST["errors"]]
    assert np.isfinite(cv.buf).all() and cv.buf.std() > 0.01


def test_product_shot_stands_as_one_group():
    from pixelcanvas import solid
    norm, errs = _validate(S.example("scene3d"))
    assert not errs and len(norm["scene3d"]["objects"]) >= 3
    lay = S._studio_layout(solid, norm["scene3d"]["objects"])
    xs = sorted(x for x, z, r in lay.values())
    assert len(xs) == 4                                     # bottle, mug, two apples
    assert max(b - a for a, b in zip(xs, xs[1:])) < 0.3     # neighbours, not metres apart
    tallest = max(lay, key=lambda k: k[0] == "scene3d.objects[0]")
    assert abs(lay[tallest][0]) < 0.15                       # the bottle stands in the middle
    objs = [dict(o) for o in norm["scene3d"]["objects"][:2]]
    objs[0]["at"], objs[1]["at"] = {"mode": "point", "values": [0.5, 0]}, {"mode": "point", "values": [-0.5, 0]}
    lay = S._studio_layout(solid, objs)
    assert lay[(objs[1]["path"], 0)][0] < lay[(objs[0]["path"], 0)][0]   # numbers keep left-to-right order


def test_two_people_stand_together_and_face_each_other():
    norm, errs = _validate({"kind": "people", "prompt": "Two friends talking in a park",
                            "background": {"sky": "day", "ground": "grass"},
                            "layers": [{"draw": "person", "describe": "a woman in a red coat", "at": "far left"},
                                       {"draw": "person", "describe": "a man in a blue jacket", "at": "far right"},
                                       {"draw": "dog", "at": "right of the man"}]})
    assert not errs and norm["layers"][2]["at"]["target_layer"] == norm["layers"][1]["label"]
    S.render(norm, 480, 270, seed=1)
    a, b = norm["layers"][0], norm["layers"][1]
    assert a["opts"]["facing"] == "right" and b["opts"]["facing"] == "left"
    pa, pb = S.LAST["placed"][a["label"]]["bbox"], S.LAST["placed"][b["label"]]["bbox"]
    gap = (pb[0] + pb[2]) / 2 - (pa[0] + pa[2]) / 2
    assert 0.08 < gap < 0.3                                 # a conversation, not two ends of the meadow
    dog = S.LAST["placed"]["dog"]["bbox"]
    assert (dog[0] + dog[2]) / 2 > (pb[0] + pb[2]) / 2      # "right of the man" found the man
    assert not [l for l in S._leftovers(norm) if "person" in l]   # "two friends" are the people


def test_under_a_tree_is_in_front_of_the_trunk_not_on_the_roots():
    norm, errs = _validate({"kind": "scene", "prompt": "A red fox asleep in the snow under a birch tree",
                            "background": {"sky": "evening", "ground": "snow", "far": "forest"},
                            "layers": [{"draw": "birch", "at": "left third", "size": "huge"},
                                       {"draw": "fox", "pose": "sleep", "at": "under birch", "size": "small"}]})
    assert not errs
    S.render(norm, 480, 270, seed=1)
    tree, fox = S.LAST["placed"]["birch"], S.LAST["placed"]["fox"]
    trunk = tree["base"][0]
    assert fox["bbox"][0] > trunk + 0.04                   # clear of the trunk and its root flare
    assert fox["base"][1] > tree["base"][1] + 0.03         # a step in front of the tree, on the ground


def test_wallpaper_text_sits_in_the_calm_sky_and_reads():
    norm, errs = _validate(S.example("wallpaper"))
    assert not errs and norm["background"]["backdrop"] == "mountains"
    cv = S.render(norm, 384, 216)
    assert not S.LAST["errors"]
    texts = cv.text_log
    assert len(texts) == 2 and all(t["bbox"][3] < 0.55 * cv.h for t in texts)    # in the sky above the ridges
    assert min(t["contrast"] for t in texts) >= 3.0
    lines, _ = S.next_steps(norm, _report())
    assert not lines, lines                                    # the mountains backdrop shows the 'mountain'


def test_room_example_is_furnished_and_small_things_read():
    norm, errs = _validate(S.example("room"))
    assert not errs and norm["size"] == [1440, 1080]
    keys = {o["src"][1] for o in norm["scene3d"]["objects"]}
    assert {"sofa", "armchair", "bookshelf", "floor_lamp", "potted_plant", "vase", "table"} <= keys
    lines, _ = S.next_steps(norm, _report())
    assert not any("lamp" in l for l in lines), lines        # a floor lamp is the requested lamp
    from pixelcanvas import solid
    vase = next(o for o in norm["scene3d"]["objects"] if o["src"][1] == "vase")
    vase = dict(vase, in_room=True)
    m, _ = S._make_model(solid, vase, 0)
    assert max(m.size) > 1.3 * max(solid.vase().size)        # scaled up on the table


def test_still_life_is_one_group_and_donut_names_the_largest_share():
    norm, errs = _validate(S.example("still_life"))
    assert not errs
    S.render(norm, 480, 270)
    cs = [(v["bbox"][0] + v["bbox"][2]) / 2 for v in S.LAST["placed"].values()]
    assert min(cs) > 0.2 and max(cs) < 0.8, cs
    import pixelcanvas.chart as C
    seen = {}
    orig = C.Axes.pie
    def spy(self, values, labels=None, **kw):
        seen.update(kw)
        return orig(self, values, labels, **kw)
    C.Axes.pie = spy
    try:
        norm, _ = _validate(S.example("dashboard"))
        S.render(norm, 480, 270)
    finally:
        C.Axes.pie = orig
    assert seen.get("center") == ("38%", "Search")


def test_cloud_reports_its_own_box_and_home_screen_is_filled():
    norm, errs = _validate({"kind": "scene", "background": {"sky": "day", "ground": "grass"},
                            "layers": [{"draw": "cloud", "at": "top right"}]})
    cv = S.render(norm, 480, 270)
    cloud = next(e for e in cv.manifest if isinstance(e, dict) and e.get("name") == "cloud")
    assert cloud["extent"][1] >= -0.02                    # not the drawing window far above the cloud
    norm, _ = _validate(S.example("ui"))
    cv = S.render(norm, 480, 300)
    assert not S.LAST["errors"]
    names = " ".join(str(e.get("name", "")) for e in cv.manifest if isinstance(e, dict))
    assert "This week" in names or "chart" in names or "ring" in names, names[:400]


# ------------------------------------------------------------------------------------------- fixes from the r14 trial
def test_poster_text_drops_quotes_and_boxes_and_sea_is_the_backdrop():
    norm, errs = _validate({"kind": "poster", "prompt": "A poster 'Wind Lanterns' with glowing kites over the sea",
                            "background": {"sky": "night"},
                            "layers": [{"text": "'Wind Lanterns'", "role": "title", "box": [0.25, 0.1, 0.75, 0.18]},
                                       {"draw": "sea"}, {"draw": "kite", "count": 3}]})
    assert not errs
    title = next(L for L in norm["layers"] if L["type"] == "text")
    assert title["text"] == "Wind Lanterns" and title["at"]["mode"] == "auto"       # laid out by the template
    assert norm["background"]["backdrop"] == "ocean" and not any(L.get("name") == "sea" for L in norm["layers"])
    S.render(norm, 270, 338)
    assert not S.LAST["errors"]
    kites = S.LAST["placed"]["kite"]["bbox"]
    assert kites[3] < 0.85                                  # flying, not standing on the stage baseline


def test_flowchart_question_is_a_diamond_and_yes_no_become_labels():
    norm, errs = _validate({"kind": "diagram", "diagram": {
        "nodes": ["Start", "Water Tank Empty?", "Refill", "Brew"],
        "edges": [["Start", "Water Tank Empty?"], ["Water Tank Empty?", "Yes -> Refill"],
                  ["Water Tank Empty?", "No -> Brew"]], "groups": {"Decision": ["Water Tank Empty?"]}}})
    assert not errs
    d = norm["diagram"]
    shapes = {n["label"]: n.get("shape") for n in d["nodes"]}
    assert shapes["Water Tank Empty?"] == "diamond" and shapes["Start"] == "pill"
    assert {(e[1], e[2]) for e in d["edges"] if e[2]} == {("refill", "Yes"), ("brew", "No")}
    assert d["groups"] == []                                # a frame around one box is left out


def test_winter_pine_forest_is_conifers_and_small_fruit_fill_the_table():
    norm, _ = _validate({"kind": "animals", "prompt": "A deer in a snowy pine forest at dawn",
                         "background": {"sky": "dawn", "ground": "snow", "far": "forest"},
                         "layers": [{"draw": "pine tree"}, {"draw": "deer"}]})
    assert S._far_forest_kw(norm)["species"] == ("spruce", "fir")
    norm, _ = _validate({"kind": "still_life", "prompt": "a lemon, a peach and three plums",
                         "layers": [{"draw": "lemon", "color": "#fffacd"}, {"draw": "peach"},
                                    {"draw": "plum", "count": 3}]})
    S.render(norm, 480, 270)
    lemon = S.LAST["placed"]["lemon"]["bbox"]
    assert lemon[3] - lemon[1] > 0.12                       # drawn larger than a lone small fruit
    assert any("washed-out" in w for w in S.LAST["warnings"])


def test_sparse_room_gets_a_rug_a_plant_and_a_lamp_on_the_desk():
    norm, errs = _validate({"kind": "room", "prompt": "An isometric home office with a desk, an office chair and a bookshelf",
                            "scene3d": {"objects": [{"draw": "desk"}, {"draw": "office chair"}, {"draw": "bookshelf"}]}})
    assert not errs
    objs = norm["scene3d"]["objects"]
    keys = [o["src"][1] for o in objs]
    assert {"rug", "potted_plant", "desk_lamp"} <= set(keys)
    assert next(o for o in objs if o["src"][1] == "desk_lamp")["on_obj"] == "desk"
    assert not S._leftovers(norm)                           # what the composer added is not "left over"


def test_campfire_friends_phrases_and_guitar_are_read():
    norm, errs = _validate({"kind": "people", "prompt": "Two friends sitting by a campfire at night, one of them playing the guitar",
                            "background": {"sky": "night stars", "ground": "forest floor with logs", "far": "dark trees"},
                            "layers": [{"draw": "campfire", "at": "center bottom"},
                                       {"draw": "person", "describe": "a person holding a guitar", "pose": "sit, hold",
                                        "at": "left third"},
                                       {"draw": "guitar", "at": "in person's hands"},
                                       {"draw": "person", "describe": "a friend listening", "pose": "sit",
                                        "at": "right of first person"}]})
    assert not errs, [str(e) for e in errs]
    bg = norm["background"]
    assert bg["sky"] == "night" and bg["far"] == "forest" and bg["ground"][0] == "soil"
    figs = [L for L in norm["layers"] if L["type"] == "thing" and L["thing"].kit == "figure"]
    assert figs[0]["opts"]["pose"] == "guitar" and not any(L.get("name") == "guitar" for L in norm["layers"])
    assert figs[1]["at"]["target_layer"] == figs[0]["label"]
    S.render(norm, 480, 270, seed=1)
    pl = S.LAST["placed"]
    a, b, fire = pl[figs[0]["label"]]["bbox"], pl[figs[1]["label"]]["bbox"], pl["campfire"]["bbox"]
    fu = (fire[0] + fire[2]) / 2
    assert (a[0] + a[2]) / 2 < fu < (b[0] + b[2]) / 2       # the fire sits between the friends
    _, errs = _validate({"kind": "scene", "layers": [{"draw": "fox", "pose": "dance around"}]})
    assert errs and "sit" in errs[0].fix and "stand" in errs[0].fix   # every pose is listed


def test_flowchart_step_without_an_arrow_is_joined_from_the_step_before():
    norm, errs = _validate({"kind": "diagram", "diagram": {
        "nodes": ["Start", "Check", "Empty?", "Refill", "Brew", "Serve"],
        "edges": [["Start", "Check"], ["Check", "Empty?"], ["Empty?", "Refill", "Yes"], ["Empty?", "Brew", "No"],
                  ["Refill", "Brew"]]}})
    assert not errs
    assert ("brew", "serve", None, "solid") in norm["diagram"]["edges"]
    assert any("'Serve' had no arrow" in n for n in norm["notes"])


# ------------------------------------------------------------------------------------------- fixes from r14b / r15
def test_home_screen_reads_loose_cards_and_never_loses_the_picture():
    for cards in ("Water the fern, New leaf on Monty and Light check",
                  {"Water the fern": "every 3 days", "Light check": "bright"},
                  [{"title": "Water the fern", "progress": -5}, {"title": "Light check", "progress": 250}]):
        spec = S.example("ui")
        spec["ui"]["content"] = {"title": "Welcome back, Lena", "cards": cards}
        norm, errs = _validate(spec)
        assert not errs
        S.render(norm, 480, 300)
        assert not S.LAST["errors"], (cards, S.LAST["errors"])
    norm, _ = _validate(S.example("ui"))
    lines, _ = S.next_steps(norm, _report(prompt=[{"rule": "prompt_text_missing", "severity": "fail",
                                                   "value": "Light check"}]))
    assert lines and '"title": "Light check"' in lines[0] and "brand" not in lines[0]


def test_night_scene_stars_and_dark_ground():
    norm, errs = _validate({"kind": "people", "prompt": "Two friends by a campfire at night",
                            "background": {"sky": "night", "ground": "grass", "far": "forest"},
                            "layers": [{"draw": "star", "at": "top left"}, {"draw": "campfire"},
                                       {"draw": "person"}, {"draw": "person"}]})
    assert not errs
    assert not any(L["type"] == "thing" and L["thing"].kit == "pictogram" for L in norm["layers"])
    assert any(L["type"] == "thing" and L["thing"].target == "stars" for L in norm["layers"])
    cv = S.render(norm, 320, 180)
    assert float(np.asarray(cv.buf)[int(0.85 * 180), :, :].mean()) < 0.35      # the ground is not daylit
