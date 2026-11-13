import pytest

from pixelcanvas import _args as A


def test_unknown_option_suggests():
    with pytest.raises(TypeError) as e:
        A.reject_unknown("kanban_card", {"tagz": 1}, ["tag", "people", "title"])
    assert "unknown option 'tagz'" in str(e.value) and "'tag'" in str(e.value) and "Options: tag, people" in str(e.value)


def test_plural_alias():
    assert A.reject_unknown("f", {"tags": [1]}, ["tag"]) == {"tag": [1]}
    assert A.reject_unknown("f", {"color": 1}, ["colors"]) == {"colors": 1}


def test_colour_compounds():
    assert A.reject_unknown("f", {"fill_colour": 1}, ["fill_color"]) == {"fill_color": 1}
    assert A.reject_unknown("f", {"grey": 1, "centre": 2}, ["gray", "center"]) == {"gray": 1, "center": 2}


def test_builtin_and_custom_aliases():
    assert A.reject_unknown("f", {"w": 3, "alpha": 1, "lw": 2}, ["width", "opacity", "line_width"]) == \
        {"width": 3, "opacity": 1, "line_width": 2}
    assert A.reject_unknown("f", {"zz": 3}, ["yy"], {"zz": "yy"}) == {"yy": 3}


def test_duplicate():
    with pytest.raises(TypeError, match="given twice"):
        A.reject_unknown("f", {"color": 1, "colour": 2}, ["color"])


def test_many_options_truncated():
    with pytest.raises(TypeError) as e:
        A.reject_unknown("f", {"nope": 1}, ["opt%d" % i for i in range(30)])
    assert "…" in str(e.value)


def test_unknown_name_and_choice():
    e = A.unknown_name("plant", "oka", ["oak", "pine"], "Try flora.names()")
    assert isinstance(e, ValueError) and "'oak'" in str(e) and "flora.names()" in str(e)
    with pytest.raises(ValueError, match="'dark'"):
        A.check_choice("axes", "theme", "dar", ["light", "dark"])
    assert A.check_choice("axes", "theme", "dark", ["light", "dark"]) == "dark"


def test_normalise_and_signature():
    assert A.normalise({"colour": 1, "x": 2}) == {"color": 1, "x": 2}
    assert A.normalise({"a": 1}, {"a": "b"}) == {"b": 1}

    def f(self, a, b=1, *args, c=2, **kw):
        pass
    assert A.signature_options(f) == ["a", "b", "c"]
