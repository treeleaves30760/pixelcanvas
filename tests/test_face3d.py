"""The anime face specs on a 3D head: every expression builds, the eye parts are there, gaze moves the irises."""
import numpy as np
import pytest

from pixelcanvas import face3d as FC


@pytest.mark.parametrize("e", FC.EXPRESSIONS)
def test_every_expression_builds(e):
    sp = FC.features(e)
    names = {s["name"] for s in sp}
    assert {"brow-r", "brow-l", "nose"} <= names
    assert any(n.startswith("mouth") for n in names)
    for s in sp:
        assert np.isfinite(np.asarray(s["pts"], float)).all()


def test_open_eye_has_the_anime_parts():
    names = {s["name"] for s in FC.features("neutral")}
    for side in "rl":
        for part in ("white", "lashshade", "iris", "irisdark", "irislight", "pupil", "shine-%s0", "shine-%s1",
                     "lash", "lower", "lid"):
            n = part % side if "%s" in part else f"{part}-{side}"
            assert n in names, n
    lash = next(s for s in FC.features("neutral") if s["name"] == "lash-r")
    assert lash["wprof"] and max(v for _, v in lash["wprof"]) > 1.0     # heavier toward the outer corner


def test_aliases_and_new_expressions():
    assert FC.expression_key("calm") == "neutral"
    assert FC.expression_key("laugh") == "laugh"
    assert FC.expression_key("yell") == "shout"
    assert FC.expression_key("blush") == "embarrassed"
    assert FC.expression_key("half-closed") == "sleepy"
    assert FC.expression_key("nonsense") is None
    wink = {s["name"] for s in FC.features("wink")}
    assert "iris-r" in wink and "iris-l" not in wink
    assert any(n.startswith("tear-") for n in {s["name"] for s in FC.features("crying")})
    assert any(n.startswith("hatch-") for n in {s["name"] for s in FC.features("embarrassed")})
    assert "teeth" in {s["name"] for s in FC.features("shout")}


def test_gaze_moves_both_irises_the_same_way():
    def ix(g):
        sp = FC.features("neutral", gaze=FC.gaze_key(g))
        return {s["name"]: np.asarray(s["pts"]).mean(0) for s in sp if s["name"] in ("iris-r", "iris-l")}
    c, lf, up = ix("center"), ix("left"), ix("up")
    for n in ("iris-r", "iris-l"):
        assert lf[n][0] > c[n][0]                  # head frame +x is the picture's left when she faces you
        assert up[n][1] > c[n][1]
    assert FC.gaze_key("0.5,-1") is not None and FC.gaze_key("sideways") is None


def test_eye_shapes_change_the_eye():
    def lash(shape):
        return next(s for s in FC.features("neutral", shape=shape) if s["name"] == "lash-r")["pts"]
    r, n = lash("round"), lash("narrow")
    assert np.ptp(n[:, 1]) < np.ptp(r[:, 1])            # a narrower eye opening
    assert any(s["name"] == "lashfill-r" for s in FC.features("neutral"))
    assert set(FC.EYE_SHAPES) == {"round", "narrow", "droopy", "sharp"}


def test_marks_add_their_shapes():
    names = {s["name"] for s in FC.features("angry", marks=FC.MARKS)}
    assert "sweat" in names and "vein0" in names and "shock0" in names
    assert "tear-r" in names and "hatch-r0" in names
    assert not any(n.startswith("vein") for n in {s["name"] for s in FC.features("angry")})
    assert any(n.startswith("flick-r") for n in {s["name"] for s in FC.features("laugh")})
