"""detail_measure: band energies, flat area, detail spread and line taper, compared with professional profiles."""
from __future__ import annotations

import importlib.util
import math
import os
import pathlib

import cv2
import numpy as np
import pytest

from pixelcanvas import detail_measure as dm

N = 256
# the reference implementation (spectrum.py, not shipped) is optional; point PIXELCANVAS_SPECTRUM
# at it to run the equivalence test, which is skipped otherwise
SPECTRUM = pathlib.Path(os.environ["PIXELCANVAS_SPECTRUM"]) if os.environ.get("PIXELCANVAS_SPECTRUM") else None


def grey(a):
    return np.repeat(np.clip(a, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)


def bands(m, c="L"):
    return np.array([m[f"band{c}_{k}"] for k in range(7)])


@pytest.mark.parametrize("k,size", list(enumerate([1.5, 3, 6, 12, 24, 48])))
def test_sinusoid_lands_in_its_band(k, size):
    xx = np.mgrid[0:N, 0:N][1].astype(np.float32)
    m = dm.measure(grey(128 + 60 * np.sin(2 * np.pi * xx / (2 * size))), jpeg=None, lines=False)
    e = bands(m)
    assert int(np.argmax(e)) == k and e[k] > 0.6 * e.sum(), np.round(e / e.sum(), 2)


def test_flat_picture():
    m = dm.measure(np.full((N, N, 3), 200, np.uint8), jpeg=None)
    assert m["flat_share"] > 0.99 and m["flat0_share"] > 0.99
    assert bands(m).sum() < 1e-9 and bands(m, "C").sum() < 1e-9
    assert m["line_len_per_mp"] == 0 and math.isnan(m["line_taper"])
    assert m["det_empty"] == 1.0


def test_detail_spread():
    rng = np.random.default_rng(0)
    img = np.full((N, N), 128, np.float32)
    img[:64, :64] += 40 * rng.standard_normal((64, 64))
    m = dm.measure(grey(img), jpeg=None, lines=False)
    assert m["det_empty"] > 0.85 and m["det_top10"] > 0.9
    m = dm.measure(grey(128 + 30 * rng.standard_normal((N, N))), jpeg=None, lines=False)
    assert m["det_empty"] == 0.0 and m["det_gini"] < 0.2


def test_jpeg_round_trip_runs():
    rng = np.random.default_rng(1)
    m = dm.measure(rng.integers(0, 255, (128, 160, 3), dtype=np.uint8))
    assert (m["width"], m["height"]) == (160, 128)


def test_delivered_size():
    a = np.zeros((300, 400, 3), np.uint8)
    assert dm.delivered(a, 1) is a
    assert dm.delivered(a, 2).shape == (150, 200, 3)


def test_area_measures_full_and_half():
    rng = np.random.default_rng(2)
    img = np.full((N, N, 3), 140, np.float32)
    img[:, : N // 2] += 30 * rng.standard_normal((N, N // 2, 1)) * [1.0, 0.6, 0.2]
    m = dm.measure(np.clip(img, 0, 255).astype(np.uint8), jpeg=None, lines=False, keep=True)
    left = np.zeros((N, N), bool)
    left[:, : N // 2] = True
    full = np.ones((N, N), bool)
    tiny = np.zeros((N, N), bool)
    tiny[:5, :5] = True
    a = dm.area_measures(m, {"full": full, "left": left, "right": ~left, "tiny": tiny})
    assert "tiny" not in a
    for k in range(7):
        assert a["full"][f"bandL_{k}"] == pytest.approx(m[f"bandL_{k}"], rel=1e-6)
        assert a["full"][f"bandC_{k}"] == pytest.approx(m[f"bandC_{k}"], rel=1e-6)
        assert (a["left"][f"bandL_{k}"] + a["right"][f"bandL_{k}"]) / 2 == pytest.approx(m[f"bandL_{k}"], rel=1e-6)
    assert a["full"]["flat_share"] == pytest.approx(m["flat_share"], abs=1e-9)
    assert a["full"]["edge_density"] == pytest.approx(m["edge_density"], abs=1e-9)
    assert a["full"]["det_empty"] == pytest.approx(m["det_empty"], abs=1e-9)
    assert a["left"]["bandL_0"] > 50 * a["right"]["bandL_0"]
    assert a["left"]["fine_rms"] > 5 * a["right"]["fine_rms"]
    assert a["left"]["det_empty"] == 0.0 and a["right"]["det_empty"] > 0.9
    assert a["right"]["flat_share"] > 0.9 > a["left"]["flat_share"]
    assert a["left"]["n_px"] == N * N // 2


@pytest.mark.parametrize("style,request_,group", [
    ("poster", "anime girl poster", "promo"),
    ("cel", "a girl, flat colours", "anime-cel"),
    ("ink", "a girl with a sword", "anime"),
    ("paint", "anime girl in a garden", "anime-paint"),
    ("photo", "a mountain lake", "landscape"),
    (None, "chibi cat", "anime"),
    ("watercolour", "a cellar door", "landscape"),
    (None, None, "landscape"),
])
def test_profile_name(style, request_, group):
    assert dm.profile_name({"style": style, "request": request_}) == group


def test_profiles_file():
    p = dm.load_profiles()
    assert set(p["groups"]) == set(dm.GROUPS) and "numbers only" in p["meta"]
    for g in p["groups"].values():
        assert g["n"] > 10
        for key in dm.RANKED_KEYS + dm.GATE_KEYS:
            assert {"min", "q1", "median", "q3", "max"} <= set(g["measures"][key])


def medians(group):
    return {k: st["median"] for k, st in dm.load_profiles()["groups"][group]["measures"].items()}


@pytest.mark.parametrize("group", dm.GROUPS)
def test_gaps_and_outside(group):
    m = medians(group)
    assert dm.outside(m, group) == []
    g = dm.gaps(m, group)
    assert len(g) == 3 and all(abs(d["z"]) < 1e-9 for d in g)
    m["det_empty"] = 0.9
    g = dm.gaps(m, group, n=1)[0]
    assert g["key"] == "det_empty" and g["low"] is False and g["z"] > 1 and "90 %" in g["text"]
    out = dm.outside(m, group)
    assert [d["key"] for d in out] == ["det_empty"]
    m["line_taper"] = float("nan")
    m["bandL_0"] = m["bandL_0"] * 0.1
    out = dm.outside(m, group)
    assert "line_taper" not in [d["key"] for d in out]
    b = next(d for d in out if d["key"] == "bandL_0")
    assert b["low"] and b["ratio"] == pytest.approx(0.1) and b["text"].startswith("fine detail 1-2 px: 0.10x")


# ----------------------------------------------------------------------------- equivalence with spectrum.py
def line_drawing(n=512):
    ss = 4
    big = np.full((n * ss, n * ss, 3), (235, 228, 220), np.uint8)
    for i in range(12):
        x0, y0 = 30 + 35 * i, 40
        for t in range(0, 100):
            f = t / 100
            x = int((x0 + 60 * math.sin(3 * f + i)) * ss)
            y = int((y0 + 420 * f) * ss)
            r = int((0.4 + 2.5 * math.sin(math.pi * f)) * ss)
            cv2.circle(big, (x, y), max(r, 1), (30, 25, 40), -1)
    img = cv2.resize(big, (n, n), interpolation=cv2.INTER_AREA)
    img[300:420, 100:300] = (200, 120, 110)
    return img


def painted(n=512):
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    base = np.stack([120 + 80 * np.sin(xx / 70), 100 + 60 * np.cos(yy / 50), 140 + 50 * np.sin((xx + yy) / 90)], -1)
    grain = cv2.GaussianBlur(rng.standard_normal((n, n)).astype(np.float32), (0, 0), 1.2) * 25
    img = base + grain[:, :, None]
    cv2.ellipse(img, (256, 256), (120, 80), 20, 0, 360, (40, 30, 30), 3)
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def spectrum():
    if SPECTRUM is None or not SPECTRUM.exists():
        pytest.skip("set PIXELCANVAS_SPECTRUM to the path of spectrum.py to compare with the reference")
    spec = importlib.util.spec_from_file_location("_spectrum_ref", SPECTRUM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("make", [line_drawing, painted])
def test_matches_spectrum(spectrum, make):
    img = make()
    ref = spectrum.measure(img, jpeg=None)
    m = dm.measure(img, jpeg=None)
    for k in range(7):
        for c in "LC":
            key = f"band{c}_{k}"
            assert m[key] == pytest.approx(ref[key], rel=1e-4, abs=1e-12), key
    for key in ("flat_share", "flat0_share", "graded_share", "textured_share", "edge_share", "det_empty", "det_top10",
                "edge_density"):
        assert m[key] == pytest.approx(ref[key], abs=1e-6), key
    for key in ("det_gini", "lc_p50", "grad_p90", "line_len_per_mp", "line_w_p50", "line_cv"):
        assert m[key] == pytest.approx(ref[key], rel=1e-4, nan_ok=True), key
    if math.isnan(ref["line_taper"]):
        assert math.isnan(m["line_taper"])
    else:
        assert m["line_taper"] == pytest.approx(ref["line_taper"], abs=1e-6)
    if make is line_drawing:
        assert np.isfinite(m["line_taper"]) and m["line_taper"] > 0.3
