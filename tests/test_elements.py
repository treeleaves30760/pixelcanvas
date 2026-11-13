# -*- coding: utf-8 -*-
"""Tests for pixelcanvas.elements: every effect runs, is deterministic, composites in place, and the docs run."""
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pixelcanvas import Canvas  # noqa: E402
from pixelcanvas import elements as E  # noqa: E402

DOC = os.path.join(ROOT, "skills", "pixelcanvas", "references", "elements.md")


def _cv(w=192, h=108, color="#20242c"):
    return Canvas(w, h, color)


@pytest.mark.parametrize("name", E.names())
def test_every_registry_entry_draws(name):
    cv = _cv()
    E.sky(cv, "dusk")
    before = cv.buf.copy()
    kw = {}
    if name == "wet ground":
        kw["mask"] = (cv.xy()[1] > cv.h * 0.6).astype(np.float32)
    if name == "snow cap":
        kw["mask"] = (cv.xy()[1] > cv.h * 0.5).astype(np.float32)
    if name == "volumetric fog":
        kw["lights"] = [E.Light(96, 54, "#ffcc88", 30, 1.0)]
    fx = E.element(cv, name, seed=1, **kw)
    assert isinstance(fx, E.FX)
    assert np.isfinite(cv.buf).all(), name
    assert np.abs(cv.buf - before).max() > 1e-4, f"{name} drew nothing"


def test_deterministic_by_seed():
    a, b, c = _cv(), _cv(), _cv()
    for cv, s in ((a, 3), (b, 3), (c, 4)):
        E.campfire(cv, at=(0.5, 0.8), size=300, seed=s)
    assert np.array_equal(a.buf, b.buf)
    assert not np.array_equal(a.buf, c.buf)


def test_resolution_independent_footprint():
    fracs = []
    for w, h in ((192, 108), (384, 216)):
        cv = _cv(w, h)
        fx = E.flame(cv, at=(0.5, 0.9), size=600, seed=2, glow=0)
        fracs.append(float((fx.mask > 0.5).mean()))
    assert abs(fracs[0] - fracs[1]) < 0.02


def test_flame_is_hot_core_and_transparent_tips():
    cv = Canvas(200, 200, "#000000")
    fx = E.flame(cv, at=(0.5, 0.95), size=1000, seed=1, glow=0)
    core = cv.buf[150:185, 95:105].max()
    assert core > 1.2                                  # HDR white-yellow core
    ys = np.nonzero(fx.mask.max(1) > 0.05)[0]
    assert ys.min() < 100                              # tall tongue
    col = cv.buf[np.argmax(fx.mask[:, 100] > 0.05), 100]
    assert col[0] >= col[2]                            # tips are warm, not blue


def test_worley_cells_and_edges():
    X, Y = np.meshgrid(np.linspace(0, 6, 120), np.linspace(0, 6, 120))
    f1, f2, edge, cid = E.worley(X, Y, seed=5)
    assert (f2 >= f1 - 1e-6).all()
    assert edge.min() >= 0 and edge.max() < 1.0
    assert 10 < len(np.unique(np.round(cid, 5))) < 100
    cv = _cv()
    ids, e = E.voronoi(cv, scale=200)
    assert ids.shape == (cv.h, cv.w) and e.min() >= 0


def test_curl_is_divergence_free():
    X, Y = np.meshgrid(np.linspace(0, 4, 200), np.linspace(0, 4, 200))
    vx, vy = E.curl(X.astype(np.float32), Y.astype(np.float32), seed=1)
    h = 4 / 199
    dvx = np.gradient(vx, h, axis=1)
    div = dvx + np.gradient(vy, h, axis=0)
    assert np.abs(div[5:-5, 5:-5]).mean() < 0.15 * np.abs(dvx[5:-5, 5:-5]).mean()


def test_illuminate_warms_near_the_light():
    cv = Canvas(200, 100, "#404040")
    lt = E.Light(50, 50, (1.0, 0.5, 0.2), 30, 1.0)
    E.illuminate(cv, lt)
    near, far = cv.buf[50, 50], cv.buf[50, 195]
    assert near[0] > far[0] + 0.1 and near[0] > near[2]


def test_lights_are_returned_and_usable():
    cv = _cv()
    fx = E.campfire(cv, at=(0.5, 0.8), size=300)
    assert fx.light is not None and fx.light.radius > 0
    lm = E.light_map(cv, fx)
    assert lm.shape == (cv.h, cv.w, 3) and lm.max() > 0


def test_reflections_sample_the_canvas():
    cv = Canvas(160, 120, "#000000")
    E.moon(cv, at=(0.5, 0.3), size=150, glow=0)
    E.lake(cv, horizon=0.5, ripple=0.2)
    top = cv.buf[:60].max()
    bottom = cv.buf[60:].max()
    assert bottom > 0.3 * top                          # the moon shows up in the water


def test_snow_cap_sits_on_top_edges():
    cv = Canvas(120, 120, "#101010")
    m = np.zeros((120, 120), np.float32)
    m[60:, 20:100] = 1
    fx = E.snow_cap(cv, m, depth=60)
    assert fx.mask[55:70, 60].max() > 0.5              # on the top edge
    assert fx.mask[110, 60] < 0.1                      # not deep inside


def test_find_ranks_sensible_names():
    assert E.find("rain on a window")[0] == "droplets"
    assert E.find("campfire at night")[0] == "campfire"
    assert E.find("northern lights")[0] == "aurora"
    assert "lightning" in E.find("thunderstorm")
    with pytest.raises(KeyError):
        E.element(_cv(), "zzzz qqqq")


def test_sky_keywords():
    for k in E.SKIES:
        cv = _cv()
        E.sky(cv, k)
        assert np.isfinite(cv.buf).all()


def test_layer_recorder_sees_effects():
    from pixelcanvas import layers
    cv = _cv()
    rec = layers.LayerRecorder(cv)
    rec.mark("sky")
    E.sky(cv, "night")
    rec.mark("fire")
    E.flame(cv, at=(0.5, 0.9), size=500, glow=0)
    stack = rec.finish()
    names = [L.name for L in stack]
    assert "sky" in names and "fire" in names


def _doc_blocks():
    text = open(DOC, encoding="utf-8").read()
    return re.findall(r"```python\n(.*?)```", text, re.S)


def test_doc_is_short():
    assert len(open(DOC, encoding="utf-8").read().splitlines()) < 250


@pytest.mark.parametrize("i", range(len(_doc_blocks())))
def test_doc_snippets_run(i):
    code = _doc_blocks()[i]
    exec(compile(code, f"elements.md block {i}", "exec"), {"__name__": "__doc__"})


def test_particle_counts_follow_the_frame_not_the_pixels():
    # a --scale 0.5 draft must look like the final: same coverage at both resolutions
    for fn in (lambda cv: E.rain(cv, seed=1), lambda cv: E.snowfall(cv, seed=1, bokeh=False)):
        cov = []
        for w, h in ((480, 270), (960, 540)):
            cv = Canvas(w, h, "#000000")
            fn(cv)
            cov.append(float(cv.buf.mean()))
        assert 0.6 < cov[0] / max(cov[1], 1e-9) < 2.5          # pixel-based counts would give ~0.25
    assert abs(E._area(Canvas(480, 270)) - E._area(Canvas(1920, 1080))) < 1e-6


def test_registry_places_region_effects_with_at_and_size():
    for name in ("stars", "rain", "droplets", "snow", "bokeh"):
        cv = Canvas(240, 135, "#000000")
        before = cv.buf.copy()
        E.element(cv, name, at=(0.2, 0.3), size=300, seed=2)
        d = np.abs(cv.buf - before).max(axis=2)
        ys, xs = np.nonzero(d > 0.02)
        assert len(xs), name
        assert xs.mean() / 240 < 0.35 and ys.mean() / 135 < 0.5, name


@pytest.mark.parametrize("name", ["embers", "god rays", "spotlight", "lens flare", "aurora", "rainbow"])
def test_registry_size_never_breaks_a_call(name):
    cv = _cv()
    E.sky(cv, "night")
    E.element(cv, name, at=(0.5, 0.5), size=200, seed=1)
    assert np.isfinite(cv.buf).all()


def test_waterfall_foot_can_be_repainted_over_the_pool():
    cv = _cv()
    full = E.waterfall(cv, box=(0.4, 0.1, 0.6, 0.8), seed=1)
    cv2 = _cv()
    foot = E.waterfall(cv2, box=(0.4, 0.1, 0.6, 0.8), seed=1, fall=False)
    assert foot.mask.sum() < full.mask.sum()
    assert foot.mask[: int(cv.h * 0.4)].max() < 0.05          # nothing high up on the fall


def test_stone_and_firewood_are_lit_by_a_fire():
    cold, warm = Canvas(200, 120, "#000000"), Canvas(200, 120, "#000000")
    E.stone(cold, at=(0.6, 0.8), size=150, seed=3)
    fire = E.Light(80, 90, (1.0, 0.5, 0.2), 120, 1.5)
    fx = E.stone(warm, at=(0.6, 0.8), size=150, seed=3, lit_by=fire)
    assert fx.mask.max() > 0.9
    assert warm.buf[..., 0].sum() > cold.buf[..., 0].sum() * 1.05
    cv = Canvas(200, 120, "#000000")
    fx = E.firewood(cv, start=(0.2, 0.8), end=(0.8, 0.7), width=80, lit_by=fire)
    assert fx.mask.max() > 0.9 and np.isfinite(cv.buf).all()


def test_moon_reflection_is_a_bright_glitter_not_a_dark_disc():
    cv = Canvas(320, 200, "#05070f")
    E.moon(cv, at=(0.5, 0.25), size=140, glow=0)
    E.lake(cv, horizon=0.5)
    col = cv.buf[110:190, 140:180].max(axis=2)
    assert col.max() > 0.6                                    # bright glints under the moon
    assert (col > 0.3).mean() > 0.05


def test_smoke_thins_as_it_spreads():
    cv = Canvas(200, 300, "#000000")
    fx = E.smoke(cv, at=(0.5, 0.95), size=1000, wind=0.0, seed=2)
    low = fx.mask[240:270].max()
    high = fx.mask[60:120].max()
    assert low > high > 0.02


def test_candle_base_stands_the_wax_on_a_surface():
    cv = Canvas(200, 300, "#000000")
    sill = np.zeros((300, 200), np.float32)
    sill[240:] = 1
    cv.paint("#402a1c", sill)
    before = cv.buf.copy()
    fx = E.candle(cv, base=(0.5, 0.8), size=60, body_height=120, light_on=sill, seed=1)
    lit = cv.buf[262:300, 10:40] - before[262:300, 10:40]        # sill away from the candle got warmer
    assert lit[..., 0].mean() > 0.003
    plain = Canvas(200, 300, "#000000")
    E.candle(plain, base=(0.5, 0.8), size=60, body_height=120, glow=0, seed=1)
    rows = np.nonzero(plain.buf.max(axis=2)[:, 100] > 0.03)[0]
    assert abs(rows.max() - 240) <= 5                            # the wax ends at the base, not below it
    assert abs(rows.min() - (240 - (120 + 60 * 1.24) * 300 / 1080)) < 8   # flame tip above the wick
    assert fx.light is not None and fx.light.y < 240 - 120 * 300 / 1080


def test_rainbow_shows_through_a_faint_spray_mask():
    cv = Canvas(240, 135, "#203040")
    before = cv.buf.copy()
    spray = np.full((135, 240), 0.1, np.float32)
    E.rainbow(cv, center=(0.5, 1.1), radius=0.8, mask=spray, secondary=False)
    assert np.abs(cv.buf - before).max() > 0.1


def test_waterfall_returns_a_spray_mask_for_the_rainbow():
    cv = Canvas(240, 135, "#203040")
    fx = E.waterfall(cv, box=(0.4, 0.1, 0.6, 0.8), seed=1)
    assert fx.spray is not None and fx.spray.max() > 0.9
    rows = np.nonzero(fx.spray.max(axis=1) > 0.2)[0]
    assert rows.min() > 135 * 0.4                                 # the mist sits at the foot of the fall


def test_droplets_cover_the_glass_with_fine_beads():
    cv = Canvas(320, 180, "#101826")
    E.bokeh(cv, n=12, size=40, seed=1)
    before = cv.buf.copy()
    E.droplets(cv, region=(0, 0, 1, 1), seed=2)
    diff = np.abs(cv.buf - before).max(axis=2)
    assert (diff > 0.02).mean() > 0.15                            # beads all over, not a few marbles
    assert np.isfinite(cv.buf).all()


def test_aurora_curtains_glow_upwards_from_a_bright_hem():
    cv = Canvas(320, 180, "#000000")
    fx = E.aurora(cv, v_range=(0.5, 0.6), seed=3, height=300, curtains=1)
    col = cv.buf.sum(axis=2)
    prof = col.mean(axis=1)
    peak = int(np.argmax(prof))
    assert prof[: max(peak - 30, 1)].mean() < prof[peak]          # fades above the hem
    assert prof[min(peak + 25, 179):].mean() < prof[peak] * 0.3     # sharp below it


def test_strict_options_aliases_and_names():
    cv = _cv()
    with pytest.raises(TypeError, match="did you mean 'size'"):
        E.element(cv, "campfire", sise=200)
    with pytest.raises(TypeError, match="did you mean 'opacity'"):
        E.chimney_smoke(cv, opacty=0.5)
    E.smoke(cv, colour="#888888", alpha=0.5)                       # colour -> color, alpha -> opacity
    with pytest.raises(ValueError, match="unknown element 'qqzx'.*elements.names"):
        E.element(cv, "qqzx")
    with pytest.raises(KeyError):                                  # older code catching KeyError keeps working
        E.element(cv, "qqzx")
    with pytest.raises(ValueError, match="did you mean 'sunset'"):
        E.sky(cv, "sunnset")


def test_smoke_phase_moves_and_loops():
    def draw(**k):
        c = _cv(200, 200)
        E.smoke(c, at=(0.5, 0.9), size=400, seed=2, **k)
        return c.buf.copy()
    a, b = draw(), draw(phase=0.25)
    assert np.abs(a - b).mean() > 1e-4                              # the turbulence moved
    assert np.abs(a - draw(phase=1.0)).max() < 1e-5                 # a full turn returns to the start
    assert np.abs(a - draw(offset=(0.0, 0.0))).max() < 1e-6


def test_lake_has_no_vertical_stripes():
    c = _cv(480, 270, "#c0a080")
    E.lake(c, horizon=0.5, seed=3)
    water = c.buf[int(270 * 0.6):, :, :].mean(-1)
    cols = (water - water.mean(0, keepdims=True) * 0 ).mean(0)
    fine = cols - np.convolve(cols, np.ones(9) / 9, "same")         # column-to-column ripple of the row average
    assert np.abs(fine[8:-8]).max() < 0.01


# ---- round 1 fixes: steam on any ground, Milky Way, reflect, pixels, every call records itself
from pixelcanvas.elements import steam, breath, stars, lake, campfire, reflect, sky, element  # noqa: E402

def _flat(color, w=360, h=300):
    return Canvas(w, h, color)


def _steam_contrast(color, **kw):
    """Largest colour difference between the steam and the bare ground."""
    cv = _flat(color)
    before = np.array(cv.buf)
    steam(cv, at=(0.5, 0.95), size=420, **kw)
    return float(np.abs(np.array(cv.buf) - before).max())


@pytest.mark.parametrize("ground", ["#efc99a", "#fbfaf5", "#e8e4dc", "#7a8aa0", "#1a2030", "#05070d"])
def test_steam_is_visible_on_light_and_dark_grounds(ground):
    assert _steam_contrast(ground) > 0.07


def test_steam_color_and_opacity_work():
    red = _flat("#fbfaf5")
    steam(red, at=(0.5, 0.95), size=420, color="#ff3030")
    px = np.array(red.buf).reshape(-1, 3)
    assert (px[:, 0] - px[:, 1]).max() > 0.12
    assert _steam_contrast("#1a2030", opacity=0.9) > _steam_contrast("#1a2030", opacity=0.3)


def test_steam_on_light_ground_has_a_darker_soft_edge():
    cv = _flat("#fbfaf5")
    steam(cv, at=(0.5, 0.95), size=420)
    lum = np.array(cv.buf).mean(-1)
    assert lum.min() < 0.93                                 # the rim is darker than the wall


def test_breath_reads_on_light_and_dark_grounds():
    for ground in ("#fbfaf5", "#1a2030"):
        cv = _flat(ground)
        before = np.array(cv.buf)
        breath(cv, at=(0.5, 0.5), size=160)
        assert np.abs(np.array(cv.buf) - before).max() > 0.05


def test_stars_with_milky_way_records_it():
    from pixelcanvas import manifest
    cv = Canvas(400, 240, "#05070d")
    stars(cv, milky_way=True)
    rows = {r["name"]: r for r in manifest.summary(cv)}
    assert "Milky Way" in rows and "stars" in rows
    assert {"stars", "night sky"} <= set(rows["Milky Way"]["aliases"])
    cv2 = Canvas(400, 240, "#05070d")
    stars(cv2, milky_way=False)
    assert [r["name"] for r in manifest.summary(cv2)] == ["stars"]


def test_backdrop_and_see_through_flags_are_passed():
    from pixelcanvas import manifest
    cv = Canvas(400, 240, "#808080")
    sky(cv, "day")
    stars(cv)
    lake(cv, horizon=0.6)
    campfire(cv, at=(0.4, 0.85), size=120)
    metas = {e["name"]: e.get("meta", {}) for e in manifest.entries(cv)}
    assert metas["sky"].get("backdrop") and metas["stars"].get("backdrop") and metas["Milky Way"].get("backdrop")
    assert metas["lake"].get("see_through")
    assert not metas["campfire"].get("see_through")


def test_lake_and_reflect_alias_water():
    from pixelcanvas import manifest
    cv = Canvas(400, 240)
    sky(cv, "night")
    lk = lake(cv, horizon=0.6)
    f = campfire(cv, at=(0.4, 0.8), size=100)
    reflect(cv, of=f, water=lk, axis=0.8)
    rows = {r["name"]: r for r in manifest.summary(cv)}
    assert "water" in rows["lake"]["aliases"] and "reflection" in rows["lake"]["aliases"]
    assert "reflection" in rows and rows["reflection"]["aliases"]


def test_reflect_mirrors_a_light_and_an_object_painted_after_the_lake():
    from pixelcanvas import sdf
    cv = Canvas(480, 320, "#05070d")
    lk = lake(cv, horizon=0.55)
    below = np.array(cv.buf)
    fire = campfire(cv, at=(0.3, 0.72), size=120)
    X, Y = cv.xy()
    tent = sdf.fill(sdf.sd_polygon(X, Y, [(0.6 * 480, 0.72 * 320), (0.8 * 480, 0.72 * 320), (0.7 * 480, 0.5 * 320)]))
    cv.paint("#d8883a", tent)
    pre = np.array(cv.buf)
    out = reflect(cv, of=[fire, tent], water=lk, axis=0.72)
    assert out.mask is not None and out.mask.max() > 0.2
    diff = np.abs(np.array(cv.buf) - pre).max(-1)
    assert diff[int(0.74 * 320):int(0.95 * 320), int(0.62 * 480):int(0.78 * 480)].max() > 0.1     # tent
    assert diff[int(0.74 * 320):int(0.95 * 320), int(0.22 * 480):int(0.38 * 480)].max() > 0.1     # fire
    assert diff[:int(0.7 * 320)].max() < 1e-6                                                    # nothing above the axis
    # a box works too, a wrong mask raises
    reflect(cv, of=(0.55, 0.5, 0.85, 0.72), water=lk)
    with pytest.raises(ValueError, match="full-canvas mask"):
        reflect(cv, of=np.ones((5, 5), np.float32))
    with pytest.raises(ValueError, match="looks like pixels"):
        reflect(cv, of=(100, 100, 200, 200))


def test_positions_in_pixels_raise():
    cv = Canvas(320, 200)
    with pytest.raises(ValueError, match=r"looks like pixels"):
        steam(cv, at=(300, 100))
    with pytest.raises(ValueError, match=r"looks like pixels"):
        campfire(cv, at=(100, 100))
    with pytest.raises(ValueError, match=r"looks like pixels"):
        lake(cv, horizon=300)
    with pytest.raises(ValueError, match=r"looks like pixels"):
        element(cv, "campfire", at=(500, 300))


def test_every_registry_effect_records_itself():
    from pixelcanvas import manifest
    from pixelcanvas.elements import REGISTRY, element, names
    missing = []
    for name in names():
        cv = Canvas(160, 100, "#30384a")
        try:
            element(cv, name, at=(0.5, 0.5), size=700, seed=1)
        except Exception as e:           # noqa: BLE001
            raise AssertionError("%s failed: %r" % (name, e))
        if not manifest.entries(cv):
            missing.append(name)
    assert not missing, missing
