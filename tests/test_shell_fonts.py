"""pixelcanvas.fonts: font lookup, masks and fitting for Latin and CJK; pixelcanvas imports no legacy pixelcanvas module."""
import ast
import pathlib

import numpy as np
import pytest

from pixelcanvas import fonts

PIXELCANVAS_DIR = pathlib.Path(fonts.__file__).resolve().parent
HAS_CJK = fonts.find_font(None, text="中文") is not None

LATIN = "Hello, Pixelcanvas"
CJK = "中文標題 測試"


def _ink_box(m, thr=0.02):
    ys, xs = np.nonzero(m > thr)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


@pytest.mark.parametrize("text", [LATIN, pytest.param(CJK, marks=pytest.mark.skipif(not HAS_CJK, reason="no CJK font"))])
def test_text_mask_draws_ink(text):
    m = fonts.text_mask(200, 400, text, (20, 30), size=40)
    assert m.shape == (200, 400) and m.dtype == np.float32
    assert 0.0 <= m.min() and m.max() <= 1.0
    assert m.sum() > 200 and (m > 0.5).sum() > 100
    x0, y0, x1, y1 = _ink_box(m)
    assert x0 >= 15 and y0 >= 25 and x1 <= 400 and y1 <= 200
    # a bigger size gives more ink; a stroke grows the glyphs
    assert fonts.text_mask(200, 400, text, (20, 30), size=60).sum() > m.sum()
    assert fonts.text_mask(200, 400, text, (20, 30), size=40, stroke=2.0).sum() > m.sum()


@pytest.mark.parametrize("text", [LATIN, pytest.param(CJK, marks=pytest.mark.skipif(not HAS_CJK, reason="no CJK font"))])
@pytest.mark.parametrize("box", [(300, 80), (180, 60), (500, 40)])
def test_fit_text_size_fits_the_box(text, box):
    bw, bh = box
    s = fonts.fit_text_size(text, bw, bh, min_size=4)
    assert s >= 4
    x0, y0, x1, y1 = fonts.text_bbox(text, (0, 0), size=s)
    assert x1 - x0 <= bw + 0.01 and y1 - y0 <= bh + 0.01
    x0, y0, x1, y1 = fonts.text_bbox(text, (0, 0), size=s + 1)
    assert x1 - x0 > bw + 0.01 or y1 - y0 > bh + 0.01          # the largest size that fits


def test_fit_wrapped_and_vertical():
    s = fonts.fit_text_size("several words wrapped into a narrow box", 160, 200, wrap=True, min_size=4)
    x0, y0, x1, y1 = fonts.text_bbox("several words wrapped into a narrow box", (0, 0), size=s, max_width=160)
    assert x1 - x0 <= 160.01 and y1 - y0 <= 200.01 and s > 12
    if HAS_CJK:
        sv = fonts.fit_text_size(CJK, 80, 400, vertical=True, min_size=4)
        m = fonts.text_mask(500, 200, CJK, (10, 10), size=sv, vertical=True)
        x0, y0, x1, y1 = _ink_box(m)
        assert y1 - y0 > x1 - x0                                 # a vertical column


def test_missing_glyphs_and_fonts():
    assert fonts.missing_glyphs(LATIN) == []
    if HAS_CJK:
        assert fonts.missing_glyphs(CJK) == []
    assert len(fonts.list_fonts("")) > 0
    f = fonts.load_font(None, 24, text=LATIN)
    assert f.getlength(LATIN) > 0
    assert fonts.has_cjk(CJK) and not fonts.has_cjk(LATIN)
    with pytest.raises(TypeError):
        fonts.text_mask(10, 10, "x", (0, 0), sise=12)


def _legacy_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "pixelcanvas" or (a.name.startswith("pixelcanvas.") and not a.name.startswith("pixelcanvas")):
                    bad.append(f"{path.name}:{node.lineno} import {a.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level >= 2:
                bad.append(f"{path.name}:{node.lineno} from {'.' * node.level}{mod} import ...")
            elif node.level == 0 and (mod == "pixelcanvas" or mod.startswith("pixelcanvas.")):
                if mod == "pixelcanvas":
                    names = [a.name for a in node.names if a.name != "pixelcanvas"]
                    if names:
                        bad.append(f"{path.name}:{node.lineno} from pixelcanvas import {', '.join(names)}")
                elif not mod.startswith("pixelcanvas"):
                    bad.append(f"{path.name}:{node.lineno} from {mod} import ...")
    return bad


def test_pixelcanvas_imports_no_legacy_module():
    files = sorted(PIXELCANVAS_DIR.rglob("*.py"))
    assert len(files) > 10
    bad = [b for f in files for b in _legacy_imports(f)]
    assert bad == [], "pixelcanvas imports legacy pixelcanvas modules: " + "; ".join(bad)
