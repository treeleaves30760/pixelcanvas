# -*- coding: utf-8 -*-
"""pixelcanvas.ui: auto layout, components, styles, hooks (glass / images), boxes, SVG, templates, doc snippets."""
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pixelcanvas import Canvas  # noqa: E402
from pixelcanvas import ui  # noqa: E402
from pixelcanvas.ui import Screen, Presentation, template, TEMPLATES, wrap  # noqa: E402
from pixelcanvas.vector import ICONS  # noqa: E402

DOC = os.path.join(ROOT, "skills", "pixelcanvas", "references", "ui.md")


def _walk(n):
    yield n
    for c in ui._kids(n):
        yield from _walk(c)


def test_ui_icons_registered():
    for name in ("chevron-right", "plus", "close", "menu", "more", "arrow-left", "send", "mic", "filter", "edit",
                 "trash", "eye", "bookmark", "share", "sliders", "moon", "sun", "credit-card", "wallet", "map",
                 "compass", "music", "pause", "skip-forward", "volume", "battery", "signal", "bluetooth"):
        assert name in ICONS
    assert ui._icon_name("settings") == "gear" and ui._icon_name("arrow_left") == "arrow-left"


def test_wrap_and_ellipsis():
    lines = wrap("The quick brown fox jumps over the lazy dog " * 4, 15, 160)
    assert len(lines) > 3
    assert all(ui._tw(l, 15) <= 161 for l in lines)
    cut = wrap("The quick brown fox jumps over the lazy dog " * 4, 15, 160, max_lines=2)
    assert len(cut) == 2 and cut[-1].endswith("…")
    assert ui._ellipsize("Supercalifragilistic", 15, 60).endswith("…")


def test_stack_sizing_fill_hug_fixed():
    s = Screen("plain", size=(400, 300), pad=0, gap=0)
    with s.hstack(gap=10, name="row") as row:
        a = s.button("Hug me", name="a")
        b = s.spacer()
        c = s.vstack(w=120, name="fixed")
        d = s.vstack(w="fill", name="fill")
    s._layout_all()
    assert row.frame[2] == 400
    assert c.frame[2] == 120
    assert a.frame[2] < 140
    assert abs((b.frame[2] + d.frame[2]) - (400 - 30 - a.frame[2] - 120)) < 0.01
    with s.grid(cols=3, gap=12) as gr:
        for i in range(5):
            s.card(name=f"cell{i}")
    s._laid = False
    s._layout_all()
    ws = [c.frame[2] for c in gr.children]
    assert max(ws) - min(ws) < 1e-6 and abs(ws[0] - (400 - 24) / 3) < 1e-6


def test_nothing_overflows_parent():
    s = template("dashboard", "phone")
    s._layout_all()
    for top in ui._kids(s.root):
        for n in _walk(top):
            if n.kind not in ("stack", "grid") or n.p.get("scroll") or n.p.get("wrap"):
                continue
            x, y, w, h = n.frame
            for c in ui._kids(n):
                cx, cy, cw, ch = c.frame
                assert cx >= x - 0.5 and cx + cw <= x + w + 0.5, (n.key(), c.key())
    assert s.root.frame[2] == s.vw


def test_text_truncates_inside_box():
    s = Screen("plain", size=(200, 100), pad=0)
    t = s.text("A very long title that cannot possibly fit on one line here", role="title", max_lines=1)
    s._layout_all()
    assert t.frame[3] == ui._lh(17, 1.3)
    assert t.frame[2] <= 200


def test_every_component_renders_in_every_style():
    for style in ui.STYLES:
        for theme in ("light", "dark"):
            s = Screen("phone", theme=theme, style=style, accent="#e4572e")
            s.app_bar("Home", leading="menu", trailing=["bell", "avatar:Ana Silva"])
            with s.scroll():
                s.heading("Hi", level=2)
                s.search()
                s.segmented(["A", "B"], active=1)
                s.tabs(["One", "Two"], counts={1: 4})
                with s.card():
                    s.stat("Revenue", "$4.2k", delta="+3%", spark=[1, 3, 2, 4])
                    s.progress(0.4, label="Sync")
                    s.ring(0.6, size=64)
                with s.list_group("Group"):
                    s.list_row("Row", "Sub", leading="bell", chevron=True)
                    s.list_row("Toggle", leading="avatar:Maya Chen", toggle=False)
                s.chips(["a", "b"], selected=[0])
                s.checkbox("Check", checked=True)
                s.radio("Radio")
                s.slider(0.3, label="Volume")
                s.stepper(2, label="Guests")
                s.badge("New")
                s.avatar_group(["A B", "C D", "E F"])
                s.table(["Name", "Amount"], [["Ana", "$12"], [{"badge": "Paid", "tone": "success"}, "-$3"]])
                s.banner("Heads up", "Something happened.", tone="warning")
                s.message("Hello there", time="9:41")
                s.message("Hi!", me=True, time="9:42")
                s.notification("Build passed", "main · 2m", time="now", unread=True)
                s.calendar(2026, 2, selected=14)
                s.timeline([("9:00", "Start", "Kick-off"), ("10:00", "Review")])
                s.image(height=80)
                s.map(height=80)
                s.video(height=90)
                s.waveform()
                s.code("x = 1  # one")
                s.terminal(["$ make", "✓ done"])
                s.skeleton(2, avatar=True)
                s.empty_state("Empty", "Nothing yet", action="Add")
                s.menu([("edit", "Edit"), "-", ("trash", "Delete")])
                s.tooltip("Tip")
                s.breadcrumb(["A", "B", "C"])
                s.pagination(2, 9)
                s.kanban({"To do": ["Task"], "Done": [dict(title="Ship", tag="Eng")]})
            s.tab_bar(["home", "user"], active=0)
            s.fab()
            s.toast("Saved", action="Undo")
            cv = s.render(scale=0.5)
            assert cv.w > 100 and np.isfinite(cv.buf).all()


def test_box_maps_to_canvas_pixels():
    s = Screen("phone", theme="light")
    s.app_bar("Inbox")
    btn = s.button("Save", name="save")
    cv = Canvas(800, 1000, "#000000")
    s.draw(cv)
    x0, y0, x1, y1 = s.box("save")
    assert 0 <= x0 < x1 <= cv.w and 0 <= y0 < y1 <= cv.h
    col = cv.buf[(y0 + y1) // 2, x0 + 4]
    assert np.allclose(col, ui._rgb(s.T.accent), atol=0.06)
    assert s.box("button:Save") == s.box(btn)
    cv2 = Canvas(800, 1000)
    s.draw(cv2, box=(400, 0, 800, 500))
    b2 = s.box("save")
    assert b2[0] >= 400 and b2[3] <= 500


def test_glass_blurs_backdrop_and_images_paint():
    s = Screen("plain", size=(300, 200), style="glass", theme="dark", pad=20)
    with s.card(h=160, name="panel"):
        s.text("Glass")
    cv = s.render(scale=1)
    hooks = []
    d = s.build()

    def walk(it):
        if ui._is_hook(it):
            hooks.append(it)
        for c in getattr(it, "children", []):
            walk(c)
    walk(d.root)
    assert any(h._ui_hook["kind"] == "glass" for h in hooks)
    src = Canvas(60, 40, "#ff0000")
    s2 = Screen("plain", size=(200, 120), pad=10)
    img = s2.image(height=100, canvas=src, name="pic")
    cv2 = s2.render(scale=1)
    x0, y0, x1, y1 = s2.box("pic")
    assert np.allclose(cv2.buf[(y0 + y1) // 2, (x0 + x1) // 2], (1, 0, 0), atol=0.02)
    svg = s2.to_svg()
    assert "<image" in svg and "data:image/png;base64" in svg
    ET.fromstring(svg.encode("utf-8"))
    assert img.frame is not None and cv.w == 300


def test_svg_is_valid_for_every_device():
    for dev in ui.DEVICES:
        s = Screen(dev, theme="dark", title="App", url="example.com")
        s.heading("Hello")
        s.button("Go")
        root = ET.fromstring(s.to_svg().encode("utf-8"))
        assert root.tag.endswith("svg")


def test_templates_build_and_render():
    for name in TEMPLATES:
        for scr in ("phone", "browser"):
            s = template(name, scr, theme="dark" if scr == "browser" else "light")
            s._layout_all()
            assert s.names()
    s = template("login", "phone", content={"title": "Hello again"})
    assert s._find("text:Hello again") is not None
    cv = template("weather", "phone").render(scale=0.4)
    assert cv.h > 300
    with pytest.raises(ValueError):
        template("nope")


def test_presentation_and_annotate():
    a = template("login", "phone")
    b = template("settings", "phone")
    p = Presentation([a, b], size=(960, 540), title="Two screens", tilt=0)
    cv = Canvas(960, 540)
    p.draw(cv)
    box = p.box("button:Sign in", screen=0)
    assert 0 <= box[0] < box[2] <= 960
    out = ui.annotate(cv, a, {"button:Sign in": "Primary action"}, kind="marker")
    assert out and out[0][1] == box
    iso = Presentation([a], size=(640, 360), tilt="iso").render()
    assert iso.w == 640


def test_doc_snippets_run(tmp_path, monkeypatch):
    if not os.path.exists(DOC):
        pytest.skip("ui.md not written")
    text = open(DOC, encoding="utf-8").read()
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    assert blocks
    monkeypatch.chdir(tmp_path)
    os.makedirs("outputs", exist_ok=True)
    for code in blocks:
        exec(compile(code, "ui.md", "exec"), {})


def test_kanban_card_tags_and_avatar_aliases():
    s = Screen("browser", theme="light")
    s.kanban({"To do": [{"title": "A", "tags": ["Dev", "UX"], "avatar": "Ana Silva", "due": "Nov 18"}]})
    cv = s.render(scale=0.5)
    assert cv.w > 0


def test_draw_backdrop_fills_canvas_and_desktop_takes_aspect():
    from pixelcanvas import Canvas
    s = Screen("browser", theme="light")
    s.heading("Hi", level=2)
    cv = Canvas(960, 540, "#ff0000")
    s.draw(cv)
    a = cv.to_array() if hasattr(cv, "to_array") else cv.buf
    assert float(a[:, :, 0].min()) < 0.99 or float(a[:, :, 1].max()) > 0.1       # no pure red left over
    assert not ((a[5, 5] > 0.99)[0] and (a[5, 5] < 0.01)[1])
    assert abs(s.vh - 800) > 1                                                        # viewport adapted to 16:9


def test_dark_modern_accent_tints_neutrals():
    from pixelcanvas.ui import tokens
    a, b = tokens("dark", "modern"), tokens("dark", "modern", "#8b5cf6")
    assert a.bg != b.bg


def test_strict_options_ui():
    s = ui.Screen()
    with pytest.raises(TypeError) as e:
        s.kanban_card("x", tagz="a")
    assert "unknown option 'tagz'" in str(e.value) and "'tag'" in str(e.value)
    s.card(background="#fff")                    # alias of bg
    with pytest.raises(ValueError, match="'phone'"):
        ui.Screen(device="phon")
    with pytest.raises(ValueError, match="'dark'"):
        ui.Screen(theme="dak")


# ---- standalone windows (ui.window) and component records -------------------------------------------------------

@pytest.mark.parametrize("kind", ui.WINDOW_KINDS)
def test_window_kinds_draw_inside_their_box(kind):
    cv = Canvas(640, 400, "#20304a")
    before = cv.buf.copy()
    wb = ui.window(cv, kind, box=(0.5, 0.1, 0.95, 0.9))
    x0, y0, x1, y1 = wb
    assert 0.5 * 640 - 1 <= x0 < x1 <= 0.95 * 640 + 1 and 0.1 * 400 - 1 <= y0 < y1 <= 0.9 * 400 + 1
    changed = np.abs(cv.buf - before).max(-1) > 1e-3
    assert not changed[:, :int(0.5 * 640) - 1].any()          # nothing outside the box
    assert changed[y0 + 5:y1 - 5, x0 + 5:x1 - 5].mean() > 0.3
    assert wb.kind == kind


def test_window_hugs_its_content():
    short = ui.window(Canvas(1000, 1000), "terminal", ["$ ls"], box=(0, 0, 1, 1), cursor=False)
    tall = ui.window(Canvas(1000, 1000), "terminal", ["$ ls"] + ["out %d" % i for i in range(12)], box=(0, 0, 1, 1))
    asp = lambda b: (b[2] - b[0]) / (b[3] - b[1])                                                     # noqa: E731
    assert asp(short) > 1.6 and asp(tall) < asp(short)       # a 1-line terminal is wide, not a tall portrait box


def test_window_terminal_records_window_and_commands():
    from pixelcanvas import manifest
    cv = Canvas(1600, 640)
    wb = ui.window(cv, "terminal", ["$ driftwood init", "$ driftwood push", "$ driftwood pull --host laptop",
                                    "✓ 42 files in sync"], box=(0.52, 0.1, 0.96, 0.9))
    assert wb.commands == 3
    names = {e["name"]: e for e in cv.manifest}
    assert "terminal window" in names and "terminal" in names["terminal window"]["aliases"]
    assert names["command"]["meta"]["count"] == 3
    issues = manifest.check(cv, "a stylised terminal window showing three commands")
    assert not [w for w in issues if w["rule"] in ("prompt_count", "object_hidden")], issues
    from pixelcanvas.text import audit_text
    assert not [i for i in (audit_text(cv) or []) if "px high" in str(i)]   # readable size, no "4 px high"


def test_window_options_are_checked():
    cv = Canvas(200, 120)
    with pytest.raises(ValueError, match="'terminal'"):
        ui.window(cv, "termnal")
    with pytest.raises(ValueError, match="'dark'"):
        ui.window(cv, "terminal", theme="drak")
    with pytest.raises(TypeError, match="did you mean 'prompt'"):
        ui.window(cv, "terminal", promt="$")


def test_window_browser_shows_a_canvas():
    page = Canvas(320, 200, "#ff0000")
    cv = Canvas(640, 480, "#000000")
    wb = ui.window(cv, "browser", page, box=(0, 0, 1, 1), url="acme.dev")
    x0, y0, x1, y1 = wb.content
    px = cv.buf[(y0 + y1) // 2, (x0 + x1) // 2]
    assert px[0] > 0.8 and px[1] < 0.2


def _meditation():
    s = Screen("phone", theme="dark", accent="#9b7cff", name="home")
    s.heading("Good evening, Sam", level=1)
    s.ring(0.62, label="4:32", sub="Breathe in", size=300)
    with s.grid(cols=3, gap=10):
        for name in ("Sleep", "Focus", "Calm"):
            with s.card(pad=14):
                s.title(name)
    s.tab_bar([("home", "Today"), ("music", "Sounds"), ("user", "Profile")], active=0)
    return s


@pytest.mark.parametrize("how", ["screen", "presentation"])
def test_components_record_themselves(how):
    from pixelcanvas import manifest
    cv = Canvas(540, 960)
    if how == "screen":
        _meditation().draw(cv)
    else:
        Presentation([_meditation()], size=(540, 960)).draw(cv)
    by = {e["name"]: e for e in cv.manifest if e["kind"] == "object"}
    assert {"ring", "tab bar", "card"} <= set(by)
    assert by["card"]["meta"]["count"] == 3 and by["ring"]["meta"]["count"] == 1
    assert "timer" in by["ring"]["aliases"] and by["tab bar"]["meta"]["see_through"]
    cov = manifest.coverage(cv, "a big circular breathing timer, three session cards and a bottom tab bar")
    assert all(c["status"] == "ok" for c in cov), cov
    bad = [w for w in manifest.check(cv, "three session cards") if w["rule"] in ("prompt_count", "object_hidden")]
    assert not bad, bad


def test_window_browser_canvas_replaces_placeholder_page():
    page = Canvas(320, 200, "#ff0000")
    cv = Canvas(960, 540, "#000000")
    wb = ui.window(cv, "browser", page)
    x0, y0, x1, y1 = wb
    for xx, yy in ((x1 - 2, (y0 + y1) // 2), (x0 + 2, y1 - 20), ((x0 + x1) // 2, y1 - 3)):
        px = cv.buf[yy, xx]
        assert px[0] > 0.9 and px[1] < 0.1 and px[2] < 0.1, (xx, yy, px)
    # no built-in page text is drawn when the caller supplies the page
    box = []
    orig = ui._window_drawing
    ui._window_drawing = lambda *a, **k: (box.append(orig(*a, **k)[0]), orig(*a, **k))[1]
    try:
        ui.window(Canvas(960, 540, "#000000"), "browser", page)
    finally:
        ui._window_drawing = orig
    svg = box[0].to_svg()
    assert "Build faster" not in svg and "Get started" not in svg


def test_hero_fits_a_portrait_screen_to_the_media_box():
    app = template("dashboard", "phone", theme="light")
    s = Screen("browser", theme="light", size=(1440, 780), url="x.io")
    with s.section(pad=(48, 32)):
        s.hero("Ship product updates", "Sub text here.", media=app)
    cv = Canvas(960, 540)
    Presentation([s], size=(960, 540), background="soft", pad=0.05).draw(cv)
    row = cv.buf[int(540 * 0.94) : int(540 * 0.95), :, :3].mean(axis=(0, 2))
    assert row.min() > 0.5, "the phone shot runs off the bottom of the page"
