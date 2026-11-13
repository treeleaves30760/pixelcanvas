# UI mock-ups (`pixelcanvas.ui`)

Describe a screen declaratively — app bars, cards, lists, forms, charts — and get a Figma / Dribbble-grade shot in a
realistic device frame. Layout is automatic (flexbox-like stacks and grids), text wraps and truncates with an
ellipsis inside its box, and every screen is a `pixelcanvas.vector` Drawing: the same picture renders to a Canvas
(one layer per section) and saves as SVG.

Options are checked: an unknown option (`tagz=` for `tag=`) raises a `TypeError` naming the closest valid one and the options that component takes, and both spellings are accepted (`tags` / `tag`, `background` / `bg`, `colour` / `color`).

```python
from pixelcanvas.ui import Screen
s = Screen("phone", theme="dark", accent="#7c5cff")            # phone tablet laptop browser desktop_window watch plain
s.app_bar("Wallet", leading="avatar:Ana Silva", trailing=["bell"])
with s.card():
    s.caption("Total balance"); s.heading("$12,480.20", level=1)
s.list_row("Coffee Lab", "Today, 8:12", leading="cart", value="-$4.50", chevron=True)
s.tab_bar([("home", "Home"), ("chart", "Stats"), ("user", "Profile")])
cv = s.render(scale=2)                                          # Canvas; s.save_svg("outputs/wallet.svg")
```

In a scene: `s.draw(cv, box=None, log=log)` fits the screen (with its frame) into the canvas or a box; with no box the backdrop fills the whole canvas and a
default-size browser / laptop window takes the canvas aspect (pass `size=` to keep a fixed viewport). Fastest start of all: `ui.template("dashboard", screen="browser", theme="dark")` returns a finished Screen.

## Windows: a terminal, browser, code editor or card in any box

`ui.window(cv, kind, content, box=...)` draws one window that hugs its content — title bar with traffic lights,
rounded corners, soft shadow — and fits it into any box of any canvas (README banners, hero images, slides).
A browser window given a Canvas shows that canvas as its page, filling the viewport; `hero(media=Screen)` fits the
screen (phone or browser) to the media box.

```python
from pixelcanvas import Canvas
from pixelcanvas import ui
cv = Canvas(1600, 640, "#0b1a3a")
ui.window(cv, "terminal", ["$ driftwood init", "$ driftwood push --all", "✓ 42 files in sync"],
          box=(0.52, 0.1, 0.96, 0.9), title="~/dotfiles — zsh", accent="#7ee8c7")
ui.window(cv, "code", "def hello(name):\n    return f'Hi {name}'", box=(0.04, 0.1, 0.48, 0.9), title="hello.py")
```

| kind | content | notes |
|---|---|---|
| `terminal` | lines; `"$ cmd"` = command (program bold, `--flags` and `"strings"` coloured), others = output (✓ / error / warn coloured) | `prompt="$"`, `cursor=True` adds a prompt line with a block cursor |
| `code` | source string or lines | syntax colours, line numbers, `title=` is the file tab |
| `browser` | `{"title", "text", "button"}` page, or a Canvas shown as the page | `url=` in the address bar |
| `card` | body lines | `title=` heading |

Common options: `theme="dark"/"light"`, `accent=`, `font_size=16` (points; the window scales with the box),
`width=` (points), `align="center"/"left"/"right"/"top"/"bottom"`, `shadow=True`, `radius=`. It returns the pixel
box (`.content` = the content area, `.commands` = number of commands) and records itself for `--check`
("terminal window", plus `command` ×N). `python tools/new_scene.py my_banner --kind banner` starts a finished
README banner built on it.

## Layout

| Call | What it does |
|---|---|
| `with s.vstack(gap=12, pad=0, align="stretch", justify="start"):` | column; `align` stretch / start / center / end; `justify` start / center / end / between |
| `with s.hstack(gap=12, align="center", wrap=False):` | row; `wrap=True` flows onto new lines (chips, tags) |
| `with s.grid(cols=2, gap=12):` | equal columns, equal-height rows; children may take `span=2` |
| `s.spacer()` · `s.spacer(24)` | flexible space (pushes siblings apart) or a fixed gap |
| `with s.card(pad=16, gap=12):` · `with s.list_group("Title"):` | card surface · grouped list with inset dividers |
| `with s.scroll(h="fill"):` | region that clips its content (the screen body always does) |
| `with s.section(bg=None, pad=(72, 32), max_w=1160):` | full-bleed web band with centred content |

Sizing keywords on **every** call: `w` / `h` = number, `"fill"` or `"hug"`; `max_w`; `grow` (share of free space);
`span` (grid); `self_align`; `pad` (around a component); `bg`; `name` (for `s.box(name)`). Pad is a number,
`(vertical, horizontal)` or `(top, right, bottom, left)`. Spacing follows a 4 / 8 point grid; type uses `ui.TYPE`
(display 44, h1 32, h2 26, h3 20, title 17, body 15, small 13, caption 12).

At the top level `app_bar` / `navbar` are pinned above the content and `tab_bar` below it; overlays (`fab`,
`toast`, `modal`, `sheet`, `menu(near=...)`, `tooltip(near=...)`) float over the screen.

## Components

| Group | Calls |
|---|---|
| Text | `heading(t, level=0..4)` `text(t, role=, color=, align=, max_lines=)` `title` `caption` `small` `label` `overline` `link` `divider(label=)` `icon(name, size, color)` `icon_badge(name, tone=)` |
| Buttons | `button(t, variant="primary"/"secondary"/"outline"/"ghost"/"link"/"destructive"/"tonal", icon=, icon_right=, size="sm"/"md"/"lg", full=)` `icon_button(icon, variant="ghost"/"filled"/"tonal"/"outline"/"surface", badge=)` `fab(icon, label=)` |
| Forms | `input(label, value=, placeholder=, icon=, trailing=, helper=, error=, focused=)` `text_area` `search` `select` `date_field` `password` `checkbox(label, checked)` `radio(label, selected)` `toggle(label, on)` `slider(value, label=)` `stepper(value, label=)` |
| Navigation | `app_bar(title, leading="back"/"menu"/"avatar:Name", trailing=[icons, "button:Text"], large=)` `tab_bar(items, active, badges=)` `sidebar(items, active, brand=, user=(name, email))` `navbar(brand, links, actions)` `tabs(items, active, counts=)` `segmented(items, active)` `breadcrumb(items)` `pagination(page, pages)` |
| Data | `list_row(title, subtitle, leading=icon/"avatar:Name"/"img:seed", value=, chevron=, toggle=, badge=, meta=)` `table(columns, rows)` `stat(label, value, delta=, icon=, spark=)` `badge(t, tone=)` `chip(t, selected=)` `chips(items, selected=)` `avatar(name, size, kind="portrait"/"initials", status=)` `avatar_group(names)` `progress(v, label=)` `ring(v, label=, sub=, size=96)` (size >= 140 gets a glow in dark themes) `skeleton(lines, avatar=, image=)` `chart(kind, data, x=, height=)` |
| Feedback | `banner(title, text, tone=)` `toast(text, action=, position="bottom"/"top"/"inline")` `tooltip(text, near=)` `menu(items, near=)` `with modal(title, text, actions=):` `with sheet(title):` `empty_state(title, text, icon, action)` |
| Content | `message(text, me=, time=)` `notification(title, text, time, unread=)` `calendar(year, month, selected=, events=)` `kanban({col: [card dicts]}, h=)` (card: `title`, `tags=[...]`, `avatar=`/`people=`, `due=`, `comments=`, `progress=`) `timeline([(time, title, text, icon)])` |
| Marketing | `hero(title, subtitle, actions=, kicker=, media=Screen/Canvas/"image"/"map")` `logo_row(names)` `feature_grid([(icon, title, text)])` `pricing(tiers, highlight=)` `testimonial(quote, name, role)` `footer(brand, columns=)` |
| Media | `image(height, scene="dawn"/"ocean"/"forest"/"dusk"/"desert"/"lilac", canvas=, caption=)` `map(height)` `video(height, title=)` `waveform(progress)` `audio(...)` `code(src, title=, highlight=)` `terminal(lines)` `screenshot(screen_or_canvas)` |

Tones (`tone=`, `color=`): `accent success warning danger info neutral` or a hex colour. Icons: every
`vector.icon_names()` name plus `ui.UI_ICONS` (chevrons, plus, close, menu, more, arrows, send, mic, filter, edit,
trash, eye, bookmark, share, sliders, moon, sun, credit-card, wallet, map, compass, music, pause, skip, volume,
battery, signal, bluetooth, message, cloud-sun, rain, ...).

**Screens and styles.** `Screen(device, theme="light"/"dark", accent="#hex", style="modern"/"glass"/"material"/"wireframe",
size=(w, h), height="auto", title=, url=, time="9:41")`. Glass blurs whatever is behind each panel at draw time over
a colourful backdrop; material tints surfaces with the accent and uses pill buttons; wireframe is greyscale with
placeholder lines for body copy. `Presentation(screens, background="soft", tilt=0 or "iso", stagger=0.05, title=)`
lays several screens out on a backdrop with soft shadows.

**Output.** `s.render(scale=2)` · `s.draw(cv, box, log)` · `s.save_svg(path)` · `s.build()` → vector Drawing ·
`s.box(name)` → pixel bbox on the last canvas (`"button:Save"`, your `name=`, or a substring) · `s.names()`.
Drawn screens record their components for `--check`: one entry per kind with a count (`card` ×3, `ring` with the
aliases timer / dial / circular, `tab bar`, `button`, `list row`, `avatar`, `chart` ... see `ui.COMPONENT_NAMES`),
so "three session cards and a tab bar" is counted without any `manifest.note`.

## Recipes

**Three screens of one app, case-study style**

```python
from pixelcanvas import Canvas
from pixelcanvas.ui import Screen, Presentation, template

def home():
    s = Screen("phone", theme="dark", accent="#ff7a59")
    s.app_bar("", leading="avatar:Ana Silva", trailing=["bell"], center=False)
    s.heading("Where to next?", level=1)
    s.search("Search destinations", trailing="sliders")
    s.chips(["All", "Beaches", "Cities"], selected=[1])
    s.image(height=180, scene="ocean", caption="Azure Bay", sub="Portugal", icon="heart")
    s.tab_bar([("home", "Explore"), ("heart", "Saved"), ("user", "Profile")])
    return s

screens = [home(), template("profile", "phone", theme="dark", accent="#ff7a59"),
           template("settings", "phone", theme="dark", accent="#ff7a59")]
cv = Canvas(1600, 900)
Presentation(screens, size=(1920, 1080), stagger=0.05).draw(cv)
```

**Web dashboard with sidebar, KPI tiles, chart and table**

```python
from pixelcanvas.ui import Screen
s = Screen("browser", theme="light", accent="#4f46e5", url="app.acme.io", title="Acme")
with s.hstack(gap=0, align="stretch", h="fill"):
    s.sidebar([("home", "Overview"), ("chart", "Analytics"), ("users", "Customers", 12)], brand="Acme",
              user=("Ana Silva", "ana@acme.io"))
    with s.vstack(gap=20, pad=28, w="fill"):
        s.heading("Overview", level=2)
        with s.grid(cols=3, gap=16):
            s.stat("Revenue", "$48.2k", delta="+12%", icon="dollar", spark=[3, 4, 3, 5, 6])
            s.stat("Orders", "1,284", delta="+8%", icon="cart")
            s.stat("Refunds", "12", delta="-2%", icon="refresh", good="down")
        with s.card():
            s.chart("area", {"2026": [3, 5, 4, 7, 6, 9]}, x=["Jan", "Feb", "Mar", "Apr", "May", "Jun"], height=180)
        s.table(["Customer", "Status", "Amount"], [[{"avatar": "Liam Carter"}, {"badge": "Paid", "tone": "success"},
                                                   "$249.00"]])
cv = s.render(scale=1)
s.save_svg("outputs/dashboard.svg")
```

**Glass style, overlays and an annotated spec**

```python
from pixelcanvas import Canvas
from pixelcanvas import ui
s = ui.Screen("phone", theme="dark", style="glass", accent="#8b5cf6")
s.app_bar("Now playing", leading="chevron-down", trailing=["more"])
s.image(aspect=1.0, scene="dusk", radius=28)
s.heading("Northern Lights", level=3)
s.waveform(0.38)
with s.hstack(justify="between", w="fill"):
    s.icon_button("skip-back"); s.icon_button("pause", variant="filled", size=64, name="play"); s.icon_button("skip-forward")
s.toast("Added to library", icon="check", action="Undo")
cv = Canvas(1400, 1100, "#0e0b1f")
s.draw(cv, box=(300, 0, 1100, 1100))
ui.annotate(cv, s, {"play": "Primary control, 64 pt", "toast": "Auto-dismiss after 4 s"}, theme="dark")
```

**Landing page with a product shot built from the kit**

```python
from pixelcanvas.ui import Screen, template
s = Screen("browser", theme="light", height="auto", url="northwave.io")
s.navbar("Northwave", links=["Product", "Pricing", "Docs"], actions=[("Sign in", "ghost"), "Start free"])
with s.section():
    s.hero("Release notes people read", "Announce every update in-app, by email and on the web.",
           kicker="New: AI summaries", media=template("dashboard", "browser", size=(1280, 760)))
with s.section(pad=(48, 32)):
    s.feature_grid([("edit", "One editor", "Write once."), ("chart", "Analytics", "See every read."),
                    ("globe", "Localised", "30 languages.")])
s.footer("Northwave")
cv = s.render(scale=0.5)
```

Templates (`ui.template(name, screen="phone"/"browser", theme=, accent=, style=, content={...})`): `login dashboard
settings chat feed profile checkout onboarding landing pricing music_player weather calendar kanban` — `content`
overrides the sample copy (e.g. `{"title": ..., "brand": ...}`), and you can keep adding components to the
returned Screen. Canvas images: `s.image(canvas=cv)` paints any Canvas into a rounded slot (cover / contain) and
embeds it in the SVG.
