# -*- coding: utf-8 -*-
"""Phone UI mock-up of a travel app, built declaratively with auto layout (ui)

* Format: 1920x1080 presentation: one phone screen, centred, on a soft backdrop
* Focal point: <the headline and the first image card>
* Screen: <app bar, greeting, search, chips, image cards, list rows, tab bar>
* Theme: <dark, one accent colour ACCENT, style STYLE>
* Layers: <one per screen>
* Exact text: <every label, name and price copied verbatim from the request>
"""
# Read skills/pixelcanvas/references/ui.md (Screen, components, Presentation, template).
from pixelcanvas import Canvas
from pixelcanvas.ui import Screen, Presentation, template

PROMPT = ""            # paste the request here, verbatim - --check compares the picture against it
KIND = "design"
NAME = "my_ui"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

# >>> data: new_scene.py --prompt writes the request's device, theme, app name and title here
DEVICE = "phone"                        # phone, tablet, browser, desktop ...
THEME = "dark"                          # dark or light
ACCENT = "#ff7a59"
STYLE = "modern"                        # modern, glass, material, wireframe
APP = "Wayfarer"
TITLE = "Where to next?"
# <<< data


# >>> screen: one function per screen; template(name, ...) gives a finished login / chat / weather ... screen
def home():
    s = Screen(DEVICE, theme=THEME, accent=ACCENT, style=STYLE, name="home")
    s.app_bar(APP, leading="avatar:Ana Silva", trailing=["bell"], center=False)      # no coordinates anywhere
    with s.vstack(gap=4):
        s.small("Good morning, Ana")
        s.heading(TITLE, level=1)
    s.search("Search destinations", trailing="sliders")
    s.chips(["All", "Beaches", "Mountains", "Cities"], selected=[1])
    with s.hstack(gap=12, align="start"):
        s.image(height=196, scene="ocean", seed=2, caption="Azure Bay", sub="Portugal", badge="\u2605 4.9", icon="heart",
                w="fill")
        s.image(height=196, scene="dusk", seed=5, caption="Red Cliffs", sub="Morocco", icon="heart", w="fill")
    with s.list_group("Popular stays"):
        s.list_row("Casa do Mar", "Lagos \u00b7 2 guests", leading="img:desert", value="$128", value_sub="/ night",
                   value_strong=True)
        s.list_row("Pine Lodge", "Sintra \u00b7 4 guests", leading="img:forest", value="$96", value_sub="/ night",
                   value_strong=True)
    s.tab_bar([("home", "Explore"), ("heart", "Saved"), ("calendar", "Trips"), ("user", "Profile")], active=0)
    return s
# <<< screen


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    log("home screen")                                   # add more screens to the list for a row of phones
    Presentation([home()], size=(width, height), background="soft").draw(cv, log=log)
    return cv
