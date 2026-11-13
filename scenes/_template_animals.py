# -*- coding: utf-8 -*-
"""Two animals in a natural setting: a horse and a fox in a summer meadow (creature + flora + elements)

* Format: 1920x1080
* Focal point: <the galloping horse on the left third, the fox watching from the right foreground>
* Light: <late-morning sun from the upper left (LIGHT); each animal casts a soft shadow to the right>
* Far / middle / near: <hazy tree line / horse on the field / fox, flowers and tall grass>
* Palette: <blue sky, warm greens, bay horse, red fox, flower accents>
* Exact text: <none>
"""
# Read skills/pixelcanvas/references/creatures.md (species, poses, coats) and flora.md (meadow, grass).
from pixelcanvas import Canvas
from pixelcanvas.creature import Creature
from pixelcanvas.elements import sky, cumulus
from pixelcanvas.flora import forest, meadow, grass
from pixelcanvas.texture import fill, plane_uv
from pixelcanvas.form import set_light
from pixelcanvas import clamp01

PROMPT = ""            # paste the request here, verbatim - --check compares the picture against it
NAME = "my_animals"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

HORIZON = 0.45
LIGHT = (-0.7, -0.62)                           # direction towards the light (upper left)
BIG = dict(species="horse", pose="gallop", at=(0.37, 0.72), size=0.36, facing="left")
SMALL = dict(species="fox", pose="sit", at=(0.8, 0.93), size=0.27, facing="left")      # species() lists 54 names
COAT = "palomino"                               # a COATS name: bay, black horse, grey, tabby ... or None
FLOWERS = ("poppy", "daisy", "cornflower")


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    X, Y = cv.xy()
    set_light(direction=LIGHT, elevation=42)
    # >>> setting: new_scene.py --prompt draws the request's place here (room, street, park, beach, night)
    log("sky")                                           # sky kind + one cloud
    sky(cv, "day", horizon=HORIZON)
    cumulus(cv, at=(0.7, 0.3), size=520, seed=seed, light=(-LIGHT[0], LIGHT[1]))
    log("trees")                                         # small and hazy = far away
    forest(cv, region=(-0.05, HORIZON - 0.04, 1.05, HORIZON + 0.01), n=40, height=(0.05, 0.11), haze=("#cfdcd2", 0.5),
           seed=seed + 1, light=LIGHT)
    log("field")                                         # texture name sets the material
    fill(cv, clamp01((Y / height - HORIZON) * height + 0.5), "moss", uv=plane_uv(height, width, horizon=HORIZON - 0.1),
         tint="#8fae52", light=LIGHT, seed=seed)
    meadow(cv, region=(0.0, 0.54, 1.0, 0.7), density=0.8, grass_height=0.03, seed=seed + 2, light=LIGHT)
    # <<< setting
    # >>> subjects: new_scene.py --prompt writes one call per subject of the request here
    log("big animal")                                    # pose, phase (stride 0~1), cast = ground shadow
    a = Creature(BIG["species"], size=BIG["size"], facing=BIG["facing"], coat=COAT, seed=seed)
    a.pose(BIG["pose"], phase=0.25).draw(cv, at=BIG["at"], cast=dict(opacity=0.35, length=0.6))
    log("meadow")                                        # flowers in front of the horse, behind the fox
    meadow(cv, region=(0.0, 0.7, 1.0, 0.86), flowers=FLOWERS, grass_height=0.05, seed=seed + 3, light=LIGHT)
    log("small animal")
    b = Creature(SMALL["species"], size=SMALL["size"], facing=SMALL["facing"], view="three-quarter", seed=seed)
    b.pose(SMALL["pose"], head_turn=12.0).draw(cv, at=SMALL["at"])
    log("foreground grass")
    grass(cv, region=(0.0, 0.88, 1.0, 1.02), tufts=40, height=0.09, blades=24, seed=seed + 4, wind=0.12, light=LIGHT)
    # <<< subjects

    log("post")
    cv.vignette(0.22, 0.7).soft_clip(0.9).grain(0.008, seed)
    return cv
