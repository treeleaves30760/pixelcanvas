# -*- coding: utf-8 -*-
"""
A picnic under the tree (clay_picnic) — a little story scene made with pixelcanvas.clay.

Two friends and a kid share a picnic on a gingham blanket under a round tree. The friend on the left leans back on
her hands, the friend on the right sits cross-legged and laughs, the kid kneels at the front holding up a
sandwich, and a labrador begs for it. A red balloon is tied to the wicker basket. Plates, a cake, juice and apples sit
on the blanket.
Composition: the blanket fills the lower middle, the tree frames the upper left, the balloon is the highest accent
on the right. Light: a warm, low afternoon sun from the upper right with soft shadows, a blue sky fill and a
gentle haze toward the horizon; depth of field on the kid and the dog.
"""
import numpy as np

from pixelcanvas import Canvas
from pixelcanvas.solid import Scene, Camera, Lighting, tree, fbm3
from pixelcanvas.clay import Character, cast, animal, prop

NAME = "clay_picnic"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 7

SKY = [(0.0, "#9cbcd8"), (0.3, "#e0dccc"), (0.45, "#f8dcb0"), (1.0, "#d8e0b0")]
BACK_TREES = [(5.0, -11.0, 3.4, "round"), (8.5, -15.0, 4.4, "pine"), (-7.5, -12.0, 4.0, "round"),
              (2.0, -16.0, 3.4, "bush"), (12.0, -18.0, 4.8, "round"), (-3.5, -17.0, 4.6, "pine"),
              (-12.0, -16.0, 4.2, "pine"), (16.0, -14.0, 3.6, "bush")]
SUN = "#ffc27e"
BLANKET = "#e0503e"
BALLOON = "#e8413a"
FOCUS = (0.3, 0.45, 0.45)
BLUR = 7.0
HAZE = 0.045


def _grass(u, v):
    """Grass colour from world x / z: two greens mixed by noise, a little lighter in the distance."""
    n = fbm3(u * 1.7, v * 1.7, np.zeros_like(u), 3, 11)
    a = np.clip((n - 0.35) * 2.2, 0, 1)
    g1, g2 = np.array([0.6, 0.72, 0.34]), np.array([0.47, 0.62, 0.27])
    return g1[None] * (1 - a[:, None]) + g2[None] * a[:, None]


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    lit = Lighting("sunny", sun=(62, 27), sun_color=SUN, sun_power=1.6, softness=0.45, ambient=0.42,
                   zenith="#a8c8e8", horizon="#f4dcb8", ground_color="#7a8a4a", fill_power=0.12, rim_power=0.35)
    sc = Scene(lighting=lit, ground=None)
    sc.floor(texture=_grass, roughness=0.9)

    sc.add(tree(3.6, kind="round", leaves="#6aa23e", trunk="#7a5236", seed=2).place(x=-2.0, z=-1.6), "tree")
    for i, (x, z, h, k) in enumerate(BACK_TREES):
        sc.add(tree(h, kind=k, leaves=["#7aaa4a", "#6a9a44", "#88b454"][i % 3], seed=10 + i).place(x=x, z=z),
               f"far tree {i + 1}")
    sc.add(prop("picnic_blanket", color=BLANKET, size=(2.3, 1.8)).rotate_y(8).place(x=0.0, z=0.0), "blanket")
    hx, hz = 0.18, -0.68
    sc.add(prop("basket", color="#c8945a").rotate_y(-14).place(x=hx, z=hz, y=0.012), "basket")
    hy = 0.012 + 0.46
    sc.add(prop("balloon", color=BALLOON, height=1.25).place(x=hx, z=hz, y=hy - 0.0), "balloon")

    # food on the blanket
    food = (prop("cake", color="#f6b8c8").place(x=-0.18, z=-0.3, y=0.012)
            + prop("juice", color="#f2a23a").place(x=-0.46, z=0.3, y=0.012)
            + prop("juice", color="#e8603a").place(x=0.5, z=-0.3, y=0.012)
            + prop("sandwich").rotate_y(30).place(x=-0.3, z=0.05, y=0.012))
    for x, z, c in [(0.12, 0.12, "#d8302a"), (0.2, 0.2, "#8ac83a"), (0.06, 0.24, "#d8302a")]:
        food = food + prop("apple", color=c).place(x=x, z=z, y=0.012)
    sc.add(food, "picnic food")

    # the people
    ana = Character("Ana", build="adult", skin=1, hair=dict(style="wavy", color="auburn"),
                    face=dict(expression="happy", freckles=True),
                    outfit=dict(top="dress", bottom="none", shoes="sandals", hat="straw_hat",
                                colours=dict(top="#f2c14e", hat="#e8cf8e", accent="#e86a8e", shoes="#c8845a")))
    ana.pose("sit_legs").face(62)
    ana = ana.place(x=-0.72, z=-0.08, y=0.012)
    ben = Character("Ben", build="adult", skin=5, hair=dict(style="short", color="black"),
                    face=dict(expression="laugh", beard="short", glasses=True),
                    outfit=dict(top="shirt", bottom="shorts", shoes="sneakers",
                                colours=dict(top="#5ab0b8", bottom="#e8d8b8", shoes="#f4f1ea", accent="#f4f1ea")))
    ben.pose("sit_floor").face(-58)
    ben = ben.place(x=0.78, z=0.1, y=0.012)
    kid = cast("kid", pose="kneel", hold=None, outfit=dict(colours=dict(top="#7a5cc8", bottom="#3f6fb0")))
    kid.pose("kneel", shoulder_r=dict(fwd=70, side=34, twist=-10), elbow_r=dict(bend=22),
             hands=dict(l="relaxed", r="grip"), head=dict(turn=-10, tilt=-6), expression="laugh")
    kid.hold(prop("sandwich"), hand="r")
    kid.face(12)
    kid = kid.place(x=0.1, z=0.55, y=0.012)
    dog = animal("labrador", pose="beg", size=0.62, tongue=True).face(-105)
    dog = dog.place(x=0.98, z=0.86)
    for who in (ana, ben, kid):
        sc.add(who, who.name)
    sc.add(dog, "dog")

    sc.camera = Camera.orbit(target=(0.05, 0.62, 0.2), distance=4.25, azimuth=-16, elevation=11, fov=34)
    res = sc.render(cv, log=log, background=SKY)

    log("post")
    res.fog(cv, "#e4e4c0", density=HAZE, start=4.0, sky=False)
    res.depth_of_field(cv, focus=FOCUS, blur=BLUR)
    cv.vignette(0.18).grain(0.006, seed=seed)
    return cv
