"""illustration_anime: an original anime character, drawn soft-cel.

girl: coral-red bob with a sheen band, teal eyes, cream hoodie with a mustard scarf; bust, centre, 80 % of the height
sky: soft gradient from deep blue to pale cyan with simple cel clouds and drifting petals
style: soft cel - base + one or two hard shadow bands shifted towards violet, rim light, coloured line art,
       multi-tone eyes, gradient blush; light from the upper left
"""
import numpy as np

from pixelcanvas import Canvas, manifest
from pixelcanvas.illustrate import Sketch, cel_tone

PROMPT = "an original anime girl with a coral-red bob and teal eyes, soft cel shading, against a summer sky"
KIND = "painting"
DEFAULT_SIZE = (1920, 1080)
HAIR = [(0, "#c8455a"), (1, "#f27f6a")]
SKIN = "#fde4d4"
EYES = "#1f9a9a"
HOODIE = "#f4ecdc"
SCARF = "#f0b43c"
SKY = ["#3f7fd8", "#7fb8ee", "#cdeefa"]
SHIFT = (-0.12, None)


def render(width=1920, height=1080, seed=1, log=lambda step: None):
    cv = Canvas(width, height)
    sk = Sketch(cv, light=(-0.6, -0.8), finish="full")
    log("sky")
    sk.ramp(sk.box(0, 0, 1, 1), SKY)
    for cx, cy, s in ((0.16, 0.3, 1.0), (0.86, 0.2, 0.8), (0.82, 0.72, 1.2)):
        with sk.at(cx, cy, 0.12 * s):
            cloud = sk.blob([(-1.4, 0.4), (-1.1, -0.2), (-0.5, -0.45), (0.0, -0.9), (0.6, -0.5), (1.2, -0.4),
                             (1.5, 0.4)], tension=0.1)
            sk.cel(cloud, "#ffffff", shift=(-0.07, 0), depth=0.35)
    manifest.note(cv, "object", "sky", bbox=(0, 0, 1, 1), backdrop=True)
    log("petals")
    for i, (px, py, a) in enumerate(((0.08, 0.62, 20), (0.27, 0.12, -30), (0.7, 0.1, 50), (0.93, 0.45, -10),
                                     (0.24, 0.85, 70), (0.76, 0.9, 10))):
        with sk.at(px, py, 0.022, rotate=a):
            sk.cel(sk.blob([(0, -1), (0.6, 0), (0, 1), (-0.6, 0)], tension=0.3), "#ffd0dc", shift=(-0.06, -10))
    with sk.at(0.5, 0.42, 0.4):
        log("back hair")
        back = sk.blob([(-0.62, -0.55), (-0.2, -0.9), (0.35, -0.86), (0.7, -0.45), (0.72, 0.2), (0.66, 0.62),
                        (0.3, 0.52), (-0.3, 0.52), (-0.68, 0.62), (-0.74, 0.15)], tension=0.2)
        sk.cel(back, HAIR, bands=1, shift=(-0.16, None), depth=0.25, line=True, line_width=0.006)
        log("body")
        neck = sk.box(-0.13, 0.3, 0.13, 0.85, radius=0.05)
        sk.cel(neck, SKIN)
        chin = sk.blob([(-0.2, 0.3), (0.2, 0.3), (0.16, 0.66), (0.0, 0.76), (-0.16, 0.66)], tension=0.2)
        sk.paint(neck * chin, cel_tone(SKIN, -0.13, -12, 1.25))                 # neck shadow under the chin
        body = sk.blob([(-1.4, 1.9), (-1.2, 1.05), (-0.95, 0.86), (-0.5, 0.76), (0.0, 0.8), (0.5, 0.76), (0.95, 0.86),
                        (1.2, 1.05), (1.4, 1.9)], tension=0.2)
        sk.cel(body, HOODIE, bands=1, depth=0.1, line=True, line_width=0.006, name="hoodie")
        fold = cel_tone(HOODIE, -0.1)                       # cloth folds and the scarf's cast shadow: hard shapes
        sk.paint(body * sk.blob([(-0.6, 0.8), (0.6, 0.8), (0.45, 1.02), (0.0, 1.08), (-0.45, 1.02)]), fold)
        deep = cel_tone(HOODIE, -0.18)
        for sx in (-1, 1):                                  # folds follow the arm seam and the collar
            sk.paint(body * sk.stroke([(0.9 * sx, 0.9), (1.02 * sx, 1.12), (1.1 * sx, 1.4), (1.16 * sx, 1.7)],
                                      [0.0, 0.07, 0.1, 0.12], taper="start"), fold)
            sk.paint(body * sk.stroke([(0.7 * sx, 0.96), (0.8 * sx, 1.18), (0.86 * sx, 1.45)], [0.0, 0.045, 0.0],
                                      taper="both"), deep)
            sk.paint(body * sk.stroke([(0.5 * sx, 1.02), (0.58 * sx, 1.2), (0.6 * sx, 1.32)], [0.0, 0.03, 0.0],
                                      taper="both"), fold)
        for sx in (-1, 1):
            sk.ink([(0.2 * sx, 0.95), (0.35 * sx, 1.15), (0.36 * sx, 1.35)], 0.012, cel_tone(HOODIE, -0.42))
        scarf = sk.blob([(-0.55, 0.78), (-0.25, 0.66), (0.0, 0.7), (0.25, 0.66), (0.55, 0.78), (0.35, 0.95),
                         (0.0, 0.9), (-0.35, 0.95)], tension=0.2)
        sk.cel(scarf, SCARF, bands=1, depth=0.6, highlight=0.7, line=True, line_width=0.006, name="scarf")
        tail = sk.blob([(0.18, 0.86), (0.4, 0.88), (0.36, 1.32), (0.2, 1.28)], tension=0.3)
        sk.cel(tail, SCARF, depth=0.7, line=True, line_width=0.005)
        log("face")
        face = sk.blob([(-0.5, -0.3), (-0.49, 0.05), (-0.36, 0.36), (-0.12, 0.56), (0.0, 0.6), (0.12, 0.56),
                        (0.36, 0.36), (0.49, 0.05), (0.5, -0.3), (0.3, -0.62), (-0.3, -0.62)], tension=0.15)
        sk.cel(face, SKIN, bands=1, depth=0.12, line=True, line_width=0.005, name="face")
        for sx in (-1, 1):
            sk.cel(sk.ellipse(0.52 * sx, 0.08, 0.06, 0.1), SKIN, line=True, line_width=0.004)
        bangs_shadow = sk.blob([(-0.5, -0.2), (-0.25, 0.0), (0.0, -0.12), (0.25, 0.02), (0.5, -0.18),
                                (0.5, -0.7), (-0.5, -0.7)], tension=0.2)
        sk.paint(face * bangs_shadow, cel_tone(SKIN, -0.08, -12, 1.2))
        log("eyes")
        for sx in (-1, 1):
            sk.eye(0.25 * sx, 0.11, 0.36, EYES, look=(0.02, 0.0), mirror=sx < 0, name="eye")
            sk.ink([(0.12 * sx, -0.14), (0.24 * sx, -0.17), (0.36 * sx, -0.13)], 0.018, cel_tone(HAIR[0][1], -0.3))
            sk.blush(0.3 * sx, 0.34, 0.09, color="#ff8e98", opacity=0.4)
        sk.ink([(0.015, 0.3), (0.0, 0.33)], 0.012, cel_tone(SKIN, -0.35))
        mouth = sk.blob([(-0.07, 0.43), (0.07, 0.43), (0.03, 0.48), (-0.03, 0.48)], tension=0.2)
        sk.paint(mouth, "#c8506a", line="#8a3048", line_width=0.003)
        sk.paint(mouth * sk.ellipse(0.0, 0.5, 0.05, 0.03), "#ff9aa6")
        log("bangs")
        locks = [[(-0.6, -0.6), (-0.4, -0.85), (-0.3, -0.4), (-0.42, -0.05), (-0.55, -0.3)],
                 [(-0.45, -0.8), (-0.1, -0.9), (-0.08, -0.45), (-0.2, -0.12), (-0.32, -0.45)],
                 [(-0.2, -0.88), (0.18, -0.9), (0.12, -0.45), (0.04, -0.06), (-0.06, -0.45)],
                 [(0.05, -0.9), (0.45, -0.82), (0.36, -0.45), (0.26, -0.1), (0.16, -0.45)],
                 [(0.3, -0.85), (0.62, -0.6), (0.56, -0.3), (0.44, -0.04), (0.36, -0.42)],
                 [(-0.6, -0.5), (-0.5, -0.2), (-0.5, 0.3), (-0.42, 0.62), (-0.6, 0.2), (-0.66, -0.2)],
                 [(0.6, -0.5), (0.5, -0.2), (0.5, 0.3), (0.42, 0.62), (0.6, 0.2), (0.66, -0.2)]]
        bangs = np.maximum.reduce([sk.blob(p, tension=0.45) for p in locks])
        crown = sk.blob([(-0.66, -0.4), (-0.4, -0.86), (0.0, -0.98), (0.4, -0.86), (0.66, -0.4), (0.0, -0.62)],
                        tension=0.2)
        hair = np.maximum(bangs, crown)
        sk.cel(hair, HAIR, bands=2, shift=(-0.14, None), depth=0.3, rim=True, line=True, line_width=0.006,
               name="hair")
        sk.sheen(hair * crown, y=0.42, width=0.07, teeth=11, bend=0.3)
        for x0 in (-0.3, 0.05, 0.3):
            sk.ink([(x0, -0.8), (x0 + 0.04, -0.55), (x0 - 0.02, -0.3)], 0.01, cel_tone(HAIR[0][1], -0.3), opacity=0.7)
    manifest.note(cv, "style", "anime")
    return cv
