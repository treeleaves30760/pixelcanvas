"""lighthouse: red and white, on a dark cliff at the right, lamp glowing, beam over the sea; title 'Harbour Light'"""
from pixelcanvas import Canvas
from pixelcanvas.illustrate import Sketch, paper, grade
PROMPT = "a red and white lighthouse at dusk, titled 'Harbour Light'"
DEFAULT_SIZE = (1600, 900)

def render(width=1600, height=900, seed=1, log=lambda step: None):
    cv = Canvas(width, height)
    sk = Sketch(cv, light=(-0.6, -0.4))
    log("sky and sea")
    cv.gradient([(0, "#1d2150"), (0.5, "#f29a72"), (0.6, "#6a78b0"), (1, "#1c2a55")])
    sk.spot(0.3, 0.6, 0.25, 0.012, "#ffd9a8", 0.6)                         # light on the water
    log("cliff")
    sk.paint(sk.blob([(0.5, 1.05), (0.6, 0.72), (0.78, 0.64), (1.05, 0.62), (1.05, 1.05)]), "#3c3048", shade=0.5)
    log("beam")
    cv.paint("#fff1c4", sk.polygon([(0.81, 0.2), (0.0, 0.3), (0.0, 0.52)]), 0.25)
    log("lighthouse")
    with sk.at(0.81, 0.42, 0.44):                                           # local frame: 1 unit = 0.44 of the height
        sk.paint(sk.polygon([(-0.12, 0.45), (-0.08, -0.35), (0.08, -0.35), (0.12, 0.45)]), "#f4efe6",
                 shade=0.4, line=True, name="lighthouse")
        for y in (-0.2, 0.1, 0.38):
            sk.paint(sk.box(-0.118, y - 0.06, 0.118, y + 0.06), "#d2423c", shade=0.4)
        sk.paint(sk.box(-0.07, -0.5, 0.07, -0.35, 0.02), "#ffe9a8", glow="#ffd27a")
        sk.paint(sk.polygon([(-0.11, -0.5), (0, -0.64), (0.11, -0.5)]), "#d2423c", line=True)
    log("title")
    cv.text("Harbour Light", box=(0.06, 0.08, 0.56, 0.22), color="#fff4e0", font="serif", bold=True)
    grade(paper(cv), "golden", 0.6)
    return cv
