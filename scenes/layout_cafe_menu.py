# -*- coding: utf-8 -*-
"""Cafe menu on cream paper: serif name between a coffee cup and a croissant, three sections of five priced items

* Format: 1080x1500 portrait
* Focal point: the name Fika & Co. at the top
* Text hierarchy: name -> section headings (Coffee, Tea, Pastries) -> items with dotted leaders and prices
* Palette: cream paper, espresso-brown ink, terracotta accent and frame
* Exact text: Fika & Co.; Coffee; Tea; Pastries; fifteen items with prices
"""
# Read skills/pixelcanvas/references/typography.md (templates: menu) and pictograms.md.
from pixelcanvas import Canvas, layout
from pixelcanvas.backdrop import backdrop
from pixelcanvas.pictograms import pictogram

PROMPT = ("A café menu with three sections — Coffee, Tea, Pastries — five items each with prices, the café name "
          "'Fika & Co.' at the top, a small illustrated coffee cup and croissant, elegant serif type on cream paper.")
NAME = "layout_cafe_menu"
KIND = "design"
DEFAULT_SIZE = (1080, 1500)
DEFAULT_SEED = 1

TITLE = "Fika & Co."
SECTIONS = [
    ("Coffee", [("Espresso", "$3.00"), ("Americano", "$3.50"), ("Cappuccino", "$4.50"), ("Flat White", "$4.50"),
                ("Cold Brew", "$5.00")]),
    ("Tea", [("English Breakfast", "$3.50"), ("Earl Grey", "$3.50"), ("Jasmine Green", "$4.00"),
             ("Chamomile", "$4.00"), ("Chai Latte", "$5.00")]),
    ("Pastries", [("Butter Croissant", "$3.80"), ("Cinnamon Bun", "$4.20"), ("Almond Danish", "$4.50"),
                  ("Blueberry Muffin", "$3.90"), ("Lemon Tart", "$5.20")]),
]
INK, ACCENT = "#3a2a20", "#8f3f1f"
ICON_SIZE = 0.11                                         # fraction of the canvas height
ICON_X = (0.13, 0.87)                                    # cup left of the name, croissant right of it


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    log("paper")
    backdrop(cv, "warm_paper", seed=seed)
    log("menu")
    th = layout.theme("paper", ink=INK, accent=ACCENT, title_font="serif bold", body_font="serif")
    boxes = layout.template("menu", cv, title=TITLE, sections=SECTIONS, theme=th, seed=seed, background=False)
    v = boxes["title"][1] / height + 0.04
    log("coffee cup")
    pictogram(cv, "mug", at=(ICON_X[0], v), size=ICON_SIZE, color="#6b4a36", style="shaded", shadow=True)
    log("croissant")
    pictogram(cv, "croissant", at=(ICON_X[1], v), size=ICON_SIZE, color="#c98a3c", style="shaded", shadow=True)
    log("post")
    cv.vignette(0.12, 0.7)
    return cv
