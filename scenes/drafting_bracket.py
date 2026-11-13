# -*- coding: utf-8 -*-
"""
Angle bracket: a complete A3 engineering drawing sheet (pixelcanvas.drafting).

* Part: L-shaped aluminium mounting bracket. Base plate 120 x 70 x 10 with rounded front corners and two
  adjustment slots; upright 120 x 75 x 10 with rounded top corners, a Ø28 H7 bore and 4x M6 on a Ø46 pitch circle.
* Sheet: A3 landscape, scale 1:1, first-angle projection (ISO). Front view in the middle, top view below it,
  right view on its left, SECTION A-A on its right, isometric pictorial at 1:2 bottom left, notes above the title
  block. Every view is its own layer.
* Model axes: X along the bracket, Y depth (front face Y = 0), Z up. Front view (X, Z), top view (X, Y),
  right view (Y, Z), section A-A looks along +X so its horizontal axis is 70 - Y.
"""
from pixelcanvas.drafting import Sheet, Profile

PROMPT = ("An A3 engineering drawing of an aluminium angle bracket with a bore, four M6 holes and two slots: front, "
          "top, right and section views, an isometric view, dimensions and a title block.")
NAME = "drafting_bracket"
KIND = "drawing"
DEFAULT_SIZE = (2480, 1754)            # A3 landscape at 150 dpi
DEFAULT_SEED = 1

STYLE = "white"                        # white | paper | blueprint
PROJECTION = "first"                   # first | third
L, D, H, T = 120, 70, 85, 10           # length, depth, height, plate thickness
BORE = 28
PCD = 46
SLOT_W, SLOT_Y0, SLOT_Y1 = 11, 18, 34
SLOT_X = (20, 100)
BORE_Z = 50
R_TOP, R_BASE = 15, 10


def build():
    sh = Sheet("A3", scale="1:1", projection=PROJECTION, title="ANGLE BRACKET", drawing_no="PC-1024-01",
               author="pixelcanvas", date="2026-11-12", material="EN AW-6082-T6", finish="Ra 3.2",
               checked="QA", revision="B", style=STYLE)
    bx, bz = L / 2, BORE_Z
    hole_r = PCD / 2 * 0.7071

    # ---- front view (X, Z)
    fr = sh.view("FRONT VIEW", origin=(150, 170))
    fr.profile(Profile(0, 0).to(L, 0).to(L, H, fillet=R_TOP).to(0, H, fillet=R_TOP).close())
    fr.line(0, T, L, T)
    for sx in SLOT_X:
        for x in (sx - SLOT_W / 2, sx + SLOT_W / 2):
            fr.line(x, 0, x, T, "hidden")
        fr.centerline(sx, 0, sx, T, extend=2)
    fr.hole(bx, bz, BORE)
    fr.bolt_circle(bx, bz, PCD, 4, 6, start=45, thread="M6")
    fr.cutting_plane([(bx, 0), (bx, H)], "A", direction="right")
    fr.dim_h((0, H), (L, H), side="top")
    fr.dim_v((0, 0), (0, H), side="left")
    fr.dim_v((L, 0), (L, bz), side="right")
    fr.dim_diameter(bx, bz, BORE, angle=200, leader=8)
    fr.dim_diameter(bx, bz, PCD, angle=160, leader=12)
    fr.dim_angle(bx, bz, 0, 45, r=34, ext_from=PCD / 2 + 4)
    fr.dim_radius(L - R_TOP, H - R_TOP, R_TOP, angle=50)
    fr.hole_callout(bx - hole_r, bz + hole_r, 6, n=4, thread="M6", angle=125, length=10)

    # ---- top view (X, Y): first angle puts it below the front view
    tp = sh.projected(fr, "top", 100)
    tp.profile(Profile(0, 0).to(L, 0, fillet=R_BASE).to(L, D).to(0, D).close(fillet=R_BASE))
    tp.line(0, D - T, L, D - T)
    for sx in SLOT_X:
        tp.slot(sx, SLOT_Y0, sx, SLOT_Y1, SLOT_W)
    for x in (bx - BORE / 2, bx + BORE / 2):
        tp.line(x, D - T, x, D, "hidden")
    tp.centerline(bx, D - T, bx, D, extend=3)
    tp.dim_chain([(0, 0), (SLOT_X[0], SLOT_Y0), (SLOT_X[1], SLOT_Y0), (L, 0)], axis="h", side="bottom")
    tp.dim_chain([(L, 0), (SLOT_X[1], SLOT_Y0), (SLOT_X[1], SLOT_Y1)], axis="v", side="right")
    tp.dim_v((L, 0), (L, D), side="right")
    tp.dim_radius(R_BASE, R_BASE, R_BASE, angle=225, leader=5)
    tp.callout(SLOT_X[0] + SLOT_W / 2, (SLOT_Y0 + SLOT_Y1) / 2, "2x SLOT 11 x 27", angle=40, length=14)

    # ---- right view (Y, Z): first angle puts it on the left
    rt = sh.projected(fr, "right", 112)
    rt.polygon([(0, 0), (D, 0), (D, H), (D - T, H), (D - T, T), (0, T)])
    for z in (bz - BORE / 2, bz + BORE / 2):
        rt.line(D - T, z, D, z, "hidden")
    rt.centerline(D - T, bz, D, bz, extend=3)
    for y in (SLOT_Y0 - SLOT_W / 2, SLOT_Y1 + SLOT_W / 2):
        rt.line(y, 0, y, T, "hidden")
    rt.dim_v((0, 0), (0, T), side="left")
    rt.dim_h((D - T, H), (D, H), side="top")
    rt.datum(D * 0.45, 0, "A", angle=-90)
    rt.datum(D, H * 0.42, "B", angle=0, length=6)
    rt.surface_finish(D * 0.3, T, "Ra 1.6", angle=90)
    rt.fcf(D - T - 40, H * 0.72, "perpendicularity", "0.05", ["A"], leader=(D - T, H * 0.72))

    # ---- section A-A (70 - Y, Z), viewed along +X: first angle puts it on the right
    sc = sh.projected(fr, "left", 150)
    sc.section_label("A")
    low = [(0, 0), (D, 0), (D, T), (T, T), (T, bz - BORE / 2), (0, bz - BORE / 2)]
    top = [(0, bz + BORE / 2), (T, bz + BORE / 2), (T, H), (0, H)]
    sc.hatch(low, material="general", outline=True)
    sc.hatch(top, material="general", outline=True)
    sc.centerline(0, bz, T, bz, extend=4)
    sc.dim_v((T, bz - BORE / 2), (T, bz + BORE / 2), side="right", prefix="Ø", suffix=" H7", tol=(0.021, 0))
    sc.dim_h((0, 0), (D, 0), side="bottom")
    sc.fcf(T + 12, bz + 6, "position", "0.1", ["A", "B"], diameter=True, leader=(T, bz + BORE / 2 - 3))

    # ---- isometric pictorial (1:2)
    iv = sh.view("ISOMETRIC", origin=(40, 82), scale="1:2")
    base = Profile(0, 0).to(L, 0, fillet=R_BASE).to(L, D).to(0, D).close(fillet=R_BASE)
    iv.iso_extrude(base, T, plane="xy")
    up = Profile(0, T).to(L, T).to(L, H, fillet=R_TOP).to(0, H, fillet=R_TOP).close()
    iv.iso_extrude(up, T, plane="xz", at=D - T)
    for sx in SLOT_X:
        iv.iso_slot(sx, SLOT_Y0, 0, sx, SLOT_Y1, 0, SLOT_W, axis="z")
    iv.iso_hole(bx, 0, bz, BORE, axis="y")
    for dx in (-hole_r, hole_r):
        for dz in (-hole_r, hole_r):
            iv.iso_hole(bx + dx, 0, bz + dz, 6, axis="y")
    iv.label("ISOMETRIC VIEW")

    sh.notes(["Break all sharp edges 0.3 x 45°.",
              "Untoleranced dimensions to ISO 2768-m.",
              "Clear anodise 15 µm after machining.",
              "Chamfer thread entries 0.5 x 90°."], x=292, y=132)
    return sh


def render(width=2480, height=1754, seed=1, log=None):
    from pixelcanvas import Canvas
    sh = build()
    cv = Canvas(width, height, (1, 1, 1))
    sh.draw(cv, log=log)
    return cv
