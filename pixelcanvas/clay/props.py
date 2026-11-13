# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import numpy as np

from .. import solid as _S
from .._args import unknown_name as _unknown, reject_unknown as _reject, signature_options as _sigopts
from ..color import F32
from ..solid import (Material, Model, Part, capsule, cone, cylinder, ellipsoid, intersect, plane, rounded_box, sphere, subtract,
                     torus)
from .shapes import _Checker, _Rx, _Ry, _Rz, _chain, _grp, _lin, _shade, _xf, round_cone, smooth_union
from .families import _Paint


# ===================================================================================================== props

def _pm(name, parts, grip=None, upright=True):
    m = Model(name, [Part(n, _grp(s) if isinstance(s, list) else s, mt) for n, s, mt in parts])
    m.grip = None if grip is None else np.asarray(grip, np.float64)
    m.upright = upright
    return m


def _gloss(c):
    return Material(color=c, roughness=0.22, specular=0.8, clearcoat=0.4)


def _matt(c, r=0.7):
    return Material(color=c, roughness=r, specular=0.25, subsurface=0.2)


class _Gingham(_Checker):
    _kind = None

    def __init__(self, a="#e04a3a", b="#f8f4ec", size=0.12):
        self.a, self.b, self.size = _lin(a), _lin(b), float(size)

    def __call__(self, P, N=None, foot=None):
        w = np.full(len(P), F32(0.01)) if foot is None else np.maximum(foot / F32(self.size) * 1.5, F32(0.01))
        u, v = P[:, 0] / F32(self.size), P[:, 2] / F32(self.size)

        def band(x):
            return np.clip(0.5 + ((_S._tri(x + w * 0.5) - _S._tri(x - w * 0.5)) / w) * 0.5, 0, 1)
        su, sv = band(u), band(v)
        k = (su + sv) / 2
        return self.b[None] * (1 - k[:, None]) + self.a[None] * k[:, None]


def balloon(color="#e8413a", height=1.0, string="#f4f2ee"):
    """Balloon on a string (grip = the string's end; it floats upright when held)."""
    r = 0.15
    top = height
    b = ellipsoid(r, r * 1.18, r).at(0, top, 0)
    knot = cone(0.025, 0.005, 0.04).at(0, top - r * 1.18 - 0.01, 0)
    pts = [np.array([0, 0.0, 0]), np.array([0.03, top * 0.35, 0.01]), np.array([-0.02, top * 0.7, 0]),
           np.array([0, top - r * 1.18 - 0.03, 0])]
    s = _chain(pts, [0.004] * 4)
    # satin latex: one soft highlight, light glowing through; no mirror reflections of the dark ground
    return _pm("balloon", [("balloon", [b, knot], Material(color=color, roughness=0.3, specular=0.55, clearcoat=0.2,
                                                           subsurface=0.5)),
                           ("string", s, _matt(string))], grip=(0, 0.0, 0))


def book(color="#c0453a", open=False, pages="#f6f1e4", w=0.16, h=0.22, d=0.035):
    """Book (closed: standing; open: lying flat). Grip at the spine."""
    if not open:
        cover = rounded_box(d, h, w, 0.004).at(0, h / 2, 0)
        pg = rounded_box(d * 0.8, h * 0.94, w * 0.95, 0.002).at(0.002, h / 2, 0.006)
        return _pm("book", [("cover", [cover], _matt(color, 0.6)), ("pages", [pg], _matt(pages, 0.8))],
                   grip=(0, h * 0.4, -w * 0.3), upright=True)
    halves = []
    pgs = []
    for s in (1, -1):
        halves.append(_xf(rounded_box(w, 0.006, h, 0.003), _Rz(s * 8), (s * w * 0.5, 0.01, 0)))
        pgs.append(_xf(rounded_box(w * 0.95, 0.012, h * 0.95, 0.004), _Rz(s * 8), (s * w * 0.5, 0.02, 0)))
    return _pm("book", [("cover", halves, _matt(color, 0.6)), ("pages", pgs, _matt(pages, 0.85))],
               grip=(0, 0.0, -h * 0.4), upright=True)


def cup(color="#f4f1ea", drink="#6a3a1e"):
    """Mug with a handle and a drink (grip = handle)."""
    body = _S.subtract(cylinder(0.045, 0.1, round=0.006).at(0, 0.05, 0), cylinder(0.039, 0.1).at(0, 0.058, 0))
    handle = _xf(torus(0.027, 0.008), _Rx(90), (0.05, 0.055, 0))
    return _pm("cup", [("cup", [body, handle], _gloss(color)),
                       ("drink", [cylinder(0.04, 0.01).at(0, 0.085, 0)], Material(color=drink, roughness=0.1, specular=0.8))],
               grip=(0.075, 0.055, 0))


def phone(color="#2a2a30", screen="#6ab0f0"):
    """Smartphone with a glowing screen (grip = lower half)."""
    b = rounded_box(0.075, 0.15, 0.01, 0.008).at(0, 0.075, 0)
    s = rounded_box(0.066, 0.135, 0.002, 0.006).at(0, 0.075, 0.005)
    return _pm("phone", [("case", [b], _gloss(color)), ("screen", [s], Material(color=screen, emissive=0.8, roughness=0.1,
                                                                                specular=0.9))],
               grip=(0, 0.05, -0.01), upright=False)


def umbrella(color="#e8413a", open=True, height=0.9):
    """Umbrella (open canopy or closed), grip = the handle."""
    shaft = capsule((0, 0.05, 0), (0, height, 0), 0.008)
    hook = intersect(_xf(torus(0.035, 0.009), _Rx(90), (0.035, 0.05, 0)), plane((0, 1, 0), 0.05))
    parts = [("shaft", [shaft, hook], Material(color="#3a2a22", roughness=0.4, specular=0.5))]
    if open:
        sh = intersect(sphere(0.55).at(0, height - 0.3, 0).shell(0.012), plane((0, -1, 0), -(height - 0.06)))
        ribs = []
        for i in range(8):
            a = 2 * math.pi * i / 8
            ribs.append(sphere(0.02).at(math.sin(a) * 0.42, height - 0.06, math.cos(a) * 0.42))
        parts.append(("canopy", [sh], Material(color=color, roughness=0.4, specular=0.4, subsurface=0.3)))
        parts.append(("tips", ribs + [sphere(0.02).at(0, height + 0.26, 0)], _matt("#2a2a2a")))
    else:
        parts.append(("canopy", [round_cone((0, height * 0.35, 0), (0, height * 0.95, 0), 0.04, 0.012)],
                      _matt(color, 0.5)))
    return _pm("umbrella", parts, grip=(0.0, 0.12, 0))


def guitar(color="#d8843a", height=0.95):
    """Acoustic guitar standing upright (grip = the neck)."""
    s = height / 0.95
    body = smooth_union(ellipsoid(0.19, 0.17, 0.05).at(0, 0.2, 0), ellipsoid(0.15, 0.13, 0.05).at(0, 0.44, 0), k=0.06)
    hole = cylinder(0.05, 0.02).rotate_x(90).at(0, 0.38, 0.045)
    neck = rounded_box(0.05, 0.4, 0.03, 0.01).at(0, 0.72, 0.01)
    head = rounded_box(0.07, 0.12, 0.03, 0.015).at(0, 0.96, 0.0)
    return _pm("guitar", [("body", [body.scale(1.0)], Material(color=color, roughness=0.25, specular=0.7, clearcoat=0.6)),
                          ("hole", [hole], Material(color="#2a1a12", roughness=0.6)),
                          ("neck", [neck, head], Material(color="#5a3a24", roughness=0.4, specular=0.5))],
               grip=(0, 0.62, 0.02), upright=False) if s == 1 else guitar(color).scale(s)


def ball(color="#e8413a", kind="beach", r=0.15):
    """Ball: kind = 'beach' (coloured gores), 'soccer', 'basket', 'tennis', 'plain'."""
    if kind == "beach":
        c = _S.stripes(r * 0.7, "#f8f6f0", color, axis="x")
    elif kind == "tennis":
        c, r = "#d8e84a", min(r, 0.035)
    elif kind == "basket":
        c = "#e0782e"
    elif kind == "soccer":
        c = _Paint("#f8f6f2", "spots", "#2a2a2e", freq=1 / (r * 0.55), width=0.28)
    else:
        c = color
    return _pm("ball", [("ball", [sphere(r).at(0, r, 0)], Material(color=c, roughness=0.3, specular=0.6, clearcoat=0.3))],
               grip=(0, r * 0.3, 0))


def flower(color="#f2647a", height=0.3, center="#f6c23a"):
    """Flower with a stem, two leaves and a ring of petals (grip = the stem)."""
    stem = _chain([np.array([0, 0, 0]), np.array([0.01, height * 0.5, 0]), np.array([0, height, 0])], [0.006, 0.005, 0.005])
    leaves = [_xf(ellipsoid(0.04, 0.008, 0.016), _Rz(s * 30), (s * 0.03, height * 0.4, 0)) for s in (1, -1)]
    pet = [_xf(ellipsoid(0.03, 0.01, 0.016), _Ry(a) @ _Rz(-12), np.array([0, height, 0]) + _Ry(a) @ np.array([0.03, 0, 0]))
           for a in range(0, 360, 60)]
    return _pm("flower", [("stem", stem + leaves, _matt("#4f9a3c")), ("petals", pet, _matt(color, 0.6)),
                          ("center", [sphere(0.016).at(0, height + 0.005, 0)], _matt(center))], grip=(0, height * 0.25, 0))


def bouquet(colors=("#f2647a", "#f6c23a", "#ffffff", "#c86ae0")):
    """A hand-held bunch of flowers in paper (grip = the wrap)."""
    parts, cols = [], list(colors)
    heads = []
    for i in range(7):
        a = i * 2.4
        r = 0.04 * (i > 0)
        p = np.array([r * math.cos(a), 0.28 + 0.02 * math.sin(i), r * math.sin(a)])
        heads.append((sphere(0.035).at(*p), cols[i % len(cols)]))
    wrap = round_cone((0, 0.02, 0), (0, 0.24, 0), 0.03, 0.08).shell(0.006)
    out = [("wrap", [intersect(wrap, plane((0, 1, 0), 0.24))], _matt("#f4e6c8", 0.8)),
           ("stems", [capsule((0, 0.0, 0), (0, 0.27, 0), 0.02)], _matt("#4f9a3c"))]
    for i, c in enumerate(cols):
        sh = [h for h, cc in heads if cc == c]
        if sh:
            out.append((f"flowers {i + 1}", sh, _matt(c, 0.6)))
    return _pm("bouquet", out, grip=(0, 0.1, 0))


def ice_cream(flavours=("#f6b8c8", "#f8f0dc", "#7a4a2a"), cherry=True):
    """Waffle cone with scoops (grip = the cone)."""
    cone_ = cone(0.0, 0.045, 0.14).at(0, 0.07, 0)
    scoops = [sphere(0.05).at(0, 0.16 + 0.06 * i, 0) for i in range(len(flavours))]
    out = [("cone", [cone_], Material(color=_S.checker(0.02, "#e0a858", "#c88a40"), roughness=0.7, specular=0.2))]
    for i, (s, c) in enumerate(zip(scoops, flavours)):
        out.append((f"scoop {i + 1}", [s.round(0.004)], _matt(c, 0.55)))
    if cherry:
        out.append(("cherry", [sphere(0.018).at(0, 0.16 + 0.06 * len(flavours) - 0.01, 0)], _gloss("#d8202a")))
    return _pm("ice cream", out, grip=(0, 0.04, 0))


def sign(text="HELLO", color="#f8f4ec", board="#3a7ad8", text_color=None, width=0.6, height=1.2):
    """Placard on a post with 3D lettering (grip = the post)."""
    post = cylinder(0.018, height * 0.75, round=0.005).at(0, height * 0.375, 0)
    bw, bh = width, width * 0.55
    brd = rounded_box(bw, bh, 0.03, 0.02).at(0, height * 0.75 + bh * 0.42, 0)
    txt = _S.extrude_text(text, height=bh * 0.32, depth=0.012)
    lo, hi = txt.bounds()
    sc = min(1.0, bw * 0.86 / max(hi[0] - lo[0], 1e-6))
    t = txt.scale((sc, sc, 1.0)).at(0, height * 0.75 + bh * 0.42, 0.02)
    return _pm("sign", [("post", [post], Material(color="#a8784a", roughness=0.6)), ("board", [brd], _matt(board, 0.5)),
                        ("text", [t], _matt(text_color or color, 0.4))], grip=(0, height * 0.3, 0))


def flag(color="#e8413a", accent="#f8f4ec", height=0.9):
    """Small flag on a pole with a waving cloth."""
    pole = capsule((0, 0, 0), (0, height, 0), 0.008)
    cloth = rounded_box(0.3, 0.18, 0.008, 0.004).bend(25).at(0.16, height - 0.1, 0)
    stripe = rounded_box(0.3, 0.05, 0.0105, 0.003).bend(25).at(0.16, height - 0.1, 0)
    return _pm("flag", [("pole", [pole, sphere(0.015).at(0, height + 0.01, 0)], Material(color="#d8b04a", roughness=0.3,
                                                                                         specular=0.7, metallic=0.5)),
                        ("cloth", [cloth], _matt(color, 0.6)), ("stripe", [stripe], _matt(accent, 0.6))],
               grip=(0, height * 0.25, 0))


def lantern(color="#2a2a30", glow="#ffcf7a"):
    """Glowing lantern with a handle (grip = handle)."""
    frame = _S.union(cylinder(0.07, 0.02, round=0.005).at(0, 0.01, 0), cone(0.075, 0.02, 0.06).at(0, 0.23, 0),
                     *[capsule((math.cos(a) * 0.065, 0.02, math.sin(a) * 0.065), (math.cos(a) * 0.065, 0.2, math.sin(a) * 0.065), 0.006)
                       for a in np.radians([45, 135, 225, 315])])
    handle = intersect(_xf(torus(0.05, 0.006), _Rx(90), (0, 0.27, 0)), plane((0, -1, 0), -0.27))
    g = cylinder(0.055, 0.17, round=0.02).at(0, 0.11, 0)
    return _pm("lantern", [("frame", [frame, handle], Material(color=color, roughness=0.35, specular=0.6, metallic=0.5)),
                           ("glass", [g], Material(color=glow, emissive=2.2, roughness=0.3, shadow=False))],
               grip=(0, 0.31, 0))


def fishing_rod(color="#4a6ad8"):
    """Fishing rod with reel and line (grip = handle)."""
    rod = round_cone((0, 0, 0), (0, 1.6, 0.35), 0.014, 0.004)
    reel = _xf(cylinder(0.03, 0.02, round=0.006), _Rz(90), (0.03, 0.22, 0.05))
    line = _chain([np.array([0, 1.6, 0.35]), np.array([0, 1.0, 0.75]), np.array([0, 0.2, 0.9])], [0.002] * 3)
    return _pm("fishing rod", [("rod", [rod], _gloss(color)), ("reel", [reel], Material(color="#c8ccd2", metallic=1.0, roughness=0.3)),
                               ("line", line, _matt("#f4f2ee")), ("float", [sphere(0.02).at(0, 0.2, 0.9)], _gloss("#e8413a"))],
               grip=(0, 0.15, 0.03), upright=True)


def skateboard(color="#3aa0d8", wheels="#f2d23a"):
    """Skateboard lying on the ground (deck, trucks, four wheels)."""
    deck = rounded_box(0.21, 0.018, 0.8, 0.012).bend(0).at(0, 0.09, 0)
    tails = [_xf(rounded_box(0.21, 0.018, 0.1, 0.012), _Rx(-s * 18), (0, 0.1, s * 0.42)) for s in (1, -1)]
    trucks = [rounded_box(0.16, 0.02, 0.04, 0.008).at(0, 0.065, s * 0.28) for s in (1, -1)]
    wh = [_xf(cylinder(0.03, 0.035, round=0.008), _Rz(90), (x, 0.03, s * 0.28)) for s in (1, -1) for x in (-0.085, 0.085)]
    return _pm("skateboard", [("deck", [deck] + tails, Material(color=color, roughness=0.3, specular=0.6, clearcoat=0.4)),
                              ("trucks", trucks, Material(color="#c8ccd2", metallic=1.0, roughness=0.3)),
                              ("wheels", wh, _gloss(wheels))], grip=(0, 0.09, 0.4), upright=False)


def bicycle(color="#e8413a", height=1.0):
    """Bicycle standing on its wheels (points along +z)."""
    R = 0.33
    wheels = [_xf(torus(R, 0.025), _Rz(90), (0, R + 0.025, s * 0.52)) for s in (1, -1)]
    hubs = [_xf(cylinder(0.03, 0.06), _Rz(90), (0, R + 0.025, s * 0.52)) for s in (1, -1)]
    spokes = [_xf(cylinder(R, 0.004), _Rz(90), (0, R + 0.025, s * 0.52)) for s in (1, -1)]
    bb = np.array([0, 0.33, 0.0])
    seat = np.array([0, 0.78, -0.12])
    head = np.array([0, 0.8, 0.38])
    rear = np.array([0, R + 0.025, -0.52])
    front = np.array([0, R + 0.025, 0.52])
    frame = _chain([rear, bb, head], [0.018] * 3) + _chain([rear, seat, bb], [0.016] * 3) + \
        [capsule(seat, head - np.array([0, 0.04, 0]), 0.02), capsule(head, front, 0.018)]
    bars = [capsule(head + np.array([0, 0.06, -0.03]), head + np.array([s * 0.24, 0.1, -0.06]), 0.014) for s in (1, -1)]
    grips = [sphere(0.025).at(*(head + np.array([s * 0.24, 0.1, -0.06]))) for s in (1, -1)]
    saddle = _xf(ellipsoid(0.08, 0.03, 0.13), np.eye(3), seat + np.array([0, 0.06, -0.02]))
    pedals = [rounded_box(0.08, 0.015, 0.05, 0.006).at(s * 0.1, 0.33 + s * 0.08, s * 0.04) for s in (1, -1)]
    m = _pm("bicycle", [("frame", frame + [capsule(head, head + np.array([0, 0.08, -0.02]), 0.02)], _gloss(color)),
                        ("tyres", wheels, Material(color="#2a2a2e", roughness=0.8, specular=0.2)),
                        ("spokes", spokes, Material(color="#c8ccd2", metallic=1.0, roughness=0.25, glass=0.0)),
                        ("hubs", hubs + bars + pedals, Material(color="#c8ccd2", metallic=1.0, roughness=0.3)),
                        ("saddle", [saddle] + grips, _matt("#2a2a2e", 0.5))], grip=(0.24, 0.9, 0.32), upright=False)
    return m if height == 1.0 else m.scale(height)


def picnic_blanket(color="#e04a3a", size=(2.0, 1.6), check=0.16):
    """Gingham picnic blanket lying on the ground (a thin, slightly rumpled slab)."""
    w, d = size
    b = rounded_box(w, 0.012, d, 0.006).at(0, 0.006, 0)
    return _pm("picnic blanket", [("blanket", [b], Material(color=_Gingham(color, "#f8f4ec", check), roughness=0.92,
                                                            specular=0.08, sheen=0.6))], upright=False)


def basket(color="#c8945a", handle=True, contents=True):
    """Wicker picnic basket with a lid and a handle."""
    body = rounded_box(0.46, 0.26, 0.32, 0.04).at(0, 0.13, 0)
    lid = rounded_box(0.48, 0.05, 0.34, 0.025).at(0, 0.28, 0)
    out = [("basket", [body], Material(color=_S.weave(color, scale=0.5), roughness=0.85, specular=0.15)),
           ("lid", [lid], Material(color=_S.weave(_shade(color, 0.92), scale=0.5), roughness=0.85, specular=0.15))]
    if handle:
        out.append(("handle", [intersect(_xf(torus(0.17, 0.016), np.eye(3) @ _Rx(90), (0, 0.3, 0)), plane((0, -1, 0), -0.3))],
                    _matt(_shade(color, 0.8), 0.6)))
    if contents:
        out.append(("napkin", [_xf(rounded_box(0.22, 0.02, 0.2, 0.01), _Ry(20) @ _Rz(-12), (0.14, 0.31, 0.0))],
                    Material(color=_Gingham("#e04a3a", "#f8f4ec", 0.04), roughness=0.9, specular=0.05)))
    return _pm("basket", out, grip=(0, 0.42, 0))


class _FlameColor(_Checker):
    """Flame colour by height: a pale yellow-white heart at the base, orange body, deep red-orange tips."""
    _kind = None

    def __init__(self, y0, y1, glow="#ffb84a"):
        self.y0, self.y1 = float(y0), float(y1)
        self.cols = np.array([_lin(c) for c in ("#fff6c8", "#ffd860", glow, "#ff7a24", "#e8401a")], F32)
        self.stops = np.array([0.0, 0.18, 0.42, 0.7, 1.0], F32)

    def __call__(self, P, N=None, foot=None):
        t = np.clip((P[:, 1] - self.y0) / max(self.y1 - self.y0, 1e-6), 0, 1).astype(F32)
        out = np.empty((len(P), 3), F32)
        for c in range(3):
            out[:, c] = np.interp(t, self.stops, self.cols[:, c])
        return out


def _tongue(base, height, radius, lean=(0.0, 0.0), sway=0.0):
    """One flame tongue: a teardrop that swells above the base, licks sideways and ends in a fine tip."""
    b = np.asarray(base, np.float64)
    lx, lz = lean
    side = np.array([-lz, 0.0, lx]) * sway
    pts = [b, b + np.array([lx * 0.15, height * 0.3, lz * 0.15]) + side * 0.4,
           b + np.array([lx * 0.45, height * 0.62, lz * 0.45]) - side * 0.5,
           b + np.array([lx * 0.9, height, lz * 0.9]) + side * 0.3]
    return smooth_union(*_chain(pts, [radius * 0.8, radius, radius * 0.55, radius * 0.04]), k=radius * 0.3)


def campfire(glow="#ffb84a"):
    """Campfire: crossed logs with glowing ember ends in a ring of stones, and layered flames (a pale core inside
    a ring of orange tongues, red at the tips) that glow. ``sc.light_from(fire)`` lights the ground and anyone
    sitting near it from the flames."""
    logs, ends = [], []
    for a in (0, 72, 144, 216, 288):
        lg = _xf(cylinder(0.05, 0.5, round=0.02), _Ry(a) @ _Rx(78), (math.sin(math.radians(a)) * 0.05, 0.14,
                                                                    math.cos(math.radians(a)) * 0.05))
        logs.append(lg)
    heart = sphere(0.15).at(0, 0.12, 0)
    wood = subtract(_S.union(*logs), heart)
    embers = intersect(_S.union(*logs), heart)
    coals = [sphere(r).scale((1.0, 0.6, 1.0)).at(x, 0.03, z)
             for x, z, r in ((0.0, 0.0, 0.09), (0.08, 0.05, 0.05), (-0.07, 0.06, 0.05), (0.04, -0.08, 0.05),
                             (-0.06, -0.06, 0.045))]
    stones = [sphere(0.07).scale((1.0, 0.7, 1.0)).at(math.cos(a) * 0.34, 0.035, math.sin(a) * 0.34)
              for a in np.linspace(0, 2 * math.pi, 11)[:-1]]
    # outer ring of tongues leaning out, a taller pale core rising through the middle
    outer = []
    for i, a in enumerate(np.linspace(0, 2 * math.pi, 6)[:-1] + 0.4):
        d = (math.cos(a), math.sin(a))
        h = (0.26, 0.32, 0.22, 0.3, 0.25)[i]
        outer.append(_tongue((d[0] * 0.07, 0.1, d[1] * 0.07), h, 0.055, lean=(d[0] * 0.06, d[1] * 0.06),
                             sway=0.25 * (1 if i % 2 else -1)))
    core = [_tongue((0.0, 0.1, 0.0), 0.44, 0.07, lean=(0.015, -0.01), sway=0.4),
            _tongue((0.035, 0.1, 0.04), 0.34, 0.05, lean=(0.04, 0.05), sway=-0.5),
            _tongue((-0.04, 0.1, -0.02), 0.3, 0.05, lean=(-0.05, -0.02), sway=0.5)]
    hot = Material(color=_FlameColor(0.04, 0.4, glow), emissive=1.9, roughness=0.6, specular=0.0, shadow=False)
    heartm = Material(color=_FlameColor(0.0, 0.62, glow), emissive=3.6, roughness=0.6, specular=0.0, shadow=False)
    m = _pm("campfire", [("logs", [wood], Material(color=_S.wood("#8a5a34", "#5a3a20"), roughness=0.7, shadow=0.35)),
                         ("embers", [embers] + coals, Material(color="#ff5a1e", emissive=1.6, roughness=0.8,
                                                              specular=0.1, shadow=False)),
                         ("stones", stones, _matt("#8a8a90", 0.8)),
                         ("flame", outer, hot),
                         ("flame core", core, heartm)], upright=False)
    return m


def bench(color="#b8844a", width=1.4, legs="#3a3a40"):
    """Park bench: wooden slats ('seat' part), backrest, metal legs."""
    slats = [rounded_box(width, 0.03, 0.09, 0.01).at(0, 0.45, z) for z in (-0.12, 0.0, 0.12)]
    back = [_xf(rounded_box(width, 0.09, 0.03, 0.01), _Rx(-12), (0, y, -0.2)) for y in (0.62, 0.76)]
    lg = []
    for x in (-width / 2 + 0.1, width / 2 - 0.1):
        lg += [rounded_box(0.04, 0.45, 0.04, 0.01).at(x, 0.225, z) for z in (-0.15, 0.15)]
        lg.append(_xf(rounded_box(0.04, 0.5, 0.04, 0.01), _Rx(-12), (x, 0.66, -0.21)))
        lg.append(rounded_box(0.04, 0.04, 0.4, 0.01).at(x, 0.42, 0))
    wm = Material(color=_S.wood(color, _shade(color, 0.7)), roughness=0.5, specular=0.3)
    return _pm("bench", [("seat", slats, wm), ("back", back, wm),
                         ("legs", lg, Material(color=legs, roughness=0.4, specular=0.5, metallic=0.6))], upright=False)


def stool(color="#d8843a", height=0.5, legs="#b88a5a"):
    """Round stool ('seat' part) on four legs."""
    seat = cylinder(0.17, 0.05, round=0.02).at(0, height - 0.025, 0)
    lg = [round_cone((math.cos(a) * 0.1, height - 0.05, math.sin(a) * 0.1), (math.cos(a) * 0.15, 0.0, math.sin(a) * 0.15), 0.018, 0.015)
          for a in np.radians([45, 135, 225, 315])]
    return _pm("stool", [("seat", [seat], _matt(color, 0.5)), ("legs", lg, Material(color=_S.wood(legs), roughness=0.5))],
               upright=False)


def spoon(color="#c8945a"):
    s = smooth_union(capsule((0, 0, 0), (0, 0.24, 0), 0.008), ellipsoid(0.03, 0.045, 0.012).at(0, 0.28, 0), k=0.01)
    return _pm("spoon", [("spoon", [s], Material(color=_S.wood(color), roughness=0.5))], grip=(0, 0.06, 0))


def flask(color="#7ae0b8"):
    body = _S.lathe([(0.0, 0.0), (0.06, 0.0), (0.07, 0.03), (0.02, 0.12), (0.018, 0.17), (0.0, 0.17)], thickness=0.004)
    liq = _S.lathe([(0.0, 0.004), (0.058, 0.004), (0.066, 0.03), (0.04, 0.07), (0.0, 0.07)])
    return _pm("flask", [("glass", [body], Material(color="#eaf6f4", roughness=0.05, specular=1.0, glass=0.85, shadow=0.3)),
                         ("liquid", [liq], Material(color=color, emissive=0.6, roughness=0.2, specular=0.6))],
               grip=(0, 0.13, 0))


def paintbrush(color="#e8413a"):
    h = round_cone((0, 0, 0), (0, 0.22, 0), 0.007, 0.01)
    tip = round_cone((0, 0.23, 0), (0, 0.27, 0), 0.01, 0.003)
    return _pm("paintbrush", [("handle", [h], _gloss("#d8b04a")), ("tip", [tip], _matt(color))], grip=(0, 0.06, 0),
               upright=False)


def staff(color="#7a5234", orb="#8ad8ff", height=1.5):
    sh = round_cone((0, 0, 0), (0, height, 0), 0.02, 0.025)
    hook = _chain([np.array([0, height, 0]), np.array([0.05, height + 0.08, 0]), np.array([0.0, height + 0.15, 0])], [0.025, 0.022, 0.02])
    return _pm("staff", [("staff", [sh] + hook, Material(color=_S.wood(color), roughness=0.6)),
                         ("orb", [sphere(0.05).at(0.0, height + 0.1, 0.0)], Material(color=orb, emissive=1.8, roughness=0.2))],
               grip=(0, height * 0.55, 0))


def sword(color="#d8dce2"):
    blade = _S.prism([(-0.025, 0), (0.025, 0), (0.025, 0.6), (0, 0.66), (-0.025, 0.6)], depth=0.012).at(0, 0.12, 0)
    guard = rounded_box(0.18, 0.025, 0.03, 0.01).at(0, 0.12, 0)
    hilt = cylinder(0.014, 0.11).at(0, 0.06, 0)
    return _pm("sword", [("blade", [blade], Material(color=color, metallic=1.0, roughness=0.15)),
                         ("hilt", [guard, hilt, sphere(0.022).at(0, 0.0, 0)], Material(color="#d8b04a", metallic=0.8, roughness=0.3))],
               grip=(0, 0.06, 0), upright=True)


def wand(color="#2a2224", tip="#ffe08a"):
    w = round_cone((0, 0, 0), (0, 0.3, 0), 0.008, 0.005)
    return _pm("wand", [("wand", [w], _gloss(color)), ("spark", [sphere(0.018).at(0, 0.31, 0)],
                                                       Material(color=tip, emissive=2.5, shadow=False))], grip=(0, 0.06, 0))


def briefcase(color="#5a3a26"):
    b = rounded_box(0.42, 0.3, 0.1, 0.03).at(0, 0.15, 0)
    h = intersect(_xf(torus(0.06, 0.012), _Rx(90), (0, 0.31, 0)), plane((0, -1, 0), -0.31))
    return _pm("briefcase", [("case", [b], Material(color=color, roughness=0.35, specular=0.5)),
                             ("handle", [h], _matt("#2a2224"))], grip=(0, 0.36, 0))


def apple(color="#d8302a"):
    a = smooth_union(sphere(0.04).at(-0.012, 0.04, 0), sphere(0.04).at(0.012, 0.04, 0), k=0.02)
    return _pm("apple", [("apple", [a], _gloss(color)), ("stem", [capsule((0, 0.075, 0), (0.005, 0.095, 0), 0.003)],
                                                         _matt("#5a3a24"))], grip=(0, 0.03, 0))


def sandwich(bread="#e8c080"):
    b = [rounded_box(0.12, 0.02, 0.1, 0.008).at(0, y, 0) for y in (0.01, 0.05)]
    fill = [rounded_box(0.125, 0.012, 0.105, 0.005).at(0, 0.026, 0), rounded_box(0.12, 0.008, 0.1, 0.004).at(0, 0.037, 0)]
    return _pm("sandwich", [("bread", b, _matt(bread, 0.8)), ("lettuce", fill[:1], _matt("#7ac84a", 0.6)),
                            ("cheese", fill[1:], _matt("#f6d23a", 0.5))], grip=(0, 0.03, 0))


def cake(color="#f6b8c8", cream="#fbf6ee"):
    c = cylinder(0.12, 0.09, round=0.015).at(0, 0.045, 0)
    top = cylinder(0.125, 0.02, round=0.01).at(0, 0.095, 0)
    berries = [sphere(0.014).at(math.cos(a) * 0.08, 0.11, math.sin(a) * 0.08) for a in np.linspace(0, 6.28, 7)[:-1]]
    plate = cylinder(0.16, 0.01, round=0.005).at(0, 0.005, 0)
    return _pm("cake", [("plate", [plate], _gloss("#f8f6f2")), ("cake", [c], _matt(color, 0.6)), ("cream", [top], _matt(cream, 0.5)),
                        ("berries", berries, _gloss("#d8202a"))], upright=False)


def juice(color="#f2a23a"):
    g = _S.lathe([(0, 0), (0.035, 0), (0.04, 0.12), (0, 0.12)], thickness=0.003)
    liq = cylinder(0.034, 0.08).at(0, 0.045, 0)
    straw = capsule((0.01, 0.06, 0), (0.025, 0.17, 0.01), 0.004)
    return _pm("juice", [("glass", [g], Material(color="#f2f8f8", roughness=0.05, specular=1.0, glass=0.85, shadow=0.25)),
                         ("juice", [liq], Material(color=color, roughness=0.15, specular=0.6, subsurface=0.4)),
                         ("straw", [straw], _matt("#e8413a"))], grip=(0, 0.06, 0))


def kite(color="#e8413a", accent="#f2d23a"):
    k = _S.prism([(0, 0.35), (0.22, 0.05), (0, -0.4), (-0.22, 0.05)], depth=0.01)
    tail = _chain([np.array([0, -0.4, 0]), np.array([0.06, -0.6, 0]), np.array([-0.03, -0.8, 0])], [0.004] * 3)
    bows = [ellipsoid(0.03, 0.015, 0.01).at(0.03 * (-1) ** i, -0.5 - 0.12 * i, 0) for i in range(3)]
    return _pm("kite", [("kite", [k], _matt(color, 0.5)), ("tail", tail + bows, _matt(accent, 0.5))], upright=True)


PROPS = {"balloon": balloon, "book": book, "cup": cup, "phone": phone, "umbrella": umbrella, "guitar": guitar,
         "ball": ball, "flower": flower, "bouquet": bouquet, "ice_cream": ice_cream, "sign": sign, "flag": flag,
         "lantern": lantern, "fishing_rod": fishing_rod, "skateboard": skateboard, "bicycle": bicycle,
         "picnic_blanket": picnic_blanket, "basket": basket, "campfire": campfire, "bench": bench, "stool": stool,
         "spoon": spoon, "flask": flask, "paintbrush": paintbrush, "staff": staff, "sword": sword, "wand": wand,
         "briefcase": briefcase, "apple": apple, "sandwich": sandwich, "cake": cake, "juice": juice, "kite": kite}


def prop(name, **kw):
    """A prop by name (``PROPS``): ``prop("balloon", color="#3a8ae8")``, ``prop("sign", text="OPEN")``. Every prop is
    a solid Model with named parts, standing on the floor; hand-held ones carry a grip point for ``Character.hold``."""
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    name = key if key in PROPS else name
    if name not in PROPS:
        raise _unknown("prop", name, PROPS, "Try clay.find('...') or clay.names('prop').")
    fn = PROPS[name]
    return fn(**_reject("prop " + name, kw, _sigopts(fn)))
