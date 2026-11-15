"""pixelcanvas.medium: a layered 2D picture delivered in a real art medium.

The picture's layers become per-pixel buffers (albedo without painted shading, a relief height per shape, normals,
depth from draw order, ids, emission); ONE key light lights them (diffuse, form shadow, cast shadows on a floor plane,
contact occlusion, specular); a MEDIUM paints the result as a process over the buffers:

    render       light, tone and bloom: a 3D-render look
    flat         flat graphic: posterised local colour in value bands
    oil          strokes along the form, coarse to fine, on canvas with impasto
    watercolour  washes in value bands on paper: pooled edges, granulation, reserved whites, pencil
    ink          contours and hatching that wraps the form, spot blacks, on paper
    pixel        a low-resolution grid in a small palette, value bands, outlines

In the studio ``set scene medium=oil`` delivers the picture in oil. Outside the studio::

    from pixelcanvas.medium import buffers_from_layers, layer, paint, record
    rec = record.start(cv)            # Sketch.paint and form solids report what they paint
    ...                               # draw the objects
    record.stop(rec)
    b = buffers_from_layers([layer(bg, role="backdrop"), layer(cv, rec, "objects")])
    img = paint(b, "watercolour")

``layers``: back to front, each ``dict(name, role, P, A)`` (premultiplied colour and alpha, full frame) with
optional ``records`` (what ``Sketch.paint`` recorded) and ``text``.
"""
MEDIA = ("render", "flat", "oil", "watercolour", "ink", "pixel")
LABELS = {"render": "render", "flat": "flat graphic", "oil": "oil", "watercolour": "watercolour", "ink": "ink",
          "pixel": "pixel art"}
ALIASES = {"watercolor": "watercolour", "water colour": "watercolour", "gouache": "oil", "3d": "render",
           "photo": "render", "flat graphic": "flat", "pixel art": "pixel", "none": "off"}


def detect(prompt):
    """The medium a request names, or None: 'a watercolour of', 'oil painting', 'ink drawing', 'pixel art',
    'gouache', '3D render' -> render. A photo or realistic request names no medium: the drawing itself is lit
    solids without outlines, and `set scene medium=render` relights it on request."""
    import re
    t = " " + str(prompt or "").lower() + " "
    rules = [("watercolour", r"\bwater\s?colou?rs?\b|\baquarelle\b"),
             ("oil", r"\boil(?:\s+painting|\s+on\s+canvas|s)?\b|\bgouache\b|\bimpasto\b|\bacrylic\b"),
             ("ink", r"\bink\b|\bpen and ink\b|\bpen drawing\b|\bcross-?hatch|\betching\b|\bengraving\b"),
             ("pixel", r"\bpixel[\s-]?art\b|\b8-?bit\b|\b16-?bit\b|\bpixelated\b"),
             ("flat", r"\bflat (?:graphic|vector|design|illustration)\b|\bvector art\b"),
             ("render", r"\b3-?d render(?:ed|ing)?\b|\brender(?:ed)? in 3-?d\b")]
    for name, pat in rules:
        if re.search(pat, t):
            return name
    return None


def canonical(name):
    n = str(name or "").strip().lower()
    n = ALIASES.get(n, n)
    return n


def paint(b, medium, style=None, **dials):
    """The medium picture (sRGB float, H x W x 3) from buffers ``b``; ``style`` a Style, or dials for
    ``style.style_for`` (strength, mark, palette, paper)."""
    from . import media
    from .style import style_for
    medium = canonical(medium)
    st = style if style is not None else style_for(medium, **dials)
    img = media.MEDIA[medium](b, st)
    return media.with_text(img, b, medium)


def stages(b, medium, style=None, **dials):
    """[(label, sRGB image)] of the stages: flat silhouettes (local colour) -> relief clay (grey, lit) -> lit colour
    -> the medium."""
    import numpy as np
    from . import media
    from .core import lin_to_srgb
    from .style import style_for, STYLES
    medium = canonical(medium)
    st = style if style is not None else style_for(medium, **dials)
    base = STYLES["render"].but(bloom=0.0)
    sil = np.where((b["id"] >= 0)[..., None], lin_to_srgb(np.clip(b["albedo"], 0, 1)), media.tone(b, b["beauty"], base,
                                                                                                    shift=False))
    grey = np.full_like(b["albedo"], 0.5)
    clay = media.tone(b, media.lit(b, base, albedo=grey), base, shift=False)
    litc = media.tone(b, media.lit(b, base), base, shift=False)
    return [("flat silhouettes", sil), ("relief clay", clay), ("lit colour", litc),
            (LABELS[medium], paint(b, medium, style=st))]


def layer(cv, recorder=None, name="layer", role="object", text=False):
    """A medium layer from a canvas drawn outside the studio: ``role="backdrop"`` takes the whole canvas; an object
    layer (drawn on a black canvas) takes what ``recorder`` (``record.start(cv)``) saw painted, with its records."""
    import numpy as np
    buf = np.clip(np.asarray(cv.buf, np.float32)[..., :3], 0, 1)
    H, W = buf.shape[:2]
    if role == "backdrop" or recorder is None:
        return dict(name=name, role=role, P=buf.copy(), A=np.ones((H, W), np.float32), text=text, records=None)
    A = np.zeros((H, W), np.float32)
    recs = []
    for r in recorder.items:
        m = r["mask"].astype(np.float32) / 255
        x0, y0 = r["off"]
        h, w = m.shape
        c = np.asarray(r["color"], np.float32)
        col = c if c.ndim == 3 else np.broadcast_to(c.reshape(1, 1, 3), (h, w, 3))
        if r["kind"] in ("body", "line", "solid"):
            A[y0:y0 + h, x0:x0 + w] = np.maximum(A[y0:y0 + h, x0:x0 + w], m)
        n = r.get("normals")
        recs.append(dict(kind=r["kind"], A=m, P=np.clip(col, 0, 1) * m[..., None], x0=x0, y0=y0,
                         volume=r["volume"], gloss=r["gloss"], strength=r["strength"], order=r["order"],
                         color=r["color"], N=None if n is None else n.astype(np.float32) / 255 * 2 - 1,
                         hollow=None if r.get("hollow") is None else r["hollow"].astype(np.float32) / 255,
                         PP=None if r.get("painted") is None else r["painted"].astype(np.float32) / 255 * m[..., None]))
    return dict(name=name, role=role, P=buf * (A[..., None] > 0), A=A, text=text, records=recs or None)


def buffers_from_layers(layers, light=(-0.6, -0.8), max_side=1200, **kw):
    from .buffers import build
    return build(layers, light=light, max_side=max_side, **kw)
