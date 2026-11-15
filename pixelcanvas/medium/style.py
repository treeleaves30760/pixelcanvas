"""Style spec: the same dials for every medium. A named look is a setting of these, not code."""
from dataclasses import dataclass, field, replace


@dataclass
class Style:
    simplify: float = 0.0             # form simplification: 0 = all detail, 1 = big shapes only
    line: float = 0.0                 # contour weight in px at 1200 wide (0 = no line)
    line_color: str = "#1b1712"
    bands: int = 0                    # value bands (0 = continuous)
    contrast: float = 1.0
    exposure: float = 1.0             # relative to auto exposure
    palette: int = 0                  # number of colours (0 = free)
    saturation: float = 1.0
    temperature: float = 0.0          # -1 cool .. +1 warm
    shadow_shift: float = 0.0         # shadows toward cool blue-violet (+) / warm (-)
    palette_colors: tuple = ()        # optional fixed palette (hex)
    edge: float = 0.5                 # 0 = soft / lost edges, 1 = hard / crisp
    mark: str = "none"                # none | band | stroke | wash | hatch | wash+hatch | pixel
    mark_size: float = 0.0            # largest mark in px at 1200 wide
    mark_min: float = 0.0             # smallest mark
    direction: str = "form"           # form | fixed
    angle: float = 35.0               # fixed angle (deg) or fallback
    jitter: float = 0.0               # colour / placement jitter
    ground: str = "none"              # none | canvas | paper
    ground_color: str = "#ffffff"
    texture: float = 0.0              # strength of ground / mark texture
    key: float = 1.0
    ambient: float = 1.0
    shadow: float = 1.0               # cast-shadow strength (1 = as lit, 0 = none)
    bloom: float = 0.0
    extra: dict = field(default_factory=dict)   # medium-specific (impasto, granulation, pixel size ...)

    def but(self, **kw):
        ex = dict(self.extra)
        ex.update(kw.pop("extra", {}))
        return replace(self, extra=ex, **kw)


STYLES = {
    "render": Style(bloom=0.35, contrast=1.05, edge=1.0),
    "flat": Style(simplify=0.35, bands=3, saturation=0.95, contrast=1.1, edge=1.0, mark="band", shadow_shift=0.5,
                  temperature=0.1),
    "oil": Style(simplify=0.2, mark="stroke", mark_size=34, mark_min=3.5, direction="form", jitter=0.06,
                 ground="canvas", ground_color="#8a6a48", texture=0.8, saturation=1.12, contrast=1.08,
                 shadow_shift=0.35, temperature=0.08, edge=0.5, extra=dict(impasto=1.0, passes=5, threshold=0.045)),
    "watercolour": Style(simplify=0.12, mark="wash", bands=3, ground="paper", ground_color="#fbf8f1", texture=0.7,
                         contrast=0.85, exposure=1.3, shadow_shift=0.7, edge=0.55,
                         extra=dict(granulation=0.4, pooling=0.9, bleed=0.35, pencil=0.5, border=1.0)),
    "ink": Style(line=2.2, mark="hatch", bands=4, mark_size=6.5, ground="paper", ground_color="#f3eddf", texture=0.25,
                 contrast=1.15, edge=1.0, direction="form", extra=dict(spot_black=0.025)),
    "pixel": Style(bands=4, palette=20, saturation=1.2, contrast=1.1, shadow_shift=0.6, line=1, mark="pixel", edge=1.0,
                   extra=dict(pixel=6, dither=0.0)),
}


def style_for(medium, strength=1.0, mark=None, palette=None, paper=None):
    """The style of a medium with the studio's dials: ``strength`` 0-1 (how much of the medium's character:
    simplification, marks, texture, colour shift), ``mark`` (a multiplier of the stroke / hatch / pixel size: 0.5 fine, 2 broad),
    ``palette`` (number of colours), ``paper`` (ground colour)."""
    st = STYLES[medium]
    s = max(0.0, min(float(strength), 2.0))
    kw = dict(simplify=st.simplify * s, texture=st.texture * s, shadow_shift=st.shadow_shift * s,
              jitter=st.jitter * s, saturation=1 + (st.saturation - 1) * s, contrast=1 + (st.contrast - 1) * s)
    ex = dict(st.extra)
    if medium == "oil":
        ex["impasto"] = st.extra.get("impasto", 1.0) * s
    if medium == "watercolour":
        ex.update(granulation=st.extra["granulation"] * s, bleed=st.extra["bleed"] * s, pencil=st.extra["pencil"] * s)
    if mark is not None:                                 # a multiplier of the medium's own mark size
        m = max(0.2, float(mark))
        if medium == "pixel":
            ex["pixel"] = max(2.0, st.extra.get("pixel", 6) * m)
        elif st.mark_size:
            kw["mark_size"] = st.mark_size * m
            kw["mark_min"] = max(1.0, st.mark_min * m) if st.mark_min else 0.0
        elif st.line:
            kw["line"] = st.line * m
    if palette is not None:
        kw["palette"] = int(palette)
    if paper is not None:
        kw["ground_color"] = paper
    kw["extra"] = ex
    return st.but(**kw)
