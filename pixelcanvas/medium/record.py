"""What a drawing call knows about what it paints, recorded while a studio section runs.

``Sketch.paint`` reports each paint: its mask, its local colour (albedo, without the painted shading), a volume
hint (from ``volume=`` / ``shade=``; negative = hollow), a gloss hint (``spec=``), and its contact / drop shadow and
glow as separate shadow / light records. The medium buffers read these instead of estimating them from the
finished layer. Nothing is recorded unless a recorder is active for that canvas (the studio starts one per section).
"""
import numpy as np

F = np.float32
ACTIVE = []                                              # recorders, innermost last
LIMIT = 600                                              # records per section (more: the layer is estimated)


class Recorder:
    def __init__(self, cv):
        self.cv = cv
        self.items = []
        self.overflow = False


def start(cv):
    r = Recorder(cv)
    ACTIVE.append(r)
    return r


def stop(r):
    if r in ACTIVE:
        ACTIVE.remove(r)
    return r


def _active(cv):
    for r in reversed(ACTIVE):
        if r.cv is cv:
            return r
    return None


def active(cv):
    return _active(cv) is not None


def note(cv, kind, mask, color, volume=0.0, gloss=0.0, strength=1.0, off=None, normals=None, hollow=None,
         painted=None):
    """Record one paint on ``cv``: kind 'body' (local colour, relief from ``volume``), 'line' (flat dark detail),
    'shadow' (darkens what is under it; not part of the thing), 'light' (a glow: emission, not paint), 'solid'
    (a form solid: local colour and its true normals). ``mask`` is full frame, or a crop at ``off`` = (x0, y0);
    ``color`` an RGB triple or a colour map of the mask's size; ``normals`` (h, w, 3) like the mask (x right,
    y down, z toward the viewer); ``hollow`` (like the mask) the part that is an opening (a bowl's inside);
    ``painted`` (like the mask) the colour the call left on the canvas (shading included), so the medium can tell
    later paint the records do not know from the recorded shape. Returns the record (or None)."""
    r = _active(cv)
    if r is None:
        return None
    if len(r.items) >= LIMIT:
        r.overflow = True
        return None
    m = np.asarray(mask, F)
    if m.ndim == 3:
        m = m.max(-1)
    nz = m > 1.0 / 255
    if not nz.any():
        return None
    ox, oy = off if off is not None else (0, 0)
    rows, cols = np.flatnonzero(nz.any(1)), np.flatnonzero(nz.any(0))
    y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    c = np.asarray(color, F)
    if c.ndim == 3:
        c = np.broadcast_to(c, m.shape + (3,))[y0:y1, x0:x1] if c.shape[:2] != m.shape else c[y0:y1, x0:x1]
    else:
        c = c.reshape(-1)[:3]
    item = {"kind": kind, "off": (int(ox) + x0, int(oy) + y0), "mask": np.round(np.clip(m[y0:y1, x0:x1], 0, 1) * 255)
            .astype(np.uint8), "color": np.clip(c, 0, 4).astype(F), "volume": float(volume), "gloss": float(gloss),
            "strength": float(strength), "order": len(r.items)}
    if normals is not None:
        n = np.asarray(normals, F)
        if n.shape[:2] == m.shape:
            item["normals"] = np.round((np.clip(n[y0:y1, x0:x1], -1, 1) * 0.5 + 0.5) * 255).astype(np.uint8)
    if hollow is not None:
        hm = np.asarray(hollow, F)
        if hm.shape == m.shape and (hm > 0.5).any():
            item["hollow"] = np.round(np.clip(hm[y0:y1, x0:x1], 0, 1) * 255).astype(np.uint8)
    if painted is not None:
        pc = np.asarray(painted, F)
        if pc.shape[:2] == m.shape:
            item["painted"] = np.round(np.clip(pc[y0:y1, x0:x1, :3], 0, 1) * 255).astype(np.uint8)
    r.items.append(item)
    return item


def painted(cv, item):
    """Store on ``item`` the colour the canvas now shows under it (after the call's shading)."""
    if item is None:
        return
    ox, oy = item["off"]
    h, w = item["mask"].shape
    buf = np.asarray(cv.buf)
    crop = buf[oy:oy + h, ox:ox + w, :3]
    if crop.shape[:2] == (h, w):
        item["painted"] = np.round(np.clip(crop, 0, 1) * 255).astype(np.uint8)
