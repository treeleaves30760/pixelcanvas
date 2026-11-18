"""Shared helpers for the export / fx / text tests: real engine, synthetic pictures only, tiny canvases."""
from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="module")
def cwd_module():
    """The tests chdir into their project folders; put the working directory back when the module is done."""
    here = os.getcwd()
    yield
    os.chdir(here)


def run(*argv):
    """Run one easel command line in-process; returns (code, text)."""
    from pixelcanvas.easel import cli
    return cli.run([str(a) for a in argv])


def ok(*argv) -> str:
    code, out = run(*argv)
    assert code == 0, out
    assert out.splitlines()[-1].startswith("NEXT: "), out
    return out


def new_project(root: Path, name: str = "pic", size: str = "160x120", ref: Optional[np.ndarray] = None) -> Path:
    os.chdir(root)
    if ref is not None:
        Image.fromarray(ref).save(root / "ref.png")
        ok("new", name, "--ref", "ref.png")
    else:
        ok("new", name, "--size", size)
    return root / f"{name}.easel"


def synthetic_picture(w: int = 160, h: int = 120) -> np.ndarray:
    """A soft gradient with two coloured blocks."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32)
    img[..., 0] = 120 + 90 * xx / w
    img[..., 1] = 150 + 60 * yy / h
    img[..., 2] = 200 - 70 * xx / w
    img[h // 4:h // 2, w // 8:w // 3] = (220, 60, 50)
    img[h // 2:3 * h // 4, w // 2:3 * w // 4] = (40, 90, 160)
    return np.clip(img, 0, 255).astype(np.uint8)


def paint_demo(root: Path, name: str = "pic") -> Path:
    """A small project with hand-tool marks on several layers, a layer setting and an undone step."""
    proj = new_project(root, name)
    ok("fill", "500,500", "--color", "#e8dcc0", "--layer", "flat", "--bounds", "canvas")
    ok("shape", "ellipse", "200,200,500,700", "--color", "#c04030", "--layer", "shade")
    ok("stroke", "100,100 300,150 500,120 700,200", "--size", "5", "--color", "#102030", "--layer", "line")
    ok("stroke", "100,800 500,700", "--size", "6", "--color", "#ff0000", "--layer", "line")      # undone below
    ok("undo")
    ok("stroke", "100,850 900,820", "--size", "6", "--color", "#00aa55", "--layer", "line")
    ok("layer", "shade", "--mode", "multiply", "--opacity", "0.8")
    ok("shape", "rect", "600,500,900,900", "--color", "#3050c0", "--layer", "detail")
    return proj


# ------------------------------------------------------------------------------------------------ PSD reader

def _unpack_bits(data: bytes, width: int) -> bytes:
    out = bytearray()
    i = 0
    while len(out) < width and i < len(data):
        n = data[i]
        i += 1
        if n < 128:
            out += data[i:i + n + 1]
            i += n + 1
        elif n > 128:
            out += bytes([data[i]]) * (257 - n)
            i += 1
    return bytes(out[:width])


def read_psd(path) -> dict:
    """Header, merged size and the layer records (name, blend key, opacity, clipping, visible, box, rgba)."""
    data = Path(path).read_bytes()
    sig, ver, _, nch, H, W, depth, mode = struct.unpack(">4sH6sHIIHH", data[:26])
    assert sig == b"8BPS" and ver == 1 and depth == 8 and mode == 3
    pos = 26
    for _ in range(2):                      # colour mode data, image resources
        (n,) = struct.unpack(">I", data[pos:pos + 4])
        pos += 4 + n
    (lmi_len,) = struct.unpack(">I", data[pos:pos + 4])
    pos += 4
    (li_len,) = struct.unpack(">I", data[pos:pos + 4])
    pos += 4
    (count,) = struct.unpack(">h", data[pos:pos + 2])
    pos += 2
    recs = []
    for _ in range(count):
        top, left, bottom, right, nc = struct.unpack(">iiiiH", data[pos:pos + 18])
        pos += 18
        chans = []
        for _ in range(nc):
            cid, ln = struct.unpack(">hI", data[pos:pos + 6])
            pos += 6
            chans.append((cid, ln))
        assert data[pos:pos + 4] == b"8BIM"
        key = data[pos + 4:pos + 8]
        opacity, clipping, flags, _ = struct.unpack(">BBBB", data[pos + 8:pos + 12])
        pos += 12
        (extra_len,) = struct.unpack(">I", data[pos:pos + 4])
        extra = data[pos + 4:pos + 4 + extra_len]
        pos += 4 + extra_len
        # extra: mask (4) + blending ranges (4) + pascal name padded to 4, then 8BIM luni ...
        k = 8
        nlen = extra[k]
        k += 1 + nlen
        k += (4 - (1 + nlen) % 4) % 4
        name = None
        while k + 12 <= len(extra):
            assert extra[k:k + 4] == b"8BIM"
            tag = extra[k + 4:k + 8]
            (tl,) = struct.unpack(">I", extra[k + 8:k + 12])
            if tag == b"luni":
                (nchars,) = struct.unpack(">I", extra[k + 12:k + 16])
                name = extra[k + 16:k + 16 + 2 * nchars].decode("utf-16-be")
            k += 12 + tl
        recs.append({"name": name, "key": key, "opacity": opacity, "clip": clipping, "visible": not (flags & 2),
                     "box": (left, top, right, bottom), "chans": chans})
    for r in recs:
        h, w = r["box"][3] - r["box"][1], r["box"][2] - r["box"][0]
        planes = {}
        for cid, ln in r["chans"]:
            blk = data[pos:pos + ln]
            pos += ln
            (comp,) = struct.unpack(">H", blk[:2])
            if h * w == 0:
                planes[cid] = np.zeros((0, 0), np.uint8)
                continue
            assert comp == 1
            counts = struct.unpack(f">{h}H", blk[2:2 + 2 * h])
            q = 2 + 2 * h
            rows = []
            for c in counts:
                rows.append(np.frombuffer(_unpack_bits(blk[q:q + c], w), np.uint8))
                q += c
            planes[cid] = np.stack(rows)
        r["rgba"] = (np.stack([planes[0], planes[1], planes[2], planes[-1]], -1)
                     if h * w else np.zeros((0, 0, 4), np.uint8))
    return {"size": (W, H), "layers": recs}
