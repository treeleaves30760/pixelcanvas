"""``easel export``: PNG, layered PSD / ORA, time-lapse film (GIF / MP4) and the stage pictures.

* PNG: the final composite, sketch layer left out.
* PSD / ORA: the project's real layers with names, blend modes, opacities, visibility and clipping.
* Time-lapse: the journal replayed from an empty canvas. Only steps that are still applied take part (undone steps
  are gone), every painting step is painted progressively with ``ops.apply_ops_progressive`` onto its layer, and the
  layer order / blend modes / visibility are those the project had at that step (``history/<n>/before/state.json``).
  The finished layer of a step is taken from the history snapshots when they exist, so the film ends exactly on the
  exported picture even for pixel filters (``fx``, ``text``, ``smooth`` ...) that have no ops.
* Stages: one PNG per closed stage plus a contact sheet ``stages.jpg``.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from . import compose, coords
from .coords import UsageError

FPS = 12.5                 # frames per second of the film (80 ms per frame: GIF delays are in 10 ms units)
HOLD_SECONDS = 2.5         # the final picture stays this long
GIF_SIDE = 640
MP4_SIDE = 1080

PSD_KEYS = {"normal": b"norm", "multiply": b"mul ", "screen": b"scrn", "overlay": b"over", "add": b"lddg"}
ORA_OPS = {"normal": "svg:src-over", "multiply": "svg:multiply", "screen": "svg:screen", "overlay": "svg:overlay",
           "add": "svg:plus"}


# ============================================================================================ small helpers

def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_layer_file(path: Path) -> Optional[np.ndarray]:
    """A layer PNG as float32 [H,W,4] 0..1; None when the file is missing or fully transparent."""
    try:
        with Image.open(path) as im:
            a = np.asarray(im.convert("RGBA"), dtype=np.float32) / 255.0
    except (OSError, ValueError):
        return None
    return a if (a[..., 3] > 0).any() else None


def _sketch_shown(state: dict) -> bool:
    e = next((x for x in state["layers"] if x["name"] == "sketch"), None)
    if e is None or not e.get("visible", True):
        return False
    if e.get("force_visible"):
        return True
    return state.get("stage") in ("plan", "sketch", "line")


def _signature(state: dict):
    return (tuple((e["name"], e.get("mode", "normal"), round(float(e.get("opacity", 1.0)), 4),
                   bool(e.get("visible", True)), bool(e.get("clip"))) for e in state["layers"]),
            _sketch_shown(state), tuple(state.get("paper", ())))


def _bbox_of(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys = np.flatnonzero(mask.any(1))
    if len(ys) == 0:
        return None
    xs = np.flatnonzero(mask.any(0))
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1


def _union(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def find_ffmpeg() -> Optional[str]:
    env = os.environ.get("EASEL_FFMPEG")
    if env and os.path.isfile(env):
        return env
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    for c in ("C:/ffmpeg/bin/ffmpeg.exe", "/c/ffmpeg/bin/ffmpeg.exe", "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg",
              "/opt/homebrew/bin/ffmpeg"):
        if os.path.isfile(c):
            return c
    return None


# ============================================================================================ layered files

@dataclass
class LayerRec:
    name: str
    rgba: np.ndarray            # uint8 [h,w,4] cropped to bbox (empty array when the layer is empty)
    bbox: Tuple[int, int, int, int]
    visible: bool
    opacity: float
    mode: str
    clip: bool


def layer_records(p, bake_clip: bool = False) -> List[LayerRec]:
    """The project's layers bottom to top. The sketch layer is always hidden (it is scaffolding, not picture).
    ``bake_clip``: multiply a clipped layer's alpha by its base's alpha (for formats without clipping)."""
    recs: List[LayerRec] = []
    base_alpha = None
    for e in p.state["layers"]:
        name = e["name"]
        arr = p.layer(name, create=False)
        clip = bool(e.get("clip"))
        if not clip:
            base_alpha = None if arr is None else arr[..., 3] * float(e.get("opacity", 1.0))
        if arr is not None and clip and bake_clip:
            arr = arr.copy()
            arr[..., 3] *= np.clip(base_alpha, 0, 1) if base_alpha is not None else 0.0
        visible = bool(e.get("visible", True)) and name != "sketch"
        if arr is None:
            recs.append(LayerRec(name, np.zeros((0, 0, 4), np.uint8), (0, 0, 0, 0), visible,
                                 float(e.get("opacity", 1.0)), e.get("mode", "normal"), clip))
            continue
        u8 = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        box = _bbox_of(u8[..., 3] > 0)
        if box is None:
            recs.append(LayerRec(name, np.zeros((0, 0, 4), np.uint8), (0, 0, 0, 0), visible,
                                 float(e.get("opacity", 1.0)), e.get("mode", "normal"), clip))
            continue
        x0, y0, x1, y1 = box
        crop = np.ascontiguousarray(u8[y0:y1, x0:x1])
        crop[crop[..., 3] == 0, :3] = 0
        recs.append(LayerRec(name, crop, box, visible, float(e.get("opacity", 1.0)), e.get("mode", "normal"), clip))
    return recs


def _packbits_rows(plane):
    """PackBits-compress each row of a (h, w) uint8 plane -> (row byte counts, data)."""
    counts, chunks = [], []
    plane = np.ascontiguousarray(plane, np.uint8)
    for row in plane:
        n = len(row)
        b = row.tobytes()
        out = bytearray()
        change = np.flatnonzero(row[1:] != row[:-1]) + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [n]))
        long = (ends - starts) >= 3
        pos = 0
        for s, e in zip(starts[long].tolist(), ends[long].tolist()):
            while pos < s:
                k = min(s, pos + 128)
                out.append(k - pos - 1)
                out += b[pos:k]
                pos = k
            while pos < e:
                k = min(e, pos + 128)
                out.append(257 - (k - pos) if k - pos >= 2 else 0)
                out.append(b[pos])
                pos = k
        while pos < n:
            k = min(n, pos + 128)
            out.append(k - pos - 1)
            out += b[pos:k]
            pos = k
        counts.append(len(out))
        chunks.append(bytes(out))
    return counts, b"".join(chunks)


def _channel_rle(plane) -> bytes:
    if plane.size == 0:
        return struct.pack(">H", 0)
    counts, data = _packbits_rows(plane)
    return struct.pack(">H", 1) + struct.pack(f">{len(counts)}H", *counts) + data


def write_psd(p, path: Path) -> int:
    """8-bit RGB PSD: one pixel layer per project layer (bottom to top) with blend mode, opacity, visibility and
    clipping, plus the merged picture. Returns the number of layers."""
    W, H = p.W, p.H
    recs = layer_records(p)
    records, chandata = [], []
    empty = np.zeros((0, 0), np.uint8)
    for r in recs:
        x0, y0, x1, y1 = r.bbox
        if r.rgba.size:
            planes = [(-1, r.rgba[..., 3]), (0, r.rgba[..., 0]), (1, r.rgba[..., 1]), (2, r.rgba[..., 2])]
        else:
            planes = [(-1, empty)] + [(c, empty) for c in range(3)]
        blocks = [(cid, _channel_rle(pl)) for cid, pl in planes]
        rec = struct.pack(">iiiiH", y0, x0, y1, x1, len(blocks))
        for cid, b in blocks:
            rec += struct.pack(">hI", cid, len(b))
        flags = 0 if r.visible else 0x02
        rec += b"8BIM" + PSD_KEYS.get(r.mode, b"norm") + struct.pack(">BBBB", int(round(r.opacity * 255)),
                                                                     1 if r.clip else 0, flags, 0)
        name_b = r.name.encode("ascii", "replace")[:255]
        pas = bytes([len(name_b)]) + name_b
        pas += b"\0" * ((4 - len(pas) % 4) % 4)
        luni = struct.pack(">I", len(r.name)) + r.name.encode("utf-16-be")
        luni += b"\0" * ((4 - len(luni) % 4) % 4)
        extra = struct.pack(">I", 0) + struct.pack(">I", 0) + pas + b"8BIMluni" + struct.pack(">I", len(luni)) + luni
        rec += struct.pack(">I", len(extra)) + extra
        records.append(rec)
        chandata.append(b"".join(b for _, b in blocks))
    layer_info = struct.pack(">h", len(records)) + b"".join(records) + b"".join(chandata)
    if len(layer_info) % 2:
        layer_info += b"\0"
    lmi = struct.pack(">I", len(layer_info)) + layer_info + struct.pack(">I", 0)
    c8 = p.composite_u8("export")
    with open(path, "wb") as f:
        f.write(b"8BPS" + struct.pack(">H6xHIIHH", 1, 3, H, W, 8, 3))
        f.write(struct.pack(">I", 0))
        f.write(struct.pack(">I", 0))
        f.write(struct.pack(">I", len(lmi)) + lmi)
        counts, datas = [], []
        for c in range(3):
            cc, d = _packbits_rows(c8[..., c])
            counts += cc
            datas.append(d)
        f.write(struct.pack(">H", 1) + struct.pack(f">{len(counts)}H", *counts) + b"".join(datas))
    return len(recs)


def write_ora(p, path: Path) -> int:
    """OpenRaster: stack.xml (top layer first) with names, composite-op, opacity, visibility; one PNG per layer."""
    import io

    def png_bytes(im):
        b = io.BytesIO()
        im.save(b, "PNG")
        return b.getvalue()

    def esc(s):
        return s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")

    recs = layer_records(p, bake_clip=True)       # ORA has no clipping flag: bake it into the alpha
    xml = [f'<?xml version="1.0" encoding="UTF-8"?>\n<image version="0.0.3" w="{p.W}" h="{p.H}">\n<stack>']
    files = []
    for i in range(len(recs) - 1, -1, -1):
        r = recs[i]
        src = f"data/{i:02d}.png"
        if r.rgba.size:
            im = Image.fromarray(r.rgba, "RGBA")
            x, y = r.bbox[0], r.bbox[1]
        else:
            im, x, y = Image.new("RGBA", (1, 1), (0, 0, 0, 0)), 0, 0
        files.append((src, png_bytes(im)))
        xml.append(f' <layer src="{src}" name="{esc(r.name)}" x="{x}" y="{y}" opacity="{r.opacity:.3f}" '
                   f'visibility="{"visible" if r.visible else "hidden"}" '
                   f'composite-op="{ORA_OPS.get(r.mode, "svg:src-over")}"/>')
    xml.append("</stack>\n</image>\n")
    merged = Image.fromarray(p.composite_u8("export"), "RGB")
    thumb = merged.copy()
    thumb.thumbnail((256, 256))
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(zipfile.ZipInfo("mimetype"), "image/openraster", compress_type=zipfile.ZIP_STORED)
        z.writestr("stack.xml", "\n".join(xml))
        for src, data in files:
            z.writestr(src, data)
        z.writestr("mergedimage.png", png_bytes(merged))
        z.writestr("Thumbnails/thumbnail.png", png_bytes(thumb))
    return len(recs)


# ============================================================================================ replay

@dataclass
class Frame:
    rgb: np.ndarray                 # uint8 [H,W,3] at canvas size
    stage: str = ""
    cmd: str = ""
    index: int = 0                  # 1-based event number (0 = the empty canvas, n+1 = the final picture)
    total: int = 0
    kind: str = "progress"          # start | progress | hold | final | closed
    step: int = 0                   # project step number
    closed_stage: str = ""


@dataclass
class _Step:
    i: int
    n: int
    cmd: str
    layer: Optional[str]
    n_ops: int
    before: dict
    after: dict


class Replay:
    """The journal of a project replayed from an empty canvas (applied steps only)."""

    def __init__(self, p):
        self.p = p
        self.hdir = p.root / "history"
        st = p._stack()
        self.applied = [int(s) for s in st.get("applied", [])]
        self.files: Dict[int, set] = {}
        for s in self.applied:
            try:
                self.files[s] = set(_read_json(self.hdir / str(s) / "meta.json")["files"])
            except Exception:  # noqa: BLE001
                self.files[s] = set()
        journal: Dict[int, dict] = {}
        jp = p.root / "journal.jsonl"
        if jp.exists():
            for ln in jp.read_text(encoding="utf-8").splitlines():
                try:
                    j = json.loads(ln)
                except ValueError:
                    continue
                if "step" in j:
                    journal[int(j["step"])] = j
        states: List[Optional[dict]] = []
        for s in self.applied:
            try:
                states.append(_read_json(self.hdir / str(s) / "before" / "state.json"))
            except Exception:  # noqa: BLE001
                states.append(None)
        states.append(p.state)
        for k in range(len(states) - 2, -1, -1):
            if states[k] is None:
                states[k] = states[k + 1]
        self.states: List[dict] = states            # type: ignore[assignment]
        self.steps: List[_Step] = []
        for i, s in enumerate(self.applied):
            j = journal.get(s, {})
            cmd = str(j.get("cmd") or "").replace("easel ", "", 1)
            self.steps.append(_Step(i, s, cmd, j.get("layer") if j.get("ops") else None, int(j.get("n_ops") or 0),
                                    self.states[i], self.states[i + 1]))
        self.canvas: Optional[np.ndarray] = None
        self._cache: Dict[Path, Optional[np.ndarray]] = {}

    # ------------------------------------------------------------------ layer snapshots
    def layer_after(self, i: int, name: str) -> Optional[np.ndarray]:
        """The layer as it was after applied step index ``i`` (-1 = before the first step)."""
        key = f"layers/{name}.png"
        for j in range(i + 1, len(self.applied)):
            s = self.applied[j]
            if key in self.files.get(s, ()):
                f = self.hdir / str(s) / "before" / key
                return _load_layer_file(f) if f.exists() else None
        return _load_layer_file(self.p.root / key)

    def _load_ops(self, st: _Step):
        f = self.p.root / "ops" / f"{st.n:04d}.json.gz"
        if not f.exists() or st.n_ops <= 0:
            return None
        try:
            from . import ops as ops_mod
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("filter"):
                return None
            return ops_mod.ops_from_jsonable(data["ops"])
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------ compositing
    def _bg(self, state):
        return tuple(v / 255.0 for v in state.get("paper", (255, 255, 255)))

    def _comp(self, layers, state, box=None, skip_sketch: Optional[bool] = None, over=None) -> np.ndarray:
        H, W = self.p.H, self.p.W
        x0, y0, x1, y1 = box if box is not None else (0, 0, W, H)
        use = dict(layers)
        if over:
            use.update(over)
        crop = {k: (None if v is None else v[y0:y1, x0:x1]) for k, v in use.items()}
        hide = (not _sketch_shown(state)) if skip_sketch is None else skip_sketch
        return compose.composite(state["layers"], crop, y1 - y0, x1 - x0, background=self._bg(state),
                                 skip=["sketch"] if hide else [])

    # ------------------------------------------------------------------ plan
    def events(self) -> List[Tuple[int, str, float]]:
        """(step index, kind, weight) of every step that shows something in the film."""
        out = []
        prev = _signature(self.states[0])
        for st in self.steps:
            sig = _signature(st.after)
            if st.layer:
                kind = "ops" if st.n_ops > 0 else "filter"
                out.append((st.i, kind, float(max(1, st.n_ops)) ** 0.8 if st.n_ops else 6.0))
            elif sig != prev:
                out.append((st.i, "state", 4.0))
            prev = sig
        return out

    @staticmethod
    def allocate(events, total_frames: int) -> Dict[int, int]:
        n = len(events)
        if n == 0:
            return {}
        spare = max(0, int(total_frames) - 2 * n)
        wsum = sum(w for _, _, w in events) or 1.0
        return {i: 2 + int(round(spare * w / wsum)) for i, _, w in events}

    # ------------------------------------------------------------------ run
    def run(self, allocs: Optional[Dict[int, int]] = None, want_closed: bool = False) -> Iterator[Frame]:
        """Replay. With ``allocs`` (step index -> frames) the film frames are produced; without, only the
        ``closed`` frames (the picture at the moment each stage was closed, if ``want_closed``)."""
        from . import brush as brush_mod
        from . import ops as ops_mod
        p = self.p
        H, W = p.H, p.W
        film = allocs is not None
        layers: Dict[str, Optional[np.ndarray]] = {e["name"]: self.layer_after(-1, e["name"])
                                                   for e in self.states[0]["layers"]}
        self.canvas = self._comp(layers, self.states[0])
        prev_sig = _signature(self.states[0])
        n_events = len(allocs or {})
        ev_no = 0

        def frame(kind, stage, cmd, step, rgb=None):
            return Frame(compose.to_uint8(self.canvas) if rgb is None else rgb, stage, cmd, ev_no, n_events, kind,
                         step)

        if film:
            yield Frame(compose.to_uint8(self.canvas), self.states[0].get("stage", "plan"), "", 0, n_events, "start")
        for st in self.steps:
            i = st.i
            sig = _signature(st.after)
            is_event = bool(st.layer) or sig != prev_sig
            stage = st.before.get("stage", "")
            alloc = (allocs or {}).get(i, 2)
            if is_event and film:
                ev_no += 1
            if st.layer:
                name = st.layer
                cur = layers.get(name)
                cur_arr = cur if cur is not None else np.zeros((H, W, 4), np.float32)
                known = f"layers/{name}.png" in self.files.get(st.n, ())    # the history holds the finished layer
                truth = self.layer_after(i, name) if known else None
                if known and truth is None:
                    truth_arr = np.zeros((H, W, 4), np.float32)
                else:
                    truth_arr = truth
                entry = next((e for e in st.after["layers"] if e["name"] == name), {})
                lock = bool(entry.get("alpha_lock"))
                meta = st.after
                if not film:
                    if known:
                        layers[name] = truth
                    else:
                        ops_list = self._load_ops(st)
                        if ops_list:
                            work = cur_arr.copy()
                            brush_mod.apply_ops(work, ops_list, alpha_lock=lock)
                            layers[name] = _quantize(work)
                else:
                    ops_list = self._load_ops(st)
                    changed = (np.abs(truth_arr - cur_arr).max(-1) > 0) if known else None
                    if known:
                        box = _bbox_of(changed)
                    else:
                        box = None
                        for op in ops_list or []:
                            box = _union(box, ops_mod.op_bbox(op, (H, W)))
                    emitted = 0
                    if ops_list and box is not None:
                        work = cur_arr.copy()
                        clip = changed.astype(np.float32) if known else None
                        n_prog = max(1, min(alloc, len(ops_list)))
                        if lock:
                            bounds = np.linspace(0, len(ops_list), n_prog + 1).round().astype(int)

                            def chunks():
                                for k in range(n_prog):
                                    brush_mod.apply_ops(work, ops_list[bounds[k]:bounds[k + 1]], clip=clip,
                                                        alpha_lock=True)
                                    yield work.copy()
                            gen = chunks()
                        else:
                            gen = ops_mod.apply_ops_progressive(work, ops_list, n_prog, clip=clip)
                        for k, arr in enumerate(gen, 1):
                            if k == n_prog and known:
                                arr = truth_arr
                            self.canvas[box[1]:box[3], box[0]:box[2]] = self._comp(layers, meta, box,
                                                                                   over={name: arr})
                            emitted += 1
                            yield frame("progress", stage, st.cmd, st.n)
                        final = truth_arr if known else _quantize(work)
                        layers[name] = final if (final[..., 3] > 0).any() else None
                    elif known and box is not None:
                        n_cf = max(2, min(alloc, 5))
                        for k in range(1, n_cf + 1):
                            arr = _lerp_layer(cur_arr, truth_arr, k / n_cf, box)
                            self.canvas[box[1]:box[3], box[0]:box[2]] = self._comp(layers, meta, box,
                                                                                   over={name: arr})
                            emitted += 1
                            yield frame("progress", stage, st.cmd, st.n)
                        layers[name] = truth
                    else:
                        if known:
                            layers[name] = truth
                    if box is not None:
                        self.canvas[box[1]:box[3], box[0]:box[2]] = self._comp(layers, meta, box)
                    if sig != prev_sig:
                        self.canvas = self._comp(layers, meta)
                    yield from self._holds(alloc - emitted, (stage, st.cmd, st.n), ev_no, n_events)
            elif sig != prev_sig and film:
                old = compose.to_uint8(self.canvas).astype(np.float32)
                self.canvas = self._comp(layers, st.after)
                new = compose.to_uint8(self.canvas)
                mid = ((old + new.astype(np.float32)) / 2).astype(np.uint8)
                yield Frame(mid, stage, st.cmd, ev_no, n_events, "progress", st.n)
                yield Frame(new, stage, st.cmd, ev_no, n_events, "progress", st.n)
                yield from self._holds(alloc - 2, (stage, st.cmd, st.n), ev_no, n_events)
            elif sig != prev_sig:
                self.canvas = self._comp(layers, st.after)
            prev_sig = sig
            if want_closed:
                newly = [s for s in st.after.get("stages_done", []) if s not in st.before.get("stages_done", [])]
                for s in newly:
                    rgb = compose.to_uint8(self._comp(layers, st.before))
                    yield Frame(rgb, s, st.cmd, ev_no, n_events, "closed", st.n, closed_stage=s)
        self.layers_final = layers
        if film:
            yield Frame(self.p.composite_u8("export"), self.states[-1].get("stage", ""), "", n_events + 1, n_events,
                        "final")

    def _holds(self, count: int, info, ev: int, n_events: int) -> Iterator[Frame]:
        stage, cmd, step = info
        for _ in range(max(0, count)):
            yield Frame(compose.to_uint8(self.canvas), stage, cmd, ev, n_events, "hold", step)


def _quantize(a: np.ndarray) -> np.ndarray:
    """What a layer file holds: 8-bit values (the project writes layers as PNG after every step)."""
    return (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8).astype(np.float32) / 255.0


def _lerp_layer(a: np.ndarray, b: np.ndarray, t: float, box) -> np.ndarray:
    """Straight-alpha layers a -> b at t, mixed in premultiplied space, only inside ``box``."""
    x0, y0, x1, y1 = box
    out = a.copy()
    pa, pb = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
    a_t = pa[..., 3] * (1 - t) + pb[..., 3] * t
    pm = pa[..., :3] * pa[..., 3:4] * (1 - t) + pb[..., :3] * pb[..., 3:4] * t
    out[y0:y1, x0:x1, 3] = a_t
    out[y0:y1, x0:x1, :3] = np.where(a_t[..., None] > 1e-6, pm / np.maximum(a_t[..., None], 1e-6), 0.0)
    return out


def film_frames(p, seconds: Optional[float] = None) -> Tuple[Iterator[Frame], int, float]:
    """(frame iterator, event count, seconds). About 8-18 s of film depending on how many steps there are."""
    rp = Replay(p)
    events = rp.events()
    secs = float(seconds) if seconds else float(min(18.0, max(8.0, 0.28 * len(events))))
    allocs = rp.allocate(events, int(round(secs * FPS)))
    return rp.run(allocs), len(events), secs


# ============================================================================================ film writers

class _Captioner:
    def __init__(self, text_blob: str, strip_h: int):
        self.strip_h = strip_h
        self.font = None
        size = max(9, int(round(strip_h * 0.52)))
        try:
            from . import fonts as easel_fonts
            self.font = easel_fonts.load_font(None, size, text=text_blob or "easel")
        except Exception:  # noqa: BLE001
            self.font = None
        if self.font is None:
            from PIL import ImageFont
            self.font = ImageFont.load_default()
        self.bold = None
        try:
            from . import fonts as easel_fonts
            self.bold = easel_fonts.load_font(None, size, bold=True, text=text_blob or "easel")
        except Exception:  # noqa: BLE001
            self.bold = self.font

    def strip(self, width: int, frame: Frame, label: str) -> np.ndarray:
        h = self.strip_h
        im = Image.new("RGB", (width, h), (22, 22, 27))
        d = ImageDraw.Draw(im)
        pad = max(6, h // 4)
        stage = frame.stage.upper() if frame.stage else ""
        right = label
        x = pad
        cy = h // 2
        if stage:
            d.text((x, cy), stage, font=self.bold, fill=(255, 196, 96), anchor="lm")
            x += int(self.bold.getlength(stage)) + pad
        rw = int(self.font.getlength(right)) if right else 0
        avail = width - x - pad - (rw + pad if right else 0)
        cmd = frame.cmd
        if self.font.getlength(cmd) > avail:
            while cmd and self.font.getlength(cmd + "...") > avail:
                cmd = cmd[:-1]
            cmd = cmd.rstrip() + "..." if cmd else ""
        if cmd:
            d.text((x, cy), cmd, font=self.font, fill=(232, 232, 236), anchor="lm")
        if right:
            d.text((width - pad, cy), right, font=self.font, fill=(150, 150, 160), anchor="rm")
        return np.asarray(im, dtype=np.uint8)


def _output_size(W: int, H: int, side: int, strip: int, even: bool) -> Tuple[int, int]:
    s = min(1.0, side / float(W), (side - strip) / float(H))
    w, h = max(8, int(round(W * s))), max(8, int(round(H * s)))
    if even:
        w, h = w - (w % 2), h - (h % 2)
    return w, h


def _film_images(frames: Iterator[Frame], W: int, H: int, side: int, even: bool, blob_cmds: List[str]):
    """Yield (RGB uint8 image of canvas + caption strip, Frame)."""
    import cv2
    strip = max(22, int(round(side * 0.046)))
    strip += strip % 2
    w, h = _output_size(W, H, side, strip, even)
    cap = _Captioner("".join(blob_cmds), strip)
    for fr in frames:
        img = fr.rgb if (w, h) == (W, H) else cv2.resize(fr.rgb, (w, h), interpolation=cv2.INTER_AREA)
        if fr.kind == "start":
            label = "start"
        elif fr.kind == "final":
            label = "final"
        else:
            label = f"{fr.index}/{fr.total}"
        out = np.concatenate([img, cap.strip(w, fr, label)], axis=0)
        yield out, fr


def write_gif(p, path: Path, seconds: Optional[float] = None) -> dict:
    frames, n_events, secs = film_frames(p, seconds)
    cmds = [s.cmd for s in Replay(p).steps]
    items: List[Image.Image] = []
    durs: List[int] = []
    prev = None
    n_frames = 0
    per = int(round(1000.0 / FPS / 10.0)) * 10
    for img, fr in _film_images(frames, p.W, p.H, GIF_SIDE, False, cmds):
        n_frames += 1
        dur = per
        if fr.kind == "final":
            dur = int(HOLD_SECONDS * 1000)
        if prev is not None and np.array_equal(img, prev):
            durs[-1] += dur
            continue
        prev = img
        items.append(Image.fromarray(img, "RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                                                           dither=Image.Dither.NONE))
        durs.append(dur)
    items[0].save(path, save_all=True, append_images=items[1:], duration=durs, loop=0, optimize=False, disposal=1)
    size = items[0].size
    return {"frames": n_frames, "distinct": len(items), "seconds": sum(durs) / 1000.0, "size": size,
            "events": n_events}


def write_mp4(p, path: Path, seconds: Optional[float] = None) -> dict:
    ff = find_ffmpeg()
    if ff is None:
        raise UsageError("an .mp4 time-lapse needs ffmpeg and it was not found (not on PATH, not in C:/ffmpeg/bin)",
                         "easel export final.png --timelapse film.gif")
    frames, n_events, secs = film_frames(p, seconds)
    cmds = [s.cmd for s in Replay(p).steps]
    tmp = Path(tempfile.mkdtemp(prefix="easel_film_"))
    n = 0
    size = (0, 0)
    try:
        for img, fr in _film_images(frames, p.W, p.H, MP4_SIDE, True, cmds):
            Image.fromarray(img, "RGB").save(tmp / f"f{n:05d}.png", compress_level=1)
            size = (img.shape[1], img.shape[0])
            n += 1
        cmd = [ff, "-nostdin", "-y", "-loglevel", "error", "-framerate", f"{FPS:g}", "-i", str(tmp / "f%05d.png"),
               "-vf", f"tpad=stop_mode=clone:stop_duration={HOLD_SECONDS:g},format=yuv420p",
               "-c:v", "libx264", "-crf", "19", "-preset", "medium", "-movflags", "+faststart", "-r", "25",
               str(path)]
        try:
            r = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, timeout=420)
        except subprocess.TimeoutExpired:
            raise UsageError("ffmpeg took longer than 7 minutes; use a .gif instead",
                             "easel export final.png --timelapse film.gif")
        if r.returncode != 0 or not path.exists():
            msg = (r.stderr or b"").decode("utf-8", "replace").strip().splitlines()
            raise UsageError("ffmpeg failed: " + (msg[-1][:160] if msg else f"exit code {r.returncode}"),
                             "easel export final.png --timelapse film.gif")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"frames": n, "seconds": n / FPS + HOLD_SECONDS, "size": size, "events": n_events}


# ============================================================================================ stages

def stage_pictures(p) -> List[Tuple[str, np.ndarray]]:
    """(stage name, picture when the stage was closed) in the order the stages were first closed; a stage that was
    reopened and closed again keeps its latest picture. An open current stage is added last."""
    rp = Replay(p)
    order: List[str] = []
    pics: Dict[str, np.ndarray] = {}
    for fr in rp.run(None, want_closed=True):
        if fr.kind == "closed":
            if fr.closed_stage not in pics:
                order.append(fr.closed_stage)
            pics[fr.closed_stage] = fr.rgb
    out = [(s, pics[s]) for s in order]
    if p.state.get("regions"):         # the plan's work is the regions: show them, not a blank canvas
        from . import view as _view
        regs = [(nm, p.region_mask(nm), 1) for nm in p.state["regions"]]
        # on blank paper, not over the reference: the stage pictures are the painter's own work
        base = np.full((p.H, p.W, 3), 255, np.uint8)
        out = [(s, _view.region_overlay(base, regs) if s == "plan" else img) for s, img in out]
    cur = p.state.get("stage")
    if cur and cur not in pics and not p.state.get("finished"):
        out.append((f"{cur} (open)", p.composite_u8("view")))
    if p.state.get("finished") and out:
        out[-1] = (out[-1][0], p.composite_u8("export"))
    return out


def write_stages(p, folder: Path) -> Tuple[List[Path], Path]:
    import cv2
    folder.mkdir(parents=True, exist_ok=True)
    pics = stage_pictures(p)
    if not pics:
        pics = [("start", p.composite_u8("view"))]
    files = []
    if hasattr(p, "to_deliver"):
        pics = [(nm, p.to_deliver(rgb)) for nm, rgb in pics]
    for k, (name, rgb) in enumerate(pics, 1):
        f = folder / f"{k:02d}-{re.sub(r'[^A-Za-z0-9]+', '-', name).strip('-').lower()}.png"
        Image.fromarray(rgb).save(f)
        files.append(f)
    # contact sheet
    n = len(pics)
    cols = n if n <= 4 else (4 if n != 5 and n != 6 else 3)
    rows = (n + cols - 1) // cols
    H, W = pics[0][1].shape[:2]
    cw = 360 if W <= H else 440
    ch = int(round(cw * H / W))
    lab = 34
    gap = 14
    sheet = np.full((rows * (ch + lab + gap) + gap, cols * (cw + gap) + gap, 3), (24, 24, 28), np.uint8)
    cap = _Captioner("".join(nm for nm, _ in pics), lab)
    for k, (name, rgb) in enumerate(pics):
        r, c = divmod(k, cols)
        x = gap + c * (cw + gap)
        y = gap + r * (ch + lab + gap)
        sheet[y:y + ch, x:x + cw] = cv2.resize(rgb, (cw, ch), interpolation=cv2.INTER_AREA)
        strip = cap.strip(cw, Frame(rgb, name, "", 0, 0), f"{k + 1}/{n}")
        sheet[y + ch:y + ch + lab, x:x + cw] = strip
    sheet_path = folder / "stages.jpg"
    Image.fromarray(sheet).save(sheet_path, quality=90)
    return files, sheet_path


# ============================================================================================ the command

def cmd_export(ctx, args: List[str]):
    from . import cli, metrics, workflow
    from .cli import Report, Spec, parse
    sp = Spec(cli.USAGE["export"], ["file"],
              {"psd": "flag", "ora": "flag", "timelapse": "value", "stages": "value"},
              alias={"film": "timelapse", "time-lapse": "timelapse", "gif": "timelapse", "video": "timelapse",
                     "layers": "psd", "stage": "stages"}, min_pos=1)
    a = parse(args, sp)
    p = ctx.project
    f = Path(str(a["file"]))
    want_psd, want_ora = bool(a.get("psd")), bool(a.get("ora"))
    film = a.get("timelapse")
    ext = f.suffix.lower()
    if ext == ".psd":
        want_psd, f = True, f.with_suffix(".png")
    elif ext == ".ora":
        want_ora, f = True, f.with_suffix(".png")
    elif ext in (".gif", ".mp4"):
        film, f = film or str(f), f.with_suffix(".png")
    elif ext not in (".png", ""):
        raise UsageError(f"the picture is written as PNG; got {f.name}. Layered files, films and stage pictures "
                         "come from options", "easel export final.png --psd --ora --timelapse film.gif")
    if f.suffix == "":
        f = f.with_suffix(".png")
    if not f.is_absolute():
        f = Path(os.getcwd()) / f
    film_path = None
    if film:
        film_path = Path(str(film))
        if film_path.suffix.lower() not in (".gif", ".mp4"):
            raise UsageError(f"--timelapse writes .gif or .mp4; got {film_path.name}",
                             "easel export final.png --timelapse film.gif")
        if not film_path.is_absolute():
            film_path = Path(os.getcwd()) / film_path
        if film_path.suffix.lower() == ".mp4" and find_ffmpeg() is None:
            raise UsageError("an .mp4 time-lapse needs ffmpeg and it was not found (not on PATH, not in "
                             "C:/ffmpeg/bin); a .gif needs nothing", "easel export final.png --timelapse film.gif")
    stages_dir = None
    if a.get("stages"):
        stages_dir = Path(str(a["stages"]))
        if not stages_dir.is_absolute():
            stages_dir = Path(os.getcwd()) / stages_dir
    f.parent.mkdir(parents=True, exist_ok=True)
    img = p.composite_u8("export")
    rep = Report(mutated=True)
    notes: List[str] = []
    from .project import save_png
    out_img = cli.delivered(p, img) if hasattr(p, "to_deliver") else img
    save_png(f, out_img)
    if out_img.shape[:2] != img.shape[:2]:
        size_words = f"{out_img.shape[1]}x{out_img.shape[0]} (drawn at {p.W}x{p.H}, reduced with an area filter" + \
            (", then the finishing sharpen" if p.state.get("deliver_filter") == "sharpen" else "") + ")"
    else:
        size_words = f"{p.W}x{p.H}"
    rep_lines = [f"exported {p.rel(f)}  {size_words}  (sketch layer left out)"]
    if want_psd:
        pf = f.with_suffix(".psd")
        n = write_psd(p, pf)
        rep_lines.append(f"layers: {p.rel(pf)}  {n} layers with blend modes, opacity, visibility")
    if want_ora:
        of = f.with_suffix(".ora")
        n = write_ora(p, of)
        rep_lines.append(f"layers: {p.rel(of)}  {n} layers (OpenRaster: Krita, GIMP)")
    if film_path is not None:
        film_path.parent.mkdir(parents=True, exist_ok=True)
        info = write_mp4(p, film_path) if film_path.suffix.lower() == ".mp4" else write_gif(p, film_path)
        mb = film_path.stat().st_size / 1e6
        rep_lines.append(f"time-lapse: {p.rel(film_path)}  {info['size'][0]}x{info['size'][1]}  "
                         f"{info['events']} steps, {info['seconds']:.0f} s, {mb:.1f} MB  (undone steps left out)")
    if stages_dir is not None:
        files, sheet = write_stages(p, stages_dir)
        rep_lines.append(f"stages: {len(files)} pictures in {p.rel(stages_dir)}/ and the contact sheet "
                         f"{p.rel(sheet)}")
    step = p.begin_step([])
    p.state["exported"] = str(f)
    workflow.tick(p.state, "finish", None, key="export")
    p.end_step(step, {"cmd": ctx.cmdline, "file": str(f)})
    rep.add(f"step {step}  " + rep_lines[0])
    for ln in rep_lines[1:]:
        rep.add(ln)
    if p.ref is not None:
        s = cli._comparison(p, img).scores()
        rep.add("whole picture:  " + metrics.fmt_scores(s))
    rep.add(cli.fingerprint_line(p))
    rep.next = cli._finish_next(ctx)
    return rep
