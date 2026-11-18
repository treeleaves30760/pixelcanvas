"""The project folder (DESIGN.md section 4): state, reference, layers, regions, history / undo / redo, journal, ops.

```
<name>.easel/
  state.json  ref.png  layers/<id>.png  regions/<name>.png  history/<step>/  journal.jsonl  ops/<step>.json.gz  views/
```
Undo works by file snapshots: before step N changes anything, the previous bytes of every file it will touch are
copied to ``history/N/before/``; undo copies the current bytes to ``history/N/after/`` and puts the old bytes back,
so undo / redo restore files byte for byte. The undo / redo stacks live in ``history/stack.json``.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from PIL import Image

from . import compose
from .coords import UsageError

MAX_SIDE = 1600
SUFFIX = ".easel"

# name, blend mode, opacity, clip, the stage that paints on it
LAYER_STACK = [
    ("paper", "normal", 1.0, False),
    ("flat", "normal", 1.0, False),
    ("shade", "normal", 1.0, False),
    ("paint", "normal", 1.0, False),
    ("line", "normal", 1.0, False),
    ("detail", "normal", 1.0, False),
    ("fx", "normal", 1.0, False),
    ("text", "normal", 1.0, False),
    ("sketch", "normal", 0.6, False),
]
LAYER_NAMES = [n for n, *_ in LAYER_STACK]


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=1, ensure_ascii=False, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_alpha(im: "Image.Image") -> "Image.Image":
    """The one rule for a reference with transparency (shell, pipeline, judge): composited over white paper. An
    image without alpha comes back unchanged; the colour profile is kept for the sRGB conversion."""
    has_alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
    if not has_alpha:
        return im
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im.convert("RGBA"))
    flat = bg.convert("RGB")
    if im.info.get("icc_profile"):
        flat.info["icc_profile"] = im.info["icc_profile"]
    return flat


def load_image_rgb(path: Path) -> np.ndarray:
    """Any image file -> uint8 RGB (alpha composited over white: flatten_alpha)."""
    with Image.open(path) as im:
        im.load()
        im = flatten_alpha(im)
        if im.info.get("icc_profile"):
            # embedded profiles (Adobe RGB, Display P3): convert to sRGB instead of copying raw values
            try:
                from .pipeline import to_srgb
            except Exception:  # noqa: BLE001
                to_srgb = _to_srgb
            return np.asarray(to_srgb(im), dtype=np.uint8).copy()
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def _to_srgb(im):
    import io as _io
    from PIL import ImageCms
    try:
        src = ImageCms.ImageCmsProfile(_io.BytesIO(im.info["icc_profile"]))
        return ImageCms.profileToProfile(im.convert("RGB"), src, ImageCms.createProfile("sRGB"),
                                         renderingIntent=0, outputMode="RGB")
    except Exception:  # noqa: BLE001
        return im.convert("RGB")


def fit_size(w: int, h: int, max_side: int = MAX_SIDE):
    s = min(1.0, max_side / float(max(w, h)))
    return max(1, int(round(w * s))), max(1, int(round(h * s)))


def save_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.png")
    Image.fromarray(arr).save(tmp, compress_level=1)
    os.replace(tmp, path)


def paper_colour(ref: Optional[np.ndarray]):
    """The reference's border colour when the border is a calm light-ish colour, else white."""
    if ref is None:
        return (255, 255, 255)
    b = max(2, min(ref.shape[:2]) // 100)
    border = np.concatenate([ref[:b].reshape(-1, 3), ref[-b:].reshape(-1, 3),
                             ref[:, :b].reshape(-1, 3), ref[:, -b:].reshape(-1, 3)]).astype(np.float32)
    med = np.median(border, axis=0)
    if np.mean(np.abs(border - med) < 12) > 0.85:
        return tuple(int(v) for v in med)
    return (255, 255, 255)


class Project:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.state: dict = _read_json(self.root / "state.json")
        self._ref: Optional[np.ndarray] = None
        self._layers: Dict[str, Optional[np.ndarray]] = {}
        self._regions: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ create / find
    @classmethod
    def create(cls, root, ref_path=None, size=None, request: str = "", notes: Optional[list] = None) -> "Project":
        root = Path(root)
        if not root.name.endswith(SUFFIX):
            root = root.with_name(root.name + SUFFIX)
        if (root / "state.json").exists():
            raise UsageError(f"a project already exists at {root}; continue it with  easel status  "
                             "or choose a new folder", f"easel new {root.with_name('picture2')} --ref <image>")
        ref = None
        if ref_path is not None:
            rp = Path(ref_path)
            if not rp.exists():
                raise UsageError(f"reference image not found: {rp}", f"easel new {root} --ref path/to/picture.png")
            try:
                ref = load_image_rgb(rp)
            except Exception as e:  # noqa: BLE001
                raise UsageError(f"cannot read the reference image {rp}: {e}", f"easel new {root} --ref picture.png")
            h, w = ref.shape[:2]
            W, H = fit_size(w, h)
            if (W, H) != (w, h):
                ref = np.asarray(Image.fromarray(ref).resize((W, H), Image.LANCZOS), dtype=np.uint8)
                if notes is not None:
                    notes.append(f"reference {w}x{h} reduced to {W}x{H} (long side at most {MAX_SIDE})")
            if size and notes is not None:
                notes.append("--size ignored: the canvas takes the reference's size")
        else:
            W, H = size or (1024, 1024)
            W, H = fit_size(W, H)
        root.mkdir(parents=True, exist_ok=True)
        for sub in ("layers", "regions", "history", "ops", "views", "cache"):
            (root / sub).mkdir(exist_ok=True)
        if ref is not None:
            save_png(root / "ref.png", ref)
        paper = paper_colour(ref)
        layers = []
        for name, mode, opacity, clip in LAYER_STACK:
            layers.append({"name": name, "mode": mode, "opacity": opacity, "visible": True, "clip": clip,
                           "alpha_lock": False, "empty": name != "paper"})
        state = {
            "version": 1, "width": W, "height": H,
            "reference": str(Path(ref_path).resolve()) if ref_path else None,
            "has_ref": ref is not None, "request": request or "", "paper": list(paper),
            "style": None, "style_guess": None, "stage": "plan", "stages_done": [],
            "layers": layers, "regions": {}, "marks": None, "todo": [], "todo_seq": 0,
            "step": 0, "attempts": {}, "exported": None, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _write_json(root / "state.json", state)
        p = cls(root)
        paper_arr = np.zeros((H, W, 4), np.float32)
        paper_arr[..., :3] = np.asarray(paper, np.float32) / 255.0
        paper_arr[..., 3] = 1.0
        p._write_layer_file("paper", paper_arr)
        for name in LAYER_NAMES[1:]:
            p._write_layer_file(name, None)
        _write_json(root / "history" / "stack.json", {"applied": [], "undone": [], "counter": 0})
        (root / "journal.jsonl").touch()
        return p

    def set_work_scale(self, s: int) -> None:
        """Make a fresh project's canvas the working canvas: s x the delivered picture (``deliver``), the reference
        enlarged to it; marks are laid at working size and the export reduces to ``deliver`` with an area filter.
        Part of the setup (no undo step); only before anything is painted."""
        import cv2
        s = max(1, int(s))
        if int(self.state.get("work_scale", 1) or 1) > 1 and self.state.get("deliver"):
            # back to the delivered size first (a fresh project only: nothing painted yet)
            dw, dh = (int(v) for v in self.state["deliver"])
            self.state["width"], self.state["height"] = dw, dh
            for e in self.state["layers"]:
                arr = self.layer(e["name"], create=False) if e["name"] == "paper" else None
                if arr is not None:
                    arr = cv2.resize(arr, (dw, dh), interpolation=cv2.INTER_NEAREST)
                self._write_layer_file(e["name"], arr)
            f = self.root / "ref_deliver.png"
            if f.exists():
                os.replace(f, self.root / "ref.png")
            self._ref = None
            self._ref_d = None
            self.state.pop("deliver", None)
        W, H = self.W, self.H
        self.state["work_scale"] = s
        if s == 1:
            self.save_state()
            return
        ref = self.ref                  # read at the delivered size, before the canvas grows
        self.state["deliver"] = [W, H]
        self.state["width"], self.state["height"] = W * s, H * s
        for e in self.state["layers"]:
            arr = self.layer(e["name"], create=False) if e["name"] == "paper" else None
            if arr is not None:
                arr = cv2.resize(arr, (W * s, H * s), interpolation=cv2.INTER_NEAREST)
            self._write_layer_file(e["name"], arr)
        if ref is not None:
            # the delivered-size reference is kept: scores, views of the error and the coarse passes use it
            save_png(self.root / "ref_deliver.png", ref)
            # the engine's own enlargement (run_reference_pipeline: cubic)
            big = cv2.resize(ref, (W * s, H * s), interpolation=cv2.INTER_CUBIC)
            save_png(self.root / "ref.png", np.clip(big, 0, 255).astype(np.uint8))
            self._ref = None
        self.save_state()

    @property
    def ref_deliver(self) -> Optional[np.ndarray]:
        """The reference at the delivered size (the reference itself on a 1x canvas)."""
        if getattr(self, "_ref_d", None) is None:
            if int(self.state.get("work_scale", 1) or 1) <= 1 or not self.state.get("has_ref"):
                return self.ref
            f = self.root / "ref_deliver.png"
            if not f.exists():
                self._rebuild_ref_files()
            self._ref_d = self._check_ref_size(load_image_rgb(f), self.deliver_size(), "ref_deliver.png")
        return self._ref_d

    def _source_ref_at(self, W: int, H: int) -> Optional[np.ndarray]:
        """The original reference file named in the state, at W x H, when it is still there and fits that size."""
        src = self.state.get("reference")
        if not src or not Path(src).exists():
            return None
        try:
            img = load_image_rgb(Path(src))
        except Exception:  # noqa: BLE001
            return None
        h, w = img.shape[:2]
        if fit_size(w, h) != (W, H):
            return None
        if (w, h) != (W, H):
            img = np.asarray(Image.fromarray(img).resize((W, H), Image.LANCZOS), dtype=np.uint8)
        return img

    def _rebuild_ref_files(self) -> None:
        """Put back ref.png / ref_deliver.png when a copied or moved folder lost one of them: from the other file,
        else from the original reference named in the state. Raises a UsageError when neither is left."""
        import cv2
        s = self.scale
        f_ref, f_d = self.root / "ref.png", self.root / "ref_deliver.png"
        dw, dh = self.deliver_size()
        if s <= 1:
            if not f_ref.exists():
                img = self._source_ref_at(self.W, self.H)
                if img is None:
                    raise self._ref_missing()
                save_png(f_ref, img)
            return
        if not f_d.exists():
            img = self._source_ref_at(dw, dh)
            if img is None and f_ref.exists():
                img = self.to_deliver(load_image_rgb(f_ref))
            if img is None:
                raise self._ref_missing()
            save_png(f_d, img)
        if not f_ref.exists():
            big = cv2.resize(load_image_rgb(f_d), (self.W, self.H), interpolation=cv2.INTER_CUBIC)
            save_png(f_ref, np.clip(big, 0, 255).astype(np.uint8))

    def _ref_missing(self) -> UsageError:
        where = self.rel(self.root / ("ref_deliver.png" if self.scale > 1 else "ref.png"))
        return UsageError(f"the project's reference picture is missing ({where}, and the original "
                          f"{self.state.get('reference')} is not there either); copy the whole project folder, "
                          f"or put the reference back as {where}",
                          f"cp reference.png {where}")

    def _check_ref_size(self, img: np.ndarray, size, name: str) -> np.ndarray:
        if (img.shape[1], img.shape[0]) != tuple(size):
            raise UsageError(f"{self.rel(self.root / name)} is {img.shape[1]}x{img.shape[0]} but this project needs "
                             f"{size[0]}x{size[1]}; copy the whole project folder, or delete that file and put the "
                             f"reference back as {self.rel(self.root / ('ref_deliver.png' if self.scale > 1 else 'ref.png'))}",
                             f"cp reference.png {self.rel(self.root / ('ref_deliver.png' if self.scale > 1 else 'ref.png'))}")
        return img

    @property
    def scale(self) -> int:
        return max(1, int(self.state.get("work_scale", 1) or 1))

    def mask_to_deliver(self, mask) -> Optional[np.ndarray]:
        import cv2
        if mask is None or self.scale == 1:
            return mask
        dw, dh = self.deliver_size()
        m = np.asarray(mask)
        return cv2.resize(m.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST) > 0

    def deliver_size(self):
        dv = self.state.get("deliver")
        if dv:
            return int(dv[0]), int(dv[1])
        s = max(1, int(self.state.get("work_scale", 1) or 1))
        return int(round(self.W / s)), int(round(self.H / s))

    def to_deliver(self, img: np.ndarray) -> np.ndarray:
        """A working-size picture reduced to the delivered size (area filter); unchanged at scale 1."""
        import cv2
        dw, dh = self.deliver_size()
        if img.shape[1] == dw and img.shape[0] == dh:
            return img
        return cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA)

    @classmethod
    def find(cls, explicit=None, cwd=None, notes: Optional[list] = None) -> "Project":
        cands = []
        if explicit:
            e = Path(explicit)
            for c in (e, e.with_name(e.name + SUFFIX)):
                if (c / "state.json").exists():
                    return cls(c)
            raise UsageError(f"no easel project at {explicit}", "easel status --project my_picture.easel")
        env = os.environ.get("EASEL_PROJECT")
        if env and (Path(env) / "state.json").exists():
            return cls(Path(env))
        here = Path(cwd or os.getcwd())
        for d in (here, *here.parents):
            if (d / "state.json").exists() and d.name.endswith(SUFFIX):
                return cls(d)
        cands = sorted((d for d in here.glob("*" + SUFFIX) if (d / "state.json").exists()),
                       key=lambda d: (d / "state.json").stat().st_mtime, reverse=True)
        if not cands:
            raise UsageError("no easel project here (no *.easel folder in the current directory)",
                             "easel new my_picture --ref reference.png")
        if len(cands) > 1 and notes is not None:
            notes.append(f"{len(cands)} projects here; using the newest, {cands[0].name} "
                         f"(choose with --project <folder>)")
        return cls(cands[0])

    # ------------------------------------------------------------------ basics
    @property
    def W(self) -> int:
        return int(self.state["width"])

    @property
    def H(self) -> int:
        return int(self.state["height"])

    @property
    def name(self) -> str:
        return self.root.name

    def save_state(self) -> None:
        _write_json(self.root / "state.json", self.state)

    def reload(self) -> None:
        self.state = _read_json(self.root / "state.json")
        self._layers.clear()
        self._regions.clear()
        self._ref = None            # undo / redo may restore another target (easel recolor ...)
        self._ref_d = None

    def rel(self, path: Path) -> str:
        """Path as printed in reports: relative to the cwd when possible, forward slashes."""
        try:
            return Path(os.path.relpath(path)).as_posix()
        except ValueError:
            return Path(path).as_posix()

    @property
    def ref(self) -> Optional[np.ndarray]:
        if self._ref is None and self.state.get("has_ref"):
            if not (self.root / "ref.png").exists():
                self._rebuild_ref_files()
            self._ref = self._check_ref_size(
                np.asarray(Image.open(self.root / "ref.png").convert("RGB"), dtype=np.uint8), (self.W, self.H), "ref.png")
        return self._ref

    def require_ref(self, what: str) -> np.ndarray:
        if self.ref is None:
            raise UsageError(f"{what} needs a reference picture and this project has none",
                             "easel new my_picture --ref reference.png")
        return self.ref

    # ------------------------------------------------------------------ layers
    def layer_entry(self, name: str) -> dict:
        n = str(name).strip().lower()
        for e in self.state["layers"]:
            if e["name"] == n:
                return e
        import difflib
        close = difflib.get_close_matches(n, LAYER_NAMES, n=1)
        hint = f" (did you mean {close[0]}?)" if close else ""
        raise UsageError(f"unknown layer {name!r}{hint}; layers: {', '.join(e['name'] for e in self.state['layers'])}",
                         f"easel look --show layers   or   --layer {close[0] if close else 'line'}")

    def layer(self, name: str, create: bool = True) -> Optional[np.ndarray]:
        """Layer as float32 [H,W,4] (a writable copy is NOT made; copy before painting if needed)."""
        e = self.layer_entry(name)
        if name not in self._layers:
            if e.get("empty"):
                self._layers[name] = None
            else:
                im = Image.open(self.root / "layers" / f"{name}.png")
                self._layers[name] = np.asarray(im.convert("RGBA"), dtype=np.float32) / 255.0
        arr = self._layers[name]
        if arr is None and create:
            return np.zeros((self.H, self.W, 4), np.float32)
        return arr

    def layers_dict(self) -> Dict[str, Optional[np.ndarray]]:
        return {e["name"]: self.layer(e["name"], create=False) for e in self.state["layers"]}

    def _write_layer_file(self, name: str, arr: Optional[np.ndarray]) -> None:
        path = self.root / "layers" / f"{name}.png"
        if arr is None:
            arr8 = np.zeros((self.H, self.W, 4), np.uint8)
        else:
            arr8 = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        save_png(path, arr8)
        e = self.layer_entry(name)
        e["empty"] = bool(arr is None or not (arr8[..., 3] > 0).any())
        # keep the in-memory copy identical to what a fresh load would give
        self._layers[name] = None if e["empty"] else arr8.astype(np.float32) / 255.0

    def save_layer(self, name: str, arr: Optional[np.ndarray]) -> None:
        self._write_layer_file(name, arr)
        self.save_state()

    # ------------------------------------------------------------------ compositing
    def background(self):
        return tuple(v / 255.0 for v in self.state.get("paper", (255, 255, 255)))

    def composite(self, mode: str = "view", upto: Optional[str] = None) -> np.ndarray:
        """float32 RGB. mode: 'view' (what the model sees: sketch shown while its stage is on),
        'score' / 'export' (no sketch layer)."""
        skip = []
        if mode in ("score", "export") or not self.sketch_shown():
            skip.append("sketch")
        return compose.composite(self.state["layers"], self.layers_dict(), self.H, self.W,
                                 background=self.background(), skip=skip, upto=upto)

    def composite_u8(self, mode: str = "view", upto: Optional[str] = None) -> np.ndarray:
        return compose.to_uint8(self.composite(mode, upto))

    def sketch_shown(self) -> bool:
        e = self.layer_entry("sketch")
        if not e.get("visible", True):
            return False
        if e.get("force_visible"):
            return True
        # without a reference the line stage inks over the construction: keep it in view while inking
        if self.state.get("stage") == "line" and self.state.get("draw") and (getattr(self, "ref", None) is None or self.state.get("mode") == "hand"):
            return True
        return self.state.get("stage") in ("plan", "sketch")

    # ------------------------------------------------------------------ regions
    def region_names(self) -> List[str]:
        return list(self.state["regions"].keys())

    def region_mask(self, name: str) -> np.ndarray:
        if name not in self._regions:
            im = Image.open(self.root / "regions" / f"{name}.png")
            self._regions[name] = np.asarray(im.convert("L"), dtype=np.uint8) > 127
        return self._regions[name]

    def save_region_mask(self, name: str, mask: np.ndarray) -> None:
        save_png(self.root / "regions" / f"{name}.png", (np.asarray(mask) > 0).astype(np.uint8) * 255)
        self._regions[name] = np.asarray(mask) > 0

    def delete_region_file(self, name: str) -> None:
        p = self.root / "regions" / f"{name}.png"
        if p.exists():
            p.unlink()
        self._regions.pop(name, None)

    # ------------------------------------------------------------------ history
    def _stack(self) -> dict:
        p = self.root / "history" / "stack.json"
        if p.exists():
            return _read_json(p)
        return {"applied": [], "undone": [], "counter": int(self.state.get("step", 0))}

    def _save_stack(self, st: dict) -> None:
        _write_json(self.root / "history" / "stack.json", st)

    def begin_step(self, files: Iterable[str]) -> int:
        """Snapshot ``files`` (paths relative to the project; state.json always included) before a change.
        Returns the new step number; the caller then changes files and calls ``end_step``."""
        st = self._stack()
        step = int(st["counter"]) + 1
        hdir = self.root / "history" / str(step)
        if hdir.exists():
            shutil.rmtree(hdir)
        (hdir / "before").mkdir(parents=True)
        files = list(dict.fromkeys(["state.json", *files]))
        absent = []
        for rel in files:
            src = self.root / rel
            if src.exists():
                dst = hdir / "before" / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
            else:
                absent.append(rel)
        _write_json(hdir / "meta.json", {"files": files, "absent_before": absent})
        # a new step discards the redo stack
        for s in st["undone"]:
            shutil.rmtree(self.root / "history" / str(s), ignore_errors=True)
        st["undone"] = []
        st["counter"] = step
        self._save_stack(st)
        self.state["step"] = step
        return step

    def end_step(self, step: int, entry: dict) -> None:
        st = self._stack()
        st["applied"].append(step)
        self._save_stack(st)
        self.save_state()
        self.journal({"step": step, **entry})

    def abort_step(self, step: int) -> None:
        """Restore the snapshot of a step that failed half-way."""
        self._restore(step, "before", save_current_to=None)
        st = self._stack()
        st["counter"] = step - 1
        self._save_stack(st)
        shutil.rmtree(self.root / "history" / str(step), ignore_errors=True)
        self.reload()

    def _restore(self, step: int, which: str, save_current_to: Optional[str]) -> None:
        hdir = self.root / "history" / str(step)
        meta = _read_json(hdir / "meta.json")
        absent = set(meta["absent_before"] if which == "before" else meta.get("absent_after", []))
        if save_current_to:
            other_absent = []
            for rel in meta["files"]:
                src = self.root / rel
                if src.exists():
                    dst = hdir / save_current_to / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
                else:
                    other_absent.append(rel)
            key = "absent_after" if save_current_to == "after" else "absent_before"
            meta[key] = other_absent
            _write_json(hdir / "meta.json", meta)
        for rel in meta["files"]:
            dst = self.root / rel
            if rel in absent:
                if dst.exists():
                    dst.unlink()
            else:
                src = hdir / which / rel
                if src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)

    def undo(self, n: int = 1) -> List[int]:
        st = self._stack()
        done = []
        for _ in range(n):
            if not st["applied"]:
                break
            step = st["applied"].pop()
            self._restore(step, "before", save_current_to="after")
            st["undone"].append(step)
            done.append(step)
        self._save_stack(st)
        self.reload()
        if done:
            self.journal({"undo": done})
        return done

    def redo(self, n: int = 1) -> List[int]:
        st = self._stack()
        done = []
        for _ in range(n):
            if not st["undone"]:
                break
            step = st["undone"].pop()
            self._restore(step, "after", save_current_to=None)
            st["applied"].append(step)
            done.append(step)
        self._save_stack(st)
        self.reload()
        if done:
            self.journal({"redo": done})
        return done

    def can_undo(self) -> int:
        return len(self._stack()["applied"])

    def can_redo(self) -> int:
        return len(self._stack()["undone"])

    # ------------------------------------------------------------------ journal / ops / views
    def journal(self, entry: dict) -> None:
        entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **entry}
        with open(self.root / "journal.jsonl", "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def journal_lines(self, n: int = 5) -> List[dict]:
        p = self.root / "journal.jsonl"
        if not p.exists():
            return []
        lines = p.read_text(encoding="utf-8").splitlines()
        out = []
        for ln in lines[-n:]:
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
        return out

    def save_ops(self, step: int, data) -> str:
        path = self.root / "ops" / f"{step:04d}.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))
        return f"ops/{step:04d}.json.gz"

    def view_path(self, tag: str) -> Path:
        return self.root / "views" / f"{tag}.png"

    def save_view(self, img: np.ndarray, tag: str) -> Path:
        path = self.view_path(tag)
        save_png(path, img)
        shutil.copyfile(path, self.root / "views" / "last.png")
        return path
