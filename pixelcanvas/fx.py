"""``pixelcanvas fx``: finishing effects an illustrator applies at the end (glow, grain, blur, vignette, paper, sharpen,
levels).

An effect is computed from the canvas itself (the composite of the layers under ``text``), never from the reference:
the reference is only used to *measure* which amount helps (DESIGN.md 10.1: no reference pixels are copied). The
result is stored on the ``fx`` layer as an ordinary RGBA layer (straight alpha) that reproduces the effect when
composited over the layers below it, so it is undoable, shows in layered exports and replays in the time-lapse.
Effects are deterministic: the same command on the same picture gives the same pixels (grain is seeded).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import coords, compose
from .coords import UsageError

EPS = 0.5 / 255.0

# name -> (default amount, one-line description)
EFFECTS: Dict[str, Tuple[float, str]] = {
    "glow": (0.5, "soft bloom around the bright areas"),
    "grain": (0.4, "fine film / paper grain"),
    "blur": (0.4, "soften (use a region for depth of field)"),
    "vignette": (0.5, "darker corners that pull the eye to the middle"),
    "paper": (0.5, "paper tooth and a warm tone"),
    "sharpen": (0.5, "crisper edges"),
    "levels": (0.6, "stretch the tones so blacks are black and whites are white"),
}
ALIASES = {"bloom": "glow", "halation": "glow", "noise": "grain", "film": "grain", "soften": "blur",
           "soft": "blur", "vignetting": "vignette", "vig": "vignette", "texture": "paper", "tooth": "paper",
           "sharp": "sharpen", "unsharp": "sharpen", "contrast": "levels", "tone": "levels", "tones": "levels",
           "level": "levels"}
CANDIDATES = (0.2, 0.4, 0.6, 0.8, 1.0)


def canonical(name: str) -> str:
    n = str(name).strip().lower()
    n = ALIASES.get(n, n)
    if n not in EFFECTS:
        import difflib
        close = difflib.get_close_matches(n, list(EFFECTS) + list(ALIASES), n=1, cutoff=0.5)
        sug = ALIASES.get(close[0], close[0]) if close else "grain"
        raise UsageError(f"unknown effect {name!r}" + (f" (did you mean {sug}?)" if close else "")
                         + f"; effects: {', '.join(EFFECTS)}", f"pixelcanvas fx {sug} --amount 0.4")
    return n


# ----------------------------------------------------------------------------------------------- the effects

def _lum(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(a, (0, 0), max(0.3, float(sigma)), borderType=cv2.BORDER_REFLECT)


def _noise(rng, H: int, W: int, cell: float) -> np.ndarray:
    """Gaussian noise whose features are about ``cell`` px, unit variance, float32 [H,W]."""
    if cell <= 1.01:
        n = rng.standard_normal((H, W)).astype(np.float32)
    else:
        h, w = max(2, int(np.ceil(H / cell))), max(2, int(np.ceil(W / cell)))
        n = cv2.resize(rng.standard_normal((h, w)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    n -= n.mean()
    return n / max(1e-6, float(n.std()))


def apply_effect(name: str, rgb: np.ndarray, amount: float, *, seed: int = 1, radius: Optional[float] = None,
                 size: Optional[float] = None, gamma: float = 1.0,
                 stat_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """The whole-picture effect on float32 RGB [H,W,3] 0..1 (the caller blends it through a region mask)."""
    rgb = np.asarray(rgb, np.float32)
    H, W = rgb.shape[:2]
    L = float(max(H, W))
    a = float(np.clip(amount, 0.0, 1.0))
    if name == "glow":
        lum = _lum(rgb)
        w = np.clip((lum - 0.5) / 0.5, 0, 1) ** 1.5
        bright = rgb * w[..., None]
        sig = float(radius) if radius else max(2.0, 0.02 * L)
        g = 0.6 * _blur(bright, sig) + 0.4 * _blur(bright, sig * 2.5)
        out = 1.0 - (1.0 - rgb) * (1.0 - np.clip(g * (1.2 * a), 0, 1))
    elif name == "grain":
        rng = np.random.default_rng(int(seed))
        cell = float(size) if size else max(1.0, L / 900.0)
        lum_n = _noise(rng, H, W, cell)
        col_n = np.stack([_noise(rng, H, W, cell) for _ in range(3)], -1)
        noise = (0.8 * lum_n[..., None] + 0.35 * col_n) / 0.95
        weight = 1.0 - 0.7 * (2.0 * _lum(rgb) - 1.0) ** 2
        out = rgb + noise * (0.075 * a) * weight[..., None]
    elif name == "blur":
        sig = float(radius) if radius else max(0.5, a * 0.012 * L)
        out = _blur(rgb, sig)
    elif name == "vignette":
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((xx - (W - 1) / 2) / (W / 2)) ** 2 + ((yy - (H - 1) / 2) / (H / 2)) ** 2) / np.sqrt(2.0)
        t = np.clip((d - 0.35) / 0.65, 0, 1)
        f = t * t * (3 - 2 * t)
        out = rgb * (1.0 - 0.75 * a * f)[..., None]
    elif name == "paper":
        rng = np.random.default_rng(int(seed))
        fine = _blur(_noise(rng, H, W, 1.0), 0.7)
        fibre = cv2.GaussianBlur(_noise(rng, H, W, 1.0), (0, 0), sigmaX=max(2.0, L / 250.0), sigmaY=0.6)
        coarse = _noise(rng, H, W, max(8.0, L / 40.0))
        tex = 0.45 * fine / max(1e-6, fine.std()) + 0.25 * fibre / max(1e-6, fibre.std()) + 0.30 * coarse
        out = rgb * (1.0 + 0.055 * a * tex)[..., None]
        out = out * np.array([1.0, 1.0 - 0.025 * a, 1.0 - 0.085 * a], np.float32)
    elif name == "sharpen":
        sig = float(radius) if radius else max(0.8, 0.002 * L)
        out = rgb + (1.6 * a) * (rgb - _blur(rgb, sig))
    elif name == "levels":
        lum = _lum(rgb)
        sel = lum[stat_mask] if stat_mask is not None and stat_mask.any() else lum.reshape(-1)
        lo, hi = np.percentile(sel, [0.5, 99.5])
        lo, hi = float(min(lo, 0.35)), float(max(hi, 0.65))
        st = np.clip((rgb - lo) / max(1e-3, hi - lo), 0, 1)
        if abs(gamma - 1.0) > 1e-3:
            st = st ** (1.0 / max(0.2, gamma))
        out = rgb + (st - rgb) * a
    else:
        raise UsageError(f"unknown effect {name!r}", "pixelcanvas fx grain")
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def blend_region(base: np.ndarray, effect: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return effect
    m = cv2.GaussianBlur(np.asarray(mask, np.float32), (0, 0), 1.2)[..., None]
    return base * (1.0 - m) + effect * m


def to_fx_layer(below: np.ndarray, target: np.ndarray) -> np.ndarray:
    """RGBA layer (straight alpha) that gives ``target`` when composited normally over ``below``."""
    d = target - below
    up = np.where(d > EPS, d / np.maximum(1.0 - below, 1e-6), 0.0)
    down = np.where(d < -EPS, -d / np.maximum(below, 1e-6), 0.0)
    a = np.clip(np.maximum(up, down).max(-1), 0.0, 1.0)
    col = np.where(a[..., None] > 0, below + d / np.maximum(a, 1e-6)[..., None], 0.0)
    out = np.zeros(below.shape[:2] + (4,), np.float32)
    a8 = np.round(a * 255.0) / 255.0
    out[..., :3] = np.clip(col, 0, 1)
    out[..., 3] = a8
    out[a8 <= 0, :3] = 0.0
    return out


# ----------------------------------------------------------------------------------------------- composite helpers

def _with_text(p, rgb: np.ndarray) -> np.ndarray:
    """``rgb`` (the picture through the fx layer) with the text layer drawn over it."""
    e = p.layer_entry("text")
    arr = p.layer("text", create=False)
    if arr is None or not e.get("visible", True):
        return rgb
    out, _ = compose.composite_over(rgb, np.ones(rgb.shape[:2], np.float32), arr, e.get("mode", "normal"),
                                    e.get("opacity", 1.0))
    return out


# ----------------------------------------------------------------------------------------------- the command

def cmd_fx(ctx, args: List[str]):
    from . import cli, metrics, workflow
    from .cli import Report, Spec, parse
    usage = cli.USAGE["fx"]
    sp = Spec(usage, ["name", "region*"],
              {"amount": "value", "radius": "value", "size": "value", "seed": "value", "gamma": "value"},
              alias={"strength": "amount", "intensity": "amount", "sigma": "radius", "grain-size": "size",
                     "colour": "color"})
    a = parse(args, sp)
    p = ctx.project
    if not a.get("name"):
        rep = Report()
        rep.add("finishing effects (limited to a region if you name one; undo with  pixelcanvas undo):")
        for n, (d, desc) in EFFECTS.items():
            rep.add(f"  fx {n:<9} {desc}  [default amount {d:g}]")
        rep.add("with a reference and no --amount the tool tries several amounts and keeps the one that raises the score")
        rep.next = "pixelcanvas fx grain --amount 0.4"
        return rep
    name = canonical(a["name"])
    u = f"pixelcanvas fx {name} hair --amount 0.4"
    region = cli.resolve_region(p, a.get("region"), ctx.notes, u)
    explicit = a.get("amount") is not None
    amount = cli._num(a, "amount", EFFECTS[name][0], 0.0, 1.0, ctx.notes, usage)
    radius = cli._num(a, "radius", 0.0, 0.5, 400.0, ctx.notes, usage) or None if a.get("radius") else None
    size = cli._num(a, "size", 0.0, 0.5, 50.0, ctx.notes, usage) or None if a.get("size") else None
    gamma = cli._num(a, "gamma", 1.0, 0.3, 3.0, ctx.notes, usage)
    seed = int(coords.parse_int(a["seed"], "seed", 0, 10 ** 9, ctx.notes, usage)) if a.get("seed") else 1
    mask = None if region.name == "all" else region.mask
    below = p.composite("score", upto="fx")                      # the picture under the fx layer
    cur = p.composite("score", upto="text")                      # ... through it (earlier effects are kept)
    kw = dict(seed=seed, radius=radius, size=size, gamma=gamma, stat_mask=mask)

    def target_for(amt: float) -> np.ndarray:
        return blend_region(cur, apply_effect(name, cur, amt, **kw), mask)
    from .draw_render import keeps_clean
    if name == "blur" and keeps_clean(p):
        # a drawing over the 3D guide: each drawn area blurs only with itself (no colour crosses a contour, no halo
        # beside a hard edge) and the line art stays sharp
        from .draw_figure2d import load_maps
        la = p.layer("line", create=False)
        line_m = (la[..., 3] > 0.02) if la is not None else np.zeros((p.H, p.W), bool)
        mp = load_maps(p)
        lab = cv2.resize(mp["lab"].astype(np.int32), (p.W, p.H), interpolation=cv2.INTER_NEAREST)             if mp is not None else np.zeros((p.H, p.W), np.int32)
        # a thin band along every contour is left as it is and lends no colour (the flat colours and the drawn
        # shapes may differ there by a pixel)
        lab_f = lab.astype(np.float32)
        k3 = np.ones((3, 3), np.uint8)
        edge = cv2.dilate(lab_f, k3) != cv2.erode(lab_f, k3)
        line_m = line_m | (cv2.dilate(edge.astype(np.uint8), k3) > 0)
        L_ = float(max(p.W, p.H))

        def target_for(amt: float) -> np.ndarray:  # noqa: F811
            sig = float(radius) if radius else max(0.5, amt * 0.012 * L_)
            m = np.ones((p.H, p.W), bool) if mask is None else np.asarray(mask) > 0.5
            rgb = np.asarray(cur, np.float32)[..., :3]
            out = rgb.copy()
            pad = int(3 * sig) + 2
            for v in np.unique(lab[m]):
                area = (lab == v) & m
                ys, xs = np.nonzero(area)
                y0, y1 = max(0, ys.min() - pad), min(p.H, ys.max() + pad + 1)
                x0, x1 = max(0, xs.min() - pad), min(p.W, xs.max() + pad + 1)
                src = ((lab[y0:y1, x0:x1] == v) & ~line_m[y0:y1, x0:x1]).astype(np.float32)
                w = cv2.GaussianBlur(src, (0, 0), sig)
                blur = np.stack([cv2.GaussianBlur(rgb[y0:y1, x0:x1, c] * src, (0, 0), sig) for c in range(3)], -1)
                blur = blur / np.maximum(w[..., None], 1e-4)
                sel = area[y0:y1, x0:x1] & ~line_m[y0:y1, x0:x1] & (w > 1e-3)
                out[y0:y1, x0:x1][sel] = blur[sel]
            res = np.array(cur, np.float32, copy=True)
            res[..., :3] = out
            return res

    extra: List[str] = []
    worse_than_before = False
    local = None
    if name == "grain" and not explicit and size is None and p.ref is not None:
        # the reference's own fine noise, area by area (none on clean flats, gradients and textured strokes)
        try:
            from .pipeline import grain_local
            cur_u8 = np.clip(np.rint(np.asarray(cur)[..., :3] * 255.0), 0, 255).astype(np.uint8)
            amap, gsize = grain_local(p.ref, cur_u8)
            local = (amap, gsize, cur_u8)
        except Exception:  # noqa: BLE001
            local = None
    if local is not None and local[0] is None:
        rep = Report()
        rep.add("grain: the reference has no fine noise the canvas lacks; nothing applied")
        if p.state.get("stage") == "finish" and workflow.tick(p.state, "finish", None, key="grain"):
            p.save_state()
        rep.next = cli._finish_next(ctx)
        return rep
    if local is not None:
        from .pipeline import add_grain_local
        amap, gsize, cur_u8 = local
        if mask is not None:
            amap = amap * np.asarray(mask, np.float32)
        out = add_grain_local(cur_u8, amap, gsize, seed=seed).astype(np.float32) / 255.0
        loc_target = np.array(cur, np.float32, copy=True)
        loc_target[..., :3] = out
        share = float((amap > 0.02).mean())
        extra.append(f"local grain: size {gsize:g}, on {100 * share:.0f}% of the canvas, as strong as the "
                     f"reference's own noise there (none on clean flats and gradients)")
        target_for = lambda amt: loc_target  # noqa: E731
        explicit = True
        amount = 1.0
    if not explicit and p.ref is not None:
        use_lines = workflow.uses_lines(p.state)
        smask = region.mask if mask is not None else None

        def overall(rgb: np.ndarray) -> Optional[float]:
            u8 = compose.to_uint8(_with_text(p, rgb))
            cmp = cli._comparison(p, u8)
            return cmp.scores(smask)["overall"]

        base_score = overall(cur)
        tried: List[Tuple[float, float]] = []
        for c in CANDIDATES:
            s = overall(target_for(c))
            if s is not None:
                tried.append((c, float(s)))
        if tried and base_score is not None:
            best_amt, best = max(tried, key=lambda t: (t[1], -t[0]))
            amount = best_amt
            row = "  ".join(f"{c:g}:{int(round(s))}" for c, s in tried)
            if best > base_score + 0.4:
                extra.append(f"amount {amount:g} chosen by score (overall now {int(round(base_score))}; tried "
                             f"amount:overall  {row})")
            else:
                amount = min(c for c, _ in tried)
                worse_than_before = True
                extra.append(f"no amount raised the score (overall now {int(round(base_score))}; tried {row}); "
                             f"applied the gentlest, {amount:g}")
        else:
            extra.append(f"amount {amount:g} (default)")
    elif not explicit:
        extra.append(f"amount {amount:g} (default; no reference to measure against)")
    target = target_for(amount)
    layer = to_fx_layer(below, target)
    if not (layer[..., 3] > 0).any():
        extra.append("the effect changed nothing visible; try a larger --amount")
    e = p.layer_entry("fx")
    fixed = [w for w, bad in (("mode normal", e.get("mode", "normal") != "normal"),
                              ("opacity 1", float(e.get("opacity", 1.0)) != 1.0),
                              ("shown", not e.get("visible", True)), ("no clip", bool(e.get("clip"))),
                              ("alpha lock off", bool(e.get("alpha_lock")))) if bad]
    if fixed:
        extra.append("the fx layer's earlier settings are baked into the picture; reset to " + ", ".join(fixed))

    def make(arr: np.ndarray) -> np.ndarray:
        ent = p.layer_entry("fx")
        ent.update({"mode": "normal", "opacity": 1.0, "visible": True, "clip": False, "alpha_lock": False})
        return layer

    label = f"fx {name} {region.label} --amount {amount:g}" + (f" --seed {seed}" if name in ("grain", "paper") else "")
    rep = cli.paint_step(ctx, label=label, layer="fx", build=lambda: [], clip=None, area_px=region.box_px,
                         region=region, pixel_fn=make, extra_lines=extra, key_metric="overall")
    missing = [ln for ln in extra if ln not in rep.lines]
    for k, ln in enumerate(missing):
        rep.lines.insert(1 + k, ln)
    if worse_than_before:
        if not any(ln.startswith("WARNING") for ln in rep.lines):
            rep.lines.append("WARNING: this effect does not help this picture; undo it with  pixelcanvas undo")
        rep.next = f"pixelcanvas undo   (fx {name} did not raise the score)"
    return rep
