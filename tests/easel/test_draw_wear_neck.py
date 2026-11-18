"""The neck, the shoulders, the scarf's overlap order and clean line paths: the construction defects a judge named in
anime busts (a skin column as wide as the face, level shoulders, a knot of crossing ovals, stray curls)."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .test_draw_wear import draw, lines_of, ok, px_path, setup_girl, state  # noqa: F401  (draw is a fixture)

BUST = ("plan subject girl --box 150,50,850,1000; plan part head --of girl --box 330,90,670,450 --focus 3; "
        "plan part eye-l --of girl --box 390,250,475,310 --focus 3; plan part eye-r --of girl --box 525,250,610,310 "
        "--focus 3; plan part hair --of girl --box 240,60,760,800 --focus 2; plan part scarf --of girl "
        "--box 300,450,700,700 --focus 2; plan part torso --of girl --box 170,560,830,1000 --focus 1; "
        "plan light top-left; palette make; palette pick 4; stage done")


def bust(run, size="400x400", wear=("scarf", "shirt"), locks=True):
    """The anime bust session that showed the defects: mannequin, head, eyes, traced jaw, face, clothes, hair."""
    ok(run("new", "w", "--mode", "free", "--size", size, "--request",
           "An anime girl with long silver hair and a red scarf, bust shot, soft cel shading, pale blue background."))
    ok(run("do", BUST))
    ok(run("sketch", "figure", "girl", "--pose", "bust"))
    ok(run("sketch", "joint", "s1", "lsh", "270,580", "rsh", "730,580"))
    ok(run("sketch", "head", "girl"))
    ok(run("stage", "done"))
    ok(run("line", "eye", "eye-l,eye-r"))
    ok(run("line", "trace", "s2"))
    ok(run("face", "s2"))
    for piece in wear:
        ok(run("wear", "girl", piece))
    if locks:
        ok(run("locks", "hair", "--cut", "long"))


def _project():
    from pixelcanvas.easel.project import Project
    return Project.find("w.easel")


def _jaw(st, W, H):
    jaw = next(r for r in st["draw"]["lines"] if str(r.get("src", "")).startswith("trace"))
    P = px_path(jaw, W, H)
    return P, float(np.ptp(P[:, 0]))


# ============================================================================================ the neck

def test_the_neck_is_about_half_the_jaw_wide(draw):
    W, H = 400, 400
    bust(draw)
    ok(draw("stage", "done"))
    ok(draw("flat", "auto"))
    st = state()
    jaw, jaw_w = _jaw(st, W, H)
    chin = jaw[np.argmax(jaw[:, 1])]
    # the skin (head and neck areas) a little below the chin: a neck, not a column as wide as the face
    p = _project()
    skin = np.zeros((H, W), bool)
    for name in st["regions"]:
        if name in st["draw"]["parts"] and st["draw"]["parts"][name].get("swatch") == "skin":
            skin |= p.region_mask(name).astype(bool)
    y = int(round(chin[1] + 0.02 * H))
    x = int(round(chin[0]))
    row = skin[y]
    assert row[x], "no skin under the chin"
    l = x
    while l > 0 and row[l - 1]:
        l -= 1
    r = x
    while r < W - 1 and row[r + 1]:
        r += 1
    width = r - l + 1
    assert width < 0.6 * jaw_w, (width, jaw_w)
    assert width > 0.35 * jaw_w, (width, jaw_w)
    # the neck is two line records of a planned neck part, coloured skin
    necks = [r for r in st["draw"]["lines"] if (r.get("part") or "").startswith("neck")]
    assert len(necks) >= 2
    assert st["draw"]["parts"]["neck"]["swatch"] == "skin"
    # the back of the scarf band, seen on both sides of the neck, is coloured as the scarf
    scarf = p.region_mask("scarf").astype(bool)
    backs = [r for r in lines_of(st, "scarf", "scarf wrap back")]
    assert {r["wear"]["item"] for r in backs} == {"scarf wrap back-l", "scarf wrap back-r"}
    for r in backs:
        P = px_path(r, W, H)
        q = P[len(P) // 2]
        assert scarf[int(q[1] + 4), int(q[0])], r["wear"]["item"]


def test_the_neck_lines_start_on_the_jaw_and_are_half_its_width_apart(draw):
    W, H = 400, 400
    bust(draw, wear=(), locks=False)
    st = state()
    jaw, jaw_w = _jaw(st, W, H)
    necks = [px_path(r, W, H) for r in st["draw"]["lines"] if (r.get("part") or "") == "neck"]
    assert len(necks) == 2
    tops = sorted((P[0] for P in necks), key=lambda q: q[0])
    gap = tops[1][0] - tops[0][0]
    assert 0.4 * jaw_w < gap < 0.6 * jaw_w, (gap, jaw_w)
    # each starts on the jaw line (within a line width), so the neck closes against the face
    for q in tops:
        assert np.hypot(*(jaw - q).T).min() < 3.0


# ============================================================================================ the shoulders

def test_the_shoulder_line_slopes_down_from_the_neck_base(draw):
    from pixelcanvas.easel import draw_wear
    W, H = 400, 400
    bust(draw, wear=(), locks=False)
    st = state()
    p = _project()
    body = draw_wear.Body(p, draw_wear.find_figure(p, "girl", "x"), draw_wear._head_of(p, "girl"))
    for s in "lr":
        pts = np.array(body.trapezius(s, np.linspace(0, 1, 11)))
        out = np.abs(pts[:, 0] - body.shc[0])
        assert np.all(np.diff(out) > 0) and np.all(np.diff(pts[:, 1]) > 0), s
        # steeper at the neck than at the shoulder, and a real slope overall
        d = np.diff(pts, axis=0)
        assert d[0, 1] / d[0, 0] * np.sign(d[0, 0]) > d[-1, 1] / d[-1, 0] * np.sign(d[-1, 0]), s
        assert pts[-1, 1] - pts[0, 1] > 0.25 * abs(pts[-1, 0] - pts[0, 0]), s
    # the drawn neck lines run out into that slope: past the neck they go down as they go out
    cx = body.shc[0]
    for r in st["draw"]["lines"]:
        if (r.get("part") or "") != "neck":
            continue
        P = px_path(r, W, H)
        flare = P[np.abs(P[:, 0] - cx) > 1.15 * body.nw["l"]]
        assert len(flare) >= 3
        o = np.abs(flare[:, 0] - cx)
        k = np.argsort(o)
        assert np.all(np.diff(flare[k, 1]) >= -0.5), r["id"]
        assert flare[k[-1], 1] - flare[k[0], 1] > 0.2 * (o.max() - o.min()), r["id"]


def test_a_shirt_shoulder_slopes_down_from_the_neck(draw):
    W, H = 400, 400
    bust(draw, wear=("shirt",), locks=False)
    st = state()
    from pixelcanvas.easel import draw_wear
    p = _project()
    body = draw_wear.Body(p, draw_wear.find_figure(p, "girl", "x"), draw_wear._head_of(p, "girl"))
    rec = next(r for r in lines_of(st, "shirt", "body") if r["wear"]["item"] == "body" and
               (r["closed"] or r.get("shape")))
    poly = px_path(rec, W, H) if rec["closed"] else np.asarray(rec["shape"], float) * [W / 1000.0, H / 1000.0]
    mask = draw_wear.shape_mask(W, H, poly)
    cx = body.shc[0]
    sp = abs(draw_wear.arm_frame(body, "l").SP[0] - cx)
    nw = body.nw["l"]
    for sg in (-1, 1):
        tops = []
        for o in np.linspace(1.2 * nw, 0.85 * sp, 6):
            col = mask[:, int(round(cx + sg * o))]
            tops.append(int(np.argmax(col)))
        assert all(b >= a for a, b in zip(tops, tops[1:])), (sg, tops)
        assert tops[-1] - tops[0] > 0.2 * (0.85 * sp - 1.2 * nw), (sg, tops)


# ============================================================================================ the scarf

ORDER = ("scarf wrap back", "scarf wrap", "scarf end-back", "scarf end-front", "scarf knot")


def _group(item: str) -> int:
    """The part a scarf record belongs to (its outline, or a fold or crease on it): the longest matching name."""
    item = item.replace("back-l", "back").replace("back-r", "back")
    hits = [(len(g), i) for i, g in enumerate(ORDER) if item == g or item.startswith(g + " ")]
    return max(hits)[1] if hits else -1


def _scarf_check(st, W, H):
    from pixelcanvas.easel import draw_wear
    recs = lines_of(st, "scarf")
    shapes = {}
    seen = set()
    for r in recs:
        it = r["wear"]["item"].replace("back-l", "back").replace("back-r", "back")
        if it in ORDER and (r["closed"] or r.get("shape")):
            poly = px_path(r, W, H) if r["closed"] else np.asarray(r["shape"], float) * [W / 1000.0, H / 1000.0]
            key = (it, tuple(np.round(poly[0], 1)))
            if key in seen:
                continue
            seen.add(key)
            shapes.setdefault(it, []).append(poly)
    assert {"scarf wrap", "scarf end-back", "scarf end-front", "scarf knot"} <= set(shapes)

    def mask(name, **kw):
        return np.logical_or.reduce([draw_wear.shape_mask(W, H, q, **kw) for q in shapes[name]])
    for r in recs:
        g = _group(r["wear"]["item"])
        assert g >= 0, r["wear"]["item"]
        P = px_path(r, W, H)
        q = np.clip(np.round(P).astype(int), 0, [W - 1, H - 1])
        # a line of a part behind stops where a nearer part covers it
        for h in range(g + 1, len(ORDER)):
            if ORDER[h] not in shapes:
                continue
            m = mask(ORDER[h], erode=2.5)
            assert not m[q[:, 1], q[:, 0]].any(), (r["id"], r["wear"]["item"], "inside", ORDER[h])
        # a fold or crease stays on its own part
        if not r["wear"]["item"].startswith(ORDER[g] + "-") and r["wear"]["item"] != ORDER[g] and ORDER[g] in shapes:
            own = mask(ORDER[g], dilate=1.5)
            assert own[q[:, 1], q[:, 0]].all(), (r["id"], r["wear"]["item"])


def test_no_scarf_line_crosses_into_a_nearer_part(draw):
    W, H = 400, 400
    bust(draw, wear=("scarf",), locks=False)
    _scarf_check(state(), W, H)


def test_scarf_on_a_standing_figure_both_knot_sides_and_tips(draw):
    W, H = 300, 400
    setup_girl(draw)
    ok(draw("line", "trace", "s2"))
    ok(draw("wear", "girl", "coat"))
    for args in (("--side", "l"), ("--side", "r", "--tips", "pointed")):
        ok(draw("wear", "girl", "scarf", *args))
        _scarf_check(state(), W, H)


def test_the_scarf_band_rises_behind_the_neck(draw):
    W, H = 400, 400
    bust(draw, wear=("scarf",), locks=False)
    st = state()
    back = [r for r in lines_of(st, "scarf", "scarf wrap back")]
    wrap = [r for r in lines_of(st, "scarf") if r["wear"]["item"] == "scarf wrap"]
    assert back and wrap
    # the back of the band is higher than the front of it: a band round the neck
    assert min(px_path(r, W, H)[:, 1].min() for r in back) < min(px_path(r, W, H)[:, 1].min() for r in wrap)
    # the neck stays in front of it: the neck lines reach down to the front of the band
    necks = [px_path(r, W, H) for r in st["draw"]["lines"] if (r.get("part") or "") == "neck"]
    wtop = min(px_path(r, W, H)[:, 1].min() for r in wrap)
    assert necks and all(P[:, 1].max() > wtop for P in necks)


# ============================================================================================ clean line paths

def _crossings(P: np.ndarray, closed: bool):
    """(i, j, loop length) of every pair of crossing segments of a polyline."""
    out = []
    n = len(P)
    if n < 4:
        return out
    A, B = P[:-1], P[1:]
    seg = np.hypot(*(B - A).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])

    def cr(o, a, b):
        return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])
    for i in range(n - 1):
        j = np.arange(i + 2, n - 1)
        if not len(j):
            continue
        d1 = cr(A[j], B[j], A[i][None])
        d2 = cr(A[j], B[j], B[i][None])
        d3 = cr(A[i][None], B[i][None], A[j])
        d4 = cr(A[i][None], B[i][None], B[j])
        hit = (d1 * d2 < 0) & (d3 * d4 < 0)
        for jj in j[hit]:
            if closed and i == 0 and jj == n - 2:
                continue
            out.append((i, int(jj), float(cum[jj] - cum[i + 1])))
    return out


def _clean(st, W, H):
    side = max(W, H)
    bad = []
    for r in st["draw"]["lines"]:
        src = str(r.get("src", ""))
        if not (r.get("wear") or src.startswith(("face ", "neck "))):
            continue
        P = px_path(r, W, H)
        for i, j, L in _crossings(P, bool(r.get("closed"))):
            if L < 0.2 * side:
                bad.append((r["id"], src, "loop", round(L, 1)))
        if not r.get("closed") and len(P) > 2:
            v = np.diff(P, axis=0)
            a = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
            turn = abs(a[-1] - a[0])
            if turn > math.radians(200) and np.hypot(*v.T).sum() < 0.1 * side:
                bad.append((r["id"], src, "curl", round(math.degrees(turn))))
    assert not bad, bad


def test_wear_and_face_lines_have_no_stray_loops(draw):
    W, H = 300, 400
    setup_girl(draw)
    ok(draw("line", "trace", "s2"))
    ok(draw("face", "s2", "--expression", "smile"))
    ok(draw("wear", "girl", "shirt"))
    ok(draw("wear", "girl", "coat"))
    ok(draw("wear", "girl", "collar"))
    ok(draw("wear", "girl", "scarf", "--side", "r", "--tips", "pointed"))
    ok(draw("hand", "girl", "l"))
    ok(draw("hand", "girl", "r", "--open"))
    _clean(state(), W, H)
    for expr in ("open", "surprised", "angry", "sleepy"):
        ok(draw("face", "s2", "--expression", expr))
    ok(draw("wear", "girl", "hood", "--down"))
    _clean(state(), W, H)


def test_the_bust_session_has_no_stray_loops_and_no_cut_buttons(draw):
    W, H = 400, 400
    bust(draw)
    st = state()
    _clean(st, W, H)
    # a button the scarf half covers goes behind it whole: no arc of it is left as a curl
    for r in lines_of(st, "shirt", "body button"):
        assert r["closed"], r["id"]
