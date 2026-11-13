# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np

from .. import solid as _S
from ..color import F32
from .shapes import _Checker, _lin


# ===================================================================================================== markings

def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _worley(P, seed=0):
    """F1 and F2 cell distances of 3D Worley noise for points P (N, 3)."""
    c = np.floor(P)
    F1 = np.full(len(P), 9.0, F32)
    F2 = np.full(len(P), 9.0, F32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                cc = c + np.array([dx, dy, dz], F32)
                ix, iy, iz = cc[:, 0].astype(np.int64), cc[:, 1].astype(np.int64), cc[:, 2].astype(np.int64)
                f = np.stack([_S._hash3(ix, iy, iz, seed), _S._hash3(ix, iy, iz, seed + 7),
                              _S._hash3(ix, iy, iz, seed + 13)], -1)
                q = cc + 0.15 + 0.7 * f - P
                d = np.sqrt((q * q).sum(1))
                F2 = np.where(d < F1, F1, np.minimum(F2, d))
                F1 = np.minimum(F1, d)
    return F1, F2


class _BodyPaint(_Checker):
    """Procedural markings that follow a posed body: every shaded point is mapped back to the rest pose through
    its nearest bone, so stripes, spots, socks and masks stay where they belong when the animal moves."""

    _kind = None

    def __init__(self, base, bones, marks, scale=1.0, head=None):
        self.base = _lin(base)
        self.bones = bones                 # list of (A, B, R, A0, R0, kind, s0, radius, f0, f1)
        self.marks = marks
        self.scale = float(scale)
        self.head = head or {}
        n = len(bones)
        self.A = np.array([b[0] for b in bones], F32)
        self.E = np.array([b[1] - b[0] for b in bones], F32)
        self.M = np.array([b[4] @ b[2].T for b in bones], F32)
        self.A0 = np.array([b[3] for b in bones], F32)
        self.kinds = [b[5] for b in bones]
        self.s0 = np.array([b[6] for b in bones], F32)
        self.rad = np.array([b[7] for b in bones], F32)
        self.len = np.linalg.norm(self.E, axis=1).astype(F32)
        self.E0 = np.array([b[4] @ (b[2].T @ (b[1] - b[0])) for b in bones], F32)
        self.f0 = np.array([b[8] for b in bones], F32)
        self.f1 = np.array([b[9] for b in bones], F32)
        self.n = n

    def __call__(self, P, N=None, foot=None):
        P = np.asarray(P, F32)
        n = len(P)
        if n == 0:
            return np.zeros((0, 3), F32)
        best = np.full(n, 1e9, F32)
        idx = np.zeros(n, np.int32)
        tb = np.zeros(n, F32)
        for i in range(self.n):
            e = self.E[i]
            ee = max(float(e @ e), 1e-12)
            w = P - self.A[i]
            t = np.clip(w @ e / F32(ee), 0, 1)
            q = w - t[:, None] * e
            d = np.sqrt((q * q).sum(1)) - self.rad[i] * 0.5
            b = d < best
            best = np.where(b, d, best)
            idx = np.where(b, i, idx)
            tb = np.where(b, t, tb)
        # rest-pose coordinates
        Pr = np.einsum("nij,nj->ni", self.M[idx], P - self.A[idx]) + self.A0[idx]
        axis = self.A0[idx] + tb[:, None] * self.E0[idx]
        rel = Pr - axis
        rad = np.maximum(self.rad[idx], F32(1e-4))
        kind = np.array(self.kinds)[idx]
        s_arc = self.s0[idx] + tb * self.len[idx]
        w = F32(0.012) if foot is None else np.maximum(foot / F32(self.scale) * 1.5, F32(0.004))
        col = np.broadcast_to(self.base, (n, 3)).copy()
        ct = self.f0[idx] + tb * (self.f1[idx] - self.f0[idx])
        ctx = dict(P=Pr, rel=rel, rad=rad, kind=kind, t=tb, s=s_arc, w=w, N=N, ct=ct)
        for m in self.marks:
            a = _mark_alpha(m, ctx, self.head)
            if a is None:
                continue
            c = _lin(m.get("color", "#ffffff"))
            col = col * (1 - a[:, None]) + c[None] * a[:, None]
        return col


def _mark_alpha(m, c, head):
    k = m["kind"]
    P, rel, rad, kind, t, s, w = c["P"], c["rel"], c["rad"], c["kind"], c["t"], c["s"], c["w"]
    yr = rel[:, 1] / rad
    torso = np.isin(kind, ("torso", "neck"))
    legs = np.isin(kind, ("leg_f", "leg_h"))
    ct = c["ct"]
    if k == "belly":
        lvl = m.get("level", -0.35)
        a = 1 - _smooth(lvl - 0.12, lvl + 0.12, yr)
        return a * (torso | (kind == "head")).astype(F32)
    if k == "socks":
        h = m.get("height", 0.6)
        return legs.astype(F32) * _smooth(h - 0.04, h + 0.04, ct)
    if k == "legs":
        h = m.get("from_", 0.0)
        return legs.astype(F32) * _smooth(h - 0.04, h + 0.04, ct)
    if k == "tail_tip":
        f = m.get("from_", 0.7)
        return (kind == "tail").astype(F32) * _smooth(f - 0.03, f + 0.03, ct)
    if k == "muzzle":
        hz = head.get("snout_z", 0.3)
        a = _smooth(hz * m.get("start", 0.35) - 0.03, hz * m.get("start", 0.35) + 0.03, rel[:, 2])
        a = a * (1 - _smooth(m.get("top", 0.25) - 0.08, m.get("top", 0.25) + 0.08, yr))
        return a * (kind == "head").astype(F32)
    if k == "mask":
        out = np.zeros(len(P), F32)
        for e in head.get("eyes", []):
            d = np.sqrt(((P - F32(e)) ** 2).sum(1))
            r = m.get("radius", 0.09)
            out = np.maximum(out, 1 - _smooth(r - w, r + w, d))
        return out * (kind == "head").astype(F32)
    if k == "band":
        lo_, hi_ = m.get("range", (0.62, 0.9))
        a = _smooth(lo_ - 0.02, lo_ + 0.02, t) * (1 - _smooth(hi_ - 0.02, hi_ + 0.02, t))
        return a * (kind == "torso").astype(F32) * (1 - (yr < -0.9).astype(F32) * 0)
    if k == "saddle":
        lvl = m.get("level", 0.35)
        return _smooth(lvl - 0.15, lvl + 0.15, yr) * torso.astype(F32)
    if k == "stripes":
        f = m.get("freq", 8.0)
        warp = _S.fbm3(P[:, 0] * 3, P[:, 1] * 3, P[:, 2] * 3, 3, m.get("seed", 3)) - 0.5
        u = s * f + warp * m.get("warp", 1.2) + (np.abs(rel[:, 0]) / rad) * m.get("chevron", 0.0)
        v = np.abs(u - np.floor(u) - 0.5) * 2
        width = m.get("width", 0.35)
        ww = np.clip(w * f * 2, 0.02, 0.5)
        a = 1 - _smooth(width - ww, width + ww, v)
        keep = m.get("on", ("torso", "neck", "tail", "head", "leg"))
        sel = np.zeros(len(P), bool)
        for kk in keep:
            sel |= np.isin(kind, ("leg_f", "leg_h")) if kk == "leg" else (kind == kk)
        a = a * sel.astype(F32)
        if m.get("top_only"):
            a = a * _smooth(-0.6, -0.25, yr)
        return a
    if k in ("spots", "patches", "giraffe", "dapple"):
        f = m.get("freq", 6.0)
        Q = P * F32(f)
        if k == "patches":
            nn = _S.fbm3(Q[:, 0], Q[:, 1], Q[:, 2], 3, m.get("seed", 5))
            th = m.get("amount", 0.5)
            a = _smooth(1 - th - 0.03, 1 - th + 0.03, nn)
        else:
            F1, F2 = _worley(Q, m.get("seed", 1))
            if k == "giraffe":
                a = _smooth(0.05, 0.12, F2 - F1)
            else:
                r = m.get("size", 0.32)
                ww = np.clip(w * f * 1.5, 0.01, 0.2)
                a = 1 - _smooth(r - ww, r + ww, F1)
        sk = m.get("skip", ())
        sel = ~np.isin(kind, tuple(k2 for k2 in sk if not k2.startswith("leg_")))
        if any(k2.startswith("leg_") for k2 in sk):
            sel &= ~(legs & (ct > 0.34))
        return a * sel.astype(F32)
    if k == "points":
        a = np.zeros(len(P), F32)
        a = np.maximum(a, legs.astype(F32) * _smooth(0.45, 0.75, ct))
        a = np.maximum(a, (kind == "tail").astype(F32) * _smooth(0.1, 0.4, ct))
        hz = head.get("snout_z", 0.3)
        a = np.maximum(a, (kind == "head").astype(F32) * _smooth(0.0, hz * 0.8, rel[:, 2]))
        return a
    if k == "region":
        return np.isin(kind, tuple(m["kinds"])).astype(F32)
    if k == "rainbow":
        return None
    return None
