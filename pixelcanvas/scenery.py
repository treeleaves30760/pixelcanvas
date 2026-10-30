# -*- coding: utf-8 -*-
"""
scenery — 可重複使用的「景物配方」：天空 / 太陽、雲、山脈、霧、河流與水面、樹木、飛鳥、光線、後製。

每個配方都接收 ``Scene``（共用狀態：畫布、座標網格、縮放係數、太陽、霧色、地形遮罩），
畫到 ``sc.cv`` 上並回傳有用的中間結果（遮罩、密度場…）。

所有「像素單位」的參數（噪聲 scale、模糊 sigma、樹高…）都以 **1080p 為基準**，
內部自動乘上 ``sc.S``（= 高度 / 1080），因此同一組參數在草稿（960x540）與全尺寸看起來一致。
畫面位置一律用比例（0~1）給。

寫新配方的原則：
* 先在遮罩 / 密度場上思考（哪裡有東西、多厚），再決定顏色（受光 / 陰影 / 霧）。
* 光照 = 大尺度（形狀的朝向）+ 小尺度（噪聲梯度對光向量）。
* 遠的東西往 ``sc.haze`` 混（大氣透視），近的東西對比高、顏色深。
* 不透光的東西記得 ``sc.occlude(mask)``，體積光才會被擋住。
"""
from __future__ import annotations

import numpy as np

from .color import F32, rgb, lerp, mix, clamp01, smoothstep
from .noise import fbm2d, fbm1d, warp2d, perlin1d_at, smooth1d
from .masks import shape_mask, band_mask
from .brush import stroke
from .filters import gaussian_blur, sample, light_shafts
from .canvas import Canvas

__all__ = [
    "Scene", "sky", "sun", "cirrus", "cumulus", "ridge", "mountain", "mountain_layer", "valley_floor", "mist",
    "haze_band", "river_course", "water_mask", "water_surface", "river_bank", "pine", "pine_mask",
    "scatter_trees", "rim_light", "ground_edge", "birds", "god_rays", "finish",
]


class Scene:
    """一張圖的共用狀態。

    ``sc.cv`` 畫布、``sc.X/Y`` 像素座標、``sc.U/V`` 0~1 座標、``sc.S`` 縮放係數（高度 / 1080）、
    ``sc.terrain`` 不透光物件聯集、``sc.sun`` 太陽位置、``sc.D`` 到太陽的距離、``sc.light`` 光向量、``sc.haze`` 霧色圖。
    """

    def __init__(self, width=1920, height=1080, seed=0, ref_height=1080, log=None):
        self.W, self.H = int(width), int(height)
        self.S = self.H / float(ref_height)
        self.seed = int(seed)
        self.cv = Canvas(self.W, self.H)
        self.X, self.Y = self.cv.xy()
        self.U, self.V = self.X / self.W, self.Y / self.H
        self.UX = self.U[0]                                   # (W,) 每一欄的 x 比例
        self.terrain = np.zeros((self.H, self.W), F32)
        self.log = log or (lambda m: None)
        self.horizon = 0.6 * self.H
        self.sun = None
        self.D = None
        self.light = (0.9, -0.42)
        self.haze = None

    def px(self, v):
        """把 1080p 基準的像素值換成目前解析度。"""
        return v * self.S

    def set_horizon(self, frac):
        """眼平線（水面鏡射軸）的高度比例。"""
        self.horizon = frac * self.H
        return self.horizon

    def set_sun(self, xfrac, yfrac, light_dir=None):
        """太陽位置（比例）；``light_dir`` = 指向光源的 (dx, dy)（螢幕座標，y 向下）。"""
        self.sun = (xfrac * self.W, yfrac * self.H)
        self.D = np.hypot(self.X - self.sun[0], self.Y - self.sun[1]).astype(F32)
        if light_dir is not None:
            self.light = light_dir
        return self.sun

    def set_haze(self, stops):
        """霧色（大氣透視的目標色）隨高度的漸層：``[(V 比例, 顏色), ...]``。"""
        self.haze = mix(stops, self.V)
        return self.haze

    def occlude(self, mask):
        """把不透光物件加入地形聯集。"""
        self.terrain = np.maximum(self.terrain, np.asarray(mask, F32))
        return self.terrain

    def _sun_dir(self):
        if self.sun is None:
            return np.zeros_like(self.X), -np.ones_like(self.Y), np.zeros_like(self.X)
        near = np.exp(-(self.D / self.px(900)) ** 2)
        return (self.sun[0] - self.X) / (self.D + 1), (self.sun[1] - self.Y) / (self.D + 1), near


# ----------------------------------------------------------------------------
# 天空與太陽
# ----------------------------------------------------------------------------
def sky(sc, stops):
    """垂直多段漸層天空：``stops = [(V 比例, 顏色), ...]``。"""
    sc.cv.paint(mix(stops, sc.V))
    return sc.cv


def sun(sc, glows=((700, 0.14, "#ff9a55"), (190, 0.14, "#ffb870"), (44, 0.22, "#fff1d6")),
        radius=27, disc=(1.8, 1.6, 1.3), horizon_glow=(110, 0.10, "#f8c48e")):
    """太陽：多層高斯光暈（sigma px, 強度, 顏色）+ HDR 太陽盤 + 地平線暖霧。回傳太陽盤遮罩。"""
    assert sc.sun is not None, "請先 sc.set_sun(...)"
    for sig, k, col in glows:
        sc.cv.paint(col, np.exp(-(sc.D / sc.px(sig)) ** 2) * k, "add")
    disc_mask = 1 - smoothstep(sc.px(radius - 2), sc.px(radius + 2), sc.D)
    if disc is not None:
        sc.cv.paint(disc, disc_mask)
    if horizon_glow:
        sig, k, col = horizon_glow
        sc.cv.paint(col, np.exp(-((sc.Y - sc.horizon) / sc.px(sig)) ** 2) * k, "add")
    return disc_mask


# ----------------------------------------------------------------------------
# 雲
# ----------------------------------------------------------------------------
def cirrus(sc, band=(0.03, 0.45, 0.08), scale=480, aspect=0.10, threshold=(0.30, 0.62), alpha=0.45,
           colors=("#8aa2c6", "#ffd9b5"), warm_range=(0.20, 0.50), seed_off=1):
    """高空卷雲：橫向拉長的 billow 噪聲。``band`` = (起 V, 迄 V, 柔邊 V)。回傳 alpha。"""
    n = fbm2d(sc.H, sc.W, sc.px(scale), 5, seed=sc.seed + seed_off, aspect=aspect, mode="billow")
    a = smoothstep(threshold[0], threshold[1], n) * band_mask(sc.H, sc.W, band[0] * sc.H, band[1] * sc.H, band[2] * sc.H) * alpha
    col = lerp(colors[0], colors[1], smoothstep(warm_range[0], warm_range[1], sc.V))
    sc.cv.paint(col, a)
    return a


def cumulus(sc, band=(0.03, 0.14, 0.33, 0.47), coverage=0.56, softness=0.20, scale=340, warp=0.35, aspect=0.5,
            shade=("#6f7ca5", "#8d7b98"), lit=("#f4e9de", "#ffd6a8"), rim=0.4, seed_off=2):
    """積雲：domain-warp 噪聲切出蓬鬆形狀；把密度當厚度，「朝太陽方向變薄」的地方受光（三個尺度疊加）。

    ``band`` = (淡入起, 淡入迄, 淡出起, 淡出迄) 的 V 比例；``coverage`` 越小雲越多；
    ``shade/lit`` = (遠離太陽, 靠近太陽) 的陰影色 / 受光色。回傳密度場（0~1，可給 god_rays 當遮擋）。
    """
    H, W, X, Y, V = sc.H, sc.W, sc.X, sc.Y, sc.V
    cum = warp2d(H, W, sc.px(scale), warp, 8, seed=sc.seed + seed_off, aspect=aspect, mode="plain") + 0.5
    b = smoothstep(band[0], band[1], V) * (1 - smoothstep(band[2], band[3], V))
    cov = coverage + 0.04 * np.sin(sc.U * 7 + 1.0) + 0.03 * np.cos(sc.U * 3)      # 覆蓋率沿 x 緩慢變化
    dens = smoothstep(cov, cov + softness, cum) * b
    dirx, diry, near = sc._sun_dir()
    clight = np.full((H, W), 0.5, F32)
    for sig, off, k in ((3, 6, 1.6), (9, 16, 1.1), (24, 40, 0.8)):
        dl = gaussian_blur(dens, sc.px(sig))
        clight += k * (dl - sample(dl, X + sc.px(off) * dirx, Y + sc.px(off) * diry))
    detail = fbm2d(H, W, sc.px(60), 4, seed=sc.seed + seed_off + 1, aspect=0.7) + 0.5
    clight = clamp01(clight + 0.25 * (detail - 0.5))
    core = smoothstep(0.5, 1.0, gaussian_blur(dens, sc.px(10)))
    col = lerp(lerp(shade[0], shade[1], near), lerp(lit[0], lit[1], near), clamp01(clight * (1 - 0.45 * core)))
    sc.cv.paint(col, clamp01(dens * 1.05))
    if rim:
        rimm = smoothstep(0.02, 0.2, dens) * (1 - smoothstep(0.2, 0.5, dens)) * clight * (0.3 + 0.7 * near)
        sc.cv.paint("#ffe2b8", rimm * rim, "add")
    return dens


# ----------------------------------------------------------------------------
# 山脈
# ----------------------------------------------------------------------------
def ridge(sc, base, amp, scale, seed_off, peaks=(), dips=(), side=None, rise=0.0, reach=0.6,
          forest=0.0, tree_px=8, octaves=7, gain=0.55):
    """稜線：回傳每一欄的 y（像素，(W,)）。

    ``base``（稜線中線高度比例）、``amp``（起伏幅度比例）；``peaks = [(x 比例, 高度比例, 半寬比例), ...]``
    指定主峰；``dips`` 同格式做鞍部；``side='left'/'right'`` + ``rise`` / ``reach`` 讓近景山坡往一側抬高；
    ``forest`` > 0 在稜線加樹形鋸齒。
    """
    W, H = sc.W, sc.H
    sd = sc.seed + seed_off
    scp = sc.px(scale)
    n = fbm1d(W, scp, octaves, gain=gain, seed=sd, mode="ridged")
    env = 0.55 + 0.45 * fbm1d(W, scp * 2.6, 3, seed=sd + 50)
    prof = base * H - amp * H * (n * env + 0.18 * fbm1d(W, scp * 0.5, 5, seed=sd + 60))
    for px_, ph, pw in peaks:
        tent = np.maximum(0, 1 - np.abs(sc.UX - px_) / pw) ** 1.35
        prof -= ph * H * tent * (1 + 0.12 * fbm1d(W, sc.px(60), 3, seed=sd + 70))
    for px_, pd, pw in dips:
        prof += pd * H * np.exp(-((sc.UX - px_) / pw) ** 2)
    if side == "left":
        prof -= rise * H * (1 - smoothstep(0.0, reach, sc.UX)) ** 1.3
    elif side == "right":
        prof -= rise * H * smoothstep(1 - reach, 1.0, sc.UX) ** 1.3
    if forest > 0:
        spikes = (1 - np.abs(fbm1d(W, sc.px(7), 2, seed=sd + 80))) * (0.45 + 0.55 * (fbm1d(W, sc.px(40), 2, seed=sd + 81) * 0.5 + 0.5))
        prof -= forest * sc.px(tree_px) * spikes
    return prof.astype(F32)


def mountain(sc, prof, base, amp, shadow, lit, fog, scale=300, seed_off=0, snow=None, forest=0.0, treeline=0.9,
             relief=30, hill=60, lift=0.0, warm=0.5, fog_down=0.9, fog_len=1.4,
             snow_colors=("#90a6c8", "#fff4e4"), forest_colors=("#0f1b17", "#4d6e46"), sun_tint="#ffd3a0"):
    """以稜線 ``prof`` 畫一層山（岩石浮雕、雪線、森林、大氣透視），回傳遮罩。

    ``shadow/lit`` 陰影色與受光色；``fog`` 0~1 這層整體的霧量（越遠越大）；``fog_down`` 山腳再加多少霧；
    ``snow`` = 雪線高度比例（None 無雪）；``forest`` 0~1 森林量、``treeline`` 森林上限高度比例；
    ``relief`` / ``hill`` 小 / 中尺度浮雕強度；``warm`` 靠近太陽的暖色量。
    """
    H, W, X, Y, S = sc.H, sc.W, sc.X, sc.Y, sc.S
    amp_px, base_px = amp * H, base * H
    sd = sc.seed + seed_off
    scp = sc.px(scale)
    rel = Y - prof[None, :]                                   # 稜線以下為正（像素）
    mask = clamp01(rel + 0.5)                                 # 1px 抗鋸齒邊緣
    # 大尺度明暗：稜線斜率（向光源那側下降 → 亮）
    slope = np.gradient(smooth1d(prof, sc.px(18)))
    big = 0.5 + 0.5 * np.tanh(slope * 1.3 * np.sign(sc.light[0] or 1.0))
    bigfade = np.exp(-np.maximum(rel, 0) / (amp_px * 1.3))
    big2 = 0.5 + (big[None, :] - 0.5) * bigfade
    # 中尺度起伏 + 小尺度岩石浮雕：當假高度場，取梯度對光向量
    hillf = fbm2d(H, W, scp * 0.9, 3, seed=sd + 8, aspect=1.4)
    rock = fbm2d(H, W, scp * 0.3, 5, gain=0.55, seed=sd + 7, aspect=2.4, mode="ridged")
    hgy, hgx = np.gradient(hillf)
    gy, gx = np.gradient(rock)
    lx, ly = sc.light
    reliefm = 0.5 + (hgx * lx + hgy * ly) * hill * S + (gx * lx + gy * ly) * relief * S
    light = clamp01(0.55 * big2 + 0.45 * reliefm + lift)
    col = lerp(shadow, lit, light) * (0.88 + 0.24 * rock)[..., None]
    if sc.D is not None and warm:
        col = lerp(col, col * rgb(sun_tint) * 1.2, light * warm * np.exp(-(sc.D / sc.px(1100)) ** 2))
    if snow:
        elev = (base_px - Y) / amp_px
        sl = snow + 0.10 * fbm1d(W, sc.px(140), 4, seed=sd + 21)[None, :] + 0.06 * fbm2d(H, W, sc.px(40), 3, seed=sd + 22)
        sm = smoothstep(sl - 0.05, sl + 0.08, elev) * smoothstep(0.18, 0.55, rock + 0.1 * (light - 0.5))
        col = lerp(col, lerp(snow_colors[0], snow_colors[1], clamp01(light * 1.15)), sm * 0.95)
    if forest > 0:
        tex = fbm2d(H, W, sc.px(22), 4, seed=sd + 33, aspect=1.6) + 0.5        # 樹冠細碎起伏
        clump = fbm2d(H, W, sc.px(70), 3, seed=sd + 35, aspect=1.4) + 0.5      # 樹叢明暗團塊
        tex2 = fbm2d(H, W, sc.px(160), 3, seed=sd + 34)                         # 大片疏密
        tl = 1 - smoothstep(treeline - 0.1, treeline + 0.1, (base_px - Y) / amp_px + 0.08 * tex2)
        fl = clamp01(light * 1.1 + 0.5 * (tex - 0.5) + 0.4 * (clump - 0.5) - 0.12)
        fcol = lerp(forest_colors[0], forest_colors[1], fl) * (0.8 + 0.4 * (tex - 0.5) + 0.3 * (clump - 0.5))[..., None]
        col = lerp(col, fcol, forest * tl * (0.7 + 0.3 * smoothstep(-0.2, 0.3, tex2)))
    if sc.haze is not None:
        fogt = fog + (1 - fog) * fog_down * smoothstep(0.0, 1.0, np.maximum(rel, 0) / (amp_px * fog_len))
        col = lerp(col, sc.haze, clamp01(fogt))
    sc.cv.paint(col, mask)
    sc.occlude(mask)
    return mask


_RIDGE_KEYS = ("peaks", "dips", "side", "rise", "reach", "forest", "tree_px", "octaves", "gain")
_MOUNTAIN_KEYS = ("snow", "forest", "treeline", "relief", "hill", "lift", "warm", "fog_down", "fog_len",
                  "snow_colors", "forest_colors", "sun_tint")


def mountain_layer(sc, L):
    """用一個參數 dict 一次完成 ridge + mountain（scenes 裡最常用的寫法）。

    必要鍵：base, amp, scale, seed, shadow, lit, fog；其餘見 ridge() / mountain() 的參數。
    """
    prof = ridge(sc, L["base"], L["amp"], L["scale"], L["seed"], **{k: L[k] for k in _RIDGE_KEYS if k in L})
    return mountain(sc, prof, L["base"], L["amp"], L["shadow"], L["lit"], L["fog"], scale=L["scale"], seed_off=L["seed"],
                    **{k: L[k] for k in _MOUNTAIN_KEYS if k in L})


# ----------------------------------------------------------------------------
# 山谷底與霧
# ----------------------------------------------------------------------------
def valley_floor(sc, top=0.655, fade=0.70, colors=("#5c6a54", "#223328"), fog_range=(0.64, 0.86), seed_off=45):
    """谷底草原：從 ``top`` 比例往下鋪，靠近地平線幾乎全是霧色。回傳遮罩。"""
    H, W, Y, V = sc.H, sc.W, sc.Y, sc.V
    m = smoothstep(top * H, fade * H, Y)
    ft = clamp01((Y - top * H) / ((1 - top) * H))
    tex = fbm2d(H, W, sc.px(60), 4, seed=sc.seed + seed_off, aspect=0.25)
    col = lerp(colors[0], colors[1], ft) * (0.85 + 0.3 * tex)[..., None]
    fogf = 1 - smoothstep(fog_range[0], fog_range[1], V)
    sc.cv.paint(lerp(col, sc.haze, clamp01(fogf * 0.95 + 0.12)), m)
    return m


def mist(sc, v_range=(0.60, 0.70, 0.78, 0.98), strength=0.5, scale=260, aspect=0.12, exclude=None, seed_off=40):
    """飄動的霧帶：橫向拉長的噪聲。``v_range`` = (淡入起, 淡入迄, 淡出起, 淡出迄)。"""
    n = fbm2d(sc.H, sc.W, sc.px(scale), 4, seed=sc.seed + seed_off, aspect=aspect)
    a = smoothstep(-0.1, 0.5, n) * smoothstep(v_range[0], v_range[1], sc.V) * (1 - smoothstep(v_range[2], v_range[3], sc.V)) * strength
    if exclude is not None:
        a = a * (1 - exclude)
    sc.cv.paint(sc.haze * 1.03, a)
    return a


def haze_band(sc, fade_in=(0.58, 0.63), fade_out=(0.615, 0.74), strength=0.8, scale=200, exclude=None, seed_off=41):
    """地平線附近的薄霧（讓遠處的河與谷底一起沒入大氣）。``exclude`` = 不該被罩住的近景遮罩。"""
    n = fbm2d(sc.H, sc.W, sc.px(scale), 3, seed=sc.seed + seed_off, aspect=0.1) + 0.5
    a = (1 - smoothstep(fade_out[0], fade_out[1], sc.V)) * smoothstep(fade_in[0], fade_in[1], sc.V) * (0.35 + 0.4 * n) * strength
    if exclude is not None:
        a = a * (1 - exclude)
    sc.cv.paint(sc.haze, a)
    return a


# ----------------------------------------------------------------------------
# 河流與水面
# ----------------------------------------------------------------------------
def river_course(sc, start=0.5, meander=0.17, freq=1.3, phase=-1.27, width_far=16, width_near=600, power=1.65,
                 wobble=70, seed_off=50):
    """蜿蜒河道：回傳 (center, half, t1)，皆為每一列一個值 (H,)。

    ``t1`` 0 = 遠（眼平線）→ 1 = 近（畫面底）；彎曲隨偽深度壓縮，寬度隨 t1^power 放大（透視）。
    """
    H, W = sc.H, sc.W
    yy = np.arange(H, dtype=F32)
    t1 = clamp01((yy - sc.horizon) / (H - sc.horizon))
    depth = 1.0 / (t1 + 0.085)
    center = start * W + W * meander * np.sin(freq * depth + phase) * (0.2 + 0.8 * t1) \
        + fbm1d(H, sc.px(170), 3, seed=sc.seed + seed_off) * sc.px(wobble) * t1
    half = sc.px(width_far + width_near * t1 ** power)
    return center.astype(F32), half.astype(F32), t1


def water_mask(sc, center, half, t1, edge_noise=(8, 50), fade_in=0.05, seed_off=51):
    """河面遮罩：回傳 (water, e)，``e`` 是到岸邊的帶號距離（< 0 在水中），後面的岸邊效果會用到。"""
    H, W, X, Y = sc.H, sc.W, sc.X, sc.Y
    T = t1[:, None]
    en = fbm2d(H, W, sc.px(110), 3, seed=sc.seed + seed_off, aspect=0.45) * sc.px(edge_noise[0] + edge_noise[1] * T)
    e = np.abs(X - center[:, None]) - half[:, None] + en
    water = (1 - smoothstep(-1.5, 1.5, e)) * smoothstep(sc.horizon, sc.horizon + fade_in * H, Y)
    return water.astype(F32), e.astype(F32)


def water_surface(sc, water, e, t1, colors=("#c8d2da", "#6b8aa6", "#2d4a62", "#16293a"), tint="#e4ecf4",
                  fresnel=(0.90, 0.45), ripple=1.0, bank_shadow="#3a4a46", glitter=0.9, sun_glow=0.45, seed_off=52):
    """水面：以眼平線鏡射「目前畫面」當倒影 + 波紋位移 + 岸邊暗影 + 太陽光帶與閃爍，然後依 ``water`` 塗上。

    ``colors`` = 水色由遠到近 4 段；``fresnel`` = (遠處反射率, 近處遞減量)；``ripple`` 波紋倍率。
    呼叫前畫面上必須已有要被倒映的東西（天空、山、岸）。回傳水色圖。
    """
    H, W, X, Y, S = sc.H, sc.W, sc.X, sc.Y, sc.S
    pre = sc.cv.buf.copy()
    T = t1[:, None]
    sd = sc.seed + seed_off
    rip = fbm2d(H, W, sc.px(70), 4, seed=sd, aspect=0.12)
    rip2 = fbm2d(H, W, sc.px(180), 3, seed=sd + 1, aspect=0.25)
    dx = (rip * (2 + 10 * T) + rip2 * 10 * T) * S * ripple
    dy = np.abs(rip) * (1 + 9 * T) * S * ripple
    src_y = 2 * sc.horizon - Y + dy
    refl = (sample(pre, X + dx, src_y) + sample(pre, X + dx, src_y + sc.px(6) * T) + sample(pre, X + dx, src_y - sc.px(6) * T)) / 3
    wbase = mix([(0, colors[0]), (0.25, colors[1]), (0.6, colors[2]), (1.0, colors[3])], T)
    fres = fresnel[0] - fresnel[1] * t1 ** 0.9
    wcol = lerp(np.broadcast_to(wbase, (H, W, 3)), refl * rgb(tint), fres[:, None])
    bank_sh = 1 - smoothstep(0, sc.px(30 + 220 * T), -e)
    wcol = lerp(wcol, wcol * rgb(bank_shadow) * 0.9, bank_sh * (0.25 + 0.45 * T))
    streak = fbm2d(H, W, sc.px(60), 3, seed=sd + 3, aspect=0.06)
    wcol = wcol * (1 + 0.12 * streak * T)[..., None]
    if sc.sun is not None and (glitter or sun_glow):
        gxs = np.exp(-((X - sc.sun[0]) / sc.px(22 + 130 * T)) ** 2)
        spark = fbm2d(H, W, sc.px(20), 3, seed=sd + 2, aspect=0.08, mode="ridged")
        patch = smoothstep(-0.15, 0.35, fbm2d(H, W, sc.px(120), 3, seed=sd + 5, aspect=0.3))
        glit = gxs * smoothstep(0.68, 0.97, spark) * (0.35 + 0.65 * T) * (0.25 + 0.75 * patch)
        wcol = wcol + rgb((1.0, 0.82, 0.58)) * (glit * glitter)[..., None]
        sry = 2 * sc.horizon - sc.sun[1]
        wcol = wcol + rgb("#ffb878") * (gxs * np.exp(-((Y - sry) / sc.px(90)) ** 2) * sun_glow)[..., None]
        wcol = wcol + rgb("#ffb878") * (np.exp(-((X - sc.sun[0]) / sc.px(260)) ** 2) * np.exp(-((Y - sry) / sc.px(400)) ** 2) * sun_glow * 0.27)[..., None]
    sc.cv.paint(wcol, water)
    return wcol


def river_bank(sc, water, e, t1, colors=("#5e5a50", "#9a9080"), width=(10, 40), seed_off=56):
    """岸邊礫石帶（在水面之外、沿著岸線）。"""
    H, W, Y = sc.H, sc.W, sc.Y
    T = t1[:, None]
    bank = (1 - smoothstep(0, sc.px(width[0] + width[1] * T), e)) * (1 - water) * smoothstep(sc.horizon, sc.horizon + sc.px(40), Y)
    btex = fbm2d(H, W, sc.px(24), 3, seed=sc.seed + seed_off) * 0.5 + 0.5
    bcol = lerp(colors[0], colors[1], btex)
    sc.cv.paint(lerp(bcol, sc.haze, (1 - T) * 0.7 + 0.1), bank * 0.9)
    return bank


# ----------------------------------------------------------------------------
# 樹木、地面、飛鳥
# ----------------------------------------------------------------------------
def pine(d, s, x, yb, h, w, rng, S=1.0):
    """用 Pillow ImageDraw 畫一棵松樹剪影（層層下垂的枝 + 樹幹，不對稱、微傾）。座標已含縮放，乘 s 為超取樣。"""
    tiers = int(8 + h / (45 * S))
    top = yb - h
    tw = w * 0.055
    d.rectangle([((x - tw) * s, (yb - h * 0.3) * s), ((x + tw) * s, yb * s)], fill=255)
    lean = rng.uniform(-0.04, 0.04)
    j = 2.5 * S
    for i in range(tiers):
        f0, f1 = i / tiers, (i + 1) / tiers
        yt = top + h * f0 * 0.9
        ybi = top + h * f1 * 0.9 + h * 0.05
        xc = x + lean * (yb - ybi)
        wl = w * (0.10 + 0.92 * f1 ** 1.15) * rng.uniform(0.75, 1.15)
        wr = w * (0.10 + 0.92 * f1 ** 1.15) * rng.uniform(0.75, 1.15)
        pts = [(xc, yt), (xc - wl * 0.55, ybi - h * 0.02), (xc - wl, ybi), (xc - wl * 0.5, ybi - h * 0.012),
               (xc, ybi - h * 0.03), (xc + wr * 0.5, ybi - h * 0.012), (xc + wr, ybi), (xc + wr * 0.55, ybi - h * 0.02)]
        d.polygon([((px + rng.uniform(-j, j)) * s, (py + rng.uniform(-j, j)) * s) for px, py in pts], fill=255)


def pine_mask(sc, trees, seed_off=100, aa=3):
    """``trees = [(x, y_base, height, width), ...]``（像素）→ 松樹剪影遮罩。"""
    rng = np.random.default_rng(sc.seed + seed_off)
    return shape_mask(sc.H, sc.W, lambda d, s: [pine(d, s, *t, rng, sc.S) for t in trees], aa=aa)


def scatter_trees(sc, n, x_range, y_range, height=(60, 120), width_ratio=0.32, avoid=None, seed_off=102):
    """在範圍內隨機撒樹：``x_range/y_range`` 為比例；``avoid`` 為不能種樹的遮罩（例如水面）。回傳樹的列表。"""
    rng = np.random.default_rng(sc.seed + seed_off)
    out = []
    for _ in range(n):
        x = rng.uniform(*x_range) * sc.W
        yb = rng.uniform(*y_range) * sc.H
        h = rng.uniform(*height) * sc.S
        if avoid is not None and avoid[int(min(yb, sc.H - 1)), int(min(x, sc.W - 1))] > 0.2:
            continue
        out.append((x, yb, h, h * width_ratio))
    return out


def rim_light(sc, mask, color="#d99a5a", offset=(3, -2), strength=0.55):
    """剪影朝光源那側的邊緣受光：``offset`` = 往光源方向的位移（px）。"""
    rim = mask * (1 - sample(mask, sc.X + sc.px(offset[0]), sc.Y + sc.px(offset[1])))
    sc.cv.paint(color, rim * strength, "add")
    return rim


def ground_edge(sc, y_frac, side="left", reach=(0.18, 0.32), color="#0a100e", wobble=22, seed_off=62):
    """畫面底部角落的暗色地面（前景樹的基座）。``reach`` = 從該側算起淡出的 U 範圍。回傳遮罩。"""
    prof = y_frac * sc.H + sc.px(wobble) * fbm1d(sc.W, sc.px(80), 3, seed=sc.seed + seed_off)
    m = clamp01(sc.Y - prof[None, :] + 0.5)
    if side == "left":
        m = m * (1 - smoothstep(reach[0], reach[1], sc.U))
    else:
        m = m * smoothstep(1 - reach[1], 1 - reach[0], sc.U)
    sc.cv.paint(color, m)
    sc.occlude(m)
    return m


def birds(sc, flock, color="#1c2a3c", alpha=0.85):
    """飛鳥（淺 V 字）：``flock = [(x 比例, y 比例, 大小 px, 傾斜), ...]``。"""
    m = np.zeros((sc.H, sc.W), F32)
    r = max(sc.px(1.0), 0.6)
    for bx, by, sz, tilt in flock:
        bx, by, sz = bx * sc.W, by * sc.H, sc.px(sz)
        stroke(m, [(bx - sz, by - sz * 0.45 + tilt * sz), (bx - sz * 0.5, by - sz * 0.12), (bx, by)], r, hardness=0.6, spacing=0.3)
        stroke(m, [(bx, by), (bx + sz * 0.5, by - sz * 0.12), (bx + sz, by - sz * 0.45 - tilt * sz)], r, hardness=0.6, spacing=0.3)
    sc.cv.paint(color, m * alpha)
    return m


# ----------------------------------------------------------------------------
# 光線與後製
# ----------------------------------------------------------------------------
def god_rays(sc, sun_disc, clouds=None, color="#ffc98a", strength=0.24, spread=240, streaks=48,
             samples=48, decay=0.96, length=0.9, seed_off=77):
    """體積光：太陽周圍未被地形 / 雲遮住的亮區（加上角度方向的條紋）朝太陽放射模糊後相加。"""
    assert sc.sun is not None
    ang = (np.arctan2(sc.Y - sc.sun[1], sc.X - sc.sun[0]) + np.pi) / (2 * np.pi)
    ray_n = 0.55 + 0.45 * smoothstep(-0.3, 0.5, perlin1d_at(ang * streaks, seed=sc.seed + seed_off))
    src = (np.exp(-(sc.D / sc.px(spread)) ** 2) * 0.8 + sun_disc * 1.5) * (1 - sc.terrain) * ray_n
    if clouds is not None:
        src = src * (1 - clamp01(clouds) * 0.8)
    rays = light_shafts(src, sc.sun, samples=samples, decay=decay, length=length, downscale=2)
    rays = rays * (0.3 + 0.7 * (1 - sc.terrain))
    sc.cv.paint(color, rays * strength, "add")
    return rays


def finish(sc, bloom=((0.90, 45, 0.18), (0.97, 9, 0.12)), knee=0.82, contrast=0.10, saturation=1.06,
           split=("#3d4a66", "#f2c58a", 0.07), vignette=(0.26, 0.55, 0.7), grain=0.010):
    """標準後製：bloom（閾值, sigma px, 強度）→ soft_clip → 對比 / 飽和 → 分離色調 → 暗角 → 顆粒。"""
    cv = sc.cv
    for th, sig, k in (bloom or ()):
        cv.bloom(th, sc.px(sig), k)
    cv.soft_clip(knee)
    if contrast:
        cv.contrast(contrast)
    if saturation != 1:
        cv.saturation(saturation)
    if split:
        cv.split_tone(*split)
    if vignette:
        cv.vignette(*vignette)
    if grain:
        cv.grain(grain, sc.seed)
    return cv
