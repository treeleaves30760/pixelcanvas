# pixelcanvas API 速查

```python
import numpy as np
from pixelcanvas import *            # 低階：Canvas、顏色、噪聲、遮罩、筆刷、濾鏡
from pixelcanvas.scenery import *    # 高階：Scene 與景物配方（風景用）
```

慣例：x 向右、y 向下，陣列 `arr[y, x]`；遮罩 float32 (H, W) 0~1；顏色可寫 `'#rrggbb'` / `'#rgb'` /
`(r, g, b)` 0~1 / `(255, 128, 0)` / 整張 (H, W, 3)；允許 HDR > 1（最後 `soft_clip`）。
所有 Canvas 方法回傳 self，可串接；`cv.buf` 就是 float32 (H, W, 3) 陣列，可以直接改。

## Canvas

| 方法 | 說明 |
|---|---|
| `Canvas(w, h, color=(0,0,0))` / `Canvas.load(path)` / `Canvas.from_array(arr)` | 建立 / 載入 / 從陣列（(H,W) 灰階自動轉 3 通道） |
| `cv.w, cv.h, cv.shape, cv.copy()` | 尺寸、複製 |
| `cv.xy()` → `(X, Y)` 像素座標；`cv.uv()` → `(U, V)` 0~1 | 座標網格（float32 (H, W)） |
| `cv.get(x, y)` / `cv.put(x, y, color)` | 讀寫像素；x, y 可為整數陣列，color 可為 (N, 3) |
| `cv.region(x0, y0, x1, y1)` | 子區域 view（直接改即生效） |
| `cv.paint(color, mask=None, mode="normal", opacity=1)` | **上色的核心**：依遮罩混合。mode：normal / add / multiply / screen / overlay / softlight / lighten / darken |
| `cv.gradient(stops, axis="y")` | 線性漸層，`stops = [(0~1, 顏色), ...]` |
| `cv.radial(cx, cy, r, stops)` | 放射漸層（0 = 中心，1 = 半徑 r） |
| `cv.glow(cx, cy, sigma, color, strength=1, mode="add")` | 高斯光暈（可 HDR） |
| `cv.paste(other, x, y, mask=None, mode=..., opacity=...)` | 貼另一張 Canvas / 陣列（自動裁切） |
| `cv.sample(X, Y)` | 在浮點座標取樣目前畫面（倒影、扭曲） |
| `cv.displace(dx, dy)` / `resize(w, h)` / `crop(...)` / `flip("y"或"x")` | 幾何 |
| `cv.blur(s)` `bloom(th, sigma, k)` `vignette(k, r, soft)` `grain(amt, seed)` `levels(black, white, gamma)` `contrast(k)` `saturation(s)` `soft_clip(knee)` `filmic()` `split_tone(shadow, highlight, k)` `clip()` | 後製 |
| `cv.apply(fn, *args)` | `cv.buf = fn(cv.buf, *args)` |
| `cv.save(path, preview=None, preview_width=960)` / `cv.stats()` | 存檔（可順便存縮圖）/ 印亮度統計（>1 比例 > 3% 通常太亮） |

## 顏色與數學（pixelcanvas.color）

| 函數 | 說明 |
|---|---|
| `rgb(c)` / `as_color(c)` | 轉成 float32 ndarray |
| `lerp(a, b, t)` | 線性內插；a, b 可為顏色或顏色圖，t 可為 (H, W) 遮罩 → 回傳 (H, W, 3) |
| `mix(stops, t)` | 多段漸層查表：`mix([(0,'#000'),(1,'#fff')], V)` → (H, W, 3) |
| `hsv_to_rgb(h, s, v)` / `luma(img)` | 色彩空間 / 亮度 |
| `clamp01(a)` / `smoothstep(e0, e1, x)` / `normalize(a, lo, hi)` | 常用數學（smoothstep 支援陣列邊界、e0 > e1 反向） |
| `grid(h, w)` | 像素座標網格 (X, Y) |
| `F32` | `np.float32` |

## 噪聲（pixelcanvas.noise）

`scale` = 最低頻一格的像素寬（越大越平滑）；`aspect` > 1 縱向拉長、< 1 橫向拉長；`mode`：
`"plain"`（約 -1~1，地形 / 雲）、`"ridged"`（0~1，尖銳稜線）、`"billow"`（0~1，鼓起的雲朵）。

| 函數 | 說明 |
|---|---|
| `fbm2d(h, w, scale, octaves=6, lacunarity=2, gain=0.5, seed=0, offset=(0,0), aspect=1, mode="plain")` | 整張 fBm（快速路徑，1080p 6 階約 1 秒） |
| `fbm2d_at(X, Y, octaves, ..., seed, mode)` | 在任意座標（以「格」為單位）取 fBm → 透視、扭曲、極座標用 |
| `perlin2d(h, w, scale, seed)` / `perlin2d_at(X, Y, seed)` | 單層 Perlin |
| `warp2d(h, w, scale, strength=1, octaves=5, seed, aspect, mode)` | domain warping：流動 / 煙 / 大理石 / 銀河 |
| `fbm1d(n, scale, octaves=5, ..., seed, mode)` / `perlin1d_at(X, seed)` | 1D（稜線、岸線、樓頂剖面） |
| `smooth1d(a, sigma)` | 1D 高斯平滑 |

## 遮罩與筆刷（pixelcanvas.masks / brush）

| 函數 | 說明 |
|---|---|
| `shape_mask(h, w, draw_fn, aa=3)` | 用 Pillow ImageDraw 畫任意形狀，`draw_fn(d, s)` 裡所有座標乘 s（超取樣） |
| `polygon_mask(h, w, points)` / `circle_mask(h, w, cx, cy, r, feather=1)` / `rect_mask(h, w, x0, y0, x1, y1, feather=0)` / `band_mask(h, w, y0, y1, soft=0)` | 幾何遮罩 |
| `text_mask(h, w, text, xy, size=40, font_path=None, anchor="la")` | 文字遮罩（中文請給 `font_path`，例如 `C:/Windows/Fonts/msjh.ttc`） |
| `feather(mask, sigma)` | 柔化邊緣 |
| `stamp(mask, cx, cy, r, hardness=0.5, opacity=1)` | 在遮罩上蓋柔邊圓點（就地修改） |
| `stroke(mask, points, r, hardness, opacity, spacing=0.3, r_end=None, jitter=0, seed=0)` | 沿折線噴點成筆觸 |

## 濾鏡（pixelcanvas.filters，陣列進陣列出）

`blend(base, top, alpha, mode)`、`gaussian_blur(a, sigma)`（2D 或 3D 皆可）、`sample(img, X, Y)`、
`resize(arr, w, h)`、`radial_blur(img, center, strength, samples)`、`light_shafts(source_mask, center, ...)`、
`bloom`、`vignette`、`grain`、`levels`、`contrast`、`saturation`、`soft_clip`、`filmic`、`split_tone`。

## 高階配方（pixelcanvas.scenery）— 風景專用

`Scene(width, height, seed, log=None)`：`sc.cv` 畫布、`sc.X/Y/U/V` 座標、`sc.S` 縮放（H/1080）、
`sc.px(v)` 把 1080p 像素換成目前解析度、`sc.terrain` 不透光聯集、`sc.haze` 霧色圖、`sc.D` 到太陽距離。

| 設定 | 說明 |
|---|---|
| `sc.set_horizon(v)` | 眼平線（水面鏡射軸） |
| `sc.set_sun(x, y, light_dir=(dx, dy))` | 太陽位置（比例）與指向光源的方向 |
| `sc.set_haze(stops)` | 霧色隨高度的漸層（大氣透視目標色） |
| `sc.occlude(mask)` | 不透光物件加入 terrain（god_rays 才會被擋） |

| 配方 | 回傳 | 重點參數 |
|---|---|---|
| `sky(sc, stops)` | — | 垂直多段漸層 |
| `sun(sc, glows, radius=27, disc=(1.8,1.6,1.3), horizon_glow)` | 太陽盤遮罩 | glows = ((sigma px, 強度, 色), ...)；`disc=None` 不畫太陽盤（已落山 / 被雲遮，只留光暈）；`horizon_glow=None` 關掉地平線暖霧 |
| `cirrus(sc, band, scale=480, aspect=0.1, threshold, alpha, colors)` | alpha | 高空卷雲 |
| `cumulus(sc, band=(淡入起, 淡入迄, 淡出起, 淡出迄), coverage=0.56, softness=0.2, scale=340, warp=0.35, aspect=0.5, shade=(色, 色), lit=(色, 色), rim=0.4)` | 雲密度（可給 god_rays 當遮擋） | **coverage 越小雲越多**；softness 越大邊越蓬；`shade` / `lit` 都是 2-tuple =（遠離太陽, 靠近太陽）的陰影色 / 受光色；`rim` 是數字（雲邊亮邊強度，0 關掉） |
| `mountain_layer(sc, dict(base, amp, scale, seed, shadow, lit, fog, ...))` | 山遮罩 | `fog` 0~1 = 往 `sc.haze` 霧色混合的比例（**越大越淡、越像遠山**；近景 0~0.1、最遠層 0.7~0.8）；`base` 稜線中線高度比例、`amp` 起伏比例、`scale` 稜線噪聲格寬（1080p px）。選填：peaks=[(x, 高, 半寬)]、dips、snow（雪線比例）、forest、treeline、relief、side/rise/reach（近景坡）、fog_down、tree_px |
| `ridge(...)` / `mountain(...)` | 稜線 (W,) / 遮罩 | mountain_layer 的兩半 |
| `valley_floor(sc, top, fade, colors, fog_range)` | 遮罩 | 谷底草原 |
| `mist(sc, v_range, strength, scale, exclude)` / `haze_band(sc, ...)` | — | 霧帶 / 地平線薄霧 |
| `river_course(sc, start, meander, width_far, width_near, ...)` | (center, half, t1) | 蜿蜒河道（每列一值） |
| `water_mask(sc, center, half, t1)` | (water, e) | e = 到岸的帶號距離 |
| `water_surface(sc, water, e, t1, colors, fresnel, ripple, glitter, ...)` | — | 倒影 + 波紋 + 閃光（要在山與岸之後） |
| `river_bank(sc, water, e, t1)` | — | 岸邊礫石 |
| `pine_mask(sc, [(x, y_base, h, w), ...])` / `scatter_trees(sc, n, x_range, y_range, height, avoid)` | 遮罩 / 樹列表 | 松樹剪影 |
| `rim_light(sc, mask, color, offset=(3,-2), strength)` | — | 剪影受光邊 |
| `ground_edge(sc, y_frac, side, reach)` | 遮罩 | 底部角落暗地面 |
| `birds(sc, [(x, y, size px, tilt), ...])` | — | 飛鳥 |
| `god_rays(sc, disc, clouds, strength=0.24, ...)` | — | 體積光（被 terrain / 雲擋住） |
| `finish(sc, bloom, knee, contrast, saturation, split, vignette, grain)` | — | 標準後製；傳 None 可關掉某一項 |

## 工具

```bash
python tools/render.py <scene> [--scale 0.5] [--seed N] [--size WxH] [--out p.png] [--crop x0 y0 x1 y1]...
python tools/render.py --list
python tools/new_scene.py <name>
python tools/look.py stats|report|crop|grid|compare|diff ...
pixelcanvas info 圖.png / pixelcanvas preview 圖.png 小圖.jpg    # pip install -e . 之後
```
