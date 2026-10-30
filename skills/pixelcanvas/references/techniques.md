# 技巧庫：怎麼用數學畫出 X

每個技巧都是可以直接貼進場景 `render()` 的片段。以下 `python` 區塊都假設已經執行過這段開頭
（`tests/test_skill_docs.py` 會逐段實際執行，確保範例不會壞掉）：

```python
import numpy as np
from pixelcanvas import *

W, H = 1920, 1080
S = H / 1080                      # 像素常數以 1080p 為基準 → 乘 S
cv = Canvas(W, H)
X, Y = cv.xy()                    # 像素座標 (H, W)
U, V = cv.uv()                    # 0~1 座標
```

完整的例子在 `scenes/`：`mountains_sunset`（高階配方）、`night_sky`、`still_life`、`city_dusk`、`sea_sunset`。

---

## 1. 剪影：1D 剖面 → 遮罩（山、丘陵、樓群、樹林邊、浪頭）

每一欄一個 y 值，`Y` 大於它的地方就是物體。`+ 0.5` 再 `clamp01` = 1px 抗鋸齒。

```python
cv.gradient([(0, "#24324f"), (1, "#d9a77c")])
for base, amp, col, seed in [(0.62, 0.12, "#7d7a92", 1), (0.72, 0.10, "#4b4e66", 2), (0.85, 0.08, "#1d2030", 3)]:
    ridge = (base - amp * fbm1d(W, 0.3 * W, 6, seed=seed, mode="ridged")) * H   # ridged = 尖峰
    cv.paint(col, clamp01(Y - ridge[None, :] + 0.5))
```
- 越遠的層：越高（base 小）、越淡（顏色接近天空底部）、起伏越小。
- 丘陵用 `mode="plain"`；岩石山用 `"ridged"`；樓群改成階梯狀（見 `scenes/city_dusk.py`）。

## 2. 有體積的雲：密度場 + 往光源方向取樣

雲的亮面 = 「朝光源方向密度變薄」的地方。把密度往光源方向平移再相減就是簡易的自陰影。

```python
cv.gradient([(0, "#3a5f95"), (1, "#a9c4dd")])
n = warp2d(H, W, 0.35 * H, 0.5, seed=4, aspect=0.55) * 0.5 + 0.5
dens = smoothstep(0.42 + 0.25 * V, 0.75 + 0.25 * V, n)       # 閾值隨高度升高 → 上方雲多、下方少
lx, ly = 25 * S, -18 * S                                  # 指向光源（右上）的位移
shifted = gaussian_blur(sample(dens, X + lx, Y + ly), 4 * S)                          # 光源方向那一點的密度
lit = clamp01(1.0 - 1.6 * shifted + 0.6 * dens)           # 光源那側越薄 → 越亮
col = lerp("#6d7896", "#fff6ea", lit)                      # 陰影色 → 受光色
cv.paint(col, smoothstep(0.0, 0.5, dens))                  # 邊緣 smoothstep 寬一點 → 蓬鬆
```
- 雲像大理石 → 閾值區間太窄或 warp 太強；雲像棉花糖 → softness 太大。
- 風景直接用 `cumulus()`（三個尺度的受光、rim light 都做好了）。

## 3. 法線 → 光照（球、物體、浮雕、地形）

任何能寫出「高度場 h(x, y)」或「表面法線」的東西都能打光：
漫射 `max(n·L, 0)`、鏡面 `(r·v)^k`、環境光、反光。

```python
L = np.array([-0.5, -0.6, 0.62]); L /= np.linalg.norm(L)  # 指向光源（左上前方）
h = fbm2d(H, W, 0.15 * H, 5, seed=7, mode="ridged") * 0.15 * H                # 高度場（像素單位）
gy, gx = np.gradient(h)
n = np.stack([-gx, -gy, np.ones_like(h)], -1)
n /= np.linalg.norm(n, axis=-1, keepdims=True)             # 表面法線
ndl = np.clip(n @ L, 0, 1)
albedo = mix([(0, "#5b4a3c"), (1, "#c9b59a")], h / h.max())                    # 低處深、高處淺
cv.paint(albedo * (0.25 + 0.9 * ndl[..., None]))
```
- 球的法線直接算：`dx, dy = (X-cx)/r, (Y-cy)/r; dz = sqrt(1-dx²-dy²)`，完整例子 `scenes/still_life.py`。
- 物體要「坐在」地面上：落影貼著接觸點 + 一道窄的 AO（接觸陰影）。

## 4. 透視地面 / 水面：反推世界座標

地平線在 `hz`。畫面上每一列對應一個深度 `z ∝ 1 / (Y - hz)`；在世界座標上取噪聲，
遠處紋理自然變細變密。遠處要淡出，否則會有摩爾紋。

```python
hz = 0.5 * H
cv.gradient([(0, "#7fa6d6"), (0.5, "#e8eef2"), (1, "#e8eef2")])
dy = np.maximum(Y - hz, 0.5)
z = 0.5 * H / dy                                           # 畫面底部 z ≈ 1
wx = (X - W / 2) / H * z
tex = fbm2d_at(wx * 14, z * 14, 5, seed=3) * (1 - smoothstep(3, 20, z))       # 草地 / 沙地紋理（遠處淡出）
ground = lerp("#3f5a2c", "#8aa05a", tex * 0.5 + 0.5)
ground = lerp(ground, "#c8d6dc", smoothstep(1, 12, z))                         # 越遠越接近霧色
cv.paint(ground, smoothstep(hz - 0.5, hz + 0.5, Y))
```

## 5. 倒影（水面、濕地、玻璃）

以眼平線為軸鏡射「目前畫面」：`cv.sample(X + 擾動, 2*hz - Y + 擾動)`。
**要先把被倒映的東西畫完**。遠處幾乎全反射（菲涅耳）、近處透出深水色；閃光只在光源正下方。

```python
hz = 0.6 * H
cv.gradient([(0, "#1b2a4a"), (0.6, "#f0a070"), (1, "#f0a070")])
ridge = (0.6 - 0.2 * fbm1d(W, 0.25 * W, 6, seed=5, mode="ridged")) * H     # ridged 0~1 → 稜線一定在水面之上
cv.paint("#2b2440", clamp01(Y - ridge[None, :] + 0.5))
ripple = fbm2d(H, W, 0.3 * W, 4, seed=6, aspect=0.04)                         # 橫向拉長 → 水波
refl = cv.sample(X + ripple * 6 * S, 2 * hz - Y + ripple * 10 * S)
t = clamp01((Y - hz) / (H - hz))                                               # 0 遠 → 1 近
water = lerp(refl * 0.92, "#0f1a28", 0.15 + 0.6 * t ** 1.5)
cv.paint(water, smoothstep(hz - 0.5, hz + 0.5, Y))
```
- 風景河流直接用 `river_course` + `water_mask` + `water_surface`；海面見 `scenes/sea_sunset.py`。

## 6. 撒點：星星、雪、螢火蟲、散景、灰塵

隨機座標 + **冪次分布亮度**（大多數暗、少數亮）→ `np.add.at` 撒到遮罩 → 微模糊。
不要用 Python 迴圈畫幾千個點；只有少數「亮的」再逐一加 glow。

```python
cv.gradient([(0, "#03050d"), (1, "#1a2a4e")])
rng = np.random.default_rng(1)
n = 2500
px, py = rng.uniform(0, W, n), rng.uniform(0, H, n)
b = rng.random(n) ** 6 * 1.5 + 0.1
pts = np.zeros((H, W), F32)
np.add.at(pts, (py.astype(int), px.astype(int)), b)
cv.paint("#ffffff", gaussian_blur(pts, 1.2 * S) * 12 * S * S, "add")         # 模糊半徑 ∝ S → 亮度補 S²
for i in np.argsort(-b)[:10]:
    cv.glow(px[i], py[i], 6 * S, "#cfe0ff", strength=0.4)
```
- 雪：亮度均勻、半徑大一點、近大遠小；散景：大而柔的圓（`stamp` hardness 0.8）、低不透明度、`add` 模式。

## 7. 格點圖樣：窗戶、磚牆、地磚、棋盤

整數格 `gx = X // cell`；同一格用同一個隨機值（`perlin2d_at(gx*a, gy*b)` 或雜湊），
格內用 `X % cell` 留邊框。

```python
cv.paint("#2a2230")
cell = 24 * S
gx, gy = X // cell, Y // cell
on = perlin2d_at(gx * 0.71, gy * 0.37, seed=2) > 0.2                          # 哪些窗亮著
pane = (X % cell > cell * 0.3) & (Y % cell > cell * 0.35)                       # 窗框
glow = gaussian_blur((on & pane).astype(F32), 2 * S)
cv.paint("#ffcf7a", (on & pane) * 0.9).paint("#ff9a40", glow * 0.4, "add")
```
- 磚牆：奇數列 `X + cell/2` 錯位；每塊磚亮度 ±10% 隨機 + 細 fBm 表面。

## 8. 光：光暈、體積光、rim light

```python
cv.gradient([(0, "#0e1424"), (1, "#3b2e3e")])
sx, sy = 0.7 * W, 0.3 * H
cv.glow(sx, sy, 0.25 * H, "#ff9a55", strength=0.35)          # 大光暈（大氣散射）
cv.glow(sx, sy, 0.04 * H, "#fff1d6", strength=0.9)           # 小光暈
disc = circle_mask(H, W, sx, sy, 0.025 * H, feather=1.5 * S)
cv.paint((1.8, 1.6, 1.3), disc)                               # HDR 光源本體 > 1
occl = clamp01(Y - ((0.75 - 0.1 * fbm1d(W, 0.2 * W, seed=9)) * H)[None, :] + 0.5)
cv.paint("#0a0c12", occl)
shafts = light_shafts(disc * (1 - occl) + 0.15 * (1 - occl) * np.exp(-((X - sx) ** 2 + (Y - sy) ** 2) / (0.2 * H) ** 2),
                      (sx, sy), samples=32, decay=0.96, length=0.9)
cv.paint("#ffc98a", shafts * 0.6, "add")
cv.bloom(0.85, 0.02 * H, 0.3).soft_clip(0.85)
```
- rim light（剪影朝光那側的亮邊）：`edge = mask - np.roll(np.roll(mask, dy, 0), dx, 1)` 取正值上暖色；
  風景用 `rim_light(sc, mask)`。

## 9. 任意形狀與筆觸：Pillow 超取樣遮罩、筆刷

```python
cv.gradient([(0, "#bcd4e6"), (1, "#eef3f6")])
def house(d, s):                                              # 所有座標乘 s（超取樣倍率）
    x0, y0 = 0.4 * W, 0.55 * H
    w, h = 0.2 * W, 0.2 * H
    d.rectangle([x0 * s, y0 * s, (x0 + w) * s, (y0 + h) * s], fill=255)
    d.polygon([(x0 - 0.02 * W) * s, y0 * s, (x0 + w / 2) * s, (y0 - 0.12 * H) * s, (x0 + w + 0.02 * W) * s, y0 * s], fill=255)
cv.paint("#6b4b3a", shape_mask(H, W, house))
grass = np.zeros((H, W), F32)
rng = np.random.default_rng(3)
for _ in range(300):                                          # 草葉：彎曲的細筆觸
    x, y = rng.uniform(0, W), rng.uniform(0.74 * H, H)
    hgt = rng.uniform(20, 60) * S
    lean = rng.uniform(-15, 15) * S
    stroke(grass, [(x, y), (x + lean * 0.4, y - hgt * 0.6), (x + lean, y - hgt)], 2.0 * S, r_end=0.5 * S, hardness=0.6)
cv.paint("#3d5a2a", band_mask(H, W, 0.75 * H, 2 * H, soft=4 * S))
cv.paint("#557a35", grass)
```
- 大量物件（上千棵樹）用一次 `shape_mask` 在 draw_fn 裡畫完，比逐個建遮罩快很多（`scenery.pine_mask`）。

## 10. 文字與海報

```python
cv.gradient([(0, "#1d1b3a"), (1, "#e2566e")], axis="x")
flow = warp2d(H, W, 0.3 * H, 1.2, seed=8) * 0.5 + 0.5
cv.paint("#ffb86b", smoothstep(0.55, 0.75, flow) * 0.5, "screen")
title = text_mask(H, W, "PIXEL", (W / 2, H / 2), size=int(0.22 * H), anchor="mm")
cv.paint("#000000", feather(title, 8 * S) * 0.5)               # 先畫柔和陰影
cv.paint("#fff6e8", title)
cv.vignette(0.3).grain(0.015)
```
- 中文字要給字型：`text_mask(..., font_path="C:/Windows/Fonts/msjh.ttc")`（macOS：`/System/Library/Fonts/PingFang.ttc`；
  Linux：`fc-list :lang=zh` 找）。

## 11. 修圖（照片 → 調整）

```python
from PIL import Image
Image.fromarray((np.random.default_rng(0).random((H, W, 3)) * 255).astype(np.uint8)).save("photo_demo.jpg")  # 示範用
cv = Canvas.load("photo_demo.jpg")                             # ← 換成使用者的照片
X, Y = cv.xy(); U, V = cv.uv()
cv.levels(0.03, 0.97, 1.05)                                    # 黑白場 + gamma
sky = smoothstep(0.55, 0.85, luma(cv.buf)) * (1 - smoothstep(0.3, 0.6, V))    # 亮且在上方 → 天空遮罩
cv.paint(cv.buf * 0.85, sky)                                   # 局部壓暗天空
cv.split_tone("#3d4a66", "#f2c58a", 0.08).saturation(1.08).vignette(0.3).grain(0.01)
cv.save("photo_demo_edit.jpg")
```
- 局部調整 = 先做遮罩（亮度、顏色、位置、`shape_mask` 手繪），再 `cv.paint(調整後的 buf, mask)`。

## 12. 雪地：藍色陰影 + 暖色受光 + 細微閃點

雪幾乎是白的，**層次全靠光**：受光面帶光源色（夕陽 → 粉橙），背光面反射天空（偏藍紫），
起伏用低頻、柔和的噪聲（雪會把細節蓋掉，別用高頻 / ridged），閃點極少、只在受光面。

```python
cv.gradient([(0, "#3a3f78"), (0.5, "#c99ab8"), (1, "#c99ab8")])
top = 0.55 * H
dy = np.maximum(Y - top, 0.5)
z = 0.45 * H / dy                                          # 透視深度（見第 4 節）
wx = (X - W / 2) / H * z
h = fbm2d_at(wx * 1.0, z * 0.8, 4, gain=0.4, seed=11) * (1 - smoothstep(3, 14, z))  # 低頻、gain 小 → 柔和雪丘
slope = np.gradient(h, axis=1) * dy / (0.02 * H)            # 左右坡向；光從右邊來 → 正值受光
lit = clamp01(0.55 + 1.6 * slope)
snow = lerp("#7d7fb8", "#ffe6e0", lit)                       # 背光 = 天空的藍紫、受光 = 夕陽粉
snow = lerp(snow, "#d9b4c8", smoothstep(3, 14, z) * 0.7)    # 遠處融入地平線霧色
rng = np.random.default_rng(5)
spark = (rng.random((H, W)) > 0.9995) * smoothstep(0.6, 0.9, lit)             # 稀疏閃點，只在受光面
snow = snow + rgb((1.2, 1.1, 1.0)) * gaussian_blur(spark.astype(F32), 0.7 * S)[..., None] * 3
cv.paint(snow, smoothstep(top - 0.5, top + 0.5, Y))
```
- 樹 / 石頭在雪地上的影子：沿光向量反方向拉長、顏色用藍紫（不是黑）；樹根處加一圈窄而深的 AO。
- 不要在雪地上用和天空相同的大斑塊噪聲（會像水面倒影或雲的影子）。

---

## 配色表（調色盤起點，(V 比例, 顏色)）

| 氣氛 | 天空漸層 | 霧色 / 遠景 | 近景 / 剪影 | 點綴 |
|---|---|---|---|---|
| 黃昏暖陽 | `#0c1c3e` → `#3f6f9f` → `#e6a86c` → `#f3c89c` | `#e2b48f` | `#10191a` | 太陽 `(1.8,1.6,1.3)` |
| 藍調時刻 | `#0b1230` → `#2c3f78` → `#8a7fb0` → `#e7a3a0` | `#9a90b8` | `#141626` | 窗燈 `#ffcf7a` |
| 星夜 | `#02040c` → `#0b1838` → `#23385f` | `#141d33` | `#04060c` | 星 `#ffffff`、月 `#f2efe4` |
| 晴天正午 | `#2f6fc4` → `#7fb2e6` → `#d6e8f5` | `#c9dbe8` | `#2f4a2a` | 雲 `#ffffff` / 雲影 `#a9b8cc` |
| 霧晨 | `#9fb0bd` → `#d7dcd9` → `#eae6dc` | `#dcdcd4` | `#4a5550` | 低對比，飽和 0.8 |
| 秋日 | `#4f7fb5` → `#d9c3a0` | `#d8b892` | `#3a2416` | 楓紅 `#c2502a`、金 `#e0a43a` |
| 雪地 | `#8fa9c8` → `#e8eef4` | `#dfe6ee` | `#2a3442` | 陰影偏藍 `#9fb3cf` |
| 水下 | `#7fd0e0` → `#1c6c8c` → `#062438` | `#1c5a74` | `#03121c` | 光束 `#bff4ff` |
| 霓虹 | `#0a0618` → `#2a0f4a` → `#5a1a6e` | `#3a1a5a` | `#07040f` | `#ff3cac` `#2bd2ff` `#ffe66d` |

原則：陰影偏冷、受光偏暖（或反之），兩者別同色系；飽和度留給焦點；大面積用低飽和。
