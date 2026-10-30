# pixelcanvas — 讓 Claude 用像素級運算畫圖的工具專案

沒有任何素材圖：天空、雲、山、雪、森林、霧、河流倒影、樹木、光線全部由噪聲函數、遮罩與混合算出來。
Claude 的工作方式是 **寫場景腳本 → 渲染 → 看預覽圖 → 修正參數 → 再渲染**，直到滿意。

![mountains_sunset](gallery/mountains_sunset_preview.jpg)

## 安裝與第一次渲染

```bash
pip install -r requirements.txt        # numpy、Pillow（scipy 可選，讓模糊 / 取樣更快）
python tools/render.py mountains_sunset --scale 0.5   # 草稿 960x540，約 6 秒
python tools/render.py mountains_sunset               # 全尺寸 1920x1080，約 25 秒
python -m pytest -q                                   # 測試
```

輸出在 `outputs/`（PNG + `_preview.jpg`）。

## 之後怎麼用

把整個資料夾交給 Claude（連結資料夾或上傳），然後直接說：

- 「畫一張 2560x1440 的雪夜森林」
- 「把 mountains_sunset 改成日出、河改成湖」
- 「幫這張照片加暖色暈影和顆粒」

Claude 會照 `CLAUDE.md` 裡的流程做：建場景 → 草稿 → 看圖 → 修 → 全尺寸 → 交付。

## 專案結構

```
pixelcanvas/            套件
  color.py              rgb / lerp / mix / hsv / smoothstep ...
  noise.py              perlin2d / fbm2d / warp2d / fbm1d（ridged、billow、domain warp）
  masks.py              shape_mask / polygon / circle / rect / band / text / feather
  brush.py              stamp / stroke
  filters.py            blend（8 種混合模式）/ gaussian_blur / sample / radial_blur / light_shafts / bloom /
                        vignette / grain / levels / contrast / saturation / soft_clip / filmic / split_tone
  canvas.py             Canvas：建立 / 載入 / 讀寫像素 / 上色 / 漸層 / 光暈 / 貼圖 / 取樣 / 後製 / 存檔
  scenery.py            高階景物配方：Scene、sky、sun、cirrus、cumulus、ridge、mountain、mountain_layer、
                        valley_floor、mist、haze_band、river_course、water_mask、water_surface、river_bank、
                        pine_mask、scatter_trees、rim_light、ground_edge、birds、god_rays、finish
  cli.py                pixelcanvas info / preview
scenes/                 場景檔（每張圖一個）：mountains_sunset.py、_template.py
tools/                  render.py（渲染）、look.py（看圖輔助）、new_scene.py（建場景）
tests/                  pytest
gallery/                定稿
CLAUDE.md               給 Claude 的工作流程與檢查清單
```

## 兩層 API

**低階**（`from pixelcanvas import *`）— 任何東西都能從這層畫出來：

```python
from pixelcanvas import *
cv = Canvas(640, 360)
cv.gradient([(0, '#0b1a3a'), (1, '#f6c28f')])          # 天空漸層
X, Y = cv.xy()
cv.glow(400, 200, 60, '#fff0c0', strength=1.2)          # 光暈 (HDR)
ridge = 220 - 60 * fbm1d(640, 120, seed=1)              # 山稜線：每一欄的 y
cv.paint('#223344', clamp01(Y - ridge[None, :] + 0.5))  # 稜線以下上色 = 山
cv.put(10, 10, '#ff0000')                               # 單一像素
cv.soft_clip(0.8).save('demo.png', preview='demo_small.jpg')
```

**高階配方**（`from pixelcanvas.scenery import *`）— 快速組一張風景；所有像素單位以 1080p 為基準，
內部依解析度縮放，所以 `--scale 0.5` 的草稿和全尺寸長得一樣：

```python
sc = Scene(1920, 1080, seed=7)
sc.set_horizon(0.62); sc.set_sun(0.6, 0.56, light_dir=(0.9, -0.4)); sc.set_haze(HAZE)
sky(sc, SKY); disc = sun(sc); clouds = cumulus(sc)
mountain_layer(sc, dict(base=0.6, amp=0.2, scale=500, seed=11, fog=0.7, shadow='#6e6e9d', lit='#c5b0bd', snow=0.5))
center, half, t1 = river_course(sc); water, e = water_mask(sc, center, half, t1); water_surface(sc, water, e, t1)
god_rays(sc, disc, clouds); finish(sc)
sc.cv.save('out.png', preview='out_preview.jpg')
```

完整範例見 `scenes/mountains_sunset.py`；要畫全新的東西（城市、海、星空…）就用低階 API 寫新配方，
放進 `scenery.py` 以便重用。

## 慣例

座標 x 向右、y 向下（`arr[y, x]`）；遮罩 float32 (H, W) 0~1；顏色 `'#rrggbb'` / `(r, g, b)` 0~1 / `(255, 128, 0)` /
整張 `(H, W, 3)`，允許 HDR > 1（最後 `soft_clip` / `bloom`）；順序 天空 → 遠 → 近 → 水 → 前景 → 光線 → 後製。
