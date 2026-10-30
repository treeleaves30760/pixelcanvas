---
name: pixelcanvas
description: >-
  Draw good-looking images with pure code (numpy + Pillow) when you cannot generate images natively —
  landscapes, skies, night scenes, sea, cities, still lifes, posters, wallpapers — and edit photos.
  Iterative loop: write a scene script → render a draft → inspect it (view the preview, or read a text
  report if you have no vision) → fix 2-4 parameters → re-render. Use whenever the user asks to draw,
  paint, render, illustrate or "generate a picture/wallpaper", or to retouch an image, and no image model
  is available. 觸發詞：畫一張、幫我畫、產生圖片、桌布、插畫、風景圖、修圖、調色。
---

# pixelcanvas：用程式 + 數學畫圖

沒有素材、沒有生成模型：每一個像素都是噪聲、遮罩、混合算出來的。你（模型）負責
**構圖 → 寫程式 → 渲染 → 檢查 → 修正**，一張好圖通常要 4~7 輪；第一版幾乎一定過曝、太平、太乾淨。
**不要寫完一次就交出去。**

## 0. 準備

1. 找專案根目錄（有 `pixelcanvas/`、`tools/render.py`、`scenes/` 的資料夾）：目前目錄 → 環境變數
   `PIXELCANVAS_HOME` → 本檔底部「本機安裝位置」。都沒有就安裝：
   ```bash
   git clone https://github.com/treeleaves30760/pixelcanvas.git && cd pixelcanvas
   pip install -r requirements.txt          # numpy、Pillow（scipy 可選，模糊 / 取樣更快）
   ```
2. 之後所有指令都在專案根目錄執行。`python -m pytest -q` 應在 2 秒內全過。

## 1. 先構圖（寫在場景檔開頭的註解）

動筆前回答五件事，寫成 docstring：
- **畫幅**：預設 1920x1080；直幅 / 手機桌布 1080x1920。
- **眼平線**在哪（V 比例，0 = 頂、1 = 底）：天空為主 → 0.6~0.75；地面為主 → 0.35~0.5。
- **光源**位置與方向（太陽 / 月亮 / 窗戶），決定所有受光面、影子、rim light 的方向。
- **遠 / 中 / 近**各有什麼；**焦點**放在三分線附近，前景只框住畫面、不搶戲。
- **調色盤**：3~5 個主色 + 霧色（見 [references/techniques.md](references/techniques.md) 的配色表）。

## 2. 建場景、渲染草稿

```bash
python tools/new_scene.py my_scene            # 從範本建立 scenes/my_scene.py
python tools/render.py my_scene --scale 0.5   # 草稿 960x540，幾秒 → outputs/my_scene_x0.5_preview.jpg
```

場景檔只要提供 `render(width, height, seed, log) -> Canvas` 和 `DEFAULT_SIZE`。
**挑最接近的範例複製來改**，比從零寫可靠：

| 要畫的東西 | 從這個開始 | 示範的技巧 |
|---|---|---|
| 山、河、森林、雲、夕陽 | `scenes/mountains_sunset.py` | 高階配方 `pixelcanvas.scenery`（Scene + sky/sun/cumulus/mountain_layer/water_surface/god_rays/finish） |
| 星空、夜景、月亮 | `scenes/night_sky.py` | 散點冪次分布、domain warp 銀河、1D 稜線剪影 |
| 物體、靜物、產品圖 | `scenes/still_life.py` | 法線 → 漫射 / 鏡面 / 落影 / AO |
| 城市、建築、幾何 | `scenes/city_dusk.py` | 高度剖面、格點圖樣（窗戶）、分層霧色 |
| 海、湖、水面 | `scenes/sea_sunset.py` | 透視反推世界座標、鏡射倒影、菲涅耳、波面閃光 |
| 修照片 | `Canvas.load("photo.jpg")` | 濾鏡與遮罩，見 techniques.md「修圖」 |

API 速查：[references/api.md](references/api.md)。技巧與配色：[references/techniques.md](references/techniques.md)。

## 3. 檢查（最重要的一步）

**有視覺能力**（能讀圖片）：直接打開 `outputs/<name>_x0.5_preview.jpg`。

**沒有視覺能力**：用文字報告「看」圖，它會給出九宮格平均色 / 亮度 / 對比 / 質感、主色佔比、
最亮區位置、過曝 / 死黑比例、ASCII 亮度圖：
```bash
python tools/look.py report outputs/my_scene_x0.5.png
```
報告還有**垂直剖面**（每段亮度 / 色相 → 地平線、天地分界在哪）、**水平剖面**、**各欄最強的上下分界**
（地平線 / 稜線 / 物體頂端的 V 位置與明暗差 Δ；Δ < 0.05 代表兩層幾乎分不開）、每格的**色相角度**（判斷「粉紫」「暖橙」）。
把報告和你的構圖意圖逐條對照（例：「太陽應在 (0.6, 0.55)」→ 看最亮區重心；「近景要暗」→ 看 r2 列亮度；
「山要和天空分開」→ 看那幾欄在山稜高度的 Δ）。

要看某個區域（例如只看雪地、不含樹）：先裁切再報告——
```bash
python tools/look.py crop outputs/my_scene_x0.5.png 300 380 660 470 -o outputs/tmp_snow.jpg
python tools/look.py report outputs/tmp_snow.jpg
```
（座標是該張圖的像素；草稿是 960x540。`outputs/` 是暫存區，crop / tile 檔可以隨時刪。）

兩種情況都照 [references/review.md](references/review.md) 的清單檢查，**寫下 2~4 個最嚴重的問題**，
再到下一步。常見第一輪問題：過曝（死白 > 3%）、層與層分不開、雲像大理石、近景太平、邊緣鋸齒。

## 4. 修正 → 重渲染

- 一次只改 2~4 個參數，seed 固定（結果才可比較）。
- 改前改後比較：`python tools/look.py compare A.png B.png`（並排）或 `diff A.png B.png`（差異熱圖 + 數字）。
- 回到第 3 步，直到清單全過。

## 5. 全尺寸、細節、交付

```bash
python tools/render.py my_scene                              # 1920x1080（或 --size 2560x1440）
python tools/render.py my_scene --crop 700 300 1500 750      # 全解析度局部裁切，看紋理與邊緣
python tools/look.py grid outputs/my_scene.png               # 3x3 九塊全解析度小圖
```
檢查細節（紋理像不像毛皮、邊緣是否太硬、閃光是否太密），滿意後把 PNG + preview 複製到 `gallery/`，
回報時附上 preview 路徑、說明改了什麼和為什麼。不必長篇描述圖的內容。

## 繪圖心法（違反這些就不會好看）

1. **先遮罩 / 密度場，再顏色**：先算「哪裡有東西、多厚」（float32 (H, W) 0~1），再決定受光色 / 陰影色 / 霧色，
   最後 `cv.paint(color, mask)`。顏色可以是整張 (H, W, 3) 顏色圖。
2. **由遠到近畫**：天空 → 遠景 → 中景 → 水面（倒影取樣「目前畫面」，所以要在岸與山之後）→ 前景 → 光線 → 後製。
3. **光源一致**：所有物件的亮面朝同一個光向量；低角度太陽 → 雲底受光、剪影邊緣加 rim light。
4. **大氣透視**：越遠越淡、越接近霧色、對比越低；近景顏色深、對比高。每層之間要分得開。
5. **HDR 再壓回**：光源可以 > 1，最後 `bloom` + `soft_clip`（或 `finish()`）。但要保留紋理的表面（月面、雲）別 > 1。
6. **解析度無關**：位置用比例（0~1），像素常數以 1080p 為基準乘上 `S = H / 1080`（或 `sc.px()`），
   草稿和全尺寸才會長一樣。
7. **抗鋸齒**：剪影用 `clamp01(Y - ridge + 0.5)`；圓用 `smoothstep` 帶 1~2px 過渡；複雜形狀用 `shape_mask`（超取樣）。
8. **質感尺度**：噪聲 scale 太小 → 毛皮感 / 雜訊；太大 → 塑膠感。圓潤物（森林、雲）用 plain / billow fBm，
   岩石稜線用 ridged。相鄰物件用不同 seed，避免重複感。
9. **全部向量化**：不要逐像素 Python 迴圈；大量小物件（樹、星星）用 `np.add.at` 撒點或 Pillow 畫到超取樣遮罩。

## 本機安裝位置

（由 `tools/install_skill.py` 自動填入；若此處空白，依第 0 步尋找或 clone 專案。）
