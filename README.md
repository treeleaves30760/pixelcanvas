# pixelcanvas — 讓 AI 用像素級運算畫圖

沒有任何素材圖、也不需要圖像生成模型：天空、雲、山、雪、森林、霧、海面倒影、城市、星空、光線，
全部由噪聲函數、遮罩與混合算出來。AI（或你）的工作方式是
**寫場景腳本 → 渲染 → 看預覽圖（或讀文字報告）→ 修正參數 → 再渲染**，直到滿意。

| | |
|---|---|
| ![mountains_sunset](gallery/mountains_sunset_preview.jpg) | ![sea_sunset](gallery/sea_sunset_preview.jpg) |
| ![night_sky](gallery/night_sky_preview.jpg) | ![city_dusk](gallery/city_dusk_preview.jpg) |
| ![still_life](gallery/still_life_preview.jpg) | |

## 安裝與第一次渲染

```bash
pip install -r requirements.txt                       # numpy、Pillow（scipy 可選，讓模糊 / 取樣更快）
python tools/render.py --list                         # 列出場景
python tools/render.py mountains_sunset --scale 0.5   # 草稿 960x540，約 6 秒
python tools/render.py mountains_sunset               # 全尺寸 1920x1080，約 25 秒
python -m pytest -q                                   # 測試（< 2 秒）
```

輸出在 `outputs/`（PNG + `_preview.jpg`）。

## 給 AI 用：pixelcanvas skill

`skills/pixelcanvas/` 是一個標準格式的 Agent Skill（`SKILL.md` + `references/`），
教「不能畫圖」的模型用這套工具畫出好看的圖：構圖 → 寫場景 → 草稿 → 檢查 → 修正 → 全尺寸。

- [`SKILL.md`](skills/pixelcanvas/SKILL.md)：完整流程、該從哪個範例場景開始、繪圖心法
- [`references/api.md`](skills/pixelcanvas/references/api.md)：API 速查
- [`references/techniques.md`](skills/pixelcanvas/references/techniques.md)：技巧庫（剪影、體積雲、法線光照、透視地面、
  倒影、撒點、格點圖樣、體積光、Pillow 形狀與筆刷、文字海報、修圖）+ 配色表
- [`references/review.md`](skills/pixelcanvas/references/review.md)：看圖檢查清單、報告數字怎麼讀、症狀 → 修法

**沒有視覺能力的模型也能用**：`python tools/look.py report 圖.png` 會把圖轉成文字
（九宮格平均色 / 亮度 / 對比 / 質感、主色佔比、最亮區位置、過曝 / 死黑比例、ASCII 亮度圖），
讓模型「讀」圖並和構圖意圖比對。

### 安裝到各種工具

| 工具 | 做法 |
|---|---|
| Claude Code | 在本專案裡直接可用（`CLAUDE.md`）；全域安裝：`python tools/install_skill.py` → `~/.claude/skills/pixelcanvas` |
| Codex CLI | `python tools/install_skill.py --tool codex` → `~/.codex/skills/pixelcanvas`；在本專案裡也會讀 `AGENTS.md` |
| 其他支援 SKILL.md 的工具 | `python tools/install_skill.py --dest <該工具的 skills 目錄>` |
| Cursor / Gemini CLI / Copilot / Aider… | 在本專案裡開啟即可（讀 `AGENTS.md`）；或把 `SKILL.md` 內容放進該工具的規則檔 |
| 一般聊天模型（有程式執行環境） | 上傳整個專案 zip，並貼上 `SKILL.md` 當指示 |

安裝腳本會把本機專案路徑寫進安裝後的 `SKILL.md`，所以在任何資料夾裡說「畫一張 ○○」模型都找得到這個專案。

## 之後怎麼用

把專案交給 AI，然後直接說：

- 「畫一張 2560x1440 的雪夜森林」
- 「把 mountains_sunset 改成日出、河改成湖」
- 「畫一張手機桌布：霓虹城市下雨的夜晚」
- 「幫這張照片加暖色暈影和顆粒」

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
scenes/                 場景檔（每張圖一個）
  mountains_sunset.py   黃昏山川：高階配方的完整示範
  night_sky.py          星空：散點、銀河、月亮、剪影
  still_life.py         靜物：法線光照、落影、AO
  city_dusk.py          城市：高度剖面、窗戶格點、分層霧
  sea_sunset.py         海面：透視、倒影、菲涅耳、閃光
  _template.py          新場景範本
skills/pixelcanvas/     給 AI 的 skill（SKILL.md + references/）
tools/
  render.py             渲染（--scale 草稿 / --size / --seed / --crop）
  look.py               看圖輔助：stats / report（文字報告）/ crop / grid / compare / diff
  new_scene.py          從範本建立新場景
  install_skill.py      把 skill 安裝到 Claude Code / Codex / 其他工具
tests/                  pytest（套件、所有場景、skill 文件裡的範例程式）
gallery/                定稿
CLAUDE.md / AGENTS.md   給 AI 助理的專案說明
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

完整範例見 `scenes/`；要畫全新的東西就用低階 API 寫新配方（技巧見 `skills/pixelcanvas/references/techniques.md`），
通用的放進 `scenery.py` 以便重用。

## 慣例

座標 x 向右、y 向下（`arr[y, x]`）；遮罩 float32 (H, W) 0~1；顏色 `'#rrggbb'` / `(r, g, b)` 0~1 / `(255, 128, 0)` /
整張 `(H, W, 3)`，允許 HDR > 1（最後 `soft_clip` / `bloom`）；順序 天空 → 遠 → 近 → 水 → 前景 → 光線 → 後製。
