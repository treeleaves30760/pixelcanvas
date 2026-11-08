# CLAUDE.md — 這個專案怎麼用（給 Claude 的工作說明）

這是一個讓 Claude 用「程式 + 數學 + 看圖」來畫圖的工具專案。沒有任何素材：
每張圖都是由噪聲、遮罩、混合算出來的像素。使用者會不斷要求畫各種東西，
請用下面的流程，而不是一次寫完就交出去。

## 專案地圖

```
pixelcanvas/            套件（低階 API）：Canvas 畫布、noise 噪聲、masks 遮罩 / 曲線、brush 筆刷、filters 濾鏡、color 顏色、
                        text 文字排版（字型、換行、自動縮放、描邊 / 陰影 / 光暈）、sdf 距離場形狀、vision 看圖用檢視
pixelcanvas/scenery.py  高階「景物配方」：Scene + sky/sun/cirrus/cumulus/ridge/mountain/mountain_layer/valley_floor/
                        mist/haze_band/river_course/water_mask/water_surface/river_bank/pine_mask/scatter_trees/
                        rim_light/ground_edge/birds/god_rays/finish
scenes/                 每張圖一個檔：render(width, height, seed, log) -> Canvas；_template.py 是範本
                        範例：mountains_sunset（高階配方）/ night_sky / still_life / city_dusk / sea_sunset（低階 API）/
                        neon_poster（文字 + SDF + 版面）/ aurora_lake（極光 + 湖面倒影 + 標題橫幅，README 首圖）；_template_poster.py 是設計範本
skills/pixelcanvas/     給 AI 的 skill（英文）：SKILL.md + references/（api / techniques 技巧與配色 / typography 文字與版面 /
                        review 看圖流程與檢查清單）
tools/render.py         渲染（--scale 0.5 草稿 / 全尺寸 / --crop 裁切 / --check 診斷圖 / --stages 逐步驟快照 /
                        --set 改常數 / --sweep、--seeds 變體對照表）
tools/look.py           看圖輔助：check（四合一診斷圖）/ guides（座標格線）/ zoom（放大）/ values（明度圖）/ edges /
                        sheet（對照表）/ critique（自動警告）/ probe（取色）/ stats / report（文字報告）/ crop / grid /
                        compare / diff
tools/fonts.py          列出字型、輸出字型樣張（--sheet）
tools/new_scene.py      從範本建立新場景（--poster 用設計範本）
tools/install_skill.py  把 skill 安裝到 ~/.claude/skills（或 --tool codex / --dest 其他工具）
tests/                  pytest（python -m pytest -q，約 5 秒；也會執行 techniques.md / typography.md 的範例程式、渲染所有場景）
AGENTS.md               給其他 AI 助理的入口（指向 skill）
outputs/                渲染輸出（不進 git）；gallery/ 放定稿；docs/ 放 README 用圖
README.md / README-zh.md 英文 / 中文說明
```

## 畫一張新圖的標準流程（渲染 → 看圖 → 修正，反覆）

1. **先想構圖再寫碼**：眼平線在哪（`set_horizon`）、光源在哪（`set_sun` + `light_dir`）、遠中近各有什麼、
   焦點是什麼。把這些寫成場景檔開頭的註解。
2. `python tools/new_scene.py <name>` → 編輯 `scenes/<name>.py`；或複製最接近的範例場景來改。先用配方組出大結構，
   再用低階 API 補特殊物件（技巧見 `skills/pixelcanvas/references/techniques.md`）。
3. **草稿**：`python tools/render.py <name> --scale 0.5 --check`（960x540，幾秒；另存四合一診斷圖並印出自動警告）。
4. **看圖**：用 Read 看 `outputs/<name>_x0.5_preview.jpg` 與 `_check.jpg`；位置用 `look.py guides` 讀座標、細節用
   `look.py zoom` 放大、構圖用 `look.py values`；不知道哪一步壞掉就 `--stages`。搭配 `look.py critique` / `report` 的數字。
   逐項檢查下面的清單，寫下 2~4 個最嚴重的問題（附座標）。
5. **修正**：一次只改 2~4 個參數，重跑草稿、再看。不確定哪個值好就 `--sweep 名稱 v1 v2 v3` 一次比較；
   不改檔案試改動用 `--set 名稱=值`；改前改後用 `tools/look.py sheet A B` / `compare A B` 並排。
6. **全尺寸**：`python tools/render.py <name> --crop x0 y0 x1 y1 ...`，用 Read 看 preview 與裁切圖，
   或 `tools/look.py grid` 看九宮格，檢查細節（紋理是否像毛皮、邊緣是否太硬、閃光是否太密）。
7. 滿意後把 PNG + preview 複製到 `gallery/`，交付給使用者。全程保持 seed 固定，結果才可重現。

經驗上一張圖要 4~7 輪；第一版通常過曝、雲像大理石、近景太平。這是正常的，不要跳過看圖。

## 看圖檢查清單

（完整版 + 症狀 → 修法對照表：`skills/pixelcanvas/references/review.md`）

- **曝光**：有沒有大片死白（太陽光暈、天空）或死黑（前景）？`cv.stats()` 的「>1 比例」> 3% 通常就太亮。
  太陽要像「一顆亮點 + 光暈」，不是一大坨白；周圍天空要比太陽暗。
- **大氣透視**：越遠越淡、越接近霧色、對比越低；近景對比高、顏色深。層與層之間要分得開。
- **光源一致**：山、雲、樹的受光面都朝同一邊；低角度太陽 → 雲的底部受光、剪影有 rim light。
- **質感尺度**：噪聲 scale 太小 → 毛皮感；太大 → 像塑膠。森林用 plain fBm（圓潤），岩石用 ridged（稜線）。
- **邊緣**：剪影要有 1px 抗鋸齒（`clamp01(rel + 0.5)`）；雲邊要蓬鬆（閾值 smoothstep 寬一點）。
- **水**：倒影以眼平線為軸鏡射（`water_surface` 已處理）、遠處幾乎全反射、近處透出深色水色；閃爍只在太陽正下方。
- **重複感**：同一個噪聲別重複用在相鄰物件；每層用不同 seed offset。
- **構圖**：焦點（主峰、太陽、河的消失點）是否在三分線附近；前景是否框住畫面但不搶戲。

## 技術慣例

- 座標 x 向右、y 向下；陣列 `arr[y, x]`；位置一律用比例（0~1），像素常數以 1080p 為基準並用 `sc.px()` 換算，
  草稿與全尺寸才會長一樣。
- 遮罩 float32 (H, W)。上色 = `sc.cv.paint(color, mask, mode)`；color 可以是整張 (H, W, 3) 顏色圖。
- 順序：天空 → 遠 → 近 → 水面（`water_surface` 取樣「目前畫面」當倒影，所以岸與山要先畫）→ 前景 → 體積光 → 後製。
- 不透光物件要 `sc.occlude(mask)`，`god_rays` 才會被擋住。
- HDR：光源可以 > 1，交給 `finish()` 的 bloom + soft_clip。
- 效能：1920x1080 一次 6 階 fbm2d 約 1 秒；整張圖 20~30 秒；草稿 1/4 時間。避免 Python 逐像素迴圈，
  用向量化；大量小物件（樹）用 Pillow 畫到超取樣遮罩（`shape_mask`）。
- 新配方寫進 `pixelcanvas/scenery.py`（通用）或場景檔（一次性），遵守「先遮罩 / 密度場，再顏色」的寫法。
- 場景的可調參數寫成模組層級的大寫常數（`--set` / `--sweep` 才改得到）；每個繪圖步驟呼叫 `log("步驟")`（`--stages` 依此快照）。
- 文字一律用 `cv.text` / `text_mask`（不指定 font 會自動選中文字型）；精確幾何形狀用 `pixelcanvas.sdf`。
- 改了 API 或新增技巧，記得同步 `skills/pixelcanvas/references/`（api.md / techniques.md / typography.md）與 README.md / README-zh.md，並跑測試。

## 使用者會怎麼要求

「畫一張 ○○」→ 走標準流程；「把 X 改成 Y」→ 改場景參數重渲染；「修這張照片」→ `Canvas.load()` 後用濾鏡 / 遮罩處理。
回報時附上 preview，說明改了什麼、為什麼；不要長篇描述圖的內容（使用者自己會看）。
