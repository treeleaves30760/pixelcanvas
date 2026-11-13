# pixelcanvas

**讓你的程式代理用程式碼畫出完成度高的圖：海報、圖解、儀表板、3D 場景與插畫，只用 numpy + Pillow。不需要圖片生成模型、
不需要 GPU、不需要 API key。**

![coding agent 用 pixelcanvas 畫出的作品](gallery/wall.jpg)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![Agent Skill](https://img.shields.io/badge/Agent%20Skill-SKILL.md-8A2BE2)
![MCP server](https://img.shields.io/badge/MCP-server-0a7)

[English](README.md)

pixelcanvas 是一個 **Agent Skill**，也是一個 **Python 套件**。Claude Code、Codex、opencode、Cursor、Gemini CLI，或任何
能執行 shell 的代理，讀了 [`SKILL.md`](skills/pixelcanvas/SKILL.md) 之後會先設計畫面、用 Python 畫出來、執行一份會指出
該修什麼的檢查，再親眼看過結果才交件。每張圖都是一個簡短的場景檔：任何語言的文字都一字不差，圖層可編輯，需要時還能輸出
SVG 與 DXF。

```bash
npx skills add treeleaves30760/pixelcanvas
```

接著就可以說：*「做一張鋼琴之夜 NOCTURNE 的中英雙語海報」*、*「畫出我們電商平台的架構圖」*、*「一個等角視角的閱讀角落，
扶手椅上有一隻貓」*。

## 可以畫什麼

下面每一張圖都是 pixelcanvas 根據圖下的需求畫出來的，場景檔都在 [`scenes/`](scenes/)。

**海報、印刷品與字體**

| | | |
|---|---|---|
| ![type poster](gallery/type_poster.jpg) | ![farmers market](gallery/farmers_market.jpg) | ![jazz poster](gallery/jazz_poster.jpg) |
| *「NOCTURNE 音樂會的中英雙語海報：一輪月亮、『一夜限定』印章、日期、場地與票價」* | *「台北週末小農市集的傳單：標題『週末小農市集』、插畫蔬菜、溫暖的紙質背景」* | *「爵士之夜 'Blue Hour' 的海報，週五 21:00 在 The Velvet Room，一支薩克斯風，深藍配金」* |
| ![cafe menu](gallery/cafe_menu.jpg) | ![readme banner](gallery/readme_banner.jpg) | ![app icons](gallery/app_icons.jpg) |
| *「'Fika & Co.' 咖啡館菜單：咖啡、茶、甜點各五項附價格，一個咖啡杯和一個可頌，米色紙上的襯線字」* | *「dotfiles 工具 'driftwood' 的 README 橫幅：粗體名稱、標語 'your shell, everywhere'、一個有三行指令的終端機」* | *「生產力套件的十二個 app 圖示，風格一致：圓角磚、柔和漸層、長陰影，各附名稱」* |

**畫面、圖表與圖解**

| | | |
|---|---|---|
| ![meditation app](gallery/meditation_app.jpg) | ![landing page](gallery/landing_page.jpg) | ![dashboard](gallery/dashboard.jpg) |
| *「冥想 app 首頁：'Good evening, Sam'、停在 4:32 的呼吸計時器、Sleep / Focus / Calm 卡片、底部分頁列，深紫主題」* | *「Northwave 的 landing page 首屏：導覽列、標題 'Ship product updates your customers actually read'、兩個按鈕和一張儀表板截圖」* | *「訂閱制 app 的分析儀表板：MRR、活躍用戶、流失率、NPS 指標卡，營收折線、各管道註冊數、方案佔比圓環」* |
| ![architecture](gallery/architecture.jpg) | ![flowchart](gallery/flowchart.jpg) | ![sequence](gallery/sequence.jpg) |
| *「電商平台架構圖：用戶端、CDN 與 gateway、各服務、事件匯流排與 worker、資料儲存。深色主題，依層分組」* | *「從合併 PR 到上線的發版流程圖，含測試、smoke test、核准三個判斷與重試迴圈」* | *「passkey 登入的循序圖：使用者、瀏覽器、API、驗證服務與使用者資料庫，簽章失敗的 alt 區塊」* |

**工程圖與教學圖**

| | | |
|---|---|---|
| ![bracket drawing](gallery/bracket_drawing.jpg) | ![floor plan](gallery/floor_plan.jpg) | ![stone lantern](gallery/stone_lantern.jpg) |
| *「鋁製角座的 A3 工程圖：前視、俯視、右視與剖面、等角視圖、尺寸標註與標題欄」* | *「兩房公寓平面圖：門的開啟方向、窗、家具、房間名稱與面積、總尺寸、指北針與比例尺」* | *「日本石燈籠的教科書插圖，用引線標出從頂端寶珠到底座的各部位名稱與高度」* |

**3D 與黏土角色**

| | | |
|---|---|---|
| ![reading nook](gallery/reading_nook.jpg) | ![clay campsite](gallery/clay_campsite.jpg) | ![earbuds](gallery/earbuds.jpg) |
| *「舒適閱讀角落的等角剖面：扶手椅、立燈、滿滿的書架、圓地毯、扶手上的貓、窗外下著雨」* | *「黏土風的登山客和她的狗在夜晚營地：發光的橘色帳篷、火星飛揚的營火、松樹與銀河」* | *「白色無線耳機放在打開的充電盒裡，淡粉色無縫背景、柔和棚拍光的產品照」* |
| ![clay cast](gallery/clay_cast.jpg) | ![clay zoo](gallery/clay_zoo.jpg) | ![toy diorama](gallery/toy_diorama.jpg) |
| *「角色介紹：十個黏土角色排成一排，從探險家到超級英雄，各附名稱」* | *「三十種黏土動物站在小底座上的圖鑑頁，各附名稱」* | *「棋盤格地板上的玩具：小汽車、球、小狗、揮手的小孩、ABC 積木和火箭」* |

**插畫、自然與畫風**

| | | |
|---|---|---|
| ![ukiyo-e wave](gallery/ukiyoe_wave.jpg) | ![animal stickers](gallery/animal_stickers.jpg) | ![campfire](gallery/campfire_lake.jpg) |
| *「浮世繪木版畫風格的巨浪，遠方是富士山，還有兩艘漁船」* | *「九隻可愛動物的貼紙頁，每隻都有白色裁切邊和小標籤，粉彩背景」* | *「夜晚山中湖畔的營火，火星上升，月亮倒映在水面」* |
| ![aurora](gallery/aurora_lake.jpg) | ![sea sunset](gallery/sea_sunset.jpg) | ![studio still life](gallery/studio_still_life.jpg) |
| *「寬幅橫幅：松林之間鏡面湖上的極光，標題 'pixelcanvas'」* | *「夕陽落在平靜的海面，波浪上一條光帶」* | *「大理石板上的棚拍靜物：玻璃瓶、陶土色馬克杯、玻璃球和一顆金球」* |
| ![art styles](gallery/art_styles.jpg) | ![pictograms](gallery/pictograms.jpg) | ![harbour light](gallery/harbour_light.jpg) |
| *「同一幅風景，十二種媒材」*：43 種具名畫風，從油畫、水彩到彩繪玻璃與藍圖 | 633 個圖示，六種樣式，用名稱就能放上畫面 | *「黃昏的紅白燈塔，標題 'Harbour Light'」*：下一節的場景 |

## 安裝

選一個適合你代理的方式，每一種都只要一步。

| 方式 | 指令 |
|---|---|
| 任何支援 Agent Skills 的工具（Claude Code、Codex、Cursor、Gemini CLI、opencode ...） | `npx skills add treeleaves30760/pixelcanvas` |
| Claude Code plugin | `/plugin marketplace add treeleaves30760/pixelcanvas` 接著 `/plugin install pixelcanvas@pixelcanvas` |
| pip，然後把 skill 交給你的代理 | `pip install pixelcanvas && pixelcanvas install-skill claude`（或 `agents`、`codex`、`cursor`、`opencode`） |
| 從 clone 安裝 | `git clone https://github.com/treeleaves30760/pixelcanvas && cd pixelcanvas && pip install -r requirements.txt && python tools/install_skill.py` |

skill 內附 `scripts/bootstrap.py`，第一次使用時會自動準備好執行環境，所以第一個需求就能直接畫。`pixelcanvas doctor`
會檢查 Python、numpy、Pillow、字型與寫入權限；`uvx pixelcanvas doctor` 不用安裝就能執行。各工具、各專案的完整選項：
[docs/INSTALL.md](docs/INSTALL.md)。

## 運作方式：規劃、繪製、檢查、檢視

**1. 規劃。** 代理在場景的 docstring 裡寫下畫面清單，並把需求一字不差地放進 `PROMPT`。

**2. 繪製。** 一次寫完整個場景。形狀用畫布的比例座標擺放，陰影、外框、光暈與精確文字都已內建：

```python
"""lighthouse: red and white, on a dark cliff at the right, lamp glowing, beam over the sea; title 'Harbour Light'"""
from pixelcanvas import Canvas
from pixelcanvas.illustrate import Sketch, paper, grade
PROMPT = "a red and white lighthouse at dusk, titled 'Harbour Light'"
DEFAULT_SIZE = (1600, 900)

def render(width=1600, height=900, seed=1, log=lambda step: None):
    cv = Canvas(width, height)
    sk = Sketch(cv, light=(-0.6, -0.4))
    log("sky and sea")
    cv.gradient([(0, "#1d2150"), (0.5, "#f29a72"), (0.6, "#6a78b0"), (1, "#1c2a55")])
    sk.spot(0.3, 0.6, 0.25, 0.012, "#ffd9a8", 0.6)                         # light on the water
    log("cliff")
    sk.paint(sk.blob([(0.5, 1.05), (0.6, 0.72), (0.78, 0.64), (1.05, 0.62), (1.05, 1.05)]), "#3c3048", shade=0.5)
    log("beam")
    cv.paint("#fff1c4", sk.polygon([(0.81, 0.2), (0.0, 0.3), (0.0, 0.52)]), 0.25)
    log("lighthouse")
    with sk.at(0.81, 0.42, 0.44):                                           # local frame: 1 unit = 0.44 of the height
        sk.paint(sk.polygon([(-0.12, 0.45), (-0.08, -0.35), (0.08, -0.35), (0.12, 0.45)]), "#f4efe6",
                 shade=0.4, line=True, name="lighthouse")
        for y in (-0.2, 0.1, 0.38):
            sk.paint(sk.box(-0.118, y - 0.06, 0.118, y + 0.06), "#d2423c", shade=0.4)
        sk.paint(sk.box(-0.07, -0.5, 0.07, -0.35, 0.02), "#ffe9a8", glow="#ffd27a")
        sk.paint(sk.polygon([(-0.11, -0.5), (0, -0.64), (0.11, -0.5)]), "#d2423c", line=True)
    log("title")
    cv.text("Harbour Light", box=(0.06, 0.08, 0.56, 0.22), color="#fff4e0", font="serif", bold=True)
    grade(paper(cv), "golden", 0.6)
    return cv
```

**3. 檢查。** `python tools/render.py gallery/src/harbour_light.py --check` 會渲染並寫出報告：畫面評析、文字檢查、每個物件
落在哪裡，以及逐項比對需求。最後的 FIX 區塊會指出要改什麼，或宣告這張圖可以交件：

```text
PROMPT "a red and white lighthouse at dusk, titled 'Harbour Light'"
  every quoted text is drawn
  COVERAGE (subjects of the request and what answers to them):
    all 1 subjects of the request are accounted for
  AS REQUESTED (count, colour, place and text the request spells out, against the picture; ✗ names what to change):
    lighthouse: 1 drawn, 1 requested ✓; red and white requested, white painted ✓
    'Harbour Light': drawn ✓
FIX: nothing flagged - look at the picture once, then deliver
```

**4. 檢視。** 代理打開 PNG 親眼看，需要細看時用 `tools/look.py`（座標格線、放大、明度圖、邊緣、取色）。這就是上面那個場景
畫出的圖：

![Harbour Light](gallery/harbour_light.jpg)

結構性的圖則從範本開始，而不是空白檔案：

```bash
pixelcanvas kit find "a corgi sitting under an oak tree at sunset"      # 每個主體該用哪個呼叫
pixelcanvas new corgi --prompt "a corgi sitting under an oak tree at sunset" --dir ./art
pixelcanvas render ./art/corgi.py --scale 0.5 --check
```

## 內容一覽

| 套件 | 能畫什麼 | 參考頁 |
|---|---|---|
| `illustrate` | 自由繪製：`Sketch` 以比例座標畫形狀、局部座標系、blob、漸細筆觸、陰影、貼紙、暈染；`cv.draw()` 提供超取樣的 Pillow 圖層 | [illustration.md](skills/pixelcanvas/references/illustration.md) |
| `text`、`layout` | 任何語言的文字（自動 CJK 字型）、自動縮放、文字特效；海報與卡片範本、主題 | [typography.md](skills/pixelcanvas/references/typography.md) |
| `chart` | 圖表、KPI 指標卡與整個儀表板 | [charts.md](skills/pixelcanvas/references/charts.md) |
| `diagram` | 流程圖、架構圖、樹狀圖、循序圖、時間軸、表格、引線標註，自動排版 | [diagrams.md](skills/pixelcanvas/references/diagrams.md) |
| `ui` | 裝置框中的 app 與網頁畫面，自動排版 | [ui.md](skills/pixelcanvas/references/ui.md) |
| `drafting` | 工程圖（視圖、剖面、尺寸、標題欄）與平面圖，輸出 PNG、SVG、DXF | [drafting.md](skills/pixelcanvas/references/drafting.md) |
| `vector` | 形狀、路徑、圖示與群組，輸出點陣圖與 SVG | [vector.md](skills/pixelcanvas/references/vector.md) |
| `pictograms` | 633 個圖示、六種樣式，也能當遮罩使用 | [pictograms.md](skills/pixelcanvas/references/pictograms.md) |
| `solid` | numpy ray marching 3D：預製物件、材質、棚拍光、透視與等角相機 | [solid.md](skills/pixelcanvas/references/solid.md) |
| `clay` | 3D 黏土角色、31 種姿勢、64 種動物、道具 | [clay.md](skills/pixelcanvas/references/clay.md) |
| `figure`、`creature` | 骨架驅動、有姿勢有穿著的人物與動物 | [figures.md](skills/pixelcanvas/references/figures.md)、[creatures.md](skills/pixelcanvas/references/creatures.md) |
| `flora`、`elements`、`scenery` | 樹木、花卉、火、煙、水、天氣、天空、風景 | [flora.md](skills/pixelcanvas/references/flora.md)、[elements.md](skills/pixelcanvas/references/elements.md) |
| `texture`、`form` | 80 種受光材質；有體積感的 2.5D 物件與房間 | [textures.md](skills/pixelcanvas/references/textures.md)、[forms.md](skills/pixelcanvas/references/forms.md) |
| `backdrop`、`palette` | 一行產生的背景與桌布，保留放文字的安靜區域；具名配色 | [backdrops.md](skills/pixelcanvas/references/backdrops.md) |
| `stylize` | 43 種畫風，`styles pick` 會依需求挑選 | [styles.md](skills/pixelcanvas/references/styles.md) |

`pixelcanvas kit find "需求"` 一次搜尋所有套件；30 個起手範本（`pixelcanvas new --list`）涵蓋海報、圖表、圖解、UI、
平面圖、產品照、人物、寵物等等。

## 小型本地模型

在你自己電腦上跑的模型透過 **pixelcanvas-lite** skill 作畫：它只寫一份小小的 JSON spec，`compose` 再用各個套件畫出來，
支援 16 種類型，從海報、桌布到儀表板、房間與 3D 場景。每次執行都以一行 `NEXT:` 結尾，告訴模型下一步做什麼。完全不用寫
Python。

```bash
pixelcanvas install-skill opencode --skill pixelcanvas-lite
pixelcanvas compose --example poster > poster.json
pixelcanvas compose poster.json --check
pixelcanvas compose poster.json --deliver final/
```

spec 格式：[spec.md](skills/pixelcanvas/references/spec.md)。

## 可編輯圖層、SVG 與 DXF

場景裡每個 `log("step")` 都是一個圖層。`--layers` 會輸出 `SCENE.psd`（Photoshop、Photopea、GIMP、Affinity）、
`SCENE.ora`（Krita、GIMP）和一個 PNG 資料夾，讓人可以移動、隱藏或重畫任何物件。`--svg` 把向量繪圖（圖解、圖表、UI、
圖示）存成 SVG，`--dxf` 把工程圖存成 CAD 可用的 DXF。

```bash
python tools/render.py drafting_bracket --svg --dxf
python tools/render.py clay_campsite --layers
```

## MCP server

`pixelcanvas mcp` 在 stdin / stdout 上說 Model Context Protocol，提供 11 個工具（`kinds`、`new_scene`、`write_scene`、
`read_scene`、`render`、`look`、`find`、`reference`、`ask`、`animate`、`layers`），每一份參考頁也都是 resource。

```bash
claude mcp add --scope project pixelcanvas -- python -m pixelcanvas.mcp_server
codex mcp add pixelcanvas -- python -m pixelcanvas.mcp_server
```

Claude Desktop、Cursor 與 opencode 的設定：[docs/INSTALL.md](docs/INSTALL.md#as-an-mcp-server)。

## 參考頁

- [SKILL.md](skills/pixelcanvas/SKILL.md)：作畫流程、自由繪製、需逐項檢查的圖、美術方向、套件表
- [api.md](skills/pixelcanvas/references/api.md)：完整 API 索引
- [illustration.md](skills/pixelcanvas/references/illustration.md)、[techniques.md](skills/pixelcanvas/references/techniques.md)、
  [typography.md](skills/pixelcanvas/references/typography.md)、[review.md](skills/pixelcanvas/references/review.md)
- 每個套件一頁，都在 [`skills/pixelcanvas/references/`](skills/pixelcanvas/references/)
- [CHANGELOG.md](CHANGELOG.md)：0.4 的新內容

## 參與貢獻

歡迎貢獻作品、套件與修正。[`CLAUDE.md`](CLAUDE.md) 有專案地圖與新增套件的步驟（registry、嚴格參數、manifest hook、程式碼
區塊都能執行的參考頁、起手範本與範例場景）。送 pull request 前請先跑測試：

```bash
pip install -r requirements.txt pytest
python -m pytest -q
```

`python tools/build_gallery.py` 會從場景重新產生這份作品集。

## 授權

MIT，見 [LICENSE](LICENSE)。
