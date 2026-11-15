# pixelcanvas

**讓你的程式代理用程式碼畫出完成度高的圖，而且一步一步越改越好。每張圖都是一疊活的圖層，代理用工具編輯它：一個指令移動、
縮放、換色或重畫一樣東西，再對照你的需求檢查整張圖，告訴代理下一步做什麼。只用 numpy + Pillow，不需要圖片生成模型、
不需要 GPU、不需要 API key。**

![第一版草稿與完成的成品](gallery/studio/campfire_fox_before_after.jpg)

*「a fox asleep beside a small campfire in a snowy pine forest at night, a full moon above the trees, cozy and
quiet」*（雪夜松林裡，一隻狐狸睡在小營火旁，樹梢上一輪滿月，溫暖而安靜）：左邊是第一版草稿，右邊是完成的成品。
[完整的工作紀錄](gallery/studio/sessions/campfire_fox.txt)。

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![Agent Skill](https://img.shields.io/badge/Agent%20Skill-SKILL.md-8A2BE2)
![MCP server](https://img.shields.io/badge/MCP-server-0a7)

[English](README.md)

pixelcanvas 是一個 **Agent Skill**，也是一個 **Python 套件**。它在 Claude Code、Codex、Cursor、Gemini CLI、opencode
或任何能執行 shell 的代理裡運作，也能用本地模型：我們用 Qwen 3.8 Flash Next 在 Claude Code 與 opencode 裡實際驅動測試過。
代理讀了 [`SKILL.md`](skills/pixelcanvas/SKILL.md) 之後，在 **`pixelcanvas studio`** 裡工作：圖放在 `./art/NAME/`，
每個物件是草稿檔裡的一個 `# %% name` 區段，草稿的清單行就是檢查清單，另外還有 look 圖與完整的修訂歷史。

```bash
npx skills add treeleaves30760/pixelcanvas
```

接著就可以說：*「做一張夜市 Lantern Street 的海報」*、*「畫出我們電商平台的架構圖」*，之後針對同一張圖再說：
*「把狐狸放大一點」*、*「把月亮移到左邊」*。

## 運作方式：一張代理持續編輯的圖

代理把整份第一版草稿寫在一個檔案 `./art/NAME/draft.py` 裡：一支腳本切成 `# %% name` 區段，由後往前，每樣東西一個區段。
區段的第一行註解就是它的清單行（數量、顏色、'引號文字'、狀態，再接 `|` 與位置）：

```python
# %% apple
# apple: 3, green, one bitten | in bowl
ball(cv, t.place(0.3, 0.5), 0.07, base="#7cb342")
```

一個指令就從草稿建出整張圖。之後每個指令只改一件小事，只重新渲染有變的部分（大約一秒），對照清單行檢查整張圖，最後印出
一行 `NEXT:`：

```bash
studio new fox --prompt "a fox carrying a lantern through a snowy wood at night, titled 'Night Walk'" --size 1920x1080 --out ./night_walk.png --from ./art/fox/draft.py
studio scale lantern 1.8                     # 不是改程式的修改：一個指令
studio add --from ./art/fox/draft.py         # 改程式：編輯一個區段，只有它會重畫
studio note add "3 the lantern glow is faint"   # 留一份缺點清單，讓 NEXT 逐條帶你處理
studio note done 1
```

`studio` 就是 `pixelcanvas studio`（從 clone 執行則是 `python tools/studio.py`）。`--out` 的檔案在每個指令之後都會重寫。
每份報告都以一行下一步收尾；沒有任何量測項目未過時，它請你打開 look 圖，看過三次之後就說圖完成了：

```text
request: 4 of 4 confirmed
look list ([x] = what can be measured holds; by eye: does the picture show each line at a glance?):
  [x] 1 wood: snowy ground receding to dark pines, night sky
  [x] 2 fox: 1, orange, walking left to right | at centre
  [x] 3 lantern: glowing warm yellow, its handle in the fox's mouth  <- worth checking: lantern covers 0.4% of the frame: scale lantern 1.8
  [x] 4 title: 'Night Walk' | at top
NEXT: look 1 of 3: open .../art/fox/look.png (the picture, then every item close up); a line above that does not show at a glance: edit its section ... then studio add --from ...
...
NEXT: finished: the picture is .../night_walk.png (open it before you tell the user)
```

代理打開的 look 圖：上方是帶座標格線與圖層框的整張圖，下面是每個清單項目的特寫。

![look 圖](gallery/studio/campfire_fox_look.jpg)

- **清單行就是合約。** 每個區段的第一行註解變成一個編號項目（數量、顏色、引號文字與位置），量測的部分成立、而且該圖層
  在圖裡看得到時才打勾。沒過的項目會直接寫出修法：`apple: 1 drawn, the request says 3: copy apple --total 3`、
  `apple is not in bowl (boxes: ...): put apple* in bowl`。量測到的疑慮以 `worth checking` 列出：被要求的東西不到畫面的
  1.5 %、互相碰到的複製品、和背後顏色融在一起的東西。
- **用最小的工具修。** `move`、`scale`、`fit`、`put`、`order`、`crop`、`recolor`、`copy --total N --row`、`set L outline=on
  shadow=ground glow=ffcf70`、`set scene light=top-right grade=golden`；形狀看不出是什麼，就編輯那一個區段，再用 `add --from`
  重畫。`scale` 與 `fit` 以 `anchor("fox", "mouth")` 定位的區段的錨點為軸心，`copy --row` 會留空隙，`put` 讓東西留在畫面內，
  指令蓋住了東西或把它推出畫面時，會建議 `undo`。
- **什麼都弄不壞整張圖。** 出錯的區段保留上一次成功的渲染，報告指出草稿的哪一行；`undo`、`redo`、`goto REV` 會一起還原程式碼
  與參數；`compare 5 34` 把兩個版本並排。
- **你要的檔案永遠是最新的。** `--out ./night_walk.png` 在每個指令之後都會重寫，完成不需要多一個指令。
- **問工具，不讀原始碼。** `studio api sk.paint creature` 印出每個呼叫的確切參數與範例；`studio api steam` 列出名稱含這個字
  的所有呼叫；`studio look --text` 把數字交給沒有視覺的模型。

所有指令與圖的資料夾結構：[studio.md](skills/pixelcanvas/references/studio.md)。

## 在 studio 裡畫出來的

這三張圖都是照著上面的流程，只用 CLI 畫出來的：第一版草稿、一個個小修正。每份工作紀錄列出每個指令與當下的需求
分數。

| | |
|---|---|
| ![Lantern Street 海報](gallery/studio/night_market.jpg) | ![靜物](gallery/studio/still_life.jpg) |
| *「A poster for a night market titled 'Lantern Street': three red paper lanterns hanging from a string across the top, a steaming bowl of noodles in the middle, the lines 'Every Friday & Saturday' and '18:00 - 23:00' at the bottom, deep indigo night」*（夜市海報：上方一條繩上掛著三盞紅紙燈籠，中間一碗冒著熱氣的麵，底部兩行字，深靛藍夜色）：rev 13 完成，[工作紀錄](gallery/studio/sessions/night_market.txt) | *「A still life on a wooden table: three green apples in a white bowl, a blue mug right of the bowl, two yellow lemons in front of the bowl, a glass bottle on the left」*（木桌上的靜物：白碗裡三顆青蘋果、碗右邊一個藍色馬克杯、碗前兩顆黃檸檬、左邊一支玻璃瓶）：rev 21 完成，[工作紀錄](gallery/studio/sessions/still_life.txt) |
| ![海報：第一版草稿與成品](gallery/studio/night_market_before_after.jpg) | ![靜物：第一版草稿與成品](gallery/studio/still_life_before_after.jpg) |
| 修正把燈籠複製成三盞、均勻掛在繩子上，把碗往下移讓熱氣離開標題，碗裡也放滿了麵、蛋、青菜與筷子。 | 修正把瓶子染成淡綠色、在放大的碗裡複製出三顆蘋果、碗前複製出兩顆檸檬，畫面也裁到桌面。 |

![交件的狐狸與營火](gallery/studio/campfire_fox.jpg)

look 圖：[海報](gallery/studio/night_market_look.jpg)、[靜物](gallery/studio/still_life_look.jpg)。

## 後續修改只要一個指令

交件後的圖保留著它的狀態。「把狐狸放大一點」就是 `studio -p campfire_fox scale fox 1.2`，使用者手上的檔案會跟著重畫。
成品完成後（rev 29 到 32）的四個要求，各一個指令：`scale fox 1.2`、`recolor moon ffb36b`、`move moon --to 0.3 0.24`、`add owl --kit "creature('owl', at=(0.885, 0.5), size=0.11, pose='perch', facing='left')"`：

![交件後的四個後續指令](gallery/studio/campfire_fox_follow_up.jpg)

會改變需求本身的要求（換顏色、數量或文字）就改 `draft.py` 裡那個區段的清單行，再執行 `studio -p NAME add --from ./art/NAME/draft.py`，
接下來交給 NEXT。

## 每個物件都是真正的圖層

`studio deliver PATH.png --layers` 在 PNG 旁邊寫出 PSD（Photoshop、Photopea、GIMP、Affinity）與 ORA（Krita、GIMP），每個物件一個
圖層，名稱和 studio 裡一樣。上面那張靜物，一層一層攤開（table、bottle、bowl、apple、apple2、apple3、mug、lemon、lemon2）：

![交件靜物的各個圖層](gallery/studio/still_life_layers.jpg)

`studio export NAME.py` 把同一張圖寫成一個給 `tools/render.py` 用的場景檔。

## 用真正的畫材交件

完成的圖可以交成 3D render、平面圖形、油畫、水彩、墨線畫或像素畫。圖層會變成逐像素的緩衝（固有色、每個形狀的浮雕、法線、
深度、發光），一盞主光照亮它們，帶出形體陰影、地面上的投影與接觸遮蔽，再由畫材用自己的方式把結果畫出來：油畫的筆觸沿著形體走，
水彩在紙上暈出帶積水邊緣的色塊，墨線畫有輪廓線與順著形體轉的排線，像素畫是小色盤的像素格。`studio new` 會從需求讀出畫材
（「水彩」、「油畫」、「墨線畫」、「像素畫」、「3D render」）；之後換畫材不會重跑任何區段。

```bash
studio set scene medium=watercolour          # render flat oil watercolour ink pixel off
studio set scene medium_mark=2 medium_strength=0.7 medium_palette=12 medium_paper=f4ecd8
studio stages                                # 平面剪影 -> 浮雕黏土 -> 上光的顏色 -> 畫材
```

上面那張靜物，每格一次 `set scene medium=`（render、flat、oil／watercolour、ink、pixel）：

![六種畫材的靜物](gallery/studio/still_life_media.jpg)

畫材檢查會確認圖保有原稿的明暗分布、要求的顏色、每樣東西的對比與清楚的文字；未過的那項就是 `NEXT:` 那一行，附修法。每個旋鈕與
各畫材畫什麼：[medium.md](skills/pixelcanvas/references/medium.md)。

## 可以畫什麼

![coding agent 用 pixelcanvas 畫出的作品](gallery/wall.jpg)

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
| *「同一幅風景，十二種媒材」*：43 種具名畫風，從油畫、水彩到彩繪玻璃與藍圖 | 633 個圖示，六種樣式，用名稱就能放上畫面 | *「黃昏的紅白燈塔，標題 'Harbour Light'」*：下面那個場景檔 |

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

## 場景檔：一張圖就是一支程式

studio 畫的每樣東西都是建立在各套件上的 Python，整張圖也能放進一個場景檔（`studio export NAME.py` 會寫出一個）。形狀用畫布
的比例座標擺放，每個物件都自帶受光與陰影；光暈、柔和賽璐璐色階、平面與墨線風格的外框，以及精確文字都已內建：

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

`python tools/render.py gallery/src/harbour_light.py --check` 會渲染並寫出報告：畫面評析、文字檢查、每個物件落在哪裡，
以及逐項比對需求，最後是 FIX 區塊：

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

這就是那個場景檔畫出的圖：

![Harbour Light](gallery/harbour_light.jpg)

圖表、儀表板、圖解、平面圖、工程圖與 UI 從 30 個起手範本之一開始，進到 studio 成為一個圖層，標註與印章再各自成為上面的圖層：

```bash
pixelcanvas new sales --dir . --prompt "quarterly sales as a bar chart, titled 'Sales 2025'"
pixelcanvas studio new sales --prompt "quarterly sales as a bar chart, titled 'Sales 2025', stamped 'Draft'"
pixelcanvas studio add base --scene sales.py
pixelcanvas studio add stamp --text "Draft" --box 0.72 0.05 0.95 0.14 --color d63c2f --bold
```

## 比對參考圖

給代理一張圖，說「照這個風格」。`match.py` 把參考圖讀成數字，代理用這些數字作畫，同一個工具再列出還有哪裡不同，差距最大的排在最前面。

```bash
python tools/match.py REF.png               # 風格表：帶角色的配色、線條、色階、高光
python tools/match.py REF.png DRAFT.png     # 草稿與參考圖的差距，以及一個相符分數
```

以 [`scenes/illustration_anime.py`](scenes/illustration_anime.py) 的動漫插畫為例：

```text
roles: background #508fdd, skin #ef8987, line #5d313a, accent #a62250
lines: #5d313a (coloured), width 0.0037 of the height (~4 px at 1080)
shading by region (largest first):
  #d1535e 11%: hard bands, 3 tones; shadow #b12854 = L -0.12, hue -13 deg; highlight band #f88f96 (L +0.17)
  #fde3d4 5% (skin): hard bands, 3 tones; shadow #dabbb0 = L -0.12, hue -10 deg
PASTE (scene docstring / constants):
  PALETTE = ["#508fdd", "#80b7ed", "#b8d4ec", "#f4efe9", "#d2555f", "#a62250", "#e89c2a", "#ef8987"]
  LINE, LINE_WIDTH = "#5d313a", 0.0037      # sk.cel(..., line=LINE, line_width=LINE_WIDTH)
  SHIFT, BANDS = (-0.119, -11.9), 2     # sk.cel(mask, base, bands=BANDS, shift=SHIFT)
```

一張頭髮畫成紫色的草稿，對照同一張參考圖：

```text
match 89/100 against the reference
  1. [palette gap 0.17] palette distance 0.021 (OKLab) - use the reference colours: #d2555f (draft nearest #896eb3), #a62250 (draft nearest #664697), #ef8987 (draft nearest #e59c35)
  2. [lines gap 0.26] line #372d3d width 0.0037 vs #5d313a width 0.0037 - line="#5d313a", line_width=0.0037
  3. [contrast gap 0.35] value range 0.35-1.00 vs 0.49-1.00 - less contrast
```

在渲染迴圈裡，`render.py SCENE --check --ref REF.png`（或在場景裡寫 `REFERENCE = "REF.png"`）會把同樣的差距加成一個
REFERENCE 區塊。完整流程：[reference.md](skills/pixelcanvas/references/reference.md)。

## 內容一覽

| 套件 | 能畫什麼 | 參考頁 |
|---|---|---|
| `illustrate` | 自由繪製：`Sketch` 以比例座標畫形狀、局部座標系、blob、漸細筆觸、受光陰影、柔和賽璐璐色階、髮絲光澤、眼睛與腮紅、貼紙、暈染；`cv.draw()` 提供超取樣的 Pillow 圖層 | [illustration.md](skills/pixelcanvas/references/illustration.md) |
| `reference` | 比對參考圖：風格表、可直接貼上的常數、草稿的差距（`tools/match.py`） | [reference.md](skills/pixelcanvas/references/reference.md) |
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

## 本地模型

studio 的流程可以交給你自己電腦上的模型：每個指令之後的那一行 `NEXT:` 讓小模型不會迷路，沒有影像輸入的模型則用
`studio look --text`（每個圖層的框、大小與平均色）檢查圖。我們用 Qwen 3.8 Flash Next 在 Claude Code 與 opencode 裡實際驅動
測試過。在 opencode 裡用有視覺的模型時，宣告這個模型能讀圖
（[docs/INSTALL.md](docs/INSTALL.md#local-models-in-opencode-the-studio-with-vision)），它就會自己看 `look` 圖。

最小的模型則有 **pixelcanvas-lite** skill：模型只寫一份小小的 JSON spec，`compose` 再用各個套件畫出來，支援 16 種類型，
從海報、桌布到儀表板、房間與 3D 場景。每次執行都以一行 `NEXT:` 結尾。完全不用寫 Python。

```bash
pixelcanvas install-skill opencode --skill pixelcanvas-lite
pixelcanvas compose --example poster > poster.json
pixelcanvas compose poster.json --check
pixelcanvas compose poster.json --deliver final/
```

spec 格式：[spec.md](skills/pixelcanvas/references/spec.md)。

## 場景檔的圖層、SVG 與 DXF

場景檔裡每個 `log("step")` 都是一個圖層：`render.py --layers` 會輸出 `SCENE.psd`、`SCENE.ora` 和一個 PNG 資料夾，就像
studio 的 `deliver PATH.png --layers`。`--svg` 把向量繪圖（圖解、圖表、UI、圖示）存成 SVG，`--dxf` 把工程圖存成 CAD 可用的 DXF。

```bash
python tools/render.py drafting_bracket --svg --dxf
python tools/render.py clay_campsite --layers
```

## MCP server

`pixelcanvas mcp` 在 stdin / stdout 上說 Model Context Protocol，提供 15 個工具，每一份參考頁也都是 resource。
其中四個就是 studio：`studio`（任何一條 studio 指令，回傳報告與 look 圖）、`studio_draft`（一次寫完整份草稿）、
`studio_layer`（重畫單一物件）、`studio_read`。其餘：`kinds`、`new_scene`、`write_scene`、`read_scene`、`render`、
`look`、`find`、`reference`、`ask`、`animate`、`layers`。

```bash
claude mcp add --scope project pixelcanvas -- python -m pixelcanvas.mcp_server
codex mcp add pixelcanvas -- python -m pixelcanvas.mcp_server
```

Claude Desktop、Cursor 與 opencode 的設定：[docs/INSTALL.md](docs/INSTALL.md#as-an-mcp-server)。

## 參考頁

- [SKILL.md](skills/pixelcanvas/SKILL.md)：studio 的作畫流程、第一版草稿、需逐項檢查的圖、美術方向、套件表
- [studio.md](skills/pixelcanvas/references/studio.md)：每個 studio 指令、圖的資料夾、草稿格式、報告、look
- [api.md](skills/pixelcanvas/references/api.md)：完整 API 索引
- [illustration.md](skills/pixelcanvas/references/illustration.md)、[techniques.md](skills/pixelcanvas/references/techniques.md)、
  [typography.md](skills/pixelcanvas/references/typography.md)、[review.md](skills/pixelcanvas/references/review.md)
- 每個套件一頁，都在 [`skills/pixelcanvas/references/`](skills/pixelcanvas/references/)
- [CHANGELOG.md](CHANGELOG.md)：0.5 的新內容

## 參與貢獻

歡迎貢獻作品、套件與修正。[`CLAUDE.md`](CLAUDE.md) 有專案地圖與新增套件的步驟（registry、嚴格參數、manifest hook、程式碼
區塊都能執行的參考頁、起手範本與範例場景）。送 pull request 前請先跑測試：

```bash
pip install -r requirements.txt pytest
python -m pytest -q
```

`python tools/build_gallery.py` 會從場景重新產生作品牆與其中的圖；`gallery/studio/` 放的是 studio 作品與它們的工作紀錄。

## 授權

MIT，見 [LICENSE](LICENSE)。
