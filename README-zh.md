# pixelcanvas：easel

**easel 是一個由 AI agent 操作的命令列繪圖程式。給它一張參考圖，agent 就像畫家一樣重畫：先規劃畫面的區域，再分階段作畫
（草稿、線稿、平塗、陰影、細節、完稿）。每個指令之後，easel 用數字回報改變了什麼，並輸出一張結果圖；agent 打開它、判斷，然後
繼續或復原。也可以要求「重畫，但是……」（薄荷綠的頭髮、做成海報、人物身後換成湖泊），或只用文字描述一張圖。**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![Agent Skill](https://img.shields.io/badge/Agent%20Skill-SKILL.md-8A2BE2)

[English](README.md)

pixelcanvas 是提供 easel 與其 Agent Skill（[`skills/easel/SKILL.md`](skills/easel/SKILL.md)）的 Python 套件。
它可在 Claude Code、opencode，或任何有 shell 並能開啟圖檔的 agent 中執行，搭配雲端模型或小型本地模型皆可。
依賴 numpy、Pillow、OpenCV、scikit-image 與 SciPy；不需要影像生成模型、GPU 或網路。

## 成果

在一組 93 張的私有參考圖（動漫、風景、宣傳圖）上，重畫流程的評審平均分數為 9.42 / 10，其中 88 張被評審評為與參考圖
同等品質。在重畫流程中，筆觸沿著參考圖下筆，並沿筆觸從參考圖取色，沒有任何一張圖是逐像素修補而成。小型開放權重模型
（Qwen 3.8 Flash 等級）在 Claude Code 或 opencode 中操作這個 skill，約 19-45 分鐘、44-86 個指令完成一張重畫；它的
作畫紀錄現已涵蓋 23 張動漫圖，評審平均分數為 9.52，其中 23 張中有 21 張被評為與參考圖同等品質（腳本流程在同樣的圖上：
9.48，23 張中 22 張）。

## 安裝

```bash
pip install pixelcanvas            # easel 指令（pixelcanvas 指令執行同一個程式）
easel-install-skill                # skill -> ~/.claude/skills/easel
easel-install-skill --opencode     # 或 -> ~/.config/opencode/skills/easel
```

接著對 agent 說：*「用 easel 重畫 reference.jpg」*、*「把 reference.jpg 重畫成極簡海報」* 或 *「畫一座夕陽下的山中湖泊」*。
其他安裝方式（Claude Code plugin、`npx skills`、clone）：[docs/INSTALL.md](docs/INSTALL.md)。

## 運作方式

圖存在一個專案資料夾 `pic.easel/` 中：參考圖、每個圖層一張 PNG、區域、復原紀錄，以及每個指令的日誌。每個指令都會
修改它並印出報告：分數 `before -> after`、警告、差異最大的位置、畫布的 `VIEW:` 圖、同一範圍參考圖的 `VIEW-REF:` 圖，
以及一行 `NEXT:` 指出下一個指令。

agent 依循一個迴圈：執行一個指令，讀報告，打開 `VIEW`，用一句話說明是否有幫助，然後照 `NEXT:` 做，或 `easel undo`。
座標在兩軸上都是 0-1000，從視圖的尺規讀取。

每個階段畫在自己的圖層上，並有一道 easel 逐區域量測的關卡。`easel stage done` 只在關卡通過時前進，或在 agent 給出
理由時（`--because "..."`）前進。`easel todo --all` 印出完整的工作清單；agent 把它抄進自己的待辦工具，逐行勾選。

### 三種模式

| 模式 | 開始 | 誰下筆 |
|---|---|---|
| trace | `easel new pic --ref reference.jpg --mode trace` | 階段指令沿著參考圖下筆並從參考圖取色；agent 規劃區域、安排順序、檢查每個指令，並決定在哪裡細修 |
| hand | `easel new pic --ref reference.jpg --mode hand` | agent 一筆一筆畫，參考圖放在旁邊（實驗性） |
| free | `easel new pic --mode free --request "..."` | agent 依文字需求，用構圖工具作畫；先規劃主體與部件，再走同樣的階段 |

## 一次重畫，逐個指令

skill 教的 trace 模式流程，以一張有深色輪廓線的動漫圖（風格 `cel`）為例。每次 shell 呼叫只執行一個指令，每次之後都看
`VIEW`：

```bash
easel new pic --ref reference.jpg --mode trace      # 猜測風格，顯示編號標記
easel region add hair --marks 4,5,9,13              # 用編號標記定義區域……
easel region add face --wand 500,550 --focus 3      # ……或在部件內點一下
easel region rest background
easel stage done                                    # 規劃完成
easel todo --all                                    # 依序列出工作清單
easel do "sketch face; sketch hair; sketch background"
easel stage done
easel ink all                                       # 線稿
easel stage done
easel flat all                                      # 平塗
easel stage done
easel shade all
easel stage done
easel detail all                                    # 全圖細筆觸，並排出區域順序
easel detail face --level 1                         # 依排序逐區域畫細線
easel look face
easel dot all                                       # 在反光與細點上點柔和的點
easel stage done
easel look
easel sharpen                                       # 依參考圖調整的完稿銳化
easel export final.png
easel stage done
```

繪畫或照片改走 block-in 與 forms（`easel paint all`），取代線稿、平塗與陰影。`easel next` 永遠會指出下一個指令。
階段工具漏掉的地方，用手繪工具修正：

```bash
easel stroke "480,590 510,585 540,590" --size 3 --color pick --snap   # 參考圖的顏色，吸附到它的邊緣
easel fill 520,640 --color pick                                       # 在線稿內油漆桶填色
easel shape ellipse 470,440,500,470 --color "#ffffff"                 # 眼睛高光
easel erase 400,300,460,360 --layer line                              # 擦除一個框內的筆跡
```

## 重畫，但是……

需求可以要求參考圖沒有的東西：換一個顏色、做成海報、換一個場景。建立專案時帶上需求，照常規劃區域，並在
`easel stage done` 之前逐項說明改動（`easel change add ...`；單純複製用 `easel change none`）。改動會修改「目標圖」，也就是
後續階段描摹並用來評分的圖；原圖保留為 `orig.png`。規劃完成後，改動階段負責做出這些改動，之後的階段描摹目標圖。
`easel look --show changes` 並排顯示原圖與目標圖；全程由 `easel next` 指出下一個指令。

換色（*「薄荷綠的頭髮與暗紅色上衣」*），區域為 `hair`、`face`、`clothes`、`background`：

```bash
easel new pic --ref reference.jpg --mode trace --request "redraw it with mint-green hair and a dark red top"
easel change add recolor hair --to "#8fdcb4"
easel change add recolor clothes --to "#7b2635"
easel stage done
easel recolor hair --to "#8fdcb4"            # 保留陰影、線條與高光；回報色相 before -> after
easel recolor clothes --to "#7b2635"
easel region refine hair --by hue            # 仍有舊顏色的零星色塊時
```

海報（*「極簡海報，標題 SLOW SUMMER，奶油底色上的海軍藍線稿，珊瑚色點綴」*）：

```bash
easel change add canvas --size 900x1200 --place 186,240,814,930 --color "#f3e9d8"
easel change add palette map --to "#1d2a44,#f3e9d8" --accent "clothes=#e8644a"
easel change add graphic disc 260,295,860,745 --color "#e97a62"
easel change add text "SLOW SUMMER" --box 70,40,930,175 --font display --weight black --color "#1d2a44"
easel stage done
easel canvas --size 900x1200 --place 186,240,814,930 --color "#f3e9d8"
easel palette map --to "#1d2a44,#f3e9d8" --accent "clothes=#e8644a"
easel graphic disc 260,295,860,745 --color "#e97a62"
easel style poster                           # 目標圖看起來像海報時，改動關卡會要求這一步
```

新場景（*「站在夕陽下的山中湖泊前」*）：背景可以是另一個 easel 專案、一張圖檔、一個漸層或單一顏色。

```bash
easel change add cutout figure --regions face,hair,clothes
easel change add background --gradient "#f2a65a,#5b5f8f"     # 或 --from lake.easel、--from sky.png、--color "#88aacc"
easel change add light
easel stage done
easel cutout figure --regions face,hair,clothes   # 回報人物內殘留的底色，並框出三個最弱的邊緣位置
easel background --gradient "#f2a65a,#5b5f8f"
easel light                                       # 把人物染向背景的光色，並加上輪廓光
easel shadow figure                               # 人物站在畫面內時加上接觸陰影
```

## 依文字需求作畫

沒有參考圖時，easel 的做法相同：agent 用方框規劃需求中的主體與部件，再用構圖工具（`sketch figure`、`sketch head`、`face`、
`line`、`flat auto`、`shade`、`eye`、`hair --flow`、`fold`、`text`）依階段畫出來，最後 `easel polish` 為完稿統一光線並做最後的整理：

```bash
easel new pic --size 1000x1200 --request "a chibi fox mage holding a staff"
easel plan subject fox --box 170,60,760,960
easel plan part head --of fox --box 200,90,720,500 --focus 3
easel sketch figure fox --pose hold --heads 2.5
```

## 指令

`easel help`：

```text
easel: a paint program. Coordinates are 0-1000 on each axis (x right, y down).
after each command: read the report, open VIEW (the canvas) and, to compare, VIEW-REF (the reference)
setup:   new  style  mode hand  status  next  todo [--all]  stage  undo [n]  redo
look:    look [region|box] [--show both|canvas|ref|diff|marks|layers] [--grid]   marks   pick   palette   score
regions: region add <name> --marks 3,4 | --box x0,y0,x1,y1 | --poly "x,y ..." | --wand x,y  [--focus 1-3]; region rest background; region focus <name> 3; region list; region remove <name>
stages:  sketch  ink  flat  shade  paint [--size N]  airbrush  detail [--level 2|1]  dot  retouch <region>   (each takes a region, default all; no options = the engine's own settings for the style)
hand:    stroke  fill  shape  gradient  erase  smooth  adjust   (act on the stage's layer or --layer)
changes: change add <kind> ...  change list  change drop <n>  change none   (state what the request changes from the reference; kinds: canvas recolor palette graphic cutout background light shadow text shape)
         recolor <region> --to "#hex"   region refine <region> --by lines|hue   canvas --size WxH --place x0,y0,x1,y1 [--color "#hex"]   palette map --to "dark,...,light" [--accent "region=#hex"]   graphic disc|rect|ring|frame <box> --color "#hex"
         cutout <subject> --regions a,b,c   background --from <easel project|picture> | --gradient "#a,#b" | --color "#hex"   light [--tint "#hex"] [--rim left|right]   shadow <subject>
free:    no reference: plan  palette make|pick|set  sketch figure|head|ellipse|box|curve  line  band  face  eye  hair  locks  blush  hand  wear  fold  crease  tufts  mark  pattern  scatter  surface  highlight  rim  shadow --from <part> --onto <part>  reflect  wet  haze  atmosphere  glow  cloud  ground  foliage  window  linestyle
finish:  polish (one finishing pass over a drawn picture; --no-<part> leaves a part out)   sharpen (the finishing sharpen, fitted to the reference)   fx <glow|grain|blur|vignette|paper|sharpen|levels> [region] [--amount 0..1]   text "<string>" --box x0,y0,x1,y1 [--font --color --align --outline px,#hex --vertical --fit]   fonts
other:   layer  export <file.png> [--psd] [--ora] [--timelapse film.gif|film.mp4] [--stages dir]   do "<cmd>; <cmd>"   help [command]
NEXT: easel next
```

任何指令後加 `--help` 會印出它的選項與範例。`easel export final.png --psd --ora` 另外輸出可在 Photoshop、Krita 或 GIMP
開啟的圖層；`--timelapse film.gif` 重播日誌；`--stages dir` 為每個階段存一張圖。

## 開發

```bash
git clone https://github.com/treeleaves30760/pixelcanvas
cd pixelcanvas
pip install -e ".[test]"
python -m pytest tests/easel -q
```

程式碼在 `pixelcanvas/easel/`，測試在 `tests/easel/`，skill 在 `skills/easel/SKILL.md`。
[CLAUDE.md](CLAUDE.md) 有專案地圖與發布步驟；[CHANGELOG.md](CHANGELOG.md) 是版本紀錄。
0.6.0 版以 easel 取代了舊的「用程式碼畫圖」工具。

## 授權

MIT
