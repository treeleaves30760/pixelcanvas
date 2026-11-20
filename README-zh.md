# pixelcanvas

**pixelcanvas 是一個由 AI agent 操作的命令列繪圖程式。給它一張參考圖，agent 就像畫家一樣重畫：先規劃畫面的區域，再分階段作畫
（草稿、線稿、平塗、陰影、細節、完稿）。每個指令之後，pixelcanvas 用數字回報改變了什麼，並輸出一張結果圖；agent 打開它、判斷，然後
繼續或復原。也可以要求「重畫，但是……」（薄荷綠的頭髮、做成海報、人物身後換成湖泊），或只用文字描述一張圖。**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![Agent Skill](https://img.shields.io/badge/Agent%20Skill-SKILL.md-8A2BE2)

[English](README.md)

pixelcanvas 是一個 Python 套件，只有一個指令 `pixelcanvas`，並附帶 Agent Skill（[`skills/pixelcanvas/SKILL.md`](skills/pixelcanvas/SKILL.md)）。
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
pip install pixelcanvas                  # pixelcanvas 指令
pixelcanvas install-skill                # skill -> ~/.claude/skills/pixelcanvas
pixelcanvas install-skill --opencode     # 或 -> ~/.config/opencode/skills/pixelcanvas
```

接著對 agent 說：*「用 pixelcanvas 重畫 reference.jpg」*、*「把 reference.jpg 重畫成極簡海報」* 或 *「畫一座夕陽下的山中湖泊」*。
其他安裝方式（Claude Code plugin、`npx skills`、clone）：[docs/INSTALL.md](docs/INSTALL.md)。

## 運作方式

圖存在一個專案資料夾 `pic.pixelcanvas/` 中：參考圖、每個圖層一張 PNG、區域、復原紀錄，以及每個指令的日誌。每個指令都會
修改它並印出報告：分數 `before -> after`、警告、差異最大的位置、畫布的 `VIEW:` 圖、同一範圍參考圖的 `VIEW-REF:` 圖，
以及一行 `NEXT:` 指出下一個指令。

agent 依循一個迴圈：執行一個指令，讀報告，打開 `VIEW`，用一句話說明是否有幫助，然後照 `NEXT:` 做，或 `pixelcanvas undo`。
座標在兩軸上都是 0-1000，從視圖的尺規讀取。

每個階段畫在自己的圖層上，並有一道 pixelcanvas 逐區域量測的關卡。`pixelcanvas stage done` 只在關卡通過時前進，或在 agent 給出
理由時（`--because "..."`）前進。`pixelcanvas todo --all` 印出完整的工作清單；agent 把它抄進自己的待辦工具，逐行勾選。

### 三種模式

| 模式 | 開始 | 誰下筆 |
|---|---|---|
| trace | `pixelcanvas new pic --ref reference.jpg --mode trace` | 階段指令沿著參考圖下筆並從參考圖取色；agent 規劃區域、安排順序、檢查每個指令，並決定在哪裡細修 |
| hand | `pixelcanvas new pic --ref reference.jpg --mode hand` | agent 一筆一筆畫，參考圖放在旁邊（實驗性） |
| free | `pixelcanvas new pic --mode free --request "..."` | agent 依文字需求，用構圖工具作畫；先規劃主體與部件，再走同樣的階段 |

## 一次重畫，逐個指令

skill 教的 trace 模式流程，以一張有深色輪廓線的動漫圖（風格 `cel`）為例。每次 shell 呼叫只執行一個指令，每次之後都看
`VIEW`：

```bash
pixelcanvas new pic --ref reference.jpg --mode trace      # 猜測風格，顯示編號標記
pixelcanvas region add hair --marks 4,5,9,13              # 用編號標記定義區域……
pixelcanvas region add face --wand 500,550 --focus 3      # ……或在部件內點一下
pixelcanvas region rest background
pixelcanvas stage done                                    # 規劃完成
pixelcanvas todo --all                                    # 依序列出工作清單
pixelcanvas do "sketch face; sketch hair; sketch background"
pixelcanvas stage done
pixelcanvas ink all                                       # 線稿
pixelcanvas stage done
pixelcanvas flat all                                      # 平塗
pixelcanvas stage done
pixelcanvas shade all
pixelcanvas stage done
pixelcanvas detail all                                    # 全圖細筆觸，並排出區域順序
pixelcanvas detail face --level 1                         # 依排序逐區域畫細線
pixelcanvas look face
pixelcanvas dot all                                       # 在反光與細點上點柔和的點
pixelcanvas stage done
pixelcanvas look
pixelcanvas sharpen                                       # 依參考圖調整的完稿銳化
pixelcanvas export final.png
pixelcanvas stage done
```

繪畫或照片改走 block-in 與 forms（`pixelcanvas paint all`），取代線稿、平塗與陰影。`pixelcanvas next` 永遠會指出下一個指令。
階段工具漏掉的地方，用手繪工具修正：

```bash
pixelcanvas stroke "480,590 510,585 540,590" --size 3 --color pick --snap   # 參考圖的顏色，吸附到它的邊緣
pixelcanvas fill 520,640 --color pick                                       # 在線稿內油漆桶填色
pixelcanvas shape ellipse 470,440,500,470 --color "#ffffff"                 # 眼睛高光
pixelcanvas erase 400,300,460,360 --layer line                              # 擦除一個框內的筆跡
```

## 重畫，但是……

需求可以要求參考圖沒有的東西：換一個顏色、做成海報、換一個場景。建立專案時帶上需求，照常規劃區域，並在
`pixelcanvas stage done` 之前逐項說明改動（`pixelcanvas change add ...`；單純複製用 `pixelcanvas change none`）。改動會修改「目標圖」，也就是
後續階段描摹並用來評分的圖；原圖保留為 `orig.png`。規劃完成後，改動階段負責做出這些改動，之後的階段描摹目標圖。
`pixelcanvas look --show changes` 並排顯示原圖與目標圖；全程由 `pixelcanvas next` 指出下一個指令。

換色（*「薄荷綠的頭髮與暗紅色上衣」*），區域為 `hair`、`face`、`clothes`、`background`：

```bash
pixelcanvas new pic --ref reference.jpg --mode trace --request "redraw it with mint-green hair and a dark red top"
pixelcanvas change add recolor hair --to "#8fdcb4"
pixelcanvas change add recolor clothes --to "#7b2635"
pixelcanvas stage done
pixelcanvas recolor hair --to "#8fdcb4"            # 保留陰影、線條與高光；回報色相 before -> after
pixelcanvas recolor clothes --to "#7b2635"
pixelcanvas region refine hair --by hue            # 仍有舊顏色的零星色塊時
```

海報（*「極簡海報，標題 SLOW SUMMER，奶油底色上的海軍藍線稿，珊瑚色點綴」*）：

```bash
pixelcanvas change add canvas --size 900x1200 --place 186,240,814,930 --color "#f3e9d8"
pixelcanvas change add palette map --to "#1d2a44,#f3e9d8" --accent "clothes=#e8644a"
pixelcanvas change add graphic disc 260,295,860,745 --color "#e97a62"
pixelcanvas change add text "SLOW SUMMER" --box 70,40,930,175 --font display --weight black --color "#1d2a44"
pixelcanvas stage done
pixelcanvas canvas --size 900x1200 --place 186,240,814,930 --color "#f3e9d8"
pixelcanvas palette map --to "#1d2a44,#f3e9d8" --accent "clothes=#e8644a"
pixelcanvas graphic disc 260,295,860,745 --color "#e97a62"
pixelcanvas style poster                           # 目標圖看起來像海報時，改動關卡會要求這一步
```

新場景（*「站在夕陽下的山中湖泊前」*）：背景可以是另一個 pixelcanvas 專案、一張圖檔、一個漸層或單一顏色。

```bash
pixelcanvas change add cutout figure --regions face,hair,clothes
pixelcanvas change add background --gradient "#f2a65a,#5b5f8f"     # 或 --from lake.pixelcanvas、--from sky.png、--color "#88aacc"
pixelcanvas change add light
pixelcanvas stage done
pixelcanvas cutout figure --regions face,hair,clothes   # 回報人物內殘留的底色，並框出三個最弱的邊緣位置
pixelcanvas background --gradient "#f2a65a,#5b5f8f"
pixelcanvas light                                       # 把人物染向背景的光色，並加上輪廓光
pixelcanvas shadow figure                               # 人物站在畫面內時加上接觸陰影
```

## 依文字需求作畫

沒有參考圖時，pixelcanvas 的做法相同：agent 用方框規劃需求中的主體與部件，再用構圖工具（`sketch figure`、`sketch head`、`face`、
`line`、`flat auto`、`shade`、`eye`、`hair --flow`、`fold`、`text`）依階段畫出來，最後 `pixelcanvas polish` 為完稿統一光線並做最後的整理：

```bash
pixelcanvas new pic --size 1000x1200 --request "a chibi fox mage holding a staff"
pixelcanvas plan subject fox --box 170,60,760,960
pixelcanvas plan part head --of fox --box 200,90,720,500 --focus 3
pixelcanvas sketch figure fox --pose hold --heads 2.5
```

場景中的人物先在 3D 骨架上擺姿勢，再以 2D 畫在上面。鏡頭、人物接觸的方塊、姿勢、衣服和頭髮都在 3D 中；`render` 把它們轉成淡色的
參考層（不會匯出），`draw` 在上面逐部件畫出動畫風格的線稿，每一筆都有編號：

```bash
pixelcanvas camera --horizon 620 --eye-height 1.2
pixelcanvas block bench box --size 1.8,0.45,0.45 --at 520,860
pixelcanvas pose girl --preset sit-ledge --on bench --yaw -20 --set "head: turn 30" --hand-shape-r relaxed
pixelcanvas dress girl blouse,skirt,thigh-highs,shoes --color navy
pixelcanvas hair3d girl --style twintails --bangs blunt --color pink
pixelcanvas render                     # 3D 參考層
pixelcanvas draw all                   # 在參考層上畫 2D 線稿
pixelcanvas flat auto
pixelcanvas shade all                  # 硬邊的賽璐璐陰影
pixelcanvas face girl --expression smile --eyes blue
```

重新擺姿勢時參考層會自動更新；`pixelcanvas draw girl --part arm-r` 只重畫一個部件。

一行設定整個人物，一個指令檔整批執行：

```bash
pixelcanvas figure girl --pose run --at 360,820 --heads 6.5 --yaw 20 --hand-shape-r fist   --wear "blouse:white,skirt:navy:pleated,thigh-highs:black,shoes:brown"   --hair "twintails,blunt,pink,bows:black" --face "smile,blue"
pixelcanvas do @steps.txt                        # 鏡頭、方塊、人物：每行一個指令
pixelcanvas do "draw all; flat auto; shade all"
pixelcanvas help pose,figure,block               # 一次列出多個指令的選項與值
```

### Claude Code 外掛

外掛提供 `/pixelcanvas:draw <需求>` 與四個子代理，從最粗的決定往細處畫一張新圖：`pixelcanvas-director` 寫出企劃（鏡頭、
每個人物的動作與姿勢、衣服、光線、配色），`pixelcanvas-blocker` 擺好人物並產生參考層，`pixelcanvas-painter` 作畫並完稿，
`pixelcanvas-critic` 檢查參考層與完稿，並為每個問題指出要執行的指令。

## 指令

`pixelcanvas help`：

```text
pixelcanvas: a paint program. Coordinates are 0-1000 on each axis (x right, y down).
after each command: read the report, open VIEW (the canvas) and, to compare, VIEW-REF (the reference)
setup:   new  style  mode hand  status  next  todo [--all]  stage  undo [n]  redo
look:    look [region|box] [--show both|canvas|ref|diff|marks|layers] [--grid]   marks   pick   palette   score
regions: region add <name> --marks 3,4 | --box x0,y0,x1,y1 | --poly "x,y ..." | --wand x,y  [--focus 1-3]; region rest background; region focus <name> 3; region list; region remove <name>
stages:  sketch  ink  flat  shade  paint [--size N]  airbrush  detail [--level 2|1]  dot  retouch <region>   (each takes a region, default all; no options = the engine's own settings for the style)
hand:    stroke  fill  shape  gradient  erase  smooth  adjust   (act on the stage's layer or --layer)
changes: change add <kind> ...  change list  change drop <n>  change none   (state what the request changes from the reference; kinds: canvas recolor palette graphic cutout background light shadow text shape)
         recolor <region> --to "#hex"   region refine <region> --by lines|hue   canvas --size WxH --place x0,y0,x1,y1 [--color "#hex"]   palette map --to "dark,...,light" [--accent "region=#hex"]   graphic disc|rect|ring|frame <box> --color "#hex"
         cutout <subject> --regions a,b,c   background --from <pixelcanvas project|picture> | --gradient "#a,#b" | --color "#hex"   light [--tint "#hex"] [--rim left|right]   shadow <subject>
free:    no reference: plan  palette make|pick|set  sketch figure|head|ellipse|box|curve  line  band  face  eye  hair  locks  blush  hand  wear  fold  crease  tufts  mark  pattern  scatter  surface  highlight  rim  shadow --from <part> --onto <part>  reflect  wet  haze  atmosphere  glow  cloud  ground  foliage  window  linestyle
3D:      camera --horizon <y> --eye-height <m>   pose <subject> --preset run --at x,y --yaw <deg> --set "arm-r: raise 90; knee-l: 60" [--on <block>] [--hand-r <block|x,y>]   pose list   block <name> box|stairs|cylinder|plane|railing --at x,y --size w,d,h   prop <name> --kind sword --hand r   form <subject>   dress <subject> blouse,skirt,shoes --color navy   dress list   hair3d <subject> --style long|twintails|... --bangs blunt --color pink   render (the 3D guide)   draw <subject>|all|scene [--part hair|face|arm-r|...]   shade <subject>|all (cel shadows)   face <subject> --expression crying
finish:  polish (one finishing pass over a drawn picture; --no-<part> leaves a part out)   sharpen (the finishing sharpen, fitted to the reference)   fx <glow|grain|blur|vignette|paper|sharpen|levels> [region] [--amount 0..1]   text "<string>" --box x0,y0,x1,y1 [--font --color --align --outline px,#hex --vertical --fit]   fonts
other:   layer  export <file.png> [--psd] [--ora] [--timelapse film.gif|film.mp4] [--stages dir]   do "<cmd>; <cmd>"   help [command]
NEXT: pixelcanvas next
```

任何指令後加 `--help` 會印出它的選項與範例。`pixelcanvas export final.png --psd --ora` 另外輸出可在 Photoshop、Krita 或 GIMP
開啟的圖層；`--timelapse film.gif` 重播日誌；`--stages dir` 為每個階段存一張圖。

## 開發

```bash
git clone https://github.com/treeleaves30760/pixelcanvas
cd pixelcanvas
pip install -e ".[test]"
python -m pytest tests -q
```

程式碼在 `pixelcanvas/`，測試在 `tests/`，skill 在 `skills/pixelcanvas/SKILL.md`。
[CLAUDE.md](CLAUDE.md) 有專案地圖與發布步驟；[CHANGELOG.md](CHANGELOG.md) 是版本紀錄。
0.6.0 版以 pixelcanvas 取代了舊的「用程式碼畫圖」工具。

## 授權

MIT
