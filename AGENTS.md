# AGENTS.md — 給所有 AI 程式助理（Codex、Cursor、Gemini CLI、Copilot、Aider…）

這個專案讓**不能生成圖片的模型**也能畫出好看的圖：用 numpy + Pillow 從噪聲、遮罩、混合算出每一個像素，
再透過「渲染 → 檢查 → 修正」的迴圈把圖修好。沒有任何素材圖。

**完整工作說明在 [`skills/pixelcanvas/SKILL.md`](skills/pixelcanvas/SKILL.md)，畫圖前先讀它。**
參考資料：[API 速查](skills/pixelcanvas/references/api.md)、[技巧庫與配色](skills/pixelcanvas/references/techniques.md)、
[檢查清單](skills/pixelcanvas/references/review.md)。

## 一分鐘版

```bash
pip install -r requirements.txt
python tools/new_scene.py my_scene             # 建立 scenes/my_scene.py（或複製最接近的範例場景）
python tools/render.py my_scene --scale 0.5    # 草稿 → outputs/my_scene_x0.5.png + _preview.jpg
python tools/look.py report outputs/my_scene_x0.5.png   # 看不到圖？用文字報告檢查
python tools/render.py my_scene                # 滿意後渲染全尺寸，複製到 gallery/
python -m pytest -q                            # 改了套件或文件後跑測試（< 2 秒）
```

1. 先構圖（眼平線、光源、遠中近、焦點、調色盤）寫成場景檔開頭的註解，再寫程式。
2. 一張圖要 4~7 輪；每輪寫下 2~4 個最嚴重的問題，一次只改 2~4 個參數，seed 固定。
3. 能讀圖片就看 `_preview.jpg`；不能就用 `tools/look.py report`。**不要沒檢查就交付。**

## 專案慣例

- 座標 x 向右、y 向下，`arr[y, x]`；遮罩 float32 (H, W) 0~1；位置用比例，像素常數以 1080p 為基準乘 `S = H / 1080`。
- 順序：天空 → 遠 → 近 → 水面（倒影取樣目前畫面）→ 前景 → 體積光 → 後製；光源可 > 1，最後 `soft_clip`。
- 全部向量化，不要逐像素 Python 迴圈。
- 通用的新配方放 `pixelcanvas/scenery.py`，一次性的放場景檔；新技巧可補進 `references/techniques.md`
  （裡面的 ```python 區塊會被 `tests/test_skill_docs.py` 實際執行）。
- `outputs/` 不進 git；定稿放 `gallery/`。
