# -*- coding: utf-8 -*-
"""命令列小工具：``pixelcanvas info 圖.png`` / ``pixelcanvas preview 圖.png 小圖.jpg [寬]``。"""
from __future__ import annotations

import sys

from .canvas import Canvas

USAGE = """用法：
  pixelcanvas info 圖.png                  尺寸與亮度統計
  pixelcanvas preview 圖.png 小圖.jpg [寬]  輸出縮圖（預設寬 960）
（渲染場景請用專案根目錄的 tools/render.py）"""


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2 or argv[0] not in ("info", "preview"):
        print(USAGE)
        return 1
    cv = Canvas.load(argv[1])
    if argv[0] == "info":
        cv.stats()
    else:
        out = argv[2] if len(argv) > 2 else "preview.jpg"
        pw = int(argv[3]) if len(argv) > 3 else 960
        cv.resize(pw, max(1, round(cv.h * pw / cv.w))).save(out)
        print("saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
