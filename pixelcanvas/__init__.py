# -*- coding: utf-8 -*-
"""
pixelcanvas — 讓 AI 用「像素級運算」畫圖 / 修圖的工具箱（numpy + Pillow）
================================================================================

兩層 API：

* 低階（``from pixelcanvas import *``）：Canvas 畫布、噪聲、遮罩、筆刷、混合、濾鏡、顏色。
  任何東西都能從這層畫出來。
* 高階配方（``from pixelcanvas.scenery import *``）：Scene + 天空 / 太陽 / 雲 / 山 / 霧 / 河流 /
  樹 / 光線 / 後製等「景物」函數，用來快速組合一張風景；也是寫新配方的範本。

慣例
----
* 座標：x 向右、y 向下；陣列索引 ``arr[y, x]``；``cv.xy()`` 取 (X, Y) 網格、``cv.uv()`` 取 0~1 的 (U, V)。
* 遮罩：float32 (H, W)，0 = 不影響、1 = 完全影響。
* 顏色：'#rrggbb' / (r, g, b) 0~1 / (255, 128, 0) / 整張 (H, W, 3)；允許 HDR > 1。
* 流程：鋪底（漸層）→ 遠到近一層層依遮罩疊上 → 後製（bloom / soft_clip / 對比 / 暗角）→ save。

最小範例
--------
>>> from pixelcanvas import *
>>> cv = Canvas(640, 360)
>>> cv.gradient([(0, '#0b1a3a'), (1, '#f6c28f')])          # 天空漸層
>>> X, Y = cv.xy()
>>> cv.glow(400, 200, 60, '#fff0c0', strength=1.2)          # 光暈 (HDR)
>>> ridge = 220 - 60 * fbm1d(640, 120, seed=1)              # 山稜線：每一欄的 y
>>> cv.paint('#223344', clamp01(Y - ridge[None, :] + 0.5))  # 稜線以下上色 = 山
>>> cv.put(10, 10, '#ff0000')                               # 單一像素
>>> cv.soft_clip(0.8).save('demo.png', preview='demo_small.jpg')
"""
from . import color, noise, masks, brush, filters
from .color import *       # noqa: F401,F403
from .noise import *       # noqa: F401,F403
from .masks import *       # noqa: F401,F403
from .brush import *       # noqa: F401,F403
from .filters import *     # noqa: F401,F403
from .canvas import Canvas

__version__ = "0.2.0"
__all__ = ["Canvas", "__version__"] + color.__all__ + noise.__all__ + masks.__all__ + brush.__all__ + filters.__all__
