# -*- coding: utf-8 -*-
"""Pictograms: kitchen tools, app and productivity glyphs, weather (100 x 100 box, y down)."""
import math

from . import circle, ellipse, ring, rect, poly, bar, arc, smooth, tube, frame, reverse, cut, rotate_points

R = rotate_points
YELLOW = "#f2b134"


def _rb(x1, y1, x2, y2, w, ang, cx=50.0, cy=50.0):
    (a, b), (c, d) = R([(x1, y1), (x2, y2)], ang, cx, cy)
    return bar(a, b, c, d, w)


def _rp(points, ang, r=0.0, cx=50.0, cy=50.0):
    return poly(R(points, ang, cx, cy), r)


def _cloud(g=0.0, dx=0.0, dy=0.0, k=1.0):
    """The shared cloud outline as a list of shapes, grown by ``g`` units (used for gaps around a cloud)."""
    def X(x):
        return 50 + (x - 50) * k + dx

    def Y(y):
        return 50 + (y - 50) * k + dy
    return [circle(X(32), Y(58), (16 + g) * k), circle(X(52), Y(46), (22 + g) * k), circle(X(72), Y(58), (16 + g) * k),
            rect(X(16) - g * k, Y(58) - g * k, 72 * k + 2 * g * k, (20 + 2 * g) * k, (10 + g) * k)]


def _rays(cx, cy, r0, r1, angles, w=5):
    out = []
    for a in angles:
        c, s_ = math.cos(math.radians(a)), math.sin(math.radians(a))
        out.append(bar(cx + r0 * c, cy + r0 * s_, cx + r1 * c, cy + r1 * s_, w))
    return out


KITCHEN = [
    dict(name="jar", category="kitchen", aliases=["glass jar", "mason jar", "jam jar", "preserving jar", "罐子", "玻璃罐"],
         keywords=["pantry", "storage", "jam", "pickle", "kitchen"], desc="glass jar with a lid and a label",
         body=[rect(22, 8, 56, 14, 3), rect(27, 22, 46, 6, 2), rect(16, 28, 68, 64, (10, 10, 12, 12))],
         detail=[rect(26, 50, 48, 28, 3), rect(22, 12, 56, 2.4, 1)],
         anchors={"label": (50, 64)}),
    dict(name="spice jar", category="kitchen", aliases=["spice", "seasoning", "herb jar", "調味罐", "香料罐"],
         keywords=["pantry", "cooking", "salt", "pepper", "kitchen"], desc="small spice jar with a screw cap",
         body=[rect(30, 8, 40, 16, 3), rect(26, 24, 48, 68, (4, 4, 8, 8))],
         detail=[rect(32, 44, 36, 32, 2.5), rect(30, 12.6, 40, 2.4, 1)],
         accent=[circle(50, 60, 8)], accent_color="#c94f3a", anchors={"label": (50, 60)}),
    dict(name="ladle", category="kitchen", aliases=["soup ladle", "scoop", "湯勺"],
         keywords=["soup", "cooking", "serving", "kitchen"], desc="soup ladle with a round bowl",
         body=[bar(44, 60, 80, 14, 6.4), ellipse(34, 68, 24, 22), circle(82, 12, 7)],
         detail=[ellipse(34, 68, 16.5, 14.5), circle(82, 12, 2.8)],
         anchors={"bowl": (34, 68)}),
    dict(name="spatula", category="kitchen", aliases=["turner", "flipper", "鍋鏟"],
         keywords=["cooking", "flip", "pan", "kitchen"], desc="slotted spatula, tilted",
         body=[_rp([(32, 8), (68, 8), (68, 46), (32, 46)], 30, 5), _rb(50, 44, 50, 92, 9, 30)],
         detail=[_rb(40, 14, 40, 38, 3.4, 30), _rb(50, 14, 50, 38, 3.4, 30), _rb(60, 14, 60, 38, 3.4, 30)],
         anchors={"blade": (50, 28)}),
    dict(name="whisk", category="kitchen", aliases=["egg whisk", "balloon whisk", "打蛋器"],
         keywords=["baking", "cooking", "eggs", "kitchen"], desc="balloon whisk with a handle",
         body=[ellipse(50, 34, 26, 30) + " " + reverse(ellipse(50, 34, 20.5, 24.5)),
               ellipse(50, 34, 13, 30) + " " + reverse(ellipse(50, 34, 8, 25)), bar(50, 6, 50, 60, 3.6),
               rect(44, 58, 12, 36, 5.5)],
         detail=[rect(47, 70, 6, 16, 3)],
         anchors={"handle": (50, 76)}),
    dict(name="rolling pin", category="kitchen", aliases=["dough roller", "擀麵棍", "桿麵棍"],
         keywords=["baking", "dough", "pastry", "kitchen"], desc="rolling pin with two handles",
         body=[rect(22, 34, 56, 32, 12), bar(6, 50, 24, 50, 10), bar(76, 50, 94, 50, 10)],
         detail=[bar(30, 43, 70, 43, 3.2)],
         anchors={"roller": (50, 50)}),
    dict(name="cutting board", category="kitchen", aliases=["chopping board", "chopping block", "砧板", "菜板"],
         keywords=["prep", "knife", "wood", "kitchen"], desc="cutting board with a hanging hole",
         body=[rect(14, 22, 72, 70, 9), rect(34, 6, 32, 26, 11)],
         detail=[circle(50, 18, 5.5), rect(28, 46, 44, 3, 1.5), rect(28, 58, 44, 3, 1.5), rect(28, 70, 30, 3, 1.5)],
         anchors={"center": (50, 55)}),
    dict(name="chef hat", category="kitchen", aliases=["toque", "chef's hat", "cook hat", "廚師帽", "大廚帽"],
         keywords=["chef", "cook", "restaurant", "uniform", "bakery"], desc="tall puffy chef's toque",
         body=[circle(30, 40, 17), circle(50, 30, 21), circle(70, 40, 17), rect(26, 42, 48, 32, 3),
               rect(26, 72, 48, 18, 4)],
         detail=[rect(36, 50, 3, 18, 1.5), rect(48.5, 50, 3, 18, 1.5), rect(61, 50, 3, 18, 1.5),
                 rect(26, 72.5, 48, 2.6, 1)],
         anchors={"top": (50, 10)}),
    dict(name="shelf", category="home", aliases=["wall shelf", "shelves", "floating shelf", "kitchen shelf", "層架", "架子"],
         keywords=["storage", "books", "plant", "kitchen", "decor"], desc="wall shelf with books and a plant",
         body=[rect(6, 66, 88, 8, 3), poly([(16, 74), (26, 74), (16, 90)], 1), poly([(74, 74), (84, 74), (84, 90)], 1),
               rect(14, 32, 9, 34, 1.5), rect(24, 26, 9, 40, 1.5), rect(34, 38, 8, 28, 1.5),
               rect(58, 48, 22, 18, (1, 1, 5, 5)), tube([(69, 50), (66, 38), (60, 28)], [4, 3.4, 2.6]),
               tube([(69, 50), (72, 36), (80, 26)], [4, 3.4, 2.6]), tube([(69, 50), (69, 36), (69, 22)], [4, 3.4, 2.6])],
         detail=[rect(15.5, 38, 6, 2.4, 1), rect(25.5, 34, 6, 2.4, 1), rect(35.5, 44, 5, 2.4, 1),
                 rect(58, 54, 22, 2.6, 0)],
         anchors={"plant": (69, 24)}),
    dict(name="refrigerator", category="kitchen", aliases=["fridge", "freezer", "冰箱"],
         keywords=["cold", "food", "appliance", "kitchen"], desc="two-door refrigerator",
         body=[rect(26, 6, 48, 88, 7)],
         detail=[rect(27, 36, 46, 2.8, 0), rect(63, 14, 4, 16, 2), rect(63, 45, 4, 24, 2)],
         anchors={"door": (50, 64)}),
    dict(name="stove", category="kitchen", aliases=["cooker", "oven", "range", "gas stove", "爐子", "烤箱", "瓦斯爐"],
         keywords=["cooking", "appliance", "kitchen", "burner"], desc="stove with two burners and an oven door",
         body=[rect(12, 38, 76, 54, 6), ellipse(32, 29, 16, 5), ellipse(68, 29, 16, 5), bar(32, 33, 32, 38, 3),
               bar(68, 33, 68, 38, 3)],
         detail=[rect(20, 44, 60, 4, 2), rect(22, 58, 56, 28, 4)],
         anchors={"oven": (50, 72)}),
    dict(name="toaster", category="kitchen", aliases=["bread toaster", "烤麵包機", "吐司機"],
         keywords=["breakfast", "toast", "appliance", "kitchen"], desc="toaster with two slices of toast",
         body=[rect(8, 40, 84, 44, (10, 10, 6, 6)), rect(18, 84, 12, 6, 2), rect(70, 84, 12, 6, 2)],
         detail=[rect(76, 50, 6, 18, 3), circle(20, 62, 4.5), rect(30, 40, 40, 4, 0)],
         accent=[rect(26, 20, 22, 24, (9, 9, 0, 0)), rect(52, 20, 22, 24, (9, 9, 0, 0))], accent_color="#d59a4a",
         anchors={"toast": (50, 30)}),
    dict(name="pepper mill", category="kitchen", aliases=["pepper grinder", "salt mill", "grinder", "胡椒研磨罐", "研磨罐"],
         keywords=["spice", "seasoning", "table", "kitchen"], desc="wooden pepper mill",
         body=[circle(50, 12, 7.5), poly([(36, 20), (64, 20), (60, 50), (68, 92), (32, 92), (40, 50)], 3)],
         detail=[bar(38, 32, 62, 32, 2.8), bar(36.5, 72, 63.5, 72, 2.8)],
         anchors={"top": (50, 12)}),
    dict(name="colander", category="kitchen", aliases=["strainer", "sieve", "濾盆", "瀝水籃"],
         keywords=["pasta", "drain", "cooking", "kitchen"], desc="bowl colander with holes and two handles",
         body=[rect(8, 38, 84, 9, 4.5), "M12 46 L88 46 Q84 90 50 92 Q16 90 12 46 Z", bar(10, 42, 6, 56, 5), bar(90, 42, 94, 56, 5)],
         detail=[circle(x, y, 2.6) for x, y in [(30, 58), (50, 58), (70, 58), (40, 70), (60, 70), (50, 82), (22, 52), (78, 52)]],
         anchors={"center": (50, 64)}),
]


def _check(x, y, w=3.4):
    return tube([(x, y + 3), (x + 4, y + 8), (x + 12, y - 3)], w)


APPS = [
    dict(name="notes", category="work", aliases=["notepad", "memo", "jotter", "便條紙", "筆記", "記事本"],
         keywords=["write", "app", "ideas", "paper", "productivity"], desc="notepad page with lines and a coloured header",
         body=[rect(18, 8, 64, 84, 8)],
         detail=[rect(28, 38, 44, 3.6, 1.8), rect(28, 50, 44, 3.6, 1.8), rect(28, 62, 44, 3.6, 1.8),
                 rect(28, 74, 28, 3.6, 1.8)],
         accent=[rect(18, 8, 64, 18, (8, 8, 0, 0))], accent_color=YELLOW, anchors={"header": (50, 17)}),
    dict(name="checklist", category="work", aliases=["tasks", "task list", "to-do list", "todo list", "to do", "todo",
                                                      "待辦清單", "清單", "任務"],
         keywords=["app", "productivity", "done", "plan", "tick"], desc="card with ticked task rows",
         body=[rect(14, 8, 72, 84, 8)],
         detail=[_check(24, 26), rect(46, 27, 30, 4.2, 2.1), _check(24, 46), rect(46, 47, 30, 4.2, 2.1),
                 frame(24, 64, 14, 14, 2.6, 2.4), rect(46, 67, 24, 4.2, 2.1)],
         anchors={"row1": (46, 28)}),
    dict(name="chat bubble", category="work", aliases=["chat", "chat bubbles", "conversation", "dialog", "messaging",
                                                        "對話框", "聊天室", "聊天泡泡"],
         keywords=["message", "talk", "app", "support", "reply"], desc="two overlapping chat bubbles",
         body=[rect(6, 8, 58, 42, 11), poly([(14, 48), (14, 64), (30, 50)], 2), cut(rect(32, 34, 66, 56, 15)),
               rect(36, 38, 58, 42, 11), poly([(86, 78), (86, 94), (70, 80)], 2)],
         detail=[circle(51, 59, 3.4), circle(65, 59, 3.4), circle(79, 59, 3.4)],
         anchors={"front": (65, 59)}),
    dict(name="sun behind cloud", category="nature", aliases=["partly cloudy", "sunny intervals", "weather",
                                                               "weather icon", "cloud and sun", "天氣", "晴時多雲"],
         keywords=["forecast", "sky", "app", "mild", "day"], desc="sun peeking out behind a cloud",
         body=_cloud(0, 4, 14),
         accent=[circle(62, 36, 17)] + _rays(62, 36, 24, 32, (-90, -45, 0, 45, 180, 225)) + [cut(s) for s in _cloud(5, 4, 14)],
         accent_color=YELLOW, anchors={"sun": (62, 36), "cloud": (50, 64)}),
    dict(name="magnifier", category="work", aliases=["magnifying glass", "search", "zoom", "find", "放大鏡", "搜尋"],
         keywords=["lookup", "app", "inspect", "detective"], desc="magnifying glass",
         body=[ring(42, 42, 32, 9), bar(66, 66, 88, 88, 11)],
         anchors={"lens": (42, 42)}),
    dict(name="bookmark", category="work", aliases=["saved", "save for later", "書籤", "收藏"],
         keywords=["read", "app", "favourite", "tag"], desc="ribbon bookmark",
         body=[poly([(24, 8), (76, 8), (76, 92), (50, 70), (24, 92)], 4)],
         detail=[rect(36, 20, 28, 3.6, 1.8)],
         anchors={"top": (50, 8)}),
    dict(name="user avatar", category="work", aliases=["profile", "account", "avatar", "person icon", "使用者", "帳號", "個人檔案"],
         keywords=["login", "app", "member", "contact", "head and shoulders"], desc="head and shoulders profile icon",
         body=[circle(50, 32, 19), "M14 92 Q14 56 50 56 Q86 56 86 92 Z"],
         anchors={"head": (50, 32)}),
    dict(name="download", category="work", aliases=["download arrow", "save", "下載"],
         keywords=["app", "file", "transfer", "arrow"], desc="arrow down into a tray",
         body=[bar(50, 10, 50, 50, 14, cap="butt"), poly([(24, 44), (76, 44), (50, 72)], 3), rect(12, 78, 76, 13, 4)],
         anchors={"tray": (50, 84)}),
    dict(name="upload", category="work", aliases=["upload arrow", "share", "上傳"],
         keywords=["app", "file", "transfer", "arrow"], desc="arrow up from a tray",
         body=[poly([(24, 36), (76, 36), (50, 8)], 3), bar(50, 34, 50, 68, 14, cap="butt"), rect(12, 78, 76, 13, 4)],
         anchors={"tray": (50, 84)}),
    dict(name="cloud upload", category="tech", aliases=["cloud storage", "cloud sync", "backup", "雲端", "雲端上傳"],
         keywords=["app", "sync", "file", "server"], desc="cloud with an upload arrow",
         body=_cloud(0, 0, 4, 1.05),
         detail=[poly([(37, 62), (63, 62), (50, 44)], 2), bar(50, 58, 50, 78, 9, cap="butt")],
         anchors={"cloud": (50, 56)}),
    dict(name="photo", category="work", aliases=["image", "gallery", "landscape photo", "照片", "圖片", "相簿"],
         keywords=["app", "album", "photograph", "mountain"], desc="photo with a sun and mountains",
         body=[rect(8, 16, 84, 68, 9)],
         detail=[circle(30, 36, 7.5), poly([(14, 74), (38, 48), (52, 64), (66, 44), (86, 74)], 2)],
         anchors={"frame": (50, 50)}),
    dict(name="kanban board", category="work", aliases=["task board", "project board", "scrum board", "看板", "任務看板"],
         keywords=["agile", "tasks", "columns", "project", "planning"], desc="board with three columns of cards",
         body=[rect(8, 14, 84, 72, 8)],
         detail=[rect(15, 22, 20, 12, 2.4), rect(15, 38, 20, 12, 2.4), rect(15, 54, 20, 12, 2.4),
                 rect(40, 22, 20, 12, 2.4), rect(40, 38, 20, 12, 2.4), rect(65, 22, 20, 12, 2.4)],
         anchors={"board": (50, 50)}),
    dict(name="funnel", category="work", aliases=["filter", "filter funnel", "sales funnel", "漏斗", "篩選"],
         keywords=["app", "sort", "conversion", "marketing"], desc="funnel",
         body=[poly([(8, 14), (92, 14), (58, 52), (58, 86), (42, 76), (42, 52)], 3)],
         anchors={"spout": (50, 80)}),
    dict(name="link", category="work", aliases=["chain link", "hyperlink", "url", "連結", "鏈結"],
         keywords=["web", "url", "attach", "chain"], desc="two chain links",
         body=[ellipse(36, 64, 24, 13, -45) + " " + reverse(ellipse(36, 64, 16.5, 6, -45)),
               ellipse(64, 36, 24, 13, -45) + " " + reverse(ellipse(64, 36, 16.5, 6, -45)), bar(41, 59, 59, 41, 7)],
         anchors={"center": (50, 50)}),
    dict(name="sliders", category="work", aliases=["equalizer", "adjustments", "controls", "preferences", "調整", "調音台"],
         keywords=["settings", "mixer", "app", "filters", "config"], desc="three sliders with round knobs",
         body=[bar(12, 24, 88, 24, 5), cut(circle(32, 24, 12)), circle(32, 24, 8),
               bar(12, 50, 88, 50, 5), cut(circle(66, 50, 12)), circle(66, 50, 8),
               bar(12, 76, 88, 76, 5), cut(circle(44, 76, 12)), circle(44, 76, 8)],
         anchors={"knob": (66, 50)}),
    dict(name="toggle switch", category="work", aliases=["switch", "toggle"],
         keywords=["settings", "app", "ui", "control"], desc="toggle switch turned on",
         body=[rect(6, 30, 88, 40, 20)],
         detail=[circle(74, 50, 13)],
         anchors={"knob": (74, 50)}),
    dict(name="video camera", category="work", aliases=["video call", "webcam", "視訊"],
         keywords=["meeting", "app", "stream", "record", "zoom"], desc="video camera facing right",
         body=[rect(8, 26, 58, 48, 9), poly([(70, 42), (92, 28), (92, 72), (70, 58)], 3)],
         detail=[circle(20, 38, 3.4)],
         anchors={"lens": (80, 50)}),
]

WEATHER = [
    dict(name="moon behind cloud", category="nature", aliases=["cloudy night", "night cloud", "night weather", "多雲夜晚"],
         keywords=["weather", "sky", "forecast", "evening", "dark"], desc="crescent moon behind a cloud",
         body=_cloud(0, 4, 14),
         accent=[circle(60, 34, 24), cut(circle(48, 42, 20))] + [cut(s) for s in _cloud(5, 4, 14)],
         accent_color="#f2d57a", anchors={"moon": (50, 34)}),
    dict(name="windy", category="nature", aliases=["wind", "breeze", "gust", "風", "颳風"],
         keywords=["weather", "air", "forecast", "storm", "blow"], desc="three wind streaks with curls",
         body=[bar(10, 36, 62, 36, 7), arc(62, 26, 10, 90, 360, 6.4), bar(10, 56, 76, 56, 7),
               arc(76, 66, 10, -90, 180, 6.4), bar(22, 80, 58, 80, 7)],
         anchors={"center": (50, 50)}),
    dict(name="snow cloud", category="nature", aliases=["snowing", "snowfall", "雪雲"],
         keywords=["weather", "winter", "forecast", "flakes", "cold"], desc="cloud with falling snowflakes",
         body=_cloud(0, 0, -16, 0.95) + [circle(28, 74, 4.6), circle(50, 84, 4.6), circle(72, 74, 4.6),
                                          circle(39, 92, 3.6), circle(61, 92, 3.6)],
         anchors={"cloud": (50, 40)}),
    dict(name="fog", category="nature", aliases=["foggy", "mist", "haze", "霧", "起霧"],
         keywords=["weather", "low visibility", "forecast", "cloud", "morning"], desc="cloud above layers of mist",
         body=_cloud(0, 0, -20, 0.9) + [bar(14, 64, 86, 64, 6.4), bar(24, 78, 76, 78, 6.4), bar(34, 92, 66, 92, 6.4)],
         anchors={"cloud": (50, 36)}),
    dict(name="storm cloud", category="nature", aliases=["thunderstorm", "lightning storm", "雷雨", "雷暴"],
         keywords=["weather", "lightning", "forecast", "rain", "severe"], desc="dark cloud with a lightning bolt",
         body=_cloud(0, 0, -14, 0.95),
         accent=[poly([(54, 52, 1), (34, 74, 1), (47, 74, 1), (40, 94, 1), (68, 68, 1), (54, 68, 1), (63, 52, 1)])],
         accent_color=YELLOW, anchors={"bolt": (52, 74)}),
]

PICTOGRAMS = KITCHEN + APPS + WEATHER
