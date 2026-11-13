# -*- coding: utf-8 -*-
"""Pictograms: devices, biology, storybook props and everyday things for posters and infographics
(100 x 100 box, y down)."""
import math

from . import circle, ellipse, ring, rect, poly, bar, arc, smooth, tube, frame, reverse, cut, rotate_points

R = rotate_points
GREEN = "#3fb37f"
RED = "#e2574c"
YELLOW = "#f2b134"


def _rb(x1, y1, x2, y2, w, ang, cx=50.0, cy=50.0):
    (a, b), (c, d) = R([(x1, y1), (x2, y2)], ang, cx, cy)
    return bar(a, b, c, d, w)


def _rp(points, ang, r=0.0, cx=50.0, cy=50.0):
    return poly(R(points, ang, cx, cy), r)


def _star(cx, cy, r1, r2, n=5, rot=-90.0, rr=0.8):
    pts = []
    for i in range(n * 2):
        a = math.radians(rot + 180.0 * i / n)
        r = r1 if i % 2 == 0 else r2
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a), rr))
    return poly(pts)


TECH = [
    dict(name="charging case", category="tech", aliases=["earbud case", "airpods case", "charging box", "耳機充電盒", "充電盒"],
         keywords=["earbuds", "wireless", "audio", "battery", "open lid"], desc="open earbud charging case with two wells",
         body=[rect(10, 46, 80, 44, 14), rect(16, 10, 68, 38, (12, 12, 0, 0))],
         detail=[circle(34, 64, 9), circle(66, 64, 9), rect(20, 16, 60, 2.6, 1.3), rect(46, 80, 8, 3, 1.5)],
         accent=[circle(50, 56, 2.6)], accent_color="#4fd1a5", anchors={"light": (50, 56)}),
    dict(name="terminal window", category="tech", aliases=["command line", "console", "cli", "terminal", "命令列", "終端機"],
         keywords=["code", "developer", "programming", "prompt", "bash"], desc="terminal window with a prompt and a cursor",
         body=[rect(6, 14, 88, 72, 9)],
         detail=[circle(17, 24, 3), circle(27, 24, 3), circle(37, 24, 3), rect(6, 32, 88, 2.6, 0),
                 tube([(18, 46), (28, 54), (18, 62)], 4.6), rect(34, 62, 18, 4.6, 2.3), rect(36, 46, 30, 3.6, 1.8)],
         anchors={"prompt": (24, 54)}),
    dict(name="game controller", category="tech", aliases=["gamepad", "joystick", "console controller", "遊戲手把", "搖桿"],
         keywords=["gaming", "play", "video games", "console", "esports"], desc="game controller with a d-pad and buttons",
         body=[smooth([(8, 58), (16, 34), (36, 30), (64, 30), (84, 34), (92, 58), (90, 78), (78, 82), (64, 66), (36, 66),
                       (22, 82), (10, 78)])],
         detail=[rect(23, 44, 18, 5.6, 1.4), rect(29.2, 37.8, 5.6, 18, 1.4), circle(68, 42, 3.8), circle(78, 50, 3.8),
                 circle(68, 58, 3.8), circle(58, 50, 3.8)],
         anchors={"pad": (32, 46)}),
    dict(name="router", category="tech", aliases=["wifi router", "wireless router", "modem", "路由器", "無線基地台"],
         keywords=["internet", "network", "wifi", "home", "broadband"], desc="wifi router with two antennas",
         body=[rect(8, 56, 84, 28, 8), bar(24, 58, 18, 16, 4.4), bar(76, 58, 82, 16, 4.4)],
         detail=[circle(24, 70, 2.8), circle(36, 70, 2.8), circle(48, 70, 2.8), rect(62, 68, 22, 4, 2)],
         anchors={"antenna": (18, 16)}),
    dict(name="drone", category="tech", aliases=["quadcopter", "uav", "multicopter", "無人機", "空拍機"],
         keywords=["camera", "flying", "aerial", "photography", "robot"], desc="quadcopter drone, front view",
         body=[rect(34, 44, 32, 16, 7), bar(34, 50, 16, 42, 4.4), bar(66, 50, 84, 42, 4.4), rect(14, 34, 4.4, 10, 1.5),
               rect(81.6, 34, 4.4, 10, 1.5), ellipse(16, 32, 14, 3.4), ellipse(84, 32, 14, 3.4), bar(40, 60, 34, 78, 3.4),
               bar(60, 60, 66, 78, 3.4), bar(28, 78, 72, 78, 3.4), circle(50, 68, 7)],
         detail=[circle(50, 68, 3.6), circle(50, 52, 2.4)],
         anchors={"camera": (50, 68)}),
    dict(name="vr headset", category="tech", aliases=["vr goggles", "virtual reality headset", "vr", "虛擬實境", "VR眼鏡"],
         keywords=["virtual reality", "gaming", "immersive", "metaverse", "wearable"], desc="VR headset seen from the front",
         body=[rect(8, 28, 84, 46, (18, 18, 14, 14)), cut(circle(50, 76, 9)), bar(8, 44, 4, 44, 6)],
         detail=[circle(32, 50, 11), circle(68, 50, 11), rect(46, 38, 8, 3.6, 1.8)],
         anchors={"lenses": (50, 50)}),
    dict(name="usb drive", category="tech", aliases=["flash drive", "thumb drive", "usb stick", "memory stick", "隨身碟"],
         keywords=["storage", "data", "file", "transfer", "computer"], desc="USB flash drive with a connector",
         body=[rect(6, 34, 62, 32, 7), rect(68, 40, 26, 20, 2)],
         detail=[circle(18, 50, 4), rect(74, 45, 5, 4.6, 1), rect(83, 45, 5, 4.6, 1), rect(30, 46, 28, 3.6, 1.8)],
         anchors={"plug": (82, 50)}),
    dict(name="smart speaker", category="tech", aliases=["voice assistant", "home speaker", "alexa", "智慧音箱", "語音助理"],
         keywords=["audio", "home", "music", "assistant", "wifi"], desc="cylindrical smart speaker with a light ring",
         body=[rect(24, 18, 52, 74, (22, 22, 8, 8))],
         detail=[circle(x, y, 1.9) for y in (50, 60, 70, 80) for x in (34, 42, 50, 58, 66)],
         accent=[rect(30, 24, 40, 4.4, 2.2)], accent_color="#4fd1ff", anchors={"ring": (50, 26)}),
    dict(name="power bank", category="tech", aliases=["portable charger", "battery pack", "external battery", "行動電源", "充電寶"],
         keywords=["battery", "charging", "travel", "usb", "phone"], desc="power bank with a charge indicator",
         body=[rect(24, 8, 52, 84, 9)],
         detail=[circle(34, 20, 2.4), circle(44, 20, 2.4), circle(54, 20, 2.4), rect(40, 84, 20, 3.4, 1.7)],
         accent=[poly([(54, 34, 1), (36, 58, 1), (48, 58, 1), (44, 78, 1), (64, 50, 1), (52, 50, 1)])],
         accent_color=YELLOW, anchors={"port": (50, 86)}),
    dict(name="hard drive", category="tech", aliases=["hdd", "ssd", "external drive", "disk drive", "硬碟"],
         keywords=["storage", "data", "backup", "computer", "disk"], desc="hard disk drive with a platter",
         body=[rect(8, 24, 84, 52, 9)],
         detail=[circle(36, 50, 18), circle(36, 50, 4), bar(36, 50, 52, 36, 3.2), circle(80, 36, 2.4), circle(80, 46, 2.4),
                 rect(66, 64, 18, 4, 2)],
         anchors={"platter": (36, 50)}),
]

BIOLOGY = [
    dict(name="cell", category="science", aliases=["animal cell", "biology cell", "cell diagram", "細胞", "動物細胞"],
         keywords=["biology", "organelle", "membrane", "science", "school"], desc="animal cell with a nucleus and organelles",
         body=[smooth([(8, 50), (14, 30), (34, 14), (60, 12), (84, 24), (92, 48), (86, 72), (62, 88), (32, 86), (14, 70)])],
         detail=[ring(42, 46, 17, 2.8), circle(42, 46, 5.4), ellipse(70, 66, 9, 5, 30), ellipse(64, 30, 6, 3.4, -30),
                 circle(24, 72, 3.6), circle(76, 44, 3), circle(58, 76, 2.6)],
         anchors={"nucleus": (42, 46)}),
    dict(name="nucleus", category="science", aliases=["cell nucleus", "nuclear envelope", "細胞核"],
         keywords=["biology", "dna", "chromatin", "organelle", "science"], desc="nucleus with a nucleolus and chromatin",
         body=[circle(50, 50, 42)],
         detail=[ring(50, 50, 35, 2.6), circle(40, 42, 11), tube([(58, 62), (66, 54), (72, 66), (60, 74), (50, 68)], 3.2),
                 tube([(58, 34), (66, 40), (72, 30)], 3)],
         anchors={"nucleolus": (40, 42)}),
    dict(name="mitochondrion", category="science", aliases=["mitochondria", "powerhouse of the cell", "粒線體"],
         keywords=["biology", "energy", "atp", "organelle", "science"], desc="mitochondrion with folded inner membrane",
         body=[ellipse(50, 50, 44, 27)],
         detail=[ellipse(50, 50, 39, 22) + " " + reverse(ellipse(50, 50, 37, 20)), bar(28, 33, 28, 52, 3), bar(40, 67, 40, 48, 3),
                 bar(52, 33, 52, 52, 3), bar(64, 67, 64, 48, 3), bar(76, 36, 76, 52, 3)],
         anchors={"matrix": (50, 50)}),
    dict(name="chloroplast", category="science", aliases=["chloroplasts", "plastid", "葉綠體"],
         keywords=["biology", "photosynthesis", "plant", "organelle", "green"], desc="chloroplast with stacks of thylakoids",
         body=[ellipse(50, 50, 44, 30, -15)],
         detail=[rect(x - 10, y + 8 * k - 2.8, 20, 5.6, 2.8) for x, y in ((30, 36), (56, 46), (72, 34)) for k in range(3)]
         + [tube([(20, 58), (30, 62), (40, 60)], 2.4)],
         anchors={"granum": (54, 54)}),
    dict(name="leaf cross-section", category="science", aliases=["leaf anatomy", "leaf section", "stomata", "葉片剖面", "葉片構造"],
         keywords=["biology", "plant", "photosynthesis", "cells", "school"], desc="leaf layers: palisade and spongy cells",
         body=[rect(6, 14, 88, 72, 6)],
         detail=[rect(10, 19, 80, 3, 1.5), rect(10, 78, 80, 3, 1.5)]
         + [frame(14 + i * 18.5, 26, 15, 24, 3, 6) for i in range(4)]
         + [ring(x, y, 8.5, 3) for x, y in ((22, 63), (42, 65), (62, 63), (80, 65))],
         anchors={"palisade": (50, 38)}),
    dict(name="bacterium", category="science", aliases=["bacteria", "microbe", "germ", "bacillus", "細菌"],
         keywords=["biology", "microbiology", "flagellum", "science", "infection"], desc="rod bacterium with a flagellum",
         body=[rect(6, 32, 62, 34, 17), tube([(68, 50), (76, 40), (84, 60), (94, 48)], 3.4), bar(24, 33, 22, 22, 2.6),
               bar(40, 33, 40, 21, 2.6), bar(54, 33, 58, 22, 2.6)],
         detail=[tube([(18, 50), (26, 42), (34, 58), (42, 44), (50, 56)], 3)],
         anchors={"flagellum": (88, 52)}),
    dict(name="virus", category="science", aliases=["coronavirus", "virion", "pathogen", "病毒"],
         keywords=["biology", "infection", "health", "disease", "microscopic"], desc="virus particle with surface spikes",
         body=[circle(50, 50, 24)] + [bar(50 + 26 * math.cos(math.radians(a)), 50 + 26 * math.sin(math.radians(a)),
                                          50 + 36 * math.cos(math.radians(a)), 50 + 36 * math.sin(math.radians(a)), 4)
                                      for a in range(0, 360, 30)]
         + [circle(50 + 38 * math.cos(math.radians(a)), 50 + 38 * math.sin(math.radians(a)), 5) for a in range(0, 360, 30)],
         detail=[circle(50, 50, 10), circle(38, 38, 2.6), circle(62, 40, 2.6), circle(60, 62, 2.6)],
         anchors={"center": (50, 50)}),
    dict(name="neuron", category="science", aliases=["nerve cell", "brain cell", "neurone", "神經元", "神經細胞"],
         keywords=["biology", "brain", "synapse", "nervous system", "science"], desc="neuron with dendrites and a long axon",
         body=[circle(38, 42, 13), tube([(34, 36), (24, 24), (14, 16)], [8, 5, 2.4]), tube([(30, 40), (18, 38), (8, 30)], [7, 4, 2.4]),
               tube([(32, 50), (22, 58), (14, 70)], [7, 4, 2.4]), tube([(40, 32), (40, 20), (46, 10)], [7, 4, 2.4]),
               tube([(46, 48), (60, 58), (72, 72), (86, 84)], [7, 5, 4, 3]), circle(88, 86, 5), bar(72, 72, 88, 62, 3), circle(90, 61, 3.6)],
         detail=[circle(38, 42, 5)],
         anchors={"soma": (38, 42)}),
    dict(name="seedling", category="nature", aliases=["sprout", "seed sprouting", "young plant", "germination", "幼苗", "發芽", "種子發芽"],
         keywords=["plant", "growth", "garden", "spring", "biology", "roots"], desc="seedling with two leaves and roots in soil",
         body=[rect(8, 62, 84, 30, 6), tube([(50, 64), (50, 48), (52, 34)], 4.4)],
         detail=[tube([(50, 66), (50, 84)], 2.6), tube([(50, 72), (38, 82)], 2.2), tube([(50, 74), (62, 84)], 2.2),
                 tube([(50, 80), (40, 90)], 2)],
         accent=[smooth([(52, 36), (62, 18), (86, 14), (84, 36), (64, 44)]), smooth([(50, 48), (34, 32), (12, 34), (18, 52), (36, 56)])],
         accent_color=GREEN, anchors={"leaf": (70, 28)}),
    dict(name="petri dish", category="science", aliases=["culture dish", "petri plate", "培養皿"],
         keywords=["biology", "lab", "bacteria", "culture", "microbiology"], desc="petri dish with colonies",
         body=[circle(50, 50, 42)],
         detail=[ring(50, 50, 37, 2.6), circle(36, 40, 7), circle(60, 34, 4.4), circle(62, 58, 9), circle(38, 64, 4.4),
                 circle(48, 52, 2.6), circle(72, 44, 2.6)],
         anchors={"center": (50, 50)}),
]

STORY = [
    dict(name="wizard tower", category="buildings", aliases=["mage tower", "wizard's tower", "magic tower", "sorcerer tower", "巫師塔", "魔法塔"],
         keywords=["fantasy", "fairy tale", "magic", "storybook", "castle"], desc="tapering wizard tower with a pointed hat roof",
         body=[poly([(32, 36), (68, 36), (74, 92), (26, 92)], 2), poly([(20, 40), (52, 4), (82, 40)], 3),
               rect(22, 48, 56, 6, 2)],
         detail=[rect(43, 62, 14, 16, (7, 7, 0, 0)), rect(41, 78, 18, 14, (9, 9, 0, 0)), circle(52, 26, 3.2),
                 rect(34, 40, 8, 5, 1)],
         anchors={"window": (50, 70)}),
    dict(name="fishing boat", category="transport", aliases=["trawler", "fishing vessel", "fisherman's boat", "漁船"],
         keywords=["sea", "fisherman", "harbour", "village", "nets"], desc="small fishing boat with a cabin and a mast",
         body=[poly([(6, 60), (30, 62), (70, 62), (94, 50), (80, 86), (20, 86)], 4), rect(14, 36, 26, 26, 2),
               bar(62, 10, 62, 62, 3.6), bar(62, 24, 88, 52, 3)],
         detail=[rect(19, 41, 16, 9, 2), rect(10, 68, 76, 3, 1.5)],
         accent=[poly([(62, 10), (80, 15), (62, 21)])], accent_color=RED, anchors={"mast": (62, 10)}),
    dict(name="magic wand", category="holidays", aliases=["wand", "wizard wand", "fairy wand", "魔杖", "仙女棒"],
         keywords=["magic", "fantasy", "spell", "wizard", "sparkle"], desc="magic wand with a star tip and sparkles",
         body=[bar(16, 86, 62, 40, 7), bar(14, 88, 24, 78, 9)],
         accent=[_star(72, 28, 20, 9, 5, -90), circle(24, 22, 3.4), circle(86, 66, 3), circle(44, 12, 2.4)],
         accent_color=YELLOW, anchors={"star": (72, 28)}),
    dict(name="spell book", category="arts", aliases=["spellbook", "magic book", "grimoire", "魔法書", "咒語書"],
         keywords=["fantasy", "wizard", "magic", "storybook", "library"], desc="closed book with a star on the cover",
         body=[rect(16, 10, 66, 80, (4, 6, 6, 4)), rect(82, 14, 4, 72, 1.5)],
         detail=[rect(26, 12, 3, 76, 1.5), _star(55, 40, 13, 6, 5, -90, 0.6), rect(40, 62, 30, 3.4, 1.7),
                 rect(46, 70, 18, 3.4, 1.7)],
         accent=[rect(74, 42, 12, 14, 2)], accent_color=YELLOW, anchors={"cover": (55, 40)}),
    dict(name="crystal ball", category="holidays", aliases=["fortune teller ball", "scrying orb", "magic orb", "水晶球"],
         keywords=["fantasy", "magic", "fortune", "mystic", "future"], desc="crystal ball on a stand with a sparkle",
         body=[circle(50, 42, 33), poly([(24, 92, 3), (76, 92, 3), (66, 76, 2), (34, 76, 2)], 2), rect(36, 70, 28, 8, 3)],
         detail=[arc(50, 42, 24, 195, 255, 4), poly([(54, 28, 0), (58, 38, 0), (68, 42, 0), (58, 46, 0), (54, 58, 0),
                                                      (50, 46, 0), (40, 42, 0), (50, 38, 0)], 1.2)],
         anchors={"orb": (50, 42)}),
    dict(name="potion bottle", category="holidays", aliases=["potion", "elixir", "magic potion", "poison bottle", "魔藥", "藥水瓶"],
         keywords=["fantasy", "witch", "alchemy", "magic", "halloween"], desc="round flask potion with a cork",
         body=[circle(50, 68, 26), rect(42, 24, 16, 32, 2), rect(37, 18, 26, 8, 3)],
         detail=[arc(50, 68, 18, 190, 245, 3.6), circle(48, 74, 3.6), circle(60, 64, 2.6), circle(56, 80, 2.4)],
         accent=[rect(43, 4, 14, 14, 3)], accent_color="#a8774a", anchors={"cork": (50, 10)}),
    dict(name="cauldron", category="holidays", aliases=["witch cauldron", "witch's pot", "bubbling cauldron", "魔女的鍋", "大鍋"],
         keywords=["witch", "potion", "halloween", "magic", "fantasy"], desc="cauldron with bubbles and three legs",
         body=["M12 54 Q10 90 50 90 Q90 90 88 54 Z", ellipse(50, 54, 40, 9), bar(28, 84, 24, 94, 6), bar(72, 84, 76, 94, 6)],
         detail=[ellipse(50, 54, 33, 4.6)],
         accent=[circle(38, 40, 7), circle(56, 32, 5.4), circle(68, 42, 4.2), circle(48, 22, 3.2)], accent_color="#7bd45a",
         anchors={"bubbles": (50, 34)}),
    dict(name="knight helmet", category="holidays", aliases=["armor helmet", "medieval helmet", "knight", "騎士頭盔", "頭盔"],
         keywords=["medieval", "armour", "castle", "fantasy", "warrior"], desc="knight helmet with a visor and a plume",
         body=["M18 58 Q18 16 52 16 Q86 16 86 58 L86 80 Q86 90 76 90 L28 90 Q18 90 18 80 Z"],
         detail=[rect(32, 42, 54, 5, 2.5), rect(52, 50, 3.4, 34, 1.7), circle(64, 62, 2.2), circle(64, 70, 2.2), circle(64, 78, 2.2),
                 circle(42, 62, 2.2), circle(42, 70, 2.2), circle(42, 78, 2.2)],
         accent=[tube([(54, 16), (48, 8), (62, 5), (76, 12)], [8, 8, 7, 5])], accent_color=RED, anchors={"plume": (62, 8)}),
]


def _qr():
    """Finder squares and a fixed module pattern for a QR-like code."""
    out = []
    for fx, fy in ((8, 8), (64, 8), (8, 64)):
        out.append(frame(fx, fy, 28, 28, 5, 3))
        out.append(rect(fx + 9, fy + 9, 10, 10, 1.5))
    n, c = 9, 9.4
    for i in range(n):
        for j in range(n):
            x, y = 8 + i * c * 1.04 + 0, 8 + j * c * 1.04
            if (i < 4 and j < 4) or (i > 5 and j < 4) or (i < 4 and j > 5):
                continue
            if (i * 5 + j * 3 + i * j) % 3 == 0:
                out.append(rect(x, y, 8, 8, 1.2))
    return out


def _barcode():
    ws, gaps = [5, 2, 3, 2, 6, 3, 2, 4, 2, 5, 3, 2, 6, 2, 3, 4], [3, 3, 2, 4, 3, 2, 3, 3, 2, 3, 4, 2, 3, 3, 2]
    out, x = [], 8.0
    for k, w in enumerate(ws):
        if x + w > 92:
            break
        out.append(rect(x, 20, w, 60, 0.6))
        x += w + gaps[k % len(gaps)]
    return out


EVERYDAY = [
    dict(name="price tag", category="money", aliases=["label", "sale tag", "discount tag", "tag", "價格標籤", "標價", "吊牌"],
         keywords=["sale", "shop", "retail", "discount", "offer"], desc="price tag with a string hole",
         body=[poly([(8, 28, 4), (64, 28, 4), (92, 50, 5), (64, 72, 4), (8, 72, 4)])],
         detail=[circle(22, 50, 5.4), rect(36, 44, 24, 5, 2.5), rect(36, 55, 16, 4, 2)],
         anchors={"hole": (22, 50)}),
    dict(name="ticket", category="money", aliases=["admission ticket", "event ticket", "cinema ticket", "coupon", "票券", "入場券", "門票"],
         keywords=["event", "concert", "cinema", "entry", "admit"], desc="ticket with notches and a perforation",
         body=[rect(6, 24, 88, 52, 6), cut(circle(6, 50, 8)), cut(circle(94, 50, 8))],
         detail=[rect(68, 30, 2.8, 6, 1), rect(68, 41, 2.8, 6, 1), rect(68, 52, 2.8, 6, 1), rect(68, 63, 2.8, 6, 1),
                 rect(18, 36, 38, 4.6, 2.3), rect(18, 48, 28, 4.6, 2.3), rect(18, 60, 34, 4.6, 2.3)],
         anchors={"stub": (82, 50)}),
    dict(name="qr code", category="tech", aliases=["qr", "scan code", "2d barcode", "二維碼", "QR碼", "條碼掃描"],
         keywords=["scan", "payment", "link", "phone", "square code"], desc="QR code with three finder squares",
         body=_qr(), anchors={"center": (50, 50)}),
    dict(name="barcode", category="tech", aliases=["bar code", "product code", "upc", "條碼", "商品條碼"],
         keywords=["scan", "retail", "product", "price", "label"], desc="barcode stripes",
         body=_barcode(), anchors={"center": (50, 50)}),
    dict(name="first-aid kit", category="medical", aliases=["first aid box", "medical kit", "emergency kit", "急救箱", "醫藥箱"],
         keywords=["health", "emergency", "safety", "bandage", "rescue"], desc="first-aid case with a cross and a handle",
         body=[rect(8, 30, 84, 58, 8), frame(34, 14, 32, 22, 5, 6)],
         detail=[rect(43, 42, 14, 34, 2), rect(33, 52, 34, 14, 2)],
         anchors={"cross": (50, 59)}),
    dict(name="water bottle", category="objects", aliases=["bottle of water", "sports bottle", "reusable bottle", "drink bottle", "水瓶", "保特瓶"],
         keywords=["drink", "hydration", "gym", "travel", "water"], desc="water bottle with a cap and a label",
         body=[rect(38, 6, 24, 11, 3), rect(42, 16, 16, 8, 1), rect(28, 22, 44, 72, (12, 12, 8, 8))],
         detail=[rect(33, 50, 34, 20, 3), rect(34, 32, 32, 2.6, 1.3)],
         anchors={"cap": (50, 10)}),
    dict(name="toothbrush", category="objects", aliases=["tooth brush", "brush teeth", "牙刷"],
         keywords=["dental", "hygiene", "bathroom", "teeth", "morning"], desc="toothbrush with bristle rows, tilted",
         body=[_rp([(40, 6), (60, 6), (60, 40), (40, 40)], 35, 4), _rp([(43, 36), (57, 36), (59, 94), (41, 94)], 35, 7)],
         detail=[_rb(44, 12, 56, 12, 3, 35), _rb(44, 20, 56, 20, 3, 35), _rb(44, 28, 56, 28, 3, 35)],
         anchors={"head": (50, 20)}),
    dict(name="face mask", category="medical", aliases=["surgical mask", "medical mask", "protective mask", "口罩"],
         keywords=["health", "hygiene", "safety", "pandemic", "doctor"], desc="surgical face mask with ear loops",
         body=[rect(16, 28, 68, 46, 13), ellipse(11, 50, 8, 16) + " " + reverse(ellipse(11, 50, 4.4, 12)),
               ellipse(89, 50, 8, 16) + " " + reverse(ellipse(89, 50, 4.4, 12))],
         detail=[rect(22, 38, 56, 2.8, 1.4), rect(22, 49, 56, 2.8, 1.4), rect(22, 60, 56, 2.8, 1.4)],
         anchors={"center": (50, 51)}),
    dict(name="soap", category="objects", aliases=["soap bar", "bar of soap", "bath soap", "肥皂", "香皂"],
         keywords=["hygiene", "bathroom", "wash", "bubbles", "clean"], desc="soap bar with bubbles",
         body=[rect(10, 40, 80, 40, 13), ring(32, 24, 9, 2.8), ring(52, 14, 6, 2.4), ring(68, 24, 8, 2.6)],
         detail=[rect(22, 49, 40, 3.4, 1.7), rect(30, 62, 40, 4.4, 2.2)],
         anchors={"bar": (50, 60)}),
    dict(name="notebook", category="work", aliases=["journal", "diary", "exercise book", "composition book", "筆記本", "日記本"],
         keywords=["school", "write", "stationery", "study", "notes"], desc="closed notebook with a label and an elastic band",
         body=[rect(18, 8, 64, 84, 6), circle(18, 22, 4), circle(18, 38, 4), circle(18, 54, 4), circle(18, 70, 4)],
         detail=[rect(34, 22, 38, 24, 3), rect(40, 28, 26, 3.4, 1.7), rect(40, 36, 18, 3.4, 1.7)],
         accent=[rect(68, 8, 5, 84, 0)], accent_color=RED, anchors={"label": (53, 34)}),
    dict(name="electric fan", category="home", aliases=["fan", "desk fan", "pedestal fan", "cooling fan", "電風扇", "風扇"],
         keywords=["summer", "cool", "air", "appliance", "wind"], desc="desk fan with three blades behind a guard",
         body=[ring(50, 40, 36, 3.6), circle(50, 40, 6.5), bar(50, 76, 50, 86, 6), rect(28, 86, 44, 7, 3.5)]
         + [ellipse(50 + 15 * math.cos(math.radians(a)), 40 + 15 * math.sin(math.radians(a)), 14, 6, a) for a in (-90 - 20, 30 - 20, 150 - 20)],
         detail=[circle(50, 40, 2.6)],
         anchors={"hub": (50, 40)}),
    dict(name="hand sanitizer", category="medical", aliases=["pump bottle", "soap dispenser", "lotion bottle", "sanitiser", "乾洗手", "洗手乳", "按壓瓶"],
         keywords=["hygiene", "clean", "germs", "health", "wash"], desc="pump bottle with a label and a drop",
         body=[rect(28, 36, 44, 56, 9), rect(42, 26, 16, 12, 2), rect(36, 16, 30, 9, 3), bar(62, 20, 80, 20, 5)],
         detail=[rect(33, 52, 34, 28, 3), circle(50, 62, 4.6)],
         anchors={"pump": (50, 20)}),
    dict(name="bandage", category="medical", aliases=["adhesive bandage", "band-aid", "plaster", "sticking plaster", "OK繃", "繃帶", "創可貼"],
         keywords=["first aid", "injury", "health", "cut", "care"], desc="adhesive bandage, tilted",
         body=[_rp([(6, 38), (94, 38), (94, 62), (6, 62)], -35, 12)],
         detail=[_rp([(36, 38), (64, 38), (64, 62), (36, 62)], -35), ] + [
             cut(circle(*R([(x, y)], -35, 50, 50)[0], 1.7)) for x, y in ((42, 44), (50, 44), (58, 44), (42, 50), (50, 50), (58, 50), (42, 56), (50, 56), (58, 56))],
         anchors={"pad": (50, 50)}),
]

PICTOGRAMS = TECH + BIOLOGY + STORY + EVERYDAY
