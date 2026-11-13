# -*- coding: utf-8 -*-
from __future__ import annotations



# ===================================================================================================== palettes

SKIN_TONES = ["#f9dcc8", "#f3c9aa", "#eab48e", "#d99e76", "#c18561", "#a36b4b", "#865239", "#683d2b"]
HAIR_COLORS = {"black": "#2b211e", "dark_brown": "#4a2f22", "brown": "#7a4a2c", "chestnut": "#94532e",
               "auburn": "#a6452a", "ginger": "#d0702e", "blonde": "#e6c06e", "platinum": "#efe2bc",
               "grey": "#b9b5ae", "white": "#eeece6", "blue": "#4a7ad8", "pink": "#ee86b4", "purple": "#8a5ccc",
               "green": "#4fae6e"}
TOP_COLORS = ["#e8613a", "#f2b33d", "#3f86d8", "#46b37a", "#7a5cc8", "#e86a8e", "#2f6f8f", "#f4efe6",
              "#c94f4f", "#5ab0b8", "#f08a5d", "#9cc95a", "#ffd166", "#264653"]
BOTTOM_COLORS = ["#384a6a", "#2f3a4a", "#6b5a4a", "#4a6a8a", "#c8b89a", "#3c3c42", "#5a6b3a", "#7a3b3b"]
SHOE_COLORS = ["#f4f1ea", "#2b2b30", "#c8462e", "#3f6fb0", "#8a5a3a", "#f2c14e"]
EYE_COLORS = {"brown": "#5a3620", "dark": "#2a1a12", "hazel": "#7a5a2a", "green": "#3f7a4a", "blue": "#3f74c0",
              "grey": "#6a7a88"}


# ===================================================================================================== builds

BUILDS = {
    # heads: total height in head heights; hipj: hip-joint height; neck: neck length; sw / hw: shoulder / hip joint
    # half widths; cw / ww: chest / waist half widths; depth: torso half depth; armr / legr: limb radii; arm:
    # shoulder-to-wrist length; hand / foot: lengths (all in head heights). headw: head width; eye: eye size;
    # belly: belly bulge; stoop: spine bend; height: default height in metres.
    "toy": dict(heads=3.0, hipj=0.84, neck=0.05, sw=0.38, hw=0.15, cw=0.35, ww=0.35, depth=0.27, armr=0.12,
                legr=0.155, arm=0.84, hand=0.46, foot=0.52, headw=1.04, eye=1.0, belly=1.05, stoop=0, height=1.1,
                head="round", eyes="bead"),
    "chibi": dict(heads=2.5, hipj=0.64, neck=0.03, sw=0.35, hw=0.14, cw=0.33, ww=0.35, depth=0.25, armr=0.115,
                  legr=0.145, arm=0.68, hand=0.42, foot=0.44, headw=1.1, eye=1.1, belly=1.1, stoop=0, height=0.9,
                  head="wide", eyes="bead"),
    "baby": dict(heads=2.7, hipj=0.68, neck=0.02, sw=0.37, hw=0.16, cw=0.35, ww=0.4, depth=0.29, armr=0.125,
                 legr=0.165, arm=0.7, hand=0.4, foot=0.4, headw=1.08, eye=1.1, belly=1.3, stoop=0, height=0.75,
                 head="round", eyes="bead"),
    "kid": dict(heads=4.0, hipj=1.48, neck=0.1, sw=0.44, hw=0.16, cw=0.4, ww=0.37, depth=0.28, armr=0.13,
                legr=0.165, arm=1.2, hand=0.5, foot=0.6, headw=1.0, eye=0.95, belly=1.0, stoop=0, height=1.2,
                head="round", eyes="round"),
    "teen": dict(heads=5.0, hipj=2.12, neck=0.16, sw=0.52, hw=0.18, cw=0.47, ww=0.4, depth=0.29, armr=0.15,
                 legr=0.19, arm=1.66, hand=0.56, foot=0.68, headw=0.97, eye=0.88, belly=0.95, stoop=0, height=1.6,
                 head="oval", eyes="round"),
    "adult": dict(heads=5.5, hipj=2.38, neck=0.2, sw=0.62, hw=0.2, cw=0.58, ww=0.48, depth=0.33, armr=0.18,
                  legr=0.23, arm=1.86, hand=0.6, foot=0.76, headw=0.96, eye=0.82, belly=1.0, stoop=0, height=1.7,
                  head="oval", eyes="round"),
    "elder": dict(heads=5.2, hipj=2.2, neck=0.15, sw=0.58, hw=0.2, cw=0.54, ww=0.54, depth=0.35, armr=0.17,
                  legr=0.21, arm=1.78, hand=0.58, foot=0.72, headw=0.98, eye=0.78, belly=1.3, stoop=9, height=1.6,
                  head="round", eyes="round"),
    "hero": dict(heads=6.5, hipj=2.95, neck=0.25, sw=0.86, hw=0.21, cw=0.8, ww=0.52, depth=0.4, armr=0.24,
                 legr=0.27, arm=2.26, hand=0.66, foot=0.82, headw=0.92, eye=0.74, belly=0.85, stoop=0, height=1.85,
                 head="square", eyes="round"),
}


# ===================================================================================================== expressions

EXPRESSIONS = {
    #             eyes ('open' | 'wide' | 'happy' | 'closed' | 'half'), upper-lid (height, inner tilt),
    #             brows (raise, inner tilt, arch), mouth, blush strength, gaze offset
    "neutral": dict(eyes="open", lid=None, brows=(0.0, 0.0, 0.25), mouth="small_smile", blush=0.6),
    "smile": dict(eyes="open", lid=None, low=0.55, brows=(0.04, -0.02, 0.35), mouth="smile", blush=0.8),
    "happy": dict(eyes="happy", lid=None, brows=(0.06, -0.02, 0.4), mouth="open_smile", blush=1.0),
    "laugh": dict(eyes="happy", lid=None, brows=(0.1, -0.04, 0.45), mouth="laugh", blush=1.0),
    "surprised": dict(eyes="wide", lid=None, brows=(0.16, 0.0, 0.6), mouth="o", blush=0.5),
    "sad": dict(eyes="open", lid=(0.55, -16), brows=(0.04, 0.16, 0.0), mouth="frown", blush=0.4, gaze=(0, -0.3)),
    "angry": dict(eyes="open", lid=(0.45, 22), brows=(-0.05, -0.15, -0.1), mouth="grit", blush=0.3),
    "wink": dict(eyes="wink", lid=None, brows=(0.05, -0.02, 0.4), mouth="smirk_open", blush=0.9),
    "sleepy": dict(eyes="open", lid=(0.05, 0), brows=(-0.02, 0.02, 0.2), mouth="small_o", blush=0.6, gaze=(0, -0.2)),
    "sleep": dict(eyes="closed", lid=None, brows=(-0.02, 0.0, 0.2), mouth="small_o", blush=0.7),
    "determined": dict(eyes="open", lid=(0.62, 14), brows=(-0.03, -0.1, 0.0), mouth="flat", blush=0.4),
    "shy": dict(eyes="open", lid=(0.75, -6), brows=(0.03, 0.08, 0.2), mouth="small_smile", blush=1.6,
                gaze=(-0.5, -0.45)),
    "cheeky": dict(eyes="happy", lid=None, brows=(0.06, -0.02, 0.4), mouth="tongue", blush=1.0),
    "worried": dict(eyes="open", lid=None, brows=(0.08, 0.18, 0.1), mouth="wavy", blush=0.4),
    "calm": dict(eyes="half", lid=(0.4, 0), brows=(0.0, 0.0, 0.25), mouth="small_smile", blush=0.6),
}


# ===================================================================================================== outfits

TOPS = {
    # sleeve: arm coverage from the shoulder (0~1); hem: where the top ends (fraction of the torso above the hip
    # joints, negative = below); neck: covers the neck; extra details
    "tshirt": dict(sleeve=0.4, hem=-0.12, cuff=1.15),
    "tank": dict(sleeve=0.0, hem=-0.1, cuff=1.0),
    "longsleeve": dict(sleeve=0.94, hem=-0.12, cuff=1.12),
    "shirt": dict(sleeve=0.94, hem=-0.14, cuff=1.15, collar=True, buttons=True),
    "polo": dict(sleeve=0.38, hem=-0.12, cuff=1.15, collar=True),
    "sweater": dict(sleeve=0.95, hem=-0.16, cuff=1.2, band=True, thick=1.5),
    "hoodie": dict(sleeve=0.95, hem=-0.18, cuff=1.2, band=True, hood=True, pocket=True, thick=1.6),
    "jacket": dict(sleeve=0.95, hem=-0.2, cuff=1.15, collar=True, open=True, thick=1.7),
    "coat": dict(sleeve=0.97, hem=-0.2, cuff=1.2, collar=True, buttons=True, tails=0.55, thick=1.8),
    "lab_coat": dict(sleeve=0.95, hem=-0.2, cuff=1.12, collar=True, open=True, tails=0.62, thick=1.4),
    "suit": dict(sleeve=0.95, hem=-0.22, cuff=1.1, lapels=True, tie=True, thick=1.5),
    "dress": dict(sleeve=0.22, hem=-0.1, cuff=1.15, skirt=0.62, flare=2.4),
    "gown": dict(sleeve=0.18, hem=-0.1, cuff=1.25, skirt=1.0, flare=3.4),
    "robe": dict(sleeve=0.97, hem=-0.1, cuff=1.8, skirt=1.0, flare=2.2, thick=1.6),
    "jersey": dict(sleeve=0.42, hem=-0.14, cuff=1.25, thick=1.2),
    "spacesuit": dict(sleeve=0.97, hem=-0.3, cuff=1.25, thick=3.2, puffy=True),
    "armor": dict(sleeve=0.6, hem=-0.25, cuff=1.25, thick=2.4, pauldrons=True),
    "onesie": dict(sleeve=0.9, hem=-0.3, cuff=1.1, thick=1.6),
    "raincoat": dict(sleeve=0.97, hem=-0.22, cuff=1.2, hood=True, buttons=True, tails=0.5, thick=1.8),
    "none": dict(sleeve=0.0, hem=1.0, cuff=1.0),
}

BOTTOMS = {
    "trousers": dict(leg=0.95, cuff=1.1),
    "jeans": dict(leg=0.95, cuff=1.12),
    "shorts": dict(leg=0.34, cuff=1.25),
    "skirt": dict(leg=0.0, skirt=0.42, flare=2.3),
    "long_skirt": dict(leg=0.0, skirt=0.86, flare=2.5),
    "leggings": dict(leg=0.96, cuff=1.0, thin=True),
    "overalls": dict(leg=0.95, cuff=1.12, bib=True),
    "puffy": dict(leg=0.97, cuff=1.2, thick=3.0),
    "none": dict(leg=0.0),
}

SHOES = ["sneakers", "boots", "shoes", "sandals", "barefoot", "rain_boots", "slippers", "heels"]

EXTRAS = ["backpack", "bag", "scarf", "cape", "glasses", "sunglasses", "headphones", "tie", "bowtie", "necklace", "medal",
          "stethoscope", "belt", "apron", "watch", "wings"]

# everyday words for outfit pieces -> the names above
_OUTFIT_ALIASES = {
    "top": {"t-shirt": "tshirt", "t_shirt": "tshirt", "tee": "tshirt", "jumper": "sweater", "sweatshirt": "sweater",
            "cardigan": "sweater", "pullover": "sweater", "blouse": "shirt", "vest": "tank", "overcoat": "coat",
            "trench_coat": "coat", "parka": "coat", "rain_coat": "raincoat", "mac": "raincoat", "anorak": "raincoat",
            "blazer": "suit", "uniform": "shirt", "tunic": "robe", "cloak": "robe", "pyjamas": "longsleeve",
            "pajamas": "longsleeve", "space_suit": "spacesuit", "armour": "armor"},
    "bottom": {"pants": "trousers", "slacks": "trousers", "chinos": "trousers", "denim": "jeans",
               "tights": "leggings", "dungarees": "overalls", "maxi_skirt": "long_skirt", "snow_pants": "puffy"},
    "shoes": {"wellies": "rain_boots", "wellingtons": "rain_boots", "rainboots": "rain_boots", "gumboots": "rain_boots",
              "trainers": "sneakers", "running_shoes": "sneakers", "socks": "slippers", "flip_flops": "sandals",
              "high_heels": "heels", "loafers": "shoes", "none": "barefoot"},
}

HAIR_STYLES = ["short", "crop", "buzz", "side_part", "quiff", "spiky", "bob", "long", "ponytail", "pigtails", "bun",
               "top_knot", "curly", "afro", "braids", "mohawk", "bald", "balding", "hijab", "pixie", "wavy",
               "baby_tuft"]

HATS = ["cap", "beanie", "sun_hat", "chef_hat", "hard_hat", "crown", "tiara", "witch_hat", "wizard_hat", "helmet",
        "space_helmet", "knight_helmet", "party_hat", "top_hat", "beret", "santa_hat", "pirate_hat", "police_cap",
        "fire_helmet", "cowboy_hat", "bow", "headband", "flower_crown", "antenna", "nurse_cap", "straw_hat"]

OUTFITS = {
    "casual": dict(top="tshirt", bottom="jeans", shoes="sneakers"),
    "school": dict(top="polo", bottom="shorts", shoes="sneakers", extras=["backpack"]),
    "office": dict(top="shirt", bottom="trousers", shoes="shoes", colours=dict(top="#eef2f6", bottom="#3a4250",
                                                                               shoes="#2b2420")),
    "suit": dict(top="suit", bottom="trousers", shoes="shoes", colours=dict(top="#2f3a4f", bottom="#2f3a4f",
                                                                           shoes="#1e1a18", accent="#c83a3a")),
    "sporty": dict(top="jersey", bottom="shorts", shoes="sneakers"),
    "winter": dict(top="coat", bottom="trousers", shoes="boots", hat="beanie", extras=["scarf"]),
    "summer": dict(top="tank", bottom="shorts", shoes="sandals", hat="sun_hat"),
    "dress": dict(top="dress", bottom="none", shoes="shoes"),
    "pyjamas": dict(top="longsleeve", bottom="trousers", shoes="slippers"),
    "hiking": dict(top="jacket", bottom="shorts", shoes="boots", extras=["backpack"]),
    "cosy": dict(top="sweater", bottom="trousers", shoes="sneakers"),
    "hoodie": dict(top="hoodie", bottom="jeans", shoes="sneakers"),
    "overalls": dict(top="tshirt", bottom="overalls", shoes="boots"),
}


# ---------------------------------------------------------------- option names accepted by the builders
MATERIALS = ("clay", "plastic", "vinyl", "plush", "wood")
ANIMAL_MATERIALS = ("clay", "plastic", "vinyl", "plush")
HAIR_KEYS = ("style", "color", "tie")
OUTFIT_KEYS = ("preset", "top", "bottom", "shoes", "hat", "extras", "colours", "colors")
FACE_KEYS = ("expression", "eyes", "eye_color", "brows", "cheeks", "glasses", "glasses_color", "beard", "beard_color",
             "freckles", "nose", "ears", "head", "mouth", "gaze", "ex", "ey", "erx", "ery", "my")
FACE_CHOICES = {"eyes": ("round", "bead", "dot"), "nose": ("button", "small", "round", "long", "pointy", "none"),
                "ears": ("round", "big", "small", "elf"), "beard": ("full", "santa", "short", "moustache", "goatee"),
                "glasses": ("round", "sunglasses")}
