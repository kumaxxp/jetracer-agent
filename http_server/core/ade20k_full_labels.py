"""ADE20K 150クラス完全マッピング

OneFormerのADE20Kモデルが出力するクラスIDと名前の対応表。
セグメンテーションマスクのピクセル値（0-149）がこのIDに対応。
"""

# ADE20K 150クラスのID→名前マッピング
# ID 0-149 が有効なクラス
ADE20K_ID_TO_NAME = {
    0: "wall",
    1: "building",
    2: "sky",
    3: "floor",
    4: "tree",
    5: "ceiling",
    6: "road",
    7: "bed",
    8: "windowpane",
    9: "grass",
    10: "cabinet",
    11: "sidewalk",
    12: "person",
    13: "earth",
    14: "door",
    15: "table",
    16: "mountain",
    17: "plant",
    18: "curtain",
    19: "chair",
    20: "car",
    21: "water",
    22: "painting",
    23: "sofa",
    24: "shelf",
    25: "house",
    26: "sea",
    27: "mirror",
    28: "rug",
    29: "field",
    30: "armchair",
    31: "seat",
    32: "fence",
    33: "desk",
    34: "rock",
    35: "wardrobe",
    36: "lamp",
    37: "bathtub",
    38: "railing",
    39: "cushion",
    40: "base",
    41: "box",
    42: "column",
    43: "signboard",
    44: "chest of drawers",
    45: "counter",
    46: "sand",
    47: "sink",
    48: "skyscraper",
    49: "fireplace",
    50: "refrigerator",
    51: "grandstand",
    52: "path",
    53: "stairs",
    54: "runway",
    55: "case",
    56: "pool table",
    57: "pillow",
    58: "screen door",
    59: "stairway",
    60: "river",
    61: "bridge",
    62: "bookcase",
    63: "blind",
    64: "coffee table",
    65: "toilet",
    66: "flower",
    67: "book",
    68: "hill",
    69: "bench",
    70: "countertop",
    71: "stove",
    72: "palm",
    73: "kitchen island",
    74: "computer",
    75: "swivel chair",
    76: "boat",
    77: "bar",
    78: "arcade machine",
    79: "hovel",
    80: "bus",
    81: "towel",
    82: "light",
    83: "truck",
    84: "tower",
    85: "chandelier",
    86: "awning",
    87: "streetlight",
    88: "booth",
    89: "television",
    90: "airplane",
    91: "dirt track",
    92: "apparel",
    93: "pole",
    94: "land",
    95: "bannister",
    96: "escalator",
    97: "ottoman",
    98: "bottle",
    99: "buffet",
    100: "poster",
    101: "stage",
    102: "van",
    103: "ship",
    104: "fountain",
    105: "conveyer belt",
    106: "canopy",
    107: "washer",
    108: "plaything",
    109: "swimming pool",
    110: "stool",
    111: "barrel",
    112: "basket",
    113: "waterfall",
    114: "tent",
    115: "bag",
    116: "minibike",
    117: "cradle",
    118: "oven",
    119: "ball",
    120: "food",
    121: "step",
    122: "tank",
    123: "trade name",
    124: "microwave",
    125: "pot",
    126: "animal",
    127: "bicycle",
    128: "lake",
    129: "dishwasher",
    130: "screen",
    131: "blanket",
    132: "sculpture",
    133: "hood",
    134: "sconce",
    135: "vase",
    136: "traffic light",
    137: "tray",
    138: "ashcan",
    139: "fan",
    140: "pier",
    141: "crt screen",
    142: "plate",
    143: "monitor",
    144: "bulletin board",
    145: "shower",
    146: "radiator",
    147: "glass",
    148: "clock",
    149: "flag",
}

# 名前→IDの逆引きマッピング
ADE20K_NAME_TO_ID = {name: id for id, name in ADE20K_ID_TO_NAME.items()}

# 特殊値
MYCAR_LABEL_ID = 255  # 自車両マーカー用


def get_label_name(label_id: int) -> str:
    """ラベルIDから名前を取得"""
    if label_id == MYCAR_LABEL_ID:
        return "mycar"
    return ADE20K_ID_TO_NAME.get(label_id, f"unknown_{label_id}")


def get_label_id(label_name: str) -> int:
    """ラベル名からIDを取得（見つからない場合は-1）"""
    if label_name == "mycar":
        return MYCAR_LABEL_ID
    return ADE20K_NAME_TO_ID.get(label_name, -1)


def get_road_label_ids(road_label_names: list) -> set:
    """ROADラベル名のリストからADE20K IDのセットを取得
    
    Args:
        road_label_names: ROADとして設定されたラベル名のリスト
    
    Returns:
        対応するADE20K IDのセット
    """
    road_ids = set()
    for name in road_label_names:
        label_id = get_label_id(name)
        if label_id >= 0:
            road_ids.add(label_id)
        else:
            print(f"[ADE20K] Warning: Unknown label name '{name}'")
    return road_ids


# デフォルトで走行可能と考えられるラベル（参考用）
DEFAULT_DRIVABLE_LABELS = {
    "road",       # ID 6
    "floor",      # ID 3
    "sidewalk",   # ID 11
    "path",       # ID 52
    "dirt track", # ID 91
    "runway",     # ID 54
    "rug",        # ID 28
    "field",      # ID 29
    "sand",       # ID 46
}


def print_all_labels():
    """全ラベルを表示（デバッグ用）"""
    print("ADE20K 150 Classes:")
    for id, name in sorted(ADE20K_ID_TO_NAME.items()):
        marker = " [DRIVABLE]" if name in DEFAULT_DRIVABLE_LABELS else ""
        print(f"  {id:3d}: {name}{marker}")


if __name__ == "__main__":
    print_all_labels()
