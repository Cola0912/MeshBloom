"""Validated, portable GUI settings; no model data or machine paths."""
import json
from pathlib import Path

from .config import SupportConfig
from .infill import InfillConfig

SUPPORT_FIELDS = {
    "layer_height": ("レイヤー高さ", "0.20"),
    "top_z_gap": ("先端 Z ギャップ", "0.20"),
    "xy_gap": ("側面ギャップ", "0.40"),
    "tip_spacing": ("先端の間隔", "2.50"),
    "tip_diameter": ("先端の直径", "0.80"),
    "overhang_angle": ("対象角度 / 鉛直から °", "45"),
    "branch_angle_max": ("枝の最大傾斜 / °", "40"),
}
INFILL_FIELDS = {
    "cell_size": ("セルの大きさ", "8.0"),
    "wall_thickness": ("格子の厚み・径", "1.2"),
    "shell_thickness": ("外殻の厚み", "1.2"),
    "pitch": ("形状の解像度", "0.4"),
}
PRESETS = {
    "標準": {},
    "素早く確認": {"layer_height": .4, "top_z_gap": .4, "tip_spacing": 3.5,
                  "cell_size": 10, "wall_thickness": 1.5, "shell_thickness": 1.5, "pitch": .5},
    "細かく生成": {"layer_height": .15, "top_z_gap": .3, "tip_spacing": 2,
                  "wall_thickness": 1, "shell_thickness": 1, "pitch": .25},
}


def configuration(values, mode, pattern="gyroid"):
    fields = SUPPORT_FIELDS if mode == "support" else INFILL_FIELDS
    data = {}
    for name, (label, _) in fields.items():
        try:
            data[name] = float(values[name])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"「{label}」に数値を入力してください。") from None
    if mode == "support":
        return SupportConfig(**data, union_mode="boolean")
    if mode == "infill":
        return InfillConfig(**data, pattern=pattern)
    raise ValueError("Unknown generation mode")


def validate_settings(data):
    if not isinstance(data, dict) or data.get("format") != "meshbloom-settings-v1":
        raise ValueError("MeshBloomの設定ファイルではありません。")
    if data.get("mode") not in ("support", "infill"):
        raise ValueError("生成モードが不正です。")
    values = data.get("values")
    if not isinstance(values, dict) or set(values) != set(SUPPORT_FIELDS) | set(INFILL_FIELDS):
        raise ValueError("設定項目が不足しているか、未対応の項目があります。")
    configuration(values, "support")
    configuration(values, "infill", data.get("pattern"))
    return data


def save_settings(path, values, mode, pattern):
    data = validate_settings({"format": "meshbloom-settings-v1", "mode": mode,
                              "pattern": pattern, "values": values})
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_settings(path):
    return validate_settings(json.loads(Path(path).read_text(encoding="utf-8")))
