"""Shared application operations for the desktop UI and CLI."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile

import numpy as np
import trimesh

from .config import SupportConfig, UnionMode
from .exporter import Exporter
from .infill import InfillConfig, generate_infill
from .mesh_loader import MeshLoader
from .pipeline import TreeSupportPipeline


SLICER_GUIDE = """Simplify3D v5 用 STL / 単位 mm

ツリーサポート:
  model_with_support.stl を読み込むと、モデルとサポートの相対位置を保持できます。
  これは通常の印刷形状です。Simplify3D側の自動サポート生成はOFFにします。
  生成時と同じレイヤー高さで、先端ギャップ・枝・接地をプレビューしてください。
  別プロセスで設定する場合は model.stl と tree_support.stl を読み込み、
  Edit > Align Selected Model Origins で原点を合わせ、グループとして配置します。
  個別の自動整列・ベッドへの落下は避け、浮いたモデルのZ位置も維持してください。

インフィル:
  model_with_infill.stl を単独で読み込みます。外殻と格子を一体化済みです。
  元の model.stl を重ねると空洞を埋めるので重ねないでください。
  infill_only.stl は内部構造の確認・独自ワークフロー用です。
  外殻・格子そのものを充填する設定として infill 100% を出発点にします。
  0%では太い格子や外殻の材料部分まで空洞になり得ます。
  自動サポートはOFF。空洞を埋める修復設定は使わず、内部空洞と薄壁が
  保持されているか各レイヤーで確認してください。細い形状では単線押出設定も確認します。

本アプリは形状を生成します。Simplify3D固有のサポート情報・ツールパス・G-codeは
含みません。v5での実スライスと実機試験は別途必要です。
ジャイロイド/ダイヤモンドは厚み付き周期曲面、cubicは直交する丸棒格子です。
スライサー内蔵パターンの押出経路を完全再現するものではありません。

公式情報:
https://www.simplify3d.com/resources/videos/selecting-grouping-models/
https://www.simplify3d.com/resources/articles/printing-thin-walls-and-small-features/
https://www.simplify3d.com/products/simplify3d-software/whats-new/version-5-0/
"""


@dataclass
class ArtifactBundle:
    kind: str
    meshes: dict[str, trimesh.Trimesh]
    report: dict


def load_model(source) -> trimesh.Trimesh:
    model = source.copy() if isinstance(source, trimesh.Trimesh) else MeshLoader().load(source).mesh
    if len(model.faces) == 0 or not np.isfinite(model.vertices).all() or not model.is_volume:
        raise ValueError("閉じたソリッドモデルが必要です。穴・法線・自己交差を確認してください。")
    return model


def build_bundle(source, kind: str, config=None, progress=None) -> ArtifactBundle:
    model = load_model(source)
    if kind == "support":
        cfg = config or SupportConfig(union_mode=UnionMode.BOOLEAN)
        cfg.validate()
        if model.bounds[0, 2] < -1e-6:
            raise ValueError("モデルがZ=0より下にあります。CAD側で印刷姿勢を設定してください。")
        if cfg.union_mode is not UnionMode.BOOLEAN:
            raise ValueError("アプリのSTL出力には boolean union が必要です。")
        result = TreeSupportPipeline(cfg).run(model, progress=progress)
        if not result.validation["passed"]:
            raise ValueError("サポート検証に失敗しました:\n" + "\n".join(result.validation["failures"]))
        support = result.support_mesh
        meshes = {"model.stl": model}
        if len(support.faces):
            meshes["tree_support.stl"] = support
            meshes["model_with_support.stl"] = trimesh.util.concatenate([model, support])
        report = result.summary()
        report["config"] = cfg.to_dict()
        report["has_support"] = bool(len(support.faces))
        report["warnings"] = list(result.validation["warnings"])
        if not len(support.faces):
            report["warnings"].append("出力できるサポートはありません。対象面・到達不能面を確認してください。")
        return ArtifactBundle(kind, meshes, report)
    if kind != "infill":
        raise ValueError("kind must be support or infill")
    result = generate_infill(model, config or InfillConfig(), progress)
    return ArtifactBundle(kind, {"model.stl": model, "infill_only.stl": result.infill,
                                "model_with_infill.stl": result.printable}, result.report)


def export_bundle(bundle: ArtifactBundle, directory: str | Path) -> Path:
    """Create a new folder for each export; never replace input or previous output."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = Path(tempfile.mkdtemp(prefix=bundle.kind + "-", dir=directory))
    exporter = Exporter()
    for name, mesh in bundle.meshes.items():
        exporter.export(mesh, target / name)
    (target / "report.json").write_text(json.dumps(bundle.report, indent=2, ensure_ascii=False,
                                                  allow_nan=False), encoding="utf-8")
    (target / "Simplify3D-v5.txt").write_text(SLICER_GUIDE, encoding="utf-8-sig")
    return target
