"""STEP 2: 固定レイヤ高でモデルを 2D ポリゴン列へ変換する。

レイヤ規約 (docs/ARCHITECTURE.md 4 節)
-------------------------------------
* モデルレイヤ ``k`` はスラブ ``[k*lh, (k+1)*lh]`` を占める。
* ``slices[k]`` はそのスラブが占める XY 領域であり、
  **スラブ内の複数平面 (既定 3 枚: 下端 + 中央 + 上端) の和** で近似する。
* ``slices[k]`` は必ず存在し、モデルが無いレイヤは空ジオメトリ。
* 2D 座標はワールド XY と厳密に一致する (座標変換なし)。

なぜ 1 平面ではなく和を取るか
-----------------------------
スラブ中央 1 枚だけでサンプリングすると、モデルの実際の下面が
サンプル平面より下にある場合を取りこぼす。例えば ``lh=0.3`` でモデル下面が
``z=8.29`` のとき、スラブ ``[8.1, 8.4]`` の中央 ``8.25`` には材料が無く、
そのスラブは「空」と判定される。結果として tip が 1 レイヤ高く置かれ、
**実座標の Z ギャップが要求値を下回る**。

下端 ``k*lh + eps`` を含む複数平面の和を取ると、

    「レイヤ k-1 が空」  =>  「実際の材料下面 >= (k)*lh - eps」

が成り立つので、tip を ``k - n_top`` に置いたときの実ギャップが
``n_top * lh - eps >= top_z_gap`` であることを構成的に保証できる。
同時に collision 側も「スラブが掃く領域」に近づくので安全側になる。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import trimesh
from shapely.geometry.base import BaseGeometry

from . import geometry2d as g2
from .config import SupportConfig

__all__ = ["LayerStack", "ModelSlicer"]


@dataclass
class LayerStack:
    """レイヤ断面の列。インデックスはビルドプレート基準の絶対レイヤ番号。"""

    layer_height: float
    slices: list[BaseGeometry]
    model_z_min: float
    model_z_max: float
    xy_bounds: tuple[float, float, float, float] | None = None
    meta: dict = field(default_factory=dict)

    # --- z 座標 ---
    def z_bottom(self, k: int) -> float:
        return k * self.layer_height

    def z_mid(self, k: int) -> float:
        return (k + 0.5) * self.layer_height

    def z_top(self, k: int) -> float:
        return (k + 1) * self.layer_height

    # --- アクセス ---
    @property
    def n_layers(self) -> int:
        return len(self.slices)

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, k: int) -> BaseGeometry:
        """範囲外は空ジオメトリ (境界処理を単純化するため)。"""
        if k < 0 or k >= len(self.slices):
            return g2.EMPTY
        return self.slices[k]

    def __iter__(self):
        return iter(self.slices)

    @property
    def first_model_layer(self) -> int:
        for k, s in enumerate(self.slices):
            if not s.is_empty:
                return k
        return -1

    @property
    def last_model_layer(self) -> int:
        for k in range(len(self.slices) - 1, -1, -1):
            if not self.slices[k].is_empty:
                return k
        return -1

    def total_area(self) -> float:
        return float(sum(s.area for s in self.slices))

    def stats(self) -> dict:
        return {
            "layer_height": self.layer_height,
            "n_layers": self.n_layers,
            "first_model_layer": self.first_model_layer,
            "last_model_layer": self.last_model_layer,
            "model_z_min": self.model_z_min,
            "model_z_max": self.model_z_max,
            "xy_bounds": list(self.xy_bounds) if self.xy_bounds else None,
            "total_slice_area": self.total_area(),
        }


class ModelSlicer:
    """メッシュ -> ``LayerStack``。"""

    def __init__(self, config: SupportConfig) -> None:
        self.config = config

    def slice(self, mesh: trimesh.Trimesh) -> LayerStack:
        cfg = self.config
        lh = cfg.layer_height
        bounds = np.asarray(mesh.bounds, dtype=float)
        z_min, z_max = float(bounds[0][2]), float(bounds[1][2])

        if z_min < -1e-6:
            raise ValueError(
                f"model extends below the build plate (z_min={z_min:.4f}). "
                "Move the model to z >= 0 before generating supports."
            )

        # レイヤ 0 は必ず含める (プレート接地判定のため)。
        last_layer = max(0, math.ceil(z_max / lh - 1e-9) - 1)
        if cfg.max_layers is not None:
            last_layer = min(last_layer, cfg.max_layers - 1)
        n_layers = last_layer + 1

        samples = max(1, cfg.slice_samples_per_layer)
        eps = lh * 1.0e-3
        offsets = self._sample_offsets(samples, lh, eps)

        heights: list[float] = []
        for k in range(n_layers):
            for off in offsets:
                heights.append(k * lh + off)

        sections = mesh.section_multiplane(
            plane_origin=[0.0, 0.0, 0.0],
            plane_normal=[0.0, 0.0, 1.0],
            heights=heights,
        )

        slices: list[BaseGeometry] = []
        for k in range(n_layers):
            parts = [self._section_to_geometry(sections[k * samples + i])
                     for i in range(samples)]
            parts = [p for p in parts if not p.is_empty]
            slices.append(g2.union(parts) if parts else g2.EMPTY)

        stack = LayerStack(
            layer_height=lh,
            slices=slices,
            model_z_min=z_min,
            model_z_max=z_max,
            xy_bounds=(float(bounds[0][0]), float(bounds[0][1]),
                       float(bounds[1][0]), float(bounds[1][1])),
        )
        return stack

    # ------------------------------------------------------------------
    @staticmethod
    def _sample_offsets(samples: int, lh: float, eps: float) -> list[float]:
        """スラブ内のサンプル平面のオフセット (レイヤ下端からの相対 z)。"""
        if samples == 1:
            return [0.5 * lh]
        lo = eps
        hi = lh - eps
        step = (hi - lo) / (samples - 1)
        return [lo + i * step for i in range(samples)]

    @staticmethod
    def _section_to_geometry(section) -> BaseGeometry:
        if section is None:
            return g2.EMPTY
        try:
            polys = section.polygons_full
        except Exception:  # pragma: no cover - trimesh が閉じた輪郭を作れない場合
            return g2.EMPTY
        if polys is None or len(polys) == 0:
            return g2.EMPTY
        return g2.union(list(polys))
