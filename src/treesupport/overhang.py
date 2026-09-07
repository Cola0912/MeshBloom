"""STEP 3: オーバーハング検出 (support demand 領域の算出)。

三角形法線ではなく**レイヤ断面の差分**で求める (指示書 7 節)。

    allowed_offset  = layer_height * tan(overhang_angle_from_vertical)
    support_demand[k] = P[k] - dilate(P[k-1], allowed_offset)

* ``k = 0`` はビルドプレートが支えるため demand は空。
* ``P[k-1]`` の膨張には**内接近似** (``dilate``) を使う。
  内接近似は真の Minkowski 和より小さいため、差分 (= demand) は真値より
  わずかに大きくなる。サポートを過剰にする方向であり安全側。
* ``min_overhang_area`` 未満のかけらは捨てる。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shapely.geometry.base import BaseGeometry

from . import geometry2d as g2
from .config import SupportConfig
from .debug_io import DebugWriter, SvgLayer
from .slicer import LayerStack

__all__ = ["OverhangResult", "OverhangDetector"]


@dataclass
class OverhangResult:
    """レイヤごとの support demand 領域。インデックスはモデルレイヤ番号。"""

    demand: list[BaseGeometry]
    layer_height: float
    allowed_offset: float
    dropped_area: float = 0.0
    meta: dict = field(default_factory=dict)

    def __getitem__(self, k: int) -> BaseGeometry:
        if k < 0 or k >= len(self.demand):
            return g2.EMPTY
        return self.demand[k]

    def __len__(self) -> int:
        return len(self.demand)

    @property
    def total_area(self) -> float:
        return float(sum(d.area for d in self.demand))

    @property
    def layers_with_demand(self) -> list[int]:
        return [k for k, d in enumerate(self.demand) if not d.is_empty]

    def stats(self) -> dict:
        layers = self.layers_with_demand
        return {
            "allowed_offset": self.allowed_offset,
            "total_demand_area": self.total_area,
            "dropped_area": self.dropped_area,
            "n_layers_with_demand": len(layers),
            "first_demand_layer": layers[0] if layers else -1,
            "last_demand_layer": layers[-1] if layers else -1,
        }


class OverhangDetector:
    def __init__(self, config: SupportConfig) -> None:
        self.config = config

    def detect(self, stack: LayerStack) -> OverhangResult:
        cfg = self.config
        allowed = cfg.overhang_allowed_offset
        demand: list[BaseGeometry] = []
        dropped = 0.0

        for k in range(stack.n_layers):
            current = stack[k]
            if current.is_empty or k == 0:
                # k == 0 はビルドプレートが支える。
                demand.append(g2.EMPTY)
                continue

            below = stack[k - 1]
            if below.is_empty:
                raw = current
            else:
                supported = g2.dilate(below, allowed, quad_segs=cfg.quad_segs)
                raw = g2.difference(current, supported)

            if raw.is_empty:
                demand.append(g2.EMPTY)
                continue

            if cfg.overhang_expansion > 0.0:
                raw = g2.dilate(raw, cfg.overhang_expansion, quad_segs=cfg.quad_segs)
                raw = g2.intersection(raw, current)

            filtered = g2.drop_small(raw, cfg.min_overhang_area)
            dropped += max(0.0, raw.area - filtered.area)
            demand.append(filtered)

        return OverhangResult(
            demand=demand,
            layer_height=stack.layer_height,
            allowed_offset=allowed,
            dropped_area=dropped,
        )

    # ------------------------------------------------------------------
    def write_debug(self, stack: LayerStack, result: OverhangResult,
                    debug: DebugWriter, every: int = 1,
                    only_nonempty: bool = True) -> int:
        """モデル断面とオーバーハング領域を SVG で出力する。"""
        if not debug.enabled("overhang") and not debug.enabled("slices"):
            return 0
        written = 0
        for k in range(0, stack.n_layers, max(1, every)):
            model = stack[k]
            over = result[k]
            if only_nonempty and model.is_empty and over.is_empty:
                continue
            title = f"layer {k:04d}  z={stack.z_mid(k):.3f}  demand={over.area:.3f} mm^2"
            if debug.enabled("slices"):
                debug.write_svg(
                    "slices", f"layer_{k:04d}_model.svg",
                    [SvgLayer("model", model, fill="#94a3b8", stroke="#334155")],
                    bounds=stack.xy_bounds, title=title,
                )
            if debug.enabled("overhang"):
                debug.write_svg(
                    "overhang", f"layer_{k:04d}_overhang.svg",
                    [
                        SvgLayer("model_below", stack[k - 1], fill="#e2e8f0",
                                 stroke="#cbd5e1", opacity=0.9),
                        SvgLayer("model", model, fill="#94a3b8", stroke="#334155",
                                 opacity=0.55),
                        SvgLayer("demand", over, fill="#ef4444", stroke="#b91c1c",
                                 opacity=0.75),
                    ],
                    bounds=stack.xy_bounds, title=title,
                )
            written += 1
        debug.write_json("overhang", "overhang_stats.json", result.stats())
        return written
