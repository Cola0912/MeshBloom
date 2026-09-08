"""STEP 14 相当: 枝径の決定規則。

**本ツール独自のヒューリスティック**であり、CuraEngine / PrusaSlicer の
再現ではない (指示書 14 節の明記要求に対応)。

規則
----
``branch_profile=linear`` の既存規則:
1. tip: ``r = tip_diameter / 2``
2. 1 レイヤ降下: ``r <- min(r_max, r + branch_diameter_growth/2 * layer_height)``
3. merge: ``r <- min(r_max, sqrt(r1^2 + r2^2))``  (断面積保存)
4. root flare: プレートから ``root_flare_height`` 以内では
   ``root_diameter_min/2`` へ向けて線形に太らせる
5. 全ての半径は ``min_printable_feature/2`` 以上

いずれも「太らせた結果 influence area が消えるなら、太らせない」という
フォールバックと組み合わせて使う (呼び出し側の責務)。

``organic`` はライン幅とノズル径で制約した先端遷移、および太り角度を使う。
参照元との対応と独自の補間は docs/TREE_SUPPORT_METHOD.md に記載する。
"""

from __future__ import annotations

import math

from .config import SupportConfig

__all__ = ["RadiusSolver"]


class RadiusSolver:
    def __init__(self, config: SupportConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    @property
    def tip_radius(self) -> float:
        return max(self.config.tip_radius, self.config.min_printable_feature * 0.5)

    def grow(self, radius: float, distance_to_tip: int | None = None) -> float:
        """1 レイヤ降下したときの半径。"""
        cfg = self.config
        if cfg.branch_profile == "organic" and distance_to_tip is not None:
            height = distance_to_tip * cfg.layer_height
            transition = max(cfg.contact_height, 2*cfg.nozzle_diameter,
                             (cfg.branch_diameter-cfg.tip_diameter)/2 / math.tan(math.radians(35)))
            t = min(1., height/transition)
            # Smooth transition from the contact neck to the nominal branch;
            # then a conical radius growth. Merged branches never shrink.
            target = cfg.tip_radius + (cfg.branch_diameter/2-cfg.tip_radius)*(t*t*(3-2*t))
            target += max(0., height-transition)*math.tan(math.radians(cfg.branch_diameter_angle))
            grown = max(radius, target)
            return min(cfg.max_radius, radius+cfg.effective_line_width/2, grown)
        return min(cfg.max_radius, radius + cfg.radius_growth_per_layer)

    def merge(self, r1: float, r2: float) -> float:
        """merge 後の半径 (断面積保存)。"""
        return min(self.config.max_radius, math.hypot(r1, r2))

    def flare(self, radius: float, layer: int) -> float:
        """ビルドプレート近傍で足を広げた半径。"""
        cfg = self.config
        if cfg.root_flare_height <= 0.0 or cfg.root_radius_min <= radius:
            return radius
        z = layer * cfg.layer_height
        if z >= cfg.root_flare_height:
            return radius
        t = 1.0 - z / cfg.root_flare_height
        target = radius + (cfg.root_radius_min - radius) * t
        return min(cfg.max_radius, max(radius, target))

    def clamp(self, radius: float) -> float:
        cfg = self.config
        return min(cfg.max_radius, max(cfg.min_printable_feature * 0.5, radius))

    def next_radius(self, radius: float, next_layer: int) -> float:
        """1 レイヤ下へ降りたときの目標半径 (成長 + root flare)。"""
        return self.clamp(self.flare(self.grow(radius), next_layer))
