"""STEP 13: 3D collision の再検証と補正。

2D レイヤ collision だけでは、斜めの太い tube が斜め上/斜め下のモデルへ
食い込む可能性がある。ここでは centerline の各点を半径 ``r`` の球とみなし、
近傍レイヤのモデル断面との 3D 距離を検査する。

レイヤ ``i`` のモデルはスラブ ``[i*lh, (i+1)*lh]`` を占めるので、
z 座標 ``zp`` の球中心からの垂直距離は::

    dz = max(0, i*lh - zp, zp - (i+1)*lh)

このとき必要な水平クリアランスは::

    need = sqrt(max(0, (r + xy_gap)^2 - dz^2))

これを下回っていれば、最近点と反対方向へ nudge し、
influence area へ射影し直してから再判定する。
"""

from __future__ import annotations

import math

from shapely.geometry import Point

from . import geometry2d as g2
from .config import SupportConfig
from .graph import TreeGraph
from .volumes import CollisionCache

__all__ = ["Collision3DCorrector"]


class Collision3DCorrector:
    def __init__(self, config: SupportConfig, cache: CollisionCache) -> None:
        self.config = config
        self.cache = cache
        self.stack = cache.stack
        self.violations_found = 0
        self.violations_fixed = 0

    # ------------------------------------------------------------------
    def correct(self, graph: TreeGraph) -> int:
        cfg = self.config
        for _ in range(max(1, cfg.collision3d_iterations)):
            moved = 0
            for node in graph.nodes.values():
                if node.position is None:
                    continue
                push = self._required_push(node)
                if push is None:
                    continue
                self.violations_found += 1
                target = (node.position[0] + push[0], node.position[1] + push[1])
                new_pos = g2.project_point(node.influence_area, target)
                if new_pos != node.position:
                    node.position = new_pos
                    moved += 1
            if moved == 0:
                break

        # 平滑化と同じ制約強制で角度不変条件を回復する
        from .smoothing import CenterlineSmoother
        smoother = CenterlineSmoother(self.config)
        smoother.enforce_max_step(graph)

        # 最終確認
        remaining = 0
        for node in graph.nodes.values():
            if node.position is None:
                continue
            if self._required_push(node) is not None:
                remaining += 1
        self.violations_fixed = max(0, self.violations_found - remaining)
        self.remaining = remaining
        return remaining

    # ------------------------------------------------------------------
    def _required_push(self, node) -> tuple[float, float] | None:
        """3D クリアランス違反があれば、押し戻すべきベクトルを返す。"""
        cfg = self.config
        lh = cfg.layer_height
        zp = node.layer * lh
        reach = node.radius + cfg.xy_gap
        span = int(math.ceil(reach / lh)) + 1

        p = Point(node.position[0], node.position[1])
        worst = None
        for i in range(node.layer - span, node.layer + span + 1):
            model = self.stack[i]
            if model.is_empty:
                continue
            dz = max(0.0, i * lh - zp, zp - (i + 1) * lh)
            if dz >= reach:
                continue
            need = math.sqrt(max(0.0, reach * reach - dz * dz))
            d = model.distance(p)
            if d >= need - 1e-9:
                continue
            deficit = need - d
            if worst is None or deficit > worst[0]:
                worst = (deficit, model)

        if worst is None:
            return None

        deficit, model = worst
        from shapely.ops import nearest_points
        q = nearest_points(model, p)[0]
        dx, dy = p.x - q.x, p.y - q.y
        norm = math.hypot(dx, dy)
        if norm < 1e-9:
            # 中心がモデル内部: 代表点から離れる方向へ逃がす
            c = model.centroid
            dx, dy = p.x - c.x, p.y - c.y
            norm = math.hypot(dx, dy)
            if norm < 1e-9:
                return (deficit, 0.0)
        scale = (deficit + 1e-6) / norm
        return (dx * scale, dy * scale)

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "collision3d_violations": self.violations_found,
            "collision3d_fixed": self.violations_fixed,
            "collision3d_remaining": getattr(self, "remaining", 0),
        }
