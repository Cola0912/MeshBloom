"""STEP 12: Centerline の平滑化。

    p[i]' = (1 - lambda) * p[i] + lambda * (p[i-1] + p[i+1]) / 2

**最重要条件**: 平滑化後の点が influence area の外へ出てはいけない。
各反復ごとに ``project_to_influence_area()`` を行う。

さらに最後に「親からの水平移動量が ``move_max`` を超えない」制約を
``influence_area & disk(parent_pos, move_max)`` への射影で強制する。
親領域と子領域の包含関係から、この交差は必ず非空である
(docs/ARCHITECTURE.md 9 節)。
"""

from __future__ import annotations

import math

from shapely.geometry import Point

from . import geometry2d as g2
from .config import SupportConfig
from .graph import TreeGraph

__all__ = ["CenterlineSmoother"]


class CenterlineSmoother:
    def __init__(self, config: SupportConfig) -> None:
        self.config = config
        self.projected = 0
        self.step_fixes = 0
        self.iterations_run = 0

    # ------------------------------------------------------------------
    def smooth(self, graph: TreeGraph) -> TreeGraph:
        cfg = self.config
        self.projected = 0
        self.step_fixes = 0

        nodes = [n for n in graph.nodes.values() if n.position is not None]
        if not nodes:
            return graph

        lam = max(0.0, min(1.0, cfg.smoothing_lambda))
        for _ in range(cfg.smoothing_iterations):
            snapshot = {n.id: n.position for n in nodes}
            for node in nodes:
                if node.locked:
                    continue
                neighbors: list[tuple[float, float]] = []
                for pid in node.parent_ids:
                    p = snapshot.get(pid)
                    if p is not None:
                        neighbors.append(p)
                for cid in node.child_ids:
                    p = snapshot.get(cid)
                    if p is not None:
                        neighbors.append(p)
                if not neighbors:
                    continue
                mx = sum(p[0] for p in neighbors) / len(neighbors)
                my = sum(p[1] for p in neighbors) / len(neighbors)
                cur = snapshot[node.id]
                nx = (1.0 - lam) * cur[0] + lam * mx
                ny = (1.0 - lam) * cur[1] + lam * my
                node.position = self._project(node, (nx, ny))
            self.iterations_run += 1

        self.enforce_max_step(graph)
        return graph

    # ------------------------------------------------------------------
    def _project(self, node, pt) -> tuple[float, float]:
        area = node.influence_area
        if area.is_empty:
            return pt
        if area.covers(Point(pt[0], pt[1])):
            return pt
        self.projected += 1
        return g2.project_point(area, pt)

    # ------------------------------------------------------------------
    def enforce_max_step(self, graph: TreeGraph, passes: int = 2) -> int:
        """親からの移動量が ``move_max`` を超えないように下から上へ矯正する。"""
        cfg = self.config
        fixed = 0
        for _ in range(max(1, passes)):
            changed = 0
            for layer in graph.layers():
                for node in graph.nodes_at(layer):
                    if node.position is None:
                        continue
                    for cid in node.child_ids:
                        child = graph.nodes.get(cid)
                        if child is None or child.position is None:
                            continue
                        d = math.hypot(child.position[0] - node.position[0],
                                       child.position[1] - node.position[1])
                        if d <= cfg.move_max + 1e-12:
                            continue
                        disk = Point(*node.position).buffer(
                            cfg.move_max, quad_segs=cfg.quad_segs)
                        region = g2.intersection(child.influence_area, disk)
                        if region.is_empty:
                            child.position = g2.project_point(
                                child.influence_area, node.position)
                        else:
                            child.position = g2.project_point(region, child.position)
                        # 射影後も超えていれば親側へ引き戻す
                        d2 = math.hypot(child.position[0] - node.position[0],
                                        child.position[1] - node.position[1])
                        if d2 > cfg.move_max:
                            s = cfg.move_max / d2
                            child.position = (
                                node.position[0] + (child.position[0] - node.position[0]) * s,
                                node.position[1] + (child.position[1] - node.position[1]) * s,
                            )
                        changed += 1
            fixed += changed
            if changed == 0:
                break
        self.step_fixes += fixed
        return fixed

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "smoothing_iterations": self.iterations_run,
            "smoothing_projections": self.projected,
            "smoothing_step_fixes": self.step_fixes,
        }
