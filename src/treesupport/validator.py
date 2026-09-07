"""STEP 15 / 指示書 26 節: 出力前の自動検証。

必ず出力するメトリクス::

    coverage_ratio
    minimum_model_clearance
    maximum_branch_angle
    tip_count
    root_count
    total_centerline_length
    support_volume
    non_manifold_edge_count
    generation_time

不変条件::

    1. 全 tip が root へ接続されている
    2. 全 root がビルドプレートへ到達している
    3. branch angle <= branch_angle_max
    4. model と support が禁止領域で交差しない
    5. tip に必要な Z ギャップがある
    6. 枝径 >= min_printable_feature
"""

from __future__ import annotations

import math

import numpy as np
import trimesh
from shapely.geometry import Point

from . import geometry2d as g2
from .config import SupportConfig

__all__ = ["Validator"]

#: 3D クリアランス測定に使うサポート頂点の上限 (性能のため)
_MAX_SAMPLE_VERTICES = 60_000


class Validator:
    def __init__(self, config: SupportConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    def validate(self, result) -> dict:
        cfg = self.config
        metrics: dict = {}
        failures: list[str] = []
        warnings: list[str] = []

        graph = result.graph
        tips = result.tips
        stack = result.stack

        # --- 基本メトリクス ---
        metrics["coverage_ratio"] = tips.coverage_ratio if tips else 1.0
        metrics["effective_coverage_ratio"] = (
            tips.effective_coverage_ratio if tips else 1.0)
        metrics["unsupportable_area"] = tips.unsupportable_area if tips else 0.0
        metrics["tip_count"] = len(graph.tips()) if graph else 0
        metrics["root_count"] = len(graph.root_ids) if graph else 0
        metrics["node_count"] = len(graph.nodes) if graph else 0
        metrics["branch_count"] = len(graph.branches) if graph else 0
        metrics["total_centerline_length"] = (
            graph.centerline_length() if graph else 0.0)
        metrics["maximum_branch_angle"] = (
            graph.max_branch_angle() if graph else 0.0)
        metrics["generation_time"] = sum(result.timings.values())

        mesh = result.support_mesh
        if mesh is not None and len(mesh.faces) > 0:
            metrics["support_volume"] = float(mesh.volume)
            metrics["support_area"] = float(mesh.area)
            metrics["mesh_watertight"] = bool(mesh.is_watertight)
            metrics["non_manifold_edge_count"] = self._non_manifold_edges(mesh)
            metrics["mesh_bounds_min"] = [float(v) for v in mesh.bounds[0]]
            metrics["mesh_bounds_max"] = [float(v) for v in mesh.bounds[1]]
        else:
            metrics["support_volume"] = 0.0
            metrics["support_area"] = 0.0
            metrics["mesh_watertight"] = True
            metrics["non_manifold_edge_count"] = 0

        # --- 不変条件 1: 全 tip が root へ接続 ---
        if graph is not None:
            roots = set(graph.root_ids)
            orphan = 0
            for tip in graph.tips():
                node = tip
                guard = 0
                while node.parent_ids and guard < 1_000_000:
                    node = graph.nodes[node.parent_ids[0]]
                    guard += 1
                if node.id not in roots:
                    orphan += 1
            metrics["orphan_tips"] = orphan
            if orphan:
                failures.append(f"{orphan} tip(s) are not connected to any root")

            # --- 不変条件 2: root がビルドプレートに接地 ---
            bad_roots = [n.id for n in graph.roots() if n.layer != 0]
            if bad_roots:
                failures.append(
                    f"{len(bad_roots)} root(s) do not reach the build plate "
                    f"(layer != 0)"
                )

            # --- 不変条件 3: 枝角度 ---
            max_angle = metrics["maximum_branch_angle"]
            if max_angle > cfg.branch_angle_max + 1e-6:
                failures.append(
                    f"maximum branch angle {max_angle:.3f} deg exceeds "
                    f"branch_angle_max {cfg.branch_angle_max:.3f} deg"
                )

            # --- 不変条件 6: 枝径 ---
            min_r = min((n.radius for n in graph.nodes.values()), default=math.inf)
            metrics["min_branch_diameter"] = (
                2.0 * min_r if math.isfinite(min_r) else 0.0)
            if math.isfinite(min_r) and 2.0 * min_r < cfg.min_printable_feature - 1e-9:
                failures.append(
                    f"minimum branch diameter {2*min_r:.4f} mm is below "
                    f"min_printable_feature {cfg.min_printable_feature:.4f} mm"
                )

        # --- 不変条件 4: 2D クリアランス (レイヤ断面基準) ---
        clearance_2d = self._layer_clearance(graph, stack)
        metrics["minimum_model_clearance_2d"] = clearance_2d
        if clearance_2d is not None and clearance_2d < cfg.xy_gap - 1e-4:
            failures.append(
                f"minimum lateral clearance {clearance_2d:.4f} mm is below "
                f"xy_gap {cfg.xy_gap:.4f} mm"
            )

        # --- 不変条件 4b: 3D クリアランス (メッシュ同士) ---
        clearance_3d = None
        if mesh is not None and len(mesh.faces) > 0 and result.model is not None:
            clearance_3d, inside = self._mesh_clearance(mesh, result.model.mesh)
            metrics["minimum_model_clearance"] = clearance_3d
            metrics["support_vertices_inside_model"] = inside
            if inside > 0:
                failures.append(
                    f"{inside} support vertices are inside the model "
                    "(support intersects the model)"
                )
        else:
            metrics["minimum_model_clearance"] = clearance_2d

        # --- 不変条件 5: Z ギャップ ---
        gap = self._tip_z_gap(graph, stack)
        metrics["minimum_tip_z_gap"] = gap
        if gap is not None and gap < cfg.top_z_gap - 1e-6:
            failures.append(
                f"minimum tip Z gap {gap:.4f} mm is below top_z_gap "
                f"{cfg.top_z_gap:.4f} mm"
            )

        # --- 警告 ---
        if tips is not None and tips.unsupportable_area > 0.0:
            warnings.append(
                f"{tips.unsupportable_area:.3f} mm^2 of overhang cannot be "
                "supported from the build plate (blocked or too close to the plate)"
            )
        if result.propagation is not None and result.propagation.dead_ends:
            warnings.append(
                f"{len(result.propagation.dead_ends)} branch(es) hit a dead end "
                "and were pruned"
            )
        if mesh is not None and len(mesh.faces) > 0 and not mesh.is_watertight:
            failures.append("support mesh is not watertight")
        if mesh is not None and len(mesh.faces) > 0:
            if mesh.bounds[0, 2] < -1e-6:
                failures.append("support mesh extends below the build plate")
            if not mesh.is_volume:
                failures.append("support mesh is not an oriented positive volume")
        if result.metrics.get("mesh_union_ok") is False:
            failures.append("support mesh union failed")

        return {
            "passed": not failures,
            "failures": failures,
            "warnings": warnings,
            "metrics": metrics,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _non_manifold_edges(mesh: trimesh.Trimesh) -> int:
        try:
            counts = np.bincount(mesh.edges_unique_inverse,
                                 minlength=len(mesh.edges_unique))
            return int(np.count_nonzero(counts != 2))
        except Exception:  # pragma: no cover
            return -1

    # ------------------------------------------------------------------
    def _layer_clearance(self, graph, stack) -> float | None:
        """側方クリアランスの最小値。

        比較対象は ``CollisionCache`` が collision に含めるのと同じレイヤ範囲、
        すなわち「垂直方向のギャップが不足していて、側方に xy_gap を要する」
        レイヤだけ。tip の真上 (ちょうど top_z_gap 離れたモデル) は
        側方クリアランスの対象ではないので除外する。
        """
        if graph is None or stack is None or not graph.nodes:
            return None
        cfg = self.config
        n_top = cfg.z_gap_layers_top
        n_bot = cfg.z_gap_layers_bottom
        worst = math.inf
        for node in graph.nodes.values():
            if node.position is None:
                continue
            p = Point(*node.position)
            t_max = n_top - 1 if node.source == "tip" else n_top
            for i in range(node.layer - 1 - n_bot, node.layer + t_max + 1):
                model = stack[i]
                if model.is_empty:
                    continue
                worst = min(worst, model.distance(p) - node.radius)
        return None if not math.isfinite(worst) else float(worst)

    # ------------------------------------------------------------------
    def _mesh_clearance(self, support: trimesh.Trimesh,
                        model: trimesh.Trimesh) -> tuple[float, int]:
        verts = np.asarray(support.vertices, dtype=float)
        if len(verts) > _MAX_SAMPLE_VERTICES:
            idx = np.linspace(0, len(verts) - 1, _MAX_SAMPLE_VERTICES).astype(int)
            verts = verts[idx]
        closest, distance, _ = model.nearest.on_surface(verts)
        inside = 0
        if model.is_watertight:
            try:
                mask = model.contains(verts)
                inside = int(np.count_nonzero(mask))
                if inside:
                    return (-float(distance[mask].max()), inside)
            except Exception:  # pragma: no cover
                inside = 0
        return (float(distance.min()), inside)

    # ------------------------------------------------------------------
    def _tip_z_gap(self, graph, stack) -> float | None:
        """tip 上端から真上のモデル下面までの最小距離。"""
        if graph is None or stack is None:
            return None
        cfg = self.config
        lh = cfg.layer_height
        worst = math.inf
        for tip in graph.tips():
            if tip.position is None:
                continue
            disk = Point(*tip.position).buffer(tip.radius, quad_segs=cfg.quad_segs)
            for i in range(tip.layer, stack.n_layers):
                model = stack[i]
                if model.is_empty or not model.intersects(disk):
                    continue
                worst = min(worst, i * lh - tip.layer * lh)
                break
        return None if not math.isfinite(worst) else float(worst)
