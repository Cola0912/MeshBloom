"""STEP 9: Centerline (枝中心線) の復元。

**各レイヤの influence polygon から独立に最近傍点を選ぶ方式は採らない。**
それでは連続する点の距離が ``move_max`` を超えうる。

influence の伝播は既に終わっているので、**root 側から tip 側へ** 復元する::

    p[root]  = argmin cost_root(p),  p in influence_area[root]

    feasible = influence_area[layer] & circle(p[parent], move_max)
    p[layer] = argmin cost(p),       p in feasible

``feasible`` から選ぶ構成なので

    distance(p[layer], p[layer-1]) <= move_max

が**構成的に**保証される。しかも伝播式
``parent.influence ⊆ dilate(child.influence, move_max)`` より
``feasible`` は必ず非空である (親の任意の点から子の領域までの最近距離は move_max 以下)。

cost
----
上位ノードのコストは重み付き距離の和 (MVP)::

    cost(p) = w_tip  * |p - tip_target|            tip 方向
            + w_par  * |p - p_parent|              parent 方向 (= 鉛直維持)
            + w_curv * |p - p_extrapolated|        曲率 (直前の進行方向の延長)
            + w_pref * max(0, |p - p_parent| - move_preferred)   preferred 角の超過罰

root のコストは::

    cost_root(p) = |p - target| - w_stab * boundary_distance(p)

``target`` は subtree の tip 重心 (preferred direction = 鉛直なので真下)、
``boundary_distance`` は influence area の縁からの距離 (大きいほど接地が安定)。

merge 地点は 1 つの ``SupportNode`` として表現されるため、
合流する 2 本の枝は**同一の junction 座標を共有する** (数値誤差で分かれない)。

平滑化はここでは行わない (STEP 12)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from . import geometry2d as g2
from .config import SupportConfig
from .debug_io import DebugWriter, SvgLayer
from .graph import SupportNode, TreeGraph

__all__ = ["CenterlineSolver", "CenterlineVerification"]


@dataclass
class CenterlineVerification:
    """STEP 9 完了時の検証結果。"""

    inside_influence: bool = True
    outside_collision: bool = True
    step_within_move_max: bool = True
    tips_connected: bool = True
    roots_on_build_plate: bool = True
    acyclic: bool = True
    angle_within_max: bool = True

    max_step: float = 0.0
    max_angle: float = 0.0
    min_clearance: float = math.inf
    max_penetration: float = 0.0
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "inside_influence": self.inside_influence,
            "outside_collision": self.outside_collision,
            "step_within_move_max": self.step_within_move_max,
            "tips_connected": self.tips_connected,
            "roots_on_build_plate": self.roots_on_build_plate,
            "acyclic": self.acyclic,
            "angle_within_max": self.angle_within_max,
            "max_step": self.max_step,
            "max_angle": self.max_angle,
            "max_collision_penetration": self.max_penetration,
            "min_clearance": (None if not math.isfinite(self.min_clearance)
                              else self.min_clearance),
            "failures": list(self.failures),
        }


class CenterlineSolver:
    def __init__(self, config: SupportConfig) -> None:
        self.config = config
        self.fallbacks = 0
        self.clamped = 0
        self.candidates_evaluated = 0

    # ------------------------------------------------------------------
    def solve(self, graph: TreeGraph) -> TreeGraph:
        self.fallbacks = 0
        self.clamped = 0
        self.candidates_evaluated = 0

        tip_targets = self._subtree_tip_centroids(graph)

        for layer in graph.layers():           # 下から上へ
            for node in graph.nodes_at(layer):
                if node.position is None:
                    node.position = self._place_root(node, tip_targets)
                for cid in node.child_ids:
                    child = graph.nodes.get(cid)
                    if child is None or child.position is not None:
                        continue
                    child.position = self._place_child(graph, node, child,
                                                       tip_targets)
        return graph

    # ------------------------------------------------------------------
    def _subtree_tip_centroids(self, graph: TreeGraph) -> dict[int, tuple[float, float]]:
        """各ノードについて、その上にある tip 群の XY 重心を求める。"""
        out: dict[int, tuple[float, float]] = {}
        for layer in reversed(graph.layers()):       # 上から下へ
            for node in graph.nodes_at(layer):
                if not node.child_ids:
                    out[node.id] = node.next_position
                    continue
                sx = sy = 0.0
                n = 0
                for cid in node.child_ids:
                    p = out.get(cid)
                    if p is None:
                        continue
                    sx += p[0]
                    sy += p[1]
                    n += 1
                out[node.id] = (sx / n, sy / n) if n else node.next_position
        return out

    # ------------------------------------------------------------------
    def _place_root(self, node: SupportNode,
                    tip_targets: dict[int, tuple[float, float]]) -> tuple[float, float]:
        """root 位置を cost 最小化で選ぶ。"""
        area = node.influence_area
        if area.is_empty:
            return node.next_position
        target = tip_targets.get(node.id, node.next_position)
        cands = self._candidate_points(area, [target, node.next_position])
        w_stab = self.config.centerline_weight_stability

        best = None
        best_cost = math.inf
        for p in cands:
            self.candidates_evaluated += 1
            stability = area.exterior.distance(Point(p)) if hasattr(area, "exterior") \
                else area.boundary.distance(Point(p))
            cost = math.dist(p, target) - w_stab * stability
            if cost < best_cost:
                best_cost = cost
                best = p
        return best if best is not None else node.next_position

    # ------------------------------------------------------------------
    def _place_child(self, graph: TreeGraph, parent: SupportNode,
                     child: SupportNode,
                     tip_targets: dict[int, tuple[float, float]]) -> tuple[float, float]:
        cfg = self.config
        px, py = parent.position

        # 親から見た子領域の最近点。伝播式より必ず move_max 以内にある。
        nearest = g2.project_point(child.influence_area, (px, py))

        # feasible 用の円は**外接**多角形にする。内接多角形だと
        # 「ちょうど move_max 離れた点」を取りこぼして交差が空になり、
        # せっかくの cost 最小化が毎回フォールバックに落ちる。
        # 行き過ぎた分は最後に _clamp_step で厳密に move_max 以内へ戻す。
        disk = Point(px, py).buffer(
            cfg.move_max * g2.arc_outer_factor(cfg.quad_segs),
            quad_segs=cfg.quad_segs)
        feasible = g2.intersection(child.influence_area, disk)
        if feasible.is_empty:
            self.fallbacks += 1
            return self._clamp_step(nearest, (px, py), child.influence_area,
                                    nearest)

        pref_region = g2.intersection(child.preferred_area, feasible)
        region = pref_region if not pref_region.is_empty else feasible

        tip_target = tip_targets.get(child.id, child.next_position)

        # 曲率: 直前の進行方向をそのまま延長した点
        extrapolated = (px, py)
        if parent.parent_ids:
            gp = graph.nodes.get(parent.parent_ids[0])
            if gp is not None and gp.position is not None:
                extrapolated = (2.0 * px - gp.position[0],
                                2.0 * py - gp.position[1])

        w_tip = cfg.centerline_weight_tip
        w_par = cfg.centerline_weight_parent
        w_curv = cfg.centerline_weight_curvature
        total = max(1e-9, w_tip + w_par + w_curv)

        # 無制約の最適点 (重み付き平均) と、いくつかの離散候補を評価する
        ideal = (
            (w_tip * tip_target[0] + w_par * px + w_curv * extrapolated[0]) / total,
            (w_tip * tip_target[1] + w_par * py + w_curv * extrapolated[1]) / total,
        )
        seeds = [ideal, (px, py), tip_target, extrapolated, nearest,
                 self._preferred_step((px, py), tip_target)]
        cands = self._candidate_points(region, seeds)

        best = None
        best_cost = math.inf
        for p in cands:
            self.candidates_evaluated += 1
            step = math.dist(p, (px, py))
            cost = (w_tip * math.dist(p, tip_target)
                    + w_par * step
                    + w_curv * math.dist(p, extrapolated)
                    + cfg.centerline_weight_preferred
                    * max(0.0, step - cfg.move_preferred))
            if cost < best_cost:
                best_cost = cost
                best = p

        if best is None:
            best = g2.project_point(region, ideal)
        best = self._nudge_inside(region, best)
        return self._clamp_step(best, (px, py), child.influence_area, nearest)

    # ------------------------------------------------------------------
    def _preferred_step(self, parent_pos, target) -> tuple[float, float]:
        """preferred angle 分だけ目標へ寄せた点。"""
        px, py = parent_pos
        dx, dy = target[0] - px, target[1] - py
        dist = math.hypot(dx, dy)
        if dist <= 1e-12:
            return (px, py)
        step = min(self.config.move_preferred, dist)
        return (px + dx / dist * step, py + dy / dist * step)

    # ------------------------------------------------------------------
    def _candidate_points(self, region: BaseGeometry,
                          seeds) -> list[tuple[float, float]]:
        """``region`` 内の候補点を作る (seed の射影 + 頂点 + 代表点)。"""
        if region.is_empty:
            return []
        out: list[tuple[float, float]] = []
        seen: set[tuple[float, float]] = set()

        def add(p) -> None:
            key = (round(p[0], 9), round(p[1], 9))
            if key in seen:
                return
            seen.add(key)
            out.append((p[0], p[1]))

        for s in seeds:
            add(g2.project_point(region, s))
        rp = g2.inside_point(region)
        if rp is not None:
            add(rp)
        limit = self.config.centerline_max_vertex_candidates
        for poly in g2.polygons(region):
            coords = list(poly.exterior.coords)[:-1]
            if not coords:
                continue
            stride = max(1, len(coords) // max(1, limit))
            for c in coords[::stride]:
                add((float(c[0]), float(c[1])))
        return out

    # ------------------------------------------------------------------
    def _nudge_inside(self, region: BaseGeometry,
                      p: tuple[float, float]) -> tuple[float, float]:
        """境界上の点をわずかに内側へ寄せる。

        候補にはポリゴン頂点も含まれるため、そのままだと collision 境界に
        ちょうど接する位置が選ばれうる。余裕があるなら内側へ入れておく。
        """
        eps = self.config.centerline_interior_nudge
        if eps <= 0.0 or region.is_empty:
            return p
        inner = g2.erode_outer(region, eps, quad_segs=self.config.quad_segs)
        if inner.is_empty:
            return p
        return g2.project_point(inner, p)

    # ------------------------------------------------------------------
    def _clamp_step(self, p, parent_pos, influence: BaseGeometry | None = None,
                    safe: tuple[float, float] | None = None) -> tuple[float, float]:
        """親からの距離を厳密に ``move_max`` 以内へ収める。

        引き戻した点が influence area から外れる場合は、
        必ず move_max 以内にある最近点 ``safe`` を使う。
        """
        cfg = self.config
        d = math.dist(p, parent_pos)
        if d <= cfg.move_max:
            return (p[0], p[1])
        self.clamped += 1
        scale = (cfg.move_max * (1.0 - 1e-9)) / d
        q = (parent_pos[0] + (p[0] - parent_pos[0]) * scale,
             parent_pos[1] + (p[1] - parent_pos[1]) * scale)
        if influence is not None and not influence.buffer(1e-9).covers(Point(*q)):
            if safe is not None:
                return safe
            return g2.project_point(influence, q)
        return q

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {"centerline_fallbacks": self.fallbacks,
                "centerline_clamped": self.clamped,
                "centerline_candidates": self.candidates_evaluated}

    # ==================================================================
    # 検証
    # ==================================================================
    def verify(self, graph: TreeGraph, cache=None,
               tolerance: float = 1e-6) -> CenterlineVerification:
        cfg = self.config
        v = CenterlineVerification()

        # 1. 全 center point が influence_area 内
        for node in graph.nodes.values():
            if node.position is None:
                v.inside_influence = False
                v.failures.append(f"node {node.id} has no position")
                continue
            if not node.influence_area.buffer(tolerance).covers(Point(*node.position)):
                v.inside_influence = False
                v.failures.append(
                    f"node {node.id} position is outside its influence area")

        # 2. 全 center point が collision 外
        #
        # influence area の境界は collision の境界と一致することが多く、
        # 選ばれた点が境界上に乗るのは正常 (クリアランスがちょうど要求値)。
        # GEOS の contains は 1e-16 の丸めでも真を返すため、
        # 「境界からどれだけ内側に食い込んでいるか」で判定する。
        if cache is not None:
            for node in graph.nodes.values():
                if node.position is None:
                    continue
                kind = "tip" if node.source == "tip" else "branch"
                p = Point(*node.position)
                col = cache.collision(node.layer, node.radius, kind=kind)
                if not col.is_empty and col.contains(p):
                    penetration = col.boundary.distance(p)
                    v.max_penetration = max(v.max_penetration, penetration)
                    if penetration > tolerance:
                        v.outside_collision = False
                        v.failures.append(
                            f"node {node.id} position is {penetration:.6g} mm "
                            "inside the collision region")
                # モデル素材そのものからのクリアランス (offset 前の断面で測る)
                for _xy, geom in cache.base(node.layer, kind):
                    if geom.is_empty:
                        continue
                    v.min_clearance = min(v.min_clearance,
                                          geom.distance(p) - node.radius)

        # 3. 隣接点距離 <= move_max / 角度 <= branch_angle_max
        lh = cfg.layer_height
        for node in graph.nodes.values():
            if node.position is None:
                continue
            for cid in node.child_ids:
                child = graph.nodes.get(cid)
                if child is None or child.position is None:
                    continue
                step = math.dist(node.position, child.position)
                v.max_step = max(v.max_step, step)
                if step > cfg.move_max + tolerance:
                    v.step_within_move_max = False
                    v.failures.append(
                        f"step {step:.6f} between nodes {node.id}->{cid} "
                        f"exceeds move_max {cfg.move_max:.6f}")
                dz = abs(child.layer - node.layer) * lh
                angle = 90.0 if dz <= 0.0 else math.degrees(math.atan2(step, dz))
                v.max_angle = max(v.max_angle, angle)
                if angle > cfg.branch_angle_max + 1e-4:
                    v.angle_within_max = False
                    v.failures.append(
                        f"branch angle {angle:.4f} deg between nodes "
                        f"{node.id}->{cid} exceeds branch_angle_max")

        # 4. 全 tip が root へ接続 / 5. 全 root が build plate
        roots = set(graph.root_ids)
        for tip in graph.tips():
            node = tip
            guard = 0
            while node.parent_ids and guard < 1_000_000:
                node = graph.nodes[node.parent_ids[0]]
                guard += 1
            if node.id not in roots:
                v.tips_connected = False
                v.failures.append(f"tip {tip.id} is not connected to a root")
        for rid in graph.root_ids:
            node = graph.nodes.get(rid)
            if node is None or node.layer != 0:
                v.roots_on_build_plate = False
                v.failures.append(f"root {rid} is not on the build plate")

        # 6. cycle なし
        color: dict[int, int] = {}
        for rid in graph.root_ids:
            stack = [(rid, iter(graph.nodes[rid].child_ids))]
            color[rid] = 1
            while stack:
                nid, it = stack[-1]
                advanced = False
                for cid in it:
                    state = color.get(cid, 0)
                    if state == 1:
                        v.acyclic = False
                        v.failures.append(f"cycle detected at node {cid}")
                        continue
                    if state == 2:
                        continue
                    color[cid] = 1
                    stack.append((cid, iter(graph.nodes[cid].child_ids)))
                    advanced = True
                    break
                if not advanced:
                    color[nid] = 2
                    stack.pop()

        return v

    # ==================================================================
    # Debug
    # ==================================================================
    def write_debug(self, stack, graph: TreeGraph, debug: DebugWriter,
                    verification: CenterlineVerification | None = None,
                    every: int = 1) -> int:
        if not debug.enabled("centerline"):
            return 0

        payload = graph.to_dict()
        payload["centerline_stats"] = self.stats()
        if verification is not None:
            payload["verification"] = verification.to_dict()
        debug.write_json("centerline", "tree.json", payload)
        debug.write_obj_polylines("centerline", "tree.obj",
                                  graph.centerline_polylines(),
                                  comment="tree support centerline")

        written = 0
        layers = graph.layers()
        for j in layers[::max(1, every)]:
            nodes = graph.nodes_at(j)
            if not nodes:
                continue
            debug.write_svg(
                "centerline", f"layer_{j:04d}.svg",
                [
                    SvgLayer("model", stack[j], fill="#475569", stroke="#0f172a",
                             opacity=0.8),
                    SvgLayer("influence", [n.influence_area for n in nodes],
                             fill="#38bdf8", stroke="#0284c7", opacity=0.3),
                    SvgLayer("preferred", [n.preferred_area for n in nodes],
                             fill="#a78bfa", stroke="#7c3aed", opacity=0.25),
                    SvgLayer("centers", None,
                             points=[n.position for n in nodes
                                     if n.position is not None],
                             point_radius=max(0.12, self.config.tip_radius * 0.5),
                             point_fill="#dc2626"),
                ],
                bounds=None,
                title=(f"layer {j:04d} z={j * graph.layer_height:.3f} "
                       f"nodes={len(nodes)}"),
            )
            written += 1
        return written
