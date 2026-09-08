"""STEP 6A/6B/7: Influence Area の top-down 伝播 (+ STEP 11: bottom-up trim)。

本ツールの中核。**最初に 3D 空間上の枝中心線を決めてから障害物回避させる方式は採らない。**
各レイヤについて「枝中心が存在可能な 2D 領域 (influence area)」を上から下へ伝播し、
具体的な XY はすべての伝播が終わってから決める (STEP 9)。

STEP 6A (``use_reach=False``)::

    raw       = dilate(A[j], move_max)
    candidate = raw - collision[j-1, radius]

STEP 6B (``use_reach=True``, 既定)::

    candidate = candidate & reach[j-1, radius]

``reach`` は collision の単なる補集合ではなく「そこからビルドプレートまで
降りられる」領域なので、枝が行き止まりに入り込むことが構造的に起こらない。

半径の扱い
----------
半径は伝播中に増加する。**その層で実際に使う半径で collision / reach を
取り直す**。目標半径では領域が消える場合の挙動は
``radius_growth_fallback`` で選ぶ:

* ``True`` (既定): その層では太らせずに降りる。
  ``reach[r][j] = (dilate(reach[r][j-1], move_max) & world) - collision[j][r]``
  の定義から、**同じ半径なら次のレイヤの領域が空にならないことが保証される**。
  ノードが持つ ``radius`` は常に「実際に collision / reach を満たした半径」
  なので、細いままであることを隠しているわけではない。
  何回抑制されたかは ``radius_growth_blocked`` に記録する。
* ``False``: 目標半径で降りられない時点で ``UNREACHABLE`` とする (厳格モード)。

いずれの場合も、実半径ですら降りられなければ ``UNREACHABLE`` になり、
その部分木は ``prune_unreachable`` で除去される。成功扱いにはしない。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from . import geometry2d as g2
from .config import SupportConfig
from .debug_io import DebugWriter, SvgLayer
from .graph import NodeStatus, SupportNode, TreeGraph
from .merge import TreeMerger
from .radius import RadiusSolver
from .slicer import LayerStack
from .tips import TipSet
from .volumes import CollisionCache

__all__ = ["PropagationResult", "TreePropagator", "StepTrace"]


@dataclass
class StepTrace:
    """1 回の伝播ステップの中間状態 (デバッグ可視化用)。"""

    node_id: int
    from_layer: int
    to_layer: int
    radius: float
    previous: BaseGeometry
    expanded: BaseGeometry
    collision: BaseGeometry
    reach: BaseGeometry
    result: BaseGeometry
    status: str


@dataclass
class PropagationResult:
    graph: TreeGraph
    unplaced_tips: list[int] = field(default_factory=list)
    dead_ends: list[int] = field(default_factory=list)
    pruned_nodes: int = 0
    radius_growth_blocked: int = 0
    #: contact id -> True/False
    tip_reachable: dict[int, bool] = field(default_factory=dict)
    #: レイヤ -> そのレイヤの influence area の和 (デバッグ用)
    influence_by_layer: dict[int, BaseGeometry] = field(default_factory=dict)
    traces: list[StepTrace] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def reachable_tips(self) -> list[int]:
        return sorted(k for k, v in self.tip_reachable.items() if v)

    @property
    def unreachable_tips(self) -> list[int]:
        return sorted(k for k, v in self.tip_reachable.items() if not v)

    def total_influence_area(self) -> float:
        return float(sum(g2.area(n.influence_area)
                         for n in self.graph.nodes.values()))

    def stats(self) -> dict:
        d = self.graph.stats()
        d.update({
            "tip_count": len(self.tip_reachable),
            "reachable_tip_count": len(self.reachable_tips),
            "unreachable_tip_count": len(self.unreachable_tips),
            "unplaced_tips": len(self.unplaced_tips),
            "dead_ends": len(self.dead_ends),
            "pruned_nodes": self.pruned_nodes,
            "radius_growth_blocked": self.radius_growth_blocked,
            "total_influence_area": self.total_influence_area(),
        })
        return d


class TreePropagator:
    def __init__(self, config: SupportConfig, cache: CollisionCache,
                 radius_solver: RadiusSolver | None = None,
                 merger: TreeMerger | None = None,
                 use_reach: bool = True,
                 record_trace: bool = False) -> None:
        self.config = config
        self.cache = cache
        self.radius = radius_solver or RadiusSolver(config)
        self.merger = merger or TreeMerger(config, cache, self.radius)
        self.use_reach = use_reach
        self.record_trace = record_trace

    # ------------------------------------------------------------------
    def allowed_region(self, layer: int, radius: float,
                       kind: str = "branch") -> BaseGeometry:
        """そのレイヤ・その半径で枝中心が存在してよい領域。"""
        if self.use_reach:
            return (self.cache.reach_tip(layer, radius) if kind == "tip"
                    else self.cache.reach(layer, radius))
        return g2.difference(self.cache.world,
                             self.cache.collision(layer, radius, kind=kind))

    # ------------------------------------------------------------------
    def propagate(self, stack: LayerStack, tipset: TipSet,
                  keep_influence: bool = False,
                  prune: bool = True) -> PropagationResult:
        cfg = self.config
        graph = TreeGraph(cfg.layer_height)
        result = PropagationResult(graph=graph)
        for tip in tipset.tips:
            result.tip_reachable[tip.id] = False

        if not tipset.by_layer:
            return result

        top = max(tipset.by_layer)
        active: list[int] = []

        for j in range(top, -1, -1):
            # --- 1. このレイヤの tip を追加 ---
            for tip in tipset.by_layer.get(j, ()):
                node = self._make_tip_node(graph, tip)
                if node is None:
                    result.unplaced_tips.append(tip.id)
                else:
                    active.append(node.id)

            if not active:
                continue

            # --- 2. merge (STEP 8。merge_enabled=False なら何もしない) ---
            active = self.merger.merge_layer(graph, j, active)

            if keep_influence:
                result.influence_by_layer[j] = g2.union(
                    [graph.nodes[i].influence_area for i in active
                     if i in graph.nodes]
                )

            if j == 0:
                for nid in active:
                    node = graph.nodes.get(nid)
                    if node is not None:
                        node.reaches_build_plate = True
                        node.status = NodeStatus.ON_BUILD_PLATE
                        graph.root_ids.append(nid)
                break

            # --- 3. 1 レイヤ下へ伝播 ---
            next_active: list[int] = []
            for nid in active:
                node = graph.nodes.get(nid)
                if node is None:
                    continue
                child = self._step_down(graph, node, j - 1, result)
                if child is None:
                    result.dead_ends.append(nid)
                else:
                    next_active.append(child.id)
            active = next_active

        if prune:
            result.pruned_nodes = graph.prune_unreachable()
        graph.build_branches()

        for node in graph.nodes.values():
            if node.reaches_build_plate:
                for cid in node.contact_ids:
                    result.tip_reachable[cid] = True
        return result

    # ------------------------------------------------------------------
    def _make_tip_node(self, graph: TreeGraph, tip) -> SupportNode | None:
        cfg = self.config
        r = self.radius.tip_radius
        influence_r = max(cfg.tip_influence, 1e-4)
        disk = Point(tip.x, tip.y).buffer(influence_r, quad_segs=cfg.quad_segs)
        allowed = self.allowed_region(tip.layer, r, kind="tip")
        area = g2.intersection(disk, allowed)
        if area.is_empty:
            # tip 中心そのものが置けるなら、微小領域として残す
            if not allowed.is_empty and allowed.covers(Point(tip.x, tip.y)):
                area = g2.intersection(
                    Point(tip.x, tip.y).buffer(cfg.area_epsilon ** 0.5,
                                               quad_segs=4),
                    allowed,
                )
            if area.is_empty:
                return None
        return graph.new_node(
            layer=tip.layer,
            radius=r,
            influence_area=area,
            preferred_area=area,
            contact_ids=[tip.id],
            next_position=(tip.x, tip.y),
            dtt=0,
            source="tip",
            status=NodeStatus.ACTIVE,
        )

    # ------------------------------------------------------------------
    def _radius_ladder(self, node: SupportNode, next_layer: int) -> list[float]:
        grown = self.radius.grow(node.radius, node.dtt + 1)
        target = self.radius.clamp(self.radius.flare(grown, next_layer))
        if not self.config.radius_growth_fallback:
            return [target]
        ladder = [target, self.radius.clamp(grown),
                  self.radius.clamp(node.radius)]
        out: list[float] = []
        for r in ladder:
            if not any(abs(r - x) < 1e-12 for x in out):
                out.append(r)
        return out

    def _step_down(self, graph: TreeGraph, node: SupportNode, next_layer: int,
                   result: PropagationResult) -> SupportNode | None:
        cfg = self.config
        expanded = g2.dilate(node.influence_area, cfg.move_max,
                             quad_segs=cfg.quad_segs)
        if expanded.is_empty:
            node.status = NodeStatus.UNREACHABLE
            node.status_reason = "expanded influence area is empty"
            return None

        chosen_area: BaseGeometry | None = None
        chosen_r = node.radius
        trace_collision = g2.EMPTY
        trace_reach = g2.EMPTY

        # collision / reach は world 全体を覆う巨大ポリゴンになりうる。
        # 着目ノードの bbox 周辺だけに矩形クリップしてからブーリアンを掛ける
        # (矩形外は expanded に含まれないので結果は変わらない)。
        clip = expanded.bounds

        ladder = self._radius_ladder(node, next_layer)
        for i, r in enumerate(ladder):
            # その層で実際に使う半径で collision / reach を取り直す
            collision = g2.clip_to_bounds(
                self.cache.collision(next_layer, r, kind="branch"), clip, pad=1e-6)
            candidate = g2.difference(expanded, collision)
            reach = g2.EMPTY
            if self.use_reach and not candidate.is_empty:
                reach = g2.clip_to_bounds(self.cache.reach(next_layer, r),
                                          clip, pad=1e-6)
                candidate = g2.intersection(candidate, reach)
            if i == 0:
                trace_collision, trace_reach = collision, reach
            if not candidate.is_empty:
                chosen_area = candidate
                chosen_r = r
                trace_collision, trace_reach = collision, reach
                if i > 0:
                    result.radius_growth_blocked += 1
                break

        if self.record_trace:
            result.traces.append(StepTrace(
                node_id=node.id, from_layer=node.layer, to_layer=next_layer,
                radius=chosen_r, previous=node.influence_area,
                expanded=expanded, collision=trace_collision, reach=trace_reach,
                result=chosen_area if chosen_area is not None else g2.EMPTY,
                status="OK" if chosen_area is not None else "UNREACHABLE",
            ))

        if chosen_area is None:
            node.status = NodeStatus.UNREACHABLE
            node.status_reason = (
                f"no valid area at layer {next_layer} for radii "
                f"{[round(r, 3) for r in ladder]}"
            )
            return None

        pref = g2.dilate(node.preferred_area, cfg.move_preferred,
                         quad_segs=cfg.quad_segs)
        pref = g2.intersection(pref, chosen_area)
        if pref.is_empty:
            pref = chosen_area

        child = graph.new_node(
            layer=next_layer,
            radius=chosen_r,
            influence_area=chosen_area,
            preferred_area=pref,
            # 伝播では変化しないのでリストを共有する (O(N^2) メモリを避ける)
            contact_ids=node.contact_ids,
            next_position=node.next_position,
            dtt=node.dtt + 1,
            source="propagate",
            status=NodeStatus.ACTIVE,
        )
        graph.link(child.id, node.id)   # child が下 (parent 役)、node が上
        return child

    # ------------------------------------------------------------------
    def trim_bottom_up(self, graph: TreeGraph) -> int:
        """STEP 11: root -> tip 方向から influence area を再制約する。

        PrusaSlicer の ``trim_influence_areas_bottom_up`` と同じ発想::

            child.influence_area &= dilate(parent.influence_area, move_max)

        これにより、下側の自由度が上側へ伝わり、不自然な蛇行が減る。
        領域が空になる制約は適用しない (到達性を壊さないため)。
        """
        cfg = self.config
        changed = 0
        for layer in graph.layers():
            for node in graph.nodes_at(layer):
                if not node.parent_ids:
                    continue
                allowed_parts = []
                for pid in node.parent_ids:
                    parent = graph.nodes.get(pid)
                    if parent is None or parent.influence_area.is_empty:
                        continue
                    allowed_parts.append(
                        g2.dilate(parent.influence_area, cfg.move_max,
                                  quad_segs=cfg.quad_segs)
                    )
                if not allowed_parts:
                    continue
                allowed = g2.union(allowed_parts)
                trimmed = g2.intersection(node.influence_area, allowed)
                if trimmed.is_empty:
                    continue
                if trimmed.area < node.influence_area.area - 1e-12:
                    node.influence_area = trimmed
                    pref = g2.intersection(node.preferred_area, trimmed)
                    node.preferred_area = pref if not pref.is_empty else trimmed
                    changed += 1
        return changed

    # ------------------------------------------------------------------
    def write_step_debug(self, stack: LayerStack, result: PropagationResult,
                         debug: DebugWriter, every: int = 1) -> int:
        """STEP 6A/6B: 1 ステップごとの中間状態を SVG で出す。

        model / collision / previous influence / expanded influence /
        result influence が識別できるように重ね描きする。
        """
        if not debug.enabled("influence") or not result.traces:
            return 0
        written = 0
        for idx, tr in enumerate(result.traces):
            if idx % max(1, every) != 0:
                continue
            layers = [
                SvgLayer("model", stack[tr.to_layer], fill="#475569",
                         stroke="#0f172a", opacity=0.85),
                SvgLayer("collision", tr.collision, fill="#fecaca",
                         stroke="#dc2626", opacity=0.45),
                SvgLayer("expanded", tr.expanded, fill="#fde68a",
                         stroke="#d97706", opacity=0.45),
                SvgLayer("previous", tr.previous, fill="#c7d2fe",
                         stroke="#4338ca", opacity=0.55),
                SvgLayer("result", tr.result, fill="#34d399",
                         stroke="#047857", opacity=0.75),
            ]
            if self.use_reach and not tr.reach.is_empty:
                layers.insert(1, SvgLayer("reach", tr.reach, fill="#bbf7d0",
                                          stroke="#16a34a", opacity=0.25))
            debug.write_svg(
                "influence",
                f"step_{idx:05d}_node{tr.node_id}_L{tr.from_layer:04d}"
                f"_to_L{tr.to_layer:04d}.svg",
                layers,
                bounds=self.cache.world.bounds,
                title=(f"node {tr.node_id}: layer {tr.from_layer} -> "
                       f"{tr.to_layer}  r={tr.radius:.3f}  {tr.status}  "
                       f"prev={g2.area(tr.previous):.3f} -> "
                       f"result={g2.area(tr.result):.3f} mm^2"),
            )
            written += 1
        debug.write_json("influence", "propagation_stats.json", result.stats())
        return written

    # ------------------------------------------------------------------
    def write_debug(self, stack: LayerStack, graph: TreeGraph,
                    debug: DebugWriter, every: int = 1) -> int:
        if not debug.enabled("influence"):
            return 0
        written = 0
        layers = graph.layers()
        for j in layers[::max(1, every)]:
            nodes = graph.nodes_at(j)
            if not nodes:
                continue
            debug.write_svg(
                "influence", f"layer_{j:04d}_influence.svg",
                [
                    SvgLayer("model", stack[j], fill="#64748b", stroke="#1e293b",
                             opacity=0.8),
                    SvgLayer("influence", [n.influence_area for n in nodes],
                             fill="#38bdf8", stroke="#0284c7", opacity=0.4),
                    SvgLayer("preferred", [n.preferred_area for n in nodes],
                             fill="#a78bfa", stroke="#7c3aed", opacity=0.35),
                    SvgLayer("targets", None,
                             points=[n.next_position for n in nodes],
                             point_radius=0.15, point_fill="#f59e0b"),
                ],
                bounds=self.cache.world.bounds,
                title=(f"layer {j:04d} nodes={len(nodes)} "
                       f"r=[{min(n.radius for n in nodes):.2f},"
                       f"{max(n.radius for n in nodes):.2f}]"),
            )
            written += 1
        debug.write_json("merge", "merge_events.json", {
            "stats": self.merger.stats(),
            "events": [e.to_dict() for e in self.merger.events],
        })
        return written
