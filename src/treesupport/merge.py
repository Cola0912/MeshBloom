"""STEP 8: Branch Merge。

同一レイヤの 2 ノード A, B について ``intersection(A.influence_area, B.influence_area)``
が有意な面積を持てば merge **候補** とする。候補であることと merge できることは別。

判定手順
--------
1. ``STRtree`` (R-tree) で AABB が重なるペアだけに絞る (総当たり O(N^2) を避ける)。
2. 精密に交差面積を求め、``merge_min_area`` 未満なら候補から外す。
3. merge 後半径を求める::

       r_merge = sqrt(r1^2 + r2^2)

   上限を超えた場合の扱いは ``merge_radius_policy`` で選ぶ:

   * ``"clamp"`` (既定): ``branch_diameter_max/2`` で頭打ちにして merge を続ける。
     上限は「印刷可能な最大フィーチャ」であって強度要件ではないため、
     上限に達したら merge をやめる方が幹が林立して不利になる。
   * ``"forbid"``: 上限を超えるなら merge しない。

4. merge 後領域を求める::

       merge_area = A.influence & B.influence
       merge_area = merge_area - collision[layer, r_merge]
       merge_area = merge_area & reach[layer, r_merge]

   **空になったら merge 禁止**。細い半径へ妥協するフォールバックはしない。

5. 候補が複数ある場合は決定論的スコアで順序を決める::

       (-merge_area, contact_distance, r_merge, min(id), max(id))

   全候補ペアを列挙 -> スコア昇順に並べる -> 貪欲に採用、という順序なので、
   入力ノードの列挙順を変えても結果は変わらない。

単純 intersection を使う理由は docs/ARCHITECTURE.md 9 節を参照
(centerline の最大移動距離が厳密に保証され、branch_angle_max 不変条件が成立する)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import shapely
from shapely.geometry.base import BaseGeometry

from . import geometry2d as g2
from .config import SupportConfig
from .debug_io import DebugWriter, SvgLayer
from .graph import NodeStatus, SupportNode, TreeGraph
from .radius import RadiusSolver
from .volumes import CollisionCache

__all__ = ["MergeEvent", "MergeRejection", "TreeMerger"]


@dataclass
class MergeEvent:
    layer: int
    node_a: int
    node_b: int
    merged: int
    contacts_a: list[int]
    contacts_b: list[int]
    radius_a: float
    radius_b: float
    radius_merged: float
    intersect_area: float
    merge_area: float
    position: tuple[float, float]
    #: デバッグ描画用 (record_geometry=True のときだけ入る)
    geometry: dict = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        return {
            "layer": self.layer, "node_a": self.node_a, "node_b": self.node_b,
            "merged": self.merged,
            "contacts_a": list(self.contacts_a),
            "contacts_b": list(self.contacts_b),
            "radius_a": self.radius_a, "radius_b": self.radius_b,
            "radius_merged": self.radius_merged,
            "intersect_area": self.intersect_area,
            "merge_area": self.merge_area,
            "position": list(self.position),
            "accepted": True,
        }


@dataclass
class MergeRejection:
    layer: int
    node_a: int
    node_b: int
    contacts_a: list[int]
    contacts_b: list[int]
    radius_a: float
    radius_b: float
    radius_merged: float
    intersect_area: float
    reason: str
    geometry: dict = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        return {
            "layer": self.layer, "node_a": self.node_a, "node_b": self.node_b,
            "contacts_a": list(self.contacts_a),
            "contacts_b": list(self.contacts_b),
            "radius_a": self.radius_a, "radius_b": self.radius_b,
            "radius_merged": self.radius_merged,
            "intersect_area": self.intersect_area,
            "reason": self.reason,
            "accepted": False,
        }


def _position_key(a: SupportNode, b: SupportNode) -> tuple:
    """ペアを一意に順序付ける幾何キー (node id に依存しない)。"""
    pa = (round(a.next_position[0], 6), round(a.next_position[1], 6))
    pb = (round(b.next_position[0], 6), round(b.next_position[1], 6))
    return (pa, pb) if pa <= pb else (pb, pa)


class TreeMerger:
    def __init__(self, config: SupportConfig, cache: CollisionCache,
                 radius_solver: RadiusSolver | None = None,
                 record_geometry: bool = False) -> None:
        self.config = config
        self.cache = cache
        self.radius = radius_solver or RadiusSolver(config)
        self.record_geometry = record_geometry
        self.events: list[MergeEvent] = []
        self.rejections: list[MergeRejection] = []
        self.candidate_count = 0

    # ------------------------------------------------------------------
    @property
    def rejected(self) -> int:
        return len(self.rejections)

    # ------------------------------------------------------------------
    def merge_layer(self, graph: TreeGraph, layer: int,
                    node_ids: list[int], max_passes: int = 8) -> list[int]:
        """``node_ids`` (全て同じレイヤ) を可能な限り統合し、残ったノード id を返す。"""
        if not self.config.merge_enabled or len(node_ids) < 2:
            return list(node_ids)

        current = list(node_ids)
        for _ in range(max_passes):
            merged_any, current = self._merge_pass(graph, layer, current)
            if not merged_any or len(current) < 2:
                break
        return sorted(current)

    # ------------------------------------------------------------------
    def _merge_pass(self, graph: TreeGraph, layer: int,
                    node_ids: list[int]) -> tuple[bool, list[int]]:
        nodes = [graph.nodes[i] for i in sorted(node_ids) if i in graph.nodes]
        nodes = [n for n in nodes if not n.influence_area.is_empty]
        if len(nodes) < 2:
            return False, [n.id for n in nodes]

        geoms = [n.influence_area for n in nodes]
        tree = shapely.STRtree(geoms)

        # --- 候補ペアの列挙 (AABB -> 精密判定) ---
        candidates: list[tuple[tuple, int, int, BaseGeometry]] = []
        for i, node in enumerate(nodes):
            for raw in tree.query(geoms[i]):
                j = int(raw)
                if j <= i:
                    continue                    # 各ペアを 1 回だけ見る
                inter = g2.intersection(geoms[i], geoms[j])
                area = g2.area(inter)
                if area < self.config.merge_min_area:
                    continue
                self.candidate_count += 1
                other = nodes[j]
                r_merge, _ = self._merged_radius(node.radius, other.radius)
                dist = math.dist(node.next_position, other.next_position)
                # 決定論的スコア。tie-break まで**幾何量だけ**で決めるので、
                # 入力ノードの列挙順 (= node id の割り当て順) に依存しない。
                score = (
                    -round(area, 9),
                    round(dist, 9),
                    round(r_merge, 9),
                    _position_key(node, other),
                )
                candidates.append((score, i, j, inter))

        if not candidates:
            return False, [n.id for n in nodes]

        candidates.sort(key=lambda c: c[0])

        consumed: set[int] = set()
        produced: list[int] = []
        for _score, i, j, inter in candidates:
            a, b = nodes[i], nodes[j]
            if a.id in consumed or b.id in consumed:
                continue
            new_id = self._try_merge(graph, layer, a, b, inter)
            if new_id is None:
                continue
            consumed.add(a.id)
            consumed.add(b.id)
            produced.append(new_id)

        survivors = [n.id for n in nodes if n.id not in consumed] + produced
        return bool(produced), sorted(survivors)

    # ------------------------------------------------------------------
    def _merged_radius(self, r1: float, r2: float) -> tuple[float, bool]:
        """(merge 後半径, 上限を超えたか)。"""
        raw = math.hypot(r1, r2)
        limit = self.config.max_radius
        if raw <= limit:
            return raw, False
        return limit, True

    # ------------------------------------------------------------------
    def _try_merge(self, graph: TreeGraph, layer: int, a: SupportNode,
                   b: SupportNode, inter: BaseGeometry) -> int | None:
        cfg = self.config
        inter_area = g2.area(inter)
        r_merged, over_limit = self._merged_radius(a.radius, b.radius)

        def reject(reason: str, geometry: dict | None = None) -> None:
            self.rejections.append(MergeRejection(
                layer=layer, node_a=a.id, node_b=b.id,
                contacts_a=list(a.contact_ids), contacts_b=list(b.contact_ids),
                radius_a=a.radius, radius_b=b.radius, radius_merged=r_merged,
                intersect_area=inter_area, reason=reason,
                geometry=geometry or {},
            ))

        if over_limit and cfg.merge_radius_policy == "forbid":
            reject("merged radius exceeds branch_diameter_max "
                   f"({math.hypot(a.radius, b.radius):.3f} > {cfg.max_radius:.3f})")
            return None

        collision = self.cache.collision(layer, r_merged, kind="branch")
        area = g2.difference(inter, collision)
        if g2.area(area) < cfg.merge_min_area:
            reject("merged radius collides with the model",
                   {"intersection": inter, "collision": collision})
            return None

        reach = self.cache.reach(layer, r_merged)
        area = g2.intersection(area, reach)
        if g2.area(area) < cfg.merge_min_area:
            reject("merged radius cannot reach the build plate",
                   {"intersection": inter, "collision": collision, "reach": reach})
            return None

        pref = g2.intersection(a.preferred_area, b.preferred_area)
        pref = g2.intersection(pref, area)
        if pref.is_empty:
            pref = area

        # merge 位置は、より太い側の目標点に最も近い merge_area 内の点。
        # 同径なら座標順で決める (node id を使うと列挙順に依存してしまう)。
        if a.radius > b.radius:
            anchor = a.next_position
        elif b.radius > a.radius:
            anchor = b.next_position
        else:
            anchor = min(a.next_position, b.next_position)
        pos = g2.project_point(area, anchor)

        node = graph.new_node(
            layer=layer,
            radius=r_merged,
            influence_area=area,
            preferred_area=pref,
            contact_ids=sorted(set(a.contact_ids) | set(b.contact_ids)),
            next_position=pos,
            dtt=max(a.dtt, b.dtt),
            source="merge",
            status=NodeStatus.ACTIVE,
        )

        for child_id in list(a.child_ids) + list(b.child_ids):
            if child_id in graph.nodes:
                graph.link(node.id, child_id)

        geometry = {}
        if self.record_geometry:
            geometry = {"a": a.influence_area, "b": b.influence_area,
                        "intersection": inter, "collision": collision,
                        "merge_area": area}
        self.events.append(MergeEvent(
            layer=layer, node_a=a.id, node_b=b.id, merged=node.id,
            contacts_a=list(a.contact_ids), contacts_b=list(b.contact_ids),
            radius_a=a.radius, radius_b=b.radius, radius_merged=r_merged,
            intersect_area=inter_area, merge_area=g2.area(area), position=pos,
            geometry=geometry,
        ))

        a.status = NodeStatus.MERGED
        b.status = NodeStatus.MERGED
        graph.remove_node(a.id)
        graph.remove_node(b.id)
        return node.id

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "merge_count": len(self.events),
            "merge_rejected": len(self.rejections),
            "merge_candidates": self.candidate_count,
            "merge_radius_policy": self.config.merge_radius_policy,
        }

    # ------------------------------------------------------------------
    def write_debug(self, stack, debug: DebugWriter) -> int:
        """merge の採否を JSON + SVG で残す。"""
        if not debug.enabled("merge"):
            return 0
        debug.write_json("merge", "merge_events.json", {
            "stats": self.stats(),
            "accepted": [e.to_dict() for e in self.events],
            "rejected": [r.to_dict() for r in self.rejections],
        })
        written = 0
        for idx, ev in enumerate(self.events):
            if not ev.geometry:
                continue
            debug.write_svg(
                "merge",
                f"L{ev.layer:04d}_merge_{idx:04d}_"
                f"n{ev.node_a}_n{ev.node_b}_accepted.svg",
                [
                    SvgLayer("model", stack[ev.layer], fill="#475569",
                             stroke="#0f172a", opacity=0.8),
                    SvgLayer("collision", ev.geometry.get("collision"),
                             fill="#fecaca", stroke="#dc2626", opacity=0.4),
                    SvgLayer("branch_a", ev.geometry.get("a"), fill="#93c5fd",
                             stroke="#1d4ed8", opacity=0.45),
                    SvgLayer("branch_b", ev.geometry.get("b"), fill="#fdba74",
                             stroke="#c2410c", opacity=0.45),
                    SvgLayer("intersection", ev.geometry.get("intersection"),
                             fill="#a78bfa", stroke="#6d28d9", opacity=0.6),
                    SvgLayer("merge_area", ev.geometry.get("merge_area"),
                             fill="#34d399", stroke="#047857", opacity=0.8),
                    SvgLayer("position", None, points=[ev.position],
                             point_radius=0.2, point_fill="#111827"),
                ],
                bounds=self.cache.world.bounds,
                title=(f"L{ev.layer} merge n{ev.node_a}(r={ev.radius_a:.2f}) + "
                       f"n{ev.node_b}(r={ev.radius_b:.2f}) -> n{ev.merged}"
                       f"(r={ev.radius_merged:.2f})  "
                       f"contacts={sorted(set(ev.contacts_a) | set(ev.contacts_b))}"),
            )
            written += 1
        for idx, rj in enumerate(self.rejections):
            if not rj.geometry:
                continue
            debug.write_svg(
                "merge",
                f"L{rj.layer:04d}_reject_{idx:04d}_"
                f"n{rj.node_a}_n{rj.node_b}.svg",
                [
                    SvgLayer("model", stack[rj.layer], fill="#475569",
                             stroke="#0f172a", opacity=0.8),
                    SvgLayer("collision", rj.geometry.get("collision"),
                             fill="#fecaca", stroke="#dc2626", opacity=0.5),
                    SvgLayer("intersection", rj.geometry.get("intersection"),
                             fill="#a78bfa", stroke="#6d28d9", opacity=0.7),
                ],
                bounds=self.cache.world.bounds,
                title=(f"L{rj.layer} REJECTED n{rj.node_a} + n{rj.node_b} "
                       f"(r_merge={rj.radius_merged:.2f}): {rj.reason}"),
            )
            written += 1
        return written
