"""ツリーグラフのデータ構造。

用語 (重要)
-----------
本モジュールでは **root = ビルドプレート側 (下)**、**tip = 上** とする::

    parent  = 1 レイヤ**下** (root に近い)
    child   = 1 レイヤ**上** (tip に近い)

伝播は上から下へ (tip -> root) 進むので、生成順は「子が先、親が後」になる。
merge は「2 つの子が 1 つの親を共有する」形で表現される。

(CuraEngine は逆向きの命名 ``parents_`` = 上位 を使うが、
 指示書 5 節の ``Tree { branches[], root }`` に合わせて本ツールでは
 一般的なツリー慣習を採る。)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from shapely.geometry.base import BaseGeometry

from . import geometry2d as g2

__all__ = ["NodeStatus", "SupportNode", "Branch", "TreeGraph"]


class NodeStatus(str, Enum):
    """ノードの状態。失敗を黙って成功扱いにしないための明示的なマーカ。"""

    #: 伝播途中 (まだ下層へ続く)
    ACTIVE = "ACTIVE"
    #: ビルドプレートに到達済み (レイヤ 0)
    ON_BUILD_PLATE = "ON_BUILD_PLATE"
    #: これ以上下へ伝播できない (collision / reach で領域が消えた)
    UNREACHABLE = "UNREACHABLE"
    #: merge によって別ノードへ置き換えられた
    MERGED = "MERGED"
    #: 到達不能な部分木として除去された
    PRUNED = "PRUNED"


@dataclass
class SupportNode:
    """1 レイヤ・1 枝断面に対応するノード。"""

    id: int
    layer: int
    radius: float

    #: このレイヤで枝中心が存在可能な領域
    influence_area: BaseGeometry = g2.EMPTY
    #: preferred angle だけで到達可能な領域 (centerline の cost 用、hard ではない)
    preferred_area: BaseGeometry = g2.EMPTY

    #: 1 レイヤ下のノード (root 側)。merge しない限り高々 1 個。
    parent_ids: list[int] = field(default_factory=list)
    #: 1 レイヤ上のノード (tip 側)。merge した親は複数持つ。
    child_ids: list[int] = field(default_factory=list)

    #: このノードが支える tip (ContactPoint) の id
    contact_ids: list[int] = field(default_factory=list)

    reaches_build_plate: bool = False
    locked: bool = False
    status: NodeStatus = NodeStatus.ACTIVE
    #: UNREACHABLE になった理由 (デバッグ用)
    status_reason: str = ""

    #: centerline 決定後の XY (未決定なら None)
    position: tuple[float, float] | None = None
    #: 目標点 (直近の merge 位置、なければ由来 tip の XY)
    next_position: tuple[float, float] = (0.0, 0.0)

    #: distance to top (レイヤ数)
    dtt: int = 0
    #: 由来 ("tip" | "propagate" | "merge")
    source: str = "propagate"

    @property
    def is_tip(self) -> bool:
        return not self.child_ids

    @property
    def is_merge(self) -> bool:
        return len(self.child_ids) > 1

    def z(self, layer_height: float) -> float:
        return self.layer * layer_height

    def to_dict(self, layer_height: float | None = None) -> dict:
        d = {
            "id": self.id,
            "layer": self.layer,
            "radius": self.radius,
            "parent_ids": list(self.parent_ids),
            "child_ids": list(self.child_ids),
            "contact_ids": list(self.contact_ids),
            "reaches_build_plate": self.reaches_build_plate,
            "locked": self.locked,
            "status": self.status.value,
            "status_reason": self.status_reason,
            "position": list(self.position) if self.position else None,
            "next_position": list(self.next_position),
            "dtt": self.dtt,
            "source": self.source,
            "influence_area": g2.area(self.influence_area),
        }
        if layer_height is not None:
            d["z"] = self.z(layer_height)
        return d


@dataclass
class Branch:
    """分岐点の間の一続きのノード列。``nodes`` は下 (root 側) から上 (tip 側) 順。"""

    id: int
    node_ids: list[int] = field(default_factory=list)
    parent_branch: int | None = None
    child_branches: list[int] = field(default_factory=list)

    @property
    def bottom(self) -> int:
        return self.node_ids[0]

    @property
    def top(self) -> int:
        return self.node_ids[-1]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "node_ids": list(self.node_ids),
            "parent_branch": self.parent_branch,
            "child_branches": list(self.child_branches),
        }


class TreeGraph:
    """ノード集合と枝分解。"""

    def __init__(self, layer_height: float) -> None:
        self.layer_height = layer_height
        self.nodes: dict[int, SupportNode] = {}
        self.by_layer: dict[int, list[int]] = {}
        self.root_ids: list[int] = []
        self.branches: dict[int, Branch] = {}
        self._next_id = 0

    # ------------------------------------------------------------------
    def new_node(self, layer: int, radius: float, **kwargs) -> SupportNode:
        node = SupportNode(id=self._next_id, layer=layer, radius=radius, **kwargs)
        self._next_id += 1
        self.nodes[node.id] = node
        self.by_layer.setdefault(layer, []).append(node.id)
        return node

    def remove_node(self, node_id: int) -> None:
        node = self.nodes.pop(node_id, None)
        if node is None:
            return
        ids = self.by_layer.get(node.layer)
        if ids and node_id in ids:
            ids.remove(node_id)
        for cid in node.child_ids:
            child = self.nodes.get(cid)
            if child and node_id in child.parent_ids:
                child.parent_ids.remove(node_id)
        for pid in node.parent_ids:
            parent = self.nodes.get(pid)
            if parent and node_id in parent.child_ids:
                parent.child_ids.remove(node_id)
        if node_id in self.root_ids:
            self.root_ids.remove(node_id)

    def link(self, parent_id: int, child_id: int) -> None:
        """``parent`` (下) と ``child`` (上) を接続する。"""
        parent = self.nodes[parent_id]
        child = self.nodes[child_id]
        if child_id not in parent.child_ids:
            parent.child_ids.append(child_id)
        if parent_id not in child.parent_ids:
            child.parent_ids.append(parent_id)

    # ------------------------------------------------------------------
    def tips(self) -> list[SupportNode]:
        return [n for n in self.nodes.values() if n.is_tip]

    def roots(self) -> list[SupportNode]:
        return [self.nodes[i] for i in self.root_ids if i in self.nodes]

    def layers(self) -> list[int]:
        return sorted(k for k, v in self.by_layer.items() if v)

    def nodes_at(self, layer: int) -> list[SupportNode]:
        return [self.nodes[i] for i in self.by_layer.get(layer, []) if i in self.nodes]

    def ascending(self):
        """下から上へノードを走査する。"""
        for layer in self.layers():
            for node in self.nodes_at(layer):
                yield node

    def descending(self):
        """上から下へノードを走査する。"""
        for layer in reversed(self.layers()):
            for node in self.nodes_at(layer):
                yield node

    # ------------------------------------------------------------------
    def prune_unreachable(self) -> int:
        """ビルドプレートへ到達しない部分木を取り除く。

        戻り値は削除したノード数。
        """
        reachable: set[int] = set()
        stack = [i for i in self.root_ids if i in self.nodes]
        while stack:
            nid = stack.pop()
            if nid in reachable:
                continue
            reachable.add(nid)
            stack.extend(self.nodes[nid].child_ids)

        dead = [i for i in self.nodes if i not in reachable]
        for nid in dead:
            self.nodes[nid].status = NodeStatus.PRUNED
            self.remove_node(nid)
        for nid in reachable:
            node = self.nodes[nid]
            node.reaches_build_plate = True
            if node.layer == 0:
                node.status = NodeStatus.ON_BUILD_PLATE
        return len(dead)

    # ------------------------------------------------------------------
    def build_branches(self) -> dict[int, Branch]:
        """分岐点で区切って枝 (Branch) に分解する。

        枝の切れ目は「親が複数の子を持つ (bifurcation)」場所。
        各 Branch の ``node_ids`` は下から上の順。
        """
        self.branches = {}
        next_bid = 0
        # node_id -> branch_id
        assign: dict[int, int] = {}

        for root in sorted(self.root_ids):
            if root not in self.nodes:
                continue
            stack: list[tuple[int, int | None]] = [(root, None)]
            while stack:
                start, parent_branch = stack.pop()
                branch = Branch(id=next_bid, parent_branch=parent_branch)
                next_bid += 1
                self.branches[branch.id] = branch
                if parent_branch is not None:
                    self.branches[parent_branch].child_branches.append(branch.id)

                cur = start
                while True:
                    branch.node_ids.append(cur)
                    assign[cur] = branch.id
                    children = [c for c in self.nodes[cur].child_ids if c in self.nodes]
                    if len(children) == 1:
                        cur = children[0]
                        continue
                    if len(children) > 1:
                        for c in children:
                            stack.append((c, branch.id))
                    break
        return self.branches

    # ------------------------------------------------------------------
    def centerline_length(self) -> float:
        """決定済み centerline の総長 [mm]。"""
        total = 0.0
        lh = self.layer_height
        for node in self.nodes.values():
            if node.position is None:
                continue
            for cid in node.child_ids:
                child = self.nodes.get(cid)
                if child is None or child.position is None:
                    continue
                dx = child.position[0] - node.position[0]
                dy = child.position[1] - node.position[1]
                dz = (child.layer - node.layer) * lh
                total += (dx * dx + dy * dy + dz * dz) ** 0.5
        return total

    def max_branch_angle(self) -> float:
        """centerline の最大傾斜角 [deg] (鉛直から)。"""
        import math

        lh = self.layer_height
        worst = 0.0
        for node in self.nodes.values():
            if node.position is None:
                continue
            for cid in node.child_ids:
                child = self.nodes.get(cid)
                if child is None or child.position is None:
                    continue
                dz = abs(child.layer - node.layer) * lh
                dxy = math.hypot(child.position[0] - node.position[0],
                                 child.position[1] - node.position[1])
                if dz <= 0.0:
                    if dxy > 0.0:
                        worst = 90.0
                    continue
                worst = max(worst, math.degrees(math.atan2(dxy, dz)))
        return worst

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "node_count": len(self.nodes),
            "tip_count": len(self.tips()),
            "root_count": len(self.root_ids),
            "branch_count": len(self.branches),
            "merge_count": sum(1 for n in self.nodes.values() if n.is_merge),
            "layers": len(self.layers()),
            "min_radius": min((n.radius for n in self.nodes.values()), default=0.0),
            "max_radius": max((n.radius for n in self.nodes.values()), default=0.0),
        }

    def to_dict(self, include_nodes: bool = True) -> dict:
        d = {"layer_height": self.layer_height, "stats": self.stats(),
             "root_ids": list(self.root_ids)}
        if include_nodes:
            d["nodes"] = [n.to_dict(self.layer_height)
                          for n in sorted(self.nodes.values(), key=lambda n: n.id)]
            d["branches"] = [b.to_dict() for b in self.branches.values()]
        return d

    def centerline_polylines(self) -> list[list[tuple[float, float, float]]]:
        """デバッグ出力用に centerline を折れ線群として返す。"""
        lh = self.layer_height
        out: list[list[tuple[float, float, float]]] = []
        for node in self.nodes.values():
            if node.position is None:
                continue
            for cid in node.child_ids:
                child = self.nodes.get(cid)
                if child is None or child.position is None:
                    continue
                out.append([
                    (node.position[0], node.position[1], node.layer * lh),
                    (child.position[0], child.position[1], child.layer * lh),
                ])
        return out
