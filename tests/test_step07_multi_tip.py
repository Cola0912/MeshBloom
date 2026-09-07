"""STEP 7: 複数 tip の独立伝播 (merge はまだ禁止)。

各 ContactPoint が独立した branch state を持つ。
1 tip = 1 branch であること、および一部が到達不能でも
到達可能な tip まで巻き添えで失敗扱いにしないことを検証する。
"""

from __future__ import annotations

import pytest
import shapely
from shapely.geometry import Point

from conftest import build_tree
from treesupport import geometry2d as g2
from treesupport import testmodels
from treesupport.debug_io import DebugWriter
from treesupport.graph import NodeStatus


def multi(mesh, **kwargs):
    kwargs.setdefault("merge_enabled", False)
    kwargs.setdefault("solve_centerline", False)
    return build_tree(mesh, **kwargs)


# ======================================================================
# 1. floating box: 全 tip が build plate へ到達
# ======================================================================
def test_floating_box_all_tips_reach_the_build_plate():
    b = multi(testmodels.floating_box(size=12.0, thickness=4.0, height=10.0),
              layer_height=0.4, tip_spacing=3.0)
    stats = b.propagation.stats()

    assert stats["tip_count"] > 5
    assert stats["reachable_tip_count"] == stats["tip_count"]
    assert stats["unreachable_tip_count"] == 0
    # merge していないので root 数 = tip 数
    assert stats["root_count"] == stats["tip_count"]
    assert stats["branch_count"] == stats["tip_count"]
    for node in b.graph.roots():
        assert node.layer == 0
        assert node.status is NodeStatus.ON_BUILD_PLATE


def test_one_tip_is_one_branch():
    b = multi(testmodels.floating_box(size=12.0, height=10.0),
              layer_height=0.4, tip_spacing=3.0)
    for node in b.graph.nodes.values():
        assert len(node.contact_ids) == 1
        assert len(node.child_ids) <= 1
        assert len(node.parent_ids) <= 1
    # 各 branch の contact_ids は 1 つで、全体で重複しない
    contact_sets = []
    for branch in b.graph.branches.values():
        ids = set()
        for nid in branch.node_ids:
            ids.update(b.graph.nodes[nid].contact_ids)
        assert len(ids) == 1
        contact_sets.append(ids.pop())
    assert len(set(contact_sets)) == len(contact_sets)


# ======================================================================
# 2. cantilever: 全 ContactPoint が branch を持つ
# ======================================================================
def test_every_contact_point_gets_a_branch():
    b = multi(testmodels.cantilever(), layer_height=0.4, tip_spacing=3.0)
    placed = set()
    for node in b.graph.nodes.values():
        placed.update(node.contact_ids)
    expected = {t.id for t in b.tips.tips}
    assert placed == expected, (
        f"missing branches for contacts {sorted(expected - placed)}")
    assert b.propagation.unplaced_tips == []


def test_tip_nodes_sit_at_the_contact_points():
    b = multi(testmodels.cantilever(), layer_height=0.4, tip_spacing=3.0)
    by_id = {t.id: t for t in b.tips.tips}
    tips = [n for n in b.graph.nodes.values() if n.source == "tip"]
    assert len(tips) == len(by_id)
    for node in tips:
        contact = by_id[node.contact_ids[0]]
        assert node.layer == contact.layer
        assert node.influence_area.covers(Point(contact.x, contact.y))
        assert node.next_position == pytest.approx((contact.x, contact.y))


# ======================================================================
# 3. 複数の孤立 island
# ======================================================================
def test_each_island_gets_its_own_roots():
    b = multi(testmodels.islands(count=3, spacing=20.0),
              layer_height=0.4, tip_spacing=3.0)
    roots = b.graph.roots()
    assert len(roots) >= 3

    # 各 island の真下付近に少なくとも 1 本の root がある
    for i in range(3):
        x0 = i * 20.0
        band = [r for r in roots
                if x0 - 8.0 <= r.influence_area.centroid.x <= x0 + 14.0]
        assert band, f"island {i} has no root beneath it"

    stats = b.propagation.stats()
    assert stats["unreachable_tip_count"] == 0
    assert stats["reachable_tip_count"] == stats["tip_count"]


def test_islands_do_not_share_branches():
    b = multi(testmodels.islands(count=3, spacing=20.0),
              layer_height=0.4, tip_spacing=3.0)
    # merge 無効なので、どのノードも複数 island の contact を持たない
    for node in b.graph.nodes.values():
        assert len(node.contact_ids) == 1


# ======================================================================
# 4. 一部だけ到達不能
# ======================================================================
def test_partial_unreachability_does_not_fail_the_reachable_tips():
    b = multi(testmodels.mixed_reachability(), layer_height=0.4,
              tip_spacing=3.0)
    stats = b.propagation.stats()

    # 浮遊板側は到達できる
    assert stats["reachable_tip_count"] > 0
    assert stats["root_count"] > 0
    # 閉じた空洞側は tip 自体が置けない (unsupportable として記録される)
    assert b.tips.unsupportable_area > 10.0
    # 到達できた tip の branch は健全
    for node in b.graph.roots():
        assert node.layer == 0
    for node in b.graph.nodes.values():
        assert node.status in (NodeStatus.ACTIVE, NodeStatus.ON_BUILD_PLATE)


def test_reachable_and_unreachable_are_recorded_separately():
    """明示的に個別記録されること。"""
    b = multi(testmodels.mixed_reachability(), layer_height=0.4,
              tip_spacing=3.0)
    res = b.propagation
    assert set(res.tip_reachable) == {t.id for t in b.tips.tips}
    assert set(res.reachable_tips) | set(res.unreachable_tips) == set(res.tip_reachable)
    assert not (set(res.reachable_tips) & set(res.unreachable_tips))


def test_unreachable_branch_is_pruned_but_others_survive():
    """到達不能な枝を刈っても、他の枝のノードは残ること。"""
    mesh = testmodels.mixed_reachability()
    b = multi(mesh, layer_height=0.4, tip_spacing=3.0)
    assert len(b.graph.nodes) > 0
    assert all(n.reaches_build_plate for n in b.graph.nodes.values())


# ======================================================================
# Metrics
# ======================================================================
def test_required_metrics_are_reported():
    b = multi(testmodels.cantilever(), layer_height=0.4, tip_spacing=3.0)
    stats = b.propagation.stats()
    for key in ("tip_count", "reachable_tip_count", "unreachable_tip_count",
                "branch_count", "total_influence_area"):
        assert key in stats, key
    assert stats["tip_count"] == len(b.tips.tips)
    assert stats["total_influence_area"] > 0.0
    assert (stats["reachable_tip_count"] + stats["unreachable_tip_count"]
            == stats["tip_count"])


# ======================================================================
# 独立性 (相互干渉しないこと)
# ======================================================================
def test_branches_are_independent_of_tip_enumeration_order():
    """tip の列挙順を変えても、生成される枝の集合が変わらないこと。"""
    mesh = testmodels.islands(count=3, spacing=20.0)
    a = multi(mesh, layer_height=0.4, tip_spacing=3.0)

    # tip の順序を逆にして伝播し直す
    from treesupport.propagate import TreePropagator
    from treesupport.tips import TipSet
    reversed_set = TipSet(
        tips=list(reversed(a.tips.tips)),
        by_layer={k: list(reversed(v)) for k, v in a.tips.by_layer.items()},
    )
    prop = TreePropagator(a.config, a.cache)
    res = prop.propagate(a.stack, reversed_set)

    def signature(result):
        out = []
        for node in result.graph.nodes.values():
            out.append((node.layer, tuple(node.contact_ids),
                        round(node.radius, 6),
                        round(g2.area(node.influence_area), 6)))
        return sorted(out)

    assert signature(res) == signature(a.propagation)


def test_no_node_area_leaks_into_another_branch_collision():
    b = multi(testmodels.islands(count=3, spacing=20.0),
              layer_height=0.4, tip_spacing=3.0)
    for node in b.graph.nodes.values():
        kind = "tip" if node.source == "tip" else "branch"
        col = b.cache.collision(node.layer, node.radius, kind=kind)
        assert g2.area(g2.intersection(node.influence_area, col)) < 1e-9


# ======================================================================
# Debug artifacts
# ======================================================================
def test_multi_tip_debug_artifacts(tmp_path):
    b = multi(testmodels.islands(count=3, spacing=20.0),
              layer_height=0.8, tip_spacing=3.0)
    dbg = DebugWriter(tmp_path)
    n = b.propagator.write_debug(b.stack, b.graph, dbg, every=2)
    assert n > 0
    dbg.write_json("influence", "propagation_stats.json",
                   b.propagation.stats())
    files = sorted((tmp_path / "influence").glob("layer_*_influence.svg"))
    assert files
    assert (tmp_path / "influence" / "propagation_stats.json").exists()
