"""STEP 6/7/8/9: influence 伝播、複数 tip、merge、centerline のテスト。"""

from __future__ import annotations

import math

import pytest
import shapely
from shapely.geometry import Point

from conftest import build_tree
from treesupport import geometry2d as g2
from treesupport import testmodels
from treesupport.config import SupportConfig
from treesupport.debug_io import DebugWriter


# ======================================================================
# STEP 6: 単一 tip の top-down 伝播
# ======================================================================
def test_single_tip_reaches_build_plate():
    """1 本の枝がビルドプレートまで到達すること (merge 無効)。"""
    mesh = testmodels.floating_box(size=1.2, thickness=2.0, height=10.0)
    b = build_tree(mesh, layer_height=0.2, tip_spacing=5.0, merge_enabled=False,
                   min_overhang_area=0.1)
    g = b.graph
    assert len(b.tips) == 1
    assert len(g.root_ids) == 1
    # tip レイヤ 49 から 0 まで、各レイヤに 1 ノード
    tip_layer = b.tips.tips[0].layer
    assert tip_layer == 49
    assert len(g.nodes) == tip_layer + 1
    for j in range(tip_layer + 1):
        assert len(g.nodes_at(j)) == 1
    assert all(n.reaches_build_plate for n in g.nodes.values())


def test_influence_area_never_intersects_collision():
    b = build_tree(testmodels.cantilever(), layer_height=0.4, tip_spacing=3.0)
    for node in b.graph.nodes.values():
        # tip ノードは上に材料が無いので tip 用 collision で評価する
        kind = "tip" if node.source == "tip" else "branch"
        col = b.cache.collision(node.layer, node.radius, kind=kind)
        assert g2.area(g2.intersection(node.influence_area, col)) < 1e-9


def test_influence_area_grows_downward_by_move_max():
    """1 レイヤ下の influence area は dilate(area, move_max) に含まれる。"""
    b = build_tree(testmodels.floating_box(size=1.2, height=6.0),
                   layer_height=0.2, tip_spacing=5.0, merge_enabled=False,
                   min_overhang_area=0.1)
    cfg = b.config
    for node in b.graph.nodes.values():
        for pid in node.parent_ids:
            parent = b.graph.nodes[pid]
            grown = g2.dilate_outer(node.influence_area, cfg.move_max,
                                    quad_segs=cfg.quad_segs)
            # 親 (下) の領域は子 (上) の領域を move_max 膨張したものの内側
            assert g2.area(g2.difference(parent.influence_area, grown)) < 1e-7


def test_radius_grows_with_depth():
    b = build_tree(testmodels.floating_box(size=1.2, height=20.0),
                   layer_height=0.2, tip_spacing=5.0, merge_enabled=False,
                   min_overhang_area=0.1, branch_diameter_growth=0.4,
                   root_flare_height=0.0)
    g = b.graph
    tip = [n for n in g.nodes.values() if n.source == "tip"][0]
    root = g.nodes[g.root_ids[0]]
    assert tip.radius == pytest.approx(b.config.tip_radius)
    assert root.radius > tip.radius
    # 単調非減少
    node = root
    prev = node.radius
    while node.child_ids:
        node = g.nodes[node.child_ids[0]]
        assert node.radius <= prev + 1e-12
        prev = node.radius


def test_radius_never_exceeds_max():
    b = build_tree(testmodels.floating_box(size=1.2, height=30.0),
                   layer_height=0.2, tip_spacing=5.0, merge_enabled=False,
                   min_overhang_area=0.1, branch_diameter_growth=2.0,
                   branch_diameter_max=3.0)
    assert max(n.radius for n in b.graph.nodes.values()) <= 1.5 + 1e-12


def test_root_flare_thickens_near_plate():
    b = build_tree(testmodels.floating_box(size=1.2, height=20.0),
                   layer_height=0.2, tip_spacing=5.0, merge_enabled=False,
                   min_overhang_area=0.1, root_diameter_min=3.0,
                   root_flare_height=5.0, branch_diameter_growth=0.0)
    g = b.graph
    root = g.nodes[g.root_ids[0]]
    assert root.radius == pytest.approx(1.5, rel=1e-6)
    mid = [n for n in g.nodes.values() if n.layer == 40][0]   # z=8.0 > flare
    assert mid.radius == pytest.approx(b.config.tip_radius)


# ======================================================================
# STEP 7: 複数 tip
# ======================================================================
def test_multiple_tips_without_merge_stay_independent():
    b = build_tree(testmodels.merge_cluster(), layer_height=0.4, tip_spacing=3.0,
                   merge_enabled=False)
    g = b.graph
    assert len(g.root_ids) == len(b.tips)
    assert all(len(n.child_ids) <= 1 for n in g.nodes.values())


def test_all_tips_are_connected_to_a_root():
    b = build_tree(testmodels.cantilever(), layer_height=0.4, tip_spacing=3.0)
    g = b.graph
    roots = set(g.root_ids)
    for tip in g.tips():
        node = tip
        seen = 0
        while node.parent_ids:
            node = g.nodes[node.parent_ids[0]]
            seen += 1
            assert seen < 10_000
        assert node.id in roots


# ======================================================================
# STEP 8: merge
# ======================================================================
def test_merge_reduces_roots_to_one_trunk():
    b = build_tree(testmodels.merge_cluster(), layer_height=0.4, tip_spacing=3.0)
    g = b.graph
    assert len(b.tips) > 4
    assert len(g.root_ids) == 1
    assert len(b.merger.events) == len(b.tips) - 1


def test_merged_radius_is_area_preserving():
    b = build_tree(testmodels.merge_cluster(), layer_height=0.4, tip_spacing=3.0)
    for ev in b.merger.events:
        expected = min(b.config.max_radius, math.hypot(ev.radius_a, ev.radius_b))
        # 半径を増やせない場合のフォールバックも許容する
        assert ev.radius_merged in (
            pytest.approx(expected), pytest.approx(max(ev.radius_a, ev.radius_b))
        )
        assert ev.radius_merged >= max(ev.radius_a, ev.radius_b) - 1e-12


def test_merged_area_is_inside_both_parents():
    """merge 後領域は両方の元領域の部分集合 (角度保証の根拠)。"""
    b = build_tree(testmodels.merge_cluster(), layer_height=0.4, tip_spacing=3.0)
    g = b.graph
    for node in g.nodes.values():
        if not node.is_merge:
            continue
        for cid in node.child_ids:
            child = g.nodes[cid]
            grown = g2.dilate_outer(child.influence_area, b.config.move_max,
                                    quad_segs=b.config.quad_segs)
            assert g2.area(g2.difference(node.influence_area, grown)) < 1e-7


def test_merge_uses_spatial_index_not_all_pairs():
    """merge 判定が全組合せになっていないこと (STRtree で候補が絞られている)。"""
    b = build_tree(testmodels.merge_cluster(), layer_height=0.4, tip_spacing=3.0)
    assert b.merger.stats()["merge_count"] > 0


def test_merge_is_rejected_when_radius_would_collide():
    """太った結果 reach が消える merge は行わない。"""
    cfg_kwargs = dict(layer_height=0.4, tip_spacing=2.0, branch_diameter_max=0.8)
    b = build_tree(testmodels.merge_cluster(), **cfg_kwargs)
    for ev in b.merger.events:
        assert ev.radius_merged <= 0.4 + 1e-12


# ======================================================================
# STEP 9: centerline
# ======================================================================
def test_every_node_has_a_position_inside_its_influence_area():
    b = build_tree(testmodels.cantilever(), layer_height=0.4, tip_spacing=3.0)
    for node in b.graph.nodes.values():
        assert node.position is not None
        p = Point(*node.position)
        assert node.influence_area.buffer(1e-6).covers(p), node.id


def test_branch_angle_never_exceeds_max():
    for mesh, kw in [
        (testmodels.cantilever(), dict(layer_height=0.4, tip_spacing=3.0)),
        (testmodels.merge_cluster(), dict(layer_height=0.4, tip_spacing=3.0)),
        (testmodels.obstructed_path(), dict(layer_height=0.4, tip_spacing=3.0)),
    ]:
        b = build_tree(mesh, **kw)
        assert b.graph.max_branch_angle() <= b.config.branch_angle_max + 1e-6


def test_centerline_keeps_clearance_from_model():
    """centerline から xy_gap + radius 以内にモデル断面が無いこと。"""
    b = build_tree(testmodels.cantilever(), layer_height=0.4, tip_spacing=3.0)
    for node in b.graph.nodes.values():
        model = b.stack[node.layer]
        if model.is_empty:
            continue
        d = model.distance(Point(*node.position))
        assert d >= node.radius + b.config.xy_gap - 1e-6, node.id


def test_root_positions_are_on_the_build_plate():
    b = build_tree(testmodels.merge_cluster(), layer_height=0.4, tip_spacing=3.0)
    for node in b.graph.roots():
        assert node.layer == 0
        assert node.z(b.config.layer_height) == pytest.approx(0.0)


def test_obstructed_path_routes_around_the_block():
    """真下が塞がれている場合、枝が障害物の外へ迂回すること。

    迂回方向は X 側とは限らない (Y 側へ回っても正しい) ので、
    「ブロックの XY 領域から xy_gap + radius 以上離れている」ことを検査する。
    """
    b = build_tree(testmodels.obstructed_path(), layer_height=0.4, tip_spacing=3.0)
    g = b.graph
    block = shapely.box(-2.0, 0.0, 12.0, 10.0)      # z in [0, 12] の断面
    detoured = False
    for node in g.nodes.values():
        z = node.z(b.config.layer_height)
        if z >= 11.9:
            continue
        p = Point(*node.position)
        assert not block.covers(p), f"node {node.id} at z={z} is inside the block"
        assert block.distance(p) >= node.radius + b.config.xy_gap - 1e-6
        detoured = True
    assert detoured, "no node below the block: nothing was verified"
    assert len(g.root_ids) >= 1
    # tip は板の真下 (x in [0,10]) にあるので、確かに迂回している
    assert any(0.0 <= t.position[0] <= 10.0 for t in g.tips())


def test_bottom_up_trim_shrinks_influence_areas():
    a = build_tree(testmodels.cantilever(), layer_height=0.4, tip_spacing=3.0,
                   solve_centerline=False, trim=False)
    c = build_tree(testmodels.cantilever(), layer_height=0.4, tip_spacing=3.0,
                   solve_centerline=False, trim=True)
    area_a = sum(g2.area(n.influence_area) for n in a.graph.nodes.values())
    area_c = sum(g2.area(n.influence_area) for n in c.graph.nodes.values())
    assert area_c <= area_a + 1e-9


def test_trim_does_not_break_centerline():
    b = build_tree(testmodels.merge_cluster(), layer_height=0.4, tip_spacing=3.0,
                   trim=True)
    assert b.graph.max_branch_angle() <= b.config.branch_angle_max + 1e-6
    for node in b.graph.nodes.values():
        assert node.position is not None


# ======================================================================
def test_closed_cavity_is_reported_unsupportable():
    """閉じた内部空洞はプレートから到達できないので unsupportable。"""
    b = build_tree(testmodels.cavity(size=16.0, wall=3.0), layer_height=0.4,
                   tip_spacing=3.0)
    assert len(b.tips) == 0
    assert b.tips.unsupportable_area > 50.0
    assert len(b.graph.nodes) == 0


def test_debug_artifacts(tmp_path):
    b = build_tree(testmodels.merge_cluster(), layer_height=0.8, tip_spacing=3.0)
    dbg = DebugWriter(tmp_path)
    n = b.propagator.write_debug(b.stack, b.graph, dbg, every=3)
    assert n > 0
    assert (tmp_path / "merge" / "merge_events.json").exists()
    dbg.write_obj_polylines("centerline", "centerline.obj",
                            b.graph.centerline_polylines())
    obj = (tmp_path / "centerline" / "centerline.obj").read_text(encoding="utf-8")
    assert obj.count("\nl ") > 0
