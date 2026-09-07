"""STEP 6B: reachability 制約。

    candidate = candidate & reach[layer, radius]

influence area を「上の tip から到達可能」かつ
「そこから build plate へ到達可能」な領域だけに絞る。

reach は「降下中も半径一定」という近似なので必要条件ではあるが十分条件ではない。
そのため伝播中に半径が変わる層では、**その層の実半径で collision と reach を
取り直す**。半径増加で経路が消えた場合の扱いは ``radius_growth_fallback``
で明示的に選ぶ (隠して成功扱いにしない)。
"""

from __future__ import annotations

import pytest
import shapely
from shapely.geometry import Point

from test_step06a_single_tip import chain_from_tip, single_tip_run
from treesupport import geometry2d as g2
from treesupport import testmodels
from treesupport.debug_io import DebugWriter
from treesupport.graph import NodeStatus


# ======================================================================
# property: reach ∩ influence == influence
# ======================================================================
@pytest.mark.parametrize("name,tip_xy,tip_layer", [
    ("cantilever", (20.0, 5.0), 79),
    ("obstructed_path", (5.0, 5.0), 99),
    ("narrow_slot", (0.0, 5.0), 69),
    ("bridge", (16.0, 4.0), 69),
])
def test_influence_is_always_a_subset_of_reach(name, tip_xy, tip_layer):
    mesh = testmodels.ALL_MODELS[name]()
    cfg, stack, cache, prop, res = single_tip_run(
        mesh, tip_xy, tip_layer=tip_layer, use_reach=True, layer_height=0.2)

    checked = 0
    for node in res.graph.nodes.values():
        if node.source == "tip":
            allowed = cache.reach_tip(node.layer, node.radius)
        else:
            allowed = cache.reach(node.layer, node.radius)
        outside = g2.area(g2.difference(node.influence_area, allowed))
        assert outside < 1e-9, (
            f"{name} node {node.id} (layer {node.layer}): {outside:.6g} mm^2 "
            "of the influence area lies outside reach"
        )
        checked += 1
    assert checked > 0


# ======================================================================
# reach があると行き止まりに入らない
# ======================================================================
def test_reach_removes_dead_ends_on_obstructed_path():
    """tip 直下は到達不能だが横へ寄れば到達可能なケース。"""
    mesh = testmodels.obstructed_path()
    tip_layer = int(20.0 / 0.2) - 1

    without = single_tip_run(mesh, (5.0, 5.0), tip_layer=tip_layer,
                             use_reach=False, layer_height=0.2)[4]
    with_reach = single_tip_run(mesh, (5.0, 5.0), tip_layer=tip_layer,
                                use_reach=True, layer_height=0.2)[4]

    # reach 無し: 障害物の上に乗り上げたまま止まる枝が残る
    chain_without = chain_from_tip(without.graph)
    chain_with = chain_from_tip(with_reach.graph)

    assert chain_with[-1].layer == 0
    assert chain_with[-1].status is NodeStatus.ON_BUILD_PLATE
    assert with_reach.dead_ends == []
    # reach 無しでも最終的にはプレートへ届くが、途中の領域は無効部分を含む
    assert len(chain_without) >= len(chain_with)


#: 障害物ブロック中央付近 (端から十分離れた領域)
_DEEP = shapely.box(0.0, 2.0, 8.0, 8.0)


def test_reach_excludes_the_area_above_the_block():
    """障害物の真上は reach から除かれ、influence が横へ誘導されること。

    ブロック上面そのもの (z=12 直上) は下側 Z ギャップの collision でも消えるので、
    reach の効果を見るには**もっと上のレイヤ**を見る必要がある。
    z=13.2 (layer 66) では collision は効かないが、そこから横へ逃げる余裕が
    無いため reach だけが領域を消す。
    """
    mesh = testmodels.obstructed_path()
    tip_layer = int(20.0 / 0.2) - 1

    with_reach = single_tip_run(mesh, (5.0, 5.0), tip_layer=tip_layer,
                                use_reach=True, layer_height=0.2)[4]
    without = single_tip_run(mesh, (5.0, 5.0), tip_layer=tip_layer,
                             use_reach=False, layer_height=0.2)[4]

    j = int(13.2 / 0.2)
    node_with = [n for n in with_reach.graph.nodes.values() if n.layer == j][0]
    node_without = [n for n in without.graph.nodes.values() if n.layer == j][0]

    # このレイヤに collision は無い (reach だけが効いている)
    cache = single_tip_run(mesh, (5.0, 5.0), tip_layer=tip_layer,
                           use_reach=True, layer_height=0.2)[2]
    assert g2.area(g2.intersection(cache.collision(j, node_with.radius),
                                   _DEEP)) < 1e-9

    block = shapely.box(-2.0, 0.0, 12.0, 10.0)
    assert g2.area(g2.intersection(node_with.influence_area, _DEEP)) < 1e-9
    assert g2.area(g2.intersection(node_without.influence_area, _DEEP)) > 10.0
    # reach 有りではブロックの足元領域から完全に押し出されている。
    # 逃げる方向は X 側とは限らない (この形状ではブロックが Y 方向にも
    # 有限なので Y 側へ回り込むのが最短)。
    assert g2.area(g2.intersection(node_with.influence_area, block)) < 1e-9
    assert g2.area(node_with.influence_area) > 0.0


def test_reach_guides_the_branch_sideways_from_high_up():
    """降りきれない高さより下では、ブロック中央部が常に除かれていること。"""
    mesh = testmodels.obstructed_path()
    tip_layer = int(20.0 / 0.2) - 1
    res = single_tip_run(mesh, (5.0, 5.0), tip_layer=tip_layer,
                         use_reach=True, layer_height=0.2)[4]
    checked = 0
    for node in chain_from_tip(res.graph):
        if node.layer > int(16.0 / 0.2):
            continue      # まだ横へ逃げる余裕がある高さ
        assert g2.area(g2.intersection(node.influence_area, _DEEP)) < 1e-9, (
            f"layer {node.layer} still covers the middle of the block")
        checked += 1
    assert checked > 20


# ======================================================================
# 半径変化時の再取得
# ======================================================================
def test_collision_and_reach_are_refetched_with_the_actual_radius():
    """各ノードの領域が「そのノードの実半径」の reach に収まること。"""
    mesh = testmodels.cantilever()
    cfg, stack, cache, prop, res = single_tip_run(
        mesh, (20.0, 5.0), tip_layer=79, use_reach=True, layer_height=0.2,
        branch_diameter_growth=0.6)
    radii = sorted({round(n.radius, 3) for n in res.graph.nodes.values()})
    assert len(radii) > 3, "radius should actually change while descending"

    for node in res.graph.nodes.values():
        if node.source == "tip":
            continue
        allowed = cache.reach(node.layer, node.radius)
        assert g2.area(g2.difference(node.influence_area, allowed)) < 1e-9


def test_radius_growth_fallback_is_counted_not_hidden():
    """狭い通路では半径増加が抑制され、その回数が記録されること。"""
    # depth=40 にして「Y 方向へ回り込む」経路を物理的に不可能にする
    # (tip の高さ 13.8mm から使える水平移動量は 13.8*tan40 = 11.6mm < 20mm)
    mesh = testmodels.narrow_slot(gap=3.0, depth=40.0)
    cfg, stack, cache, prop, res = single_tip_run(
        mesh, (0.0, 20.0), tip_layer=69, use_reach=True, layer_height=0.2,
        xy_gap=0.4, tip_diameter=0.8, branch_diameter_growth=0.4,
        root_flare_height=0.0, radius_growth_fallback=True)
    chain = chain_from_tip(res.graph)
    assert chain[-1].layer == 0
    assert res.radius_growth_blocked > 0
    # 実半径は通路を通れる範囲に収まっている (細さを隠していない)
    inside = [n for n in chain if 5 <= n.layer <= 55]
    assert max(n.radius for n in inside) <= 1.5 - 0.4 + 1e-6


def test_strict_mode_marks_unreachable_instead_of_silently_thinning():
    """radius_growth_fallback=False では、太れない時点で UNREACHABLE。"""
    mesh = testmodels.narrow_slot(gap=3.0, depth=40.0)
    cfg, stack, cache, prop, res = single_tip_run(
        mesh, (0.0, 20.0), tip_layer=69, use_reach=True, layer_height=0.2,
        xy_gap=0.4, tip_diameter=0.8, branch_diameter_growth=0.4,
        root_flare_height=0.0, radius_growth_fallback=False)
    chain = chain_from_tip(res.graph)
    assert chain[-1].status is NodeStatus.UNREACHABLE
    assert chain[-1].layer > 0
    assert res.stats()["unreachable_tip_count"] == 1
    assert res.radius_growth_blocked == 0     # 妥協していない


def test_unreachable_subtree_is_pruned_not_reported_as_success():
    mesh = testmodels.cavity(size=16.0, wall=3.0)
    cfg = dict(layer_height=0.2, use_reach=True)
    _, _, _, _, res = single_tip_run(mesh, (8.0, 8.0),
                                     tip_layer=int(13.0 / 0.2) - 1, **cfg)
    stats = res.stats()
    assert stats["reachable_tip_count"] == 0
    assert stats["unreachable_tip_count"] == 1
    assert res.graph.root_ids == []


# ======================================================================
# Debug artifacts
# ======================================================================
def test_reach_debug_svgs(tmp_path):
    mesh = testmodels.obstructed_path()
    cfg, stack, cache, prop, res = single_tip_run(
        mesh, (5.0, 5.0), tip_layer=int(20.0 / 0.4) - 1, use_reach=True,
        layer_height=0.4, record_trace=True)
    dbg = DebugWriter(tmp_path)
    n = prop.write_step_debug(stack, res, dbg, every=1)
    assert n > 0
    files = sorted((tmp_path / "influence").glob("step_*.svg"))
    text = files[len(files) // 2].read_text(encoding="utf-8")
    assert 'id="reach"' in text
    assert 'id="result"' in text
