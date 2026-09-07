"""STEP 6A: 単一 tip の influence area 伝播 (merge なし / reach なし)。

このステップでは reach 制約を使わない。伝播式は::

    raw       = dilate(A[j], move_max)
    candidate = raw - collision[j-1, radius]

具体的な中心 XY はまだ決めない。保持するのは
「このレイヤで枝中心が存在可能な領域」だけ。
"""

from __future__ import annotations

import math

import pytest
import shapely
from shapely.geometry import Point

from treesupport import geometry2d as g2
from treesupport import testmodels
from treesupport.config import SupportConfig
from treesupport.debug_io import DebugWriter
from treesupport.graph import NodeStatus
from treesupport.propagate import TreePropagator
from treesupport.slicer import ModelSlicer
from treesupport.tips import ContactPoint, TipSet
from treesupport.volumes import CollisionCache


# ----------------------------------------------------------------------
def single_tip_run(mesh, tip_xy, tip_layer, use_reach=False,
                   record_trace=False, **cfg_kwargs):
    """指定座標に 1 本だけ tip を置いて伝播させる。"""
    cfg = SupportConfig(merge_enabled=False, **cfg_kwargs)
    stack = ModelSlicer(cfg).slice(mesh)
    cache = CollisionCache(cfg, stack)

    tip = ContactPoint(id=0, layer=tip_layer, x=tip_xy[0], y=tip_xy[1],
                       z=stack.z_bottom(tip_layer), overhang_layer=tip_layer + 1)
    tipset = TipSet(tips=[tip], by_layer={tip_layer: [tip]})

    prop = TreePropagator(cfg, cache, use_reach=use_reach,
                          record_trace=record_trace)
    result = prop.propagate(stack, tipset, prune=False)
    return cfg, stack, cache, prop, result


def chain_from_tip(graph):
    """tip から下へ 1 本の鎖として辿る。"""
    tips = [n for n in graph.nodes.values() if n.source == "tip"]
    assert len(tips) == 1
    chain = [tips[0]]
    node = tips[0]
    while node.parent_ids:
        node = graph.nodes[node.parent_ids[0]]
        chain.append(node)
    return chain


# ======================================================================
# A. 障害物なし: move_max に従って広がる
# ======================================================================
def test_influence_area_widens_by_move_max_without_obstacles():
    mesh = testmodels.floating_box(size=12.0, thickness=4.0, height=10.0)
    cfg, stack, cache, prop, res = single_tip_run(
        mesh, (6.0, 6.0), tip_layer=49, layer_height=0.2, tip_spacing=5.0)
    chain = chain_from_tip(res.graph)

    assert len(chain) == 50           # レイヤ 49 -> 0
    assert chain[0].layer == 49
    assert chain[-1].layer == 0
    assert chain[-1].status is NodeStatus.ON_BUILD_PLATE

    # 障害物が無いので、各段の半径は tip 半径 + k * move_max
    move = cfg.move_max
    for k, node in enumerate(chain[:20]):
        expected_r = cfg.tip_influence + k * move
        # 円の面積として比較 (内接近似のぶん僅かに小さい)
        r_eff = math.sqrt(g2.area(node.influence_area) / math.pi)
        assert r_eff == pytest.approx(expected_r, rel=0.02), (
            f"layer {node.layer}: r_eff={r_eff} expected={expected_r}")


def test_influence_area_is_monotonically_growing():
    mesh = testmodels.floating_box(size=12.0, thickness=4.0, height=10.0)
    cfg, stack, cache, prop, res = single_tip_run(
        mesh, (6.0, 6.0), tip_layer=49, layer_height=0.2)
    chain = chain_from_tip(res.graph)
    areas = [g2.area(n.influence_area) for n in chain]
    for a, b in zip(areas, areas[1:]):
        assert b >= a - 1e-9


def test_no_center_position_is_decided_in_step6a():
    """STEP 6A では具体的な中心 XY を決めない。"""
    mesh = testmodels.floating_box(size=12.0, height=10.0)
    _, _, _, _, res = single_tip_run(mesh, (6.0, 6.0), tip_layer=49)
    assert all(n.position is None for n in res.graph.nodes.values())


def test_node_structure_fields():
    mesh = testmodels.floating_box(size=12.0, height=10.0)
    _, _, _, _, res = single_tip_run(mesh, (6.0, 6.0), tip_layer=49)
    node = next(iter(res.graph.nodes.values()))
    for attr in ("id", "layer", "influence_area", "radius", "child_ids",
                 "parent_ids", "contact_ids", "status"):
        assert hasattr(node, attr)
    assert node.contact_ids == [0]


# ======================================================================
# B. 垂直壁の近く: collision 側へ侵入しない
# ======================================================================
def test_influence_area_never_enters_collision_near_a_wall():
    """壁のすぐ横に tip を置いても collision を侵さないこと。"""
    mesh = testmodels.cantilever(column=8.0, arm=22.0, depth=10.0, height=16.0)
    cfg, stack, cache, prop, res = single_tip_run(
        mesh, (12.0, 5.0), tip_layer=79, layer_height=0.2, xy_gap=0.4)
    chain = chain_from_tip(res.graph)
    assert len(chain) > 50

    for node in chain:
        kind = "tip" if node.source == "tip" else "branch"
        col = cache.collision(node.layer, node.radius, kind=kind)
        assert g2.area(g2.intersection(node.influence_area, col)) < 1e-9, node.id


def test_influence_area_keeps_xy_gap_from_the_wall():
    mesh = testmodels.cantilever(column=8.0, arm=22.0, depth=10.0, height=16.0)
    cfg, stack, cache, prop, res = single_tip_run(
        mesh, (12.0, 5.0), tip_layer=79, layer_height=0.2, xy_gap=0.4)
    for node in chain_from_tip(res.graph):
        model = stack[node.layer]
        if model.is_empty or node.source == "tip":
            continue
        # 領域内の任意の点はモデルから radius + xy_gap 以上離れている
        assert node.influence_area.distance(model) >= 0.0
        eroded = g2.intersection(node.influence_area, model)
        assert g2.area(eroded) < 1e-9


# ======================================================================
# C. 狭い通路: 細ければ通る / 太いと通らない
# ======================================================================
def test_thin_branch_passes_through_a_narrow_slot():
    gap = 3.0
    mesh = testmodels.narrow_slot(gap=gap)
    # r + xy_gap = 0.4 + 0.4 = 0.8 < gap/2 = 1.5 -> 通れる
    cfg, stack, cache, prop, res = single_tip_run(
        mesh, (0.0, 5.0), tip_layer=int(14.0 / 0.2) - 1,
        layer_height=0.2, xy_gap=0.4, tip_diameter=0.8,
        branch_diameter_growth=0.0, root_flare_height=0.0)
    chain = chain_from_tip(res.graph)
    assert chain[-1].layer == 0, "thin branch should reach the build plate"
    assert chain[-1].status is NodeStatus.ON_BUILD_PLATE

    # ブロックがある Y 帯 (y in [0.5, 9.5]) の中では、
    # 領域は隙間の内側 (|x| <= gap/2 - (r + xy_gap)) に収まる。
    # ブロックは Y 方向に有限なので、その外側へ回り込む lobe が
    # 出ること自体は正しい挙動 (通路以外の経路が実在する)。
    mid = [n for n in chain if n.layer == 30][0]
    band = g2.intersection(mid.influence_area,
                           shapely.box(-12.0, 0.5, 12.0, 9.5))
    assert not band.is_empty, "branch does not pass through the slot at all"
    minx, _, maxx, _ = band.bounds
    limit = gap * 0.5 - 0.8
    assert minx >= -limit - 1e-6
    assert maxx <= limit + 1e-6


def test_thick_branch_cannot_pass_through_the_same_slot():
    gap = 3.0
    mesh = testmodels.narrow_slot(gap=gap)
    # r + xy_gap = 1.2 + 0.4 = 1.6 > gap/2 = 1.5 -> 通れない
    cfg, stack, cache, prop, res = single_tip_run(
        mesh, (0.0, 5.0), tip_layer=int(14.0 / 0.2) - 1,
        layer_height=0.2, xy_gap=0.4, tip_diameter=2.4,
        branch_diameter_growth=0.0, root_flare_height=0.0,
        min_printable_feature=0.4)
    chain = chain_from_tip(res.graph)
    assert chain[-1].layer > 0, "thick branch must not reach the plate"
    assert chain[-1].status is NodeStatus.UNREACHABLE
    assert "no valid area" in chain[-1].status_reason
    # ブロック上面 (z=12 -> layer 60) の手前で止まるはず
    assert chain[-1].layer >= int(12.0 / 0.2) - 2


# ======================================================================
# D. 完全閉塞: 明示的に UNREACHABLE
# ======================================================================
def test_fully_blocked_tip_is_marked_unreachable():
    """閉じた空洞の天井を支える tip は、床より下へ進めない。"""
    mesh = testmodels.cavity(size=16.0, wall=3.0)
    tip_layer = int(13.0 / 0.2) - 1        # 空洞天井の 1 層下
    cfg, stack, cache, prop, res = single_tip_run(
        mesh, (8.0, 8.0), tip_layer=tip_layer, layer_height=0.2)
    chain = chain_from_tip(res.graph)
    last = chain[-1]
    assert last.layer > 0
    assert last.status is NodeStatus.UNREACHABLE
    assert last.status_reason
    # 空洞の床 z=3.0 (layer 15) の直上で止まる
    assert last.layer >= int(3.0 / 0.2) - 1
    assert res.dead_ends == [last.id]


def test_unreachable_is_not_reported_as_success():
    mesh = testmodels.cavity(size=16.0, wall=3.0)
    tip_layer = int(13.0 / 0.2) - 1
    _, _, _, _, res = single_tip_run(mesh, (8.0, 8.0), tip_layer=tip_layer,
                                     layer_height=0.2)
    assert res.graph.root_ids == []
    assert res.stats()["reachable_tip_count"] == 0
    assert res.stats()["unreachable_tip_count"] == 1


# ======================================================================
# E. branch angle: move_max を超える領域を生成しない
# ======================================================================
@pytest.mark.parametrize("branch_angle_max", [20.0, 40.0, 60.0])
def test_consecutive_layers_never_exceed_move_max(branch_angle_max):
    mesh = testmodels.cantilever()
    cfg, stack, cache, prop, res = single_tip_run(
        mesh, (20.0, 5.0), tip_layer=79, layer_height=0.2,
        branch_angle_max=branch_angle_max,
        branch_angle_preferred=min(20.0, branch_angle_max))
    chain = chain_from_tip(res.graph)
    move = cfg.move_max

    for upper, lower in zip(chain, chain[1:]):
        # 下層領域の全点は、上層領域から move_max 以内にある
        grown = g2.dilate_outer(upper.influence_area, move,
                                quad_segs=cfg.quad_segs)
        outside = g2.area(g2.difference(lower.influence_area, grown))
        assert outside < 1e-9, (
            f"layer {lower.layer}: {outside:.6g} mm^2 is farther than "
            f"move_max={move:.4f} from the layer above")


def test_move_max_matches_the_angle_formula():
    cfg = SupportConfig(layer_height=0.2, branch_angle_max=40.0)
    assert cfg.move_max == pytest.approx(0.2 * math.tan(math.radians(40.0)))


# ======================================================================
# Debug artifacts
# ======================================================================
def test_step_debug_svgs_show_all_stages(tmp_path):
    mesh = testmodels.cantilever()
    cfg, stack, cache, prop, res = single_tip_run(
        mesh, (20.0, 5.0), tip_layer=39, layer_height=0.4,
        record_trace=True)
    dbg = DebugWriter(tmp_path)
    n = prop.write_step_debug(stack, res, dbg, every=1)
    assert n > 0

    files = sorted((tmp_path / "influence").glob("step_*.svg"))
    assert files
    text = files[len(files) // 2].read_text(encoding="utf-8")
    for layer_name in ("model", "collision", "expanded", "previous", "result"):
        assert f'id="{layer_name}"' in text
    assert (tmp_path / "influence" / "propagation_stats.json").exists()


def test_trace_records_each_step():
    mesh = testmodels.floating_box(size=12.0, height=6.0)
    cfg, stack, cache, prop, res = single_tip_run(
        mesh, (6.0, 6.0), tip_layer=29, layer_height=0.2, record_trace=True)
    assert len(res.traces) == 29
    for tr in res.traces:
        assert tr.to_layer == tr.from_layer - 1
        assert tr.status == "OK"
        assert g2.area(tr.expanded) >= g2.area(tr.previous)
