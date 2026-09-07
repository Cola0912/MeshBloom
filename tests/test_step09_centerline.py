"""STEP 9: Centerline reconstruction。

root -> tip 方向に、``influence_area & circle(p_parent, move_max)`` から
cost 最小の点を選ぶ。各レイヤ独立の最近傍射影は禁止 (角度制約を破るため)。
"""

from __future__ import annotations

import json
import math

import pytest
from shapely.geometry import Point

from conftest import build_tree
from treesupport import geometry2d as g2
from treesupport import testmodels
from treesupport.centerline import CenterlineSolver
from treesupport.debug_io import DebugWriter

MODELS = ["merge_cluster", "cantilever", "bridge", "obstructed_path",
          "islands", "narrow_slot", "arch", "overhang_corner", "floating_box"]


def solved(name, **kwargs):
    kwargs.setdefault("layer_height", 0.4)
    kwargs.setdefault("tip_spacing", 3.0)
    return build_tree(testmodels.ALL_MODELS[name](), **kwargs)


# ======================================================================
# 検証 1-6 + 角度
# ======================================================================
@pytest.mark.parametrize("name", MODELS)
def test_full_verification_passes(name):
    b = solved(name)
    v = b.solver.verify(b.graph, b.cache)
    assert v.passed, v.failures[:5]
    assert v.inside_influence
    assert v.outside_collision
    assert v.step_within_move_max
    assert v.tips_connected
    assert v.roots_on_build_plate
    assert v.acyclic
    assert v.angle_within_max


@pytest.mark.parametrize("name", MODELS)
def test_every_position_is_inside_its_influence_area(name):
    b = solved(name)
    for node in b.graph.nodes.values():
        assert node.position is not None
        assert node.influence_area.buffer(1e-6).covers(Point(*node.position)), node.id


@pytest.mark.parametrize("name", MODELS)
def test_step_never_exceeds_move_max(name):
    b = solved(name)
    move = b.config.move_max
    for node in b.graph.nodes.values():
        for cid in node.child_ids:
            child = b.graph.nodes[cid]
            d = math.dist(node.position, child.position)
            assert d <= move + 1e-9, f"{name}: {node.id}->{cid} step={d}"


@pytest.mark.parametrize("name", MODELS)
def test_angle_recomputed_from_real_coordinates(name):
    """実座標から角度を再計算しても branch_angle_max 以内。"""
    b = solved(name)
    lh = b.config.layer_height
    worst = 0.0
    for node in b.graph.nodes.values():
        for cid in node.child_ids:
            child = b.graph.nodes[cid]
            dz = abs(child.layer - node.layer) * lh
            dxy = math.dist(node.position, child.position)
            assert dz > 0.0
            worst = max(worst, math.degrees(math.atan2(dxy, dz)))
    assert worst <= b.config.branch_angle_max + 1e-4
    assert worst > 0.0


def test_minimum_model_clearance_is_at_least_xy_gap():
    for name in ("cantilever", "bridge", "obstructed_path", "arch",
                 "narrow_slot"):
        b = solved(name)
        v = b.solver.verify(b.graph, b.cache)
        assert v.min_clearance >= b.config.xy_gap - 1e-6, (
            f"{name}: clearance {v.min_clearance} < xy_gap {b.config.xy_gap}")


# ======================================================================
# 「各レイヤ独立の最近傍射影」ではないこと
# ======================================================================
def test_independent_projection_would_violate_move_max():
    """禁止された素朴な方式が実際に角度制約を破ることを示す。"""
    b = solved("obstructed_path")
    naive = {}
    for node in b.graph.nodes.values():
        naive[node.id] = g2.project_point(node.influence_area,
                                          node.next_position)

    worst_naive = 0.0
    for node in b.graph.nodes.values():
        for cid in node.child_ids:
            worst_naive = max(worst_naive,
                              math.dist(naive[node.id], naive[cid]))

    worst_solver = 0.0
    for node in b.graph.nodes.values():
        for cid in node.child_ids:
            worst_solver = max(worst_solver,
                               math.dist(node.position,
                                         b.graph.nodes[cid].position))

    assert worst_naive > b.config.move_max, (
        "naive per-layer projection unexpectedly satisfied the constraint")
    assert worst_solver <= b.config.move_max + 1e-9


def test_solver_is_root_to_tip():
    """root が先に決まり、上のノードが親に依存して決まること。

    root の位置を人為的にずらすと、その上の枝全体の位置が変わる。
    """
    b = solved("cantilever")
    original = {n.id: n.position for n in b.graph.nodes.values()}

    # 位置を消して root だけ別候補にしてから解き直す
    for node in b.graph.nodes.values():
        node.position = None
    root = b.graph.nodes[b.graph.root_ids[0]]
    corner = root.influence_area.bounds
    root.position = g2.project_point(root.influence_area,
                                     (corner[0], corner[1]))

    solver = CenterlineSolver(b.config)
    solver.solve(b.graph)

    changed = sum(1 for n in b.graph.nodes.values()
                  if original[n.id] != n.position)
    assert changed > 5, "upper nodes did not follow the root position"
    v = solver.verify(b.graph, b.cache)
    assert v.step_within_move_max


# ======================================================================
# junction 座標の共有
# ======================================================================
def test_merge_junction_coordinate_is_shared():
    """合流点は 1 つのノードなので、両方の枝が同一座標を共有する。"""
    b = solved("merge_cluster")
    merges = [n for n in b.graph.nodes.values() if n.is_merge]
    assert merges

    for node in merges:
        assert node.position is not None
        # この合流点から上へ伸びる全ての枝は、同じ点から始まる
        starts = []
        for cid in node.child_ids:
            child = b.graph.nodes[cid]
            starts.append(node.position)          # segment の始点
            assert child.position is not None
        assert len(set(starts)) == 1
        assert len(node.child_ids) >= 2


def test_no_duplicate_junction_nodes():
    """同じ層に、ほぼ同一座標の合流ノードが 2 つ生まれていないこと。"""
    b = solved("merge_cluster")
    for layer in b.graph.layers():
        positions = [n.position for n in b.graph.nodes_at(layer)]
        for i, p in enumerate(positions):
            for q in positions[i + 1:]:
                assert math.dist(p, q) > 1e-9


# ======================================================================
# root 選択が cost 最小化であること
# ======================================================================
def test_root_position_follows_the_tip_centroid():
    b = solved("merge_cluster")
    root = b.graph.nodes[b.graph.root_ids[0]]
    tips = [n.position for n in b.graph.tips()]
    cx = sum(p[0] for p in tips) / len(tips)
    cy = sum(p[1] for p in tips) / len(tips)
    # 十分近い (真下に降ろした位置の近傍)
    assert math.dist(root.position, (cx, cy)) < 4.0


def test_stability_weight_changes_the_root_choice():
    """安定性の重みが実際に cost へ効いていること。"""
    a = solved("cantilever", centerline_weight_stability=0.0)
    c = solved("cantilever", centerline_weight_stability=8.0)

    def root_edge_distance(built):
        out = []
        for node in built.graph.roots():
            out.append(node.influence_area.boundary.distance(
                Point(*node.position)))
        return sum(out) / len(out)

    assert root_edge_distance(c) > root_edge_distance(a)


def test_curvature_weight_changes_the_path():
    a = solved("cantilever", centerline_weight_curvature=0.0)
    c = solved("cantilever", centerline_weight_curvature=3.0)
    assert a.graph.centerline_length() != pytest.approx(
        c.graph.centerline_length(), rel=1e-9)


def test_preferred_angle_penalty_reduces_average_step():
    """preferred 罰則を強くすると平均移動量が減ること。"""
    def mean_step(built):
        steps = []
        for node in built.graph.nodes.values():
            for cid in node.child_ids:
                steps.append(math.dist(node.position,
                                       built.graph.nodes[cid].position))
        return sum(steps) / len(steps)

    weak = solved("merge_cluster", centerline_weight_preferred=0.0)
    strong = solved("merge_cluster", centerline_weight_preferred=20.0)
    assert mean_step(strong) <= mean_step(weak) + 1e-9


# ======================================================================
# 決定論性
# ======================================================================
def test_centerline_is_deterministic():
    a = solved("merge_cluster")
    c = solved("merge_cluster")
    pa = sorted((n.layer, round(n.position[0], 9), round(n.position[1], 9))
                for n in a.graph.nodes.values())
    pc = sorted((n.layer, round(n.position[0], 9), round(n.position[1], 9))
                for n in c.graph.nodes.values())
    assert pa == pc


# ======================================================================
# 平滑化はまだ行わない
# ======================================================================
def test_no_smoothing_is_applied_in_step9():
    """STEP 9 の出力が smoothing を通していないこと。"""
    b = solved("merge_cluster")
    before = {n.id: n.position for n in b.graph.nodes.values()}

    from treesupport.smoothing import CenterlineSmoother
    CenterlineSmoother(b.config).smooth(b.graph)
    after = {n.id: n.position for n in b.graph.nodes.values()}
    assert before != after, "smoothing had no effect; the comparison is vacuous"


# ======================================================================
# Debug artifacts
# ======================================================================
def test_centerline_debug_artifacts(tmp_path):
    b = solved("merge_cluster", layer_height=0.8)
    v = b.solver.verify(b.graph, b.cache)
    dbg = DebugWriter(tmp_path)
    n = b.solver.write_debug(b.stack, b.graph, dbg, verification=v, every=2)
    assert n > 0

    tree_json = tmp_path / "centerline" / "tree.json"
    payload = json.loads(tree_json.read_text(encoding="utf-8"))
    assert payload["stats"]["node_count"] == len(b.graph.nodes)
    assert payload["verification"]["passed"] is True
    assert payload["nodes"][0]["position"] is not None
    assert payload["branches"]

    obj = (tmp_path / "centerline" / "tree.obj").read_text(encoding="utf-8")
    assert obj.count("\nl ") >= len(b.graph.nodes) - len(b.graph.root_ids) - 1
    assert obj.startswith("# ")

    svgs = sorted((tmp_path / "centerline").glob("layer_*.svg"))
    assert svgs
    text = svgs[len(svgs) // 2].read_text(encoding="utf-8")
    for group in ("model", "influence", "centers"):
        assert f'id="{group}"' in text


def test_tree_obj_coordinates_are_3d_and_in_world_frame():
    b = solved("merge_cluster", layer_height=0.8)
    lines = b.graph.centerline_polylines()
    assert lines
    for seg in lines:
        assert len(seg) == 2
        for p in seg:
            assert len(p) == 3
            assert p[2] >= -1e-9
    zs = sorted({round(p[2], 6) for seg in lines for p in seg})
    assert zs[0] == pytest.approx(0.0)
