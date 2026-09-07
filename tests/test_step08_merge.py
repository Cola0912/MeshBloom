"""STEP 8: Branch Merge。

* AABB (STRtree) で候補を絞ってから精密判定
* 交差があるだけでは merge しない。merge 後半径で collision と reach を再評価する
* 候補が複数あるときは決定論的スコアで順序付け
* 入力 tip の列挙順を変えても結果が変わらない
"""

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
from treesupport.graph import NodeStatus
from treesupport.merge import TreeMerger
from treesupport.propagate import TreePropagator
from treesupport.radius import RadiusSolver
from treesupport.tips import ContactPoint, TipSet


def run(mesh, **kwargs):
    kwargs.setdefault("solve_centerline", False)
    return build_tree(mesh, **kwargs)


def synthetic_tips(stack, points, layer):
    """指定座標に tip を並べた TipSet を作る。"""
    tips = []
    for i, (x, y) in enumerate(points):
        tips.append(ContactPoint(id=i, layer=layer, x=x, y=y,
                                 z=stack.z_bottom(layer),
                                 overhang_layer=layer + 1))
    return TipSet(tips=tips, by_layer={layer: list(tips)})


def propagate_with(cfg, stack, cache, tipset, record_geometry=False):
    solver = RadiusSolver(cfg)
    merger = TreeMerger(cfg, cache, solver, record_geometry=record_geometry)
    prop = TreePropagator(cfg, cache, solver, merger)
    res = prop.propagate(stack, tipset)
    return merger, res


# ======================================================================
# A. tip 2 本 -> 1 trunk
# ======================================================================
def test_two_tips_merge_into_one_trunk():
    from treesupport.slicer import ModelSlicer
    from treesupport.volumes import CollisionCache

    cfg = SupportConfig(layer_height=0.4, tip_spacing=3.0)
    mesh = testmodels.floating_box(size=12.0, thickness=2.0, height=12.0)
    stack = ModelSlicer(cfg).slice(mesh)
    cache = CollisionCache(cfg, stack)
    tipset = synthetic_tips(stack, [(4.0, 6.0), (8.0, 6.0)], layer=29)

    merger, res = propagate_with(cfg, stack, cache, tipset)
    g = res.graph

    assert len(merger.events) == 1
    assert len(g.root_ids) == 1
    root = g.nodes[g.root_ids[0]]
    assert sorted(root.contact_ids) == [0, 1]

    merged = [n for n in g.nodes.values() if n.is_merge]
    assert len(merged) == 1
    assert len(merged[0].child_ids) == 2


def test_merged_node_children_are_the_two_branches():
    from treesupport.slicer import ModelSlicer
    from treesupport.volumes import CollisionCache

    cfg = SupportConfig(layer_height=0.4, tip_spacing=3.0)
    mesh = testmodels.floating_box(size=12.0, thickness=2.0, height=12.0)
    stack = ModelSlicer(cfg).slice(mesh)
    cache = CollisionCache(cfg, stack)
    tipset = synthetic_tips(stack, [(4.0, 6.0), (8.0, 6.0)], layer=29)
    merger, res = propagate_with(cfg, stack, cache, tipset)

    ev = merger.events[0]
    merged = res.graph.nodes[ev.merged]
    children = [res.graph.nodes[c] for c in merged.child_ids]
    assert len(children) == 2
    assert {tuple(c.contact_ids) for c in children} == {(0,), (1,)}
    assert sorted(merged.contact_ids) == [0, 1]


# ======================================================================
# B. tip 3 本 -> tree topology が正しい
# ======================================================================
def test_three_tips_form_a_valid_tree():
    from treesupport.slicer import ModelSlicer
    from treesupport.volumes import CollisionCache

    cfg = SupportConfig(layer_height=0.4, tip_spacing=3.0)
    mesh = testmodels.floating_box(size=16.0, thickness=2.0, height=14.0)
    stack = ModelSlicer(cfg).slice(mesh)
    cache = CollisionCache(cfg, stack)
    tipset = synthetic_tips(stack, [(4.0, 8.0), (8.0, 8.0), (12.0, 8.0)],
                            layer=34)
    merger, res = propagate_with(cfg, stack, cache, tipset)
    g = res.graph

    assert len(g.root_ids) == 1
    assert len(merger.events) == 2
    root = g.nodes[g.root_ids[0]]
    assert sorted(root.contact_ids) == [0, 1, 2]

    # 各ノードの親は高々 1 つ (= 木構造)
    for node in g.nodes.values():
        assert len(node.parent_ids) <= 1
    # 3 本の tip、2 回の分岐
    assert len(g.tips()) == 3
    assert sum(1 for n in g.nodes.values() if n.is_merge) == 2


def test_graph_has_no_cycles():
    b = run(testmodels.merge_cluster(), layer_height=0.4, tip_spacing=3.0)
    g = b.graph
    color: dict[int, int] = {}

    def visit(nid: int) -> None:
        state = color.get(nid, 0)
        assert state != 1, f"cycle detected at node {nid}"
        if state == 2:
            return
        color[nid] = 1
        for cid in g.nodes[nid].child_ids:
            visit(cid)
        color[nid] = 2

    import sys
    sys.setrecursionlimit(20000)
    for root in g.root_ids:
        visit(root)
    assert len(color) == len(g.nodes)


def test_every_node_is_reachable_from_exactly_one_root():
    b = run(testmodels.merge_cluster(), layer_height=0.4, tip_spacing=3.0)
    g = b.graph
    seen: dict[int, int] = {}
    for root in g.root_ids:
        stack_ = [root]
        while stack_:
            nid = stack_.pop()
            assert nid not in seen or seen[nid] == root
            seen[nid] = root
            stack_.extend(g.nodes[nid].child_ids)
    assert len(seen) == len(g.nodes)


# ======================================================================
# C. 交差はあるが merged radius では衝突 -> merge 禁止
# ======================================================================
def test_merge_rejected_when_merged_radius_collides():
    """influence は重なるが、太った半径ではモデルに当たる配置。"""
    from treesupport.slicer import ModelSlicer
    from treesupport.volumes import CollisionCache

    # 幅 3mm のスロットの真上に 2 本の tip。
    # 個別半径 r=1.0 なら r+xy=1.4 < 1.5 で通るが、
    # merge すると sqrt(2)*1.0 = 1.414 -> 1.414+0.4 = 1.81 > 1.5 で通らない。
    cfg = SupportConfig(layer_height=0.4, tip_diameter=2.0, xy_gap=0.4,
                        branch_diameter_growth=0.0, root_flare_height=0.0,
                        tip_influence_radius=0.6, merge_min_area=1e-3)
    mesh = testmodels.narrow_slot(gap=3.0, depth=40.0, wall_height=12.0,
                                  plate_half=1.0, plate_z=13.0)
    stack = ModelSlicer(cfg).slice(mesh)
    cache = CollisionCache(cfg, stack)
    layer = int(13.0 / 0.4) - 1
    tipset = synthetic_tips(stack, [(0.0, 19.6), (0.0, 20.4)], layer=layer)

    merger, res = propagate_with(cfg, stack, cache, tipset,
                                 record_geometry=True)
    # スロット内では merge が拒否され続ける
    reasons = {r.reason for r in merger.rejections}
    assert merger.rejections, "expected at least one rejected merge"
    assert any("collides" in r or "reach" in r for r in reasons), reasons
    # 個別の枝はプレートまで到達している
    assert res.stats()["reachable_tip_count"] == 2


def test_forbid_policy_blocks_oversized_merges():
    cfg_common = dict(layer_height=0.4, tip_spacing=3.0,
                      branch_diameter_max=1.0, tip_diameter=0.8)
    clamp = run(testmodels.merge_cluster(), merge_radius_policy="clamp",
                **cfg_common)
    forbid = run(testmodels.merge_cluster(), merge_radius_policy="forbid",
                 **cfg_common)

    assert len(clamp.merger.events) > len(forbid.merger.events)
    assert max(n.radius for n in clamp.graph.nodes.values()) <= 0.5 + 1e-9
    assert any("branch_diameter_max" in r.reason for r in forbid.merger.rejections)


# ======================================================================
# D. 細い通路: 個別なら通るが merged では通れない
# ======================================================================
def test_merge_forbidden_in_a_narrow_passage():
    from treesupport.slicer import ModelSlicer
    from treesupport.volumes import CollisionCache

    cfg = SupportConfig(layer_height=0.4, tip_diameter=1.6, xy_gap=0.3,
                        branch_diameter_growth=0.0, root_flare_height=0.0,
                        tip_influence_radius=0.8)
    # gap=2.6 -> 個別 (0.8+0.3=1.1 < 1.3) は通る、
    #            merge (sqrt(2)*0.8=1.13, +0.3=1.43 > 1.3) は通らない
    mesh = testmodels.narrow_slot(gap=2.6, depth=40.0, wall_height=12.0,
                                  plate_half=0.9, plate_z=13.0)
    stack = ModelSlicer(cfg).slice(mesh)
    cache = CollisionCache(cfg, stack)
    layer = int(13.0 / 0.4) - 1
    tipset = synthetic_tips(stack, [(0.0, 19.5), (0.0, 20.5)], layer=layer)
    merger, res = propagate_with(cfg, stack, cache, tipset)

    # 通路内 (z<12 -> layer<30) では merge されていない
    in_passage = [e for e in merger.events if e.layer < 30]
    assert not in_passage, f"merged inside the passage: {in_passage}"
    assert res.stats()["reachable_tip_count"] == 2


# ======================================================================
# E. merge 後も全 contact が root へ接続される
# ======================================================================
def test_all_contacts_stay_connected_after_merges():
    b = run(testmodels.merge_cluster(), layer_height=0.4, tip_spacing=3.0)
    g = b.graph
    root_contacts: set[int] = set()
    for root in g.roots():
        root_contacts.update(root.contact_ids)

    all_contacts = {t.id for t in b.tips.tips}
    assert root_contacts == all_contacts

    # 各 tip から root まで親を辿れる
    roots = set(g.root_ids)
    for tip in g.tips():
        node = tip
        guard = 0
        while node.parent_ids:
            node = g.nodes[node.parent_ids[0]]
            guard += 1
            assert guard < 10000
        assert node.id in roots


def test_merge_reduces_root_count():
    merged = run(testmodels.merge_cluster(), layer_height=0.4, tip_spacing=3.0)
    separate = run(testmodels.merge_cluster(), layer_height=0.4,
                   tip_spacing=3.0, merge_enabled=False)
    assert len(merged.graph.root_ids) == 1
    assert len(separate.graph.root_ids) == len(separate.tips.tips)
    assert len(merged.merger.events) == len(merged.tips.tips) - 1


def test_merged_radius_is_area_preserving_or_clamped():
    b = run(testmodels.merge_cluster(), layer_height=0.4, tip_spacing=3.0)
    for ev in b.merger.events:
        raw = math.hypot(ev.radius_a, ev.radius_b)
        expected = min(b.config.max_radius, raw)
        assert ev.radius_merged == pytest.approx(expected)
        assert ev.radius_merged >= max(ev.radius_a, ev.radius_b) - 1e-12


def test_merged_area_is_inside_both_inputs():
    """merge 後領域は両方の元領域の部分集合 (角度保証の根拠)。"""
    b = run(testmodels.merge_cluster(), layer_height=0.4, tip_spacing=3.0,
            record_geometry=True)
    for ev in b.merger.events:
        if not ev.geometry:
            continue
        merge_area = ev.geometry["merge_area"]
        for key in ("a", "b"):
            outside = g2.area(g2.difference(merge_area, ev.geometry[key]))
            assert outside < 1e-9


# ======================================================================
# 決定論性
# ======================================================================
def _signature(result):
    out = []
    for node in result.graph.nodes.values():
        out.append((node.layer, tuple(sorted(node.contact_ids)),
                    round(node.radius, 6),
                    round(g2.area(node.influence_area), 6),
                    len(node.child_ids)))
    return sorted(out)


@pytest.mark.parametrize("model", ["merge_cluster", "floating_box", "cantilever"])
def test_merge_result_is_independent_of_tip_order(model):
    import random

    from treesupport.slicer import ModelSlicer
    from treesupport.volumes import CollisionCache

    cfg = SupportConfig(layer_height=0.4, tip_spacing=3.0)
    mesh = testmodels.ALL_MODELS[model]()
    stack = ModelSlicer(cfg).slice(mesh)
    cache = CollisionCache(cfg, stack)

    base = build_tree(mesh, layer_height=0.4, tip_spacing=3.0,
                      solve_centerline=False)
    tips = list(base.tips.tips)

    rng = random.Random(12345)
    shuffled = list(tips)
    rng.shuffle(shuffled)
    by_layer: dict[int, list] = {}
    for t in shuffled:
        by_layer.setdefault(t.layer, []).append(t)
    tipset = TipSet(tips=shuffled, by_layer=by_layer)

    merger, res = propagate_with(cfg, stack, cache, tipset)
    assert _signature(res) == _signature(base.propagation)
    assert len(merger.events) == len(base.merger.events)


def test_merge_is_reproducible_across_runs():
    a = run(testmodels.merge_cluster(), layer_height=0.4, tip_spacing=3.0)
    b = run(testmodels.merge_cluster(), layer_height=0.4, tip_spacing=3.0)
    assert _signature(a.propagation) == _signature(b.propagation)
    assert [e.to_dict() for e in a.merger.events] == \
           [e.to_dict() for e in b.merger.events]


# ======================================================================
# 候補探索が総当たりでないこと
# ======================================================================
def test_candidate_search_uses_spatial_index():
    """候補ペア数がノード数の 2 乗よりずっと小さいこと。"""
    b = run(testmodels.islands(count=3, spacing=20.0), layer_height=0.4,
            tip_spacing=3.0)
    stats = b.merger.stats()
    n = len(b.tips.tips)
    assert stats["merge_candidates"] > 0
    # 総当たりなら 1 レイヤあたり n*(n-1)/2、全レイヤで膨大になる
    assert stats["merge_candidates"] < n * n * 4


# ======================================================================
# Debug artifacts
# ======================================================================
def test_merge_debug_artifacts(tmp_path):
    import json

    b = run(testmodels.merge_cluster(), layer_height=0.8, tip_spacing=3.0,
            record_geometry=True)
    dbg = DebugWriter(tmp_path)
    n = b.merger.write_debug(b.stack, dbg)
    assert n > 0

    path = tmp_path / "merge" / "merge_events.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["stats"]["merge_count"] > 0
    for ev in payload["accepted"]:
        for key in ("layer", "node_a", "node_b", "merged", "contacts_a",
                    "contacts_b", "radius_merged", "intersect_area",
                    "merge_area", "accepted"):
            assert key in ev
    for rj in payload["rejected"]:
        assert rj["reason"]
        assert rj["accepted"] is False

    svgs = sorted((tmp_path / "merge").glob("L*_merge_*.svg"))
    assert svgs
    text = svgs[0].read_text(encoding="utf-8")
    for group in ("branch_a", "branch_b", "intersection", "merge_area",
                  "collision"):
        assert f'id="{group}"' in text
