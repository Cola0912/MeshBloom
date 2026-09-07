"""STEP 6 へ進む前の事前確認 (3 項目)。

1. world polygon が「最大水平移動 + 最大枝半径 + xy_gap + 安全余裕」を確保し、
   本来存在する迂回経路を境界で消さないこと。
2. tip の Z ギャップが**実座標**で ``model_bottom_z - tip_top_z >= top_z_gap``
   を満たすこと (レイヤ番号だけの確認では不十分)。
3. ``base[j] -> dilate(radius)`` の最適化が、直接計算より
   **危険側 (小さい側)** へずれないこと。
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
from treesupport.overhang import OverhangDetector
from treesupport.slicer import ModelSlicer
from treesupport.tips import ContactSampler
from treesupport.volumes import CollisionCache


def make(mesh, **kwargs):
    cfg = SupportConfig(**kwargs)
    stack = ModelSlicer(cfg).slice(mesh)
    return CollisionCache(cfg, stack), stack, cfg


# ======================================================================
# 1. world polygon
# ======================================================================
def test_world_margin_covers_max_lateral_travel():
    mesh = testmodels.cantilever()
    cache, stack, cfg = make(mesh)
    travel = cfg.max_lateral_travel(stack.model_z_max)
    need = travel + cfg.max_radius + cfg.xy_gap
    mb = stack.xy_bounds
    wb = cache.world.bounds
    for i, sign in ((0, -1), (1, -1), (2, 1), (3, 1)):
        margin = (wb[i] - mb[i]) * sign
        assert margin >= need, f"world margin {margin} < required {need}"
    assert cfg.reach_safety_margin > 0.0


def test_world_margin_is_not_capped_for_tall_models():
    """高いモデルでも余白が頭打ちにならないこと (旧実装は 50mm で上限)。"""
    tall = testmodels.unit_cube(size=200.0)
    cache, stack, cfg = make(tall, layer_height=2.0)
    travel = cfg.max_lateral_travel(200.0)
    assert travel > 100.0
    margin = stack.xy_bounds[0] - cache.world.bounds[0]
    assert margin >= travel + cfg.max_radius + cfg.xy_gap


def test_world_boundary_does_not_delete_a_valid_detour():
    """world 境界のせいで実在する迂回経路が消えていないこと。

    obstructed_path はブロックを迂回しないと着地できない。
    十分大きい world なら到達し、極端に小さい world なら到達しない
    (= 検査が実効的である) ことを両方確認する。
    """
    mesh = testmodels.obstructed_path()
    ok = build_tree(mesh, layer_height=0.4, tip_spacing=3.0)
    assert len(ok.graph.root_ids) >= 1
    assert len(ok.tips) > 0
    assert len(ok.propagation.unplaced_tips) == 0

    # world をモデル bbox ぎりぎりに絞ると、迂回できず到達不能になるはず
    tiny = build_tree(mesh, layer_height=0.4, tip_spacing=3.0, reach_margin=0.5)
    assert len(tiny.graph.nodes) < len(ok.graph.nodes)


def test_build_plate_option_clips_the_world():
    mesh = testmodels.cantilever()
    plate = (-5.0, -5.0, 35.0, 15.0)
    cache, stack, cfg = make(mesh, build_plate=plate)
    wb = cache.world.bounds
    assert wb[0] >= plate[0] - 1e-9
    assert wb[2] <= plate[2] + 1e-9


# ======================================================================
# 2. 実座標での Z ギャップ
# ======================================================================
@pytest.mark.parametrize("layer_height,top_z_gap", [
    (0.2, 0.2), (0.2, 0.5), (0.3, 0.2), (0.15, 0.3), (0.25, 0.25),
])
def test_tip_z_gap_in_real_coordinates(layer_height, top_z_gap):
    """レイヤ番号ではなく実 Z 座標でギャップを検査する。

    * モデル下面の実 Z = (最初にモデルが存在するレイヤ) * layer_height
      (スライス位置はスラブ中央だが、材料の下面はスラブ下端)
    * tip 上端の実 Z = tip.layer * layer_height (tip は枝の最上点)
    """
    height = 8.0
    mesh = testmodels.floating_box(size=12.0, thickness=4.0, height=height)
    b = build_tree(mesh, layer_height=layer_height, top_z_gap=top_z_gap,
                   tip_spacing=3.0, solve_centerline=False)
    cfg = b.config
    stack = b.stack

    m = stack.first_model_layer
    model_bottom_z = m * layer_height
    assert model_bottom_z <= height + 1e-9
    assert model_bottom_z > height - layer_height - 1e-9

    assert len(b.tips) > 0
    for tip in b.tips.tips:
        tip_top_z = tip.z
        assert tip_top_z == pytest.approx(tip.layer * layer_height)
        gap = model_bottom_z - tip_top_z
        assert gap >= top_z_gap - 1e-12, (
            f"real z gap {gap} < top_z_gap {top_z_gap}"
        )


def test_slice_sampling_z_is_not_the_material_bottom():
    """スライス位置 (スラブ中央) と材料下端を混同していないこと。"""
    cfg = SupportConfig(layer_height=0.2)
    stack = ModelSlicer(cfg).slice(testmodels.floating_box(height=8.0))
    m = stack.first_model_layer
    assert stack.z_mid(m) == pytest.approx(m * 0.2 + 0.1)
    assert stack.z_bottom(m) == pytest.approx(m * 0.2)
    assert stack.z_bottom(m) == pytest.approx(8.0)


def test_tip_z_gap_holds_for_a_mid_air_overhang():
    """途中の高さに現れるオーバーハングでも実座標ギャップが成立すること。"""
    b = build_tree(testmodels.cantilever(height=16.0), layer_height=0.2,
                   top_z_gap=0.2, tip_spacing=3.0, solve_centerline=False)
    for tip in b.tips.tips:
        model_bottom_z = tip.overhang_layer * b.config.layer_height
        assert model_bottom_z - tip.z >= b.config.top_z_gap - 1e-12


# ======================================================================
# 3. collision 最適化の安全性 (property test)
# ======================================================================
@pytest.mark.parametrize("name", ["cantilever", "bridge", "arch", "cavity",
                                  "obstructed_path", "overhang_corner"])
@pytest.mark.parametrize("radius", [0.4, 1.3, 3.0])
def test_optimized_collision_is_never_smaller_than_direct(name, radius):
    """最適化 collision が直接計算より危険側 (小さい側) にならないこと。

    完全一致は要求しない。``direct ⊆ optimized`` (面積差が負でない) を要求する。
    """
    mesh = testmodels.ALL_MODELS[name]()
    cache, stack, cfg = make(mesh, layer_height=0.4)
    checked = 0
    for j in range(0, stack.n_layers, 5):
        for kind in ("branch", "tip"):
            opt = cache.collision(j, radius, kind=kind)
            direct = cache.direct_collision(j, radius, kind=kind)
            if direct.is_empty and opt.is_empty:
                continue
            missing = g2.area(g2.difference(direct, opt))
            assert missing <= 1e-9, (
                f"{name} layer {j} kind={kind}: optimized collision misses "
                f"{missing:.6g} mm^2 of the direct collision (unsafe)"
            )
            checked += 1
    assert checked > 0


def test_optimized_collision_is_close_to_direct():
    """安全側であっても、過剰に大きくないこと (実用上の精度確認)。"""
    cache, stack, cfg = make(testmodels.cantilever(), layer_height=0.4)
    for j in range(0, stack.n_layers, 7):
        opt = cache.collision(j, 1.0)
        direct = cache.direct_collision(j, 1.0)
        if direct.is_empty:
            continue
        extra = g2.area(g2.difference(opt, direct))
        assert extra <= 0.02 * g2.area(direct) + 1e-6


def test_collision_offset_is_conservative_versus_true_circle():
    """離散化した offset が真円の Minkowski 和を必ず含むこと。"""
    cache, stack, cfg = make(testmodels.unit_cube(size=10.0), xy_gap=0.4)
    col = cache.collision(20, 1.0)
    truth = shapely.box(0, 0, 10, 10).buffer(1.4, quad_segs=256)
    assert g2.area(g2.difference(truth, col)) <= 1e-9
