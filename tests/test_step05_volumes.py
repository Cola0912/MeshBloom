"""STEP 5: collision mask と reachability のテスト。"""

from __future__ import annotations

import pytest
import shapely
from shapely.geometry import Point

from treesupport import geometry2d as g2
from treesupport import testmodels
from treesupport.config import SupportConfig
from treesupport.debug_io import DebugWriter
from treesupport.slicer import ModelSlicer
from treesupport.volumes import CollisionCache


def make(mesh, **kwargs):
    cfg = SupportConfig(**kwargs)
    stack = ModelSlicer(cfg).slice(mesh)
    return CollisionCache(cfg, stack), stack, cfg


# ----------------------------------------------------------------------
def test_collision_is_model_offset_by_radius_plus_xy_gap():
    cache, stack, cfg = make(testmodels.unit_cube(size=10.0), xy_gap=0.4)
    r = 1.0
    col = cache.collision(20, r, kind="branch")
    minx, miny, maxx, maxy = col.bounds
    expected = 1.0 + 0.4
    # 円弧の外接補正 (x1.0048) と collision_safety_ratio (x1.02) の分だけ
    # 真値よりわずかに大きい。「必ず安全側 (大きい方) へ丸める」ことが要件。
    upper = 1.0 + cfg.collision_safety_ratio + 0.01
    assert -expected >= minx >= -expected * upper
    assert 10.0 + expected <= maxx <= 10.0 + expected * upper
    # 外接近似なので真の Minkowski 和を必ず含む
    truth = shapely.box(0, 0, 10, 10).buffer(expected, quad_segs=64)
    assert col.covers(truth.buffer(-1e-9))


def test_collision_radius_is_quantized_upward():
    cache, stack, cfg = make(testmodels.unit_cube(size=10.0),
                             xy_gap=0.0, radius_quantum=0.1)
    a = cache.collision(20, 0.41)
    b = cache.collision(20, 0.50)
    assert a.bounds == pytest.approx(b.bounds)
    assert cache.quantize(0.41) == pytest.approx(0.5)
    assert cache.quantize(0.40) == pytest.approx(0.4)


def test_collision_cache_reuses_entries():
    cache, stack, cfg = make(testmodels.unit_cube(size=10.0))
    cache.collision(10, 1.0)
    misses = cache.stats.collision_misses
    for _ in range(20):
        cache.collision(10, 1.0)
        cache.collision(10, 0.95)     # 同じ量子化半径
    assert cache.stats.collision_misses == misses
    assert cache.stats.collision_hits >= 40


def test_collision_includes_layers_below_for_bottom_gap():
    """モデル上面の 1 層上でも、下側 z ギャップのために collision が残る。"""
    cfg_kwargs = dict(layer_height=0.2, xy_gap=0.0, bottom_z_gap=0.2)
    cache, stack, cfg = make(testmodels.unit_cube(size=10.0), **cfg_kwargs)
    top = stack.last_model_layer            # 49
    assert not cache.collision(top + 1, 0.1).is_empty
    assert not cache.collision(top + 2, 0.1).is_empty   # j-1-n_bot = top
    assert cache.collision(top + 3, 0.1).is_empty


def test_tip_collision_is_looser_above_than_branch():
    """tip 用 collision は 1 層分だけ上方向にゆるい。"""
    cache, stack, cfg = make(testmodels.floating_box(size=12.0, height=8.0),
                             layer_height=0.2, top_z_gap=0.2, xy_gap=0.4)
    m = stack.first_model_layer            # 40
    j = m - cfg.z_gap_layers_top           # 39
    branch = cache.collision(j, 0.4, kind="branch")
    tip = cache.collision(j, 0.4, kind="tip")
    assert not branch.is_empty             # レイヤ 40 のモデルが効く
    assert tip.is_empty                    # tip は上を見ない
    assert branch.covers(Point(6.0, 6.0))
    assert not tip.covers(Point(6.0, 6.0))


def test_reach_on_empty_plate_is_the_whole_world():
    cache, stack, cfg = make(testmodels.floating_box(size=12.0, height=8.0))
    reach0 = cache.reach(0, 0.4)
    assert reach0.area == pytest.approx(cache.world.area, rel=1e-9)


def test_reach_excludes_model_and_grows_slowly():
    cache, stack, cfg = make(testmodels.unit_cube(size=10.0),
                             layer_height=0.2, xy_gap=0.4,
                             branch_angle_max=40.0)
    r = 0.4
    for j in (0, 10, 30):
        reach = cache.reach(j, r)
        assert not reach.covers(Point(5.0, 5.0))      # モデル内部は不可
        assert reach.covers(Point(5.0, -3.0))         # 十分離れた外側は可


def test_reach_is_empty_under_a_full_overhead_block():
    """真上をモデルが完全に覆う位置は、そこから降りられても collision で消える。"""
    cache, stack, cfg = make(testmodels.unit_cube(size=10.0), xy_gap=0.4)
    reach = cache.reach(25, 0.4)
    assert not reach.covers(Point(5.0, 5.0))


def test_reach_shrinks_inside_a_narrow_pocket():
    """袋小路 (下が塞がっている場所) が reach から除かれること。

    obstructed_path: x in [-2,12] z in [0,12] のブロックの真上は、
    ブロック上面の直上レイヤでは降下できない。
    """
    cache, stack, cfg = make(testmodels.obstructed_path(), layer_height=0.2,
                             xy_gap=0.4, branch_angle_max=40.0)
    j = int(round(12.0 / 0.2)) + 2          # ブロック上面のすぐ上
    reach = cache.reach(j, 0.4)
    assert not reach.covers(Point(5.0, 5.0))     # ブロック直上は不可
    assert reach.covers(Point(20.0, 5.0))        # 横に外れれば可


def test_reach_is_monotone_in_radius():
    """半径が大きいほど reach は狭くなる。"""
    cache, stack, cfg = make(testmodels.unit_cube(size=10.0), xy_gap=0.4)
    small = cache.reach(20, 0.4)
    large = cache.reach(20, 2.0)
    assert large.area < small.area
    assert small.buffer(1e-6).covers(large)


def test_reach_never_intersects_collision():
    cache, stack, cfg = make(testmodels.bridge(), layer_height=0.4, xy_gap=0.4)
    for j in range(0, stack.n_layers, 5):
        reach = cache.reach(j, 0.6)
        col = cache.collision(j, 0.6)
        assert g2.area(g2.intersection(reach, col)) < 1e-9


def test_tip_allowed_region_for_floating_box():
    cache, stack, cfg = make(testmodels.floating_box(size=12.0, height=8.0),
                             layer_height=0.2, top_z_gap=0.2)
    m = stack.first_model_layer
    j = m - cfg.z_gap_layers_top
    allowed = cache.tip_allowed_by_layer([j])[j]
    # 何も遮るものが無いので板の真下は全部置ける
    assert allowed.covers(shapely.box(0.1, 0.1, 11.9, 11.9))


def test_debug_svgs(tmp_path):
    cache, stack, cfg = make(testmodels.bridge(), layer_height=0.8)
    dbg = DebugWriter(tmp_path)
    n = cache.write_debug(dbg, radius=0.6, every=4)
    assert n > 0
    assert (tmp_path / "collision" / "collision_stats.json").exists()
    files = sorted((tmp_path / "collision").glob("layer_*_collision_*.svg"))
    assert files and files[0].read_text(encoding="utf-8").startswith("<svg")
