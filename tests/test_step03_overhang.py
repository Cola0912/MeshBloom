"""STEP 3: オーバーハング検出のテスト。

テストモデル: 垂直壁 / 45 度斜面 / 水平片持ち板 / 空中直方体 / bridge。
"""

from __future__ import annotations

import pytest

from treesupport import testmodels
from treesupport.config import SupportConfig
from treesupport.debug_io import DebugWriter
from treesupport.overhang import OverhangDetector
from treesupport.slicer import ModelSlicer


def detect(mesh, **kwargs):
    cfg = SupportConfig(**kwargs)
    stack = ModelSlicer(cfg).slice(mesh)
    result = OverhangDetector(cfg).detect(stack)
    return stack, result, cfg


def test_allowed_offset_formula():
    cfg = SupportConfig(layer_height=0.2, overhang_angle=45.0)
    assert cfg.overhang_allowed_offset == pytest.approx(0.2)
    cfg = SupportConfig(layer_height=0.2, overhang_angle=60.0)
    assert cfg.overhang_allowed_offset == pytest.approx(0.2 * 1.7320508, rel=1e-6)


def test_vertical_wall_needs_no_support():
    _, result, _ = detect(testmodels.vertical_wall())
    assert result.total_area == pytest.approx(0.0)
    assert result.layers_with_demand == []


def test_plate_on_build_plate_needs_no_support():
    _, result, _ = detect(testmodels.horizontal_plate_on_plate())
    assert result.total_area == pytest.approx(0.0)


def test_slope_at_exactly_45_is_self_supporting():
    """45 度斜面 + overhang_angle=45 -> demand ほぼゼロ。"""
    _, result, _ = detect(testmodels.slope_45(), layer_height=0.2, overhang_angle=45.0)
    assert result.total_area < 1e-6


def test_slope_at_45_needs_support_when_threshold_is_40():
    """しきい値を厳しくすると 45 度斜面はサポート対象になる。"""
    stack, result, cfg = detect(
        testmodels.slope_45(size=16.0, depth=10.0),
        layer_height=0.2, overhang_angle=40.0, min_overhang_area=0.0,
    )
    assert result.total_area > 0.0
    # 1 レイヤあたりの張り出し 0.2 - allowed(=0.2*tan40=0.1678) を幅 10 で
    per_layer = (0.2 - cfg.overhang_allowed_offset) * 10.0
    k = 40
    assert stack[k].area > 0
    assert result[k].area == pytest.approx(per_layer, rel=1e-3)


def test_floating_box_demand_is_full_footprint():
    stack, result, _ = detect(
        testmodels.floating_box(size=12.0, thickness=4.0, height=8.0),
        layer_height=0.2,
    )
    first = stack.first_model_layer
    assert first == 40
    assert result[first].area == pytest.approx(144.0)
    # 2 層目以降は真下に同じ断面があるので demand なし
    assert result[first + 1].is_empty
    assert result.layers_with_demand == [first]


def test_cantilever_demand_matches_arm():
    column, arm, depth, height = 8.0, 22.0, 10.0, 16.0
    stack, result, cfg = detect(
        testmodels.cantilever(column=column, arm=arm, depth=depth, height=height),
        layer_height=0.2, overhang_angle=45.0,
    )
    k = int(round(height / 0.2))          # 板の最初のレイヤ
    assert not result[k].is_empty
    expected = (arm - cfg.overhang_allowed_offset) * depth
    assert result[k].area == pytest.approx(expected, rel=1e-3)
    minx, miny, maxx, maxy = result[k].bounds
    assert minx == pytest.approx(column + cfg.overhang_allowed_offset, abs=1e-6)
    assert maxx == pytest.approx(column + arm)


def test_bridge_demand_is_the_span():
    pillar, span, depth, height = 6.0, 20.0, 8.0, 14.0
    stack, result, cfg = detect(
        testmodels.bridge(pillar=pillar, span=span, depth=depth, height=height),
        layer_height=0.2, overhang_angle=45.0,
    )
    k = int(round(height / 0.2))
    expected = (span - 2 * cfg.overhang_allowed_offset) * depth
    assert result[k].area == pytest.approx(expected, rel=1e-3)
    minx, _, maxx, _ = result[k].bounds
    assert minx == pytest.approx(pillar + cfg.overhang_allowed_offset, abs=1e-6)
    assert maxx == pytest.approx(pillar + span - cfg.overhang_allowed_offset, abs=1e-6)


def test_overhang_corner_keeps_sharp_corners():
    stack, result, _ = detect(testmodels.overhang_corner(), layer_height=0.2)
    k = stack.first_model_layer
    demand = result[k]
    assert not demand.is_empty
    # L 字全体が浮いているので断面と一致する
    assert demand.area == pytest.approx(stack[k].area, rel=1e-9)


def test_arch_has_demand_near_apex_only():
    stack, result, cfg = detect(testmodels.arch(), layer_height=0.2, overhang_angle=45.0)
    layers = result.layers_with_demand
    assert layers, "arch should require some support near the apex"
    # アーチの立ち上がり (z が小さい領域) では自己支持
    assert min(layers) > 0.5 * (9.0 / 0.2)


def test_cavity_internal_overhang_detected():
    stack, result, cfg = detect(testmodels.cavity(size=16.0, wall=3.0), layer_height=0.2)
    # 空洞の天井 z=13 のレイヤに内部オーバーハングが出る。
    # 天井の外周 allowed_offset 分は壁が支えるので、その分だけ内側に縮む。
    k = int(round(13.0 / 0.2))
    assert not result[k].is_empty
    side = 10.0 - 2.0 * cfg.overhang_allowed_offset
    assert result[k].area == pytest.approx(side * side, rel=1e-3)


def test_min_overhang_area_filter():
    mesh = testmodels.floating_box(size=0.5, thickness=1.0, height=4.0)
    _, result_keep, _ = detect(mesh, min_overhang_area=0.0)
    _, result_drop, _ = detect(mesh, min_overhang_area=1.0)
    assert result_keep.total_area == pytest.approx(0.25)
    assert result_drop.total_area == pytest.approx(0.0)
    assert result_drop.dropped_area == pytest.approx(0.25)


def test_debug_svgs_are_written(tmp_path):
    stack, result, cfg = detect(testmodels.cantilever(), layer_height=0.4)
    dbg = DebugWriter(tmp_path)
    n = OverhangDetector(cfg).write_debug(stack, result, dbg, every=5)
    assert n > 0
    files = sorted((tmp_path / "overhang").glob("layer_*_overhang.svg"))
    assert files
    assert (tmp_path / "overhang" / "overhang_stats.json").exists()
    assert files[0].read_text(encoding="utf-8").startswith("<svg")
