"""STEP 4: tip サンプリングと coverage のテスト。"""

from __future__ import annotations

import math

import pytest
import shapely
from shapely.geometry import MultiPoint, Point, Polygon

from treesupport import geometry2d as g2
from treesupport import testmodels
from treesupport.config import SupportConfig
from treesupport.debug_io import DebugWriter
from treesupport.overhang import OverhangDetector
from treesupport.slicer import ModelSlicer
from treesupport.tips import ContactSampler, TipSet


def pipeline(mesh, **kwargs):
    cfg = SupportConfig(**kwargs)
    stack = ModelSlicer(cfg).slice(mesh)
    over = OverhangDetector(cfg).detect(stack)
    tips = ContactSampler(cfg).sample(stack, over)
    return stack, over, tips, cfg


def coverage_of(region, points, radius):
    if not points:
        return 0.0
    cover = MultiPoint([Point(p[0], p[1]) for p in points]).buffer(radius, quad_segs=16)
    return region.intersection(cover).area / region.area


# ----------------------------------------------------------------------
def test_square_region_is_fully_covered():
    cfg = SupportConfig(tip_spacing=2.5)
    sampler = ContactSampler(cfg)
    region = shapely.box(0.0, 0.0, 20.0, 20.0)
    pts = sampler.sample_region(region)
    assert len(pts) > 0
    assert coverage_of(region, pts, cfg.coverage_radius) == pytest.approx(1.0, abs=1e-3)


def test_thin_strip_is_covered():
    """格子だけでは漏れる細長い領域が、輪郭サンプリングで拾われること。"""
    cfg = SupportConfig(tip_spacing=2.5)
    sampler = ContactSampler(cfg)
    region = shapely.box(0.05, 0.05, 30.0, 0.45)   # 幅 0.4mm、格子点が乗らない位置
    pts = sampler.sample_region(region)
    assert len(pts) >= 10
    assert coverage_of(region, pts, cfg.coverage_radius) == pytest.approx(1.0, abs=1e-3)
    assert all(region.buffer(1e-6).covers(Point(p[0], p[1])) for p in pts)


def test_sharp_corner_gets_a_tip():
    """鋭角コーナーが tip から漏れないこと (CuraEngine #2037 対策)。"""
    cfg = SupportConfig(tip_spacing=4.0)
    sampler = ContactSampler(cfg)
    # 非常に鋭い三角形の先端 (30, 0.0)
    region = Polygon([(0.0, 0.0), (30.0, 0.0), (0.0, 6.0)])
    pts = sampler.sample_region(region)
    tip_apex = min(pts, key=lambda p: math.hypot(p[0] - 30.0, p[1] - 0.0))
    assert math.hypot(tip_apex[0] - 30.0, tip_apex[1]) < 1.5
    assert coverage_of(region, pts, cfg.coverage_radius) > 0.99


def test_isolated_small_region_gets_a_tip():
    cfg = SupportConfig(tip_spacing=5.0, min_overhang_area=0.01)
    sampler = ContactSampler(cfg)
    region = shapely.MultiPolygon([
        shapely.box(0.0, 0.0, 1.0, 1.0),
        shapely.box(40.0, 40.0, 40.6, 40.6),
    ])
    pts = sampler.sample_region(region)
    near_a = [p for p in pts if p[0] < 10]
    near_b = [p for p in pts if p[0] > 30]
    assert near_a and near_b


def test_grid_is_globally_aligned():
    """レイヤ間で格子が揃うこと (merge しやすさのため)。"""
    cfg = SupportConfig(tip_spacing=2.5, tip_min_distance_factor=0.5)
    sampler = ContactSampler(cfg)
    a = sampler._grid_points(shapely.box(0.0, 0.0, 10.0, 10.0))
    b = sampler._grid_points(shapely.box(1.3, 1.3, 8.8, 8.8))
    for x, y, _ in a + b:
        assert abs(x / 2.5 - round(x / 2.5)) < 1e-9
        assert abs(y / 2.5 - round(y / 2.5)) < 1e-9


def test_min_distance_is_respected():
    cfg = SupportConfig(tip_spacing=2.5, tip_min_distance_factor=0.5)
    sampler = ContactSampler(cfg)
    region = shapely.box(0.0, 0.0, 15.0, 15.0)
    pts = sampler.sample_region(region)
    # fill 段階は coverage を優先するため、格子・輪郭段階の点のみ検査
    base = [p for p in pts if p[2] != "fill"]
    for i, p in enumerate(base):
        for q in base[i + 1:]:
            assert math.hypot(p[0] - q[0], p[1] - q[1]) >= cfg.tip_min_distance - 1e-9


# ----------------------------------------------------------------------
def test_floating_box_tip_layer_and_z_gap():
    stack, over, tipset, cfg = pipeline(
        testmodels.floating_box(size=12.0, thickness=4.0, height=8.0),
        layer_height=0.2, top_z_gap=0.2, tip_spacing=2.5,
    )
    assert len(tipset) > 0
    m = stack.first_model_layer          # 40 (z=8.0)
    j = m - cfg.z_gap_layers_top         # 39
    assert set(tipset.by_layer) == {j}
    tip = tipset.tips[0]
    assert tip.z == pytest.approx(j * 0.2)
    # tip 上端とモデル下面の距離
    assert m * 0.2 - tip.z == pytest.approx(cfg.actual_top_z_gap)
    assert m * 0.2 - tip.z >= cfg.top_z_gap - 1e-12


def test_larger_z_gap_moves_tips_further_down():
    stack, over, tipset, cfg = pipeline(
        testmodels.floating_box(height=8.0), layer_height=0.2, top_z_gap=0.5,
    )
    m = stack.first_model_layer
    assert cfg.z_gap_layers_top == 3
    assert set(tipset.by_layer) == {m - 3}
    assert cfg.actual_top_z_gap == pytest.approx(0.6)


def test_coverage_ratio_of_cantilever():
    _, _, tipset, cfg = pipeline(testmodels.cantilever(), tip_spacing=2.5)
    assert tipset.effective_coverage_ratio == pytest.approx(1.0)
    assert tipset.coverage_ratio > 0.999
    assert tipset.demand_area > 100.0
    assert tipset.unsupportable_area == pytest.approx(0.0)


def test_overhang_corner_coverage():
    _, _, tipset, cfg = pipeline(testmodels.overhang_corner(), tip_spacing=2.5)
    # min_overhang_area 未満のかけらを除けば完全被覆
    assert tipset.effective_coverage_ratio == pytest.approx(1.0)
    assert tipset.coverage_ratio > 0.998
    assert any(t.source == "corner" for t in tipset.tips)


def test_tip_too_close_to_plate_is_unsupportable():
    """Z ギャップを確保できない高さの overhang は unsupportable として記録される。"""
    mesh = testmodels.floating_box(size=4.0, thickness=1.0, height=0.2)
    _, _, tipset, cfg = pipeline(mesh, layer_height=0.2, top_z_gap=0.6)
    assert len(tipset) == 0
    assert tipset.unsupportable_area == pytest.approx(16.0)


def test_allowed_region_restricts_tips():
    cfg = SupportConfig(layer_height=0.2, tip_spacing=2.0)
    mesh = testmodels.floating_box(size=12.0, thickness=4.0, height=8.0)
    stack = ModelSlicer(cfg).slice(mesh)
    over = OverhangDetector(cfg).detect(stack)
    j = stack.first_model_layer - cfg.z_gap_layers_top
    allowed = {j: shapely.box(0.0, 0.0, 6.0, 12.0)}
    tipset = ContactSampler(cfg).sample(stack, over, allowed_by_layer=allowed)
    assert all(t.x <= 6.0 + 1e-9 for t in tipset.tips)
    assert tipset.unsupportable_area == pytest.approx(72.0, rel=1e-3)
    assert tipset.coverage_ratio < 0.6


def test_debug_artifacts(tmp_path):
    cfg = SupportConfig(layer_height=0.4, tip_spacing=2.5)
    mesh = testmodels.bridge()
    stack = ModelSlicer(cfg).slice(mesh)
    over = OverhangDetector(cfg).detect(stack)
    sampler = ContactSampler(cfg)
    tipset = sampler.sample(stack, over)
    dbg = DebugWriter(tmp_path)
    n = sampler.write_debug(stack, over, tipset, dbg)
    assert n > 0
    assert (tmp_path / "tips" / "tips.json").exists()
    assert sorted((tmp_path / "tips").glob("layer_*_tips.svg"))
