"""STEP 2: レイヤスライスのテスト。"""

from __future__ import annotations

import math

import pytest

from treesupport import geometry2d as g2
from treesupport import testmodels
from treesupport.config import SupportConfig
from treesupport.debug_io import DebugWriter, SvgLayer
from treesupport.slicer import ModelSlicer


def make_stack(mesh, **kwargs):
    cfg = SupportConfig(**kwargs)
    return ModelSlicer(cfg).slice(mesh), cfg


def test_cube_slices_are_exact_squares():
    mesh = testmodels.unit_cube(size=10.0, origin=(0.0, 0.0, 0.0))
    stack, cfg = make_stack(mesh, layer_height=0.2)

    assert stack.n_layers == 50
    assert stack.first_model_layer == 0
    assert stack.last_model_layer == 49

    for k in (0, 1, 25, 49):
        s = stack[k]
        assert not s.is_empty
        assert s.area == pytest.approx(100.0, rel=1e-9)
        assert s.bounds == pytest.approx((0.0, 0.0, 10.0, 10.0))


def test_slice_z_convention():
    stack, cfg = make_stack(testmodels.unit_cube(size=1.0), layer_height=0.25)
    assert stack.z_bottom(0) == pytest.approx(0.0)
    assert stack.z_mid(0) == pytest.approx(0.125)
    assert stack.z_top(0) == pytest.approx(0.25)
    assert stack.z_bottom(4) == pytest.approx(1.0)


def test_offset_cube_keeps_world_coordinates():
    mesh = testmodels.unit_cube(size=6.0, origin=(13.0, -7.5, 0.0))
    stack, _ = make_stack(mesh, layer_height=0.2)
    s = stack[10]
    assert s.bounds == pytest.approx((13.0, -7.5, 19.0, -1.5))


def test_floating_model_has_empty_lower_layers():
    mesh = testmodels.floating_box(size=12.0, thickness=4.0, height=8.0)
    stack, cfg = make_stack(mesh, layer_height=0.2)

    # z<8 のレイヤは空、z>=8 は 12x12
    assert stack[0].is_empty
    assert stack[39].is_empty          # z in [7.8, 8.0]
    assert not stack[40].is_empty      # z in [8.0, 8.2]
    assert stack[40].area == pytest.approx(144.0)
    assert stack.first_model_layer == 40
    assert stack.last_model_layer == 59
    assert stack.n_layers == 60


def test_cavity_slice_has_hole():
    mesh = testmodels.cavity(size=16.0, wall=3.0)
    stack, _ = make_stack(mesh, layer_height=0.2)
    s = stack[40]  # z ~ 8.1 -> 空洞の内部
    polys = g2.polygons(s)
    assert len(polys) == 1
    assert len(polys[0].interiors) == 1
    assert s.area == pytest.approx(16.0 * 16.0 - 10.0 * 10.0, rel=1e-9)


def test_slope_area_decreases_linearly():
    mesh = testmodels.slope_45(size=16.0, depth=10.0)
    stack, cfg = make_stack(mesh, layer_height=0.2)
    # 断面は三角形 (0,0)-(size,size)-(0,size) を高さ z で切ったもの。
    # 高さ z における x の範囲は [0, z] なので 面積 = z * depth。
    # x=z の面が鉛直から 45 度の下向き面になる。
    # slices[k] はスラブが掃く領域 (下端/中央/上端の和) なので、
    # 上に向かって広がるこの形状では上端サンプルの面積になる。
    eps = 0.2 * 1e-3
    for k in (5, 30, 60):
        z_top = stack.z_top(k) - eps
        assert stack[k].area == pytest.approx(z_top * 10.0, rel=1e-6)


def test_slice_is_the_union_over_the_slab():
    """1 平面サンプリングに落とすと、スラブ中央の断面そのものになること。"""
    mesh = testmodels.slope_45(size=16.0, depth=10.0)
    stack, cfg = make_stack(mesh, layer_height=0.2, slice_samples_per_layer=1)
    for k in (5, 30, 60):
        assert stack[k].area == pytest.approx(stack.z_mid(k) * 10.0, rel=1e-6)


def test_model_below_plate_rejected():
    mesh = testmodels.unit_cube(size=4.0, origin=(0.0, 0.0, -1.0))
    cfg = SupportConfig()
    with pytest.raises(ValueError, match="below the build plate"):
        ModelSlicer(cfg).slice(mesh)


def test_layer_count_rounds_up():
    mesh = testmodels.unit_cube(size=1.05, origin=(0.0, 0.0, 0.0))
    stack, _ = make_stack(mesh, layer_height=0.2)
    assert stack.n_layers == math.ceil(1.05 / 0.2)


def test_out_of_range_access_is_empty():
    stack, _ = make_stack(testmodels.unit_cube(size=2.0), layer_height=0.2)
    assert stack[-1].is_empty
    assert stack[10_000].is_empty


def test_debug_svg_written(tmp_path):
    mesh = testmodels.bridge()
    stack, _ = make_stack(mesh, layer_height=0.4)
    dbg = DebugWriter(tmp_path)
    for k in range(0, stack.n_layers, 10):
        p = dbg.write_svg(
            "slices",
            f"layer_{k:04d}_model.svg",
            [SvgLayer("model", stack[k])],
            bounds=stack.xy_bounds,
            title=f"layer {k} z={stack.z_mid(k):.3f}",
        )
        assert p is not None and p.exists()
        text = p.read_text(encoding="utf-8")
        assert text.startswith("<svg")
        assert "path" in text or stack[k].is_empty

    assert (tmp_path / "slices").is_dir()


def test_debug_writer_disabled_is_noop():
    dbg = DebugWriter(None)
    assert dbg.write_svg("slices", "x.svg", []) is None
    assert dbg.write_json("slices", "x.json", {}) is None
    assert not dbg.active
