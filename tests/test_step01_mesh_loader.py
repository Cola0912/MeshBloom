"""STEP 1: メッシュ読み込み・診断・テスト形状のテスト。"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from treesupport import testmodels
from treesupport.mesh_loader import MeshLoader


def test_unit_cube_diagnostics():
    mesh = testmodels.unit_cube(size=10.0, origin=(1.0, 2.0, 3.0))
    loaded = MeshLoader().from_trimesh(mesh)
    d = loaded.diagnostics

    assert d.triangle_count == 12
    assert d.is_watertight
    assert d.is_winding_consistent
    assert d.open_edge_count == 0
    assert d.volume == pytest.approx(1000.0)
    assert d.bounds_min == pytest.approx((1.0, 2.0, 3.0))
    assert d.bounds_max == pytest.approx((11.0, 12.0, 13.0))
    assert d.extents == pytest.approx((10.0, 10.0, 10.0))


def test_loader_does_not_move_the_model(tmp_path):
    """読み込みで座標が変わらないこと (T_model == T_support の前提)。"""
    mesh = testmodels.unit_cube(size=5.0, origin=(37.0, -11.5, 2.25))
    path = tmp_path / "cube.stl"
    mesh.export(path)

    loaded = MeshLoader().load(path)
    assert loaded.diagnostics.bounds_min == pytest.approx((37.0, -11.5, 2.25))
    assert loaded.diagnostics.bounds_max == pytest.approx((42.0, -6.5, 7.25))


def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        MeshLoader().load("does_not_exist_1234.stl")


@pytest.mark.parametrize("name", sorted(testmodels.ALL_MODELS))
def test_all_test_models_are_valid_solids(name):
    mesh = testmodels.ALL_MODELS[name]()
    loaded = MeshLoader().from_trimesh(mesh, source=name)
    d = loaded.diagnostics

    assert d.triangle_count > 0, name
    assert d.is_watertight, f"{name} is not watertight"
    assert d.is_winding_consistent, f"{name} winding inconsistent"
    assert d.open_edge_count == 0, f"{name} has open edges"
    assert d.volume > 0.0, f"{name} has non-positive volume"
    assert d.bounds_min[2] >= -1e-9, f"{name} extends below the build plate"


def test_floating_box_is_above_plate():
    mesh = testmodels.floating_box(size=12.0, thickness=4.0, height=8.0)
    assert mesh.bounds[0][2] == pytest.approx(8.0)
    assert mesh.bounds[1][2] == pytest.approx(12.0)


def test_slope_45_geometry():
    """45 度斜面の断面が想定通りか (XZ の三角形)。"""
    mesh = testmodels.slope_45(size=16.0, depth=10.0)
    assert mesh.bounds[0] == pytest.approx([0.0, 0.0, 0.0])
    assert mesh.bounds[1] == pytest.approx([16.0, 10.0, 16.0])
    # 断面積 = 16*16/2、体積 = 断面積 * 奥行
    assert mesh.volume == pytest.approx(16.0 * 16.0 / 2.0 * 10.0, rel=1e-6)


def test_cavity_has_two_shells():
    mesh = testmodels.cavity(size=16.0, wall=3.0)
    loaded = MeshLoader().from_trimesh(mesh)
    assert loaded.diagnostics.body_count == 2
    # 外形 - 空洞
    expected = 16.0 ** 3 - 10.0 ** 3
    assert loaded.diagnostics.volume == pytest.approx(expected, rel=1e-6)


def test_merge_cluster_has_four_pads():
    mesh = testmodels.merge_cluster()
    loaded = MeshLoader().from_trimesh(mesh)
    assert loaded.diagnostics.body_count == 4


def test_repair_fixes_flipped_normals():
    mesh = testmodels.unit_cube(size=4.0)
    broken = mesh.copy()
    broken.invert()
    repaired = MeshLoader(repair=True).from_trimesh(broken)
    assert repaired.diagnostics.volume > 0.0
    assert repaired.diagnostics.is_winding_consistent


def test_roundtrip_stl_binary(tmp_path):
    mesh = testmodels.bridge()
    path = tmp_path / "bridge.stl"
    mesh.export(path)
    assert path.stat().st_size > 84  # binary STL header + count

    loaded = MeshLoader().load(path)
    assert loaded.diagnostics.is_watertight
    assert np.allclose(loaded.bounds, mesh.bounds, atol=1e-4)


def test_scene_is_merged(tmp_path):
    scene = trimesh.Scene()
    scene.add_geometry(testmodels.unit_cube(size=3.0, origin=(0, 0, 0)))
    scene.add_geometry(testmodels.unit_cube(size=3.0, origin=(10, 0, 0)))
    path = tmp_path / "scene.obj"
    scene.export(path)

    loaded = MeshLoader().load(path)
    assert loaded.diagnostics.triangle_count == 24
