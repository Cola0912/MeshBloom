"""Import the bundled user model without changing its source or valid surfaces."""
import hashlib
from importlib.resources import files

import numpy as np
import pytest
import trimesh

from treesupport.application import load_model
from treesupport.mesh_loader import MeshLoader
from treesupport.sample_models import load_benchy


def test_bundled_benchy_preserves_source_and_coordinates():
    resource = files("treesupport").joinpath("assets", "3DBenchy.stl")
    source = resource.read_bytes()
    assert hashlib.sha256(source).hexdigest() == "6ab57f1c3f8e86bc3cbd302c6fa6270acf06277c6335454e922419c25d42e97e"
    raw = trimesh.load_mesh(resource, process=True)
    model = load_benchy()
    assert len(raw.faces) == 225706
    assert len(model.faces) == 225154
    assert model.metadata["meshbloom_removed_collapsed_faces"] == 552
    assert model.is_volume
    np.testing.assert_array_equal(model.vertices, raw.vertices)
    np.testing.assert_array_equal(model.bounds, raw.bounds)
    assert model.extents == pytest.approx([60.001, 31.004, 48], abs=.001)
    assert resource.read_bytes() == source


def test_discard_collapsed_faces_without_mutating_input():
    cube = trimesh.creation.box()
    broken = trimesh.Trimesh(cube.vertices, np.vstack([cube.faces, [0, 0, 1]]), process=False)
    before = broken.faces.copy()
    result = MeshLoader().from_trimesh(broken).mesh
    assert result.is_volume
    assert len(result.faces) == 12
    np.testing.assert_array_equal(broken.faces, before)
    np.testing.assert_array_equal(result.vertices, broken.vertices)


def test_thin_valid_faces_are_not_discarded():
    mesh = trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [.5, 1e-10, 0]], [[0, 1, 2]], process=False)
    result = MeshLoader().from_trimesh(mesh)
    assert result.diagnostics.triangle_count == 1
    assert result.diagnostics.area > 0


def test_only_collapsed_triangles_rejected():
    mesh = trimesh.Trimesh([[0, 0, 0], [1, 0, 0]], [[0, 0, 1]], process=False)
    with pytest.raises(ValueError, match="only collapsed"):
        MeshLoader().from_trimesh(mesh)


def test_import_does_not_silently_fill_holes(tmp_path):
    mesh = trimesh.creation.box()
    mesh.update_faces(np.arange(len(mesh.faces) - 1))
    path = tmp_path / "open.stl"
    mesh.export(path)
    with pytest.raises(ValueError, match="閉じたソリッド"):
        load_model(path)
