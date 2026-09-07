import json

import numpy as np
import pytest
import trimesh

from treesupport.application import build_bundle, export_bundle
from treesupport.cli import main
from treesupport.config import SupportConfig
from treesupport.infill import InfillConfig, generate_infill, lattice_field
from treesupport.testmodels import floating_box, unit_cube, cantilever


@pytest.mark.parametrize("pattern", ["gyroid", "diamond", "cubic"])
def test_infill_preserves_exterior_and_real_internal_voids(pattern):
    model = unit_cube(10, origin=(13, -7, 2))
    original_vertices = model.vertices.copy()
    config = InfillConfig(pattern=pattern, cell_size=6)
    result = generate_infill(model, config)
    assert result.printable.is_volume
    assert result.infill.is_volume
    np.testing.assert_allclose(result.printable.bounds, model.bounds, atol=1e-5)
    np.testing.assert_array_equal(model.vertices, original_vertices)
    assert 0 < result.printable.volume < model.volume * .95
    # Query points well away from the sampled boundaries, so these assertions
    # test actual geometry rather than reproducing marching-cubes triangles.
    points = np.random.default_rng(30).uniform(model.bounds[0]+2, model.bounds[1]-2, (300, 3))
    field = lattice_field(points, config)
    void = points[field < -config.pitch]
    material = points[field > config.pitch]
    assert len(void) > 5
    assert not result.printable.contains(void).any()
    if len(material):
        assert result.printable.contains(material).all()
    shell_points = np.array([[13.3, -2, 7], [22.7, -2, 7], [18, -6.7, 7], [18, -2, 2.3]])
    assert result.printable.contains(shell_points).all()
    assert model.contains(result.infill.vertices).all()


def test_infill_keeps_existing_hole():
    outer = unit_cube(12)
    hole = trimesh.creation.cylinder(radius=2, height=16, sections=24)
    hole.apply_translation([6, 6, 6])
    model = trimesh.boolean.difference([outer, hole], engine="manifold")
    result = generate_infill(model, InfillConfig())
    assert result.printable.is_volume
    assert not result.printable.contains([[6, 6, 6], [6, 6, 2], [6, 6, 10]]).any()
    np.testing.assert_allclose(result.printable.bounds, model.bounds, atol=1e-5)


@pytest.mark.parametrize("kwargs", [
    {"pitch": 0}, {"cell_size": float("nan")}, {"wall_thickness": float("inf")},
    {"shell_thickness": -1}, {"pattern": "lightning"}, {"pitch": .6}, {"cell_size": 2},
])
def test_invalid_infill_parameters(kwargs):
    with pytest.raises(ValueError):
        InfillConfig(**kwargs)


def test_grid_limit_and_open_mesh_rejected():
    with pytest.raises(ValueError, match="Grid needs"):
        generate_infill(unit_cube(10), InfillConfig(max_grid_points=100))
    model = unit_cube(10)
    model.update_faces(np.arange(len(model.faces)-1))
    with pytest.raises(ValueError, match="closed"):
        generate_infill(model, InfillConfig())


@pytest.mark.parametrize("model", [floating_box(size=8, height=4, thickness=2),
                                    cantilever(column=4, arm=6, depth=6, height=6)])
def test_support_boolean_export_roundtrip_and_gap(model, tmp_path):
    original = model.vertices.copy()
    cfg = SupportConfig(layer_height=.4, tip_spacing=4, union_mode="boolean")
    bundle = build_bundle(model, "support", cfg)
    support = bundle.meshes["tree_support.stl"]
    assert bundle.report["metrics"]["mesh_union_ok"]
    assert support.is_volume
    assert support.bounds[0, 2] >= -1e-6
    underside = 4 if model.extents[0] == 8 else 6
    assert support.bounds[1, 2] <= underside-cfg.actual_top_z_gap+1e-5
    assert trimesh.boolean.intersection([model, support], engine="manifold").volume == 0
    np.testing.assert_array_equal(model.vertices, original)
    folder = export_bundle(bundle, tmp_path)
    again = export_bundle(bundle, tmp_path)
    assert again != folder
    loaded = trimesh.load_mesh(folder / "tree_support.stl")
    assert loaded.is_volume
    np.testing.assert_allclose(loaded.bounds, support.bounds, atol=1e-5)
    assert json.loads((folder/"report.json").read_text(encoding="utf-8"))["validation"]["passed"]


def test_no_support_is_a_reported_empty_result(tmp_path):
    bundle = build_bundle(unit_cube(4), "support")
    assert bundle.report["has_support"] is False
    assert list(bundle.meshes) == ["model.stl"]
    folder = export_bundle(bundle, tmp_path)
    assert not (folder / "tree_support.stl").exists()


def test_cli_infill_and_invalid_input(tmp_path, capsys):
    source = tmp_path / "input.stl"
    unit_cube(8).export(source)
    assert main(["infill", str(source), "--pattern", "cubic", "-o", str(tmp_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    path = next(tmp_path.glob("infill-*/model_with_infill.stl"))
    assert "model_with_infill.stl" in result["files"]
    assert trimesh.load_mesh(path).is_volume
    assert main(["support", str(tmp_path / "missing.stl")]) == 1


def test_nonfinite_support_values_rejected():
    with pytest.raises(ValueError, match="finite"):
        SupportConfig(top_z_gap=float("nan"))
