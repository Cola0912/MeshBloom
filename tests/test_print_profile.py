import json
import numpy as np
import pytest
import trimesh

from treesupport.application import build_bundle, export_bundle
from treesupport.config import SupportConfig
from treesupport.graph import TreeGraph
from treesupport.infill import InfillConfig, generate_infill
from treesupport.meshbuilder import MeshBuilder
from treesupport.print_profile import PrintProfile
from treesupport.radius import RadiusSolver
from treesupport.testmodels import floating_box, unit_cube
from treesupport.ui_settings import validate_settings


def test_nozzle_and_independent_line_width_size_real_infill():
    small = PrintProfile(.4, .4)
    big = PrintProfile(.6, .6)
    assert PrintProfile(.6, 0).width == pytest.approx(.675)
    assert PrintProfile(.4, .6).dimensions() == {**big.dimensions(), "contact_height": .8}
    results = []
    for profile in (small, big):
        dims = profile.dimensions()
        cfg = InfillConfig(pattern="cubic", cell_size=8, nozzle_diameter=profile.nozzle_diameter,
                           line_width=profile.line_width, wall_thickness=dims["wall_thickness"],
                           shell_thickness=dims["shell_thickness"], pitch=.4)
        results.append(generate_infill(unit_cube(10), cfg))
    assert results[1].infill.volume > results[0].infill.volume
    assert results[1].printable.volume > results[0].printable.volume


@pytest.mark.parametrize("shape", ["flat", "tapered", "rounded"])
def test_contact_shape_stays_inside_safe_tube_and_has_requested_contact_diameter(shape):
    graph = TreeGraph(.2)
    root = graph.new_node(0, 1., position=(0, 0))
    tip = graph.new_node(20, 1., position=(0, 0))
    graph.link(root.id, tip.id)
    graph.root_ids = [root.id]
    cfg = SupportConfig(tip_diameter=2, contact_diameter=.6, contact_height=1.2, contact_shape=shape)
    mesh = MeshBuilder(cfg).build(graph)
    assert mesh.is_volume
    assert mesh.bounds[0, 2] == 0 and mesh.bounds[1, 2] == 4
    cap = mesh.vertices[np.isclose(mesh.vertices[:, 2], 4)]
    assert np.linalg.norm(cap[:, :2], axis=1).max() == pytest.approx(.3)
    assert np.linalg.norm(mesh.vertices[:, :2], axis=1).max() <= 1+1e-9
    middle = mesh.section([0, 0, 1], [0, 0, 3.4])
    radius = np.linalg.norm(middle.vertices[:, :2], axis=1).max()
    assert radius == pytest.approx({"flat": .3, "tapered": .65, "rounded": .3+.7*np.sqrt(.75)}[shape], abs=.002)


@pytest.mark.parametrize("gap,shape", [(.2, "flat"), (.6, "tapered"), (.4, "rounded")])
def test_organic_support_roundtrip_keeps_gap_and_no_model_intersection(gap, shape, tmp_path):
    model = floating_box(size=6, height=4, thickness=2)
    cfg = SupportConfig(layer_height=.4, tip_spacing=4, union_mode="boolean", branch_profile="organic",
                        contact_shape=shape, contact_diameter=.4, top_z_gap=gap)
    bundle = build_bundle(model, "support", cfg)
    support = bundle.meshes["tree_support.stl"]
    assert support.is_volume
    assert support.bounds[1, 2] <= 4-cfg.actual_top_z_gap+1e-5
    assert support.bounds[0, 2] >= -1e-6
    assert trimesh.boolean.intersection([model, support], engine="manifold").volume == 0
    folder = export_bundle(bundle, tmp_path)
    assert trimesh.load_mesh(folder / "tree_support.stl").is_volume
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    assert report["config"]["contact_shape"] == shape
    assert report["config"]["derived"]["effective_line_width"] == .4
    assert "ライン幅 0.4 mm" in (folder / "Simplify3D-v5.txt").read_text(encoding="utf-8-sig")


def test_organic_radius_grows_and_keeps_merged_radius():
    cfg = SupportConfig(branch_profile="organic", branch_diameter=2.4)
    solver = RadiusSolver(cfg)
    radius = cfg.tip_radius
    for distance in range(1, 60):
        next_radius = solver.grow(radius, distance)
        assert radius <= next_radius <= radius+cfg.effective_line_width/2+1e-9
        radius = next_radius
    assert radius > cfg.branch_diameter/2
    assert solver.grow(3, 2) >= 3


@pytest.mark.parametrize("kwargs", [{"nozzle_diameter": 0}, {"line_width": -1},
                                  {"nozzle_diameter": float("nan")}, {"line_width": float("inf")}])
def test_invalid_print_profile(kwargs):
    with pytest.raises(ValueError):
        PrintProfile(**kwargs)


def test_invalid_contact_and_old_profile_migration():
    assert SupportConfig(.32).layer_height == .32
    assert InfillConfig("cubic").pattern == "cubic"
    with pytest.raises(ValueError):
        SupportConfig(contact_diameter=.2)
    with pytest.raises(ValueError):
        SupportConfig(contact_diameter=2)
    legacy = {"layer_height": ".2", "top_z_gap": ".2", "xy_gap": ".4", "tip_spacing": "2.5",
              "tip_diameter": ".8", "overhang_angle": "45", "branch_angle_max": "40",
              "cell_size": "8", "wall_thickness": "1.2", "shell_thickness": "1.2", "pitch": ".4"}
    data = validate_settings({"format": "meshbloom-settings-v1", "mode": "support", "pattern": "gyroid", "values": legacy})
    assert data["values"]["branch_profile"] == "linear"
    assert data["values"]["contact_diameter"] == "0"
    assert data["values"]["wall_thickness"] == "1.2"
