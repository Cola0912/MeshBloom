"""Exercise the real Tk event loop and worker handoff without interactive dialogs."""
import time
import tkinter as tk

import pytest

from treesupport import gui
from treesupport.testmodels import unit_cube


@pytest.fixture(scope="module")
def tk_owner():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display is unavailable: {exc}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def studio(monkeypatch, tk_owner):
    # Keep one Tcl interpreter per process and isolate each app in a Toplevel.
    # Repeated Tk creation/destruction can race Tcl cleanup after worker tests.
    root = tk.Toplevel(tk_owner)
    root.withdraw()
    errors = []
    tk_owner.report_callback_exception = lambda *a: errors.append(str(a))
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *a, **k: errors.append(str(a)))
    app = gui.Studio(root)
    root.update()
    yield app, errors
    if not app.closed:
        app.close()
    assert not errors, errors


def pump(app, timeout=10):
    deadline = time.monotonic() + timeout
    while app.busy and time.monotonic() < deadline:
        app.root.update()
        time.sleep(.02)
    app.root.update()
    assert not app.busy


def test_pose_reset_and_plate_sections(studio):
    import numpy as np
    app, _ = studio
    model = unit_cube(4, origin=(2, 3, 8))
    original = model.vertices.copy()
    app.set_model(model, "floating")
    assert float(app.zslider.cget("from")) == 0
    app.transform_model("X")
    app.transform_model("bed")
    assert app.model.bounds[0, 2] == pytest.approx(0)
    app.transform_model("reset")
    np.testing.assert_allclose(app.model.vertices, original)
    np.testing.assert_allclose(model.vertices, original)
    app.z.set("bad")
    app._draw_slice()
    assert app.z.get() == pytest.approx(6)
    for mode in ("3D", "断面", "3D + 断面"):
        app.view_mode.set(mode)
        app.draw()
        app.root.update()
        assert (app.viewport.winfo_manager() == "") == (mode == "断面")
        assert (app.ax2d is None) == (mode == "3D")


def test_cancel_discards_completed_work_and_allows_retry(studio, monkeypatch):
    import threading
    app, _ = studio
    app.set_model(unit_cube(4), "cube")
    release = threading.Event()
    started = threading.Event()
    called = []

    def job():
        started.set()
        release.wait(5)
        return "result"
    app._work(job, called.append, cancellable=True)
    assert started.wait(2)
    app.cancel()
    release.set()
    pump(app)
    assert called == []
    assert app.bundle is None
    assert app.export_button.instate(["disabled"])
    assert not app.generate_button.instate(["disabled"])
    app._work(lambda: "second", called.append, cancellable=True)
    pump(app)
    assert called == ["second"]


def test_settings_load_is_atomic_and_invalidates_old_output(studio, tmp_path, monkeypatch):
    import json
    from treesupport.ui_settings import save_settings
    app, errors = studio
    app.set_model(unit_cube(4), "cube")
    values = {k: v.get() for k, v in app.values.items()}
    values["cell_size"] = "12"
    path = tmp_path / "profile.json"
    save_settings(path, values, "infill", "diamond")
    monkeypatch.setattr(gui.filedialog, "askopenfilename", lambda **k: str(path))
    app.load_profile()
    app.root.update()
    assert app._mode() == "infill" and app.pattern.get() == "diamond"
    assert app.values["cell_size"].get() == "12"
    before = {k: v.get() for k, v in app.values.items()}
    invalid = json.loads(path.read_text(encoding="utf-8"))
    invalid["values"]["cell_size"] = "broken"
    path.write_text(json.dumps(invalid), encoding="utf-8")
    app.load_profile()
    assert len(errors) == 1
    errors.clear()
    assert {k: v.get() for k, v in app.values.items()} == before


def test_worker_keeps_mode_and_invalidates_stale_exports(tmp_path, monkeypatch, studio):
    app, errors = studio
    root = app.root
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *a, **k: errors.append(str(a)))
    monkeypatch.setattr(gui.filedialog, "askdirectory", lambda **k: str(tmp_path))

    def finish_worker():
        deadline = time.monotonic() + 45
        while app.busy and time.monotonic() < deadline:
            root.update()
            time.sleep(.02)
        root.update()
        assert not errors, errors
        assert not app.busy, "geometry worker did not finish"

    try:
        root.update()
        app.tabs.select(1)
        root.update()
        app.set_model(unit_cube(8), "test cube")
        app.generate()
        finish_worker()
        assert app.tabs.index(app.tabs.select()) == 1
        assert app.bundle is not None and app.bundle.kind == "infill"
        assert not app.export_button.instate(["disabled"])
        app.show_model.set(False)
        assert app._meshes()[0][0] is app.bundle.meshes["infill_only.stl"]
        app.show_generated.set(False)
        assert app._meshes() == []
        app.show_model.set(True)
        assert app._meshes()[0][0] is app.model
        app.show_generated.set(True)
        app.cutaway.set(True)
        app.view_mode.set("3D + 断面")
        app.draw()
        root.update()
        app.figure.savefig(tmp_path / "preview.png")
        app.export()
        finish_worker()
        assert list(tmp_path.glob("infill-*/model_with_infill.stl"))
        app.values["cell_size"].set("9")
        root.update()
        assert app.bundle is None
        assert app.export_button.instate(["disabled"])
        assert not errors
    finally:
        app.close()


def test_close_waits_for_export_callback(studio):
    import threading
    app, _ = studio
    release = threading.Event()
    finished = []
    app._work(lambda: release.wait(3), lambda result: finished.append(result))
    app.close()
    assert not app.closed
    release.set()
    deadline = time.monotonic()+5
    while not app.closed and time.monotonic() < deadline:
        app.root.update()
        time.sleep(.02)
    assert app.closed and finished == [True]


def test_bundled_benchy_import_preserves_selected_mode(studio):
    app, _ = studio
    app.tabs.select(1)
    app.root.update()
    app.open_benchy()
    pump(app, timeout=60)
    assert app.model_name == "3DBenchy.stl"
    assert app.model.is_volume
    assert len(app.model.faces) == 225154
    assert app._mode() == "infill"
    assert "552" in app.log.get("1.0", "end")
    assert not app.generate_button.instate(["disabled"])
    assert app.bundle is None


def test_gpu_camera_does_not_rebuild_mesh_or_section(studio, monkeypatch, tmp_path):
    import numpy as np
    from PIL import Image
    app, _ = studio
    app.root.deiconify()
    app.set_model(unit_cube(8), "cube")
    app.root.update()
    viewport = app.viewport
    assert viewport.context_created and not viewport.error
    viewport.tkMakeCurrent()
    viewport.redraw()
    uploads = viewport.upload_count
    vertices = app.model.vertices.copy()
    monkeypatch.setattr(app.model, "section", lambda **k: pytest.fail("Camera must not recompute sections"))
    viewport.camera.pan(30, 20, 400)
    viewport.orbit(40, 10)
    viewport.zoom(2)
    viewport.redraw()
    assert viewport.upload_count == uploads
    np.testing.assert_array_equal(app.model.vertices, vertices)
    viewport.fit()
    viewport.save_image(tmp_path / "gpu.png")
    pixels = np.asarray(Image.open(tmp_path / "gpu.png"))
    assert pixels.std() > 10  # A visible shaded model, not just a cleared framebuffer.
    app.close()
    assert viewport._disposed and not viewport._buffers and viewport._frame_job is None


def test_open_surface_is_viewable_but_cannot_generate(studio, monkeypatch, tmp_path):
    app, _ = studio
    mesh = unit_cube(4)
    mesh.update_faces(list(range(11)))
    path = tmp_path / "open.stl"
    mesh.export(path)
    monkeypatch.setattr(gui.filedialog, "askopenfilename", lambda **kw: str(path))
    app.open_model()
    pump(app)
    assert app.model is not None and not app.model.is_volume
    assert app.generate_button.instate(["disabled"])
    app.generate()
    assert not app.busy


def test_nozzle_line_width_apply_and_profile_roundtrip(studio, monkeypatch, tmp_path):
    from treesupport.ui_settings import configuration
    app, _ = studio
    app.values["nozzle_diameter"].set(".6")
    app.values["line_width"].set("0")
    app.apply_print_dimensions()
    assert float(app.values["wall_thickness"].get()) == pytest.approx(2.025)
    assert float(app.values["tip_diameter"].get()) == pytest.approx(1.35)
    assert float(app.values["contact_diameter"].get()) == pytest.approx(.675)
    app.values["contact_shape"].set("rounded")
    app.values["line_width"].set(".5")
    app.apply_print_dimensions()
    assert float(app.values["tip_diameter"].get()) == pytest.approx(1.)
    path = tmp_path / "nozzle.json"
    monkeypatch.setattr(gui.filedialog, "asksaveasfilename", lambda **kw: str(path))
    monkeypatch.setattr(gui.filedialog, "askopenfilename", lambda **kw: str(path))
    app.save_profile()
    app.values["contact_shape"].set("flat")
    app.load_profile()
    config = configuration({k: v.get() for k, v in app.values.items()}, "support")
    assert config.contact_shape == "rounded" and config.nozzle_diameter == .6
    assert config.effective_line_width == .5 and config.branch_profile == "organic"
