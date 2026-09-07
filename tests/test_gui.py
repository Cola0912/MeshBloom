"""Exercise the real Tk event loop and worker handoff without interactive dialogs."""
import time
import tkinter as tk

import pytest

from treesupport import gui
from treesupport.testmodels import unit_cube


def test_worker_keeps_mode_and_invalidates_stale_exports(tmp_path, monkeypatch):
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display is unavailable")
    root.withdraw()
    errors = []
    root.report_callback_exception = lambda *args: errors.append(str(args))
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *a, **k: errors.append(str(a)))
    monkeypatch.setattr(gui.filedialog, "askdirectory", lambda **k: str(tmp_path))
    app = gui.Studio(root)

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
        app.cutaway.set(True)
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
        root.after_cancel(app._poll_id)
        if app._slice_job:
            root.after_cancel(app._slice_job)
        root.destroy()
