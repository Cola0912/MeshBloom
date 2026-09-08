"""Run with the project Python on a desktop with OpenGL. Uses the bundled Benchy."""
import json
from pathlib import Path
import time
import tkinter as tk

import numpy as np
from OpenGL import GL
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from treesupport.sample_models import load_benchy
from treesupport.viewer import MeshViewport


def main():
    mesh = load_benchy()
    figure = Figure(figsize=(9, 6), dpi=100)
    canvas = FigureCanvasAgg(figure)
    ax = figure.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(mesh.triangles, facecolors="#9baebd", shade=True))
    ax.auto_scale_xyz(*mesh.vertices.T)
    old_times = []
    for angle in range(4):
        start = time.perf_counter()
        ax.view_init(25, -55+angle)
        canvas.draw()
        old_times.append(time.perf_counter()-start)
    root = tk.Tk()
    root.title("MeshBloom · Viewport benchmark")
    root.geometry("900x600")
    view = MeshViewport(root)
    view.pack(fill="both", expand=True)
    root.update()
    view.set_meshes([(mesh, "#9baebd")], reset_camera=True)
    view.tkMakeCurrent()
    view.redraw()
    GL.glFinish()
    timings = []
    for angle in range(120):
        start = time.perf_counter()
        view.camera.azimuth = -55+angle*2
        view.redraw()
        view.tkSwapBuffers()
        GL.glFinish()
        timings.append(time.perf_counter()-start)
    result = {"triangles": len(mesh.faces), "renderer": view.renderer,
              "viewport": [view.winfo_width(), view.winfo_height()], "frames": len(timings),
              "opengl_median_ms": float(np.median(timings)*1000),
              "opengl_p95_ms": float(np.percentile(timings, 95)*1000),
              "matplotlib_median_ms": float(np.median(old_times[1:])*1000),
              "mesh_uploads": view.upload_count}
    out = Path("out/viewer-benchmark")
    out.mkdir(parents=True, exist_ok=True)
    view.save_image(out / "benchy.png")
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    view.dispose()
    root.destroy()


if __name__ == "__main__":
    main()
