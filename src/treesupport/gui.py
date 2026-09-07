"""Local desktop UI. Geometry runs in a worker; Tk is used only on its main thread."""
from __future__ import annotations

from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .application import SLICER_GUIDE, build_bundle, export_bundle, load_model
from .config import SupportConfig
from .infill import InfillConfig
from .testmodels import cantilever, unit_cube

BG, PANEL, TEXT, MUTED, ACCENT = "#111820", "#1b2530", "#e4edf4", "#91a5b7", "#59d9b4"


class Studio:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.model = None
        self.bundle = None
        self.busy = False
        self.events = queue.Queue()
        self.controls = []
        self.values = {}
        self._slice_job = None
        root.title("MeshBloom · Simplify3D v5")
        root.geometry("1260x850")
        root.minsize(1050, 740)
        root.configure(bg=BG)
        self._style()
        self._layout()
        self._poll_id = root.after(100, self._poll)
        root.protocol("WM_DELETE_WINDOW", root.destroy)

    def _style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background=PANEL, foreground=TEXT, font=("Yu Gothic UI", 10))
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG)
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Header.TLabel", font=("Yu Gothic UI", 23, "bold"))
        style.configure("TButton", padding=(10, 8), background="#2b3d4d")
        style.map("TButton", background=[("active", "#3a5468"), ("disabled", "#202a34")])
        style.configure("Accent.TButton", background=ACCENT, foreground="#102720",
                        font=("Yu Gothic UI", 11, "bold"))
        style.map("Accent.TButton", background=[("active", "#8ae9ce"), ("disabled", "#2a4540")])
        style.configure("TEntry", fieldbackground="#253341", foreground=TEXT, padding=5)
        style.configure("TCombobox", fieldbackground="#253341", foreground=TEXT, padding=5)
        style.map("TCombobox", fieldbackground=[("readonly", "#253341")],
                  foreground=[("readonly", TEXT)])
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(15, 8))
        style.map("TNotebook.Tab", background=[("selected", "#31485a")])

    def _button(self, parent, text, command, **kwargs):
        button = ttk.Button(parent, text=text, command=command, **kwargs)
        self.controls.append(button)
        return button

    def _layout(self):
        header = ttk.Frame(self.root, padding=(24, 18, 24, 14))
        header.pack(fill="x")
        ttk.Label(header, text="MeshBloom", style="Header.TLabel").pack(anchor="w")
        ttk.Label(header, text="形状をつくる。いつものスライサーで印刷する。  /  Simplify3D v5",
                  style="Muted.TLabel").pack(anchor="w", pady=(5, 0))
        body = ttk.Frame(self.root, padding=(24, 0, 24, 12))
        body.pack(fill="both", expand=True)
        side_container = ttk.Frame(body, width=340)
        side_container.pack(side="left", fill="y", padx=(0, 20))
        side_container.pack_propagate(False)
        side_canvas = tk.Canvas(side_container, width=320, bg=BG, highlightthickness=0)
        scroll = ttk.Scrollbar(side_container, orient="vertical", command=side_canvas.yview)
        scroll.pack(side="right", fill="y")
        side_canvas.pack(side="left", fill="both", expand=True)
        side_canvas.configure(yscrollcommand=scroll.set)
        sidebar = ttk.Frame(side_canvas)
        item = side_canvas.create_window((0, 0), window=sidebar, anchor="nw", width=320)
        sidebar.bind("<Configure>", lambda e: side_canvas.configure(scrollregion=side_canvas.bbox("all")))
        side_canvas.bind("<Configure>", lambda e: side_canvas.itemconfigure(item, width=e.width))

        def scroll_sidebar(event):
            if str(event.widget).startswith(str(side_container)):
                side_canvas.yview_scroll(-int(event.delta/120), "units")
        self.root.bind("<MouseWheel>", scroll_sidebar, add="+")
        ttk.Label(sidebar, text="01   モデル", font=("Yu Gothic UI", 12, "bold")).pack(anchor="w")
        self._button(sidebar, "STL / OBJ / 3MF を開く", self.open_model).pack(fill="x", pady=8)
        examples = ttk.Frame(sidebar)
        examples.pack(fill="x")
        self._button(examples, "片持ち板デモ", lambda: self.demo("support")).pack(side="left", expand=True, fill="x")
        self._button(examples, "キューブデモ", lambda: self.demo("infill")).pack(side="left", expand=True, fill="x", padx=(6, 0))
        self.model_info = tk.StringVar(value="モデル未選択\n単位 mm · 元の座標と印刷姿勢を保持")
        ttk.Label(sidebar, textvariable=self.model_info, wraplength=310, style="Muted.TLabel").pack(anchor="w", pady=10)
        ttk.Label(sidebar, text="02   生成する形状", font=("Yu Gothic UI", 12, "bold")).pack(anchor="w", pady=(8, 10))
        self.tabs = ttk.Notebook(sidebar)
        self.tabs.pack(fill="x")
        support = ttk.Frame(self.tabs, padding=(8, 12))
        infill = ttk.Frame(self.tabs, padding=(8, 12))
        self.tabs.add(support, text="ツリーサポート")
        self.tabs.add(infill, text="インフィル")
        self.tabs.bind("<<NotebookTabChanged>>", lambda e: self.invalidate())
        for key, label, value in [
            ("layer_height", "レイヤー高さ / mm", "0.20"),
            ("top_z_gap", "先端 Z ギャップ / mm", "0.20"),
            ("xy_gap", "側面ギャップ / mm", "0.40"),
            ("tip_spacing", "先端の間隔 / mm", "2.50"),
            ("overhang_angle", "対象角度 / 鉛直から °", "45"),
        ]:
            self._field(support, key, label, value)
        ttk.Label(support, text="ビルドプレートから伸びる枝を生成。\n先端の直径 0.8 mm / 枝の最大傾斜 40°",
                  style="Muted.TLabel").pack(anchor="w", pady=(10, 0))
        self.pattern = tk.StringVar(value="gyroid")
        ttk.Label(infill, text="パターン").pack(anchor="w")
        combo = ttk.Combobox(infill, textvariable=self.pattern, state="readonly",
                             values=("gyroid", "diamond", "cubic"))
        combo.pack(fill="x", pady=(3, 6))
        self.controls.append(combo)
        self.pattern.trace_add("write", lambda *a: self.invalidate())
        for key, label, value in [
            ("cell_size", "セルの大きさ / mm", "8.0"),
            ("wall_thickness", "格子の厚み・径 / mm", "1.2"),
            ("shell_thickness", "外殻の厚み / mm", "1.2"),
            ("pitch", "形状の解像度 / mm", "0.4"),
        ]:
            self._field(infill, key, label, value)
        ttk.Label(infill, text="解像度は厚みの 1/3 以下。\n外殻を含めて、内部空洞を形状化します。",
                  style="Muted.TLabel").pack(anchor="w", pady=(10, 0))
        self.generate_button = self._button(sidebar, "形状を生成", self.generate, style="Accent.TButton")
        self.generate_button.pack(fill="x", pady=(16, 8))
        self.export_button = self._button(sidebar, "03   STL 一式を書き出す", self.export)
        self.export_button.pack(fill="x")
        self.export_button.state(["disabled"])
        self._button(sidebar, "Simplify3D v5 での使い方", self.help).pack(fill="x", pady=8)

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)
        top = ttk.Frame(right)
        top.pack(fill="x")
        ttk.Label(top, text="プレビュー", font=("Yu Gothic UI", 12, "bold")).pack(side="left")
        self.cutaway = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="半分を隠して内部を見る", variable=self.cutaway,
                        command=self.draw).pack(side="right")
        self.figure = Figure(figsize=(8, 5), dpi=100, facecolor=BG)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, pady=(8, 0))
        self.toolbar = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill="x")
        self.toolbar.configure(background=PANEL)
        zbar = ttk.Frame(right)
        zbar.pack(fill="x", pady=6)
        self.zlabel = ttk.Label(zbar, text="断面 Z = —", width=21)
        self.zlabel.pack(side="left")
        self.z = tk.DoubleVar(value=0)
        self.zslider = ttk.Scale(zbar, from_=0, to=1, variable=self.z, command=self._queue_slice)
        self.zslider.pack(side="left", fill="x", expand=True)
        self.log = tk.Text(right, height=6, bg=PANEL, fg=TEXT, relief="flat", wrap="word",
                           font=("Yu Gothic UI", 10), padx=12, pady=8, state="disabled")
        self.log.pack(fill="x", pady=(6, 0))
        self.status = tk.StringVar(value="STLを開くか、デモモデルを選んでください。")
        ttk.Label(self.root, textvariable=self.status, padding=(24, 8), style="Muted.TLabel").pack(fill="x")
        self.draw()

    def _field(self, parent, key, label, value):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label).pack(side="left")
        var = tk.StringVar(value=value)
        self.values[key] = var
        entry = ttk.Entry(row, textvariable=var, width=8)
        entry.pack(side="right")
        self.controls.append(entry)
        var.trace_add("write", lambda *a: self.invalidate())

    def invalidate(self):
        if self.busy:
            return
        if self.bundle is not None:
            self.bundle = None
            self.export_button.state(["disabled"])
            self.status.set("設定が変わりました。形状を再生成してください。")
            self._log("生成結果をクリアしました。現在の設定で再生成してください。")
            self.draw()

    def _log(self, text):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.insert("1.0", text)
        self.log.configure(state="disabled")

    def _work(self, function, callback):
        if self.busy:
            return
        self.busy = True
        for control in self.controls:
            control.state(["disabled"])
        for tab in self.tabs.tabs():
            if tab != self.tabs.select():
                self.tabs.tab(tab, state="disabled")
        self.status.set("処理中…")

        def run():
            try:
                result = function()
                self.events.put(("done", (callback, result)))
            except Exception as exc:
                self.events.put(("error", str(exc)))
        threading.Thread(target=run, daemon=True).start()

    def _poll(self):
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == "progress":
                    self.status.set(data)
                    continue
                self.busy = False
                for control in self.controls:
                    control.state(["!disabled"])
                for tab in self.tabs.tabs():
                    self.tabs.tab(tab, state="normal")
                if kind == "done":
                    callback, result = data
                    callback(result)
                else:
                    self.status.set("処理を完了できませんでした。設定・モデルを確認してください。")
                    self._log(data)
                    messagebox.showerror("生成できませんでした", data, parent=self.root)
                self.export_button.state(["!disabled"] if self.bundle else ["disabled"])
        except queue.Empty:
            pass
        self._poll_id = self.root.after(100, self._poll)

    def open_model(self):
        path = filedialog.askopenfilename(filetypes=[("3D model", "*.stl *.obj *.3mf *.ply"), ("All", "*.*")])
        if path:
            self._work(lambda: load_model(path), lambda m: self.set_model(m, Path(path).name))

    def demo(self, kind):
        self.tabs.select(0 if kind == "support" else 1)
        factory = (lambda: cantilever(column=5, arm=10, depth=8, height=8, plate=2)) if kind == "support" else (lambda: unit_cube(16))
        self._work(factory, lambda m: self.set_model(m, "片持ち板デモ" if kind == "support" else "16 mm キューブ"))

    def set_model(self, model, name):
        self.model = model
        self.bundle = None
        self.export_button.state(["disabled"])
        x, y, z = model.extents
        self.model_info.set(f"{name}\n{x:.2f} × {y:.2f} × {z:.2f} mm · {len(model.faces):,} triangles")
        lo, hi = model.bounds[:, 2]
        self.zslider.configure(from_=lo+1e-4, to=hi-1e-4)
        self.z.set((lo+hi)/2)
        self.status.set("モデルを読み込みました。設定を確認して生成してください。")
        self._log("緑: 生成形状 / グレー: 元モデル\n左: 回転できる3D表示 / 右: 指定したZ高さの断面\n元の座標を保持します。生成前にCAD側で印刷姿勢を決めてください。")
        self.draw()

    def generate(self):
        if self.model is None:
            messagebox.showinfo("モデルを選択", "STLを開くか、デモを選択してください。")
            return
        kind = "support" if self.tabs.index(self.tabs.select()) == 0 else "infill"
        try:
            if kind == "support":
                cfg = SupportConfig(**{k: float(self.values[k].get()) for k in
                    ("layer_height", "top_z_gap", "xy_gap", "tip_spacing", "overhang_angle")}, union_mode="boolean")
            else:
                cfg = InfillConfig(pattern=self.pattern.get(), **{k: float(self.values[k].get()) for k in
                    ("cell_size", "wall_thickness", "shell_thickness", "pitch")})
        except ValueError as exc:
            messagebox.showerror("設定を確認してください", str(exc))
            return
        self.bundle = None
        self.draw()
        self._log("形状を生成しています。モデルの大きさと解像度によって数分かかります。")
        self._work(lambda: build_bundle(self.model, kind, cfg,
                    lambda s: self.events.put(("progress", s))), self.generated)

    def generated(self, bundle):
        self.bundle = bundle
        warnings = bundle.report.get("warnings", [])
        if bundle.kind == "support":
            v = bundle.report["validation"]["metrics"]
            detail = f"先端 {v['tip_count']} / 根元 {v['root_count']} / 対象領域の被覆率 {v['coverage_ratio']:.1%}"
        else:
            detail = f"外殻込みの材料体積比 {bundle.report['material_fraction_including_shell']:.1%}"
            self.cutaway.set(True)
        self._log(detail + "\n出力: " + ", ".join(bundle.meshes) + "\n" + "\n".join(warnings))
        self.status.set("生成完了。断面を確認してSTLを書き出せます。")
        self.draw()

    def export(self):
        if self.bundle is None:
            return
        directory = filedialog.askdirectory(title="出力先を選択（新しいサブフォルダーを作成します）")
        if directory:
            bundle = self.bundle
            self._work(lambda: export_bundle(bundle, directory), self.exported)

    def exported(self, path):
        self.status.set(f"書き出しました: {path}")
        self._log(f"保存先: {path}\nSTL・report.json・Simplify3D-v5.txt を保存しました。")

    def help(self):
        window = tk.Toplevel(self.root)
        window.title("Simplify3D v5 での使い方")
        window.geometry("820x660")
        text = tk.Text(window, wrap="word", bg=PANEL, fg=TEXT, padx=20, pady=20,
                       font=("Yu Gothic UI", 11))
        text.pack(fill="both", expand=True)
        text.insert("1.0", SLICER_GUIDE)
        text.configure(state="disabled")

    def _queue_slice(self, value):
        if self._slice_job:
            self.root.after_cancel(self._slice_job)
        self._slice_job = self.root.after(160, self._draw_slice)

    def _meshes(self):
        if self.model is None:
            return []
        if self.bundle and self.bundle.kind == "infill":
            return [(self.bundle.meshes["model_with_infill.stl"], ACCENT)]
        meshes = [(self.model, "#9baebd")]
        if self.bundle and "tree_support.stl" in self.bundle.meshes:
            meshes.append((self.bundle.meshes["tree_support.stl"], ACCENT))
        return meshes

    def draw(self):
        if not hasattr(self, "figure"):
            return
        self.figure.clear()
        self.ax3d = self.figure.add_subplot(121, projection="3d")
        self.ax2d = self.figure.add_subplot(122)
        ax = self.ax3d
        ax.set_facecolor(BG)
        ax.set_axis_off()
        ax.view_init(elev=24, azim=-55)
        meshes = self._meshes()
        for mesh, color in meshes:
            if self.cutaway.get():
                view = mesh.slice_plane(plane_origin=[self.model.bounds[:, 0].mean(), 0, 0],
                                        plane_normal=[-1, 0, 0], cap=False)
                triangles = view.triangles
            else:
                triangles = mesh.triangles
            if not len(triangles):
                continue
            # Display budget only. Export always uses every triangle.
            step = max(1, int(np.ceil(len(triangles)/60000)))
            ax.add_collection3d(Poly3DCollection(triangles[::step], facecolors=color,
                                                 edgecolors=color, linewidths=0, alpha=1, shade=True,
                                                 zsort="average"))
        if meshes:
            bounds = np.vstack([m.bounds for m, _ in meshes])
            low, high = bounds.min(axis=0), bounds.max(axis=0)
            mid = (low+high)/2
            radius = max(high-low)/2*1.08
            ax.set_xlim(mid[0]-radius, mid[0]+radius)
            ax.set_ylim(mid[1]-radius, mid[1]+radius)
            ax.set_zlim(mid[2]-radius, mid[2]+radius)
            ax.set_box_aspect((1, 1, 1))
        else:
            ax.text2D(.5, .5, "Open a model to begin", transform=ax.transAxes, ha="center", color=MUTED)
        ax.set_title("3D / mm", color=MUTED, fontsize=10)
        self.figure.subplots_adjust(left=.04, right=.97, bottom=.13, top=.90, wspace=.12)
        self._draw_slice()

    def _draw_slice(self):
        self._slice_job = None
        if not hasattr(self, "ax2d"):
            return
        ax = self.ax2d
        ax.clear()
        ax.set_facecolor(BG)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#334654")
        height = self.z.get()
        self.zlabel.configure(text=f"断面 Z = {height:.2f} mm")
        for mesh, color in self._meshes():
            section = mesh.section(plane_origin=[0, 0, height], plane_normal=[0, 0, 1])
            if section is not None:
                for line in section.discrete:
                    ax.plot(line[:, 0], line[:, 1], color=color, linewidth=.85)
        if self.model is not None:
            low, high = self.model.bounds[:, :2]
            margin = max(self.model.extents[:2])*.12
            if self.bundle and "tree_support.stl" in self.bundle.meshes:
                support_bounds = self.bundle.meshes["tree_support.stl"].bounds[:, :2]
                low, high = np.minimum(low, support_bounds[0]), np.maximum(high, support_bounds[1])
            ax.set_xlim(low[0]-margin, high[0]+margin)
            ax.set_ylim(low[1]-margin, high[1]+margin)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"SECTION / Z {height:.2f} mm", color=MUTED, fontsize=10)
        ax.set_xlabel("X / mm", color=MUTED, fontsize=8)
        ax.set_ylabel("Y / mm", color=MUTED, fontsize=8)
        self.canvas.draw_idle()


def main():
    root = tk.Tk()
    Studio(root)
    root.mainloop()


if __name__ == "__main__":
    main()
