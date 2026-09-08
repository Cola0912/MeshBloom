"""Local desktop UI. Geometry runs in a worker; Tk is used only on its main thread."""
from __future__ import annotations

from pathlib import Path
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .application import SLICER_GUIDE, build_bundle, export_bundle
from .testmodels import cantilever, unit_cube
from .ui_settings import (SUPPORT_FIELDS, INFILL_FIELDS, PRINT_FIELDS, CHOICE_FIELDS, PRESETS,
                          configuration, load_settings, save_settings)
from .print_profile import PrintProfile
from .mesh_loader import MeshLoader
from .viewer import MeshViewport

BG, PANEL, TEXT, MUTED, ACCENT = "#111820", "#1b2530", "#e4edf4", "#91a5b7", "#59d9b4"

STAGES = {"load": "モデルを読み込み中", "slice": "レイヤー断面を解析中",
          "overhang": "オーバーハングを検出中", "tips": "先端の配置を計算中",
          "propagate": "枝を生成・合流中", "trim": "枝の接続を整理中",
          "centerline": "枝の中心線を計算中", "smoothing": "枝を滑らかに調整中",
          "collision3d": "モデルとの距離を確認中", "mesh": "サポートをメッシュ化中",
          "validate": "生成結果を検証中"}


class GenerationCancelled(Exception):
    pass


class Studio:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.model = None
        self.bundle = None
        self.busy = False
        self.events = queue.Queue()
        self.controls = []
        self.values = {}
        self.source_model = None
        self.model_name = ""
        self.output_path = None
        self.cancel_event = threading.Event()
        self.cancellable = False
        self.closed = False
        self._close_when_idle = False
        self._applying_settings = False
        self._started = 0
        self._slice_job = None
        root.title("MeshBloom · Simplify3D v5")
        root.geometry("1260x850")
        root.minsize(1020, 700)
        root.configure(bg=BG)
        self._style()
        self._layout()
        self._poll_id = root.after(100, self._poll)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Control-o>", lambda e: self.open_model() if not self.busy else None)
        root.bind("<Control-s>", lambda e: self.export() if not self.busy else None)
        root.bind("<F5>", lambda e: self.generate() if not self.busy else None)
        self._sync_controls()

    def _style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background=PANEL, foreground=TEXT, font=("Yu Gothic UI", 10))
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG)
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Header.TLabel", font=("Yu Gothic UI", 23, "bold"))
        style.configure("TButton", padding=(10, 8), background="#2b3d4d", borderwidth=1,
                        bordercolor="#425566", lightcolor="#425566", darkcolor="#425566")
        style.map("TButton", background=[("active", "#3a5468"), ("disabled", "#202a34")])
        style.configure("Accent.TButton", background=ACCENT, foreground="#102720",
                        font=("Yu Gothic UI", 11, "bold"))
        style.map("Accent.TButton", background=[("active", "#8ae9ce"), ("disabled", "#2a4540")])
        style.configure("TEntry", fieldbackground="#253341", foreground=TEXT, padding=5)
        style.configure("TSpinbox", fieldbackground="#253341", foreground=TEXT, padding=4, arrowcolor=TEXT)
        style.configure("TCheckbutton", background=BG)
        style.map("TCheckbutton", background=[("active", BG)])
        style.configure("Horizontal.TScale", background=BG, troughcolor="#293d4c")
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
        brand = ttk.Frame(header)
        brand.pack(fill="x")
        ttk.Label(brand, text="MeshBloom", style="Header.TLabel").pack(side="left")
        ttk.Label(brand, text="STL WORKSPACE  /  Simplify3D v5", style="Muted.TLabel").pack(side="right")
        ttk.Label(header, text="形状をつくる。いつものスライサーで印刷する。  /  Simplify3D v5",
                  style="Muted.TLabel").pack(anchor="w", pady=(5, 0))
        body = ttk.Frame(self.root, padding=(24, 0, 24, 12))
        body.pack(fill="both", expand=True)
        side_container = ttk.Frame(body, width=340)
        side_container.pack(side="left", fill="y", padx=(0, 20))
        side_container.pack_propagate(False)
        actions = ttk.Frame(side_container, padding=(0, 12, 0, 0))
        actions.pack(side="bottom", fill="x")
        self.generate_button = self._button(actions, "形状を生成   F5", self.generate, style="Accent.TButton")
        self.generate_button.pack(fill="x", pady=(0, 6))
        self.export_button = self._button(actions, "STL 一式を書き出す   Ctrl+S", self.export)
        self.export_button.pack(fill="x")
        self.export_button.state(["disabled"])
        self._button(actions, "Simplify3D v5 での使い方", self.help).pack(fill="x", pady=(6, 0))
        self.open_output_button = ttk.Button(actions, text="保存先フォルダーを開く", command=self.open_output)
        self.open_output_button.pack(fill="x", pady=(6, 0))
        self.open_output_button.state(["disabled"])
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
        self._button(sidebar, "3DBenchy を開く", self.open_benchy).pack(fill="x", pady=(6, 0))
        self.model_info = tk.StringVar(value="モデル未選択\n単位 mm · 元の座標と印刷姿勢を保持")
        ttk.Label(sidebar, textvariable=self.model_info, wraplength=310, style="Muted.TLabel").pack(anchor="w", pady=10)
        pose = ttk.Frame(sidebar)
        pose.pack(fill="x")
        for axis in ("X", "Y", "Z"):
            self._button(pose, f"{axis} +90°", lambda a=axis: self.transform_model(a)).pack(side="left", fill="x", expand=True)
        place = ttk.Frame(sidebar)
        place.pack(fill="x", pady=5)
        self._button(place, "ベッドに置く", lambda: self.transform_model("bed")).pack(side="left", expand=True, fill="x")
        self._button(place, "元の姿勢に戻す", lambda: self.transform_model("reset")).pack(side="left", expand=True, fill="x", padx=(4, 0))
        ttk.Label(sidebar, text="ノズルとライン幅 / mm", font=("Yu Gothic UI", 11, "bold")).pack(anchor="w", pady=(10, 4))
        for key, (label, value) in PRINT_FIELDS.items():
            self._field(sidebar, key, label, value)
        self._button(sidebar, "ライン幅を太さに適用", self.apply_print_dimensions).pack(fill="x", pady=5)
        ttk.Label(sidebar, text="先端2本・接触1本・枝5本・格子/外殻3本分。\n適用後も各寸法を個別に変更できます。",
                  style="Muted.TLabel").pack(anchor="w")
        ttk.Label(sidebar, text="02   生成する形状", font=("Yu Gothic UI", 12, "bold")).pack(anchor="w", pady=(8, 10))
        self.tabs = ttk.Notebook(sidebar)
        self.tabs.pack(fill="x")
        support = ttk.Frame(self.tabs, padding=(8, 12))
        infill = ttk.Frame(self.tabs, padding=(8, 12))
        self.tabs.add(support, text="ツリーサポート")
        self.tabs.add(infill, text="インフィル")
        self.tabs.bind("<<NotebookTabChanged>>", lambda e: self.invalidate())
        for key, (label, default, options) in CHOICE_FIELDS.items():
            ttk.Label(support, text=label).pack(anchor="w")
            var = tk.StringVar(value=default)
            self.values[key] = var
            combo = ttk.Combobox(support, textvariable=var, values=options, state="readonly")
            combo.pack(fill="x", pady=(2, 5))
            self.controls.append(combo)
            var.trace_add("write", lambda *a: self.invalidate())
        ttk.Label(support, text="flat: 平面 / tapered: 絞り / rounded: 丸み\n接触面は指定Zギャップの下で止まります。", style="Muted.TLabel").pack(anchor="w", pady=(0, 5))
        for key, (label, value) in SUPPORT_FIELDS.items():
            self._field(support, key, label, value)
        ttk.Label(support, text="長さは mm。角度は鉛直からの傾き。\nビルドプレートから伸びる枝を生成します。",
                  style="Muted.TLabel").pack(anchor="w", pady=(10, 0))
        self.pattern = tk.StringVar(value="gyroid")
        ttk.Label(infill, text="パターン").pack(anchor="w")
        combo = ttk.Combobox(infill, textvariable=self.pattern, state="readonly",
                             values=("gyroid", "diamond", "cubic"))
        combo.pack(fill="x", pady=(3, 6))
        self.controls.append(combo)
        self.pattern.trace_add("write", lambda *a: self.invalidate())
        for key, (label, value) in INFILL_FIELDS.items():
            self._field(infill, key, label, value)
        ttk.Label(infill, text="解像度は厚みの 1/3 以下。\n外殻を含めて、内部空洞を形状化します。",
                  style="Muted.TLabel").pack(anchor="w", pady=(10, 0))
        preset_row = ttk.Frame(sidebar)
        preset_row.pack(fill="x", pady=(12, 6))
        self.preset = tk.StringVar(value="標準")
        presets = ttk.Combobox(preset_row, textvariable=self.preset, values=tuple(PRESETS), state="readonly", width=13)
        presets.pack(side="left", expand=True, fill="x")
        self.controls.append(presets)
        self._button(preset_row, "プリセット適用", self.apply_preset).pack(side="right", padx=(4, 0))
        settings_row = ttk.Frame(sidebar)
        settings_row.pack(fill="x", pady=(0, 12))
        self._button(settings_row, "設定を保存", self.save_profile).pack(side="left", expand=True, fill="x")
        self._button(settings_row, "設定を開く", self.load_profile).pack(side="left", expand=True, fill="x", padx=(4, 0))

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)
        top = ttk.Frame(right)
        top.pack(fill="x")
        self.view_mode = tk.StringVar(value="3D")
        views = ttk.Combobox(top, textvariable=self.view_mode, state="readonly", values=("3D + 断面", "3D", "断面"), width=13)
        views.pack(side="left")
        views.bind("<<ComboboxSelected>>", lambda e: self.draw())
        self.perspective = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="透視投影", variable=self.perspective,
                        command=self.update_view_options).pack(side="left", padx=8)
        self.cutaway = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="半分を隠して内部を見る", variable=self.cutaway,
                        command=self.draw).pack(side="right")
        objects = ttk.Frame(right)
        objects.pack(fill="x", pady=(8, 0))
        self.show_model = tk.BooleanVar(value=True)
        self.show_generated = tk.BooleanVar(value=True)
        self.wireframe = tk.BooleanVar(value=False)
        self.show_grid = tk.BooleanVar(value=True)
        for text, var in (("外形 / モデル", self.show_model), ("生成した形状", self.show_generated), ("メッシュ線", self.wireframe)):
            ttk.Checkbutton(objects, text=text, variable=var, command=self.draw).pack(side="left", padx=(0, 8))
        ttk.Checkbutton(objects, text="グリッド", variable=self.show_grid,
                        command=self.update_view_options).pack(side="left")
        camera = ttk.Frame(right)
        camera.pack(fill="x", pady=(6, 0))
        directions = {"斜め": (25, -55), "上": (90, -90), "下": (-90, -90),
                      "前": (0, -90), "後": (0, 90), "右": (0, 0), "左": (0, 180)}
        self.camera_direction = tk.StringVar(value="斜め")
        camera_combo = ttk.Combobox(camera, textvariable=self.camera_direction,
                                    values=tuple(directions), state="readonly", width=6)
        camera_combo.pack(side="left", padx=(0, 5))
        camera_combo.bind("<<ComboboxSelected>>", lambda e: self.set_camera(*directions[self.camera_direction.get()]))
        ttk.Button(camera, text="全体表示  F", command=lambda: self.viewport.fit()).pack(side="left", padx=3)
        ttk.Button(camera, text="＋", width=3, command=lambda: self.viewport.zoom(1)).pack(side="left", padx=3)
        ttk.Button(camera, text="−", width=3, command=lambda: self.viewport.zoom(-1)).pack(side="left", padx=3)
        ttk.Button(camera, text="画像を保存", command=self.save_view_image).pack(side="right")
        self.viewport_host = ttk.Frame(right)
        self.viewport_host.rowconfigure(0, weight=1)
        self.viewport_host.columnconfigure(0, weight=1, uniform="preview")
        self.viewport_host.columnconfigure(1, weight=1, uniform="preview")
        self.viewport = MeshViewport(self.viewport_host)
        self.section_panel = ttk.Frame(self.viewport_host)
        self.figure = Figure(figsize=(8, 5), dpi=100, facecolor=BG)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.section_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, pady=(8, 0))
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.section_panel, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill="x")
        self.toolbar.configure(background=PANEL)
        zbar = ttk.Frame(right)
        self.zbar = zbar
        zbar.pack(fill="x", pady=6)
        self.zlabel = ttk.Label(zbar, text="断面 Z = —", width=21)
        self.zlabel.pack(side="left")
        self.z = tk.DoubleVar(value=0)
        self.zslider = ttk.Scale(zbar, from_=0, to=1, variable=self.z, command=self._queue_slice)
        self.zslider.pack(side="left", fill="x", expand=True)
        self.zentry = ttk.Spinbox(zbar, textvariable=self.z, from_=0, to=1, increment=.2, width=7, command=self._queue_slice)
        self.zentry.pack(side="right", padx=(8, 0))
        self.zentry.bind("<Return>", self._queue_slice)
        self.zentry.bind("<FocusOut>", self._queue_slice)
        self.log = tk.Text(right, height=6, bg=PANEL, fg=TEXT, relief="flat", wrap="word",
                           font=("Yu Gothic UI", 10), padx=12, pady=8, state="disabled")
        self.log.pack(fill="x", pady=(6, 0))
        self.result_info = tk.StringVar(value="モデルを読み込むと、ここに生成結果を表示します。")
        result_label = ttk.Label(right, textvariable=self.result_info, style="Muted.TLabel")
        result_label.pack(fill="x", pady=(6, 0))
        # Reserve the controls/results first; only the viewport should shrink.
        self.navigation_hint = ttk.Label(right, text="左ドラッグ: 回転  /  右・中・Shift＋左: 移動  /  ホイール: 拡大縮小  /  ダブルクリック: 全体表示",
                                         style="Muted.TLabel", wraplength=800)
        right.bind("<Configure>", lambda e: self.navigation_hint.configure(wraplength=max(200, e.width-10)))
        preview_widgets = (top, objects, camera, self.viewport_host,
                           self.navigation_hint, zbar, self.log, result_label)
        for widget in preview_widgets:
            widget.pack_forget()
        for row, widget in enumerate(preview_widgets):
            widget.grid(row=row, column=0, sticky="nsew" if row == 3 else "ew", pady=(0, 6))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(3, weight=1)
        footer = ttk.Frame(self.root, padding=(24, 8))
        footer.pack(side="bottom", fill="x", before=body)
        self.status = tk.StringVar(value="STLを開くか、デモモデルを選んでください。")
        ttk.Label(footer, textvariable=self.status, style="Muted.TLabel", width=68).pack(side="left", fill="x", expand=True)
        self.elapsed = tk.StringVar(value="")
        ttk.Label(footer, textvariable=self.elapsed, width=8).pack(side="left")
        self.progressbar = ttk.Progressbar(footer, mode="indeterminate", length=100)
        self.progressbar.pack(side="left", padx=8)
        self.cancel_button = ttk.Button(footer, text="生成を中止", command=self.cancel)
        self.cancel_button.pack(side="right")
        self.cancel_button.state(["disabled"])
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
        if self.busy or self._applying_settings:
            return
        if self.bundle is not None:
            self.bundle = None
            self.export_button.state(["disabled"])
            self.status.set("設定が変わりました。形状を再生成してください。")
            self._log("生成結果をクリアしました。現在の設定で再生成してください。")
            self.result_info.set("設定が変わりました。再生成が必要です。")
            self.draw()

    def _mode(self):
        return "support" if self.tabs.index(self.tabs.select()) == 0 else "infill"

    def _sync_controls(self):
        for control in self.controls:
            control.state(["disabled"] if self.busy else ["!disabled"])
        self.generate_button.state(["disabled"] if self.busy or self.model is None or not self.model.is_volume else ["!disabled"])
        self.export_button.state(["disabled"] if self.busy or self.bundle is None else ["!disabled"])
        self.cancel_button.state(["!disabled"] if self.busy and self.cancellable and not self.cancel_event.is_set() else ["disabled"])
        self.open_output_button.state(["!disabled"] if self.output_path else ["disabled"])

    def transform_model(self, action):
        if self.model is None or self.busy:
            return
        import trimesh
        mesh = self.model.copy()
        if action == "reset":
            mesh = self.source_model.copy()
        elif action == "bed":
            mesh.apply_translation([0, 0, -mesh.bounds[0, 2]])
        else:
            axis = np.eye(3)[("X", "Y", "Z").index(action)]
            matrix = trimesh.transformations.rotation_matrix(np.pi/2, axis, point=mesh.bounds.mean(axis=0))
            mesh.apply_transform(matrix)
        self.set_model(mesh, self.model_name, remember_original=False)
        self.status.set("姿勢を変更しました。出力にもこの座標が反映されます。")

    def apply_preset(self):
        if self.busy:
            return
        fields = SUPPORT_FIELDS if self._mode() == "support" else INFILL_FIELDS
        self._applying_settings = True
        try:
            for key, (_, default) in fields.items():
                self.values[key].set(str(PRESETS[self.preset.get()].get(key, default)))
        finally:
            self._applying_settings = False
        self.invalidate()
        self.status.set(f"「{self.preset.get()}」を適用しました。")

    def apply_print_dimensions(self):
        if self.busy:
            return
        try:
            profile = PrintProfile(float(self.values["nozzle_diameter"].get()), float(self.values["line_width"].get()))
        except ValueError as exc:
            messagebox.showerror("ノズル・ライン幅を確認してください", str(exc), parent=self.root)
            return
        self._applying_settings = True
        try:
            for key, value in profile.dimensions().items():
                self.values[key].set(f"{value:.6g}")
        finally:
            self._applying_settings = False
        self.invalidate()
        self.status.set(f"ライン幅 {profile.width:.3f} mm から太さを設定しました。必要に応じて個別に調整できます。")

    def save_profile(self):
        if self.busy:
            return
        path = filedialog.asksaveasfilename(title="生成設定を保存", defaultextension=".json",
                                           initialfile="meshbloom-settings.json", filetypes=[("MeshBloom settings", "*.json")])
        if path:
            try:
                save_settings(path, {k: v.get() for k, v in self.values.items()}, self._mode(), self.pattern.get())
                self.status.set(f"設定を保存しました: {Path(path).name}")
            except (ValueError, OSError) as exc:
                messagebox.showerror("設定を保存できません", str(exc), parent=self.root)

    def load_profile(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(title="生成設定を開く", filetypes=[("MeshBloom settings", "*.json")])
        if not path:
            return
        try:
            data = load_settings(path)
        except (ValueError, OSError) as exc:
            messagebox.showerror("設定を読み込めません", str(exc), parent=self.root)
            return
        # Validate everything before changing any widget or invalidating output.
        self.invalidate()
        self._applying_settings = True
        try:
            for key, value in data["values"].items():
                self.values[key].set(str(value))
            self.pattern.set(data["pattern"])
            self.tabs.select(0 if data["mode"] == "support" else 1)
        finally:
            self._applying_settings = False
        self.status.set(f"設定を読み込みました: {Path(path).name}")

    def open_output(self):
        if self.output_path:
            try:
                os.startfile(self.output_path)
            except OSError as exc:
                messagebox.showerror("フォルダーを開けません", str(exc), parent=self.root)

    def set_camera(self, elevation, azimuth):
        self.viewport.set_camera(elevation, azimuth)

    def update_view_options(self):
        self.viewport.camera.perspective = self.perspective.get()
        self.viewport.wireframe = self.wireframe.get()
        self.viewport.show_grid = self.show_grid.get()
        self.viewport.cutaway = self.cutaway.get()
        if self.model is not None:
            self.viewport.cut_x = float(self.model.bounds[:, 0].mean())
        self.viewport.request_render()

    def save_view_image(self):
        is_section = self.view_mode.get() == "断面"
        if not is_section and (not self.viewport.context_created or self.viewport.error):
            return
        path = filedialog.asksaveasfilename(title="断面画像を保存" if is_section else "3D画像を保存",
                                          defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            try:
                if is_section:
                    self.figure.savefig(path)
                else:
                    self.viewport.save_image(path)
                self.status.set(f"画像を保存しました: {path}")
            except Exception as exc:
                messagebox.showerror("画像を保存できません", str(exc), parent=self.root)

    def cancel(self):
        if self.busy and self.cancellable:
            self.cancel_event.set()
            self.status.set("中止を受け付けました。現在の処理工程が終わるまでお待ちください。")
            self._sync_controls()

    def close(self):
        if self.busy and not self.cancellable:
            self._close_when_idle = True
            self.status.set("現在の読み込み・保存が完了したら終了します。")
            return
        self.closed = True
        self.cancel_event.set()
        self.progressbar.stop()
        self.root.after_cancel(self._poll_id)
        if self._slice_job:
            self.root.after_cancel(self._slice_job)
        self.viewport.dispose()
        self.root.destroy()

    def _log(self, text):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.insert("1.0", text)
        self.log.configure(state="disabled")

    def _work(self, function, callback, cancellable=False):
        if self.busy:
            return
        self.busy = True
        self.cancellable = cancellable
        self.cancel_event.clear()
        self._started = time.monotonic()
        self.progressbar.start(12)
        self._sync_controls()
        for tab in self.tabs.tabs():
            if tab != self.tabs.select():
                self.tabs.tab(tab, state="disabled")
        self.status.set("処理中…")

        def run():
            try:
                result = function()
                if cancellable and self.cancel_event.is_set():
                    raise GenerationCancelled()
                self.events.put(("done", (callback, result)))
            except GenerationCancelled:
                self.events.put(("cancelled", None))
            except Exception as exc:
                self.events.put(("error", str(exc)))
        threading.Thread(target=run, daemon=True).start()

    def _poll(self):
        if self.closed:
            return
        if self.busy:
            self.elapsed.set(f"{time.monotonic()-self._started:.0f} 秒")
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == "done" and self.cancellable and self.cancel_event.is_set():
                    kind, data = "cancelled", None
                if kind == "progress":
                    if not self.cancel_event.is_set():
                        self.status.set(STAGES.get(data, data))
                    continue
                self.busy = False
                self.progressbar.stop()
                for tab in self.tabs.tabs():
                    self.tabs.tab(tab, state="normal")
                if kind == "done":
                    callback, result = data
                    try:
                        callback(result)
                    except Exception as exc:
                        self._log(f"画面の更新に失敗しました: {exc}")
                        self.status.set("画面を更新できませんでした。")
                elif kind == "cancelled":
                    self.status.set("生成を中止しました。設定を変更して再実行できます。")
                    self._log("生成を中止しました。新しいSTLは保存していません。")
                    self.result_info.set("中止")
                else:
                    self.status.set("処理を完了できませんでした。設定・モデルを確認してください。")
                    self._log(data)
                    messagebox.showerror("生成できませんでした", data, parent=self.root)
                    self.result_info.set("処理に失敗しました。下の詳細を確認してください。")
                self._sync_controls()
                if self._close_when_idle:
                    self.close()
                    return
        except queue.Empty:
            pass
        self._poll_id = self.root.after(100, self._poll)

    def open_model(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(filetypes=[("3D model", "*.stl *.obj *.3mf *.ply"), ("All", "*.*")])
        if path:
            self._work(lambda: MeshLoader().load(path).mesh, lambda m: self.set_model(m, Path(path).name))

    def demo(self, kind):
        if self.busy:
            return
        self.tabs.select(0 if kind == "support" else 1)
        factory = (lambda: cantilever(column=5, arm=10, depth=8, height=8, plate=2)) if kind == "support" else (lambda: unit_cube(16))
        self._work(factory, lambda m: self.set_model(m, "片持ち板デモ" if kind == "support" else "16 mm キューブ"))

    def open_benchy(self):
        if self.busy:
            return
        from .sample_models import load_benchy
        self._work(load_benchy, lambda m: self.set_model(m, "3DBenchy.stl"))

    def set_model(self, model, name, remember_original=True):
        self.model = model
        self.model_name = name
        if remember_original:
            self.source_model = model.copy()
        self.show_model.set(True)
        self.show_generated.set(True)
        self.cutaway.set(False)
        self.bundle = None
        self.export_button.state(["disabled"])
        x, y, z = model.extents
        solid = "閉じたソリッド" if model.is_volume else "開いた面 / ソリッドではありません"
        self.model_info.set(f"{name}\n{x:.2f} × {y:.2f} × {z:.2f} mm\n{len(model.faces):,} 面  /  最下点 Z {model.bounds[0,2]:.2f} mm\n{solid}")
        lo, hi = self._z_bounds()
        self._update_z_bounds()
        self.z.set((lo+hi)/2)
        self.status.set("モデルを読み込みました。設定を確認して生成してください。")
        self._log("緑: 生成形状 / グレー: 元モデル\n姿勢調整はモデル全体に作用します。読み込んだだけでは元の座標を保持します。\n浮いたモデルの下を支える場合は、そのままサポートを生成してください。")
        removed = model.metadata.get("meshbloom_removed_collapsed_faces", 0)
        if removed and remember_original:
            self._log(f"読み込み時にゼロ面積の面 {removed:,} 枚を除外しました。\n元のSTLと頂点座標は変更していません。\n緑: 生成形状 / グレー: 元モデル")
        self.result_info.set("準備完了  /  パラメータを確認して生成してください。")
        if not model.is_volume:
            self.result_info.set("閲覧モード  /  形状生成には閉じたソリッドが必要です。")
        self._sync_controls()
        self.draw(reset_camera=True)

    def generate(self):
        if self.busy:
            return
        if self.model is None:
            messagebox.showinfo("モデルを選択", "STLを開くか、デモを選択してください。")
            return
        if not self.model.is_volume:
            self.status.set("閲覧できますが、形状生成には閉じたソリッドが必要です。")
            return
        kind = self._mode()
        try:
            cfg = configuration({k: v.get() for k, v in self.values.items()}, kind, self.pattern.get())
        except ValueError as exc:
            messagebox.showerror("設定を確認してください", str(exc))
            return
        self.bundle = None
        self.draw()
        self._log("形状を生成しています。モデルの大きさと解像度によって数分かかります。")
        self.result_info.set("生成中…")
        model = self.model.copy()

        def progress(stage):
            if self.cancel_event.is_set():
                raise GenerationCancelled()
            self.events.put(("progress", stage))
        self._work(lambda: build_bundle(model, kind, cfg, progress), self.generated, cancellable=True)

    def generated(self, bundle):
        self.bundle = bundle
        warnings = bundle.report.get("warnings", [])
        if bundle.kind == "support":
            v = bundle.report["validation"]["metrics"]
            detail = f"先端 {v['tip_count']} / 根元 {v['root_count']} / 対象領域の被覆率 {v['coverage_ratio']:.1%}"
            gap = bundle.report["config"]["derived"]["actual_top_z_gap"]
            detail += f" / Zギャップ下限 {gap:g} mm"
        else:
            detail = f"外殻込みの材料体積比 {bundle.report['material_fraction_including_shell']:.1%}"
            self.cutaway.set(True)
        self._log(detail + "\n出力: " + ", ".join(bundle.meshes) + "\n" + "\n".join(warnings))
        self.status.set("生成完了。断面を確認してSTLを書き出せます。")
        self.result_info.set(detail + (f"  /  注意 {len(warnings)} 件" if warnings else "  /  検証済み"))
        self._update_z_bounds()
        self.draw()

    def export(self):
        if self.bundle is None or self.busy:
            return
        directory = filedialog.askdirectory(title="出力先を選択（新しいサブフォルダーを作成します）")
        if directory:
            bundle = self.bundle
            self._work(lambda: export_bundle(bundle, directory), self.exported)

    def exported(self, path):
        self.output_path = Path(path)
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

    def _z_bounds(self):
        if self.model is None:
            return 0., 1.
        return min(0., float(self.model.bounds[0, 2])), max(0., float(self.model.bounds[1, 2]))

    def _update_z_bounds(self):
        lo, hi = self._z_bounds()
        self.zslider.configure(from_=lo, to=hi)
        self.zentry.configure(from_=lo, to=hi)

    def _section_z(self):
        lo, hi = self._z_bounds()
        try:
            height = float(self.z.get())
            if not np.isfinite(height):
                raise ValueError()
        except (ValueError, tk.TclError):
            height = (lo+hi)/2
        height = float(np.clip(height, lo, hi))
        self.z.set(height)
        return height

    def _queue_slice(self, value=None):
        if self._slice_job:
            self.root.after_cancel(self._slice_job)
        self._slice_job = self.root.after(160, self._draw_slice)

    def _meshes(self):
        if self.model is None:
            return []
        if self.bundle and self.bundle.kind == "infill":
            if self.show_generated.get():
                name = "model_with_infill.stl" if self.show_model.get() else "infill_only.stl"
                return [(self.bundle.meshes[name], ACCENT)]
            return [(self.model, "#9baebd")] if self.show_model.get() else []
        meshes = [(self.model, "#9baebd")] if self.show_model.get() else []
        if self.bundle and self.show_generated.get() and "tree_support.stl" in self.bundle.meshes:
            meshes.append((self.bundle.meshes["tree_support.stl"], ACCENT))
        return meshes

    def draw(self, reset_camera=False):
        if not hasattr(self, "figure"):
            return
        mode = self.view_mode.get()
        self.viewport.grid_remove()
        self.section_panel.grid_remove()
        if mode != "断面":
            self.viewport.grid(row=0, column=0, columnspan=2 if mode == "3D" else 1, sticky="nsew")
        if mode != "3D":
            self.section_panel.grid(row=0, column=1 if mode == "3D + 断面" else 0,
                                    columnspan=1 if mode == "3D + 断面" else 2, sticky="nsew")
            self.zbar.grid()
            if getattr(self, "ax2d", None) is None:
                self.figure.clear()
                self.ax2d = self.figure.add_subplot(111)
                self.figure.subplots_adjust(left=.18, right=.96, bottom=.18, top=.9)
                self.toolbar.update()
            self._draw_slice()
        else:
            self.zbar.grid_remove()
            self.ax2d = None
        self.viewport.set_meshes(self._meshes(), reset_camera=reset_camera)
        self.update_view_options()

    def _draw_slice(self):
        if self._slice_job:
            self.root.after_cancel(self._slice_job)
        self._slice_job = None
        height = self._section_z()
        self.zlabel.configure(text=f"断面 Z = {height:.2f} mm")
        if getattr(self, "ax2d", None) is None:
            return
        ax = self.ax2d
        ax.clear()
        ax.set_facecolor(BG)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#334654")
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
