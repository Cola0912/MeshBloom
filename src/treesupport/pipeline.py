"""パイプライン全体の組み立て。

    Mesh -> Slices -> Support Demand -> Tree Graph -> 3D Geometry

各段階のオブジェクトを ``PipelineResult`` に保持するので、
CLI からもテストからも同じ経路で中間状態を検査できる。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import trimesh

from . import geometry2d as g2
from .centerline import CenterlineSolver
from .config import SupportConfig
from .debug_io import DebugWriter
from .graph import TreeGraph
from .merge import TreeMerger
from .mesh_loader import LoadedMesh, MeshLoader
from .overhang import OverhangDetector, OverhangResult
from .propagate import PropagationResult, TreePropagator
from .radius import RadiusSolver
from .slicer import LayerStack, ModelSlicer
from .tips import ContactSampler, TipSet
from .volumes import CollisionCache

__all__ = ["PipelineResult", "TreeSupportPipeline"]


@dataclass
class PipelineResult:
    config: SupportConfig
    model: LoadedMesh | None = None
    stack: LayerStack | None = None
    overhang: OverhangResult | None = None
    cache: CollisionCache | None = None
    tips: TipSet | None = None
    propagation: PropagationResult | None = None
    graph: TreeGraph | None = None
    support_mesh: trimesh.Trimesh | None = None
    validation: dict = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    def summary(self) -> dict:
        d = {"timings": self.timings, "metrics": self.metrics}
        if self.stack is not None:
            d["slices"] = self.stack.stats()
        if self.overhang is not None:
            d["overhang"] = self.overhang.stats()
        if self.tips is not None:
            d["tips"] = self.tips.stats()
        if self.propagation is not None:
            d["tree"] = self.propagation.stats()
        if self.cache is not None:
            d["cache"] = self.cache.stats.to_dict()
        if self.validation:
            d["validation"] = self.validation
        return d


class _Timer:
    def __init__(self, store: dict, key: str, progress=None) -> None:
        self.store = store
        self.key = key
        self.progress = progress

    def __enter__(self):
        if self.progress:
            self.progress(self.key)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.store[self.key] = time.perf_counter() - self.t0
        return False


class TreeSupportPipeline:
    def __init__(self, config: SupportConfig,
                 debug: DebugWriter | None = None) -> None:
        self.config = config
        self.debug = debug or DebugWriter(config.debug_dir,
                                          config.debug_stages or None)

    # ------------------------------------------------------------------
    def run(self, mesh_or_path, build_mesh: bool = True, progress=None) -> PipelineResult:
        cfg = self.config
        res = PipelineResult(config=cfg)

        # --- STEP 1: load ---
        with _Timer(res.timings, "load", progress):
            loader = MeshLoader()
            if isinstance(mesh_or_path, trimesh.Trimesh):
                res.model = loader.from_trimesh(mesh_or_path)
            elif isinstance(mesh_or_path, LoadedMesh):
                res.model = mesh_or_path
            else:
                res.model = loader.load(mesh_or_path)

        # --- STEP 2: slice ---
        with _Timer(res.timings, "slice", progress):
            res.stack = ModelSlicer(cfg).slice(res.model.mesh)

        # --- STEP 3: overhang ---
        detector = OverhangDetector(cfg)
        with _Timer(res.timings, "overhang", progress):
            res.overhang = detector.detect(res.stack)

        # --- STEP 5: collision / reach ---
        res.cache = CollisionCache(cfg, res.stack)

        # --- STEP 4: tips ---
        sampler = ContactSampler(cfg)
        with _Timer(res.timings, "tips", progress):
            tip_layers = sorted({m - cfg.z_gap_layers_top
                                 for m in res.overhang.layers_with_demand})
            allowed = res.cache.tip_allowed_by_layer([j for j in tip_layers if j >= 0])
            res.tips = sampler.sample(res.stack, res.overhang,
                                      allowed_by_layer=allowed)

        # --- STEP 6-8: propagate + merge ---
        radius_solver = RadiusSolver(cfg)
        merger = TreeMerger(cfg, res.cache, radius_solver)
        propagator = TreePropagator(cfg, res.cache, radius_solver, merger)
        with _Timer(res.timings, "propagate", progress):
            res.propagation = propagator.propagate(
                res.stack, res.tips,
                keep_influence=self.debug.enabled("influence"),
            )
        res.graph = res.propagation.graph

        # --- STEP 11: bottom-up trim ---
        if cfg.bottom_up_trim:
            with _Timer(res.timings, "trim", progress):
                trimmed = propagator.trim_bottom_up(res.graph)
            res.metrics["bottom_up_trimmed_nodes"] = trimmed

        # --- STEP 9: centerline ---
        solver = CenterlineSolver(cfg)
        with _Timer(res.timings, "centerline", progress):
            solver.solve(res.graph)
        res.metrics.update(solver.stats())
        res.metrics.update(merger.stats())

        # --- STEP 12: smoothing ---
        if cfg.smoothing_iterations > 0:
            from .smoothing import CenterlineSmoother
            smoother = CenterlineSmoother(cfg)
            with _Timer(res.timings, "smoothing", progress):
                smoother.smooth(res.graph)
            res.metrics.update(smoother.stats())

        # --- STEP 13: 3D collision 補正 ---
        if cfg.collision3d_enabled:
            from .collision3d import Collision3DCorrector
            corrector = Collision3DCorrector(cfg, res.cache)
            with _Timer(res.timings, "collision3d", progress):
                corrector.correct(res.graph)
            res.metrics.update(corrector.stats())

        # --- STEP 10/14: メッシュ化 ---
        if build_mesh:
            from .meshbuilder import MeshBuilder
            builder = MeshBuilder(cfg)
            with _Timer(res.timings, "mesh", progress):
                res.support_mesh = builder.build(res.graph)
            res.metrics.update(builder.stats())

        # --- STEP 15: 検証 ---
        from .validator import Validator
        with _Timer(res.timings, "validate", progress):
            res.validation = Validator(cfg).validate(res)

        self._write_debug(res, detector, sampler, propagator)
        return res

    # ------------------------------------------------------------------
    def _write_debug(self, res: PipelineResult, detector, sampler,
                     propagator) -> None:
        dbg = self.debug
        if not dbg.active:
            return
        every = max(1, (res.stack.n_layers // 60) or 1)
        detector.write_debug(res.stack, res.overhang, dbg, every=every)
        sampler.write_debug(res.stack, res.overhang, res.tips, dbg)
        if dbg.enabled("collision"):
            res.cache.write_debug(dbg, radius=self.config.tip_radius, every=every)
        propagator.write_debug(res.stack, res.graph, dbg, every=every)
        if dbg.enabled("centerline"):
            dbg.write_obj_polylines("centerline", "centerline.obj",
                                    res.graph.centerline_polylines(),
                                    comment="tree support centerline")
            dbg.write_json("centerline", "graph.json", res.graph.to_dict())
        dbg.write_json("report", "summary.json", res.summary())
        dbg.write_json("report", "config.json", self.config.to_dict())
