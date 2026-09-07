"""テスト共通のヘルパ。"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from treesupport.centerline import CenterlineSolver
from treesupport.config import SupportConfig
from treesupport.graph import TreeGraph
from treesupport.merge import TreeMerger
from treesupport.overhang import OverhangDetector, OverhangResult
from treesupport.propagate import PropagationResult, TreePropagator
from treesupport.radius import RadiusSolver
from treesupport.slicer import LayerStack, ModelSlicer
from treesupport.tips import ContactSampler, TipSet
from treesupport.volumes import CollisionCache


@dataclass
class Built:
    config: SupportConfig
    stack: LayerStack
    overhang: OverhangResult
    cache: CollisionCache
    tips: TipSet
    propagation: PropagationResult
    graph: TreeGraph
    merger: TreeMerger
    propagator: TreePropagator
    solver: CenterlineSolver | None = None


def build_tree(mesh, solve_centerline: bool = True, trim: bool = False,
               use_reach: bool = True, record_trace: bool = False,
               record_geometry: bool = False, **cfg_kwargs) -> Built:
    """STEP 1-9 をまとめて実行する (メッシュ化は行わない)。"""
    cfg = SupportConfig(**cfg_kwargs)
    stack = ModelSlicer(cfg).slice(mesh)
    overhang = OverhangDetector(cfg).detect(stack)
    cache = CollisionCache(cfg, stack)

    tip_layers = sorted({m - cfg.z_gap_layers_top
                         for m in overhang.layers_with_demand})
    allowed = cache.tip_allowed_by_layer([j for j in tip_layers if j >= 0])
    tips = ContactSampler(cfg).sample(stack, overhang, allowed_by_layer=allowed)

    radius_solver = RadiusSolver(cfg)
    merger = TreeMerger(cfg, cache, radius_solver,
                        record_geometry=record_geometry)
    propagator = TreePropagator(cfg, cache, radius_solver, merger,
                                use_reach=use_reach, record_trace=record_trace)
    propagation = propagator.propagate(stack, tips)

    if trim:
        propagator.trim_bottom_up(propagation.graph)

    solver = None
    if solve_centerline:
        solver = CenterlineSolver(cfg)
        solver.solve(propagation.graph)

    return Built(config=cfg, stack=stack, overhang=overhang, cache=cache,
                 tips=tips, propagation=propagation, graph=propagation.graph,
                 merger=merger, propagator=propagator, solver=solver)


@pytest.fixture(scope="session")
def fast_config() -> dict:
    """テストを速く回すための粗めの設定。"""
    return dict(layer_height=0.4, tip_spacing=3.0)
