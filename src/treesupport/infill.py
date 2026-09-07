"""Sampled solid infill, with the original exterior retained by boolean cutting.

Positive fields represent material. Gyroid/diamond wall distances are local
gradient approximations, not a constant-thickness offset or a slicer toolpath.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable

import numpy as np
import trimesh
from skimage.measure import marching_cubes


@dataclass
class InfillConfig:
    pattern: str = "gyroid"
    cell_size: float = 8.0
    wall_thickness: float = 1.2
    shell_thickness: float = 1.2
    pitch: float = 0.4
    max_grid_points: int = 2_000_000

    def __post_init__(self):
        if self.pattern not in ("gyroid", "diamond", "cubic"):
            raise ValueError("pattern must be gyroid, diamond or cubic")
        for name in ("cell_size", "wall_thickness", "shell_thickness", "pitch"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        if self.wall_thickness >= self.cell_size / 2:
            raise ValueError("wall_thickness must be less than half the cell size")
        if self.pitch > min(self.wall_thickness, self.shell_thickness) / 3 + 1e-9:
            raise ValueError("pitch must be <= one third of wall and shell thickness")
        if self.cell_size / self.pitch < 12:
            raise ValueError("at least 12 samples per cell are required")
        if self.max_grid_points < 1:
            raise ValueError("max_grid_points must be positive")


@dataclass
class InfillResult:
    model: trimesh.Trimesh
    infill: trimesh.Trimesh
    printable: trimesh.Trimesh
    report: dict


def lattice_field(points: np.ndarray, config: InfillConfig) -> np.ndarray:
    """World-anchored periodic field (mm), positive inside lattice material."""
    w = 2 * np.pi / config.cell_size
    x, y, z = (points * w).T
    if config.pattern == "cubic":
        d = np.abs(points % config.cell_size - config.cell_size / 2)
        dx, dy, dz = d.T
        distance = np.sqrt(np.minimum.reduce((dx*dx+dy*dy, dy*dy+dz*dz, dz*dz+dx*dx)))
        return config.wall_thickness / 2 - distance
    sx, sy, sz = np.sin(x), np.sin(y), np.sin(z)
    cx, cy, cz = np.cos(x), np.cos(y), np.cos(z)
    if config.pattern == "gyroid":
        f = sx*cy + sy*cz + sz*cx
        gx, gy, gz = cx*cy-sz*sx, cy*cz-sx*sy, cz*cx-sy*sz
    else:
        f = sx*sy*sz + sx*cy*cz + cx*sy*cz + cx*cy*sz
        gx = cx*sy*sz + cx*cy*cz - sx*sy*cz - sx*cy*sz
        gy = sx*cy*sz - sx*sy*cz + cx*cy*cz - cx*sy*sz
        gz = sx*sy*cz - sx*cy*sz - cx*sy*sz + cx*cy*cz
    distance = np.abs(f) / (w * np.maximum(np.sqrt(gx*gx+gy*gy+gz*gz), 0.1))
    return config.wall_thickness / 2 - distance


def _surface(field: np.ndarray, origin: np.ndarray, pitch: float) -> trimesh.Trimesh:
    if field.max() <= 0:
        return trimesh.Trimesh()
    # Avoid nearly zero samples producing coincident float32 vertices at
    # grid-aligned shell boundaries. Bias only this tiny tie region outward
    # from the positive solid, by much less than the sampling resolution.
    field = np.where(np.abs(field) < pitch*1e-4, -pitch*1e-4, field)
    vertices, faces, _, _ = marching_cubes(field, level=0,
                                          allow_degenerate=False)
    mesh = trimesh.Trimesh(vertices=vertices.astype(float)*pitch + origin, faces=faces, process=True)
    # Preserve relative orientation of enclosed voids (do not flip each body).
    if mesh.volume < 0:
        mesh.invert()
    if not mesh.is_volume:
        raise ValueError("Sampled surface is not a closed volume; use a finer pitch")
    return mesh


def generate_infill(model: trimesh.Trimesh, config: InfillConfig,
                    progress: Callable[[str], None] | None = None) -> InfillResult:
    config.__post_init__()
    tell = progress or (lambda message: None)
    if len(model.faces) == 0 or not np.isfinite(model.vertices).all() or not model.is_volume:
        raise ValueError("Infill requires a closed, consistently oriented solid mesh")
    origin = np.floor(model.bounds[0] / config.pitch) * config.pitch - 2*config.pitch
    dims = np.ceil((model.bounds[1] - origin) / config.pitch).astype(np.int64) + 3
    count = math.prod(int(n) for n in dims)
    if count > config.max_grid_points:
        raise ValueError(f"Grid needs {count:,} points (limit {config.max_grid_points:,}). "
                         "Increase pitch together with wall/shell thickness, or use a smaller model.")
    distance = np.empty(count, dtype=np.float32)
    lattice = np.empty(count, dtype=np.float32)
    tell(f"内部形状を計算中 / {count:,} grid points")
    for start in range(0, count, 4096):
        end = min(start + 4096, count)
        ijk = np.column_stack(np.unravel_index(np.arange(start, end), tuple(dims)))
        points = origin + ijk * config.pitch
        distance[start:end] = trimesh.proximity.signed_distance(model, points)
        lattice[start:end] = lattice_field(points, config)
        if start % (4096*8) == 0:
            tell(f"内部形状を計算中 / {end/count:.0%}")
    if distance.max() <= config.shell_thickness + config.pitch:
        raise ValueError("Model is too thin for this shell thickness and resolution")
    tell("空洞をメッシュ化中")
    # Only cut voids deeper than the shell. The original outer triangles remain
    # exact; the shell and internal walls are approximated at the grid pitch.
    void_field = np.minimum(distance-config.shell_thickness, -lattice).reshape(tuple(dims))
    void = _surface(void_field, origin, config.pitch)
    if len(void.faces) == 0:
        raise ValueError("No internal void resolved; reduce pitch or wall thickness")
    tell("外殻とインフィルを一体化中")
    printable = trimesh.boolean.difference([model, void], engine="manifold")
    if printable is None or not printable.is_volume:
        raise ValueError("Boolean operation did not produce a closed positive volume")
    # Standalone lattice overlaps half the shell, so its ends reach the shell
    # when assembled. This file is primarily for inspection / custom workflows.
    tell("インフィル単体をメッシュ化中")
    infill_field = np.minimum(distance-config.shell_thickness/2, lattice).reshape(tuple(dims))
    infill = _surface(infill_field, origin, config.pitch)
    if len(infill.faces) == 0:
        raise ValueError("No infill resolved at this resolution")
    return InfillResult(model, infill, printable, {
        "config": asdict(config), "grid_points": count,
        "model_volume_mm3": float(model.volume),
        "printable_volume_mm3": float(printable.volume),
        "material_fraction_including_shell": float(printable.volume / model.volume),
        "watertight": bool(printable.is_watertight),
        "warnings": ["壁厚・外殻厚は格子解像度に依存する近似値です。",
                     "密度は押出経路の充填率ではなく、外殻を含む形状の体積比です。",
                     "Simplify3D v5 のスライス結果と試験印刷で確認してください。"],
    })
