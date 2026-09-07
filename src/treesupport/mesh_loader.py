"""STEP 1: メッシュ読み込みと診断。

* 座標変換は一切行わない (センタリング禁止)。``T_model == T_support`` を守るため。
* 読み込み時に watertight / manifold / winding を診断する。
* 同じ頂点を複数回参照するゼロ面積の面は除外する。頂点座標は変更しない。
* ``repair=True`` のときのみ、位相修復を試みる (頂点マージ・法線整合・穴埋め)。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

__all__ = ["MeshDiagnostics", "LoadedMesh", "MeshLoader"]


@dataclass(frozen=True)
class MeshDiagnostics:
    triangle_count: int
    vertex_count: int
    body_count: int
    is_watertight: bool
    is_winding_consistent: bool
    is_volume: bool
    euler_number: int
    volume: float
    area: float
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    open_edge_count: int

    @property
    def extents(self) -> tuple[float, float, float]:
        return (
            self.bounds_max[0] - self.bounds_min[0],
            self.bounds_max[1] - self.bounds_min[1],
            self.bounds_max[2] - self.bounds_min[2],
        )

    def to_dict(self) -> dict:
        d = {
            "triangle_count": self.triangle_count,
            "vertex_count": self.vertex_count,
            "body_count": self.body_count,
            "is_watertight": self.is_watertight,
            "is_winding_consistent": self.is_winding_consistent,
            "is_volume": self.is_volume,
            "euler_number": self.euler_number,
            "volume": self.volume,
            "area": self.area,
            "bounds_min": list(self.bounds_min),
            "bounds_max": list(self.bounds_max),
            "extents": list(self.extents),
            "open_edge_count": self.open_edge_count,
        }
        return d

    def summary(self) -> str:
        return (
            f"triangles={self.triangle_count} vertices={self.vertex_count} "
            f"bodies={self.body_count} watertight={self.is_watertight} "
            f"winding_ok={self.is_winding_consistent} open_edges={self.open_edge_count}\n"
            f"bbox min=({self.bounds_min[0]:.3f}, {self.bounds_min[1]:.3f}, {self.bounds_min[2]:.3f}) "
            f"max=({self.bounds_max[0]:.3f}, {self.bounds_max[1]:.3f}, {self.bounds_max[2]:.3f}) "
            f"extents=({self.extents[0]:.3f}, {self.extents[1]:.3f}, {self.extents[2]:.3f})\n"
            f"volume={self.volume:.4f} mm^3 area={self.area:.4f} mm^2"
        )


@dataclass
class LoadedMesh:
    """読み込んだモデルとその診断結果。"""

    mesh: trimesh.Trimesh
    source: str
    diagnostics: MeshDiagnostics

    @property
    def bounds(self) -> np.ndarray:
        return np.asarray(self.mesh.bounds, dtype=float)

    @property
    def z_min(self) -> float:
        return float(self.mesh.bounds[0][2])

    @property
    def z_max(self) -> float:
        return float(self.mesh.bounds[1][2])


class MeshLoader:
    """STL / OBJ / PLY / 3MF などを読み込む。"""

    SUPPORTED = (".stl", ".obj", ".ply", ".3mf", ".off", ".glb", ".gltf")

    def __init__(self, repair: bool = False) -> None:
        self.repair = repair

    # ------------------------------------------------------------------
    def load(self, path: str | Path) -> LoadedMesh:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"mesh file not found: {p}")
        # process=True で一致する頂点を溶接する。残る縮退面は from_trimesh で扱う。
        # 平行移動・スケール・センタリングは一切行わないため T_model は保存される。
        # STL は頂点を共有しない形式なので、溶接しないと watertight 判定が常に偽になる。
        obj = trimesh.load(p, force="mesh", process=True)
        mesh = self._as_trimesh(obj)
        return self.from_trimesh(mesh, source=str(p))

    def from_trimesh(self, mesh: trimesh.Trimesh, source: str = "<memory>") -> LoadedMesh:
        mesh = self._as_trimesh(mesh)
        if len(mesh.faces) == 0 or not np.isfinite(mesh.vertices).all():
            raise ValueError("mesh must contain finite triangle geometry")
        # STL vertex welding can leave faces such as (a, a, b). They have no
        # surface area but incorrectly add edge references to manifold checks.
        # Do not use a geometric height threshold: thin valid triangles must
        # survive, and the source mesh/file must remain unchanged.
        keep = np.all(np.diff(np.sort(mesh.faces, axis=1), axis=1) != 0, axis=1)
        removed = int(np.count_nonzero(~keep))
        if removed:
            mesh = mesh.copy()
            mesh.update_faces(keep)
            mesh.metadata["meshbloom_removed_collapsed_faces"] = removed
            if len(mesh.faces) == 0:
                raise ValueError("mesh contains only collapsed triangles")
        if self.repair:
            mesh = self._repair(mesh)
        return LoadedMesh(mesh=mesh, source=source, diagnostics=self.diagnose(mesh))

    # ------------------------------------------------------------------
    @staticmethod
    def _as_trimesh(obj) -> trimesh.Trimesh:
        if isinstance(obj, trimesh.Trimesh):
            return obj
        if isinstance(obj, trimesh.Scene):
            if len(obj.geometry) == 0:
                raise ValueError("scene contains no geometry")
            merged = trimesh.util.concatenate(
                [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
            )
            if merged is None:
                raise ValueError("scene contains no triangle mesh")
            return merged
        raise TypeError(f"unsupported mesh object: {type(obj)!r}")

    @staticmethod
    def _repair(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """位相修復。頂点座標そのものは変更しない (マージのみ)。"""
        mesh = mesh.copy()
        mesh.merge_vertices()
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        if not mesh.is_watertight:
            mesh.fill_holes()
        if not mesh.is_winding_consistent or mesh.volume < 0:
            mesh.fix_normals()
        return mesh

    @staticmethod
    def diagnose(mesh: trimesh.Trimesh) -> MeshDiagnostics:
        bounds = np.asarray(mesh.bounds, dtype=float)
        try:
            body_count = int(mesh.body_count)
        except Exception:  # pragma: no cover - trimesh の稀な失敗への保険
            body_count = 1
        try:
            open_edges = int(len(mesh.edges_sorted) - len(mesh.edges_unique) * 2 + len(mesh.edges_unique))
        except Exception:  # pragma: no cover
            open_edges = -1
        # より素直な open edge 数: 参照回数が 1 回だけの unique edge
        try:
            counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
            open_edges = int(np.count_nonzero(counts == 1))
        except Exception:  # pragma: no cover
            pass
        return MeshDiagnostics(
            triangle_count=int(len(mesh.faces)),
            vertex_count=int(len(mesh.vertices)),
            body_count=body_count,
            is_watertight=bool(mesh.is_watertight),
            is_winding_consistent=bool(mesh.is_winding_consistent),
            is_volume=bool(mesh.is_volume),
            euler_number=int(mesh.euler_number),
            volume=float(mesh.volume) if mesh.is_watertight else float("nan"),
            area=float(mesh.area),
            bounds_min=(float(bounds[0][0]), float(bounds[0][1]), float(bounds[0][2])),
            bounds_max=(float(bounds[1][0]), float(bounds[1][1]), float(bounds[1][2])),
            open_edge_count=open_edges,
        )
