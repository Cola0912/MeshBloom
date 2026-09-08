"""STEP 14: 分岐部のメッシュ結合。

* ``UnionMode.BOOLEAN``: manifold3d による厳密ブーリアン union。
  途中でTrimeshへ戻さず、Manifold側に全入力の結合を任せる。
* ``UnionMode.VOXEL``: 各セグメントを round-cone (両端半径の異なるカプセル) の
  符号付き距離場として評価し、その最小値の等値面を marching cubes で取り出す。
  厳密ブーリアンで開発が止まるより、まず voxel を優先してよい (指示書 20 節)。

voxel の SDF は「線分への垂線足で半径を線形補間する」近似を使う。
真の round-cone SDF より僅かに大きい値を返しうるが、
``pitch`` に対して十分小さい誤差であり MVP では許容する。
"""

from __future__ import annotations

import numpy as np
import trimesh

from .config import SupportConfig, UnionMode

__all__ = ["union_meshes", "voxel_union_from_segments"]


def union_meshes(parts: list[trimesh.Trimesh], mode: UnionMode,
                 config: SupportConfig) -> trimesh.Trimesh | None:
    if mode is UnionMode.BOOLEAN:
        return _boolean_union(parts)
    if mode is UnionMode.VOXEL:
        return _voxel_union_meshes(parts, config)
    return None


# ----------------------------------------------------------------------
def _boolean_union(parts: list[trimesh.Trimesh]) -> trimesh.Trimesh | None:
    # Let Manifold perform the complete reduction. Repeated Trimesh round-trips
    # can merge nearly coincident vertices and invalidate intermediate meshes.
    if not parts:
        return None
    result = trimesh.boolean.union(parts, engine="manifold")
    # The adapter welds Manifold's float32 output. Coincident vertices can
    # leave zero-area (a, a, b) faces at a junction. Remove only those faces;
    # do not fill holes or discard thin, valid geometry. The validator still
    # checks the resulting solid before the app allows export.
    keep = np.all(np.diff(np.sort(result.faces, axis=1), axis=1) != 0, axis=1)
    if not keep.all():
        result.update_faces(keep)
    return result


# ----------------------------------------------------------------------
def _voxel_union_meshes(parts: list[trimesh.Trimesh],
                        config: SupportConfig) -> trimesh.Trimesh | None:
    """メッシュ群を voxel 化して union する (汎用フォールバック)。"""
    pitch = config.voxel_pitch
    grids = [p.voxelized(pitch=pitch).fill() for p in parts]
    combined = grids[0]
    for g in grids[1:]:
        combined = combined.union(g)
    return combined.marching_cubes


# ----------------------------------------------------------------------
def voxel_union_from_segments(segments, pitch: float,
                              padding: float = 2.0) -> trimesh.Trimesh | None:
    """``(p0, r0, p1, r1)`` の列から SDF を作り marching cubes で復元する。

    centerline から直接 union メッシュを得る経路 (メッシュ化を経由しない)。
    """
    from skimage import measure

    segs = list(segments)
    if not segs:
        return None

    pts = np.array([[s[0], s[2]] for s in segs], dtype=float).reshape(-1, 3)
    radii = np.array([[s[1], s[3]] for s in segs], dtype=float).reshape(-1)
    lo = pts.min(axis=0) - (radii.max() + padding)
    hi = pts.max(axis=0) + (radii.max() + padding)

    dims = np.maximum(np.ceil((hi - lo) / pitch).astype(int) + 1, 2)
    sdf = np.full(tuple(dims), 1e9, dtype=np.float32)

    xs = lo[0] + np.arange(dims[0]) * pitch
    ys = lo[1] + np.arange(dims[1]) * pitch
    zs = lo[2] + np.arange(dims[2]) * pitch

    for (p0, r0, p1, r1) in segs:
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        rmax = max(r0, r1)
        bmin = np.minimum(p0, p1) - rmax - pitch
        bmax = np.maximum(p0, p1) + rmax + pitch
        i0 = np.maximum(((bmin - lo) / pitch).astype(int), 0)
        i1 = np.minimum(((bmax - lo) / pitch).astype(int) + 1, dims - 1)
        if np.any(i0 > i1):
            continue

        gx, gy, gz = np.meshgrid(
            xs[i0[0]:i1[0] + 1], ys[i0[1]:i1[1] + 1], zs[i0[2]:i1[2] + 1],
            indexing="ij",
        )
        q = np.stack([gx, gy, gz], axis=-1)
        d = p1 - p0
        denom = float(np.dot(d, d))
        if denom < 1e-18:
            t = np.zeros(q.shape[:-1])
        else:
            t = np.clip(((q - p0) @ d) / denom, 0.0, 1.0)
        proj = p0 + t[..., None] * d
        r = r0 + (r1 - r0) * t
        dist = np.linalg.norm(q - proj, axis=-1) - r

        block = sdf[i0[0]:i1[0] + 1, i0[1]:i1[1] + 1, i0[2]:i1[2] + 1]
        np.minimum(block, dist.astype(np.float32), out=block)

    if not (sdf.min() < 0.0 < sdf.max()):
        return None

    verts, faces, _, _ = measure.marching_cubes(sdf, level=0.0)
    verts = verts * pitch + lo
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.fix_normals()
    return mesh
