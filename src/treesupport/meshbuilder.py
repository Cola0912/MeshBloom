"""STEP 10/19: centerline + 半径 -> 3D tube メッシュ。

各 centerline 点に水平なXYリングを作り、隣接する円同士を
triangle strip で接続する。半径は2D collisionと同じ水平断面の半径。

* 水平リングはZ方向に単調に並び、曲がった太い枝でもリングが折り返さない。
* tip と root には**平面キャップ**を付ける。
  - root キャップは z=0 平面に一致し、ビルドプレートへ密着する。
  - tip キャップは ``z_tip = tip_layer * layer_height`` に一致するので、
    モデル下面との Z ギャップが構成的に保証される (半球キャップだと
    半径分だけ上へはみ出してギャップを侵食する)。
* 分岐部は「子の枝を 1 ノード分だけ親側へ延長し、親の tube の内側から
  生やす」ことで実体的に重ねる。MVP では重なった solid tube のままとし、
  必要に応じて ``union_mode`` で厳密ブーリアン / voxel union を選ぶ。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import trimesh

from .config import SupportConfig, UnionMode
from .graph import TreeGraph

__all__ = ["MeshBuilder"]


@dataclass
class _Path:
    """メッシュ化の単位となる 1 本の連続経路。"""

    points: np.ndarray          # (n, 3)
    radii: np.ndarray           # (n,)
    cap_bottom: bool
    cap_top: bool


@dataclass
class BuildStats:
    path_count: int = 0
    ring_count: int = 0
    vertex_count: int = 0
    face_count: int = 0
    dropped_paths: int = 0
    union_mode: str = "none"
    union_ok: bool = True

    def to_dict(self) -> dict:
        return {
            "mesh_path_count": self.path_count,
            "mesh_ring_count": self.ring_count,
            "mesh_vertex_count": self.vertex_count,
            "mesh_face_count": self.face_count,
            "mesh_dropped_paths": self.dropped_paths,
            "mesh_union_mode": self.union_mode,
            "mesh_union_ok": self.union_ok,
        }


class MeshBuilder:
    def __init__(self, config: SupportConfig) -> None:
        self.config = config
        self._stats = BuildStats(union_mode=config.union_mode.value)

    # ------------------------------------------------------------------
    def build(self, graph: TreeGraph) -> trimesh.Trimesh:
        paths = self._collect_paths(graph)
        self._stats.path_count = len(paths)

        meshes: list[trimesh.Trimesh] = []
        for path in paths:
            m = self._tube(path)
            if m is not None:
                meshes.append(m)
            else:
                self._stats.dropped_paths += 1

        if not meshes:
            return trimesh.Trimesh()

        merged = trimesh.util.concatenate(meshes)
        merged = self._apply_union(merged, meshes)

        self._stats.vertex_count = int(len(merged.vertices))
        self._stats.face_count = int(len(merged.faces))
        return merged

    # ------------------------------------------------------------------
    def _collect_paths(self, graph: TreeGraph) -> list[_Path]:
        """root から tip へ、分岐で区切った経路群を作る。"""
        lh = graph.layer_height
        out: list[_Path] = []

        # (開始ノード, 接続元ノード) のスタック
        stack: list[tuple[int, int | None]] = [(r, None) for r in graph.root_ids]
        while stack:
            start, junction = stack.pop()
            if start not in graph.nodes:
                continue

            chain: list[int] = [start]
            cur = start
            while True:
                children = [c for c in graph.nodes[cur].child_ids if c in graph.nodes]
                if len(children) == 1:
                    cur = children[0]
                    chain.append(cur)
                    continue
                if len(children) > 1:
                    for c in children:
                        stack.append((c, cur))
                break

            prefix: list[int] = []
            if junction is not None:
                # 親の tube の内側から生やすため、分岐ノードとその 1 つ下を前置する
                jnode = graph.nodes.get(junction)
                if jnode is not None:
                    for pid in jnode.parent_ids[:1]:
                        if pid in graph.nodes:
                            prefix.append(pid)
                    prefix.append(junction)

            ids = prefix + chain
            pts = []
            radii = []
            r_head = graph.nodes[chain[0]].radius
            # 前置部分は親 tube の**厳密に内側**に収める (面の一致を避ける)
            r_prefix = max(self.config.min_printable_feature * 0.5, r_head * 0.98)
            for i, nid in enumerate(ids):
                node = graph.nodes[nid]
                if node.position is None:
                    continue
                pts.append((node.position[0], node.position[1], node.layer * lh))
                radii.append(min(node.radius, r_prefix) if i < len(prefix)
                             else node.radius)

            if len(pts) < 2:
                if len(pts) == 1:
                    out.append(_Path(np.asarray(pts * 2, dtype=float),
                                     np.asarray(radii * 2, dtype=float),
                                     cap_bottom=True, cap_top=True))
                continue

            points = np.asarray(pts, dtype=float)
            rr = np.asarray(radii, dtype=float)
            points, rr = self._simplify(points, rr)
            if graph.nodes[chain[-1]].is_tip:
                # Keep junction overlaps intact; only shape the free terminal.
                min_z = graph.nodes[chain[0]].layer * lh
                points, rr = self._contact_profile(points, rr, min_z)
            # 分岐した子の tube も必ず閉じる。下端キャップは親 tube の内側に
            # 埋まるので実害が無く、各成分が閉じていることで
            # watertight / manifold を満たせる。
            out.append(_Path(points, rr, cap_bottom=True, cap_top=True))
        return out

    def _contact_profile(self, points, radii, min_z=0.):
        cfg = self.config
        if cfg.contact_shape == "flat" and cfg.contact_diameter == 0:
            return points, radii
        top = points[-1, 2]
        bottom = max(points[0, 2], min_z, top-cfg.contact_height)
        if top-bottom <= 1e-10:
            return points, radii
        # Additional rings approximate the shoulder without extending the top
        # or growing beyond the collision-checked tube at any height.
        extra = np.linspace(bottom, top, 13)
        extra = extra[np.min(np.abs(extra[:, None]-points[None, :, 2]), axis=1) > 1e-8]
        zs = np.sort(np.r_[points[:, 2], extra])
        result = np.column_stack([np.interp(zs, points[:, 2], points[:, axis]) for axis in range(3)])
        rr = np.interp(zs, points[:, 2], radii)
        mask = zs >= bottom
        t = (zs[mask]-bottom)/(top-bottom)
        contact = cfg.effective_contact_diameter/2
        base = float(np.interp(bottom, points[:, 2], radii))
        if cfg.contact_shape == "rounded":
            weight = np.sqrt(np.maximum(0., 1-t*t))
        elif cfg.contact_shape == "flat":
            weight = np.maximum(0., 1-2*t)  # straight contact neck in the top half
        else:
            weight = 1-t
        rr[mask] = np.minimum(rr[mask], contact + (base-contact)*weight)
        return result, rr

    # ------------------------------------------------------------------
    def _simplify(self, points: np.ndarray,
                  radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """直線的で半径変化の小さい区間の ring を間引く。"""
        tol = self.config.mesh_simplify_deviation
        if tol <= 0.0 or len(points) < 3:
            return points, radii

        keep = [0]
        anchor = 0
        for i in range(1, len(points) - 1):
            a = points[anchor]
            b = points[i + 1]
            p = points[i]
            ab = b - a
            denom = float(np.dot(ab, ab))
            if denom <= 1e-18:
                dev = float(np.linalg.norm(p - a))
            else:
                t = float(np.dot(p - a, ab)) / denom
                t = min(1.0, max(0.0, t))
                dev = float(np.linalg.norm(p - (a + t * ab)))
            r_lin = radii[anchor] + (radii[i + 1] - radii[anchor]) * 0.5
            if dev > tol or abs(radii[i] - r_lin) > tol:
                keep.append(i)
                anchor = i
        keep.append(len(points) - 1)
        idx = np.asarray(sorted(set(keep)), dtype=int)
        return points[idx], radii[idx]

    # ------------------------------------------------------------------
    def _tube(self, path: _Path) -> trimesh.Trimesh | None:
        cfg = self.config
        pts = path.points
        radii = np.maximum(path.radii, cfg.min_printable_feature * 0.5)
        n = len(pts)
        if n < 2 or pts[-1, 2] - pts[0, 2] <= 1e-10:
            return None

        # Collision/reach radii are XY disks at fixed Z. Horizontal rings
        # preserve that convention, including at sharp bends near tips/roots.
        # Perpendicular rings can fold over each other, dip below the plate,
        # and extend above the tip even when the end cap itself is horizontal.
        normals = np.tile([0.0, 0.0, 1.0], (n, 1))
        # キャップ端の ring は**水平**にする。
        # 傾いた ring のままだと端の頂点が z を r*sin(tilt) だけはみ出し、
        #   - tip 側: Z ギャップを侵食する
        #   - root 側: ビルドプレートより下へ潜る
        # という致命的な問題が起きる。水平にすれば端点の z は節点の z と厳密に一致する。
        if path.cap_bottom:
            normals[0] = np.array([0.0, 0.0, 1.0])
        if path.cap_top:
            normals[-1] = np.array([0.0, 0.0, 1.0])
        seg = cfg.ring_segments

        # 参照ベクトルを平行移動フレームで運ぶ
        refs = np.zeros((n, 3), dtype=float)
        refs[0] = self._initial_reference(normals[0])
        for i in range(1, n):
            r = refs[i - 1] - normals[i] * float(np.dot(refs[i - 1], normals[i]))
            norm = float(np.linalg.norm(r))
            if norm < 1e-9:
                r = self._initial_reference(normals[i])
                norm = float(np.linalg.norm(r))
            refs[i] = r / norm

        angles = np.arange(seg, dtype=float) * (2.0 * math.pi / seg)
        cos_a = np.cos(angles)[:, None]
        sin_a = np.sin(angles)[:, None]

        vertices: list[np.ndarray] = []
        for i in range(n):
            u = refs[i]
            v = np.cross(normals[i], u)
            v /= max(1e-12, float(np.linalg.norm(v)))
            ring = pts[i] + radii[i] * (cos_a * u + sin_a * v)
            vertices.append(ring)

        verts = np.vstack(vertices)
        faces: list[tuple[int, int, int]] = []
        for i in range(n - 1):
            lo = i * seg
            hi = (i + 1) * seg
            for k in range(seg):
                k2 = (k + 1) % seg
                faces.append((lo + k, lo + k2, hi + k))
                faces.append((lo + k2, hi + k2, hi + k))

        verts_list = [verts]
        offset = len(verts)
        if path.cap_bottom:
            center_idx = offset
            verts_list.append(pts[0].reshape(1, 3))
            offset += 1
            for k in range(seg):
                k2 = (k + 1) % seg
                faces.append((center_idx, k2, k))
        if path.cap_top:
            center_idx = offset
            verts_list.append(pts[-1].reshape(1, 3))
            offset += 1
            base = (n - 1) * seg
            for k in range(seg):
                k2 = (k + 1) % seg
                faces.append((center_idx, base + k, base + k2))

        mesh = trimesh.Trimesh(
            vertices=np.vstack(verts_list),
            faces=np.asarray(faces, dtype=np.int64),
            process=False,
        )
        self._stats.ring_count += n
        return mesh

    # ------------------------------------------------------------------
    @staticmethod
    def _ring_normals(pts: np.ndarray) -> np.ndarray:
        n = len(pts)
        d = pts[1:] - pts[:-1]
        lengths = np.linalg.norm(d, axis=1, keepdims=True)
        lengths[lengths < 1e-12] = 1.0
        d = d / lengths

        normals = np.zeros((n, 3), dtype=float)
        normals[0] = d[0]
        normals[-1] = d[-1]
        for i in range(1, n - 1):
            v = d[i - 1] + d[i]
            norm = float(np.linalg.norm(v))
            normals[i] = d[i] if norm < 1e-9 else v / norm
        return normals

    @staticmethod
    def _initial_reference(normal: np.ndarray) -> np.ndarray:
        axis = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(normal, axis))) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        r = axis - normal * float(np.dot(axis, normal))
        norm = float(np.linalg.norm(r))
        if norm < 1e-9:  # pragma: no cover
            r = np.array([0.0, 0.0, 1.0]) - normal * float(normal[2])
            norm = float(np.linalg.norm(r))
        return r / norm

    # ------------------------------------------------------------------
    def _apply_union(self, merged: trimesh.Trimesh,
                     parts: list[trimesh.Trimesh]) -> trimesh.Trimesh:
        mode = self.config.union_mode
        if mode is UnionMode.NONE or len(parts) < 2:
            return merged
        try:
            from .booleanop import union_meshes
            result = union_meshes(parts, mode, self.config)
            if result is not None and len(result.faces) > 0:
                return result
            self._stats.union_ok = False
        except Exception:  # pragma: no cover - 環境依存
            self._stats.union_ok = False
        return merged

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return self._stats.to_dict()
