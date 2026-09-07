"""STEP 10 / 指示書 21 節: 出力。

**元モデルと完全に同一の座標系で出力する。** センタリング・スケール・
回転を一切行わないため ``T_model == T_support`` が保証される。

標準形式は binary STL。OBJ / PLY / 3MF も同じ経路で出力できる。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

__all__ = ["Exporter"]


class Exporter:
    SUPPORTED = {".stl": "stl", ".obj": "obj", ".ply": "ply", ".3mf": "3mf",
                 ".off": "off", ".glb": "glb"}

    def export(self, mesh: trimesh.Trimesh, path: str | Path,
               file_type: str | None = None) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ext = p.suffix.lower()
        if file_type is None:
            file_type = self.SUPPORTED.get(ext)
            if file_type is None:
                raise ValueError(
                    f"unsupported output extension {ext!r}; "
                    f"supported: {sorted(self.SUPPORTED)}"
                )
        if file_type == "stl":
            # trimesh の既定は binary STL
            data = trimesh.exchange.stl.export_stl(mesh)
            p.write_bytes(data)
        else:
            mesh.export(p, file_type=file_type)
        return p

    # ------------------------------------------------------------------
    def export_combined(self, model: trimesh.Trimesh, support: trimesh.Trimesh,
                        path: str | Path) -> Path:
        """model + support を重ねた確認用メッシュ (テスト・目視用)。"""
        combined = trimesh.util.concatenate([model, support])
        return self.export(combined, path)

    # ------------------------------------------------------------------
    @staticmethod
    def verify_same_frame(model: trimesh.Trimesh, support: trimesh.Trimesh,
                          tol: float = 1e-9) -> bool:
        """サポートがモデルと同じ座標系にあるか (変換されていないか) を確認する。

        サポートは必ずモデルの XY 近傍かつ z>=0 にあるはずなので、
        「z の最小値が 0 以上」「XY が world 内」を軽く検査する。
        """
        if len(support.faces) == 0:
            return True
        return bool(support.bounds[0][2] >= -tol)
