"""デバッグ成果物の出力 (SVG / JSON / OBJ / PLY)。

指示書 25 節に対応。各処理段階の中間状態を外部ファイルとして確認できるようにする。

出力レイアウト::

    debug/
        slices/     layer_XXXX_model.svg
        overhang/   layer_XXXX_overhang.svg
        tips/       layer_XXXX_tips.svg, tips.json
        collision/  layer_XXXX_collision.svg
        influence/  layer_XXXX_influence.svg
        merge/      merge_events.json
        centerline/ centerline.obj, graph.json
        mesh/       branches.ply など
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from . import geometry2d as g2

__all__ = ["SvgLayer", "DebugWriter", "STAGES"]

#: 既知のステージ名
STAGES = (
    "slices",
    "overhang",
    "tips",
    "collision",
    "influence",
    "merge",
    "centerline",
    "mesh",
    "report",
)


@dataclass
class SvgLayer:
    """SVG に重ね描きする 1 レイヤ分の描画指定。"""

    name: str
    geometry: BaseGeometry | Sequence[BaseGeometry] | None = None
    points: Sequence[Sequence[float]] | None = None
    lines: Sequence[Sequence[Sequence[float]]] | None = None
    fill: str | None = "#3b82f6"
    stroke: str | None = "#1d4ed8"
    stroke_width: float = 0.06
    opacity: float = 0.45
    point_radius: float = 0.25
    point_fill: str = "#ef4444"

    def geometries(self) -> list[BaseGeometry]:
        if self.geometry is None:
            return []
        if isinstance(self.geometry, BaseGeometry):
            return [self.geometry]
        return [g for g in self.geometry if g is not None]


class DebugWriter:
    """デバッグ出力の窓口。``root`` が None なら全て no-op。"""

    def __init__(self, root: str | Path | None,
                 stages: Iterable[str] | None = None) -> None:
        self.root = Path(root) if root else None
        self.stages: set[str] | None = set(stages) if stages else None
        self._counters: dict[str, int] = {}

    # ------------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self.root is not None

    def enabled(self, stage: str) -> bool:
        if self.root is None:
            return False
        if self.stages is None:
            return True
        return stage in self.stages

    def stage_dir(self, stage: str) -> Path:
        assert self.root is not None
        d = self.root / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    def write_json(self, stage: str, name: str, payload) -> Path | None:
        if not self.enabled(stage):
            return None
        path = self.stage_dir(stage) / name
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                   default=_json_default), encoding="utf-8")
        return path

    def write_text(self, stage: str, name: str, text: str) -> Path | None:
        if not self.enabled(stage):
            return None
        path = self.stage_dir(stage) / name
        path.write_text(text, encoding="utf-8")
        return path

    def write_svg(self, stage: str, name: str, layers: Sequence[SvgLayer],
                  bounds: Sequence[float] | None = None,
                  title: str | None = None,
                  px_per_mm: float = 12.0,
                  margin_mm: float = 2.0) -> Path | None:
        if not self.enabled(stage):
            return None
        svg = render_svg(layers, bounds=bounds, title=title,
                         px_per_mm=px_per_mm, margin_mm=margin_mm)
        path = self.stage_dir(stage) / name
        path.write_text(svg, encoding="utf-8")
        return path

    def write_obj_polylines(self, stage: str, name: str,
                            polylines: Sequence[Sequence[Sequence[float]]],
                            comment: str | None = None) -> Path | None:
        """3D 折れ線群を OBJ (l 要素) として書き出す。"""
        if not self.enabled(stage):
            return None
        lines: list[str] = []
        if comment:
            lines.append(f"# {comment}")
        index = 1
        for poly in polylines:
            if len(poly) < 2:
                continue
            start = index
            for p in poly:
                lines.append(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
                index += 1
            idx = " ".join(str(i) for i in range(start, index))
            lines.append(f"l {idx}")
        path = self.stage_dir(stage) / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def write_ply_points(self, stage: str, name: str,
                         points: Sequence[Sequence[float]],
                         colors: Sequence[Sequence[int]] | None = None) -> Path | None:
        if not self.enabled(stage):
            return None
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        header = [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(pts)}",
            "property float x",
            "property float y",
            "property float z",
        ]
        if colors is not None:
            header += ["property uchar red", "property uchar green", "property uchar blue"]
        header.append("end_header")
        body = []
        for i, p in enumerate(pts):
            if colors is not None:
                c = colors[i]
                body.append(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}")
            else:
                body.append(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
        path = self.stage_dir(stage) / name
        path.write_text("\n".join(header + body) + "\n", encoding="utf-8")
        return path


# ----------------------------------------------------------------------
# SVG レンダリング
# ----------------------------------------------------------------------
def _polygon_path_data(poly: Polygon) -> str:
    def ring(coords) -> str:
        pts = list(coords)
        if not pts:
            return ""
        head = f"M {pts[0][0]:.4f} {pts[0][1]:.4f}"
        rest = " ".join(f"L {x:.4f} {y:.4f}" for x, y in pts[1:])
        return f"{head} {rest} Z"

    parts = [ring(poly.exterior.coords)]
    parts.extend(ring(r.coords) for r in poly.interiors)
    return " ".join(p for p in parts if p)


def render_svg(layers: Sequence[SvgLayer],
               bounds: Sequence[float] | None = None,
               title: str | None = None,
               px_per_mm: float = 12.0,
               margin_mm: float = 2.0) -> str:
    """複数レイヤを重ねた SVG 文字列を返す。Y 軸は上向き (数学的向き)。"""
    if bounds is None:
        collected: list[BaseGeometry] = []
        for layer in layers:
            collected.extend(layer.geometries())
        b = g2.bounds_union(collected)
        if b is None:
            pts: list[Sequence[float]] = []
            for layer in layers:
                if layer.points is not None:
                    pts.extend(layer.points)
            if pts:
                arr = np.asarray(pts, dtype=float)
                b = (float(arr[:, 0].min()), float(arr[:, 1].min()),
                     float(arr[:, 0].max()), float(arr[:, 1].max()))
            else:
                b = (0.0, 0.0, 1.0, 1.0)
        bounds = b

    minx, miny, maxx, maxy = bounds
    minx -= margin_mm
    miny -= margin_mm
    maxx += margin_mm
    maxy += margin_mm
    w_mm = max(maxx - minx, 1e-6)
    h_mm = max(maxy - miny, 1e-6)
    w_px = w_mm * px_per_mm
    h_px = h_mm * px_per_mm

    out: list[str] = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w_px:.2f}" '
        f'height="{h_px + 18:.2f}" viewBox="0 0 {w_px:.2f} {h_px + 18:.2f}">'
    )
    out.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    if title:
        out.append(
            f'<text x="4" y="13" font-family="monospace" font-size="11" '
            f'fill="#111111">{_escape(title)}</text>'
        )
    # mm -> px 変換 + Y 反転
    out.append(
        f'<g transform="translate(0,{h_px + 18:.4f}) scale({px_per_mm:.6f},{-px_per_mm:.6f}) '
        f'translate({-minx:.6f},{-miny:.6f})">'
    )

    for layer in layers:
        fill = layer.fill or "none"
        stroke = layer.stroke or "none"
        style = (
            f'fill="{fill}" fill-opacity="{layer.opacity}" fill-rule="evenodd" '
            f'stroke="{stroke}" stroke-width="{layer.stroke_width}" '
            f'stroke-linejoin="round" vector-effect="non-scaling-stroke"'
        )
        out.append(f'<g id="{_escape(layer.name)}" {style}>')
        for geom in layer.geometries():
            for poly in g2.polygons(geom):
                d = _polygon_path_data(poly)
                if d:
                    out.append(f'<path d="{d}"/>')
        if layer.lines:
            for line in layer.lines:
                if len(line) < 2:
                    continue
                pts = " ".join(f"{p[0]:.4f},{p[1]:.4f}" for p in line)
                out.append(
                    f'<polyline points="{pts}" fill="none" stroke="{stroke}" '
                    f'stroke-width="{layer.stroke_width}" vector-effect="non-scaling-stroke"/>'
                )
        if layer.points:
            for p in layer.points:
                out.append(
                    f'<circle cx="{p[0]:.4f}" cy="{p[1]:.4f}" r="{layer.point_radius}" '
                    f'fill="{layer.point_fill}" fill-opacity="0.95" stroke="none"/>'
                )
        out.append("</g>")

    out.append("</g>")
    out.append("</svg>")
    return "\n".join(out)


def _escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, BaseGeometry):
        return obj.wkt
    raise TypeError(f"not JSON serializable: {type(obj)!r}")
