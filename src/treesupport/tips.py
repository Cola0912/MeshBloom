"""STEP 4: Contact / Tip サンプリング。

support demand polygon を巨大な板状サポートにせず、ツリーの先端 (tip) 群へ変換する。

指示書 9 節の 3 段階構成:

1. **通常サンプリング** — グローバル整列した正方格子 (レイヤ間で XY が揃うので merge しやすい)
2. **輪郭頂点補助 tip** — 鋭角頂点 + 輪郭上の等間隔点
   (CuraEngine Issue #2037 のような corner omission 対策)
3. **coverage 不足領域への追加 tip** — 被覆されなかった領域へ反復追加

tip は「サポートレイヤ」に属する。オーバーハングがモデルレイヤ ``m`` にあるとき

    j = m - z_gap_layers_top

に置かれ、tip の材料上端 ``j*lh`` とモデル下面 ``m*lh`` の差が Z ギャップになる。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPoint, Point
from shapely.geometry.base import BaseGeometry

from . import geometry2d as g2
from .config import SupportConfig
from .debug_io import DebugWriter, SvgLayer
from .overhang import OverhangResult
from .slicer import LayerStack

__all__ = ["ContactPoint", "TipSet", "ContactSampler"]


@dataclass
class ContactPoint:
    """1 本の tip。"""

    id: int
    layer: int              # サポートレイヤ番号 (z = layer * layer_height)
    x: float
    y: float
    z: float
    overhang_layer: int     # 支える対象のモデルレイヤ
    supported_area: float = 0.0
    source: str = "grid"    # grid | corner | contour | fill

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "layer": self.layer, "x": self.x, "y": self.y, "z": self.z,
            "overhang_layer": self.overhang_layer,
            "supported_area": self.supported_area, "source": self.source,
        }


@dataclass
class TipSet:
    """全 tip とサンプリング統計。"""

    tips: list[ContactPoint] = field(default_factory=list)
    #: サポートレイヤ -> そのレイヤの tip
    by_layer: dict[int, list[ContactPoint]] = field(default_factory=dict)
    #: モデルレイヤ -> 被覆できなかった demand 領域
    unsupportable: dict[int, BaseGeometry] = field(default_factory=dict)
    #: モデルレイヤ -> tip で被覆されなかった (がサポート可能な) demand 領域
    uncovered: dict[int, BaseGeometry] = field(default_factory=dict)
    demand_area: float = 0.0
    covered_area: float = 0.0
    unsupportable_area: float = 0.0
    #: min_overhang_area 以上の大きさで残った未被覆面積
    uncovered_significant_area: float = 0.0

    def __len__(self) -> int:
        return len(self.tips)

    @property
    def coverage_ratio(self) -> float:
        """厳密な被覆率。min_overhang_area 未満のかけらも未被覆として数える。"""
        if self.demand_area <= 0.0:
            return 1.0
        return self.covered_area / self.demand_area

    @property
    def effective_coverage_ratio(self) -> float:
        """min_overhang_area 以上の未被覆領域だけを不足とみなした被覆率。"""
        if self.demand_area <= 0.0:
            return 1.0
        return 1.0 - self.uncovered_significant_area / self.demand_area

    def stats(self) -> dict:
        return {
            "tip_count": len(self.tips),
            "tip_layers": len(self.by_layer),
            "demand_area": self.demand_area,
            "covered_area": self.covered_area,
            "uncovered_area": max(0.0, self.demand_area - self.covered_area),
            "uncovered_significant_area": self.uncovered_significant_area,
            "unsupportable_area": self.unsupportable_area,
            "coverage_ratio": self.coverage_ratio,
            "effective_coverage_ratio": self.effective_coverage_ratio,
            "by_source": {
                s: sum(1 for t in self.tips if t.source == s)
                for s in ("grid", "corner", "contour", "fill")
            },
        }


class ContactSampler:
    """support demand -> tip 群。"""

    def __init__(self, config: SupportConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    def sample(self, stack: LayerStack, overhang: OverhangResult,
               allowed_by_layer: dict[int, BaseGeometry] | None = None) -> TipSet:
        """全レイヤの demand から tip を生成する。

        Parameters
        ----------
        allowed_by_layer:
            サポートレイヤ番号 -> tip 中心が存在してよい領域。
            ``None`` なら制限なし (STEP 4 単体テスト用)。
            STEP 5 以降は ``CollisionCache`` から得た reach 領域を渡す。
        """
        cfg = self.config
        result = TipSet()
        next_id = 0

        for m in range(len(overhang)):
            demand = overhang[m]
            if demand.is_empty:
                continue
            j = m - cfg.z_gap_layers_top
            result.demand_area += demand.area
            if j < 0:
                # ビルドプレートに近すぎてギャップを確保できない
                result.unsupportable[m] = demand
                result.unsupportable_area += demand.area
                continue

            region = demand
            if allowed_by_layer is not None:
                allowed = allowed_by_layer.get(j, g2.EMPTY)
                region = g2.intersection(demand, allowed)
                blocked = g2.difference(demand, allowed)
                if not blocked.is_empty:
                    result.unsupportable[m] = blocked
                    result.unsupportable_area += blocked.area
            if region.is_empty:
                continue

            points = self.sample_region(region)
            z = stack.z_bottom(j)
            layer_tips: list[ContactPoint] = []
            for (px, py, src) in points:
                tip = ContactPoint(id=next_id, layer=j, x=px, y=py, z=z,
                                   overhang_layer=m, source=src)
                next_id += 1
                layer_tips.append(tip)
            if not layer_tips:
                continue

            covered = self._covered_area(region, layer_tips)
            result.covered_area += covered
            uncovered = self._uncovered(region, layer_tips)
            if not uncovered.is_empty:
                result.uncovered[m] = uncovered
                result.uncovered_significant_area += uncovered.area

            result.tips.extend(layer_tips)
            result.by_layer.setdefault(j, []).extend(layer_tips)

        return result

    # ------------------------------------------------------------------
    def sample_region(self, region: BaseGeometry) -> list[tuple[float, float, str]]:
        """1 レイヤ分の領域から tip 候補を 3 段階で生成する。"""
        cfg = self.config
        candidates: list[tuple[float, float, str]] = []

        for poly in g2.polygons(region):
            candidates.extend(self._corner_points(poly))
        for poly in g2.polygons(region):
            candidates.extend(self._grid_points(poly))
        for poly in g2.polygons(region):
            candidates.extend(self._contour_points(poly))

        accepted = self._dedupe(candidates, cfg.tip_min_distance)
        accepted = self._fill_uncovered(region, accepted)
        return accepted

    # --- stage 1: 格子 -------------------------------------------------
    def _grid_points(self, poly) -> list[tuple[float, float, str]]:
        s = self.config.tip_spacing
        minx, miny, maxx, maxy = poly.bounds
        i0 = math.floor(minx / s)
        i1 = math.ceil(maxx / s)
        j0 = math.floor(miny / s)
        j1 = math.ceil(maxy / s)
        if (i1 - i0 + 1) * (j1 - j0 + 1) > 4_000_000:  # pragma: no cover - 安全弁
            return []
        xs = np.arange(i0, i1 + 1, dtype=float) * s
        ys = np.arange(j0, j1 + 1, dtype=float) * s
        if xs.size == 0 or ys.size == 0:
            return []
        gx, gy = np.meshgrid(xs, ys)
        gx = gx.ravel()
        gy = gy.ravel()
        mask = shapely.contains_xy(poly, gx, gy)
        return [(float(x), float(y), "grid") for x, y in zip(gx[mask], gy[mask])]

    # --- stage 2: 輪郭 -------------------------------------------------
    def _corner_points(self, poly) -> list[tuple[float, float, str]]:
        """鋭角頂点を内側へわずかにずらして tip 候補にする。"""
        cfg = self.config
        out: list[tuple[float, float, str]] = []
        # 内側への inset は行わない。鋭い楔では inset が頂点を大きく後退させ
        # (後退量 = inset / sin(半角))、コーナーの取りこぼしを招くため。
        # tip 中心が demand の縁に乗ること自体は問題ない
        # (モデルとのクリアランスは region = demand & allowed 側で担保済み)。
        target = poly
        limit = math.radians(cfg.tip_sharp_angle)

        for ring in [poly.exterior, *poly.interiors]:
            coords = list(ring.coords)[:-1]
            n = len(coords)
            if n < 3:
                continue
            for i in range(n):
                prev = np.asarray(coords[i - 1], dtype=float)
                cur = np.asarray(coords[i], dtype=float)
                nxt = np.asarray(coords[(i + 1) % n], dtype=float)
                v1 = prev - cur
                v2 = nxt - cur
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 < 1e-9 or n2 < 1e-9:
                    continue
                cosang = float(np.dot(v1, v2) / (n1 * n2))
                ang = math.acos(max(-1.0, min(1.0, cosang)))
                if ang > limit:
                    continue
                px, py = g2.project_point(target, (float(cur[0]), float(cur[1])))
                out.append((px, py, "corner"))
        return out

    def _contour_points(self, poly) -> list[tuple[float, float, str]]:
        """輪郭上の等間隔点。細長い領域が格子から漏れるのを防ぐ。"""
        cfg = self.config
        step = max(1e-3, cfg.tip_spacing * cfg.tip_corner_spacing_factor)
        target = poly

        out: list[tuple[float, float, str]] = []
        for ring in [poly.exterior, *poly.interiors]:
            line = LineString(ring.coords)
            length = line.length
            if length <= 0.0:
                continue
            n = max(1, int(round(length / step)))
            for i in range(n):
                d = (i + 0.5) * (length / n)
                p = line.interpolate(d)
                px, py = g2.project_point(target, (p.x, p.y))
                out.append((px, py, "contour"))
        return out

    # --- stage 3: coverage 補完 ---------------------------------------
    def _fill_uncovered(self, region: BaseGeometry,
                        accepted: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
        cfg = self.config
        r = cfg.coverage_radius
        min_area = max(cfg.min_overhang_area, cfg.area_epsilon)
        points = list(accepted)

        for _ in range(max(0, cfg.tip_fill_iterations)):
            uncovered = self._uncovered(region, points, min_area=min_area)
            if uncovered.is_empty:
                break
            added = False
            for part in g2.polygons(uncovered):
                if part.area < min_area:
                    continue
                px, py = self._deep_point(part)
                if any(math.hypot(px - q[0], py - q[1]) < 1e-6 for q in points):
                    continue
                points.append((px, py, "fill"))
                added = True
            if not added:
                break
        return points

    @staticmethod
    def _deep_point(poly) -> tuple[float, float]:
        """ポリゴン内部の「深い」点 (境界から最も遠いあたり) を返す。"""
        # 二分探索的に内側へ削り、消える直前の代表点を採る。
        lo, hi = 0.0, math.sqrt(poly.area) * 0.5 + 1e-6
        best = poly.representative_point()
        for _ in range(12):
            mid = (lo + hi) * 0.5
            eroded = poly.buffer(-mid, quad_segs=4)
            if eroded.is_empty:
                hi = mid
            else:
                lo = mid
                best = eroded.representative_point()
        return (float(best.x), float(best.y))

    # --- 共通 ----------------------------------------------------------
    @staticmethod
    def _dedupe(candidates, min_dist: float) -> list[tuple[float, float, str]]:
        """優先順 (先頭優先) に、最小距離を満たす点だけ採用する。

        空間ハッシュ (セル幅 = min_dist) を使うので候補数に対して線形。
        """
        if min_dist <= 0.0:
            return list(candidates)
        cell = min_dist
        d2 = min_dist * min_dist
        buckets: dict[tuple[int, int], list[tuple[float, float]]] = {}
        accepted: list[tuple[float, float, str]] = []

        for (x, y, src) in candidates:
            cx = int(math.floor(x / cell))
            cy = int(math.floor(y / cell))
            ok = True
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for (ax, ay) in buckets.get((cx + dx, cy + dy), ()):
                        if (ax - x) ** 2 + (ay - y) ** 2 < d2:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                accepted.append((x, y, src))
                buckets.setdefault((cx, cy), []).append((x, y))
        return accepted

    def _coverage_geometry(self, points) -> BaseGeometry:
        if not points:
            return g2.EMPTY
        r = self.config.coverage_radius
        pts = MultiPoint([Point(p[0], p[1]) for p in points])
        return g2.clean(pts.buffer(r, quad_segs=self.config.quad_segs))

    def _uncovered(self, region: BaseGeometry, points,
                   min_area: float | None = None) -> BaseGeometry:
        if min_area is None:
            min_area = max(self.config.min_overhang_area, self.config.area_epsilon)
        xy = [(p[0], p[1]) if not isinstance(p, ContactPoint) else (p.x, p.y)
              for p in points]
        cover = self._coverage_geometry(xy)
        return g2.drop_small(g2.difference(region, cover), min_area)

    def _covered_area(self, region: BaseGeometry, tips) -> float:
        cover = self._coverage_geometry([(t.x, t.y) for t in tips])
        return g2.area(g2.intersection(region, cover))

    # ------------------------------------------------------------------
    def write_debug(self, stack: LayerStack, overhang: OverhangResult,
                    tipset: TipSet, debug: DebugWriter) -> int:
        if not debug.enabled("tips"):
            return 0
        written = 0
        for j, tips in sorted(tipset.by_layer.items()):
            m = tips[0].overhang_layer
            demand = overhang[m]
            debug.write_svg(
                "tips", f"layer_{j:04d}_tips.svg",
                [
                    SvgLayer("model", stack[m], fill="#94a3b8", stroke="#334155",
                             opacity=0.35),
                    SvgLayer("demand", demand, fill="#fca5a5", stroke="#b91c1c",
                             opacity=0.6),
                    SvgLayer("coverage",
                             self._coverage_geometry([(t.x, t.y) for t in tips]),
                             fill="#86efac", stroke="#16a34a", opacity=0.35),
                    SvgLayer("tips", None, points=[(t.x, t.y) for t in tips],
                             point_radius=self.config.tip_radius,
                             point_fill="#1d4ed8"),
                ],
                bounds=stack.xy_bounds,
                title=(f"support layer {j:04d} (overhang layer {m}) "
                       f"tips={len(tips)}"),
            )
            written += 1
        debug.write_json("tips", "tips.json", {
            "stats": tipset.stats(),
            "tips": [t.to_dict() for t in tipset.tips],
        })
        return written
