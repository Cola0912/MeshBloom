"""2D 幾何ユーティリティ (shapely / GEOS ラッパ)。

設計方針
--------
* 全ての公開関数は ``Polygon`` / ``MultiPolygon`` / 空ジオメトリを受け取り、
  ``Polygon`` 単体または ``MultiPolygon`` を返す。
* shapely の ``buffer`` は円弧を**内接**多角形で近似する。
  用途に応じて内接 / 外接を明示的に選ぶ:

  - ``dilate``       : 内接 = 真の Minkowski 和より小さい -> 「動けること」の主張に安全
  - ``dilate_outer`` : 外接 = 真の Minkowski 和より大きい -> 「禁止領域」の主張に安全
  - ``erode_outer``  : 外接量だけ余分に削る -> 残った領域の主張に安全
* 面積 epsilon 未満のかけら (GEOS の丸め誤差由来) は常に捨てる。
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import shapely
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, unary_union

__all__ = [
    "EMPTY",
    "AREA_EPS",
    "arc_outer_factor",
    "as_multipolygon",
    "clean",
    "polygons",
    "is_empty",
    "area",
    "union",
    "intersection",
    "difference",
    "dilate",
    "dilate_outer",
    "erode_outer",
    "project_point",
    "inside_point",
    "simplify_outward",
    "bounds_union",
    "box_polygon",
    "clip_to_bounds",
    "drop_small",
]

EMPTY: Polygon = Polygon()

#: これ未満の面積 (mm^2) はブーリアン誤差由来のかけらとして扱う。
AREA_EPS: float = 1.0e-7


def arc_outer_factor(quad_segs: int) -> float:
    """円弧を内接多角形で近似したときの外接補正係数。

    ``quad_segs`` 個の線分で 90 度を近似するとき、内接多角形の辺の中点は
    半径 ``r*cos(pi/(4*quad_segs))`` にある。よって距離をこの逆数倍すれば
    近似多角形が真円を**含む** (外接する)。
    """
    return 1.0 / math.cos(math.pi / (4.0 * max(1, quad_segs)))


def as_multipolygon(geom: BaseGeometry | None) -> BaseGeometry:
    """ポリゴン成分だけを取り出して返す (線・点は捨てる)。"""
    if geom is None or geom.is_empty:
        return EMPTY
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    flat: list[Polygon] = []
    for g in getattr(geom, "geoms", []):
        if isinstance(g, Polygon):
            flat.append(g)
        elif isinstance(g, MultiPolygon):
            flat.extend(list(g.geoms))
    if not flat:
        return EMPTY
    if len(flat) == 1:
        return flat[0]
    return MultiPolygon(flat)


def polygons(geom: BaseGeometry | None) -> list[Polygon]:
    """``Polygon`` のリストに展開する。"""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [p for p in geom.geoms if not p.is_empty]
    out: list[Polygon] = []
    for g in getattr(geom, "geoms", []):
        out.extend(polygons(g))
    return out


def drop_small(geom: BaseGeometry, min_area: float) -> BaseGeometry:
    """面積が ``min_area`` 未満のポリゴンを捨てる。"""
    if geom.is_empty:
        return EMPTY
    if min_area <= 0.0:
        return geom
    keep = [p for p in polygons(geom) if p.area >= min_area]
    if not keep:
        return EMPTY
    if len(keep) == 1:
        return keep[0]
    return MultiPolygon(keep)


def clean(geom: BaseGeometry | None, min_area: float = AREA_EPS) -> BaseGeometry:
    """不正ジオメトリを修復し、微小片を除去する。"""
    if geom is None or geom.is_empty:
        return EMPTY
    if not geom.is_valid:
        geom = shapely.make_valid(geom)
    geom = as_multipolygon(geom)
    if geom.is_empty:
        return EMPTY
    return drop_small(geom, min_area)


def is_empty(geom: BaseGeometry | None, min_area: float = AREA_EPS) -> bool:
    return geom is None or geom.is_empty or geom.area < min_area


def area(geom: BaseGeometry | None) -> float:
    return 0.0 if geom is None or geom.is_empty else float(geom.area)


def union(geoms: Iterable[BaseGeometry], min_area: float = AREA_EPS) -> BaseGeometry:
    items = [g for g in geoms if g is not None and not g.is_empty]
    if not items:
        return EMPTY
    if len(items) == 1:
        return clean(items[0], min_area)
    return clean(unary_union(items), min_area)


def intersection(a: BaseGeometry, b: BaseGeometry, min_area: float = AREA_EPS) -> BaseGeometry:
    if a is None or b is None or a.is_empty or b.is_empty:
        return EMPTY
    try:
        res = a.intersection(b)
    except shapely.errors.GEOSException:  # pragma: no cover - GEOS の稀な失敗への保険
        res = shapely.make_valid(a).intersection(shapely.make_valid(b))
    return clean(res, min_area)


def difference(a: BaseGeometry, b: BaseGeometry, min_area: float = AREA_EPS) -> BaseGeometry:
    if a is None or a.is_empty:
        return EMPTY
    if b is None or b.is_empty:
        return clean(a, min_area)
    try:
        res = a.difference(b)
    except shapely.errors.GEOSException:  # pragma: no cover
        res = shapely.make_valid(a).difference(shapely.make_valid(b))
    return clean(res, min_area)


def dilate(geom: BaseGeometry, dist: float, quad_segs: int = 8,
           min_area: float = AREA_EPS) -> BaseGeometry:
    """内接近似の膨張。真の Minkowski 和の**部分集合**になる (安全側に小さい)。"""
    if geom is None or geom.is_empty:
        return EMPTY
    if dist <= 0.0:
        return clean(geom, min_area)
    return clean(geom.buffer(dist, quad_segs=quad_segs, join_style="round"), min_area)


def dilate_outer(geom: BaseGeometry, dist: float, quad_segs: int = 8,
                 min_area: float = AREA_EPS) -> BaseGeometry:
    """外接近似の膨張。真の Minkowski 和を**含む** (禁止領域用の安全側)。"""
    if geom is None or geom.is_empty:
        return EMPTY
    if dist <= 0.0:
        return clean(geom, min_area)
    d = dist * arc_outer_factor(quad_segs)
    return clean(geom.buffer(d, quad_segs=quad_segs, join_style="round"), min_area)


def erode_outer(geom: BaseGeometry, dist: float, quad_segs: int = 8,
                min_area: float = AREA_EPS) -> BaseGeometry:
    """外接量だけ余分に削る収縮。真の収縮の**部分集合**になる (安全側に小さい)。"""
    if geom is None or geom.is_empty:
        return EMPTY
    if dist <= 0.0:
        return clean(geom, min_area)
    d = dist * arc_outer_factor(quad_segs)
    return clean(geom.buffer(-d, quad_segs=quad_segs, join_style="round"), min_area)


def simplify_outward(geom: BaseGeometry, tol: float, quad_segs: int = 8) -> BaseGeometry:
    """簡略化。ただし元の領域を必ず含むよう union を取る (禁止領域向け)。"""
    if geom is None or geom.is_empty or tol <= 0.0:
        return geom if geom is not None else EMPTY
    simplified = clean(geom.simplify(tol, preserve_topology=True))
    if simplified.is_empty:
        return clean(geom)
    return union([geom, simplified])


def project_point(geom: BaseGeometry, pt: Sequence[float],
                  nudge: float = 0.0) -> tuple[float, float]:
    """``pt`` を ``geom`` 内部 (または境界) へ射影する。

    ``geom`` が空の場合は ``pt`` をそのまま返す。

    既定では内側への nudge を行わない。``nearest_points`` の結果は
    定義上ジオメトリ上の点なので ``covers`` は必ず真であり、
    余分に内側へ動かすと「最近点までの距離」がわずかに増えて
    ``move_max`` 制約を 1e-6 単位で破る原因になる。
    内側へ寄せたい場合は呼び出し側で明示的に ``nudge`` を渡すこと。
    """
    x, y = float(pt[0]), float(pt[1])
    if geom is None or geom.is_empty:
        return (x, y)
    p = Point(x, y)
    if geom.covers(p):
        return (x, y)
    q = nearest_points(geom, p)[0]
    qx, qy = float(q.x), float(q.y)
    dx, dy = qx - x, qy - y
    norm = math.hypot(dx, dy)
    if norm > 0.0 and nudge > 0.0:
        cx = qx + dx / norm * nudge
        cy = qy + dy / norm * nudge
        if geom.covers(Point(cx, cy)):
            return (cx, cy)
    return (qx, qy)


def inside_point(geom: BaseGeometry) -> tuple[float, float] | None:
    """領域内部の代表点。空なら ``None``。"""
    if geom is None or geom.is_empty:
        return None
    p = geom.representative_point()
    return (float(p.x), float(p.y))


def bounds_union(geoms: Iterable[BaseGeometry]) -> tuple[float, float, float, float] | None:
    box: list[float] | None = None
    for g in geoms:
        if g is None or g.is_empty:
            continue
        b = g.bounds
        if box is None:
            box = [b[0], b[1], b[2], b[3]]
        else:
            box[0] = min(box[0], b[0])
            box[1] = min(box[1], b[1])
            box[2] = max(box[2], b[2])
            box[3] = max(box[3], b[3])
    return None if box is None else (box[0], box[1], box[2], box[3])


def box_polygon(bounds: Sequence[float], margin: float = 0.0) -> Polygon:
    minx, miny, maxx, maxy = bounds
    return shapely.box(minx - margin, miny - margin, maxx + margin, maxy + margin)


def clip_to_bounds(geom: BaseGeometry, bounds: Sequence[float],
                   pad: float = 0.0) -> BaseGeometry:
    """``geom`` を矩形で切り取る (高速な矩形クリップ)。

    大きな collision / reach ポリゴンを、着目ノードの bbox 周辺だけに絞ってから
    ブーリアンを掛けるための前処理。矩形の外側は元々使わないので結果は変わらない。
    """
    if geom is None or geom.is_empty:
        return EMPTY
    minx, miny, maxx, maxy = bounds
    gminx, gminy, gmaxx, gmaxy = geom.bounds
    if (gminx >= minx - pad and gminy >= miny - pad
            and gmaxx <= maxx + pad and gmaxy <= maxy + pad):
        return geom
    try:
        clipped = shapely.clip_by_rect(geom, minx - pad, miny - pad,
                                       maxx + pad, maxy + pad)
    except shapely.errors.GEOSException:  # pragma: no cover
        return intersection(geom, box_polygon(bounds, pad))
    # 後段のブーリアンで clean() が走るので、ここでは検証コストを掛けない。
    return as_multipolygon(clipped)
