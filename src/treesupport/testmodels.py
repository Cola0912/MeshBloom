"""外部 STL 無しで unit / integration test を回すための手続き生成テスト形状。

指示書 29 節に対応:

1. ``floating_box``       空中直方体
2. ``slope_45``           45 度斜面
3. ``cantilever``         水平片持ち板
4. ``bridge``             ブリッジ
5. ``overhang_corner``    鋭角コーナーを持つオーバーハング
6. ``arch``               アーチ
7. ``cavity``             内部空洞
8. ``obstructed_path``    ビルドプレート直下経路をモデルが遮る形状
9. ``merge_cluster``      複数 tip が 1 本の trunk に統合される形状

全て z=0 のビルドプレート座標系で、モデルは z>=0 に置かれる。
"""

from __future__ import annotations

import numpy as np
import shapely
import trimesh
from shapely.geometry import Polygon

__all__ = [
    "floating_box",
    "slope_45",
    "cantilever",
    "bridge",
    "overhang_corner",
    "arch",
    "cavity",
    "obstructed_path",
    "merge_cluster",
    "narrow_slot",
    "islands",
    "mixed_reachability",
    "vertical_wall",
    "horizontal_plate_on_plate",
    "unit_cube",
    "ALL_MODELS",
]


# ----------------------------------------------------------------------
# 内部ヘルパ
# ----------------------------------------------------------------------
def _box(extents, center) -> trimesh.Trimesh:
    """軸平行ボックス。``center`` は中心座標。"""
    tf = np.eye(4)
    tf[:3, 3] = np.asarray(center, dtype=float)
    return trimesh.creation.box(extents=np.asarray(extents, dtype=float), transform=tf)


def _box_span(x0, x1, y0, y1, z0, z1) -> trimesh.Trimesh:
    """min/max 指定のボックス。"""
    return _box(
        (x1 - x0, y1 - y0, z1 - z0),
        ((x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5),
    )


def _prism_xz(polygon_xz: Polygon, y0: float, y1: float) -> trimesh.Trimesh:
    """XZ 平面上の断面を Y 方向に押し出した角柱を作る。"""
    solid = trimesh.creation.extrude_polygon(polygon_xz, height=(y1 - y0))
    # extrude_polygon は XY 断面を +Z に押し出す。X をそのまま、Y->Z、Z->-Y に写す。
    rot = trimesh.transformations.rotation_matrix(np.pi / 2.0, [1.0, 0.0, 0.0])
    solid.apply_transform(rot)
    # 回転後、押し出し方向は -Y。y0 側に合わせる。
    solid.apply_translation([0.0, y1, 0.0])
    return solid


def _union(*meshes: trimesh.Trimesh) -> trimesh.Trimesh:
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.boolean.union(list(meshes))


def _difference(a: trimesh.Trimesh, *others: trimesh.Trimesh) -> trimesh.Trimesh:
    return trimesh.boolean.difference([a, *others])


# ----------------------------------------------------------------------
# 1. 空中直方体
# ----------------------------------------------------------------------
def floating_box(size: float = 12.0, thickness: float = 4.0,
                 height: float = 8.0) -> trimesh.Trimesh:
    """完全に浮いた直方体。底面全体がサポート対象になる。"""
    return _box_span(0.0, size, 0.0, size, height, height + thickness)


# ----------------------------------------------------------------------
# 2. 45 度斜面
# ----------------------------------------------------------------------
def slope_45(size: float = 16.0, depth: float = 10.0) -> trimesh.Trimesh:
    """下向き面がちょうど鉛直から 45 度の斜面 (自己支持の境界)。"""
    poly = Polygon([(0.0, 0.0), (size, size), (0.0, size)])
    return _prism_xz(poly, 0.0, depth)


# ----------------------------------------------------------------------
# 3. 水平片持ち板
# ----------------------------------------------------------------------
def cantilever(column: float = 8.0, arm: float = 22.0, depth: float = 10.0,
               height: float = 16.0, plate: float = 2.0) -> trimesh.Trimesh:
    """柱 + 水平に張り出した板。板の下面が完全な水平オーバーハング。"""
    col = _box_span(0.0, column, 0.0, depth, 0.0, height)
    top = _box_span(0.0, column + arm, 0.0, depth, height, height + plate)
    return _union(col, top)


# ----------------------------------------------------------------------
# 4. ブリッジ
# ----------------------------------------------------------------------
def bridge(pillar: float = 6.0, span: float = 20.0, depth: float = 8.0,
           height: float = 14.0, beam: float = 2.5) -> trimesh.Trimesh:
    """2 本の柱とその間を渡る梁。梁の下面が水平オーバーハング。"""
    total = pillar * 2.0 + span
    a = _box_span(0.0, pillar, 0.0, depth, 0.0, height)
    b = _box_span(pillar + span, total, 0.0, depth, 0.0, height)
    top = _box_span(0.0, total, 0.0, depth, height, height + beam)
    return _union(a, b, top)


# ----------------------------------------------------------------------
# 5. 鋭角コーナーを持つオーバーハング
# ----------------------------------------------------------------------
def overhang_corner(height: float = 10.0, thickness: float = 2.0,
                    arm: float = 18.0, width: float = 5.0) -> trimesh.Trimesh:
    """XY 平面で L 字 (鋭角コーナーあり) の薄板が空中に浮いた形状。

    CuraEngine Issue #2037 のような corner omission を検出するためのモデル。
    """
    poly = Polygon([
        (0.0, 0.0),
        (arm, 0.0),
        (arm, width),
        (width, width),
        (width, arm),
        (0.0, arm),
    ])
    solid = trimesh.creation.extrude_polygon(poly, height=thickness)
    solid.apply_translation([0.0, 0.0, height])
    return solid


# ----------------------------------------------------------------------
# 6. アーチ
# ----------------------------------------------------------------------
def arch(width: float = 24.0, height: float = 18.0, depth: float = 8.0,
         radius: float = 9.0) -> trimesh.Trimesh:
    """半円アーチ。下向き面の傾斜が連続的に変化する。"""
    outer = shapely.box(0.0, 0.0, width, height)
    cx = width * 0.5
    hole = shapely.Point(cx, 0.0).buffer(radius, quad_segs=32)
    hole = hole.intersection(shapely.box(cx - radius, 0.0, cx + radius, radius))
    hole = hole.union(shapely.box(cx - radius, -1.0, cx + radius, 0.0))
    poly = outer.difference(hole)
    return _prism_xz(poly, 0.0, depth)


# ----------------------------------------------------------------------
# 7. 内部空洞
# ----------------------------------------------------------------------
def cavity(size: float = 16.0, wall: float = 3.0) -> trimesh.Trimesh:
    """内部に閉じた直方体空洞を持つ箱。空洞天井が内部オーバーハング。"""
    outer = _box_span(0.0, size, 0.0, size, 0.0, size)
    inner = _box_span(wall, size - wall, wall, size - wall, wall, size - wall)
    return _difference(outer, inner)


# ----------------------------------------------------------------------
# 8. 直下経路が塞がれた形状
# ----------------------------------------------------------------------
def obstructed_path(height: float = 20.0, depth: float = 10.0) -> trimesh.Trimesh:
    """浮いた板の真下に別の実体があり、枝が迂回しないと着地できない形状。

    上部の板は x in [0, 10]、その真下 z in [4, 12] に x in [-2, 12] の障害物ブロックが
    あるため、枝は x>12 側 (または x<-2 側) へ逃げてから降りる必要がある。
    """
    top = _box_span(0.0, 10.0, 0.0, depth, height, height + 2.5)
    block = _box_span(-2.0, 12.0, 0.0, depth, 0.0, 12.0)
    return _union(block, top)


# ----------------------------------------------------------------------
# 9. merge テスト
# ----------------------------------------------------------------------
def merge_cluster(height: float = 22.0, pad: float = 3.0, gap: float = 4.0,
                  thickness: float = 2.0) -> trimesh.Trimesh:
    """互いに近接した 4 枚の浮遊小板。降下に伴い 1 本の trunk へ統合されるべき形状。"""
    parts = []
    for ix in (0, 1):
        for iy in (0, 1):
            x0 = ix * (pad + gap)
            y0 = iy * (pad + gap)
            parts.append(_box_span(x0, x0 + pad, y0, y0 + pad, height, height + thickness))
    merged = trimesh.util.concatenate(parts)
    return merged


# ----------------------------------------------------------------------
# 補助 (STEP 3 の基本形状)
# ----------------------------------------------------------------------
def vertical_wall(width: float = 10.0, depth: float = 4.0,
                  height: float = 12.0) -> trimesh.Trimesh:
    """垂直壁。サポートは一切不要。"""
    return _box_span(0.0, width, 0.0, depth, 0.0, height)


def horizontal_plate_on_plate(width: float = 12.0, depth: float = 12.0,
                              thickness: float = 2.0) -> trimesh.Trimesh:
    """ビルドプレートに直接載る水平板。サポート不要。"""
    return _box_span(0.0, width, 0.0, depth, 0.0, thickness)


def unit_cube(size: float = 10.0, origin=(0.0, 0.0, 0.0)) -> trimesh.Trimesh:
    """スライス検証用の単純な立方体。"""
    x0, y0, z0 = origin
    return _box_span(x0, x0 + size, y0, y0 + size, z0, z0 + size)


def narrow_slot(gap: float = 3.0, block: float = 12.0, depth: float = 10.0,
                wall_height: float = 12.0, plate_half: float = 1.5,
                plate_z: float = 14.0, plate_thickness: float = 2.0) -> trimesh.Trimesh:
    """幅 ``gap`` の隙間を挟む 2 つのブロックと、その真上に浮いた板。

    枝は隙間を通り抜けないとビルドプレートへ到達できない。
    通過可否は ``gap > 2 * (branch_radius + xy_gap)`` で決まるので、
    「細ければ通る / 太いと通らない」を検証できる。
    """
    left = _box_span(-block, -gap * 0.5, 0.0, depth, 0.0, wall_height)
    right = _box_span(gap * 0.5, block, 0.0, depth, 0.0, wall_height)
    plate = _box_span(-plate_half, plate_half, 0.0, depth,
                      plate_z, plate_z + plate_thickness)
    return _union(_union(left, right), plate)


def islands(count: int = 3, pad: float = 6.0, spacing: float = 20.0,
            depth: float = 6.0, height: float = 10.0,
            thickness: float = 2.0) -> trimesh.Trimesh:
    """互いに遠く離れた複数の浮遊板。merge しない独立 island として使う。"""
    parts = []
    for i in range(count):
        x0 = i * spacing
        parts.append(_box_span(x0, x0 + pad, 0.0, depth,
                               height, height + thickness))
    return trimesh.util.concatenate(parts)


def mixed_reachability(size: float = 16.0, wall: float = 3.0,
                       plate_x: float = 24.0, plate: float = 8.0,
                       plate_z: float = 12.0) -> trimesh.Trimesh:
    """到達可能な浮遊板と、到達不能な閉じた空洞を併せ持つ形状。

    一部の tip が到達不能でも、到達可能な tip まで巻き添えで失敗扱いに
    しないことを検証するために使う。
    """
    box_with_cavity = cavity(size=size, wall=wall)
    floating = _box_span(plate_x, plate_x + plate, 0.0, plate,
                         plate_z, plate_z + 2.0)
    return trimesh.util.concatenate([box_with_cavity, floating])


#: 名前 -> 生成関数
ALL_MODELS = {
    "floating_box": floating_box,
    "slope_45": slope_45,
    "cantilever": cantilever,
    "bridge": bridge,
    "overhang_corner": overhang_corner,
    "arch": arch,
    "cavity": cavity,
    "obstructed_path": obstructed_path,
    "merge_cluster": merge_cluster,
    "narrow_slot": narrow_slot,
    "islands": islands,
    "mixed_reachability": mixed_reachability,
    "vertical_wall": vertical_wall,
    "horizontal_plate_on_plate": horizontal_plate_on_plate,
    "unit_cube": unit_cube,
}
