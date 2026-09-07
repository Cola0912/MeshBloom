# ADR-0001: 実装言語とライブラリ

- Status: Accepted
- Date: 2026-08-24

## 決定

コアアルゴリズムを **Python 3.12 + shapely 2.1 (GEOS) + trimesh 4.6 + numpy** で実装する。

## 背景

指示書 32 節は「新規プロジェクトなら幾何処理コアは C++ が第一候補。ただし MVP を
高速に検証するため Python プロトタイプは許可」としている。

## 理由

1. 本リポジトリ (`F:\Projects`) は Python 資産のみで、C++ ツールチェーン
   (CMake / Clipper2 / Eigen / CGAL) は未整備。ジオメトリを 1 つも検証しないうちに
   ビルド環境構築で数ステップ消費するのは STEP 主義に反する。
2. shapely 2.1 は GEOS バックエンドで Clipper2 と同等以上に堅牢な多角形ブーリアン・
   オフセットを提供し、`STRtree` による空間索引も同梱する。
3. trimesh は STL/OBJ/3MF I/O、`section_multiplane` による高速スライス、
   watertight/manifold 診断、`manifold3d` 経由のブーリアンを一括で提供する。
4. 責務分離 (Slice -> Polygon -> Tree Graph -> Mesh) を保てば、
   性能が問題になった時点でモジュール単位で C++ 化できる。

## 帰結

- 単位は mm の float64。Cura の整数マイクロメートル座標系は採らない。
  代わりに `geometry2d.py` で**保守側への丸め** (円弧近似の外接補正、面積 epsilon) を徹底する。
- 性能は C++ 実装に劣る。31 節の禁止事項 (全組合せ intersection、O(N^2) merge、
  offset 再計算) を回避する設計で吸収する。
- 将来の C++ 移植先: `volumes.py` (collision/reach) と `propagate.py` が支配的コスト。

## 保守的丸めについて

shapely の `buffer(d, join_style="round", quad_segs=n)` は円弧を**内接**多角形で
近似するため、真の Minkowski 和より小さくなる。collision に使うと危険側。
そこで `geometry2d.grow()` は距離を `1/cos(pi/(4*quad_segs))` 倍して**外接**させる。
