# サードパーティ

## 同梱モデル

`src/treesupport/assets/3DBenchy.stl`: ユーザー提供の#3DBenchyテストモデル。
公式モデルのライセンスはCC0 1.0 Universal。
作者・参照元・元ファイルのSHA-256は[同梱モデルの説明](src/treesupport/assets/README.md)を参照。

## 実行時依存ライブラリ

| library | version | license | usage |
|---|---|---|---|
| numpy | >=1.26 (検証 2.2.5) | BSD-3-Clause | 数値配列全般 |
| shapely | >=2.0 (検証 2.1.2) | BSD-3-Clause | 2D 多角形ブーリアン / offset / STRtree |
| trimesh | >=4.0 (検証 4.6.8) | MIT | メッシュ I/O、スライス、manifold 診断 |
| scipy | >=1.11 (検証 1.18.0) | BSD-3-Clause | KDTree (tip coverage / 3D collision) |
| networkx | >=3.0 (検証 3.6.1) | BSD-3-Clause | ツリーグラフの連結成分・走査 |
| manifold3d | >=3.0 (検証 3.5.2) | Apache-2.0 | 分岐部のunion、インフィル空洞の差分 |
| scikit-image | >=0.22 (検証 0.26.0) | BSD-3-Clause | インフィルとvoxel/SDFのmarching cubes |
| mapbox_earcut | 任意 (検証 2.0.0) | ISC | trimesh の多角形三角形分割補助 |
| rtree | >=1.2 (検証 1.4.1) | MIT | trimesh の距離計算・空間索引 |
| matplotlib | >=3.8 (検証 3.11.1) | Matplotlib license (同梱LICENSE参照) | デスクトップの3D・断面プレビュー |
| pytest | 開発時 | MIT | テスト |

(間接依存: GEOS (LGPL-2.1, shapely が同梱バイナリとして配布))

## アルゴリズム参照元

以下は**読解・再実装のみ**行い、ソースコードの直接流用はしていない。

| project | license | 参照ファイル | 抽出した概念 |
|---|---|---|---|
| CuraEngine | AGPL-3.0 | `src/TreeSupport.cpp` | influence area の top-down 伝播、merge 手順、centerline の bottom-up 確定 |
| CuraEngine | AGPL-3.0 | `src/TreeSupportTipGenerator.cpp` | overhang からの tip サンプリング、線分上への最大距離サンプリング、輪郭補助 tip |
| CuraEngine | AGPL-3.0 | `src/TreeModelVolumes.cpp` | collision の半径量子化キャッシュ、avoidance の下から上への構築、z 方向 collision 累積と xy テーパ |
| PrusaSlicer | AGPL-3.0 | `src/libslic3r/Support/OrganicSupport.cpp` | bottom-up influence trim、influence area 内 Laplacian smoothing、tube 押し出しと分岐処理 |
| Slic3r (legacy) | AGPL-3.0 | `xs/src/libslic3r/SupportMaterial.cpp` | レイヤ差分によるオーバーハング検出の考え方 |

参考論文: Vanek et al., *Clever Support: Efficient Support Structure Generation for
Digital Fabrication*, CGF 2014 (DOI 10.1111/cgf.12437) — 枝の合流と傾斜角制約の考え方。

**AGPL-3.0 の扱い**: 上記プロジェクトのソースコードは一切コピーしていない。
本ツールの実装はアルゴリズム記述 (公開 Wiki / 論文 / ソース読解による理解) に基づく
独立実装であり、変数名・関数構造・定数テーブルの流用も行っていない。
