# MeshBloom — アーキテクチャ

## 2026-09-07: アプリケーション層とインフィル

`gui.py` (Tkinter / Matplotlib) と `cli.py` は `application.py` の同じ生成・出力処理を使う。
サポートの既存パイプラインに加え、`infill.py` が厚み付きgyroid / diamond / cubic棒格子を生成。
符号付き距離と周期場から内部空洞をサンプリングし、元モデルとの差分で外形を保持する。
アプリの出力はManifoldによるBoolean結合を必須とし、失敗時に重複面のまま成功扱いしない。

メッシュ生成のリングはすべて水平なXY断面とした。従来の中心線に直交するリングは
短い曲がりで重なり、ベッド下・tip上へはみ出すため、2D collisionが扱う半径と規約を合わせる。
ゼロ長のroot経路はメッシュ化せず、接続する子経路のキャップで接地面を作る。
GUIは別スレッドで生成し、メインスレッドでプレビュー・進捗・エラーを表示する。
出力形式とSimplify3D v5の受け渡しはルートのREADME.mdを参照。

## 0. 目的

`model.stl` を入力し、**ツリーサポート形状だけ**を独立した watertight メッシュ
`tree_support.stl` として出力する。両者を同一座標系のままレガシースライサー
(Simplify3D / 旧 Slic3r / 旧 ideaMaker 等) へ同時読み込みし、擬似ツリーサポートを実現する。

G-code は生成しない。責務は「印刷可能なツリーサポート形状そのものの生成」。

## 1. レイヤ構造 (概念4層)

```
Mesh          : MeshLoader
   |
Slices        : ModelSlicer / OverhangDetector / CollisionCache
   |
Support Demand: ContactSampler
   |
Tree Graph    : TreePropagator / TreeMerger / CenterlineSolver / RadiusSolver
   |
3D Geometry   : MeshBuilder / BooleanUnion / Validator / Exporter
```

## 2. パイプライン

```
3D Mesh
  -> 2D Layer Slicing              (slicer.py)
  -> Overhang Detection            (overhang.py)
  -> Support Demand Region         (overhang.py)
  -> Contact / Tip Sampling        (tips.py)
  -> Collision / Avoidance Region  (volumes.py)
  -> Top-down Influence Propagation(propagate.py)
  -> Branch Merge                  (merge.py)
  -> Tree Graph                    (graph.py)
  -> Bottom-up Constraint          (propagate.py::trim_bottom_up)
  -> Centerline Placement          (centerline.py)
  -> Centerline Smoothing          (smoothing.py)
  -> Branch Radius Assignment      (radius.py, 伝播中に確定)
  -> 3D Tube Meshing               (meshbuilder.py)
  -> Mesh Union / Repair           (booleanop.py)
  -> Support-only STL              (exporter.py)
```

## 3. モジュール一覧

| モジュール | 責務 |
|---|---|
| `config.py` | `SupportConfig` 全パラメータ + 派生量 |
| `geometry2d.py` | shapely ラッパ。保守的 offset / union / 面積 / 領域への射影 |
| `mesh_loader.py` | `MeshLoader`: STL/OBJ/PLY/3MF 読み込み・診断・修復(任意) |
| `slicer.py` | `ModelSlicer`: 固定レイヤ高で 2D ポリゴン列 `P[k]` を生成 |
| `overhang.py` | `OverhangDetector`: `P[k] - offset(P[k-1], allowed)` |
| `volumes.py` | `CollisionCache`: collision / reachability(avoidance) キャッシュ、半径量子化 |
| `tips.py` | `ContactSampler`: 3段階 tip サンプリング + coverage 計測 |
| `graph.py` | `SupportNode` / `ContactPoint` / `Branch` / `TreeGraph` |
| `propagate.py` | `TreePropagator`: top-down influence area 伝播 + bottom-up trim |
| `merge.py` | `TreeMerger`: STRtree による AABB 事前判定 + 精密 intersection |
| `centerline.py` | `CenterlineSolver`: influence area 内の具体的 XY 決定 |
| `smoothing.py` | Laplacian / bi-Laplacian + influence area への再射影 |
| `collision3d.py` | 3D カプセル衝突の再検証と nudge |
| `meshbuilder.py` | `MeshBuilder`: 平行移動フレームによる tube メッシュ |
| `booleanop.py` | 分岐部の union (manifold3d boolean / voxel-SDF) |
| `validator.py` | `Validator`: 不変条件チェックとメトリクス |
| `exporter.py` | `Exporter`: binary STL / OBJ (座標系は絶対保持) |
| `debug_io.py` | SVG / JSON / OBJ / PLY による中間状態可視化 |
| `testmodels.py` | 手続き生成のテスト形状 (外部 STL 不要) |
| `pipeline.py` | 上記を束ねる `generate_tree_support()` |
| `cli.py` | CLI エントリポイント |

## 4. 座標系とレイヤ規約

- 単位は **mm (float64)**。z=0 はビルドプレート。
- モデルレイヤ `k` はスラブ `[k*lh, (k+1)*lh]` を占め、断面は中央 `z=(k+0.5)*lh` でサンプリングする。
- **サポートノード**はレイヤ境界平面上にある: ノードレイヤ `j` の z は `j*lh`。
  - よってノード 0 は z=0、すなわちビルドプレート面にちょうど接地する。
  - tip ノードは枝の**最上点**であり、その上に材料は存在しない。
- 入力メッシュの座標は一切変更しない (センタリング禁止)。`T_model == T_support`。

## 5. Z ギャップの幾何的保証

`z_gap_layers = ceil(top_z_gap / layer_height)` (最低1)。
モデルのオーバーハングがレイヤ `m` にあるとき、tip ノードは `j = m - z_gap_layers` に置く。
tip の材料上端は `j*lh`、モデル下面は `m*lh` なので
実ギャップ `= z_gap_layers*lh >= top_z_gap` が構成的に保証される。

## 6. Collision の Z 方向考慮 (CuraEngine の TreeModelVolumes 相当)

ノードレイヤ `j`・半径 `r` に対して

```
collision_branch[j][r] =
    U_{i in [j-1-n_bot, j]}  offset(P[i], r + xy_gap)
  U U_{t in [1, n_top]}      offset(P[j+t], r + xy_gap * (1 - (t-0.5)/n_top))
```

- 下側は「枝下端 `(j-1)*lh`」とモデル上端の間に `bottom_z_gap` を確保するため。
- 上側は xy_gap をテーパさせる。これは CuraEngine が
  「オーバーハングの真下が xy_distance で塗り潰されて tip を置けなくなる」問題を
  回避するために使っている手法と同趣旨 (式は独自)。
- **tip 用変種** `collision_tip[j][r]` は上側の範囲を `t in [1, n_top-1]` とする。
  tip は枝の最上点で上に材料が無いため、`t = n_top` のモデルとは
  ちょうど `top_z_gap` 離れており阻害する必要がない。
  (CuraEngine は代わりに tip を 1 層余分に下げる `z_distance_delta = n_top + 1` を採る。
   本ツールは要求ギャップぴったりを実現するため per-kind collision を採用した。)

## 7. Reachability (Cura の avoidance に相当)

`avoidance` の補集合 `reach[r][j]` を**下から上へ**構築する:

```
reach[r][0] = world_box - collision_branch[0][r]
reach[r][j] = (dilate(reach[r][j-1], move_max) & world_box) - collision_branch[j][r]
```

`reach[r][j]` は「半径 r の枝中心をレイヤ j のその点に置いたとき、
最大傾斜角を守って降下すればビルドプレートまで到達できる」領域。
伝播時に collision ではなく **reach** で交差を取ることで、
行き止まりに枝が入り込むことを構造的に防ぐ (Cura と同じ発想)。

## 8. Influence Area 伝播

```
next_area = offset(area, move_max) & reach[r_next][j-1]
```

具体的な中心座標は伝播完了まで確定しない。preferred angle は
hard constraint ではなく centerline 決定時の cost として使う。

## 9. Merge 条件

同一レイヤの 2 ノード A, B について
`intersection(A.influence_area, B.influence_area)` が有意面積を持てば merge 候補。
merge 後半径 `r = min(r_max, sqrt(rA^2 + rB^2))` で reach を再評価し、
`intersection & reach[r_merged][j]` が空になる場合は merge 禁止。
候補探索は shapely `STRtree` による AABB 事前判定で O(N log N) 相当。

**単純 intersection を採用した理由**: merge 後の領域が両親の領域の部分集合になるため、
centerline を子から親へ射影する際の移動距離が `move_max` 以下であることが
数学的に保証され、`branch_angle <= branch_angle_max` の不変条件が厳密に成立する。
(CuraEngine は半径差だけ膨張させた intersection を使い、より積極的に merge するが
 その分だけ角度制約が緩む。)

## 10. Centerline 決定

Cura と同様に **bottom-up** で確定する。

1. 各木の最下ノードで `next_position` (最後の merge 位置 / tip 中心の投影) を採用し、
   influence area 外なら射影。
2. 親 (上位) ノードは子の点を目標とし、preferred angle 由来のバイアスを加えた
   目標点を influence area へ射影。
   - `child.area ⊆ dilate(parent.area, move_max)` なので、
     子の点から親領域への最近点距離は必ず `move_max` 以下。

## 11. 半径

```
r(tip)          = tip_diameter / 2
r(下layer)      = min(r_max, r + growth_radius_per_mm * layer_height)
r(merge)        = min(r_max, sqrt(r1^2 + r2^2))          # 断面積保存ヒューリスティック
r(root flare)   = 底面近傍 root_flare_height 以内で root_diameter_min/2 まで線形に増加
```

これは Cura/Prusa の完全再現ではなく**本ツール独自のヒューリスティック**である。
