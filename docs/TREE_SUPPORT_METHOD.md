# ツリーサポートの寸法と接触部

2026-09-08に公開実装を読み、現在のパイプラインと比較した。

## 参照した実装

- [OrcaSlicer TreeSupportCommon.hpp](https://github.com/OrcaSlicer/OrcaSlicer/blob/main/src/libslic3r/Support/TreeSupportCommon.hpp):
  ライン幅、先端径、基準枝径を別に持ち、先端から枝への移行と下方への太り角度を扱う。
- [OrcaSlicer TreeSupport3D.cpp](https://github.com/OrcaSlicer/OrcaSlicer/blob/main/src/libslic3r/Support/TreeSupport3D.cpp):
  オーバーハングからの先端・接触インターフェースの配置を参照。
- [PrusaSlicer 2.9.4 OrganicSupport.cpp](https://github.com/prusa3d/PrusaSlicer/blob/version_2.9.4/src/libslic3r/Support/OrganicSupport.cpp):
  到達可能領域内での中心線の平滑化、分岐点と通常の枝の扱い、衝突回避を参照。

MeshBloomは独立したPython実装。上記エンジンのコードや生成結果の完全な移植ではない。
Prusaのbilaplacianや連続衝突処理全体、Orcaの接触インターフェース層・押出経路は再現していない。

## MeshBloomでの生成

1. モデルを層に切り、支持が必要な領域へ先端を配置する。
2. 先端から下へ、衝突と到達可能領域で許される範囲を伝播し、枝を合流させる。
3. 下からの領域制約とLaplacian平滑化で、制約内の中心線を求める。
4. 水平リングから閉じた枝を生成し、Manifoldで合流部を結合する。
5. ギャップ、到達性、閉じた体積などの既存検証を通ってから書き出す。

追加した`organic`径プロファイルでは、ノズル径、先端と枝の径の差、接触高さから
先端遷移の高さを決める。遷移には独自のsmoothstep補間を使い、その後は太り角度で
半径を増やす。1層の半径増加はライン幅の半分以下。合流後の半径は縮小せず、
衝突制約を満たせない増加は従来の細い半径へ戻して再試行する。

## 接触形状

接触面は先端Z位置の水平円盤。接触径はライン幅以上、元の先端径以下とする。
指定高さ内で、円柱状のネック、直線テーパー、丸み付きの肩を選べる。
リングを補間して元の管の内側だけを削り、頂部のZを上げない。
合流部の重なりを維持するため、短い終端枝では接触高さを制限する。
これは取り外しやすさを調整するための実体形状で、スライサーのインターフェース層ではない。

Zギャップは層単位へ切り上げ、最低1層分。任意面への連続的な最短距離や
フィラメント収縮、ブリッジの垂れまで保証するものではない。Simplify3Dでのスライスと
実機での接触強度・除去性の確認が必要。
