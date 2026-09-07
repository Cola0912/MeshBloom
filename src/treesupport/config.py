"""ツリーサポート生成の全パラメータ。

角度の規約
----------
* ``overhang_angle``          : **鉛直からの角度** [deg]。0 = 垂直壁、90 = 水平天井。
                                この角度を超える面がサポート対象。
                                1 レイヤあたりの許容張り出しは ``lh * tan(angle)``。
* ``branch_angle_max``        : **鉛直からの角度** [deg]。枝中心線の最大傾斜。
* ``branch_angle_preferred``  : 同上。hard constraint ではなく centerline のコスト。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum


class RootMode(str, Enum):
    """枝の着地先。MVP では ``BUILD_PLATE_ONLY`` のみ実装する。"""

    BUILD_PLATE_ONLY = "BUILD_PLATE_ONLY"
    MODEL_ALLOWED = "MODEL_ALLOWED"


class UnionMode(str, Enum):
    """分岐部のメッシュ結合方式。"""

    NONE = "none"          # 重なった solid tube をそのまま結合 (MVP)
    BOOLEAN = "boolean"    # manifold3d による厳密ブーリアン
    VOXEL = "voxel"        # SDF/voxel union -> marching cubes


@dataclass
class SupportConfig:
    """生成パラメータ一式。全て mm / deg。"""

    # --- スライス ---
    layer_height: float = 0.20
    #: 1 スラブあたりのサンプル平面数。>=2 なら下端と上端を含む和を取る。
    #: 1 にすると従来のスラブ中央 1 枚サンプリング (Z ギャップ保証が弱くなる)。
    slice_samples_per_layer: int = 3

    # --- オーバーハング検出 ---
    overhang_angle: float = 45.0
    min_overhang_area: float = 0.20      # mm^2 これ未満の demand 領域は無視
    overhang_expansion: float = 0.0      # demand 領域の外側への拡張

    # --- ギャップ ---
    top_z_gap: float = 0.20
    bottom_z_gap: float = 0.20
    xy_gap: float = 0.40
    #: True にすると、枝の上側にあるモデルへの xy クリアランスを
    #: 垂直距離に応じてテーパさせる (CuraEngine 相当の緩和)。
    #: 既定は False = 厳密。レガシースライサーへ渡す独立メッシュでは
    #: 融着を避けるため厳密側を既定とする。
    xy_taper: bool = False

    # --- tip ---
    tip_diameter: float = 0.80
    tip_spacing: float = 2.50
    tip_corner_spacing_factor: float = 1.0   # 輪郭補助 tip の間隔倍率
    tip_coverage_radius: float = 0.0         # 0 なら tip_spacing から自動
    tip_min_distance_factor: float = 0.5     # tip 同士の最小距離 = spacing * この値
    tip_sharp_angle: float = 120.0           # これ未満の内角を鋭角コーナーとみなす [deg]
    tip_fill_iterations: int = 12            # coverage 不足領域への追加 tip の反復上限
    tip_influence_radius: float = 0.0        # 0 なら tip_radius

    # --- 枝 ---
    branch_angle_preferred: float = 25.0
    branch_angle_max: float = 40.0
    branch_diameter_max: float = 8.0
    branch_diameter_growth: float = 0.175    # mm(直径)/mm(高さ)。約 5 deg 相当
    root_diameter_min: float = 3.0
    root_flare_height: float = 5.0
    min_printable_feature: float = 0.40

    # --- 木構造 ---
    root_mode: RootMode = RootMode.BUILD_PLATE_ONLY
    merge_enabled: bool = True
    merge_min_area: float = 1.0e-3           # mm^2 merge 判定の最小交差面積
    #: merge 後半径 sqrt(r1^2+r2^2) が branch_diameter_max を超えたときの方針。
    #: "clamp"  = 上限で頭打ちにして merge を続ける (既定)
    #: "forbid" = 上限を超えるなら merge しない
    merge_radius_policy: str = "clamp"
    #: 目標半径では降下できないとき、その層で太らせずに降りることを許すか。
    #: False にすると即 UNREACHABLE (厳格モード)。
    #: True でもノードの radius は「実際に collision/reach を満たした半径」なので
    #: 細さを隠すことにはならない (radius_growth_blocked に回数を記録)。
    radius_growth_fallback: bool = True

    # --- centerline (STEP 9) ---
    #: tip 方向 (subtree の tip 重心) へ寄せる重み
    centerline_weight_tip: float = 1.0
    #: parent 方向 (鉛直維持) の重み
    centerline_weight_parent: float = 0.6
    #: 曲率 (直前の進行方向の延長) の重み
    centerline_weight_curvature: float = 0.4
    #: preferred angle を超えた分への罰則の重み
    centerline_weight_preferred: float = 1.5
    #: root 選択時、接地の安定性 (領域の縁からの距離) を評価する重み
    centerline_weight_stability: float = 0.5
    #: 候補点として拾う influence polygon の頂点数の上限
    centerline_max_vertex_candidates: int = 24
    #: 境界上に乗った候補点を内側へ寄せる量 [mm] (0 で無効)
    centerline_interior_nudge: float = 0.02

    # --- smoothing (STEP 12) ---
    smoothing_iterations: int = 40
    smoothing_lambda: float = 0.5
    bottom_up_trim: bool = True
    collision3d_enabled: bool = False
    collision3d_iterations: int = 8

    # --- メッシュ化 ---
    ring_segments: int = 16
    union_mode: UnionMode = UnionMode.NONE
    voxel_pitch: float = 0.25
    #: 直線的な区間の ring を間引く許容偏差 [mm]。0 で無効。
    mesh_simplify_deviation: float = 0.02

    # --- 数値 ---
    quad_segs: int = 8
    radius_quantum: float = 0.10
    area_epsilon: float = 1.0e-6
    #: world polygon の余白を明示指定する [mm]。0 なら自動計算 (下記 world_margin)。
    reach_margin: float = 0.0
    #: 自動計算時に上乗せする安全余裕 [mm]。
    reach_safety_margin: float = 5.0
    #: ビルドプレート矩形 (minx, miny, maxx, maxy)。指定すると world をここへ制限する。
    build_plate: tuple[float, float, float, float] | None = None
    #: collision offset に上乗せする絶対安全余裕 [mm]。
    collision_safety_epsilon: float = 0.002
    #: collision offset に上乗せする相対安全余裕 (offset 距離に対する比)。
    #: GEOS の buffer が入力を distance*0.01 相当で簡略化することへの保険。
    collision_safety_ratio: float = 0.02

    # --- 実行制御 ---
    debug_dir: str | None = None
    debug_stages: tuple[str, ...] = field(default_factory=tuple)
    max_layers: int | None = None            # デバッグ用の上限
    verbose: bool = False

    # ------------------------------------------------------------------
    # 派生量
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.layer_height <= 0.0:
            raise ValueError("layer_height must be > 0")
        if not (0.0 < self.overhang_angle < 90.0):
            raise ValueError("overhang_angle must be in (0, 90) degrees from vertical")
        if not (0.0 <= self.branch_angle_preferred <= self.branch_angle_max < 90.0):
            raise ValueError(
                "require 0 <= branch_angle_preferred <= branch_angle_max < 90"
            )
        if self.tip_diameter < self.min_printable_feature:
            raise ValueError("tip_diameter must be >= min_printable_feature")
        if self.branch_diameter_max < self.tip_diameter:
            raise ValueError("branch_diameter_max must be >= tip_diameter")
        if self.top_z_gap < 0.0 or self.bottom_z_gap < 0.0 or self.xy_gap < 0.0:
            raise ValueError("gaps must be >= 0")
        if self.tip_spacing <= 0.0:
            raise ValueError("tip_spacing must be > 0")
        if self.ring_segments < 6:
            raise ValueError("ring_segments must be >= 6")
        if self.merge_radius_policy not in ("clamp", "forbid"):
            raise ValueError("merge_radius_policy must be 'clamp' or 'forbid'")
        if isinstance(self.root_mode, str):
            self.root_mode = RootMode(self.root_mode)
        if isinstance(self.union_mode, str):
            self.union_mode = UnionMode(self.union_mode)
        if self.root_mode is RootMode.MODEL_ALLOWED:
            raise NotImplementedError(
                "root_mode=MODEL_ALLOWED is not implemented yet (MVP is BUILD_PLATE_ONLY)"
            )

    # --- 角度由来 ---
    @property
    def overhang_allowed_offset(self) -> float:
        """1 レイヤで許容される張り出し量 [mm]。"""
        return self.layer_height * math.tan(math.radians(self.overhang_angle))

    @property
    def move_max(self) -> float:
        """1 レイヤで枝中心が移動できる最大距離 [mm]。"""
        return self.layer_height * math.tan(math.radians(self.branch_angle_max))

    @property
    def move_preferred(self) -> float:
        """1 レイヤの推奨移動距離 [mm]。"""
        return self.layer_height * math.tan(math.radians(self.branch_angle_preferred))

    # --- 半径由来 ---
    @property
    def tip_radius(self) -> float:
        return self.tip_diameter * 0.5

    @property
    def max_radius(self) -> float:
        return self.branch_diameter_max * 0.5

    @property
    def root_radius_min(self) -> float:
        return self.root_diameter_min * 0.5

    @property
    def coverage_radius(self) -> float:
        """1 本の tip が支えたとみなす半径 [mm]。

        既定は ``tip_spacing / sqrt(2) * 1.05``。間隔 ``s`` の正方格子で
        最悪距離が ``s/sqrt(2)`` なので、理想的な格子なら coverage 100% になる。
        """
        if self.tip_coverage_radius > 0.0:
            return self.tip_coverage_radius
        return self.tip_spacing / math.sqrt(2.0) * 1.05

    @property
    def tip_min_distance(self) -> float:
        return self.tip_spacing * self.tip_min_distance_factor

    @property
    def tip_influence(self) -> float:
        """tip ノードの初期 influence area 半径 [mm]。"""
        if self.tip_influence_radius > 0.0:
            return self.tip_influence_radius
        return self.tip_radius

    @property
    def radius_growth_per_layer(self) -> float:
        """1 レイヤ降下あたりの半径増加 [mm]。"""
        return self.branch_diameter_growth * 0.5 * self.layer_height

    # --- z ギャップのレイヤ数 ---
    @property
    def z_gap_layers_top(self) -> int:
        return max(1, math.ceil(self.top_z_gap / self.layer_height - 1.0e-9))

    @property
    def z_gap_layers_bottom(self) -> int:
        return max(1, math.ceil(self.bottom_z_gap / self.layer_height - 1.0e-9))

    @property
    def actual_top_z_gap(self) -> float:
        """構成上保証される実際の Z ギャップ [mm]。"""
        return self.z_gap_layers_top * self.layer_height

    # --- world polygon ---
    def max_lateral_travel(self, model_height: float) -> float:
        """モデル頂部から z=0 まで最大傾斜で降りたときの水平移動量 [mm]。"""
        return max(0.0, model_height) * math.tan(math.radians(self.branch_angle_max))

    def world_margin(self, model_height: float) -> float:
        """モデル bbox に加える world polygon の余白 [mm]。

        枝がモデル外側へ迂回する経路を world 境界で不当に削らないよう、

            max_lateral_travel + branch_diameter_max/2 + xy_gap + safety_margin

        を確保する。``reach_margin`` を明示指定した場合はそちらを優先する。
        """
        if self.reach_margin > 0.0:
            return self.reach_margin
        return (self.max_lateral_travel(model_height)
                + self.max_radius + self.xy_gap + self.reach_safety_margin)

    # --- 便利メソッド ---
    def quantize_radius(self, radius: float) -> float:
        """半径を安全側 (大きい方) へ量子化する。"""
        q = self.radius_quantum
        n = math.ceil(radius / q - 1.0e-9)
        return max(q, n * q)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["root_mode"] = self.root_mode.value
        d["union_mode"] = self.union_mode.value
        d["derived"] = {
            "overhang_allowed_offset": self.overhang_allowed_offset,
            "move_max": self.move_max,
            "move_preferred": self.move_preferred,
            "tip_radius": self.tip_radius,
            "max_radius": self.max_radius,
            "z_gap_layers_top": self.z_gap_layers_top,
            "z_gap_layers_bottom": self.z_gap_layers_bottom,
            "actual_top_z_gap": self.actual_top_z_gap,
        }
        return d
