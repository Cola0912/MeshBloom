"""STEP 5: Collision 領域と Reachability (Cura の avoidance に相当) のキャッシュ。

Collision
---------
ノードレイヤ ``j``・枝半径 ``r`` に対して、枝中心が入ってはいけない領域::

    base[j]          = U_{i in [j-1-n_bot, j]} dilate(P[i], xy_gap)
                     U U_{t in [1, n_top_k]}   dilate(P[j+t], xy_t)
    collision[j][r]  = dilate(base[j], r)

``dilate(A, a+b) == dilate(dilate(A, a), b)`` (Minkowski 和の結合律) かつ
``dilate(A U B, r) == dilate(A, r) U dilate(B, r)`` なので、
半径に依存しない ``base[j]`` を 1 度だけ作れば、半径ごとの offset は 1 回で済む。

Z 方向の範囲 (docs/ARCHITECTURE.md 6 節):

* 下側 ``i in [j-1-n_bot, j]``: 枝下端 ``(j-1)*lh`` とモデル上端の間に
  ``bottom_z_gap`` を確保する。
* 上側 (``kind="branch"``) ``t in [1, n_top]``: 枝上端 ``(j+1)*lh`` から見て
  垂直距離が ``top_z_gap`` 未満のモデル。
* 上側 (``kind="tip"``) ``t in [1, n_top-1]``: tip は枝の最上点で上に材料が無いため
  1 層分ゆるい。これにより「オーバーハングの真下に tip を置けない」問題を
  ギャップを広げずに解決する。

Reachability
------------
``reach[r][j]`` = 「半径 r の枝中心をレイヤ j のその点に置いたとき、
1 レイヤあたり ``move_max`` 以内の移動でビルドプレートまで降りられる」領域::

    reach[r][0] = world - collision[0][r]
    reach[r][j] = (dilate(reach[r][j-1], move_max) & world) - collision[j][r]

伝播時に collision ではなく reach を使うことで、枝が行き止まりに入り込むのを
構造的に防ぐ。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from shapely.geometry.base import BaseGeometry

from . import geometry2d as g2
from .config import SupportConfig
from .debug_io import DebugWriter, SvgLayer
from .slicer import LayerStack

__all__ = ["CollisionCache"]


@dataclass
class CacheStats:
    collision_hits: int = 0
    collision_misses: int = 0
    reach_layers_computed: int = 0
    base_computed: int = 0
    radii: set = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "collision_hits": self.collision_hits,
            "collision_misses": self.collision_misses,
            "reach_layers_computed": self.reach_layers_computed,
            "base_computed": self.base_computed,
            "distinct_radii": sorted(round(r, 4) for r in self.radii),
        }


class CollisionCache:
    """collision / reach の遅延計算キャッシュ。"""

    def __init__(self, config: SupportConfig, stack: LayerStack) -> None:
        self.config = config
        self.stack = stack
        self.stats = CacheStats()

        self._base: dict[tuple[str, int], BaseGeometry] = {}
        self._collision: dict[tuple[str, float, int], BaseGeometry] = {}
        #: 量子化半径 -> レイヤごとの reach。未計算部分は None。
        self._reach: dict[float, list[BaseGeometry | None]] = {}

        self.world = self._make_world()

    # ------------------------------------------------------------------
    def _make_world(self) -> BaseGeometry:
        """枝が存在しうる有限領域。

        余白は ``max_lateral_travel + max_radius + xy_gap + safety_margin``。
        上限クリップは行わない。上限を掛けると、高いモデルで
        「本来存在する迂回経路が world 境界で消える」ことがあるため。
        ``build_plate`` を指定した場合のみ、その矩形で制限する。
        """
        cfg = self.config
        b = self.stack.xy_bounds or (0.0, 0.0, 1.0, 1.0)
        margin = cfg.world_margin(self.stack.model_z_max)
        world = g2.box_polygon(b, margin)
        if cfg.build_plate is not None:
            world = g2.intersection(world, g2.box_polygon(cfg.build_plate, 0.0))
        return world

    # ------------------------------------------------------------------
    def quantize(self, radius: float) -> float:
        return self.config.quantize_radius(radius)

    # --- base (offset 前の生の和。半径にも xy_gap にも非依存) -----------
    def base(self, layer: int, kind: str = "branch") -> list[tuple[float, BaseGeometry]]:
        """``(必要な xy クリアランス, その値を要求するモデル断面の和)`` の列。

        **offset を掛ける前の生の和**を返すことが重要。
        ここで xy_gap を掛けてしまうと、後段の半径 offset と合わせて
        buffer を 2 回通ることになり、GEOS が内部で行う入力簡略化
        (``distance * 0.01`` 相当) の影響で結果が**危険側に小さく**なりうる。
        生の和にしておけば ``dilate(base, xy_gap + r)`` の 1 回で済み、
        直接計算と厳密に一致する。

        ``xy_taper`` が無効 (既定) なら要素は 1 つだけなので、
        半径ごとの offset も 1 回で済む。
        """
        key = (kind, layer)
        cached = self._base.get(key)
        if cached is not None:
            return cached

        cfg = self.config
        n_top = cfg.z_gap_layers_top
        n_bot = cfg.z_gap_layers_bottom
        groups: dict[float, list[BaseGeometry]] = {}

        def add(xy: float, geom: BaseGeometry) -> None:
            if geom.is_empty:
                return
            groups.setdefault(round(xy, 9), []).append(geom)

        # 同一高さ〜下側: フルの xy_gap
        for i in range(layer - 1 - n_bot, layer + 1):
            add(cfg.xy_gap, self.stack[i])

        # 上側 (tip は 1 層分ゆるい)
        t_max = n_top if kind == "branch" else n_top - 1
        for t in range(1, t_max + 1):
            xy = (cfg.xy_gap * max(0.0, 1.0 - (t - 0.5) / n_top)
                  if cfg.xy_taper else cfg.xy_gap)
            add(xy, self.stack[layer + t])

        result = [(xy, g2.union(parts)) for xy, parts in sorted(groups.items())]
        self.stats.base_computed += 1
        self._base[key] = result
        return result

    # --- collision ------------------------------------------------------
    def collision(self, layer: int, radius: float,
                  kind: str = "branch") -> BaseGeometry:
        rq = self.quantize(radius)
        key = (kind, rq, layer)
        cached = self._collision.get(key)
        if cached is not None:
            self.stats.collision_hits += 1
            return cached
        self.stats.collision_misses += 1
        self.stats.radii.add(rq)

        cfg = self.config
        eps = cfg.collision_safety_epsilon
        parts = []
        for xy, geom in self.base(layer, kind):
            if geom.is_empty:
                continue
            d = xy + rq
            # 離散化誤差で危険側へ縮まないよう、わずかに安全側へ寄せる
            d += max(eps, d * cfg.collision_safety_ratio)
            parts.append(g2.dilate_outer(geom, d, quad_segs=cfg.quad_segs))

        result = g2.union(parts)
        self._collision[key] = result
        return result

    def direct_collision(self, layer: int, radius: float,
                         kind: str = "branch") -> BaseGeometry:
        """最適化を使わずに ``U_i dilate(P[i], r + xy_i)`` を直接計算する。

        ``collision()`` の base 再利用が離散化誤差で**危険側 (小さい側)** へ
        ずれていないことを確認するための参照実装。性能は考慮していない。
        """
        cfg = self.config
        rq = self.quantize(radius)
        n_top = cfg.z_gap_layers_top
        n_bot = cfg.z_gap_layers_bottom
        parts: list[BaseGeometry] = []

        for i in range(layer - 1 - n_bot, layer + 1):
            s = self.stack[i]
            if not s.is_empty:
                parts.append(g2.dilate_outer(s, cfg.xy_gap + rq,
                                             quad_segs=cfg.quad_segs))

        t_max = n_top if kind == "branch" else n_top - 1
        for t in range(1, t_max + 1):
            s = self.stack[layer + t]
            if s.is_empty:
                continue
            xy = (cfg.xy_gap * max(0.0, 1.0 - (t - 0.5) / n_top)
                  if cfg.xy_taper else cfg.xy_gap)
            parts.append(g2.dilate_outer(s, xy + rq, quad_segs=cfg.quad_segs))

        return g2.union(parts)

    # --- reach ----------------------------------------------------------
    def reach(self, layer: int, radius: float) -> BaseGeometry:
        """レイヤ ``layer`` でビルドプレートへ到達可能な枝中心の領域。"""
        if layer < 0:
            return g2.EMPTY
        rq = self.quantize(radius)
        column = self._reach.get(rq)
        if column is None:
            column = [None] * max(1, self.stack.n_layers + 1)
            self._reach[rq] = column
        if layer >= len(column):
            column.extend([None] * (layer + 1 - len(column)))

        if column[layer] is not None:
            return column[layer]

        # 未計算の最下層から順に埋める。
        start = 0
        for k in range(layer, -1, -1):
            if column[k] is not None:
                start = k + 1
                break

        cfg = self.config
        simplify_tol = cfg.radius_quantum * 0.25
        for k in range(start, layer + 1):
            if k == 0:
                geom = g2.difference(self.world, self.collision(0, rq, "branch"))
            else:
                prev = column[k - 1]
                if prev is None or prev.is_empty:
                    geom = g2.EMPTY
                else:
                    grown = g2.dilate(prev, cfg.move_max, quad_segs=cfg.quad_segs)
                    if simplify_tol > 0.0 and not grown.is_empty:
                        # 簡略化は「縮む方向のみ」に限定する。外側へ膨らむと
                        # 「reach[j] の点から reach[j-1] へ move_max で降りられる」
                        # という不変条件が破れ、伝播が行き止まりになる。
                        simplified = g2.clean(
                            grown.simplify(simplify_tol, preserve_topology=True))
                        if not simplified.is_empty:
                            grown = g2.intersection(simplified, grown)
                    grown = g2.intersection(grown, self.world)
                    geom = g2.difference(grown, self.collision(k, rq, "branch"))
            column[k] = geom
            self.stats.reach_layers_computed += 1
        return column[layer]

    def reach_tip(self, layer: int, radius: float) -> BaseGeometry:
        """tip を置ける領域 (最上層だけ tip 用 collision を使う)。"""
        if layer < 0:
            return g2.EMPTY
        rq = self.quantize(radius)
        cfg = self.config
        if layer == 0:
            grown = self.world
        else:
            below = self.reach(layer - 1, rq)
            if below.is_empty:
                return g2.EMPTY
            grown = g2.dilate(below, cfg.move_max, quad_segs=cfg.quad_segs)
            grown = g2.intersection(grown, self.world)
        return g2.difference(grown, self.collision(layer, rq, "tip"))

    # ------------------------------------------------------------------
    def tip_allowed_by_layer(self, layers) -> dict[int, BaseGeometry]:
        """``ContactSampler.sample`` に渡す allowed 領域を作る。"""
        r = self.config.tip_radius
        return {j: self.reach_tip(j, r) for j in layers}

    # ------------------------------------------------------------------
    def write_debug(self, debug: DebugWriter, radius: float,
                    layers=None, every: int = 1) -> int:
        if not debug.enabled("collision"):
            return 0
        rq = self.quantize(radius)
        if layers is None:
            layers = range(0, self.stack.n_layers, max(1, every))
        written = 0
        for j in layers:
            col = self.collision(j, rq, "branch")
            rch = self.reach(j, rq)
            debug.write_svg(  # noqa: E501
                "collision", f"layer_{j:04d}_collision_r{rq:.2f}.svg",
                [
                    SvgLayer("reach", rch, fill="#bbf7d0", stroke="#16a34a",
                             opacity=0.5),
                    SvgLayer("collision", col, fill="#fecaca", stroke="#dc2626",
                             opacity=0.7),
                    SvgLayer("model", self.stack[j], fill="#64748b",
                             stroke="#1e293b", opacity=0.9),
                ],
                bounds=self.world.bounds,
                title=f"layer {j:04d} r={rq:.2f} collision/reach",
            )
            written += 1
        debug.write_json("collision", "collision_stats.json", self.stats.to_dict())
        return written
