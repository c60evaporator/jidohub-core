"""タスク別の出力スキーマ。

:class:`~jidohub.core.tasks.TaskType` と 1 対 1 で対応する出力型を定義する。

設計方針
    E2E（``sensing_to_planning``）の中間出力は、単体タスクの出力型を
    **再利用**して表現する。これにより UniAD の中間出力可視化を、
    Detection / Map 単体タスク用の可視化コードのまま実現できる。
    新しい中間出力を追加する際も、まず単体タスクの型として定義してから
    :class:`E2EOutput` に載せること。

規約（詳細は CLAUDE.md 3 章）
    - 出力の座標系は **ego 座標系を既定**とし、``frame`` フィールドで必ず明示する。
    - 回転は quaternion ``(w, x, y, z)`` を正とし、``yaw`` は派生プロパティ。
    - 寸法は ``(length, width, height)`` の順（ego 座標の x, y, z 軸方向に対応）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from jidohub.core.geometry import quaternion_to_yaw

__all__ = [
    "CoordinateFrame",
    "Box3D",
    "Detection3DOutput",
    "MapElementType",
    "MapElement",
    "MapOutput",
    "AgentForecast",
    "MotionForecastOutput",
    "PlanningOutput",
    "E2EOutput",
]


class CoordinateFrame(str, Enum):
    """出力が表現されている座標系。

    暗黙の前提を作らないため、座標を持つ出力型は必ずこのフィールドを持つ。
    """

    EGO = "ego"
    """ego 座標系（x = 前方、y = 左方、z = 上方）。既定値。"""

    GLOBAL = "global"
    """global（マップ）座標系。"""


@dataclass
class Box3D:
    """3D バウンディングボックス 1 個。

    検出・追跡・E2E の中間出力で共通して使う。

    Attributes:
        center: shape ``(3,)``、``np.float64``。ボックス中心の座標[m]。
            **底面中心ではなく重心（幾何中心）**である点に注意（nuScenes 準拠）。
        size: shape ``(3,)``、``np.float64``。``(length, width, height)``[m] の順で、
            ego 座標系の x / y / z 軸方向の寸法に対応する。
            **nuScenes の ``sample_annotation.size`` は ``(width, length, height)`` 順**で
            あり、Adapter で必ず入れ替えが必要（取り違えても例外は出ないため注意）。
            個別の値には ``size[0]`` 等でインデックスせず、
            :attr:`length` / :attr:`width` / :attr:`height` プロパティを使うこと。
        rotation: shape ``(4,)``、``np.float64``。``(w, x, y, z)`` 順のクォータニオン。
            yaw のみを扱うモデルからの変換には
            :func:`~jidohub.core.geometry.yaw_to_quaternion` を使う。
        label: クラス名の文字列（例: ``"car"``）。
            クラス体系はデータセット・モデル依存のため core では enum 化しない。
        score: 信頼度スコア ``[0, 1]``。GT の場合は ``None``。
        velocity: shape ``(2,)`` または ``(3,)``、``np.float64``。速度[m/s]。
            ``frame`` と同じ座標系で表す。
        track_id: 追跡 ID。単体検出では ``None``。
        attributes: データセット固有の属性（nuScenes の ``vehicle.moving`` 等）。
    """

    center: np.ndarray
    size: np.ndarray
    rotation: np.ndarray
    label: str
    score: float | None = None
    velocity: np.ndarray | None = None
    track_id: int | None = None
    attributes: dict = field(default_factory=dict)

    @property
    def yaw(self) -> float:
        """z 軸まわりの yaw 角[rad]（読み取り専用の派生値）。

        BEV 描画や yaw ベースのフレームワークとの連携用。
        **この値を正として保持しないこと**（情報が欠落するため）。
        """
        return quaternion_to_yaw(self.rotation)

    @property
    def length(self) -> float:
        """車長[m]。ego 座標系の x 軸方向の寸法（``size[0]``）。"""
        return float(self.size[0])

    @property
    def width(self) -> float:
        """車幅[m]。ego 座標系の y 軸方向の寸法（``size[1]``）。"""
        return float(self.size[1])

    @property
    def height(self) -> float:
        """車高[m]。ego 座標系の z 軸方向の寸法（``size[2]``）。"""
        return float(self.size[2])

    @classmethod
    def from_dimensions(
        cls,
        center: np.ndarray,
        length: float,
        width: float,
        height: float,
        rotation: np.ndarray,
        label: str,
        **kwargs: object,
    ) -> Box3D:
        """寸法をキーワードで指定して構築する。

        ``size`` の並び順を意識せずに済むため、データセット Adapter からの
        構築ではこちらを使うことを推奨する。nuScenes のように
        ``(width, length, height)`` 順で寸法を保持する形式からの変換で、
        取り違えを防げる。

        ``score`` / ``velocity`` / ``track_id`` / ``attributes`` は ``kwargs`` で
        そのまま渡せる。
        """
        size = np.array([length, width, height], dtype=np.float64)
        return cls(center=center, size=size, rotation=rotation, label=label, **kwargs)  # type: ignore[arg-type]


@dataclass
class Detection3DOutput:
    """3D 物体検出・追跡の出力。

    ``object_detection_3d`` と ``object_tracking_3d`` で共通。
    追跡の場合は各 :class:`Box3D` の ``track_id`` が埋まる。

    Attributes:
        boxes: 検出されたボックスのリスト。スコア降順である必要はない。
        frame: ``boxes`` が表現されている座標系。
    """

    boxes: list[Box3D] = field(default_factory=list)
    frame: CoordinateFrame = CoordinateFrame.EGO


class MapElementType(str, Enum):
    """HD マップ要素の種別。

    オンラインマップ構築（``sensing_to_map``）の出力と、
    nuScenes Map Expansion の GT の双方で使う共通語彙。
    """

    DIVIDER = "divider"
    """車線境界線。"""

    PED_CROSSING = "ped-crossing"
    """横断歩道。"""

    BOUNDARY = "boundary"
    """走行可能領域の境界。"""

    CENTERLINE = "centerline"
    """車線中心線。"""


@dataclass
class MapElement:
    """HD マップの 1 要素（ポリライン / ポリゴン）。

    PostGIS へそのまま格納できる形を意図している。GeoJSON への変換は
    利用側（nuscenes-viewer 等）が行い、core では変換を提供しない。

    Attributes:
        points: shape ``(P, 2)`` または ``(P, 3)``、``np.float64``。頂点列[m]。
        element_type: 要素種別。
        is_closed: 閉じた形状（ポリゴン）なら ``True``、ポリラインなら ``False``。
        score: 信頼度スコア ``[0, 1]``。GT の場合は ``None``。
        element_id: 要素の識別子（GT との対応付けや追跡に使う）。
    """

    points: np.ndarray
    element_type: MapElementType
    is_closed: bool = False
    score: float | None = None
    element_id: str | None = None


@dataclass
class MapOutput:
    """オンライン HD マップ構築の出力（``sensing_to_map``）。

    Attributes:
        elements: マップ要素のリスト。
        frame: ``elements`` が表現されている座標系。
    """

    elements: list[MapElement] = field(default_factory=list)
    frame: CoordinateFrame = CoordinateFrame.EGO


@dataclass
class AgentForecast:
    """他車・歩行者 1 体分の動作予測。

    Attributes:
        trajectories: shape ``(M, T, 2)``、``np.float64``。
            ``M`` = マルチモーダル予測のモード数、``T`` = 予測ステップ数。
            単一モードの場合も ``M = 1`` として 3 次元で持つ（形状を分岐させない）。
        probabilities: shape ``(M,)``、``np.float64``。各モードの確率。総和 1。
        dt: 予測ステップの時間間隔[s]。
        track_id: 対応する :class:`Box3D` の ``track_id``。
        frame: 座標系。
    """

    trajectories: np.ndarray
    probabilities: np.ndarray
    dt: float
    track_id: int | None = None
    frame: CoordinateFrame = CoordinateFrame.EGO


@dataclass
class MotionForecastOutput:
    """動作予測の出力。

    Attributes:
        forecasts: 各対象の予測のリスト。
    """

    forecasts: list[AgentForecast] = field(default_factory=list)


@dataclass
class PlanningOutput:
    """自車の走行計画（``track_map_to_planning``）。

    Attributes:
        trajectory: shape ``(T, 2)`` または ``(T, 3)``、``np.float64``。
            **ego 座標系**での自車の将来位置[m]。現在位置（原点）は含めず、
            ``dt`` 後の点から始める。
        dt: 軌跡の時間間隔[s]（例: ``0.5``）。
        frame: 座標系。通常 ``EGO``。
        confidence: 計画の信頼度 ``[0, 1]``。モデルが出力しない場合は ``None``。
    """

    trajectory: np.ndarray
    dt: float
    frame: CoordinateFrame = CoordinateFrame.EGO
    confidence: float | None = None


@dataclass
class E2EOutput:
    """E2E モデルの出力（``sensing_to_planning``）。

    主たる出力は ``planning``。それ以外は**中間出力**であり、モデルが
    公開している場合のみ埋まる（``agent_config.json`` の
    ``intermediate_outputs`` で宣言されたものが対応する）。

    中間出力はすべて単体タスクの出力型を再利用しているため、
    可視化側は「出力型 → 描画レイヤー」の対応表を 1 つ持てばよい。

    Attributes:
        planning: 自車の走行計画。E2E の主出力。
        detection: 中間出力の 3D 検出結果。
        tracking: 中間出力の追跡結果（``Box3D.track_id`` が埋まる）。
        map: 中間出力のオンラインマップ。
        motion_forecast: 中間出力の他車動作予測。
        occupancy: 中間出力の占有予測。shape ``(T, H, W)``、``np.float32``。
            ``T`` = 予測ステップ数、``H``/``W`` = BEV グリッドの縦横。
            グリッドの分解能と原点は ``occupancy_meta`` で表す。
        occupancy_meta: 占有グリッドのメタ情報。
            ``{"resolution": float[m/cell], "origin": (x, y)[m], "dt": float[s]}``。
        bev_feature: 中間出力の BEV 特徴量。shape ``(C, H, W)``、``np.float32``。
            デバッグ・可視化用途。サイズが大きいため既定では返さない実装を推奨。
    """

    planning: PlanningOutput
    detection: Detection3DOutput | None = None
    tracking: Detection3DOutput | None = None
    map: MapOutput | None = None
    motion_forecast: MotionForecastOutput | None = None
    occupancy: np.ndarray | None = None
    occupancy_meta: dict | None = None
    bev_feature: np.ndarray | None = None
