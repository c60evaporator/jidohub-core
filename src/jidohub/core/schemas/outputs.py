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
    "Box2D",
    "Instance2D",
    "Detection2DOutput",
    "InstanceSegmentation2DOutput",
    "Classification2DOutput",
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

    CAMERA = "camera"
    """カメラ座標系。単眼深度（``depth_estimation`` の ``DepthOutput``、将来定義）の
    出力は ego 座標に置けない（外部パラメータが必要で、それを持つのは ``CameraFrame`` を
    握っている呼び出し側）ため予約する。後から追加すると ``schema_version`` を上げる
    変更になるため、`2d_tasks.md` 6.4 に従い今のうちに値だけ用意する。"""


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


@dataclass
class Box2D:
    """2D バウンディングボックス 1 個（画像平面）。

    検出・追跡・インスタンス分割で共通して使う。

    Attributes:
        xyxy: shape ``(4,)``、``np.float64``。``(x0, y0, x1, y1)``[px]。原点は左上、
            ``x`` 右・``y`` 下（`2d_tasks.md` 3.2）。フィールド名を ``xyxy`` にすることで
            xywh との取り違えをコード上で防ぐ（``Box3D.size`` の並び順と同じ発想）。
            COCO 形式（xywh）への変換は評価層の責務であり core では扱わない。
        label: クラス名の**文字列**。ゼロショット分類・オープン語彙検出では
            ラベル集合が実行時に決まるためクラスインデックスにできない
            （`2d_tasks.md` 1 章）。プロンプタブル系では ``None`` になり得る。
        score: 信頼度スコア ``[0, 1]``。GT や単純なプロンプト応答では ``None``。
        track_id: 追跡 ID。単発検出では ``None``。``object_tracking_2d`` 追加時に
            フィールドを足すと破壊的変更になるため、**最初から**用意する（`2d_tasks.md` 9.1）。
        attributes: データセット固有の属性。
    """

    xyxy: np.ndarray
    label: str | None = None
    score: float | None = None
    track_id: int | None = None
    attributes: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.xyxy.shape != (4,):
            raise ValueError(f"Box2D.xyxy must have shape (4,), got {self.xyxy.shape}")


@dataclass
class Instance2D:
    """インスタンスセグメンテーションの 1 インスタンス。

    マスクは**全画面ではなく bbox 内に限定**して持つ。素朴な ``(N, H, W)`` の全画面
    マスクは 1600x900 で 50 インスタンスなら約 72MB になり、プロセス境界を越える設計では
    実用にならないため（`2d_tasks.md` 6.2）。

    Attributes:
        box: このインスタンスのバウンディングボックス。
        mask: shape ``(h, w)``、``np.bool_``。``mask_region`` が示す整数画素領域を覆う。
        mask_region: ``(x0, y0, x1, y1)`` の**整数**画素領域。``x1 - x0 == w``、
            ``y1 - y0 == h`` を満たす。``box.xyxy``（float）とは別に整数領域を持つのは、
            float から暗黙に丸めると実装ごとに 1 画素ずれるため（曖昧さの排除）。
    """

    box: Box2D
    mask: np.ndarray | None = None
    mask_region: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if (self.mask is None) != (self.mask_region is None):
            raise ValueError("Instance2D.mask and mask_region must be provided together")
        if self.mask is not None:
            if self.mask.ndim != 2 or self.mask.dtype != np.bool_:
                raise ValueError(
                    "Instance2D.mask must be a 2-D bool array, "
                    f"got shape={self.mask.shape} dtype={self.mask.dtype}"
                )
            assert self.mask_region is not None  # 上の同時指定チェックで保証済み
            x0, y0, x1, y1 = self.mask_region
            height, width = self.mask.shape
            if (x1 - x0, y1 - y0) != (width, height):
                raise ValueError(
                    "Instance2D.mask_region size must match mask shape "
                    f"(expected x1-x0={width}, y1-y0={height}; got {(x1 - x0, y1 - y0)})"
                )


@dataclass
class Detection2DOutput:
    """2D 物体検出の出力（``object_detection_2d``）。

    座標は**入力 :class:`~jidohub.core.schemas.Image` の現サイズ基準**（`2d_tasks.md` 3.2）。
    3D 出力と異なり座標系の選択の余地がない（画像平面に一意）ため
    :class:`CoordinateFrame` フィールドは持たない。元画像へ戻す必要がある場合は
    :attr:`~jidohub.core.schemas.image.Image.source` を用いる。

    Attributes:
        boxes: 検出されたボックスのリスト。スコア降順である必要はない。
    """

    boxes: list[Box2D] = field(default_factory=list)


@dataclass
class InstanceSegmentation2DOutput:
    """インスタンスセグメンテーションの出力（``instance_segmentation_2d``）。

    座標系の扱いは :class:`Detection2DOutput` と同じ（現サイズ基準・``frame`` なし）。

    Attributes:
        instances: 検出されたインスタンスのリスト。
    """

    instances: list[Instance2D] = field(default_factory=list)


@dataclass
class Classification2DOutput:
    """画像分類の出力（``image_classification_2d``）。

    Attributes:
        labels: クラス名の**文字列**リスト。スコア降順。ゼロショット分類では
            ラベル集合が実行時に決まるため文字列で持つ（`2d_tasks.md` 1 章）。
        scores: shape ``(K,)``、``np.float64``。``labels`` と同順・同数。
            モデルがスコアを出さない場合は ``None``。
    """

    labels: list[str] = field(default_factory=list)
    scores: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.scores is not None and self.scores.shape[0] != len(self.labels):
            raise ValueError(
                "Classification2DOutput.scores length must match labels "
                f"({self.scores.shape[0]} != {len(self.labels)})"
            )


class MapElementType(str, Enum):
    """HD マップ要素の種別。

    オンラインマップ構築（``map_construction``）の出力と、
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
    """オンライン HD マップ構築の出力（``map_construction``）。

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
