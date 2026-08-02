"""タスク種別の定義。

タスク種別は jidohub プラットフォーム全体の語彙であり、**このファイルが唯一の正**。
DB のマスタテーブル等に複製すると必ず乖離するため、Web/server/viewer はすべて
ここから読んだ定義を表示に使うこと。

命名は原則 ``<入力>_to_<出力>`` のスキームに従う。これにより
「Agent の入出力形式でフィルタする」というプラットフォームの検索軸と
タスク定義が 1 対 1 で対応する。ただしPerception系の単体タスクは
慣用に従い ``object_detection_3d`` のようにタスク内容そのものを名前にする。
"""

from __future__ import annotations

import re
from enum import Enum

__all__ = [
    "TaskType",
    "InputKind",
    "TASK_INPUT_KINDS",
    "IntermediateOutput",
    "Platform",
]

# タスク値は snake_case のみ許可（先頭は英小文字、以降は英小文字・数字・アンダースコア）。
# ハイフン（kebab-case）を弾くための唯一のパターン定義。
_TASK_VALUE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class TaskType(str, Enum):
    """Agent のタスク種別。

    ``agent_config.json`` の ``task`` フィールドの値であり、
    入力スキーマ（常に :class:`~jidohub.core.schemas.Sample`）と
    出力スキーマの組を決定する。

    値は snake_case の文字列。JSON への直列化時はそのまま文字列になる
    （``str`` を継承しているため）。値の形式は :data:`_TASK_VALUE_PATTERN` で
    強制され、ハイフンを含む値は enum の定義（import）時点で ``ValueError`` になる。
    """

    def __init__(self, value: str) -> None:
        # 値は snake_case のみ。新しい値にハイフン（kebab-case）が混入したら、
        # この enum を import した時点で落とす（テスト収集前に気付ける）。
        if not _TASK_VALUE_PATTERN.fullmatch(value):
            raise ValueError(
                f"TaskType value {value!r} must match "
                f"{_TASK_VALUE_PATTERN.pattern!r} (snake_case; ハイフン不可)"
            )

    # --- Perception系（単体タスク） -------------------------------------------
    OBJECT_DETECTION_3D = "object_detection_3d"
    """センサ入力 → 3D 物体検出。出力は ``Detection3DOutput``。

    例: CenterPoint, TransFusion, BEVFusion
    """

    OBJECT_TRACKING_3D = "object_tracking_3d"
    """複数フレームのセンサ入力 → 3D 物体追跡。出力は ``Tracking3DOutput``（予約）。

    出力スキーマ未定義。enum の値のみ予約する。

    例: AB3DMOT, CenterPoint Tracking
    """

    MAP_CONSTRUCTION = "map_construction"
    """センサ入力 → オンライン HD マップ構築。出力は ``MapOutput``。

    例: MapTR, HDMapNet
    """

    OBJECT_DETECTION_2D = "object_detection_2d"
    """画像入力 → 2D 物体検出。出力は ``Detection2DOutput``。

    例: YOLO, GroundingDINO
    """

    OBJECT_TRACKING_2D = "object_tracking_2d"
    """複数フレームの画像入力 → 2D 物体追跡。出力は ``Tracking2DOutput``（予約）。

    出力スキーマ未定義。enum の値のみ予約する。

    例: SORT, DeepSORT
    """

    INSTANCE_SEGMENTATION_2D = "instance_segmentation_2d"
    """画像入力 → インスタンスマスク。出力は ``InstanceSegmentation2DOutput``。

    例: Mask2Former, SAM
    """

    INSTANCE_SEGMENTATION_TRACKING_2D = "instance_segmentation_tracking_2d"
    """複数フレーム画像入力 → インスタンスマスク追跡。
    出力は ``InstanceSegmentation2DTrackingOutput``（予約）。

    出力スキーマ未定義。enum の値のみ予約する。サフィックス ``_2d`` は必ず末尾に置く
    （``instance_segmentation_2d_tracking`` は規約違反。`2d_tasks.md` 2 章）。

    例: SAM2, SAM3
    """

    SEMANTIC_SEGMENTATION_2D = "semantic_segmentation_2d"
    """画像入力 → セマンティックセグメンテーション。出力は ``SemanticSegmentation2DOutput``（予約）。

    出力スキーマ未定義。enum の値のみ予約する。
    """

    PANOPTIC_SEGMENTATION_2D = "panoptic_segmentation_2d"
    """画像入力 → パノプティックセグメンテーション。出力は ``PanopticSegmentation2DOutput``（予約）。

    出力スキーマ未定義。enum の値のみ予約する。
    """

    IMAGE_CLASSIFICATION_2D = "image_classification_2d"
    """画像入力 → 画像分類。出力は ``Classification2DOutput``。

    例: SigLIP2（ゼロショット）, ResNet, ViT
    """

    DEPTH_ESTIMATION = "depth_estimation"
    """画像入力 → 深度マップ。出力は ``DepthOutput``（予約）。

    出力スキーマ未定義。enum の値のみ予約する。相対 / 絶対は
    ``agent_config.json`` の宣言で区別する（`2d_tasks.md` 9.5）。
    次元サフィックスは付けない（画像平面にも 3D 空間にも一意に属さないため。2 章）。
    """

    MULTI_VIEW_RECONSTRUCTION = "multi_view_reconstruction"
    """複数視点画像 → 3D 再構成。出力は ``ReconstructionOutput``（予約）。

    出力スキーマ未定義。enum の値のみ予約する。

    例: VGGT, DUSt3R
    """

    VIDEO_TEXT_TO_TEXT = "video_text_to_text"
    """映像 + テキスト → テキスト。出力は ``TextOutput``（予約）。

    出力スキーマ未定義。enum の値のみ予約する。
    """

    # --- Prediction系 -------------------------------------------------------
    MOTION_FORECASTING = "motion_forecasting"
    """追跡結果 + マップ → 他車・歩行者の動作予測。出力は ``MotionForecastOutput``。

    次元サフィックスは付けない（`2d_tasks.md` 2 章）。
    """

    OCCUPANCY_PREDICTION = "occupancy_prediction"
    """センサ入力 → 占有グリッド予測。出力は ``OccupancyOutput``（予約）。

    出力スキーマ未定義。enum の値のみ予約する。
    """

    # --- Planning系 ---------------------------------------------------------
    SENSING_TO_PLANNING = "sensing_to_planning"
    """センサ入力 → 走行軌跡（E2E）。出力は ``E2EOutput``。

    中間出力（検出・追跡・マップ・動作予測・占有）を併せて返すことができる。

    例: UniAD, VAD
    """

    VISION_LANGUAGE_ACTION = "vision_language_action"
    """センサ入力 → 走行軌跡 + 自然言語（VLA）。出力は ``VLAOutput``（予約）。

    出力スキーマ未定義。enum の値のみ予約する。
    """

    TRACK_MAP_TO_PLANNING = "track_map_to_planning"
    """追跡結果 + マップ → 走行軌跡（プランナ単体）。出力は ``PlanningOutput``。"""

    # --- Control系 ---------------------------------------------------------
    CONTROL = "control"
    """走行軌跡 → 制御指令（コントローラ単体）。出力は ``ControlOutput``（予約）。

    出力スキーマ未定義。enum の値のみ予約する。
    """


class InputKind(str, Enum):
    """タスクが入力に取るコンテナの種別。

    2D タスクは :class:`~jidohub.core.schemas.Sample`（センサ入力）ではなく
    :class:`~jidohub.core.schemas.ImageSample`（画像入力）を取るため、config の
    センサ要求検証（:mod:`jidohub.core.config.agent`）が入力種別ごとに規則を分ける。
    """

    SENSOR = "sensor"
    """:class:`~jidohub.core.schemas.Sample` を入力に取る（センサ 1 つ以上が必要）。"""

    IMAGE = "image"
    """:class:`~jidohub.core.schemas.ImageSample` を入力に取る（センサを宣言しない）。"""

    COMPOSITE = "composite"
    """上流タスクの出力を含む複合入力（``track_map_to_planning`` 等）。"""


# タスク → 入力種別の対応表。TaskType と**同じファイル**に置き、語彙の唯一の正を分散させない。
# 予約タスク（出力型未定義）の分類は暫定であり、実装時に見直す余地がある。
TASK_INPUT_KINDS: dict["TaskType", InputKind] = {
    # SENSOR: Sample を入力に取る
    TaskType.OBJECT_DETECTION_3D: InputKind.SENSOR,
    TaskType.OBJECT_TRACKING_3D: InputKind.SENSOR,
    TaskType.MAP_CONSTRUCTION: InputKind.SENSOR,
    TaskType.OCCUPANCY_PREDICTION: InputKind.SENSOR,
    TaskType.SENSING_TO_PLANNING: InputKind.SENSOR,
    # IMAGE: ImageSample を入力に取る
    TaskType.OBJECT_DETECTION_2D: InputKind.IMAGE,
    TaskType.OBJECT_TRACKING_2D: InputKind.IMAGE,
    TaskType.INSTANCE_SEGMENTATION_2D: InputKind.IMAGE,
    TaskType.INSTANCE_SEGMENTATION_TRACKING_2D: InputKind.IMAGE,
    TaskType.SEMANTIC_SEGMENTATION_2D: InputKind.IMAGE,
    TaskType.PANOPTIC_SEGMENTATION_2D: InputKind.IMAGE,
    TaskType.IMAGE_CLASSIFICATION_2D: InputKind.IMAGE,
    TaskType.DEPTH_ESTIMATION: InputKind.IMAGE,
    TaskType.MULTI_VIEW_RECONSTRUCTION: InputKind.IMAGE,
    TaskType.VIDEO_TEXT_TO_TEXT: InputKind.IMAGE,
    # COMPOSITE: 複合入力（上流タスクの出力を含むケースが典型的）
    TaskType.VISION_LANGUAGE_ACTION: InputKind.COMPOSITE,  # VLA は Sample + テキストを取るため COMPOSITE
    TaskType.MOTION_FORECASTING: InputKind.COMPOSITE,
    TaskType.TRACK_MAP_TO_PLANNING: InputKind.COMPOSITE,
    TaskType.CONTROL: InputKind.COMPOSITE,
}


class IntermediateOutput(str, Enum):
    """E2E Agent が公開できる中間出力の種別。

    値は :class:`~jidohub.core.schemas.E2EOutput` の**フィールド名と一致させる**。
    ``agent_config.json`` の ``intermediate_outputs`` で宣言された項目が、
    実際に ``E2EOutput`` の対応フィールドに埋まることを保証する契約になる。

    可視化側は「この enum 値 → 描画レイヤー」の対応表を 1 つ持てばよく、
    Agent が増えても可視化の変更が不要になる。
    """

    DETECTION = "detection"
    TRACKING = "tracking"
    MAP = "map"
    MOTION_FORECAST = "motion_forecast"
    OCCUPANCY = "occupancy"
    BEV_FEATURE = "bev_feature"


class Platform(str, Enum):
    """Agent が動作する実行プラットフォーム。

    jidohub-web の検索フィルタ軸（スライドの「動作プラットフォームをフィルタ」）と
    1 対 1 で対応する。
    """

    PYTHON = "python"
    """Python API から利用できる（jidohub-agents 経由）。"""

    ROS = "ros"
    """ROS ノードとして起動できる（jidohub-interfaces 経由）。"""
