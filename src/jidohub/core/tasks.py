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

__all__ = ["TaskType", "IntermediateOutput", "Platform"]

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
    """
    複数フレームのセンサ or Detection3DOutputを入力 → 3D 物体追跡。出力は ``Tracking3DOutput``（``track_id`` 付き）。

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
    """
    複数フレームの画像 or Detection2DOutputを入力 → 2D 物体追跡。出力は ``Tracking2DOutput``（``track_id`` 付き）。

    例: SORT, DeepSORT
    """

    INSTANCE_SEGMENTATION_2D = "instance_segmentation_2d"
    """
    画像入力 → インスタンスマスク。出力は ``InstanceSegmentation2DOutput``。

    例: Mask2Former, SAM
    """

    INSTANCE_SEGMENTATION_2D_TRACKING = "instance_segmentation_2d_tracking"
    """
    複数フレーム画像入力 → インスタンスマスク追跡。出力は ``InstanceSegmentation2DTrackingOutput``（``track_id`` 付き）。

    例: SAM2, SAM3
    """

    # --- Prediction系（単体タスク） -------------------------------------------

    # --- Planning系 ---------------------------------------------------------
    SENSING_TO_PLANNING = "sensing_to_planning"
    """センサ入力 → 走行軌跡（E2E）。出力は ``E2EOutput``。

    中間出力（検出・追跡・マップ・動作予測・占有）を併せて返すことができる。

    例: UniAD, VAD
    """

    VISION_LANGUAGE_ACTION = "vision_language_action"
    """センサ入力 → 走行軌跡 + 自然言語（VLA）。出力は ``VLAOutput``（将来定義）。

    Phase 1 では出力スキーマ未定義。enum の値のみ予約する。
    """

    TRACK_MAP_TO_PLANNING = "track_map_to_planning"
    """追跡結果 + マップ → 走行軌跡（プランナ単体）。出力は ``PlanningOutput``。"""

    # --- Control系 ---------------------------------------------------------
    PLANNING_TO_CONTROL = "planning_to_control"
    """走行軌跡 → 制御指令（コントローラ単体）。出力は ``ControlOutput``（将来定義）。

    Phase 1 では出力スキーマ未定義。enum の値のみ予約する。
    """


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
