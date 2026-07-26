"""jidohub 標準スキーマ。

プラットフォーム全体の契約であり、**この定義が唯一の正**。
他リポジトリで同等の型を再定義しないこと。

型の追加・変更は :data:`SCHEMA_VERSION` への影響とセットで判断する。
"""

from __future__ import annotations

from jidohub.core.schemas.outputs import (
    AgentForecast,
    Box3D,
    CoordinateFrame,
    Detection3DOutput,
    E2EOutput,
    MapElement,
    MapElementType,
    MapOutput,
    MotionForecastOutput,
    PlanningOutput,
)
from jidohub.core.schemas.sample import (
    CameraFrame,
    DrivingCommand,
    EgoState,
    LidarSweep,
    RadarSweep,
    Sample,
)
from jidohub.core.schemas.version import SCHEMA_VERSION

__all__ = [
    "SCHEMA_VERSION",
    # 入力
    "Sample",
    "CameraFrame",
    "LidarSweep",
    "RadarSweep",
    "EgoState",
    "DrivingCommand",
    # 出力
    "CoordinateFrame",
    "Box3D",
    "Detection3DOutput",
    "MapElement",
    "MapElementType",
    "MapOutput",
    "AgentForecast",
    "MotionForecastOutput",
    "PlanningOutput",
    "E2EOutput",
]
