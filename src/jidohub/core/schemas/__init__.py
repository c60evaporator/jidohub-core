"""jidohub 標準スキーマ。

プラットフォーム全体の契約であり、**この定義が唯一の正**。
他リポジトリで同等の型を再定義しないこと。

型の追加・変更は :data:`SCHEMA_VERSION` への影響とセットで判断する。
"""

from __future__ import annotations

from jidohub.core.schemas.image import (
    EncodedPixels,
    Image,
    ImageDecodeError,
    ImageDecoder,
    ImageFormat,
    ImageSource,
    decode_image,
    get_image_decoder,
    register_image_decoder,
)
from jidohub.core.schemas.outputs import (
    AgentForecast,
    Box2D,
    Box3D,
    Classification2DOutput,
    CoordinateFrame,
    Detection2DOutput,
    Detection3DOutput,
    E2EOutput,
    Instance2D,
    InstanceSegmentation2DOutput,
    MapElement,
    MapElementType,
    MapOutput,
    MotionForecastOutput,
    PlanningOutput,
)
from jidohub.core.schemas.prompts import (
    ImagePrompt,
    ImageSample,
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
    # 入力（センサ）
    "Sample",
    "CameraFrame",
    "LidarSweep",
    "RadarSweep",
    "EgoState",
    "DrivingCommand",
    # 入力（2D / 画像）
    "ImageSample",
    "ImagePrompt",
    # 画像
    "Image",
    "ImageSource",
    "EncodedPixels",
    "ImageFormat",
    "ImageDecoder",
    "ImageDecodeError",
    "register_image_decoder",
    "get_image_decoder",
    "decode_image",
    # 出力（3D / E2E）
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
    # 出力（2D）
    "Box2D",
    "Instance2D",
    "Detection2DOutput",
    "InstanceSegmentation2DOutput",
    "Classification2DOutput",
]
