"""スキーマオブジェクトと JSON 可能ツリーの相互変換。

numpy 配列は JSON に載せず、**別途バイナリバッファとして取り出す**。
ツリー側には配列の参照（インデックス）だけを残す。
バイナリ framing は :mod:`jidohub.core.serialization.envelope` が担う。

分離の理由
    server の API 境界では multipart や事前署名 URL で本体を別送したい場合があり、
    「ツリー + バッファ列」の形で取り出せると再利用が効く。
    バイト列が欲しいだけなら :func:`~jidohub.core.serialization.pack` を使う。

セキュリティ
    デコード時に復元できる型は :data:`TYPE_REGISTRY` の**許可リストに限る**。
    ペイロードに書かれた任意のクラス名を import して構築することはしない
    （信頼できない送信元からのペイロードを受け取り得るため）。
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any

import numpy as np

from jidohub.core import schemas, tasks

__all__ = ["TYPE_REGISTRY", "encode", "decode", "SerializationError"]


class SerializationError(ValueError):
    """シリアライズ／デシリアライズに失敗した場合に送出される例外。"""


_DATACLASS_TYPES: tuple[type, ...] = (
    schemas.Sample,
    schemas.CameraFrame,
    schemas.LidarSweep,
    schemas.RadarSweep,
    schemas.EgoState,
    schemas.EncodedImage,
    schemas.Box3D,
    schemas.Detection3DOutput,
    schemas.MapElement,
    schemas.MapOutput,
    schemas.AgentForecast,
    schemas.MotionForecastOutput,
    schemas.PlanningOutput,
    schemas.E2EOutput,
)

_ENUM_TYPES: tuple[type[Enum], ...] = (
    schemas.DrivingCommand,
    schemas.ImageFormat,
    schemas.CoordinateFrame,
    schemas.MapElementType,
    tasks.TaskType,
    tasks.IntermediateOutput,
    tasks.Platform,
)

TYPE_REGISTRY: dict[str, type] = {cls.__name__: cls for cls in (*_DATACLASS_TYPES, *_ENUM_TYPES)}
"""デコード時に復元を許可する型の許可リスト。

**新しいスキーマ型を追加したら必ずここに登録する。** 登録漏れは
エンコード時に :class:`SerializationError` として検出される。
"""

_TYPE_KEY = "__type__"
_ENUM_KEY = "__enum__"
_ARRAY_KEY = "__array__"
_TUPLE_KEY = "__tuple__"


def encode(obj: Any) -> tuple[Any, list[np.ndarray]]:
    """オブジェクトを JSON 可能ツリーとバッファ列に変換する。

    Args:
        obj: スキーマのインスタンス、またはそれらを含むコンテナ。

    Returns:
        ``(tree, buffers)``。``tree`` は JSON 直列化可能。
        ``buffers`` は C 連続化された numpy 配列のリスト。

    Raises:
        SerializationError: 未対応の型が含まれる場合。
    """
    buffers: list[np.ndarray] = []
    tree = _encode_value(obj, buffers)
    return tree, buffers


def decode(tree: Any, buffers: list[np.ndarray]) -> Any:
    """:func:`encode` の逆変換。

    Args:
        tree: :func:`encode` が返したツリー。
        buffers: :func:`encode` が返したバッファ列（順序を保つこと）。

    Raises:
        SerializationError: 未登録の型や不整合な参照が含まれる場合。
    """
    return _decode_value(tree, buffers)


def _encode_value(value: Any, buffers: list[np.ndarray]) -> Any:
    # Enum の判定は primitive より**先**に行う。TaskType 等は str を継承しているため、
    # 順序を逆にすると enum が素の文字列として直列化され、復元時に型が失われる。
    if isinstance(value, Enum):
        name = type(value).__name__
        if name not in TYPE_REGISTRY:
            raise SerializationError(f"enum type {name!r} is not registered in TYPE_REGISTRY")
        return {_ENUM_KEY: name, "value": value.value}

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, np.ndarray):
        buffers.append(np.ascontiguousarray(value))
        return {_ARRAY_KEY: len(buffers) - 1}

    if isinstance(value, np.generic):
        # numpy スカラは Python スカラに落とす（JSON 化のため）
        return value.item()

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        name = type(value).__name__
        if name not in TYPE_REGISTRY:
            raise SerializationError(f"dataclass {name!r} is not registered in TYPE_REGISTRY")
        encoded: dict[str, Any] = {_TYPE_KEY: name}
        for field in dataclasses.fields(value):
            field_value = getattr(value, field.name)
            # None は省略する。デコード時に dataclass の既定値が入る
            if field_value is None:
                continue
            encoded[field.name] = _encode_value(field_value, buffers)
        return encoded

    if isinstance(value, tuple):
        return {_TUPLE_KEY: [_encode_value(item, buffers) for item in value]}

    if isinstance(value, list):
        return [_encode_value(item, buffers) for item in value]

    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise SerializationError(f"dict keys must be str, got {type(key).__name__}")
        return {key: _encode_value(item, buffers) for key, item in value.items()}

    raise SerializationError(f"unsupported type for serialization: {type(value).__name__}")


def _decode_value(value: Any, buffers: list[np.ndarray]) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, list):
        return [_decode_value(item, buffers) for item in value]

    if not isinstance(value, dict):
        raise SerializationError(f"unexpected node type in payload: {type(value).__name__}")

    if _ARRAY_KEY in value:
        index = value[_ARRAY_KEY]
        if not isinstance(index, int) or not 0 <= index < len(buffers):
            raise SerializationError(f"array reference out of range: {index!r}")
        return buffers[index]

    if _TUPLE_KEY in value:
        return tuple(_decode_value(item, buffers) for item in value[_TUPLE_KEY])

    if _ENUM_KEY in value:
        enum_type = _lookup_type(value[_ENUM_KEY])
        try:
            return enum_type(value["value"])
        except (KeyError, ValueError) as error:
            raise SerializationError(
                f"invalid value for enum {value[_ENUM_KEY]!r}: {value.get('value')!r}"
            ) from error

    if _TYPE_KEY in value:
        cls = _lookup_type(value[_TYPE_KEY])
        field_names = {field.name for field in dataclasses.fields(cls)}
        kwargs = {}
        for key, item in value.items():
            if key == _TYPE_KEY:
                continue
            if key not in field_names:
                raise SerializationError(f"unknown field {key!r} for {cls.__name__}")
            kwargs[key] = _decode_value(item, buffers)
        try:
            # dataclass の __post_init__ による検証がここで走る
            return cls(**kwargs)
        except (TypeError, ValueError) as error:
            raise SerializationError(f"failed to construct {cls.__name__}: {error}") from error

    return {key: _decode_value(item, buffers) for key, item in value.items()}


def _lookup_type(name: Any) -> type:
    if not isinstance(name, str) or name not in TYPE_REGISTRY:
        raise SerializationError(f"type {name!r} is not allowed (not in TYPE_REGISTRY)")
    return TYPE_REGISTRY[name]
