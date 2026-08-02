"""直列化（JSON ヘッダ + 生バッファの自己記述コンテナ）の round-trip と異常系。"""

from __future__ import annotations

import dataclasses
import enum
import struct

import numpy as np
import pytest

from jidohub.core import schemas
from jidohub.core.schemas import (
    DrivingCommand,
    E2EOutput,
    Sample,
)
from jidohub.core.serialization import (
    MAGIC,
    TYPE_REGISTRY,
    SerializationError,
    decode,
    pack,
    unpack,
)

from .conftest import make_e2e_output, make_sample

# --- round-trip -------------------------------------------------------------


def test_sample_roundtrip_preserves_structure() -> None:
    sample = make_sample()
    restored = unpack(pack(sample))

    assert isinstance(restored, Sample)
    # int timestamp が保持される。
    assert isinstance(restored.timestamp, int)
    assert restored.timestamp == sample.timestamp
    # dtype が保持される。
    assert restored.lidar is not None
    assert restored.lidar.points.dtype == np.float32
    assert np.array_equal(restored.lidar.points, sample.lidar.points)
    # 変換行列も一致。
    assert np.allclose(restored.ego_to_global, sample.ego_to_global)
    # ネストした history / cameras / radars。
    assert set(restored.cameras) == {"CAM_FRONT", "CAM_BACK"}
    assert len(restored.history) == 1
    assert isinstance(restored.history[0], Sample)


def test_enum_type_is_preserved_not_just_value() -> None:
    sample = make_sample()
    restored = unpack(pack(sample))
    # str を継承しているため == "turn-left" では型崩壊を検出できない。
    assert restored.command is DrivingCommand.TURN_LEFT
    assert isinstance(restored.command, DrivingCommand)


def test_tuple_vs_list_distinction_preserved() -> None:
    sample = make_sample()
    restored = unpack(pack(sample))
    assert restored.lidar is not None
    # LidarSweep.fields は tuple。
    assert isinstance(restored.lidar.fields, tuple)
    assert restored.lidar.fields == ("x", "y", "z", "intensity")
    # history は list。
    assert isinstance(restored.history, list)


def test_none_fields_restored_to_default() -> None:
    sample = make_sample()
    assert sample.ego_state is not None
    assert sample.ego_state.acceleration is None  # 前提
    restored = unpack(pack(sample))
    assert restored.ego_state is not None
    assert restored.ego_state.acceleration is None
    assert restored.ego_state.angular_velocity is None


def test_e2e_output_roundtrip_all_intermediates() -> None:
    output = make_e2e_output()
    restored = unpack(pack(output))
    assert isinstance(restored, E2EOutput)
    assert restored.detection is not None and len(restored.detection.boxes) == 1
    assert restored.tracking is not None
    assert restored.map is not None
    assert restored.motion_forecast is not None
    assert restored.occupancy is not None
    assert restored.bev_feature is not None
    assert np.array_equal(restored.occupancy, output.occupancy)
    # 中間出力内の enum / dtype も保持。
    assert restored.map.elements[0].element_type is output.map.elements[0].element_type
    assert restored.motion_forecast.forecasts[0].trajectories.dtype == np.float32


# --- 2D タスクの round-trip -------------------------------------------------


def _image_2d() -> "schemas.Image":
    return schemas.Image(pixels=np.zeros((4, 6, 3), dtype=np.uint8), intrinsic=np.eye(3))


def test_image_sample_roundtrip_with_prompt() -> None:
    sample = schemas.ImageSample(
        image=_image_2d(),
        prompt=schemas.ImagePrompt(
            points=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            point_labels=np.array([1, 0], dtype=np.int64),
            boxes=np.array([[0.0, 0.0, 2.0, 2.0]], dtype=np.float64),
            texts=["car", "person"],
        ),
        sample_id="img-0",
    )
    restored = unpack(pack(sample))
    assert isinstance(restored, schemas.ImageSample)
    assert restored.sample_id == "img-0"
    # texts は list[str] のまま（tuple 化しない）。
    assert restored.prompt.texts == ["car", "person"]
    assert isinstance(restored.prompt.texts, list)
    assert np.array_equal(restored.prompt.point_labels, sample.prompt.point_labels)


def test_image_sample_roundtrip_without_prompt() -> None:
    restored = unpack(pack(schemas.ImageSample(image=_image_2d())))
    assert isinstance(restored, schemas.ImageSample)
    assert restored.prompt is None


def test_detection_2d_roundtrip() -> None:
    output = schemas.Detection2DOutput(
        boxes=[
            schemas.Box2D(xyxy=np.array([1.0, 2.0, 5.0, 6.0]), label="car", score=0.9, track_id=3),
            schemas.Box2D(xyxy=np.array([0.0, 0.0, 1.0, 1.0])),  # プロンプタブル: label None
        ]
    )
    restored = unpack(pack(output))
    assert isinstance(restored, schemas.Detection2DOutput)
    assert restored.boxes[0].label == "car" and restored.boxes[0].track_id == 3
    assert restored.boxes[1].label is None
    assert np.array_equal(restored.boxes[0].xyxy, output.boxes[0].xyxy)


def test_instance_segmentation_2d_roundtrip() -> None:
    mask = np.zeros((4, 4), dtype=np.bool_)
    mask[1:3, 0:2] = True
    output = schemas.InstanceSegmentation2DOutput(
        instances=[
            schemas.Instance2D(
                box=schemas.Box2D(xyxy=np.array([2.0, 1.0, 6.0, 5.0]), label="person"),
                mask=mask,
                mask_region=(2, 1, 6, 5),
            )
        ]
    )
    restored = unpack(pack(output))
    instance = restored.instances[0]
    # bool dtype 維持・mask_region の tuple が list に落ちない。
    assert instance.mask.dtype == np.bool_
    assert np.array_equal(instance.mask, mask)
    assert isinstance(instance.mask_region, tuple)
    assert instance.mask_region == (2, 1, 6, 5)


def test_classification_2d_roundtrip() -> None:
    output = schemas.Classification2DOutput(
        labels=["car", "truck"], scores=np.array([0.8, 0.2], dtype=np.float64)
    )
    restored = unpack(pack(output))
    assert restored.labels == ["car", "truck"]
    assert np.allclose(restored.scores, output.scores)


# --- copy / read-only -------------------------------------------------------


def test_unpack_default_is_readonly_view() -> None:
    sample = make_sample()
    restored = unpack(pack(sample))
    assert restored.lidar is not None
    assert restored.lidar.points.flags.writeable is False
    with pytest.raises(ValueError):
        restored.lidar.points[0, 0] = 1.0


def test_unpack_copy_is_writable() -> None:
    sample = make_sample()
    restored = unpack(pack(sample), copy=True)
    assert restored.lidar is not None
    assert restored.lidar.points.flags.writeable is True
    restored.lidar.points[0, 0] = 42.0  # 例外にならない


# --- 異常系（envelope レベル）----------------------------------------------


def test_bad_magic() -> None:
    packed = pack(make_sample())
    corrupt = b"NOTMAGIC" + packed[len(MAGIC) :]
    with pytest.raises(SerializationError, match="magic"):
        unpack(corrupt)


def test_too_short() -> None:
    with pytest.raises(SerializationError, match="too short"):
        unpack(b"JIDO")


def test_oversized_header_length() -> None:
    packed = bytearray(pack(make_sample()))
    # header_len フィールド（MAGIC の直後 8 バイト, uint64 LE）を過大宣言する。
    struct.pack_into("<Q", packed, len(MAGIC), 10**9)
    with pytest.raises(SerializationError, match="header length exceeds"):
        unpack(bytes(packed))


@pytest.mark.parametrize("version", ["0.1", "1.0"])
def test_incompatible_schema_version(version: str) -> None:
    # 単純なオブジェクトを pack し、ヘッダの現行バージョンだけを差し替える（長さ不変）。
    packed = pack({"a": 1})
    corrupt = packed.replace(b'"0.2"', f'"{version}"'.encode(), 1)
    assert corrupt != packed
    with pytest.raises(SerializationError):
        unpack(corrupt)


# --- 異常系（codec レベル、decode を直接叩く）------------------------------


def test_unknown_type_name_rejected() -> None:
    with pytest.raises(SerializationError, match="not in TYPE_REGISTRY|not allowed"):
        decode({"__type__": "TotallyUnknown"}, [])


def test_array_reference_out_of_range() -> None:
    with pytest.raises(SerializationError, match="out of range"):
        decode({"__array__": 5}, [])


def test_unknown_dataclass_field_rejected() -> None:
    with pytest.raises(SerializationError, match="unknown field"):
        decode({"__type__": "Box3D", "not_a_field": 1}, [])


# --- TYPE_REGISTRY 網羅性 ---------------------------------------------------


def test_type_registry_covers_all_public_schema_types() -> None:
    missing = []
    for name in schemas.__all__:
        obj = getattr(schemas, name)
        if isinstance(obj, type) and (dataclasses.is_dataclass(obj) or issubclass(obj, enum.Enum)):
            if name not in TYPE_REGISTRY:
                missing.append(name)
    assert not missing, f"types missing from TYPE_REGISTRY: {missing}"
