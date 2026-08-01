"""標準スキーマ（dataclass）の構築と __post_init__ 検証。"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from jidohub.core.geometry import yaw_to_quaternion
from jidohub.core.schemas import (
    Box3D,
    CameraFrame,
    Image,
    LidarSweep,
    RadarSweep,
    Sample,
)
from jidohub.core.serialization import pack, unpack

from .conftest import make_box, make_lidar_sweep, make_sample, make_transform


def test_sample_builds() -> None:
    sample = make_sample()
    assert sample.timestamp == 1_600_000_000_000_000
    assert sample.ego_to_global.shape == (4, 4)
    assert set(sample.cameras) == {"CAM_FRONT", "CAM_BACK"}
    assert len(sample.history) == 1


def test_sample_ego_to_global_wrong_shape() -> None:
    with pytest.raises(ValueError, match="Sample.ego_to_global must have shape"):
        Sample(timestamp=1, ego_to_global=np.eye(3, dtype=np.float64))


def test_sample_timestamp_must_be_int() -> None:
    with pytest.raises(TypeError, match="Sample.timestamp must be an int"):
        Sample(timestamp=1.5, ego_to_global=make_transform())  # type: ignore[arg-type]


def test_sample_timestamp_accepts_numpy_integer() -> None:
    sample = Sample(timestamp=np.int64(42), ego_to_global=make_transform())  # type: ignore[arg-type]
    assert int(sample.timestamp) == 42


def test_camera_frame_requires_image_intrinsic() -> None:
    # intrinsic の唯一の正は Image。カメラフレームは intrinsic 未知の Image を拒否する。
    with pytest.raises(ValueError, match="CameraFrame.image.intrinsic must not be None"):
        CameraFrame(
            image=Image(pixels=np.zeros((4, 6, 3), dtype=np.uint8)),  # intrinsic なし
            sensor_to_ego=make_transform(),
            channel="CAM_FRONT",
        )


def test_camera_frame_sensor_to_ego_wrong_shape() -> None:
    with pytest.raises(ValueError, match="CameraFrame.sensor_to_ego must have shape"):
        CameraFrame(
            image=Image(
                pixels=np.zeros((4, 6, 3), dtype=np.uint8),
                intrinsic=np.eye(3, dtype=np.float64),
            ),
            sensor_to_ego=np.eye(3, dtype=np.float64),
            channel="CAM_FRONT",
        )


def test_lidar_points_need_three_columns() -> None:
    with pytest.raises(ValueError, match=r"LidarSweep.points must have shape"):
        LidarSweep(
            points=np.zeros((5, 2), dtype=np.float32),
            sensor_to_ego=make_transform(),
        )


def test_lidar_sensor_to_ego_wrong_shape() -> None:
    with pytest.raises(ValueError, match="LidarSweep.sensor_to_ego must have shape"):
        LidarSweep(
            points=np.zeros((5, 4), dtype=np.float32),
            sensor_to_ego=np.eye(3, dtype=np.float64),
        )


def test_radar_points_need_three_columns() -> None:
    with pytest.raises(ValueError, match=r"RadarSweep.points must have shape"):
        RadarSweep(
            points=np.zeros((5, 2), dtype=np.float32),
            sensor_to_ego=make_transform(),
            channel="RADAR_FRONT",
        )


def test_points_in_ego_does_not_mutate_and_preserves_extra_columns() -> None:
    sweep = make_lidar_sweep()
    original = sweep.points.copy()
    ego_points = sweep.points_in_ego()

    # 元配列は不変。
    assert np.array_equal(sweep.points, original)
    # 出力は形状・dtype を保持。
    assert ego_points.shape == sweep.points.shape
    assert ego_points.dtype == sweep.points.dtype
    # 4 列目以降（intensity）は変換されず保持される。
    assert np.array_equal(ego_points[:, 3:], original[:, 3:])
    # 先頭 3 列は sensor_to_ego で変換される（恒等でないので値が変わる）。
    assert not np.array_equal(ego_points[:, :3], original[:, :3])


@pytest.mark.parametrize("yaw", [0.0, 0.5, -1.2, 3.0, np.pi / 2])
def test_box_yaw_roundtrip(yaw: float) -> None:
    box = make_box()
    box.rotation = yaw_to_quaternion(yaw)
    assert box.yaw == pytest.approx(yaw, abs=1e-9)


# --- Box3D の寸法アクセサ ----------------------------------------------------


def test_box_dimension_properties_match_size() -> None:
    box = make_box()
    assert box.length == box.size[0]
    assert box.width == box.size[1]
    assert box.height == box.size[2]
    # プロパティは float を返す（np スカラのままにしない）。
    assert isinstance(box.length, float)
    assert isinstance(box.width, float)
    assert isinstance(box.height, float)


def test_from_dimensions_stores_size_in_lwh_order() -> None:
    box = Box3D.from_dimensions(
        center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        length=4.5,
        width=1.9,
        height=1.6,
        rotation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        label="car",
    )
    assert box.size.dtype == np.float64
    np.testing.assert_array_equal(box.size, np.array([4.5, 1.9, 1.6]))
    assert box.length == 4.5
    assert box.width == 1.9
    assert box.height == 1.6


def test_from_dimensions_passes_through_optional_kwargs() -> None:
    velocity = np.array([2.0, 0.0, 0.0], dtype=np.float64)
    box = Box3D.from_dimensions(
        center=np.array([1.0, 2.0, 0.5], dtype=np.float64),
        length=4.5,
        width=1.9,
        height=1.6,
        rotation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        label="car",
        score=0.8,
        track_id=42,
        velocity=velocity,
        attributes={"moving": True},
    )
    assert box.score == 0.8
    assert box.track_id == 42
    np.testing.assert_array_equal(box.velocity, velocity)
    assert box.attributes == {"moving": True}


def test_dimension_properties_are_not_dataclass_fields() -> None:
    # フィールド化されると直列化の二重の正になるため、機械的に防ぐ。
    field_names = {f.name for f in dataclasses.fields(Box3D)}
    assert {"length", "width", "height"} & field_names == set()


def test_from_dimensions_roundtrips_through_pack_unpack() -> None:
    box = Box3D.from_dimensions(
        center=np.array([1.0, 2.0, 0.5], dtype=np.float64),
        length=4.5,
        width=1.9,
        height=1.6,
        rotation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        label="car",
        score=0.8,
    )
    restored = unpack(pack(box))
    np.testing.assert_array_equal(restored.size, box.size)
    assert restored.label == box.label
    assert restored.score == box.score


def test_realistic_dimensions_avoid_lw_swap() -> None:
    # 乗用車相当。順序を取り違えた実装ではこの assert が失敗する。
    box = Box3D.from_dimensions(
        center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        length=4.5,
        width=1.9,
        height=1.6,
        rotation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        label="car",
    )
    assert box.length > box.width
