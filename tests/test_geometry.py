"""幾何ヘルパ（純 numpy）の検証。"""

from __future__ import annotations

import numpy as np
import pytest

from jidohub.core.geometry import (
    invert_transform,
    quaternion_to_yaw,
    transform_points,
    yaw_to_quaternion,
)

from .conftest import make_transform


@pytest.mark.parametrize("yaw", [0.0, 0.7, -2.0, np.pi / 3, np.pi - 1e-6])
def test_yaw_quaternion_roundtrip(yaw: float) -> None:
    quaternion = yaw_to_quaternion(yaw)
    assert quaternion.shape == (4,)
    assert quaternion_to_yaw(quaternion) == pytest.approx(yaw, abs=1e-9)


def test_invert_transform_is_true_inverse() -> None:
    transform = make_transform()
    inverse = invert_transform(transform)
    assert np.allclose(transform @ inverse, np.eye(4))
    assert np.allclose(inverse @ transform, np.eye(4))
    assert inverse.dtype == np.float64


def test_transform_points_only_touches_xyz() -> None:
    points = np.arange(20, dtype=np.float32).reshape(5, 4)
    original = points.copy()
    transform = make_transform()

    result = transform_points(points, transform)

    # 入力は不変。
    assert np.array_equal(points, original)
    # dtype 保持。
    assert result.dtype == np.float32
    # 4 列目以降は保持。
    assert np.array_equal(result[:, 3:], original[:, 3:])
    # 先頭 3 列は期待どおり変換されている（float64 で参照計算）。
    expected = original[:, :3].astype(np.float64) @ transform[:3, :3].T + transform[:3, 3]
    assert np.allclose(result[:, :3], expected, atol=1e-4)


def test_transform_points_handles_non_contiguous_input() -> None:
    base = np.arange(40, dtype=np.float64).reshape(10, 4)
    sliced = base[::2]  # 非連続ビュー
    assert not sliced.flags["C_CONTIGUOUS"]

    transform = make_transform()
    result = transform_points(sliced, transform)

    expected = sliced[:, :3] @ transform[:3, :3].T + transform[:3, 3]
    assert np.allclose(result[:, :3], expected)
    assert np.array_equal(result[:, 3:], sliced[:, 3:])


def test_transform_points_rejects_too_few_columns() -> None:
    with pytest.raises(ValueError, match=r"points must have shape"):
        transform_points(np.zeros((4, 2)), make_transform())
