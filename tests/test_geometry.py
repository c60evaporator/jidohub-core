"""幾何ヘルパ（純 numpy）の検証。"""

from __future__ import annotations

import numpy as np
import pytest

from jidohub.core.geometry import (
    crop_intrinsic,
    invert_transform,
    quaternion_to_yaw,
    scale_intrinsic,
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


# --- 内部パラメータの crop / scale 追従（手計算で検証）----------------------


def _intrinsic(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def test_crop_intrinsic_shifts_principal_point() -> None:
    k = _intrinsic(1000.0, 1000.0, 800.0, 450.0)
    cropped = crop_intrinsic(k, x0=100, y0=50)
    # 主点だけが (x0, y0) 平行移動し、焦点距離は不変。
    assert cropped[0, 2] == 700.0
    assert cropped[1, 2] == 400.0
    assert cropped[0, 0] == 1000.0
    assert cropped[1, 1] == 1000.0


def test_crop_intrinsic_does_not_mutate_input() -> None:
    k = _intrinsic(1000.0, 1000.0, 800.0, 450.0)
    original = k.copy()
    crop_intrinsic(k, x0=100, y0=50)
    assert np.array_equal(k, original)


def test_scale_intrinsic_scales_focal_and_principal() -> None:
    k = _intrinsic(1000.0, 800.0, 640.0, 360.0)
    scaled = scale_intrinsic(k, scale_x=0.5, scale_y=2.0)
    assert scaled[0, 0] == 500.0  # fx * 0.5
    assert scaled[1, 1] == 1600.0  # fy * 2.0
    assert scaled[0, 2] == 320.0  # cx * 0.5
    assert scaled[1, 2] == 720.0  # cy * 2.0


def test_scale_intrinsic_does_not_mutate_input() -> None:
    k = _intrinsic(1000.0, 800.0, 640.0, 360.0)
    original = k.copy()
    scale_intrinsic(k, scale_x=0.5, scale_y=2.0)
    assert np.array_equal(k, original)


def test_crop_then_scale_differs_from_scale_then_crop() -> None:
    # 順序が意味を持つ（主点の平行移動とスケールは非可換）ことを確認する。
    k = _intrinsic(1000.0, 1000.0, 800.0, 450.0)
    crop_then_scale = scale_intrinsic(crop_intrinsic(k, 100, 50), 0.5, 0.5)
    scale_then_crop = crop_intrinsic(scale_intrinsic(k, 0.5, 0.5), 100, 50)
    assert not np.array_equal(crop_then_scale, scale_then_crop)
