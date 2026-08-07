"""幾何ヘルパ（純 numpy）の検証。"""

from __future__ import annotations

import numpy as np
import pytest

from jidohub.core.geometry import (
    boxes_from_source,
    boxes_to_source,
    crop_intrinsic,
    denormalize_boxes,
    invert_transform,
    normalize_boxes,
    points_from_source,
    points_to_source,
    quaternion_to_rotation_matrix,
    quaternion_to_yaw,
    rotate_vectors,
    rotation_matrix_to_quaternion,
    scale_intrinsic,
    scaled_source,
    transform_points,
    transform_quaternion,
    yaw_to_quaternion,
)
from jidohub.core.schemas import ImageSource

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


# --- 3D ベクトル・クォータニオン変換 ----------------------------------------


def test_rotate_vectors_applies_rotation_only() -> None:
    # z 軸まわり 90 度回転で (1, 0, 0) -> (0, 1, 0)。平行移動は含まれない。
    transform = make_transform(translation=(100.0, 200.0, 300.0))
    transform[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    result = rotate_vectors(np.array([1.0, 0.0, 0.0]), transform)
    # 4x4 を渡しても回転部のみが使われる（平行移動 100,200,300 は無視）。
    assert np.allclose(result, [0.0, 1.0, 0.0])


def test_rotate_vectors_batched_and_non_destructive() -> None:
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    original = vectors.copy()
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    result = rotate_vectors(vectors, rotation)
    assert np.allclose(result, [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    assert np.array_equal(vectors, original)


@pytest.mark.parametrize("yaw", [0.0, 0.7, -2.0, np.pi / 3, np.pi - 1e-6])
def test_rotation_matrix_quaternion_roundtrip(yaw: float) -> None:
    quaternion = yaw_to_quaternion(yaw)
    matrix = quaternion_to_rotation_matrix(quaternion)
    recovered = rotation_matrix_to_quaternion(matrix)
    # 符号の不定性を許容して回転行列で比較する。
    assert np.allclose(quaternion_to_rotation_matrix(recovered), matrix, atol=1e-9)


def test_rotation_matrix_to_quaternion_180deg_is_stable() -> None:
    # w が 0 に近い 180 度回転（素朴実装が精度を失う領域）。
    matrix = quaternion_to_rotation_matrix(yaw_to_quaternion(np.pi))
    recovered = rotation_matrix_to_quaternion(matrix)
    assert np.isfinite(recovered).all()
    assert np.allclose(quaternion_to_rotation_matrix(recovered), matrix, atol=1e-9)


def test_transform_quaternion_composes_known_rotation() -> None:
    # 45 度の姿勢に 45 度の変換を合成すると 90 度になる。
    q45 = yaw_to_quaternion(np.pi / 4)
    transform = np.eye(4)
    transform[:3, :3] = quaternion_to_rotation_matrix(yaw_to_quaternion(np.pi / 4))
    result = transform_quaternion(q45, transform)
    assert quaternion_to_yaw(result) == pytest.approx(np.pi / 2, abs=1e-9)


# --- 2D 画素座標の変換（手計算で検証）---------------------------------------


def test_denormalize_normalize_roundtrip() -> None:
    xyxy = np.array([[0.1, 0.2, 0.5, 0.8], [0.0, 0.0, 1.0, 1.0]])
    pixels = denormalize_boxes(xyxy, width=1600, height=900)
    assert np.allclose(pixels[0], [160.0, 180.0, 800.0, 720.0])
    assert np.allclose(normalize_boxes(pixels, 1600, 900), xyxy)


def test_denormalize_boxes_accepts_1d_and_2d() -> None:
    one = denormalize_boxes(np.array([0.5, 0.5, 1.0, 1.0]), 100, 200)
    assert one.shape == (4,)
    assert np.allclose(one, [50.0, 100.0, 100.0, 200.0])
    many = denormalize_boxes(np.zeros((3, 4)), 100, 200)
    assert many.shape == (3, 4)


def test_boxes_to_from_source_roundtrip_crop_and_scale() -> None:
    # 元画像を crop (400,200) して scale (0.8,1.28) した現画像。
    source = ImageSource(crop=(400, 200, 1200, 700), scale=(0.8, 1.28))
    current = np.array([[320.0, 320.0, 336.0, 384.0], [0.0, 0.0, 8.0, 12.8]])
    original = boxes_to_source(current, source)
    # (320/0.8 + 400, 320/1.28 + 200) = (800, 450)
    assert np.allclose(original[0, :2], [800.0, 450.0])
    assert np.allclose(boxes_from_source(original, source), current)


def test_boxes_to_source_none_is_identity() -> None:
    xyxy = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(boxes_to_source(xyxy, None), xyxy)


def test_points_to_from_source_roundtrip() -> None:
    source = ImageSource(crop=(400, 200, 1200, 700), scale=(0.8, 1.28))
    current = np.array([[320.0, 320.0], [0.0, 0.0]])
    original = points_to_source(current, source)
    assert np.allclose(original[0], [800.0, 450.0])
    assert np.allclose(points_from_source(original, source), current)


def test_scaled_source_composes_scale_and_keeps_crop() -> None:
    base = ImageSource(channel="CAM_FRONT", crop=(400, 200, 1200, 700), scale=(0.8, 1.28))
    result = scaled_source(base, 0.5, 0.5)
    assert result.crop == (400, 200, 1200, 700)
    assert result.channel == "CAM_FRONT"
    assert result.scale == pytest.approx((0.4, 0.64))
    # 入力は破壊しない。
    assert base.scale == (0.8, 1.28)


def test_scaled_source_from_none() -> None:
    result = scaled_source(None, 0.5, 2.0)
    assert result.crop is None
    assert result.scale == (0.5, 2.0)
