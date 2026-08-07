"""出力型の座標変換メソッド（層 3）の検証。

取り違えても例外が出ない処理のため、**手計算で検証できる値**で機械的に固定する。
基準の状況: ego が global の ``(10, 20)`` にあり 90 度左（+x → +y）を向いている。
このとき ego 座標 ``(5, 0, 0)`` は global 座標 ``(10, 25, 0)`` に写る。
"""

from __future__ import annotations

import numpy as np
import pytest

from jidohub.core.schemas import (
    AgentForecast,
    Box2D,
    Box3D,
    CoordinateFrame,
    Detection2DOutput,
    Detection3DOutput,
    Image,
    ImageSource,
    Instance2D,
    InstanceSegmentation2DOutput,
    MapElement,
    MapElementType,
    MapOutput,
    MotionForecastOutput,
    PlanningOutput,
)


def _ego_to_global() -> np.ndarray:
    """ego が global の (10, 20) で 90 度左向き。"""
    theta = np.pi / 2
    cos, sin = np.cos(theta), np.sin(theta)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    transform[:3, 3] = [10.0, 20.0, 0.0]
    return transform


def _box(velocity: np.ndarray | None = None) -> Box3D:
    return Box3D(
        center=np.array([5.0, 0.0, 0.0], dtype=np.float64),
        size=np.array([4.0, 2.0, 1.5], dtype=np.float64),
        rotation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        label="car",
        score=0.9,
        velocity=velocity,
        track_id=3,
    )


# --- 3D: Detection3DOutput --------------------------------------------------


def test_detection3d_to_global_hand_computed() -> None:
    det = Detection3DOutput(boxes=[_box()], frame=CoordinateFrame.EGO)
    out = det.to_global(_ego_to_global())
    assert out.frame == CoordinateFrame.GLOBAL
    assert np.allclose(out.boxes[0].center, [10.0, 25.0, 0.0])
    # size / label / score / track_id は不変。
    assert np.allclose(out.boxes[0].size, [4.0, 2.0, 1.5])
    assert out.boxes[0].track_id == 3


def test_detection3d_roundtrip_is_identity() -> None:
    det = Detection3DOutput(boxes=[_box(velocity=np.array([2.0, 1.0, 0.0]))])
    back = det.to_global(_ego_to_global()).to_ego(_ego_to_global())
    assert np.allclose(back.boxes[0].center, det.boxes[0].center)
    assert np.allclose(back.boxes[0].velocity, det.boxes[0].velocity)
    assert np.allclose(back.boxes[0].rotation, det.boxes[0].rotation)


def test_detection3d_idempotent_returns_self() -> None:
    det = Detection3DOutput(boxes=[_box()], frame=CoordinateFrame.EGO)
    once = det.to_global(_ego_to_global())
    twice = once.to_global(_ego_to_global())
    assert twice is once  # 既に GLOBAL なら self
    assert det.to_ego(_ego_to_global()) is det  # 既に EGO なら self


def test_detection3d_velocity_has_no_translation() -> None:
    # 静止物体は変換後も静止（平行移動が velocity に効かないことの検出）。
    det = Detection3DOutput(boxes=[_box(velocity=np.array([0.0, 0.0, 0.0]))])
    out = det.to_global(_ego_to_global())
    assert np.allclose(out.boxes[0].velocity, [0.0, 0.0, 0.0])


def test_detection3d_velocity_rotates_only() -> None:
    # (1, 0) の速度は 90 度回転で (0, 1) になる（大きさ保存・原点シフトなし）。2 次元速度。
    det = Detection3DOutput(boxes=[_box(velocity=np.array([1.0, 0.0]))])
    out = det.to_global(_ego_to_global())
    assert np.allclose(out.boxes[0].velocity, [0.0, 1.0], atol=1e-12)


def test_detection3d_rotation_transformed() -> None:
    # 前方 +x を向いた姿勢（yaw=0）は 90 度左回転後に yaw=pi/2 になる。
    det = Detection3DOutput(boxes=[_box()])
    out = det.to_global(_ego_to_global())
    assert out.boxes[0].yaw == pytest.approx(np.pi / 2, abs=1e-9)


def test_detection3d_input_not_mutated() -> None:
    det = Detection3DOutput(boxes=[_box(velocity=np.array([2.0, 0.0, 0.0]))])
    center_before = det.boxes[0].center.copy()
    det.to_global(_ego_to_global())
    assert np.array_equal(det.boxes[0].center, center_before)
    assert det.frame == CoordinateFrame.EGO


def test_camera_frame_raises_not_implemented() -> None:
    det = Detection3DOutput(boxes=[_box()], frame=CoordinateFrame.CAMERA)
    with pytest.raises(NotImplementedError, match="CAMERA"):
        det.to_ego(_ego_to_global())
    with pytest.raises(NotImplementedError, match="CAMERA"):
        det.to_global(_ego_to_global())


# --- 3D: MapOutput / MotionForecastOutput / PlanningOutput ------------------


def test_map_output_transforms_points() -> None:
    element = MapElement(
        points=np.array([[5.0, 0.0], [0.0, 0.0]], dtype=np.float64),
        element_type=MapElementType.DIVIDER,
    )
    out = MapOutput(elements=[element]).to_global(_ego_to_global())
    assert out.frame == CoordinateFrame.GLOBAL
    # (5,0) -> (10,25), (0,0) -> (10,20)
    assert np.allclose(out.elements[0].points, [[10.0, 25.0], [10.0, 20.0]])
    # 往復恒等・入力不変。
    back = out.to_ego(_ego_to_global())
    assert np.allclose(back.elements[0].points, element.points)


def test_motion_forecast_frame_on_container_and_transforms() -> None:
    forecast = AgentForecast(
        trajectories=np.array([[[5.0, 0.0]]], dtype=np.float64),  # (M=1, T=1, 2)
        probabilities=np.array([1.0]),
        dt=0.5,
        track_id=3,
    )
    motion = MotionForecastOutput(forecasts=[forecast])
    assert motion.frame == CoordinateFrame.EGO  # frame はコンテナが持つ
    out = motion.to_global(_ego_to_global())
    assert out.frame == CoordinateFrame.GLOBAL
    assert np.allclose(out.forecasts[0].trajectories[0, 0], [10.0, 25.0])
    assert out.forecasts[0].track_id == 3


def test_agent_forecast_has_no_frame_field() -> None:
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(AgentForecast)}
    assert "frame" not in field_names


def test_planning_output_transforms_trajectory() -> None:
    planning = PlanningOutput(
        trajectory=np.array([[5.0, 0.0]], dtype=np.float64), dt=0.5, confidence=0.8
    )
    out = planning.to_global(_ego_to_global())
    assert out.frame == CoordinateFrame.GLOBAL
    assert np.allclose(out.trajectory[0], [10.0, 25.0])
    assert out.confidence == 0.8
    assert out.to_ego(_ego_to_global()).to_global(_ego_to_global()).frame == CoordinateFrame.GLOBAL


# --- 2D: to_source_image ----------------------------------------------------


def _resized_image() -> Image:
    """元 1600x900 → crop (400,200,1200,700)=800x500 → scale (0.8,1.28)=640x640。"""
    source = ImageSource(crop=(400, 200, 1200, 700), scale=(0.8, 1.28))
    return Image(pixels=np.zeros((640, 640, 3), dtype=np.uint8), source=source)


def test_detection2d_source_none_only_denormalizes() -> None:
    image = Image(pixels=np.zeros((900, 1600, 3), dtype=np.uint8))
    det = Detection2DOutput(boxes=[Box2D(xyxy=np.array([0.5, 0.5, 1.0, 1.0]))], normalized=True)
    out = det.to_source_image(image)
    assert out.normalized is False
    assert np.allclose(out.boxes[0].xyxy, [800.0, 450.0, 1600.0, 900.0])


def test_detection2d_crop_only_hand_computed() -> None:
    # crop のみ（scale なし）: 現画像座標 (10,20) -> 元画像 (410,220)。
    crop = Image(pixels=np.zeros((900, 1600, 3), dtype=np.uint8)).cropped(400, 200, 1200, 700)
    det = Detection2DOutput(boxes=[Box2D(xyxy=np.array([10.0, 20.0, 30.0, 40.0]))])
    out = det.to_source_image(crop)
    assert np.allclose(out.boxes[0].xyxy, [410.0, 220.0, 430.0, 240.0])


def test_detection2d_crop_and_scale_normalized() -> None:
    det = Detection2DOutput(boxes=[Box2D(xyxy=np.array([0.5, 0.5, 0.5, 0.5]))], normalized=True)
    out = det.to_source_image(_resized_image())
    # (0.5*640, 0.5*640)=(320,320) -> /(0.8,1.28)=(400,250) -> +(400,200)=(800,450)
    assert np.allclose(out.boxes[0].xyxy, [800.0, 450.0, 800.0, 450.0])
    assert out.normalized is False


def test_detection2d_crop_of_crop() -> None:
    # crop の crop（source が 1 組に合成されるケース）。
    base = Image(pixels=np.zeros((900, 1600, 3), dtype=np.uint8))
    inner = base.cropped(400, 200, 1200, 700).cropped(50, 30, 200, 150)
    # 合成後の元画像 crop 原点 = (450, 230)。
    assert inner.source is not None and inner.source.crop == (450, 230, 600, 350)
    det = Detection2DOutput(boxes=[Box2D(xyxy=np.array([10.0, 20.0, 30.0, 40.0]))])
    out = det.to_source_image(inner)
    assert np.allclose(out.boxes[0].xyxy, [460.0, 250.0, 480.0, 270.0])


def test_detection2d_idempotent_no_op_returns_self() -> None:
    image = Image(pixels=np.zeros((10, 10, 3), dtype=np.uint8))
    det = Detection2DOutput(boxes=[Box2D(xyxy=np.array([1.0, 2.0, 3.0, 4.0]))])
    assert det.to_source_image(image) is det


def test_detection2d_input_not_mutated() -> None:
    crop = Image(pixels=np.zeros((900, 1600, 3), dtype=np.uint8)).cropped(400, 200, 1200, 700)
    box = Box2D(xyxy=np.array([10.0, 20.0, 30.0, 40.0]))
    det = Detection2DOutput(boxes=[box])
    det.to_source_image(crop)
    assert np.array_equal(box.xyxy, [10.0, 20.0, 30.0, 40.0])


def test_instance_seg_moves_mask_region_no_scale() -> None:
    crop = Image(pixels=np.zeros((900, 1600, 3), dtype=np.uint8)).cropped(400, 200, 1200, 700)
    inst = Instance2D(
        box=Box2D(xyxy=np.array([10.0, 20.0, 14.0, 23.0])),
        mask=np.ones((3, 4), dtype=np.bool_),
        mask_region=(10, 20, 14, 23),
    )
    out = InstanceSegmentation2DOutput(instances=[inst]).to_source_image(crop)
    # 領域は crop 原点 (400,200) だけ平行移動、サイズは不変（mask.shape と一致）。
    assert out.instances[0].mask_region == (410, 220, 414, 223)
    assert out.instances[0].mask.shape == (3, 4)


def test_instance_seg_mask_under_scale_raises() -> None:
    inst = Instance2D(
        box=Box2D(xyxy=np.array([0.0, 0.0, 4.0, 4.0])),
        mask=np.ones((4, 4), dtype=np.bool_),
        mask_region=(0, 0, 4, 4),
    )
    seg = InstanceSegmentation2DOutput(instances=[inst])
    with pytest.raises(NotImplementedError, match="resampl"):
        seg.to_source_image(_resized_image())


def test_instance_seg_boxes_only_under_scale_ok() -> None:
    # mask を持たないインスタンスは scale 付きでも box を変換できる。
    inst = Instance2D(box=Box2D(xyxy=np.array([320.0, 320.0, 320.0, 320.0])))
    out = InstanceSegmentation2DOutput(instances=[inst]).to_source_image(_resized_image())
    assert np.allclose(out.instances[0].box.xyxy, [800.0, 450.0, 800.0, 450.0])
