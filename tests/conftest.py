"""共有フィクスチャとファクトリ。

正常系のスキーマ・出力インスタンスを最小限の労力で組み立てるためのヘルパを提供する。
テストは実データ（実 numpy 配列）を使い、ネットワークにはアクセスしない。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jidohub.core.schemas import (
    Box3D,
    CameraFrame,
    Detection3DOutput,
    DrivingCommand,
    E2EOutput,
    EgoState,
    Image,
    LidarSweep,
    MapElement,
    MapElementType,
    MapOutput,
    MotionForecastOutput,
    PlanningOutput,
    RadarSweep,
    Sample,
)
from jidohub.core.schemas.outputs import AgentForecast

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"


def identity_transform() -> np.ndarray:
    """恒等な 4x4 同次変換行列（float64）。"""
    return np.eye(4, dtype=np.float64)


def make_transform(translation: tuple[float, float, float] = (1.0, 2.0, 3.0)) -> np.ndarray:
    """z 軸まわり回転 + 並進を含む剛体変換（float64, shape (4,4)）。"""
    theta = np.pi / 6.0
    cos, sin = np.cos(theta), np.sin(theta)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array(
        [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def make_camera_frame(channel: str = "CAM_FRONT") -> CameraFrame:
    return CameraFrame(
        image=Image(
            pixels=np.zeros((4, 6, 3), dtype=np.uint8),
            intrinsic=np.eye(3, dtype=np.float64),
        ),
        sensor_to_ego=make_transform(),
        channel=channel,
        timestamp=1_600_000_000_000_000,
    )


def make_lidar_sweep() -> LidarSweep:
    points = np.arange(20, dtype=np.float32).reshape(5, 4)
    return LidarSweep(points=points, sensor_to_ego=make_transform())


def make_radar_sweep() -> RadarSweep:
    points = np.arange(15, dtype=np.float32).reshape(5, 3)
    return RadarSweep(points=points, sensor_to_ego=make_transform(), channel="RADAR_FRONT")


def make_sample(*, with_history: bool = True) -> Sample:
    """カメラ複数・lidar・radar・ego_state・command・history・metadata を含む Sample。"""
    history: list[Sample] = []
    if with_history:
        history = [make_sample(with_history=False)]
    return Sample(
        timestamp=1_600_000_000_000_000,
        ego_to_global=make_transform(),
        cameras={
            "CAM_FRONT": make_camera_frame("CAM_FRONT"),
            "CAM_BACK": make_camera_frame("CAM_BACK"),
        },
        lidar=make_lidar_sweep(),
        radars={"RADAR_FRONT": make_radar_sweep()},
        ego_state=EgoState(
            velocity=np.array([1.0, 0.0, 0.0]),
            steering_angle=0.1,
        ),
        command=DrivingCommand.TURN_LEFT,
        history=history,
        sample_id="sample-0",
        scene_id="scene-0",
        metadata={"note": "unit-test", "index": 3},
    )


def make_box() -> Box3D:
    return Box3D(
        center=np.array([1.0, 2.0, 0.5], dtype=np.float64),
        size=np.array([4.0, 1.8, 1.5], dtype=np.float64),
        rotation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        label="car",
        score=0.9,
        velocity=np.array([2.0, 0.0, 0.0], dtype=np.float64),
        track_id=7,
        attributes={"moving": True},
    )


def make_e2e_output() -> E2EOutput:
    """全中間出力を含む E2EOutput。"""
    detection = Detection3DOutput(boxes=[make_box()])
    tracking = Detection3DOutput(boxes=[make_box()])
    map_out = MapOutput(
        elements=[
            MapElement(
                points=np.zeros((3, 2), dtype=np.float32),
                element_type=MapElementType.DIVIDER,
            )
        ]
    )
    motion = MotionForecastOutput(
        forecasts=[
            AgentForecast(
                trajectories=np.zeros((3, 6, 2), dtype=np.float32),
                probabilities=np.array([0.5, 0.3, 0.2], dtype=np.float32),
                dt=0.5,
                track_id=7,
            )
        ]
    )
    planning = PlanningOutput(
        trajectory=np.zeros((6, 2), dtype=np.float32),
        dt=0.5,
        confidence=0.8,
    )
    return E2EOutput(
        planning=planning,
        detection=detection,
        tracking=tracking,
        map=map_out,
        motion_forecast=motion,
        occupancy=np.zeros((2, 2, 2), dtype=np.float32),
        occupancy_meta={"resolution": 0.5},
        bev_feature=np.zeros((4, 4), dtype=np.float32),
    )


@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    """JIDOHUB_HOME を tmp_path に向け、実ホームの ~/.cache を汚さない。"""
    home = tmp_path / "jidohub_home"
    home.mkdir()
    monkeypatch.setenv("JIDOHUB_HOME", str(home))
    return home
