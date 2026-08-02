"""2D タスクの入力型（``ImagePrompt`` / ``ImageSample``）の構築と検証。"""

from __future__ import annotations

import numpy as np
import pytest

from jidohub.core.schemas import Image, ImagePrompt, ImageSample


def _image() -> Image:
    return Image(pixels=np.zeros((4, 6, 3), dtype=np.uint8), intrinsic=np.eye(3, dtype=np.float64))


def _points(n: int = 2) -> np.ndarray:
    return np.arange(n * 2, dtype=np.float64).reshape(n, 2)


# --- ImagePrompt ------------------------------------------------------------


def test_points_without_labels_rejected() -> None:
    with pytest.raises(ValueError, match="points and point_labels must be provided together"):
        ImagePrompt(points=_points())


def test_labels_without_points_rejected() -> None:
    with pytest.raises(ValueError, match="points and point_labels must be provided together"):
        ImagePrompt(point_labels=np.array([1, 0], dtype=np.int64))


def test_points_labels_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="point_labels must have shape"):
        ImagePrompt(points=_points(2), point_labels=np.array([1], dtype=np.int64))


def test_points_wrong_shape_rejected() -> None:
    with pytest.raises(ValueError, match=r"points must have shape \(P, 2\)"):
        ImagePrompt(
            points=np.zeros((3, 3), dtype=np.float64),
            point_labels=np.zeros((3,), dtype=np.int64),
        )


def test_boxes_wrong_shape_rejected() -> None:
    with pytest.raises(ValueError, match=r"boxes must have shape \(B, 4\)"):
        ImagePrompt(boxes=np.zeros((3, 5), dtype=np.float64))


def test_points_with_labels_builds() -> None:
    prompt = ImagePrompt(points=_points(2), point_labels=np.array([1, 0], dtype=np.int64))
    assert prompt.points.shape == (2, 2)


def test_texts_only_builds() -> None:
    prompt = ImagePrompt(texts=["car", "person"])
    assert prompt.texts == ["car", "person"]


def test_boxes_only_builds() -> None:
    prompt = ImagePrompt(boxes=np.array([[0.0, 0.0, 2.0, 2.0]], dtype=np.float64))
    assert prompt.boxes.shape == (1, 4)


# --- ImageSample ------------------------------------------------------------


def test_image_sample_without_prompt_builds() -> None:
    sample = ImageSample(image=_image())
    assert sample.prompt is None
    assert sample.image.width == 6


def test_image_sample_with_prompt_builds() -> None:
    sample = ImageSample(image=_image(), prompt=ImagePrompt(texts=["car"]), sample_id="s0")
    assert sample.prompt is not None
    assert sample.sample_id == "s0"
