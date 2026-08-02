"""2D 出力型（``Box2D`` / ``Instance2D`` / 各 Output）の構築と検証。"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from jidohub.core.schemas import (
    Box2D,
    Classification2DOutput,
    Instance2D,
)


def _box() -> Box2D:
    return Box2D(xyxy=np.array([2.0, 1.0, 6.0, 5.0], dtype=np.float64))


# --- Box2D ------------------------------------------------------------------


def test_box2d_xyxy_wrong_shape_rejected() -> None:
    with pytest.raises(ValueError, match=r"Box2D.xyxy must have shape \(4,\)"):
        Box2D(xyxy=np.array([1.0, 2.0, 3.0], dtype=np.float64))


def test_box2d_track_id_is_a_field_not_property() -> None:
    # object_tracking_2d 追加時にフィールドを足すと破壊的になるため、最初から field で持つ。
    field_names = {f.name for f in dataclasses.fields(Box2D)}
    assert "track_id" in field_names


def test_box2d_label_optional() -> None:
    # プロンプタブル系では label が None になり得る。
    assert _box().label is None


# --- Instance2D -------------------------------------------------------------


def test_mask_without_region_rejected() -> None:
    with pytest.raises(ValueError, match="mask and mask_region must be provided together"):
        Instance2D(box=_box(), mask=np.ones((4, 4), dtype=np.bool_))


def test_region_without_mask_rejected() -> None:
    with pytest.raises(ValueError, match="mask and mask_region must be provided together"):
        Instance2D(box=_box(), mask_region=(2, 1, 6, 5))


def test_mask_region_size_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="mask_region size must match mask shape"):
        # mask は 4x4 だが region は 4x4 でない（幅 5）。
        Instance2D(box=_box(), mask=np.ones((4, 4), dtype=np.bool_), mask_region=(2, 1, 7, 5))


def test_mask_must_be_2d() -> None:
    with pytest.raises(ValueError, match="mask must be a 2-D bool array"):
        Instance2D(
            box=_box(),
            mask=np.ones((2, 4, 4), dtype=np.bool_),  # 3 次元
            mask_region=(2, 1, 6, 5),
        )


def test_mask_must_be_bool() -> None:
    with pytest.raises(ValueError, match="mask must be a 2-D bool array"):
        Instance2D(box=_box(), mask=np.ones((4, 4), dtype=np.uint8), mask_region=(2, 1, 6, 5))


def test_instance_with_mask_builds() -> None:
    instance = Instance2D(
        box=_box(), mask=np.ones((4, 4), dtype=np.bool_), mask_region=(2, 1, 6, 5)
    )
    assert instance.mask.shape == (4, 4)


def test_instance_without_mask_builds() -> None:
    # ボックスのみ（マスクなし）も許容する。
    instance = Instance2D(box=_box())
    assert instance.mask is None and instance.mask_region is None


def test_mask_paste_back_into_full_canvas() -> None:
    """``mask_region`` に従って全画面キャンバスへ貼り戻し、期待位置に載ることを確認する。

    座標の取り違え（x/y や w/h の入れ替え）を検出する唯一の手段。
    """
    height, width = 10, 12
    x0, y0, x1, y1 = 3, 2, 7, 5  # 幅 4, 高さ 3
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.bool_)  # (3, 4)
    mask[0, 0] = True  # crop 内左上 → 全画面では (y0, x0)
    mask[2, 3] = True  # crop 内右下 → 全画面では (y1-1, x1-1)

    instance = Instance2D(box=_box(), mask=mask, mask_region=(x0, y0, x1, y1))

    canvas = np.zeros((height, width), dtype=np.bool_)
    rx0, ry0, rx1, ry1 = instance.mask_region
    canvas[ry0:ry1, rx0:rx1] = instance.mask

    expected = np.zeros((height, width), dtype=np.bool_)
    expected[y0, x0] = True
    expected[y1 - 1, x1 - 1] = True
    assert np.array_equal(canvas, expected)


# --- Classification2DOutput -------------------------------------------------


def test_classification_labels_scores_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="scores length must match labels"):
        Classification2DOutput(labels=["a", "b"], scores=np.array([0.5], dtype=np.float64))


def test_classification_without_scores_builds() -> None:
    output = Classification2DOutput(labels=["a", "b"])
    assert output.scores is None


def test_classification_with_matching_scores_builds() -> None:
    output = Classification2DOutput(
        labels=["a", "b"], scores=np.array([0.7, 0.3], dtype=np.float64)
    )
    assert output.scores.shape == (2,)
