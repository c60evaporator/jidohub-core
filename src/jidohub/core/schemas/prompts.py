"""2D タスクの入力スキーマ。

2D タスク（``object_detection_2d`` など）の入力は :class:`~jidohub.core.schemas.Sample`
（センサ入力）ではなく :class:`ImageSample`（画像 1 枚 + オプションのプロンプト）である。
``predict(input) -> output`` の契約を崩さないため、画像とプロンプトを単一の
:class:`ImageSample` にまとめる。

規約
    - プロンプトの座標（``points`` / ``boxes``）も出力の座標も、すべて**入力
      :class:`~jidohub.core.schemas.image.Image` の現サイズ基準**（左上原点、
      ``x`` 右・``y`` 下。`2d_tasks.md` 3.2）。元画像へ戻す場合は ``Image.source`` を使う。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from jidohub.core.schemas.image import Image

__all__ = ["ImagePrompt", "ImageSample"]


@dataclass
class ImagePrompt:
    """2D タスクのオプション入力。指定された項目のみが有効。

    Attributes:
        points: shape ``(P, 2)``、``np.float64``。画像座標の点プロンプト。
            ``point_labels`` と**必ず同時に**指定する。
        point_labels: shape ``(P,)``、``np.int_``。各点が前景（``1``）か背景（``0``）か。
        boxes: shape ``(B, 4)``、``np.float64``。``(x0, y0, x1, y1)`` のボックスプロンプト。
        texts: テキストプロンプト / ラベル候補。**分類のラベル候補と検出のテキスト
            プロンプトで共用する**（対象を言語で指定する点で意味的に同じであり、
            分けると型が増えるだけ。`2d_tasks.md` 4 章）。
    """

    points: np.ndarray | None = None
    point_labels: np.ndarray | None = None
    boxes: np.ndarray | None = None
    texts: list[str] | None = None

    def __post_init__(self) -> None:
        if (self.points is None) != (self.point_labels is None):
            raise ValueError("ImagePrompt.points and point_labels must be provided together")
        if self.points is not None:
            if self.points.ndim != 2 or self.points.shape[1] != 2:
                raise ValueError(
                    f"ImagePrompt.points must have shape (P, 2), got {self.points.shape}"
                )
            assert self.point_labels is not None  # 上の同時指定チェックで保証済み
            if self.point_labels.ndim != 1 or self.point_labels.shape[0] != self.points.shape[0]:
                raise ValueError(
                    "ImagePrompt.point_labels must have shape (P,) matching points "
                    f"(got labels {self.point_labels.shape}, points {self.points.shape})"
                )
        if self.boxes is not None and (self.boxes.ndim != 2 or self.boxes.shape[1] != 4):
            raise ValueError(f"ImagePrompt.boxes must have shape (B, 4), got {self.boxes.shape}")


@dataclass
class ImageSample:
    """2D タスクの入力。:class:`~jidohub.core.schemas.Sample`（センサ入力）と対になる。

    Attributes:
        image: 入力画像。
        prompt: オプションのプロンプト。プロンプトを取らないタスク・Agent では ``None``。
        sample_id: データセット内での一意識別子（評価時の対応付けに使う）。
        metadata: データセット固有の付加情報。標準スキーマで表せるものは入れない。
    """

    image: Image
    prompt: ImagePrompt | None = None
    sample_id: str | None = None
    metadata: dict = field(default_factory=dict)
