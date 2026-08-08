"""座標変換・回転表現のヘルパ。

純粋な numpy 実装のみ。scipy / torch などに依存しないこと。

**座標系の規約（CLAUDE.md 3 章と同一）**

- ego 座標系は右手系で ``x`` = 前方、``y`` = 左方、``z`` = 上方。
- 変換行列はすべて 4x4 同次変換行列（``np.float64``）。
- 変換行列の変数名・引数名は必ず ``<from>_to_<to>`` の向きを明示する。
  「extrinsic」のような向きの曖昧な名前を単体で使わない。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from jidohub.core.schemas.image import ImageSource

__all__ = [
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
    "quaternion_to_yaw",
    "yaw_to_quaternion",
    "invert_transform",
    "transform_points",
    "rotate_vectors",
    "transform_quaternion",
    "crop_intrinsic",
    "scale_intrinsic",
    "denormalize_boxes",
    "normalize_boxes",
    "boxes_to_source",
    "boxes_from_source",
    "points_to_source",
    "points_from_source",
    "scaled_source",
    "resize_mask_nearest",
]


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """クォータニオンを 3x3 回転行列に変換する。

    Args:
        quaternion: shape ``(4,)``、``(w, x, y, z)`` の順。正規化されている前提。

    Returns:
        shape ``(3, 3)`` の回転行列（``np.float64``）。
    """
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """3x3 回転行列を ``(w, x, y, z)`` のクォータニオンに変換する。

    数値的に安定な Shepperd 法（対角成分の trace の符号で場合分けする）を用いる。
    素朴な実装は ``w`` が 0 に近い（180 度付近の）回転で精度を失うため。
    :func:`quaternion_to_rotation_matrix` の逆写像であり、往復で一致する
    （符号の不定性はあるが回転としては同値）。

    Args:
        matrix: shape ``(3, 3)`` の回転行列（直交・行列式 +1 を前提）。

    Returns:
        shape ``(4,)``、``(w, x, y, z)`` 順の**正規化された**クォータニオン（``np.float64``）。
    """
    m = np.asarray(matrix, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]

    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (m[2, 1] - m[1, 2]) / scale
        y = (m[0, 2] - m[2, 0]) / scale
        z = (m[1, 0] - m[0, 1]) / scale
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        scale = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / scale
        x = 0.25 * scale
        y = (m[0, 1] + m[1, 0]) / scale
        z = (m[0, 2] + m[2, 0]) / scale
    elif m[1, 1] > m[2, 2]:
        scale = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / scale
        x = (m[0, 1] + m[1, 0]) / scale
        y = 0.25 * scale
        z = (m[1, 2] + m[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / scale
        x = (m[0, 2] + m[2, 0]) / scale
        y = (m[1, 2] + m[2, 1]) / scale
        z = 0.25 * scale

    quaternion = np.array([w, x, y, z], dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def quaternion_to_yaw(quaternion: np.ndarray) -> float:
    """クォータニオンから z 軸まわりの yaw 角[rad]を取り出す。

    x 軸ベクトルを回転させて水平面に射影する方式で算出する
    （nuScenes devkit の ``quaternion_yaw`` と同じ定義）。
    ロール・ピッチを含む姿勢に対しても「車両が向いている方位」として
    直感に合う値を返すため、オイラー角分解より本用途に適する。

    Args:
        quaternion: shape ``(4,)``、``(w, x, y, z)`` の順。

    Returns:
        yaw 角[rad]。範囲は ``(-pi, pi]``。ego 座標系では ``0`` が前方（+x）、
        正の値が左回り（+y 方向へ向く）。
    """
    rotation = quaternion_to_rotation_matrix(quaternion)
    forward = rotation @ np.array([1.0, 0.0, 0.0])
    return float(np.arctan2(forward[1], forward[0]))


def yaw_to_quaternion(yaw: float) -> np.ndarray:
    """z 軸まわりの yaw 角[rad]をクォータニオンに変換する。

    yaw のみを扱うモデル（mmdetection3d 系など）の出力を
    標準スキーマの ``Box3D.rotation`` に載せる際に使う。

    Args:
        yaw: yaw 角[rad]。

    Returns:
        shape ``(4,)``、``(w, x, y, z)`` の順のクォータニオン。
    """
    half = float(yaw) / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """4x4 同次変換行列の逆変換を返す。

    回転部が直交行列であることを利用するため、一般の逆行列計算より安定かつ高速。

    Args:
        transform: shape ``(4, 4)`` の同次変換行列。

    Returns:
        shape ``(4, 4)`` の逆変換行列。``a_to_b`` を渡すと ``b_to_a`` が返る。
    """
    transform = np.asarray(transform, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    inverted = np.eye(4, dtype=np.float64)
    inverted[:3, :3] = rotation.T
    inverted[:3, 3] = -rotation.T @ translation
    return inverted


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """点群に 4x4 同次変換を適用する。

    先頭 3 列（x, y, z）のみを変換し、4 列目以降（intensity, ring など）は
    そのまま保持する。入力の dtype を維持する。

    Args:
        points: shape ``(N, C)``、``C >= 3``。先頭 3 列が x, y, z。
        transform: shape ``(4, 4)`` の同次変換行列。

    Returns:
        変換後の点群。shape・dtype は入力と同じ。
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"points must have shape (N, C>=3), got {points.shape}")

    transform = np.asarray(transform, dtype=np.float64)
    xyz = points[:, :3].astype(np.float64)
    transformed = xyz @ transform[:3, :3].T + transform[:3, 3]

    result = points.copy()
    result[:, :3] = transformed.astype(points.dtype)
    return result


def rotate_vectors(vectors: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """ベクトル量に**回転のみ**を適用する（平行移動を加えない）。

    速度のようなベクトル量に使う。位置（点）と異なり平行移動を適用してはならない。
    この誤りは「静止物体が高速で動いて見える」形で現れ、例外は出ない（4.3）。

    Args:
        vectors: shape ``(3,)`` または ``(N, 3)``。
        rotation_matrix: shape ``(3, 3)`` の回転行列。4x4 同次変換の回転部
            （``transform[:3, :3]``）をそのまま渡してよい。

    Returns:
        回転後のベクトル。shape は入力と同じ、``np.float64``。**入力は破壊しない。**
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    rotation = np.asarray(rotation_matrix, dtype=np.float64)[:3, :3]
    if vectors.shape[-1] != 3:
        raise ValueError(f"rotate_vectors expects last dim 3, got shape {vectors.shape}")
    return vectors @ rotation.T


def transform_quaternion(quaternion: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """回転を表すクォータニオンに、4x4 同次変換の**回転部**を合成する。

    ``Box3D.rotation`` を別座標系へ移す際に使う。平行移動は姿勢に影響しないため
    ``transform`` の回転部のみを用いる。合成後の回転行列を
    :func:`rotation_matrix_to_quaternion` で戻す。

    Args:
        quaternion: shape ``(4,)``、``(w, x, y, z)`` 順。
        transform: shape ``(4, 4)`` の同次変換行列（``a_to_b``）。

    Returns:
        変換後の姿勢を表す shape ``(4,)`` のクォータニオン（``np.float64``）。
    """
    transform = np.asarray(transform, dtype=np.float64)
    rotated = transform[:3, :3] @ quaternion_to_rotation_matrix(quaternion)
    return rotation_matrix_to_quaternion(rotated)


def crop_intrinsic(intrinsic: np.ndarray, x0: float, y0: float) -> np.ndarray:
    """crop 後の画像に対応する内部パラメータ ``K`` を返す（主点を平行移動）。

    画素座標の原点は左上（``x`` 右・``y`` 下）。``(x0, y0)`` を新しい原点にする
    切り出しでは、主点 ``(cx, cy)`` が同じだけ左上へ移動する（焦点距離は不変）。
    元画像上の点 ``(u, v)`` は crop 後に ``(u - x0, v - y0)`` に写る。

    Args:
        intrinsic: shape ``(3, 3)`` のピンホール内部パラメータ。
        x0: crop 左端の x 座標[px]。
        y0: crop 上端の y 座標[px]。

    Returns:
        shape ``(3, 3)``、``np.float64`` の新しい ``K``。**入力は破壊しない。**
    """
    result = np.array(intrinsic, dtype=np.float64, copy=True)
    result[0, 2] -= x0
    result[1, 2] -= y0
    return result


def scale_intrinsic(intrinsic: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    """リサイズ後の画像に対応する内部パラメータ ``K`` を返す（焦点距離と主点をスケール）。

    画像を ``(scale_x, scale_y)`` 倍にリサンプルすると、焦点距離 ``(fx, fy)`` と
    主点 ``(cx, cy)`` が同じ倍率でスケールする。実際の画素のリサンプリングは行わない
    （それは画像処理ライブラリの責務であり core は持たない）。``K`` の更新のみを担う。

    Args:
        intrinsic: shape ``(3, 3)`` のピンホール内部パラメータ。
        scale_x: x 方向の拡大率（現サイズ / 元サイズ）。
        scale_y: y 方向の拡大率。

    Returns:
        shape ``(3, 3)``、``np.float64`` の新しい ``K``。**入力は破壊しない。**
    """
    result = np.array(intrinsic, dtype=np.float64, copy=True)
    result[0, 0] *= scale_x
    result[0, 2] *= scale_x
    result[1, 1] *= scale_y
    result[1, 2] *= scale_y
    return result


# --- 2D（画像平面）座標の変換 -------------------------------------------------
#
# 画素座標の原点は左上（x 右・y 下）。crop / resize / 正規化を経た座標を元画像へ
# 戻す際の取り違えを防ぐため、変換の実体をここに集約する（`coordinate_transforms.md` 6 章）。
# すべて入力を破壊せず新しい配列を返す。


def denormalize_boxes(xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    """正規化座標 ``[0, 1]`` の box を画素座標へ戻す。

    Args:
        xyxy: shape ``(4,)`` または ``(N, 4)``。``(x0, y0, x1, y1)`` の正規化座標。
        width: 対象画像の幅[px]。
        height: 対象画像の高さ[px]。

    Returns:
        画素座標の box。shape は入力と同じ、``np.float64``。
    """
    xyxy = np.asarray(xyxy, dtype=np.float64)
    return xyxy * np.array([width, height, width, height], dtype=np.float64)


def normalize_boxes(xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    """画素座標の box を正規化座標 ``[0, 1]`` へ変換する（:func:`denormalize_boxes` の逆）。

    Args:
        xyxy: shape ``(4,)`` または ``(N, 4)``。``(x0, y0, x1, y1)`` の画素座標。
        width: 対象画像の幅[px]。
        height: 対象画像の高さ[px]。

    Returns:
        正規化座標の box。shape は入力と同じ、``np.float64``。
    """
    xyxy = np.asarray(xyxy, dtype=np.float64)
    return xyxy / np.array([width, height, width, height], dtype=np.float64)


def _source_params(source: "ImageSource | None") -> tuple[float, float, float, float]:
    """``ImageSource`` から実効的な ``(crop_x0, crop_y0, scale_x, scale_y)`` を取り出す。

    ``source`` が ``None`` なら恒等 ``(0, 0, 1, 1)``。``crop`` が ``None`` なら原点 ``(0, 0)``、
    ``scale`` が ``None`` なら等倍 ``(1, 1)`` とみなす。属性のみを読むため ``ImageSource`` を
    import しない（モジュール循環回避。型注釈は ``TYPE_CHECKING`` 下でのみ解決）。
    """
    if source is None:
        return 0.0, 0.0, 1.0, 1.0
    x0, y0 = (float(source.crop[0]), float(source.crop[1])) if source.crop else (0.0, 0.0)
    sx, sy = (float(source.scale[0]), float(source.scale[1])) if source.scale else (1.0, 1.0)
    return x0, y0, sx, sy


def boxes_to_source(xyxy: np.ndarray, source: "ImageSource | None") -> np.ndarray:
    """現画像基準の box を、``source`` に従って元画像基準へ戻す。

    元画像 ``(u, v)`` は ``crop`` と ``scale`` を経て現画像で ``((u - x0) * sx, (v - y0) * sy)``
    になる。本関数はその逆写像 ``(x / sx + x0, y / sy + y0)`` を適用する。まず ``scale`` を
    逆適用し、次に ``crop`` 原点を加算する（順序を誤ると crop 原点の扱いがずれる）。

    Args:
        xyxy: shape ``(4,)`` または ``(N, 4)``。現画像の**画素**座標
            （正規化座標には使わない。先に :func:`denormalize_boxes` で戻すこと）。
        source: 現画像の由来。``None`` / ``crop`` None / ``scale`` None は恒等成分として扱う。

    Returns:
        元画像基準の box。shape は入力と同じ、``np.float64``。
    """
    xyxy = np.asarray(xyxy, dtype=np.float64)
    x0, y0, sx, sy = _source_params(source)
    return xyxy / np.array([sx, sy, sx, sy]) + np.array([x0, y0, x0, y0])


def boxes_from_source(xyxy: np.ndarray, source: "ImageSource | None") -> np.ndarray:
    """元画像基準の box を、``source`` に従って現画像基準へ写す（:func:`boxes_to_source` の逆）。

    Args:
        xyxy: shape ``(4,)`` または ``(N, 4)``。元画像の画素座標。
        source: 現画像の由来。

    Returns:
        現画像基準の box。shape は入力と同じ、``np.float64``。
    """
    xyxy = np.asarray(xyxy, dtype=np.float64)
    x0, y0, sx, sy = _source_params(source)
    return (xyxy - np.array([x0, y0, x0, y0])) * np.array([sx, sy, sx, sy])


def points_to_source(points: np.ndarray, source: "ImageSource | None") -> np.ndarray:
    """現画像基準の点列を元画像基準へ戻す（プロンプト座標などに使う）。

    Args:
        points: shape ``(2,)`` または ``(P, 2)``。現画像の画素座標 ``(x, y)``。
        source: 現画像の由来。

    Returns:
        元画像基準の点列。shape は入力と同じ、``np.float64``。
    """
    points = np.asarray(points, dtype=np.float64)
    x0, y0, sx, sy = _source_params(source)
    return points / np.array([sx, sy]) + np.array([x0, y0])


def points_from_source(points: np.ndarray, source: "ImageSource | None") -> np.ndarray:
    """元画像基準の点列を現画像基準へ写す（:func:`points_to_source` の逆）。

    Args:
        points: shape ``(2,)`` または ``(P, 2)``。元画像の画素座標 ``(x, y)``。
        source: 現画像の由来。

    Returns:
        現画像基準の点列。shape は入力と同じ、``np.float64``。
    """
    points = np.asarray(points, dtype=np.float64)
    x0, y0, sx, sy = _source_params(source)
    return (points - np.array([x0, y0])) * np.array([sx, sy])


def scaled_source(source: "ImageSource | None", scale_x: float, scale_y: float) -> "ImageSource":
    """``source`` に resize 分の ``scale`` を合成した新しい :class:`ImageSource` を返す。

    agents 側の Processor は「画素の resize」「:func:`scale_intrinsic`」「本関数」を
    **3 点セット**で行う。ここで ``source`` の更新が漏れると 2D の変換全体が破綻するため、
    core が更新手段を提供する（5.4）。``crop`` / ``channel`` は保ち、``scale`` のみ
    乗算合成する（``scale`` は crop 後サイズ→現サイズの拡大率であり、resize は乗算で積み重なる）。

    Args:
        source: 元の由来。``None`` なら crop なし・等倍から開始する。
        scale_x: 今回の resize の x 方向拡大率（新サイズ / 現サイズ）。
        scale_y: y 方向拡大率。

    Returns:
        ``scale`` を合成した新しい :class:`ImageSource`。**入力は破壊しない。**
    """
    from jidohub.core.schemas.image import ImageSource

    if source is None:
        return ImageSource(scale=(scale_x, scale_y))
    old_sx, old_sy = source.scale if source.scale else (1.0, 1.0)
    return ImageSource(
        channel=source.channel,
        crop=source.crop,
        scale=(old_sx * scale_x, old_sy * scale_y),
    )


def resize_mask_nearest(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """bool マスクを最近傍補間で ``(height, width)`` にリサイズする。

    純 numpy のインデックス操作のみで実装する（画像処理ライブラリを import しない）。
    bool マスクに適用できる補間は**最近傍のみ**である（bilinear で補間すると bool でなくなり
    再度しきい値処理が必要になる）ため、最近傍リサイズは numpy で完結する。写像は
    **画素中心基準**（``align_corners=False`` 相当）で行う。

    Args:
        mask: shape ``(h, w)``、``np.bool_``。
        height: 出力の高さ[px]（1 以上）。
        width: 出力の幅[px]（1 以上）。

    Returns:
        shape ``(height, width)``、``np.bool_`` の **C 連続**配列。**入力は破壊しない。**

    Raises:
        ValueError: ``mask`` が 2 次元でない、または ``height`` / ``width`` が 1 未満の場合。

    Note:
        二値化済みマスクの**拡大**は境界がブロック状になる（情報は増えない）。品質が必要な場合は
        Agent 側で**二値化の前**に float（logit）マスクを入力解像度へ補間するのが本来の経路であり、
        本関数はそれを経ていない bool 出力を元画像座標へ移すための**フォールバック**である。
    """
    if mask.ndim != 2:
        raise ValueError(f"resize_mask_nearest expects a 2-D mask, got shape {mask.shape}")
    if height < 1 or width < 1:
        raise ValueError(f"resize_mask_nearest size must be >= 1, got {width}x{height}")
    h, w = mask.shape
    rows = ((np.arange(height) + 0.5) * h / height).astype(np.int64).clip(0, h - 1)
    cols = ((np.arange(width) + 0.5) * w / width).astype(np.int64).clip(0, w - 1)
    resized = mask[rows][:, cols]
    return np.ascontiguousarray(resized, dtype=np.bool_)
