"""タスク別の出力スキーマ。

:class:`~jidohub.core.tasks.TaskType` と 1 対 1 で対応する出力型を定義する。

設計方針
    E2E（``sensing_to_planning``）の中間出力は、単体タスクの出力型を
    **再利用**して表現する。これにより UniAD の中間出力可視化を、
    Detection / Map 単体タスク用の可視化コードのまま実現できる。
    新しい中間出力を追加する際も、まず単体タスクの型として定義してから
    :class:`E2EOutput` に載せること。

規約（詳細は CLAUDE.md 3 章）
    - 出力の座標系は **ego 座標系を既定**とし、``frame`` フィールドで必ず明示する。
    - 回転は quaternion ``(w, x, y, z)`` を正とし、``yaw`` は派生プロパティ。
    - 寸法は ``(length, width, height)`` の順（ego 座標の x, y, z 軸方向に対応）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from jidohub.core.geometry import (
    boxes_to_source,
    denormalize_boxes,
    invert_transform,
    quaternion_to_yaw,
    resize_mask_nearest,
    rotate_vectors,
    transform_quaternion,
)

if TYPE_CHECKING:
    from jidohub.core.schemas.image import Image

__all__ = [
    "CoordinateFrame",
    "Box3D",
    "Detection3DOutput",
    "Box2D",
    "Instance2D",
    "Detection2DOutput",
    "InstanceSegmentation2DOutput",
    "Classification2DOutput",
    "MapElementType",
    "MapElement",
    "MapOutput",
    "AgentForecast",
    "MotionForecastOutput",
    "PlanningOutput",
    "E2EOutput",
]


class CoordinateFrame(str, Enum):
    """出力が表現されている座標系。

    暗黙の前提を作らないため、座標を持つ出力型は必ずこのフィールドを持つ。
    """

    EGO = "ego"
    """ego 座標系（x = 前方、y = 左方、z = 上方）。既定値。"""

    GLOBAL = "global"
    """global（マップ）座標系。"""

    CAMERA = "camera"
    """カメラ座標系。単眼深度（``depth_estimation`` の ``DepthOutput``、将来定義）の
    出力は ego 座標に置けない（外部パラメータが必要で、それを持つのは ``CameraFrame`` を
    握っている呼び出し側）ため予約する。後から追加すると ``schema_version`` を上げる
    変更になるため、`2d_tasks.md` 6.4 に従い今のうちに値だけ用意する。"""


@dataclass
class Box3D:
    """3D バウンディングボックス 1 個。

    検出・追跡・E2E の中間出力で共通して使う。

    Attributes:
        center: shape ``(3,)``、``np.float64``。ボックス中心の座標[m]。
            **底面中心ではなく重心（幾何中心）**である点に注意（nuScenes 準拠）。
        size: shape ``(3,)``、``np.float64``。``(length, width, height)``[m] の順で、
            ego 座標系の x / y / z 軸方向の寸法に対応する。
            **nuScenes の ``sample_annotation.size`` は ``(width, length, height)`` 順**で
            あり、Adapter で必ず入れ替えが必要（取り違えても例外は出ないため注意）。
            個別の値には ``size[0]`` 等でインデックスせず、
            :attr:`length` / :attr:`width` / :attr:`height` プロパティを使うこと。
        rotation: shape ``(4,)``、``np.float64``。``(w, x, y, z)`` 順のクォータニオン。
            yaw のみを扱うモデルからの変換には
            :func:`~jidohub.core.geometry.yaw_to_quaternion` を使う。
        label: クラス名の文字列（例: ``"car"``）。
            クラス体系はデータセット・モデル依存のため core では enum 化しない。
        score: 信頼度スコア ``[0, 1]``。GT の場合は ``None``。
        velocity: shape ``(2,)`` または ``(3,)``、``np.float64``。速度[m/s]。
            ``frame`` と同じ座標系で表す。
        track_id: 追跡 ID。単体検出では ``None``。
        attributes: データセット固有の属性（nuScenes の ``vehicle.moving`` 等）。
    """

    center: np.ndarray
    size: np.ndarray
    rotation: np.ndarray
    label: str
    score: float | None = None
    velocity: np.ndarray | None = None
    track_id: int | None = None
    attributes: dict = field(default_factory=dict)

    @property
    def yaw(self) -> float:
        """z 軸まわりの yaw 角[rad]（読み取り専用の派生値）。

        BEV 描画や yaw ベースのフレームワークとの連携用。
        **この値を正として保持しないこと**（情報が欠落するため）。
        """
        return quaternion_to_yaw(self.rotation)

    @property
    def length(self) -> float:
        """車長[m]。ego 座標系の x 軸方向の寸法（``size[0]``）。"""
        return float(self.size[0])

    @property
    def width(self) -> float:
        """車幅[m]。ego 座標系の y 軸方向の寸法（``size[1]``）。"""
        return float(self.size[1])

    @property
    def height(self) -> float:
        """車高[m]。ego 座標系の z 軸方向の寸法（``size[2]``）。"""
        return float(self.size[2])

    @classmethod
    def from_dimensions(
        cls,
        center: np.ndarray,
        length: float,
        width: float,
        height: float,
        rotation: np.ndarray,
        label: str,
        **kwargs: object,
    ) -> Box3D:
        """寸法をキーワードで指定して構築する。

        ``size`` の並び順を意識せずに済むため、データセット Adapter からの
        構築ではこちらを使うことを推奨する。nuScenes のように
        ``(width, length, height)`` 順で寸法を保持する形式からの変換で、
        取り違えを防げる。

        ``score`` / ``velocity`` / ``track_id`` / ``attributes`` は ``kwargs`` で
        そのまま渡せる。
        """
        size = np.array([length, width, height], dtype=np.float64)
        return cls(center=center, size=size, rotation=rotation, label=label, **kwargs)  # type: ignore[arg-type]


# --- 3D 変換の内部ヘルパ（層 3 メソッドが委譲する）---------------------------
#
# 単一要素を変換する**公開**関数は置かない（座標系はコンテナが持つ原則。2 章）。
# geometry には schemas 非依存のプリミティブのみを置き、要素の再構築はここで行う
# （`transform_boxes` を geometry に置くと geometry→schemas の結合が生じるため）。


def _resolve_transform(
    frame: CoordinateFrame, target: CoordinateFrame, ego_to_global: np.ndarray
) -> np.ndarray | None:
    """``frame`` から ``target`` へ移すために適用する 4x4 変換を返す。

    既に ``target`` なら ``None``（呼び出し側は ``self`` を返して冪等にする）。
    ``CAMERA`` は ``ego_to_global`` だけでは変換できないため
    :class:`NotImplementedError`（黙って誤変換しない。4.2）。
    """
    if frame == CoordinateFrame.CAMERA:
        raise NotImplementedError(
            "cannot transform a CAMERA-frame output with ego_to_global alone; "
            "camera extrinsics are required and are held by the caller (CameraFrame)."
        )
    if frame == target:
        return None
    if target == CoordinateFrame.GLOBAL:  # EGO -> GLOBAL
        return np.asarray(ego_to_global, dtype=np.float64)
    return invert_transform(ego_to_global)  # GLOBAL -> EGO


def _transform_positions(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """位置（点）に回転 + 平行移動を適用する。新しい ``np.float64`` 配列を返す。

    最終軸が 2（BEV の平面座標）なら z = 0 を補って変換し、2 成分に戻す。
    最終軸が 3 ならそのまま。先頭次元は任意（``(3,)`` / ``(P, 2)`` / ``(M, T, 2)`` などに対応）。
    """
    pts = np.asarray(points, dtype=np.float64)
    rotation = np.asarray(transform, dtype=np.float64)[:3, :3]
    translation = np.asarray(transform, dtype=np.float64)[:3, 3]
    if pts.shape[-1] == 2:
        padded = np.concatenate([pts, np.zeros((*pts.shape[:-1], 1))], axis=-1)
        return (padded @ rotation.T + translation)[..., :2]
    return pts @ rotation.T + translation


def _transform_box3d(box: Box3D, transform: np.ndarray) -> Box3D:
    """:class:`Box3D` 1 個を変換した新しい :class:`Box3D` を返す（入力は不変）。

    ``center`` は回転 + 平行移動、``rotation`` は姿勢の合成、``velocity`` は**回転のみ**
    （平行移動を適用しない。速度はベクトル量。4.3）。``size`` 等は不変。
    """
    velocity = box.velocity
    if velocity is not None:
        if velocity.shape == (2,):
            padded = np.array([velocity[0], velocity[1], 0.0], dtype=np.float64)
            velocity = rotate_vectors(padded, transform)[:2]
        else:
            velocity = rotate_vectors(velocity, transform)
    return Box3D(
        center=_transform_positions(box.center, transform),
        size=box.size,
        rotation=transform_quaternion(box.rotation, transform),
        label=box.label,
        score=box.score,
        velocity=velocity,
        track_id=box.track_id,
        attributes=box.attributes,
    )


@dataclass
class Detection3DOutput:
    """3D 物体検出・追跡の出力。

    ``object_detection_3d`` と ``object_tracking_3d`` で共通。
    追跡の場合は各 :class:`Box3D` の ``track_id`` が埋まる。

    Attributes:
        boxes: 検出されたボックスのリスト。スコア降順である必要はない。
        frame: ``boxes`` が表現されている座標系。
    """

    boxes: list[Box3D] = field(default_factory=list)
    frame: CoordinateFrame = CoordinateFrame.EGO

    def to_ego(self, ego_to_global: np.ndarray) -> Detection3DOutput:
        """ego 座標系での新しい出力を返す（冪等・非破壊）。詳細は :meth:`to_global`。"""
        return self._to_frame(CoordinateFrame.EGO, ego_to_global)

    def to_global(self, ego_to_global: np.ndarray) -> Detection3DOutput:
        """global 座標系での新しい出力を返す。

        引数は常に ``ego_to_global``（``Sample.ego_to_global`` をそのまま渡せる）。
        既に目的の座標系なら ``self`` を返す（二重変換を構造的に防ぐ）。``frame`` は
        更新済みで返る。``CoordinateFrame.CAMERA`` の出力には
        :class:`NotImplementedError`（4.2）。
        """
        return self._to_frame(CoordinateFrame.GLOBAL, ego_to_global)

    def _to_frame(self, target: CoordinateFrame, ego_to_global: np.ndarray) -> Detection3DOutput:
        transform = _resolve_transform(self.frame, target, ego_to_global)
        if transform is None:
            return self
        return Detection3DOutput(
            boxes=[_transform_box3d(b, transform) for b in self.boxes], frame=target
        )


@dataclass
class Box2D:
    """2D バウンディングボックス 1 個（画像平面）。

    検出・追跡・インスタンス分割で共通して使う。

    Attributes:
        xyxy: shape ``(4,)``、``np.float64``。``(x0, y0, x1, y1)``[px]。原点は左上、
            ``x`` 右・``y`` 下（`2d_tasks.md` 3.2）。フィールド名を ``xyxy`` にすることで
            xywh との取り違えをコード上で防ぐ（``Box3D.size`` の並び順と同じ発想）。
            COCO 形式（xywh）への変換は評価層の責務であり core では扱わない。
        label: クラス名の**文字列**。ゼロショット分類・オープン語彙検出では
            ラベル集合が実行時に決まるためクラスインデックスにできない
            （`2d_tasks.md` 1 章）。プロンプタブル系では ``None`` になり得る。
        score: 信頼度スコア ``[0, 1]``。GT や単純なプロンプト応答では ``None``。
        track_id: 追跡 ID。単発検出では ``None``。``object_tracking_2d`` 追加時に
            フィールドを足すと破壊的変更になるため、**最初から**用意する（`2d_tasks.md` 9.1）。
        attributes: データセット固有の属性。
    """

    xyxy: np.ndarray
    label: str | None = None
    score: float | None = None
    track_id: int | None = None
    attributes: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.xyxy.shape != (4,):
            raise ValueError(f"Box2D.xyxy must have shape (4,), got {self.xyxy.shape}")


@dataclass
class Instance2D:
    """インスタンスセグメンテーションの 1 インスタンス。

    マスクは**全画面ではなく bbox 内に限定**して持つ。素朴な ``(N, H, W)`` の全画面
    マスクは 1600x900 で 50 インスタンスなら約 72MB になり、プロセス境界を越える設計では
    実用にならないため（`2d_tasks.md` 6.2）。

    Attributes:
        box: このインスタンスのバウンディングボックス。
        mask: shape ``(h, w)``、``np.bool_``。``mask_region`` が示す整数画素領域を覆う。
        mask_region: ``(x0, y0, x1, y1)`` の**整数**画素領域。``x1 - x0 == w``、
            ``y1 - y0 == h`` を満たす。``box.xyxy``（float）とは別に整数領域を持つのは、
            float から暗黙に丸めると実装ごとに 1 画素ずれるため（曖昧さの排除）。
    """

    box: Box2D
    mask: np.ndarray | None = None
    mask_region: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if (self.mask is None) != (self.mask_region is None):
            raise ValueError("Instance2D.mask and mask_region must be provided together")
        if self.mask is not None:
            if self.mask.ndim != 2 or self.mask.dtype != np.bool_:
                raise ValueError(
                    "Instance2D.mask must be a 2-D bool array, "
                    f"got shape={self.mask.shape} dtype={self.mask.dtype}"
                )
            assert self.mask_region is not None  # 上の同時指定チェックで保証済み
            x0, y0, x1, y1 = self.mask_region
            height, width = self.mask.shape
            if (x1 - x0, y1 - y0) != (width, height):
                raise ValueError(
                    "Instance2D.mask_region size must match mask shape "
                    f"(expected x1-x0={width}, y1-y0={height}; got {(x1 - x0, y1 - y0)})"
                )

    def paste(self, height: int, width: int) -> np.ndarray:
        """このインスタンスの ``mask`` を ``(height, width)`` の全画面キャンバスへ貼り戻す。

        可視化・評価では全画面マスクが必要になる。``mask_region`` の位置に ``mask`` を置いた
        shape ``(height, width)`` の ``np.bool_`` 配列を返す（非破壊。新しい配列を返す）。

        画像外へはみ出す領域は**切り落とす**。元画像座標へ戻した ``mask_region`` は crop の
        逆変換で負座標や画像サイズ超過を含み得るため、クリップしないと ``IndexError`` や
        意図しない位置への書き込みになる。領域が完全に画像外なら全 ``False`` を返す（例外にしない）。

        Args:
            height: キャンバスの高さ[px]。
            width: キャンバスの幅[px]。

        Returns:
            shape ``(height, width)``、``np.bool_`` のキャンバス。

        Raises:
            ValueError: ``mask`` または ``mask_region`` が ``None`` の場合。

        Note:
            **全インスタンスをまとめて貼り戻す API は提供しない。** 1600x900 で 50 インスタンスなら
            約 72MB になり、bbox + crop 内マスクという表現を選んだ意味が失われる（`2d_tasks.md` 6.2）。
            全画面が必要な利用側は本メソッドをループで呼ぶ。
        """
        if self.mask is None or self.mask_region is None:
            raise ValueError("Instance2D.paste requires both mask and mask_region to be set")
        x0, y0, x1, y1 = self.mask_region
        canvas = np.zeros((height, width), dtype=np.bool_)
        # キャンバスに収まる範囲へクリップ（負座標・サイズ超過を切り落とす）。
        cx0, cy0 = max(x0, 0), max(y0, 0)
        cx1, cy1 = min(x1, width), min(y1, height)
        if cx1 <= cx0 or cy1 <= cy0:
            return canvas  # 完全に画像外。
        canvas[cy0:cy1, cx0:cx1] = self.mask[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]
        return canvas


def _xyxy_to_source(xyxy: np.ndarray, image: Image, normalized: bool) -> np.ndarray:
    """box の ``xyxy`` を、推論に使った ``image`` を基準に元画像座標へ戻す。

    ``normalized`` なら ``image`` の現サイズで画素へ戻し（正規化解除）、続いて
    ``image.source`` の ``scale`` → ``crop`` を逆適用する。順序が重要
    （正規化解除 → スケール逆適用 → crop 原点加算）。
    """
    if normalized:
        xyxy = denormalize_boxes(xyxy, image.width, image.height)
    return boxes_to_source(xyxy, image.source)


@dataclass
class Detection2DOutput:
    """2D 物体検出の出力（``object_detection_2d``）。

    座標は**入力 :class:`~jidohub.core.schemas.Image` の現サイズ基準**（`2d_tasks.md` 3.2）。
    3D 出力と異なり座標系の選択の余地がない（画像平面に一意）ため
    :class:`CoordinateFrame` フィールドは持たない。元画像へ戻す必要がある場合は
    :meth:`to_source_image`（内部で :attr:`~jidohub.core.schemas.image.Image.source` を使う）。

    Attributes:
        boxes: 検出されたボックスのリスト。スコア降順である必要はない。
        normalized: ``True`` なら :attr:`Box2D.xyxy` は ``[0, 1]`` の正規化座標、
            ``False``（既定）なら画素座標。**どの画像に対する座標かはここに持たない**
            （crop 位置とスケールは連続量で enum に収まらないため、
            :attr:`~jidohub.core.schemas.image.Image.source` が保持する。5.1）。
            正規化は要素間で不整合になり得ないコンテナ単位の性質なので
            :class:`Box2D` ではなくここに持つ。
    """

    boxes: list[Box2D] = field(default_factory=list)
    normalized: bool = False

    def to_source_image(self, image: Image) -> Detection2DOutput:
        """推論に使った ``image`` を基準に、元画像座標の新しい出力を返す。

        正規化解除・``image.source`` の逆適用を 1 回で行い、``normalized=False`` にする。
        crop 位置・スケール・正規化の有無を利用者が覚える必要がなくなる（5.3）。

        冪等・非破壊。``image.source`` が ``None`` かつ ``normalized`` が ``False`` なら
        変換対象がないため ``self`` を返す。
        """
        if not self.normalized and image.source is None:
            return self
        boxes = [
            Box2D(
                xyxy=_xyxy_to_source(b.xyxy, image, self.normalized),
                label=b.label,
                score=b.score,
                track_id=b.track_id,
                attributes=b.attributes,
            )
            for b in self.boxes
        ]
        return Detection2DOutput(boxes=boxes, normalized=False)


@dataclass
class InstanceSegmentation2DOutput:
    """インスタンスセグメンテーションの出力（``instance_segmentation_2d``）。

    座標系の扱いは :class:`Detection2DOutput` と同じ（現サイズ基準・``frame`` なし）。

    Attributes:
        instances: 検出されたインスタンスのリスト。
        normalized: ``True`` なら各 :attr:`Instance2D.box` の ``xyxy`` は ``[0, 1]`` の
            正規化座標、``False``（既定）なら画素座標（意味は :class:`Detection2DOutput` と同じ）。
            **マスクは常に画素座標**であり ``normalized`` の対象外
            （:attr:`Instance2D.mask_region` は整数）。1 つの :class:`Instance2D` 内で
            box（正規化され得る）と mask（常に画素）の座標系が混在する点に注意。
    """

    instances: list[Instance2D] = field(default_factory=list)
    normalized: bool = False

    def to_source_image(self, image: Image) -> InstanceSegmentation2DOutput:
        """推論に使った ``image`` を基準に、元画像座標の新しい出力を返す。

        各 box を元画像座標へ戻し、``mask`` と ``mask_region`` も元画像基準へ移す。``scale`` を
        伴う場合はマスクを**最近傍リサイズ**で移す（bool マスクに使える補間は最近傍のみで、
        純 numpy で書けるため画像処理依存は生じない。5.3）。品質面では Agent 側で二値化前に
        logit を補間するのが本来の経路であり、ここはそのフォールバック。

        冪等・非破壊。``image.source`` が ``None`` かつ ``normalized`` が ``False`` なら ``self``。
        """
        if not self.normalized and image.source is None:
            return self
        source = image.source
        instances = []
        for inst in self.instances:
            new_box = Box2D(
                xyxy=_xyxy_to_source(inst.box.xyxy, image, self.normalized),
                label=inst.box.label,
                score=inst.box.score,
                track_id=inst.box.track_id,
                attributes=inst.box.attributes,
            )
            mask, mask_region = inst.mask, inst.mask_region
            if mask is not None and mask_region is not None:
                # mask_region は常に画素座標なので、normalized によらず crop/scale 逆適用のみ行う
                # （_xyxy_to_source を使うと normalized 時に誤って正規化解除される）。
                moved = boxes_to_source(np.asarray(mask_region, dtype=np.float64), source)
                # 先に整数の目標領域を確定し、そのサイズへマスクを合わせる（逆順にすると
                # 丸めで 1 画素ずれ、Instance2D の size 一致検証に落ちる）。
                x0, y0 = int(np.floor(moved[0])), int(np.floor(moved[1]))
                x1, y1 = int(np.ceil(moved[2])), int(np.ceil(moved[3]))
                x1 = max(x1, x0 + 1)  # 極端な縮小で幅・高さが 0 になるのを防ぐ。
                y1 = max(y1, y0 + 1)
                if (y1 - y0, x1 - x0) != mask.shape:
                    mask = resize_mask_nearest(mask, y1 - y0, x1 - x0)
                mask_region = (x0, y0, x1, y1)
            instances.append(Instance2D(box=new_box, mask=mask, mask_region=mask_region))
        return InstanceSegmentation2DOutput(instances=instances, normalized=False)


@dataclass
class Classification2DOutput:
    """画像分類の出力（``image_classification_2d``）。

    Attributes:
        labels: クラス名の**文字列**リスト。スコア降順。ゼロショット分類では
            ラベル集合が実行時に決まるため文字列で持つ（`2d_tasks.md` 1 章）。
        scores: shape ``(K,)``、``np.float64``。``labels`` と同順・同数。
            モデルがスコアを出さない場合は ``None``。
    """

    labels: list[str] = field(default_factory=list)
    scores: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.scores is not None and self.scores.shape[0] != len(self.labels):
            raise ValueError(
                "Classification2DOutput.scores length must match labels "
                f"({self.scores.shape[0]} != {len(self.labels)})"
            )


class MapElementType(str, Enum):
    """HD マップ要素の種別。

    オンラインマップ構築（``map_construction``）の出力と、
    nuScenes Map Expansion の GT の双方で使う共通語彙。
    """

    DIVIDER = "divider"
    """車線境界線。"""

    PED_CROSSING = "ped-crossing"
    """横断歩道。"""

    BOUNDARY = "boundary"
    """走行可能領域の境界。"""

    CENTERLINE = "centerline"
    """車線中心線。"""


@dataclass
class MapElement:
    """HD マップの 1 要素（ポリライン / ポリゴン）。

    PostGIS へそのまま格納できる形を意図している。GeoJSON への変換は
    利用側（nuscenes-viewer 等）が行い、core では変換を提供しない。

    Attributes:
        points: shape ``(P, 2)`` または ``(P, 3)``、``np.float64``。頂点列[m]。
        element_type: 要素種別。
        is_closed: 閉じた形状（ポリゴン）なら ``True``、ポリラインなら ``False``。
        score: 信頼度スコア ``[0, 1]``。GT の場合は ``None``。
        element_id: 要素の識別子（GT との対応付けや追跡に使う）。
    """

    points: np.ndarray
    element_type: MapElementType
    is_closed: bool = False
    score: float | None = None
    element_id: str | None = None


@dataclass
class MapOutput:
    """オンライン HD マップ構築の出力（``map_construction``）。

    Attributes:
        elements: マップ要素のリスト。
        frame: ``elements`` が表現されている座標系。
    """

    elements: list[MapElement] = field(default_factory=list)
    frame: CoordinateFrame = CoordinateFrame.EGO

    def to_ego(self, ego_to_global: np.ndarray) -> MapOutput:
        """ego 座標系での新しい出力を返す（冪等・非破壊）。詳細は :meth:`Detection3DOutput.to_global`。"""
        return self._to_frame(CoordinateFrame.EGO, ego_to_global)

    def to_global(self, ego_to_global: np.ndarray) -> MapOutput:
        """global 座標系での新しい出力を返す（冪等・非破壊）。"""
        return self._to_frame(CoordinateFrame.GLOBAL, ego_to_global)

    def _to_frame(self, target: CoordinateFrame, ego_to_global: np.ndarray) -> MapOutput:
        transform = _resolve_transform(self.frame, target, ego_to_global)
        if transform is None:
            return self
        elements = [
            MapElement(
                points=_transform_positions(e.points, transform),
                element_type=e.element_type,
                is_closed=e.is_closed,
                score=e.score,
                element_id=e.element_id,
            )
            for e in self.elements
        ]
        return MapOutput(elements=elements, frame=target)


@dataclass
class AgentForecast:
    """他車・歩行者 1 体分の動作予測。

    座標の解釈に関わるメタ情報（座標系）は**要素ではなくコンテナ**
    :class:`MotionForecastOutput` が持つ（2 章）。要素に ``frame`` を持たせると
    要素間で不整合な座標系が型として表現可能になり、コンテナ単位の変換 API が
    定義できなくなるため、ここには持たせない。

    Attributes:
        trajectories: shape ``(M, T, 2)``、``np.float64``。
            ``M`` = マルチモーダル予測のモード数、``T`` = 予測ステップ数。
            単一モードの場合も ``M = 1`` として 3 次元で持つ（形状を分岐させない）。
        probabilities: shape ``(M,)``、``np.float64``。各モードの確率。総和 1。
        dt: 予測ステップの時間間隔[s]。
        track_id: 対応する :class:`Box3D` の ``track_id``。
    """

    trajectories: np.ndarray
    probabilities: np.ndarray
    dt: float
    track_id: int | None = None


@dataclass
class MotionForecastOutput:
    """動作予測の出力。

    Attributes:
        forecasts: 各対象の予測のリスト。
        frame: ``forecasts`` の軌跡が表現されている座標系（要素の
            :class:`AgentForecast` から移設。2 章・4.1）。
    """

    forecasts: list[AgentForecast] = field(default_factory=list)
    frame: CoordinateFrame = CoordinateFrame.EGO

    def to_ego(self, ego_to_global: np.ndarray) -> MotionForecastOutput:
        """ego 座標系での新しい出力を返す（冪等・非破壊）。"""
        return self._to_frame(CoordinateFrame.EGO, ego_to_global)

    def to_global(self, ego_to_global: np.ndarray) -> MotionForecastOutput:
        """global 座標系での新しい出力を返す（冪等・非破壊）。"""
        return self._to_frame(CoordinateFrame.GLOBAL, ego_to_global)

    def _to_frame(self, target: CoordinateFrame, ego_to_global: np.ndarray) -> MotionForecastOutput:
        transform = _resolve_transform(self.frame, target, ego_to_global)
        if transform is None:
            return self
        forecasts = [
            AgentForecast(
                trajectories=_transform_positions(f.trajectories, transform),
                probabilities=f.probabilities,
                dt=f.dt,
                track_id=f.track_id,
            )
            for f in self.forecasts
        ]
        return MotionForecastOutput(forecasts=forecasts, frame=target)


@dataclass
class PlanningOutput:
    """自車の走行計画（``track_map_to_planning``）。

    Attributes:
        trajectory: shape ``(T, 2)`` または ``(T, 3)``、``np.float64``。
            **ego 座標系**での自車の将来位置[m]。現在位置（原点）は含めず、
            ``dt`` 後の点から始める。
        dt: 軌跡の時間間隔[s]（例: ``0.5``）。
        frame: 座標系。通常 ``EGO``。
        confidence: 計画の信頼度 ``[0, 1]``。モデルが出力しない場合は ``None``。
    """

    trajectory: np.ndarray
    dt: float
    frame: CoordinateFrame = CoordinateFrame.EGO
    confidence: float | None = None

    def to_ego(self, ego_to_global: np.ndarray) -> PlanningOutput:
        """ego 座標系での新しい計画を返す（冪等・非破壊）。"""
        return self._to_frame(CoordinateFrame.EGO, ego_to_global)

    def to_global(self, ego_to_global: np.ndarray) -> PlanningOutput:
        """global 座標系での新しい計画を返す（冪等・非破壊）。"""
        return self._to_frame(CoordinateFrame.GLOBAL, ego_to_global)

    def _to_frame(self, target: CoordinateFrame, ego_to_global: np.ndarray) -> PlanningOutput:
        transform = _resolve_transform(self.frame, target, ego_to_global)
        if transform is None:
            return self
        return PlanningOutput(
            trajectory=_transform_positions(self.trajectory, transform),
            dt=self.dt,
            frame=target,
            confidence=self.confidence,
        )


@dataclass
class E2EOutput:
    """E2E モデルの出力（``sensing_to_planning``）。

    主たる出力は ``planning``。それ以外は**中間出力**であり、モデルが
    公開している場合のみ埋まる（``agent_config.json`` の
    ``intermediate_outputs`` で宣言されたものが対応する）。

    中間出力はすべて単体タスクの出力型を再利用しているため、
    可視化側は「出力型 → 描画レイヤー」の対応表を 1 つ持てばよい。

    Attributes:
        planning: 自車の走行計画。E2E の主出力。
        detection: 中間出力の 3D 検出結果。
        tracking: 中間出力の追跡結果（``Box3D.track_id`` が埋まる）。
        map: 中間出力のオンラインマップ。
        motion_forecast: 中間出力の他車動作予測。
        occupancy: 中間出力の占有予測。shape ``(T, H, W)``、``np.float32``。
            ``T`` = 予測ステップ数、``H``/``W`` = BEV グリッドの縦横。
            グリッドの分解能と原点は ``occupancy_meta`` で表す。
        occupancy_meta: 占有グリッドのメタ情報。
            ``{"resolution": float[m/cell], "origin": (x, y)[m], "dt": float[s]}``。
        bev_feature: 中間出力の BEV 特徴量。shape ``(C, H, W)``、``np.float32``。
            デバッグ・可視化用途。サイズが大きいため既定では返さない実装を推奨。
    """

    planning: PlanningOutput
    detection: Detection3DOutput | None = None
    tracking: Detection3DOutput | None = None
    map: MapOutput | None = None
    motion_forecast: MotionForecastOutput | None = None
    occupancy: np.ndarray | None = None
    occupancy_meta: dict | None = None
    bev_feature: np.ndarray | None = None
