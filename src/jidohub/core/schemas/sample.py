"""標準入力スキーマ。

すべての Agent は :class:`Sample` を入力として受け取る（``predict(sample) -> Output``）。
データセット固有の形式から ``Sample`` への正規化は jidohub-datasets の Adapter が担い、
``Sample`` からモデル固有のテンソルへの変換は各 Agent の Processor が担う。
**core はこの型の定義だけを持ち、変換ロジックは持たない。**

規約（詳細は CLAUDE.md 3 章）
    - ego 座標系は右手系で ``x`` = 前方、``y`` = 左方、``z`` = 上方。
    - 変換行列は 4x4 同次変換行列（``np.float64``）で、名前が向きを表す。
    - 時刻は UNIX epoch のマイクロ秒 ``int``。
    - **画像は生画素（``pixels``）か符号化バイト列（``encoded``）のどちらか一方**で
      保持する（排他）。生画素は ``np.uint8`` の ``(H, W, 3)``、**RGB 順**。
      利用側は表現を意識せず :attr:`CameraFrame.image` を使う。常に RGB が返る。
    - プロセス境界を越える経路では ``encoded`` を推奨する
      （生画素の約 1/15 のサイズで運べる）。デコードは
      :func:`~jidohub.core.schemas.image.register_image_decoder` で
      jidohub-datasets / jidohub-agents 側が注入する。
    - 点群は ``np.float32`` の ``(N, C)`` で、先頭 3 列が x, y, z。
    - **点群はセンサ座標系のまま保持する。** ego 座標が必要な場合は
      各 sweep の :meth:`LidarSweep.points_in_ego` を使う。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from jidohub.core.geometry import transform_points
from jidohub.core.schemas.image import EncodedImage, decode_image

__all__ = [
    "DrivingCommand",
    "CameraFrame",
    "LidarSweep",
    "RadarSweep",
    "EgoState",
    "Sample",
]


class DrivingCommand(str, Enum):
    """高レベル走行指令。

    UniAD / VAD 系のプランナが入力として要求するため標準スキーマに含める。
    データセットから復元できない場合は ``Sample.command`` を ``None`` にする。
    """

    GO_STRAIGHT = "go-straight"
    TURN_LEFT = "turn-left"
    TURN_RIGHT = "turn-right"


@dataclass
class CameraFrame:
    """単一カメラの 1 フレーム。

    画素は **生配列（``pixels``）か符号化バイト列（``encoded``）のどちらか一方**で
    保持する。両方を同時に持つことも、両方 ``None`` にすることもできない
    （どちらが正かが曖昧になり、二重の正が生まれるため）。

    利用側は表現を意識せず :attr:`image` にアクセスすればよい。
    符号化されている場合は初回アクセス時にデコードされ、以後はキャッシュされる。

    Attributes:
        intrinsic: shape ``(3, 3)``、``np.float64``。ピンホールカメラの内部パラメータ。
        sensor_to_ego: shape ``(4, 4)``、``np.float64``。
            カメラ座標系 → ego 座標系の同次変換。逆向きではない。
        channel: センサチャンネル名（例: ``"CAM_FRONT"``）。
            :class:`Sample` の ``cameras`` のキーと一致させる。
        pixels: shape ``(H, W, 3)``、``np.uint8``、**RGB 順**。BGR で入れないこと。
            デコード済みの画素を直接持つ場合に使う。
        encoded: 符号化された画像。プロセス境界を越える経路ではこちらを推奨する
            （生配列の約 1/15 のサイズで運べる）。
        distortion: レンズ歪み係数。OpenCV の ``(k1, k2, p1, p2, k3, ...)`` 順。
            歪み補正済み画像の場合は ``None``。
        timestamp: このフレーム固有の取得時刻[μs]。
            センサ間で取得時刻が異なる場合に使う。``None`` なら ``Sample.timestamp`` を用いる。
    """

    intrinsic: np.ndarray
    sensor_to_ego: np.ndarray
    channel: str
    pixels: np.ndarray | None = None
    encoded: EncodedImage | None = None
    distortion: np.ndarray | None = None
    timestamp: int | None = None

    def __post_init__(self) -> None:
        _check_shape("CameraFrame.intrinsic", self.intrinsic, (3, 3))
        _check_shape("CameraFrame.sensor_to_ego", self.sensor_to_ego, (4, 4))

        if (self.pixels is None) == (self.encoded is None):
            raise ValueError(
                "CameraFrame requires exactly one of 'pixels' or 'encoded' to be set"
            )
        if self.pixels is not None:
            if self.pixels.ndim != 3 or self.pixels.shape[2] != 3:
                raise ValueError(
                    f"CameraFrame.pixels must have shape (H, W, 3), got {self.pixels.shape}"
                )
            if self.pixels.dtype != np.uint8:
                raise ValueError(
                    f"CameraFrame.pixels must be uint8 (RGB), got {self.pixels.dtype}"
                )

        # デコード結果のキャッシュ。dataclass のフィールドではないため直列化されない
        self._decoded: np.ndarray | None = None

    @property
    def image(self) -> np.ndarray:
        """画素を shape ``(H, W, 3)``、``np.uint8``、RGB 順で返す。

        ``encoded`` のみを保持している場合は初回アクセス時にデコードし、
        結果をインスタンス内にキャッシュする（キャッシュは直列化されない）。

        Raises:
            ImageDecodeError: デコーダが未登録の場合。
                jidohub-datasets を import するか、
                :func:`~jidohub.core.schemas.image.register_image_decoder` を呼ぶこと。
        """
        if self.pixels is not None:
            return self.pixels
        if self._decoded is None:
            assert self.encoded is not None  # __post_init__ で保証済み
            self._decoded = decode_image(self.encoded)
        return self._decoded

    @property
    def height(self) -> int:
        """画像の高さ[px]。**デコードせずに**取得できる。"""
        return int(self.pixels.shape[0]) if self.pixels is not None else self.encoded.height

    @property
    def width(self) -> int:
        """画像の幅[px]。**デコードせずに**取得できる。"""
        return int(self.pixels.shape[1]) if self.pixels is not None else self.encoded.width

    @property
    def is_encoded(self) -> bool:
        """符号化された表現を保持しているか。

        直列化コストの見積もりなど、表現の違いが本質的に効く場面でのみ使う。
        画素が欲しいだけなら :attr:`image` を使い、分岐を書かないこと。
        """
        return self.encoded is not None


@dataclass
class LidarSweep:
    """LiDAR の 1 スイープ。

    Attributes:
        points: shape ``(N, C)``、``np.float32``、``C >= 3``。
            **センサ座標系**で保持する。先頭 3 列は必ず x, y, z[m]。
            4 列目以降の意味は ``fields`` で表す。
        sensor_to_ego: shape ``(4, 4)``、``np.float64``。
            LiDAR 座標系 → ego 座標系の同次変換。
        channel: センサチャンネル名（例: ``"LIDAR_TOP"``）。
        fields: 各列の名前。先頭は必ず ``("x", "y", "z", ...)``。
            例: ``("x", "y", "z", "intensity", "ring")``。
        timestamp: このスイープ固有の取得時刻[μs]。``None`` なら ``Sample.timestamp``。

    Note:
        点群をセンサ座標のまま保持するのは、生データの可逆性を保ち、
        Adapter 実装を単純にするため。モデルによって必要な座標系が異なるので、
        変換は利用側（Agent の Processor）が明示的に行う。
    """

    points: np.ndarray
    sensor_to_ego: np.ndarray
    channel: str = "LIDAR_TOP"
    fields: tuple[str, ...] = ("x", "y", "z", "intensity")
    timestamp: int | None = None

    def __post_init__(self) -> None:
        _check_shape("LidarSweep.sensor_to_ego", self.sensor_to_ego, (4, 4))
        if self.points.ndim != 2 or self.points.shape[1] < 3:
            raise ValueError(
                f"LidarSweep.points must have shape (N, C>=3), got {self.points.shape}"
            )

    def points_in_ego(self) -> np.ndarray:
        """点群を ego 座標系に変換して返す（元の ``points`` は変更しない）。

        Returns:
            shape・dtype は ``points`` と同じ。先頭 3 列のみ変換される。
        """
        return transform_points(self.points, self.sensor_to_ego)


@dataclass
class RadarSweep:
    """RADAR の 1 スイープ。

    Attributes:
        points: shape ``(N, C)``、``np.float32``、``C >= 3``。
            **センサ座標系**で保持する。先頭 3 列は x, y, z[m]。
            速度成分を含む場合は ``fields`` で明示する。
        sensor_to_ego: shape ``(4, 4)``、``np.float64``。
        channel: センサチャンネル名（例: ``"RADAR_FRONT"``）。
        fields: 各列の名前。例: ``("x", "y", "z", "vx", "vy", "rcs")``。
            速度[m/s]は補償前後の区別が実装で混乱しやすいため、
            補償済みの場合は ``"vx_comp"`` のように名前で区別する。
        timestamp: このスイープ固有の取得時刻[μs]。
    """

    points: np.ndarray
    sensor_to_ego: np.ndarray
    channel: str
    fields: tuple[str, ...] = ("x", "y", "z")
    timestamp: int | None = None

    def __post_init__(self) -> None:
        _check_shape("RadarSweep.sensor_to_ego", self.sensor_to_ego, (4, 4))
        if self.points.ndim != 2 or self.points.shape[1] < 3:
            raise ValueError(
                f"RadarSweep.points must have shape (N, C>=3), got {self.points.shape}"
            )

    def points_in_ego(self) -> np.ndarray:
        """点群を ego 座標系に変換して返す（元の ``points`` は変更しない）。"""
        return transform_points(self.points, self.sensor_to_ego)


@dataclass
class EgoState:
    """自車の運動状態。

    E2E / プランナ系モデルが要求する（nuScenes の CAN bus 情報に相当）。
    取得できない場合は :class:`Sample` の ``ego_state`` を ``None`` にする。

    Attributes:
        velocity: shape ``(3,)``、``np.float64``。**ego 座標系**での速度[m/s]。
            グローバル座標系の速度ではない点に注意。
        acceleration: shape ``(3,)``、``np.float64``。ego 座標系での加速度[m/s^2]。
        angular_velocity: shape ``(3,)``、``np.float64``。ego 座標系での角速度[rad/s]。
            z 成分がヨーレート。
        steering_angle: ステアリング角[rad]。左が正。
    """

    velocity: np.ndarray | None = None
    acceleration: np.ndarray | None = None
    angular_velocity: np.ndarray | None = None
    steering_angle: float | None = None


@dataclass
class Sample:
    """センサ入力の標準コンテナ。データセット非依存。

    すべての Agent の入力型。1 タイムスタンプ分のセンサ観測と自車状態を保持する。
    時系列を必要とするモデルのために、過去フレームを ``history`` に持てる。

    Attributes:
        timestamp: 取得時刻。**UNIX epoch のマイクロ秒 ``int``**（nuScenes 準拠）。
            秒や float にしないこと。
        ego_to_global: shape ``(4, 4)``、``np.float64``。
            ego 座標系 → global（マップ）座標系の同次変換。逆向きではない。
        cameras: チャンネル名 → :class:`CameraFrame`。キーは ``CameraFrame.channel`` と一致させる。
        lidar: 主 LiDAR のスイープ。存在しない構成では ``None``。
        radars: チャンネル名 → :class:`RadarSweep`。
        ego_state: 自車の運動状態。取得できない場合は ``None``。
        command: 高レベル走行指令。プランナ系モデルが要求する。
        history: 過去フレームのリスト。**時刻の昇順**（末尾が直近の過去）で並べる。
            ``history`` の各要素はさらに ``history`` を持たない（1 段のみ）。
        sample_id: データセット内での一意識別子（nuScenes の sample_token 等）。
        scene_id: 所属シーンの識別子。
        metadata: データセット固有の付加情報。**標準スキーマで表現できるものを
            ここに入れない**（メタデータへの逃避は互換性を壊す）。
    """

    timestamp: int
    ego_to_global: np.ndarray
    cameras: dict[str, CameraFrame] = field(default_factory=dict)
    lidar: LidarSweep | None = None
    radars: dict[str, RadarSweep] = field(default_factory=dict)
    ego_state: EgoState | None = None
    command: DrivingCommand | None = None
    history: list["Sample"] = field(default_factory=list)
    sample_id: str | None = None
    scene_id: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_shape("Sample.ego_to_global", self.ego_to_global, (4, 4))
        if not isinstance(self.timestamp, (int, np.integer)):
            raise TypeError(
                "Sample.timestamp must be an int in microseconds, "
                f"got {type(self.timestamp).__name__}"
            )


def _check_shape(name: str, array: np.ndarray, expected: tuple[int, ...]) -> None:
    """形状の軽量チェック。毎フレーム呼ばれるため重い検証は行わない。"""
    if array.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {array.shape}")
