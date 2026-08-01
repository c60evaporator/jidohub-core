"""画像 1 枚を表す :class:`Image` と、その符号化表現・デコーダ注入。

画像そのものを表す型が :class:`Image` である。``pixels``（生画素）か
``encoded``（符号化バイト列）の**どちらか一方**を保持し、利用側は表現を意識せず
:attr:`Image.array` を使う（常に ``(H, W, 3)`` の ``np.uint8`` RGB が返る）。

2D タスク（`object_detection_2d` など）の入力は「カメラ 1 台の 1 フレーム」ではなく
「画像 1 枚」であり、crop された領域や外部から与えられた画像も対象になる。そのため
画像そのものを :class:`Image` として分離し、:class:`~jidohub.core.schemas.sample.CameraFrame`
は「:class:`Image` + カメラの取り付け情報（``sensor_to_ego``）」の合成として定義する。

なぜ符号化のまま運ぶか
    1600x900 の画像 6 枚は生配列で約 26MB、JPEG なら約 1.7MB。
    プロセス境界（docker runner との RPC）を越えるたびに 26MB を直列化・転送するのは
    転送コストとメモリコピーの両面で無駄が大きい。
    デコード自体のコストは**どちらの設計でも 1 回**発生するため、
    符号化のまま運べば直列化と転送のコストだけが削減される。

    さらに nuScenes をはじめ多くのデータセットは画像を JPEG で保持しているため、
    Adapter は読み込んだバイト列をそのまま載せるだけでよく、
    データ供給側でのデコードが不要になる。

なぜ :class:`EncodedPixels` を入力型にしないか
    符号化は :class:`Image` の 2 表現のうちの片方にすぎず、画像そのものではない
    （``pixels`` との is-a が成立しない。名前も ``pixels`` / ``encoded`` が対称になるよう改名した）。
    排他の吸収は :class:`Image` が担い、利用側は :attr:`Image.array` だけを見る。

core がコーデックに依存しない理由
    バイト列を運ぶだけならコーデックは要らない。実際のデコードは
    :func:`register_image_decoder` で datasets / agents 側が注入する
    （Pillow、OpenCV、nvJPEG など環境に応じて選べる）。
    これにより core の依存を numpy と pydantic に保てる（CLAUDE.md 2.1）。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from jidohub.core.geometry import crop_intrinsic

__all__ = [
    "ImageFormat",
    "EncodedPixels",
    "ImageSource",
    "Image",
    "ImageDecoder",
    "register_image_decoder",
    "get_image_decoder",
    "decode_image",
    "ImageDecodeError",
]


class ImageFormat(str, Enum):
    """符号化画像のフォーマット。"""

    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"


class ImageDecodeError(RuntimeError):
    """デコーダが未登録、またはデコードに失敗した場合に送出される例外。"""


@dataclass
class EncodedPixels:
    """符号化された画素データ 1 枚（:class:`Image` の一表現）。

    Attributes:
        data: shape ``(N,)``、``np.uint8``。**ファイルそのもののバイト列**
            （JPEG なら SOI から EOI まで）を 1 次元配列として保持する。

            ``bytes`` ではなく ``np.uint8`` 配列にしているのは、既存の直列化機構
            （:mod:`jidohub.core.serialization`）が numpy 配列をそのまま
            バイナリバッファとして扱えるため。型を増やさずに済む。
            バイト列が必要な場合は :meth:`to_bytes` を使う。
        format: 符号化フォーマット。
        height: デコード後の画像の高さ[px]。
        width: デコード後の画像の幅[px]。

    Note:
        ``height`` / ``width`` を明示的に保持するのは、**デコードせずに画像サイズを
        知りたい場面が多い**ため（内部パラメータのスケーリング、UI のレイアウト、
        入力サイズの検証など）。デコードは重い操作なので、
        サイズを得るためだけに実行させない。

        色空間はフィールドとして持たない。デコード結果は
        :data:`ImageDecoder` の契約により**常に RGB**であり、
        符号化ストリーム内部の色空間はコーデックの実装詳細だからである。
    """

    data: np.ndarray
    format: ImageFormat
    height: int
    width: int

    def __post_init__(self) -> None:
        if self.data.ndim != 1 or self.data.dtype != np.uint8:
            raise ValueError(
                "EncodedPixels.data must be a 1-D uint8 array of raw file bytes, "
                f"got shape={self.data.shape} dtype={self.data.dtype}"
            )
        if self.height <= 0 or self.width <= 0:
            raise ValueError(f"EncodedPixels size must be positive, got {self.width}x{self.height}")

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        format: ImageFormat | str,
        height: int,
        width: int,
    ) -> "EncodedPixels":
        """バイト列から構築する。Adapter からはこちらを使う。"""
        return cls(
            data=np.frombuffer(data, dtype=np.uint8),
            format=ImageFormat(format),
            height=height,
            width=width,
        )

    def to_bytes(self) -> bytes:
        """保持しているバイト列を ``bytes`` として返す。"""
        return self.data.tobytes()

    @property
    def nbytes(self) -> int:
        """符号化データのサイズ[byte]。"""
        return int(self.data.nbytes)


ImageDecoder = Callable[[EncodedPixels], np.ndarray]
"""デコーダの型。

:class:`EncodedPixels` を受け取り、shape ``(H, W, 3)`` の ``np.uint8`` 配列を
**RGB 順**で返すこと。
"""

_decoder: ImageDecoder | None = None


def register_image_decoder(decoder: ImageDecoder | None) -> None:
    """既定のデコーダを登録する。

    jidohub-datasets や jidohub-agents が import 時に呼ぶことを想定している。
    core 自身はコーデックに依存しないため、登録がない状態で
    :func:`decode_image` を呼ぶと :class:`ImageDecodeError` になる。

    Args:
        decoder: デコード関数。``None`` を渡すと登録を解除する（主にテスト用）。
    """
    global _decoder
    _decoder = decoder


def get_image_decoder() -> ImageDecoder | None:
    """現在登録されているデコーダを返す。未登録なら ``None``。"""
    return _decoder


def decode_image(encoded: EncodedPixels, decoder: ImageDecoder | None = None) -> np.ndarray:
    """符号化画素をデコードして ``(H, W, 3)`` の ``np.uint8`` 配列を返す。

    Args:
        encoded: デコード対象。
        decoder: 使用するデコーダ。``None`` なら登録済みの既定デコーダを使う。

    Raises:
        ImageDecodeError: デコーダが未登録、または結果が規約に反する場合。
    """
    active = decoder or _decoder
    if active is None:
        raise ImageDecodeError(
            "no image decoder is registered. "
            "Install and import jidohub-datasets, or call register_image_decoder()."
        )

    array = active(encoded)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ImageDecodeError(
            "decoder must return a uint8 array of shape (H, W, 3), "
            f"got shape={array.shape} dtype={array.dtype}"
        )
    if array.shape[0] != encoded.height or array.shape[1] != encoded.width:
        raise ImageDecodeError(
            f"decoded size {array.shape[1]}x{array.shape[0]} does not match "
            f"the declared size {encoded.width}x{encoded.height}"
        )
    return array


@dataclass
class ImageSource:
    """この画像の由来。crop / resize を経ても元画像へ写像できるようにする。

    **連鎖（リンクリスト）にはしない。** crop の crop を表す際も、元画像に対する
    実効的な crop + scale を **1 組に合成**して保持する。深さが無制限に伸びず、
    元画像座標への逆写像が 1 回で済むため（`2d_tasks.md` 3.2）。

    Attributes:
        channel: 元画像のセンサチャンネル名（あれば）。
        crop: **元画像上**の切り出し領域 ``(x0, y0, x1, y1)``（画素・整数）。
            原点は左上、``x`` 右・``y`` 下。
        scale: crop 後のサイズから現サイズへの拡大率 ``(scale_x, scale_y)``。
            crop 単独では解像度は変わらないため、core の :meth:`Image.cropped` は
            この値を変更せず引き継ぐ。実際のリサンプリングは agents 側の責務。
    """

    channel: str | None = None
    crop: tuple[int, int, int, int] | None = None
    scale: tuple[float, float] | None = None


@dataclass
class Image:
    """画像 1 枚。``pixels`` か ``encoded`` のどちらか一方を保持する（排他）。

    両方を同時に持つことも、両方 ``None`` にすることもできない
    （どちらが正かが曖昧になり、二重の正が生まれるため）。利用側は表現を意識せず
    :attr:`array` にアクセスすればよい。符号化されている場合は初回アクセス時に
    デコードされ、以後はキャッシュされる（キャッシュは直列化されない）。

    Attributes:
        pixels: shape ``(H, W, 3)``、``np.uint8``、**RGB 順**。BGR で入れないこと。
            デコード済みの画素を直接持つ場合に使う。
        encoded: 符号化された画素。プロセス境界を越える経路ではこちらを推奨する
            （生配列の約 1/15 のサイズで運べる）。
        intrinsic: shape ``(3, 3)``、``np.float64``。ピンホールカメラの内部パラメータ。
            未知の画像（外部入力など）では ``None``。
        distortion: レンズ歪み係数。OpenCV の ``(k1, k2, p1, p2, k3, ...)`` 順。
            歪み補正済み画像の場合は ``None``。
        source: この画像の由来（crop / scale の履歴）。元画像座標へ戻す際に使う。
    """

    pixels: np.ndarray | None = None
    encoded: EncodedPixels | None = None
    intrinsic: np.ndarray | None = None
    distortion: np.ndarray | None = None
    source: ImageSource | None = None

    def __post_init__(self) -> None:
        if (self.pixels is None) == (self.encoded is None):
            raise ValueError("Image requires exactly one of 'pixels' or 'encoded' to be set")
        if self.pixels is not None:
            if self.pixels.ndim != 3 or self.pixels.shape[2] != 3:
                raise ValueError(f"Image.pixels must have shape (H, W, 3), got {self.pixels.shape}")
            if self.pixels.dtype != np.uint8:
                raise ValueError(f"Image.pixels must be uint8 (RGB), got {self.pixels.dtype}")
        if self.intrinsic is not None and self.intrinsic.shape != (3, 3):
            raise ValueError(f"Image.intrinsic must have shape (3, 3), got {self.intrinsic.shape}")

        # デコード結果のキャッシュ。dataclass のフィールドではないため直列化されない。
        self._decoded: np.ndarray | None = None

    @property
    def array(self) -> np.ndarray:
        """画素を shape ``(H, W, 3)``、``np.uint8``、RGB 順で返す。

        ``encoded`` のみを保持している場合は初回アクセス時にデコードし、
        結果をインスタンス内にキャッシュする（キャッシュは直列化されない）。

        Raises:
            ImageDecodeError: デコーダが未登録の場合。
                jidohub-datasets を import するか、
                :func:`register_image_decoder` を呼ぶこと。
        """
        if self.pixels is not None:
            return self.pixels
        if self._decoded is None:
            assert self.encoded is not None  # __post_init__ で排他保証済み
            self._decoded = decode_image(self.encoded)
        return self._decoded

    @property
    def height(self) -> int:
        """画像の高さ[px]。**デコードせずに**取得できる。"""
        if self.pixels is not None:
            return int(self.pixels.shape[0])
        assert self.encoded is not None  # __post_init__ で排他保証済み
        return self.encoded.height

    @property
    def width(self) -> int:
        """画像の幅[px]。**デコードせずに**取得できる。"""
        if self.pixels is not None:
            return int(self.pixels.shape[1])
        assert self.encoded is not None  # __post_init__ で排他保証済み
        return self.encoded.width

    @property
    def is_encoded(self) -> bool:
        """符号化された表現を保持しているか。

        直列化コストの見積もりなど、表現の違いが本質的に効く場面でのみ使う。
        画素が欲しいだけなら :attr:`array` を使い、分岐を書かないこと。
        """
        return self.encoded is not None

    def cropped(self, x0: int, y0: int, x1: int, y1: int) -> "Image":
        """指定領域を切り出した新しい :class:`Image` を返す。

        画素の切り出し・``intrinsic`` の更新・``source`` の合成を**一貫して**行う。
        これを core が提供しないと、パイプライン側は ``intrinsic=None`` で逃げるか、
        親の ``intrinsic`` をそのままコピーする（後者は誤りだが実行時には通るため
        最も危険。`2d_tasks.md` 3.3）。``intrinsic`` の更新は
        :func:`~jidohub.core.geometry.crop_intrinsic` の薄いラッパである。

        座標は**この :class:`Image` の現サイズ基準**（左上原点、``x`` 右・``y`` 下）。
        crop 後の :class:`Image` に対してはさらにその crop 後の座標系になる。

        Args:
            x0, y0, x1, y1: 切り出し領域 ``[x0, x1) × [y0, y1)``（画素）。

        Returns:
            切り出した画素を ``pixels`` として持つ新しい :class:`Image`。
            ``intrinsic`` は更新され、``source`` は元画像基準の実効 crop + scale に
            合成される。``intrinsic`` が ``None`` の場合は ``None`` のまま。

        Raises:
            ValueError: 領域が画像の範囲外、または幅・高さが 0 以下の場合。

        Note:
            **``distortion`` は変更しない。** OpenCV の歪み係数は正規化座標
            ``((u - cx) / fx など)`` に対して定義されるため、``K`` を正しく更新している
            限り係数は不変である。「crop したから歪み係数も補正しなければ」という
            誤った処理を防ぐため、ここで明示する。

            **``encoded`` を保持している場合は切り出しにデコードが必要**であり、
            デコードした ``pixels`` ベースの :class:`Image` を返す（返り値の
            :attr:`is_encoded` は ``False`` になり、符号化のまま運ぶ利点は失われる）。
        """
        width, height = self.width, self.height
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(
                f"Image.cropped region ({x0}, {y0}, {x1}, {y1}) is out of bounds "
                f"or has non-positive size for a {width}x{height} image"
            )
        # encoded の場合はここでデコードが走り、pixels ベースの Image になる。
        cropped_pixels = self.array[y0:y1, x0:x1]
        new_intrinsic = (
            crop_intrinsic(self.intrinsic, x0, y0) if self.intrinsic is not None else None
        )
        return Image(
            pixels=cropped_pixels,
            intrinsic=new_intrinsic,
            distortion=self.distortion,
            source=self._compose_source(x0, y0, x1, y1),
        )

    def _compose_source(self, x0: int, y0: int, x1: int, y1: int) -> ImageSource:
        """現サイズ基準の crop を、元画像基準の実効 crop + scale へ 1 組に合成する。

        親が ``scale`` を持つ場合、子の crop 座標は親の scale が適用された現サイズ上で
        指定されるため、``x / scale`` で元画像の crop 座標に戻す。``scale`` 自体は
        crop で解像度が変わらないためそのまま引き継ぐ。連鎖はさせない。
        """
        source = self.source
        scale_x, scale_y = source.scale if source and source.scale else (1.0, 1.0)
        base_x, base_y = (source.crop[0], source.crop[1]) if source and source.crop else (0, 0)
        crop = (
            base_x + round(x0 / scale_x),
            base_y + round(y0 / scale_y),
            base_x + round(x1 / scale_x),
            base_y + round(y1 / scale_y),
        )
        return ImageSource(
            channel=source.channel if source else None,
            crop=crop,
            scale=source.scale if source else None,
        )
