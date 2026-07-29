"""符号化済み画像の保持とデコーダの注入。

カメラ画像を**符号化されたバイト列のまま**運ぶための型を定義する。

なぜ生配列で運ばないか
    1600x900 の画像 6 枚は生配列で約 26MB、JPEG なら約 1.7MB。
    プロセス境界（docker runner との RPC）を越えるたびに 26MB を直列化・転送するのは
    転送コストとメモリコピーの両面で無駄が大きい。
    デコード自体のコストは**どちらの設計でも 1 回**発生するため、
    符号化のまま運べば直列化と転送のコストだけが削減される。

    さらに nuScenes をはじめ多くのデータセットは画像を JPEG で保持しているため、
    Adapter は読み込んだバイト列をそのまま載せるだけでよく、
    データ供給側でのデコードが不要になる。

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

__all__ = [
    "ImageFormat",
    "EncodedImage",
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
class EncodedImage:
    """符号化された画像 1 枚。

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
                "EncodedImage.data must be a 1-D uint8 array of raw file bytes, "
                f"got shape={self.data.shape} dtype={self.data.dtype}"
            )
        if self.height <= 0 or self.width <= 0:
            raise ValueError(f"EncodedImage size must be positive, got {self.width}x{self.height}")

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        format: ImageFormat | str,
        height: int,
        width: int,
    ) -> "EncodedImage":
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


ImageDecoder = Callable[[EncodedImage], np.ndarray]
"""デコーダの型。

:class:`EncodedImage` を受け取り、shape ``(H, W, 3)`` の ``np.uint8`` 配列を
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


def decode_image(encoded: EncodedImage, decoder: ImageDecoder | None = None) -> np.ndarray:
    """符号化画像をデコードして ``(H, W, 3)`` の ``np.uint8`` 配列を返す。

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
