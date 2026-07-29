"""符号化画像（``EncodedImage`` / デコーダ注入 / ``CameraFrame`` の pixels・encoded）のテスト。

core は画像コーデックに依存しないため、Pillow / OpenCV を持ち込まない。
生画素をそのままバイト列にした「符号化」形式（reshape で復元できる偽デコーダ）で検証する。
"""

from __future__ import annotations

import numpy as np
import pytest

from jidohub.core.schemas import (
    CameraFrame,
    EncodedImage,
    ImageDecodeError,
    ImageFormat,
    register_image_decoder,
)
from jidohub.core.serialization import pack, unpack

from .conftest import make_transform

# --- ヘルパ -----------------------------------------------------------------


def make_pixels(height: int = 4, width: int = 6) -> np.ndarray:
    """一意な値を持つ (H, W, 3) uint8 RGB 画素。"""
    return np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)


def encode_pixels(pixels: np.ndarray, format: ImageFormat | str = ImageFormat.JPEG) -> EncodedImage:
    """画素を「符号化」した EncodedImage を作る（バイト列は画素そのもの）。"""
    height, width = pixels.shape[:2]
    return EncodedImage.from_bytes(pixels.tobytes(), format, height=height, width=width)


def raw_reshape_decoder(encoded: EncodedImage) -> np.ndarray:
    """バイト列を (H, W, 3) に戻すだけの偽デコーダ（RGB を返す契約を満たす）。"""
    return encoded.data.reshape(encoded.height, encoded.width, 3)


class CountingDecoder:
    """呼び出し回数を数える偽デコーダ。"""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, encoded: EncodedImage) -> np.ndarray:
        self.calls += 1
        return raw_reshape_decoder(encoded)


def make_frame(**kwargs: object) -> CameraFrame:
    common = dict(
        intrinsic=np.eye(3, dtype=np.float64),
        sensor_to_ego=make_transform(),
        channel="CAM_FRONT",
    )
    common.update(kwargs)
    return CameraFrame(**common)  # type: ignore[arg-type]


# --- fixtures ---------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_decoder():
    """デコーダはモジュールグローバル状態。各テストの前後で必ず解除し、漏れを防ぐ。"""
    register_image_decoder(None)
    yield
    register_image_decoder(None)


@pytest.fixture
def raw_decoder():
    """生バイト列をそのまま画素に戻す偽デコーダを登録する。"""
    register_image_decoder(raw_reshape_decoder)
    yield raw_reshape_decoder
    register_image_decoder(None)


@pytest.fixture
def counting_decoder():
    """呼び出し回数を数える偽デコーダを登録し、そのカウンタを返す。"""
    decoder = CountingDecoder()
    register_image_decoder(decoder)
    yield decoder
    register_image_decoder(None)


# --- CameraFrame: pixels / encoded の排他 -----------------------------------


def test_pixels_only_builds() -> None:
    frame = make_frame(pixels=make_pixels())
    assert not frame.is_encoded
    assert frame.height == 4 and frame.width == 6


def test_encoded_only_builds() -> None:
    frame = make_frame(encoded=encode_pixels(make_pixels()))
    assert frame.is_encoded
    assert frame.height == 4 and frame.width == 6


def test_both_pixels_and_encoded_rejected() -> None:
    with pytest.raises(ValueError, match="exactly one of 'pixels' or 'encoded'"):
        make_frame(pixels=make_pixels(), encoded=encode_pixels(make_pixels()))


def test_neither_pixels_nor_encoded_rejected() -> None:
    with pytest.raises(ValueError, match="exactly one of 'pixels' or 'encoded'"):
        make_frame()


def test_pixels_must_be_3d() -> None:
    with pytest.raises(ValueError, match="CameraFrame.pixels must have shape"):
        make_frame(pixels=np.zeros((4, 6), dtype=np.uint8))


def test_pixels_must_have_three_channels() -> None:
    with pytest.raises(ValueError, match="CameraFrame.pixels must have shape"):
        make_frame(pixels=np.zeros((4, 6, 4), dtype=np.uint8))


def test_pixels_must_be_uint8() -> None:
    with pytest.raises(ValueError, match="CameraFrame.pixels must be uint8"):
        make_frame(pixels=np.zeros((4, 6, 3), dtype=np.float32))


# --- image プロパティ / デコード / キャッシュ ------------------------------


def test_pixels_image_returns_same_array() -> None:
    pixels = make_pixels()
    frame = make_frame(pixels=pixels)
    assert frame.image is pixels  # 生画素はそのまま返る


def test_encoded_image_decodes_correctly(raw_decoder) -> None:
    pixels = make_pixels()
    frame = make_frame(encoded=encode_pixels(pixels))
    assert np.array_equal(frame.image, pixels)


def test_encoded_image_is_cached(counting_decoder) -> None:
    frame = make_frame(encoded=encode_pixels(make_pixels()))
    first = frame.image
    second = frame.image
    assert first is second  # 2 回目は同一オブジェクト（再デコードしない）
    assert counting_decoder.calls == 1


def test_image_without_decoder_raises() -> None:
    frame = make_frame(encoded=encode_pixels(make_pixels()))
    with pytest.raises(ImageDecodeError, match="no image decoder"):
        _ = frame.image


def test_decoder_size_mismatch_raises() -> None:
    def wrong_size(encoded: EncodedImage) -> np.ndarray:
        return np.zeros((encoded.height + 1, encoded.width, 3), dtype=np.uint8)

    register_image_decoder(wrong_size)
    frame = make_frame(encoded=encode_pixels(make_pixels()))
    with pytest.raises(ImageDecodeError, match="does not match the declared size"):
        _ = frame.image


def test_decoder_wrong_dtype_raises() -> None:
    def wrong_dtype(encoded: EncodedImage) -> np.ndarray:
        return np.zeros((encoded.height, encoded.width, 3), dtype=np.float32)

    register_image_decoder(wrong_dtype)
    frame = make_frame(encoded=encode_pixels(make_pixels()))
    with pytest.raises(ImageDecodeError, match="must return a uint8 array"):
        _ = frame.image


def test_decoder_wrong_shape_raises() -> None:
    def wrong_shape(encoded: EncodedImage) -> np.ndarray:
        return np.zeros((encoded.height, encoded.width), dtype=np.uint8)  # 2 次元

    register_image_decoder(wrong_shape)
    frame = make_frame(encoded=encode_pixels(make_pixels()))
    with pytest.raises(ImageDecodeError, match="must return a uint8 array"):
        _ = frame.image


# --- サイズはデコードせずに取得できる（Task 2.3）----------------------------


def test_size_does_not_trigger_decode(counting_decoder) -> None:
    frame = make_frame(encoded=encode_pixels(make_pixels(8, 5)))
    assert frame.height == 8
    assert frame.width == 5
    assert frame.is_encoded is True
    # サイズ参照でデコードが走っていないこと。
    assert counting_decoder.calls == 0


# --- EncodedImage 自体の検証 ------------------------------------------------


def test_encoded_from_bytes_to_bytes_roundtrip() -> None:
    raw = bytes(range(50))
    encoded = EncodedImage.from_bytes(raw, ImageFormat.JPEG, height=4, width=6)
    assert encoded.to_bytes() == raw
    assert encoded.nbytes == len(raw)


def test_encoded_data_must_be_1d_uint8() -> None:
    with pytest.raises(ValueError, match="1-D uint8 array"):
        EncodedImage(
            data=np.zeros((2, 3), dtype=np.uint8), format=ImageFormat.JPEG, height=4, width=6
        )


def test_encoded_data_must_be_uint8() -> None:
    with pytest.raises(ValueError, match="1-D uint8 array"):
        EncodedImage(
            data=np.zeros((6,), dtype=np.float32), format=ImageFormat.JPEG, height=4, width=6
        )


@pytest.mark.parametrize("height,width", [(0, 6), (4, 0), (-1, 6)])
def test_encoded_size_must_be_positive(height: int, width: int) -> None:
    with pytest.raises(ValueError, match="size must be positive"):
        EncodedImage(
            data=np.zeros((6,), dtype=np.uint8), format=ImageFormat.JPEG, height=height, width=width
        )


# --- 直列化との整合（Task 2.4）---------------------------------------------


def test_encoded_frame_roundtrip_preserves_format_type() -> None:
    frame = make_frame(encoded=encode_pixels(make_pixels(), ImageFormat.JPEG))
    restored = unpack(pack(frame))
    assert isinstance(restored, CameraFrame)
    assert restored.is_encoded is True
    assert restored.encoded is not None
    # str を継承しているため == "jpeg" では型崩壊を検出できない。is で確認する。
    assert restored.encoded.format is ImageFormat.JPEG
    assert np.array_equal(restored.encoded.data, frame.encoded.data)


def test_pixels_frame_roundtrip_preserves_dtype() -> None:
    frame = make_frame(pixels=make_pixels())
    restored = unpack(pack(frame))
    assert restored.pixels is not None
    assert restored.pixels.dtype == np.uint8
    assert np.array_equal(restored.pixels, frame.pixels)


def test_decode_cache_is_not_serialized(raw_decoder) -> None:
    """``image`` によるデコード結果はフィールドでないため直列化されない。

    デコード前後で pack のペイロード長が変わらないことで確認する。
    """
    frame = make_frame(encoded=encode_pixels(make_pixels()))
    before = pack(frame)
    _ = frame.image  # デコードを発生させ、_decoded にキャッシュさせる
    after = pack(frame)
    assert len(after) == len(before)
