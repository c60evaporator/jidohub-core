"""``Image`` / ``EncodedPixels`` / デコーダ注入 / ``CameraFrame`` の画像まわりのテスト。

core は画像コーデックに依存しないため、Pillow / OpenCV を持ち込まない。
生画素をそのままバイト列にした「符号化」形式（reshape で復元できる偽デコーダ）で検証する。
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from jidohub.core.schemas import (
    CameraFrame,
    EncodedPixels,
    Image,
    ImageDecodeError,
    ImageFormat,
    ImageSource,
    register_image_decoder,
)
from jidohub.core.serialization import pack, unpack

from .conftest import make_transform

# --- ヘルパ -----------------------------------------------------------------


def make_pixels(height: int = 4, width: int = 6) -> np.ndarray:
    """一意な値を持つ (H, W, 3) uint8 RGB 画素。"""
    return np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)


def encode_pixels(
    pixels: np.ndarray, format: ImageFormat | str = ImageFormat.JPEG
) -> EncodedPixels:
    """画素を「符号化」した EncodedPixels を作る（バイト列は画素そのもの）。"""
    height, width = pixels.shape[:2]
    return EncodedPixels.from_bytes(pixels.tobytes(), format, height=height, width=width)


def raw_reshape_decoder(encoded: EncodedPixels) -> np.ndarray:
    """バイト列を (H, W, 3) に戻すだけの偽デコーダ（RGB を返す契約を満たす）。"""
    return encoded.data.reshape(encoded.height, encoded.width, 3)


class CountingDecoder:
    """呼び出し回数を数える偽デコーダ。"""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, encoded: EncodedPixels) -> np.ndarray:
        self.calls += 1
        return raw_reshape_decoder(encoded)


def make_image(**kwargs: object) -> Image:
    return Image(**kwargs)  # type: ignore[arg-type]


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


# --- Image: pixels / encoded の排他 -----------------------------------------


def test_pixels_only_builds() -> None:
    image = make_image(pixels=make_pixels())
    assert not image.is_encoded
    assert image.height == 4 and image.width == 6


def test_encoded_only_builds() -> None:
    image = make_image(encoded=encode_pixels(make_pixels()))
    assert image.is_encoded
    assert image.height == 4 and image.width == 6


def test_both_pixels_and_encoded_rejected() -> None:
    with pytest.raises(ValueError, match="exactly one of 'pixels' or 'encoded'"):
        make_image(pixels=make_pixels(), encoded=encode_pixels(make_pixels()))


def test_neither_pixels_nor_encoded_rejected() -> None:
    with pytest.raises(ValueError, match="exactly one of 'pixels' or 'encoded'"):
        make_image()


def test_pixels_must_be_3d() -> None:
    with pytest.raises(ValueError, match="Image.pixels must have shape"):
        make_image(pixels=np.zeros((4, 6), dtype=np.uint8))


def test_pixels_must_have_three_channels() -> None:
    with pytest.raises(ValueError, match="Image.pixels must have shape"):
        make_image(pixels=np.zeros((4, 6, 4), dtype=np.uint8))


def test_pixels_must_be_uint8() -> None:
    with pytest.raises(ValueError, match="Image.pixels must be uint8"):
        make_image(pixels=np.zeros((4, 6, 3), dtype=np.float32))


def test_intrinsic_must_be_3x3() -> None:
    with pytest.raises(ValueError, match="Image.intrinsic must have shape"):
        make_image(pixels=make_pixels(), intrinsic=np.eye(4, dtype=np.float64))


# --- array プロパティ / デコード / キャッシュ ------------------------------


def test_pixels_array_returns_same_array() -> None:
    pixels = make_pixels()
    image = make_image(pixels=pixels)
    assert image.array is pixels  # 生画素はそのまま返る


def test_encoded_array_decodes_correctly(raw_decoder) -> None:
    pixels = make_pixels()
    image = make_image(encoded=encode_pixels(pixels))
    assert np.array_equal(image.array, pixels)


def test_encoded_array_is_cached(counting_decoder) -> None:
    image = make_image(encoded=encode_pixels(make_pixels()))
    first = image.array
    second = image.array
    assert first is second  # 2 回目は同一オブジェクト（再デコードしない）
    assert counting_decoder.calls == 1


def test_array_without_decoder_raises() -> None:
    image = make_image(encoded=encode_pixels(make_pixels()))
    with pytest.raises(ImageDecodeError, match="no image decoder"):
        _ = image.array


def test_decoder_size_mismatch_raises() -> None:
    def wrong_size(encoded: EncodedPixels) -> np.ndarray:
        return np.zeros((encoded.height + 1, encoded.width, 3), dtype=np.uint8)

    register_image_decoder(wrong_size)
    image = make_image(encoded=encode_pixels(make_pixels()))
    with pytest.raises(ImageDecodeError, match="does not match the declared size"):
        _ = image.array


def test_decoder_wrong_dtype_raises() -> None:
    def wrong_dtype(encoded: EncodedPixels) -> np.ndarray:
        return np.zeros((encoded.height, encoded.width, 3), dtype=np.float32)

    register_image_decoder(wrong_dtype)
    image = make_image(encoded=encode_pixels(make_pixels()))
    with pytest.raises(ImageDecodeError, match="must return a uint8 array"):
        _ = image.array


def test_decoder_wrong_shape_raises() -> None:
    def wrong_shape(encoded: EncodedPixels) -> np.ndarray:
        return np.zeros((encoded.height, encoded.width), dtype=np.uint8)  # 2 次元

    register_image_decoder(wrong_shape)
    image = make_image(encoded=encode_pixels(make_pixels()))
    with pytest.raises(ImageDecodeError, match="must return a uint8 array"):
        _ = image.array


# --- サイズはデコードせずに取得できる ---------------------------------------


def test_size_does_not_trigger_decode(counting_decoder) -> None:
    image = make_image(encoded=encode_pixels(make_pixels(8, 5)))
    assert image.height == 8
    assert image.width == 5
    assert image.is_encoded is True
    # サイズ参照でデコードが走っていないこと。
    assert counting_decoder.calls == 0


# --- cropped() --------------------------------------------------------------


def test_cropped_pixels_match_parent_region() -> None:
    pixels = make_pixels(8, 10)
    image = make_image(pixels=pixels)
    cropped = image.cropped(2, 1, 6, 4)
    assert np.array_equal(cropped.array, pixels[1:4, 2:6])


def test_cropped_size_matches_region() -> None:
    image = make_image(pixels=make_pixels(8, 10))
    cropped = image.cropped(2, 1, 6, 4)
    assert cropped.width == 4 and cropped.height == 3


def test_cropped_shifts_principal_point() -> None:
    intrinsic = np.array([[100.0, 0.0, 5.0], [0.0, 120.0, 4.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    image = make_image(pixels=make_pixels(8, 10), intrinsic=intrinsic)
    cropped = image.cropped(2, 1, 6, 4)
    # 主点だけが (x0, y0) 移動し、焦点距離は不変。
    assert cropped.intrinsic is not None
    assert cropped.intrinsic[0, 2] == 3.0  # 5 - 2
    assert cropped.intrinsic[1, 2] == 3.0  # 4 - 1
    assert cropped.intrinsic[0, 0] == 100.0
    assert cropped.intrinsic[1, 1] == 120.0
    # 元の intrinsic は破壊されない。
    assert intrinsic[0, 2] == 5.0


def test_cropped_leaves_distortion_unchanged() -> None:
    distortion = np.array([0.1, -0.2, 0.0, 0.0, 0.05], dtype=np.float64)
    image = make_image(
        pixels=make_pixels(8, 10),
        intrinsic=np.eye(3, dtype=np.float64),
        distortion=distortion,
    )
    cropped = image.cropped(2, 1, 6, 4)
    # 歪み係数は正規化座標に対して定義されるため crop で不変。
    assert np.array_equal(cropped.distortion, distortion)


def test_cropped_keeps_none_intrinsic() -> None:
    image = make_image(pixels=make_pixels(8, 10))  # intrinsic なし
    cropped = image.cropped(2, 1, 6, 4)
    assert cropped.intrinsic is None


@pytest.mark.parametrize(
    "region",
    [
        (-1, 0, 4, 3),  # 範囲外（左）
        (0, 0, 11, 3),  # 範囲外（右、幅 10 超）
        (0, 0, 0, 3),  # 幅 0
        (0, 0, 4, 0),  # 高さ 0
        (4, 0, 2, 3),  # x1 < x0
    ],
)
def test_cropped_rejects_bad_region(region: tuple[int, int, int, int]) -> None:
    image = make_image(pixels=make_pixels(8, 10))
    with pytest.raises(ValueError, match="out of bounds"):
        image.cropped(*region)


def test_cropped_encoded_returns_pixels_based(raw_decoder) -> None:
    pixels = make_pixels(8, 10)
    image = make_image(encoded=encode_pixels(pixels))
    cropped = image.cropped(2, 1, 6, 4)
    # 符号化画像の crop はデコードを要するため pixels ベースになる。
    assert cropped.is_encoded is False
    assert np.array_equal(cropped.array, pixels[1:4, 2:6])


def test_cropped_source_from_original() -> None:
    image = make_image(pixels=make_pixels(8, 10))
    cropped = image.cropped(2, 1, 6, 4)
    assert cropped.source is not None
    assert cropped.source.crop == (2, 1, 6, 4)
    assert cropped.source.scale is None


def test_crop_of_crop_composes_to_single_source() -> None:
    image = make_image(pixels=make_pixels(20, 20))
    first = image.cropped(3, 2, 15, 14)  # 現サイズ 12x12
    second = first.cropped(1, 1, 5, 6)  # first 基準
    # 連鎖せず、元画像基準の 1 組の crop に合成される。
    assert isinstance(second.source, ImageSource)
    assert second.source.crop == (4, 3, 8, 8)  # (3+1, 2+1, 3+5, 2+6)
    assert second.source.scale is None


# --- _decoded がフィールドでない --------------------------------------------


def test_decoded_cache_is_not_a_dataclass_field() -> None:
    field_names = {f.name for f in dataclasses.fields(Image)}
    assert "_decoded" not in field_names


# --- EncodedPixels 自体の検証 -----------------------------------------------


def test_encoded_from_bytes_to_bytes_roundtrip() -> None:
    raw = bytes(range(50))
    encoded = EncodedPixels.from_bytes(raw, ImageFormat.JPEG, height=4, width=6)
    assert encoded.to_bytes() == raw
    assert encoded.nbytes == len(raw)


def test_encoded_data_must_be_1d_uint8() -> None:
    with pytest.raises(ValueError, match="1-D uint8 array"):
        EncodedPixels(
            data=np.zeros((2, 3), dtype=np.uint8), format=ImageFormat.JPEG, height=4, width=6
        )


def test_encoded_data_must_be_uint8() -> None:
    with pytest.raises(ValueError, match="1-D uint8 array"):
        EncodedPixels(
            data=np.zeros((6,), dtype=np.float32), format=ImageFormat.JPEG, height=4, width=6
        )


@pytest.mark.parametrize("height,width", [(0, 6), (4, 0), (-1, 6)])
def test_encoded_size_must_be_positive(height: int, width: int) -> None:
    with pytest.raises(ValueError, match="size must be positive"):
        EncodedPixels(
            data=np.zeros((6,), dtype=np.uint8), format=ImageFormat.JPEG, height=height, width=width
        )


# --- 直列化との整合 ---------------------------------------------------------


def test_encoded_image_roundtrip_preserves_format_type() -> None:
    image = make_image(encoded=encode_pixels(make_pixels(), ImageFormat.JPEG))
    restored = unpack(pack(image))
    assert isinstance(restored, Image)
    assert restored.is_encoded is True
    assert restored.encoded is not None
    # str を継承しているため == "jpeg" では型崩壊を検出できない。is で確認する。
    assert restored.encoded.format is ImageFormat.JPEG
    assert np.array_equal(restored.encoded.data, image.encoded.data)


def test_pixels_image_roundtrip_preserves_dtype() -> None:
    image = make_image(pixels=make_pixels())
    restored = unpack(pack(image))
    assert restored.pixels is not None
    assert restored.pixels.dtype == np.uint8
    assert np.array_equal(restored.pixels, image.pixels)


def test_image_source_tuple_survives_roundtrip() -> None:
    image = make_image(pixels=make_pixels(8, 10)).cropped(2, 1, 6, 4)
    restored = unpack(pack(image))
    assert restored.source is not None
    # tuple が list に落ちていないこと。
    assert isinstance(restored.source.crop, tuple)
    assert restored.source.crop == (2, 1, 6, 4)


def test_camera_frame_roundtrip() -> None:
    frame = CameraFrame(
        image=Image(pixels=make_pixels(), intrinsic=np.eye(3, dtype=np.float64)),
        sensor_to_ego=make_transform(),
        channel="CAM_FRONT",
        timestamp=1_600_000_000_000_000,
    )
    restored = unpack(pack(frame))
    assert isinstance(restored, CameraFrame)
    assert restored.channel == "CAM_FRONT"
    assert restored.image.intrinsic is not None
    assert np.array_equal(restored.image.array, frame.image.array)


def test_decode_cache_is_not_serialized(raw_decoder) -> None:
    """``array`` によるデコード結果はフィールドでないため直列化されない。

    デコード前後で pack のペイロード長が変わらないことで確認する。
    """
    image = make_image(encoded=encode_pixels(make_pixels()))
    before = pack(image)
    _ = image.array  # デコードを発生させ、_decoded にキャッシュさせる
    after = pack(image)
    assert len(after) == len(before)
