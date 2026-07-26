"""バイナリコンテナ形式（pack / unpack）。

「JSON ヘッダ + 生バッファ」の自己記述コンテナ。safetensors と同じ発想で、
**追加依存なし（numpy と標準ライブラリのみ）**で numpy 配列を効率よく運ぶ。

形式
    ::

        +----------------------------------+
        | magic     : 8 bytes  b"JIDOHUB\\x01" |
        | header_len: 8 bytes  uint64 LE   |
        | header    : header_len bytes     |  UTF-8 JSON、8 バイト境界にパディング
        | buffers   : 残り                 |  各バッファを 8 バイト境界に整列
        +----------------------------------+

    ヘッダ JSON は ``{"schema_version", "root", "buffers": [{dtype, shape, offset, nbytes}]}``。

JSON を使わず生バッファで運ぶ理由
    点群 1 スイープは約 3 万点 × 4 列 × 4 バイトで 500KB、
    カメラ 6 枚は 1 フレームで約 26MB になる。base64 化すると 33% 増える上に
    エンコード／デコードの CPU コストが無視できない。

画像の圧縮について
    このコンテナは配列を**生のまま**運ぶ。JPEG/PNG 圧縮は画像コーデックへの依存が
    必要になるため core では扱わない。帯域が問題になる経路（server の API 境界など）は、
    画像を事前署名 URL で別送するか、圧縮を各サービス側で行うこと。
"""

from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np

from jidohub.core.schemas.version import SCHEMA_VERSION, assert_compatible
from jidohub.core.serialization.codec import SerializationError, decode, encode

__all__ = ["MAGIC", "pack", "unpack"]

MAGIC = b"JIDOHUB\x01"
"""コンテナ先頭のマジックバイト。末尾 1 バイトはコンテナ形式のバージョン。"""

_ALIGNMENT = 8
_HEADER_LEN_SIZE = 8


def pack(obj: Any) -> bytes:
    """スキーマオブジェクトをバイト列に直列化する。

    Args:
        obj: スキーマのインスタンス、またはそれらを含むコンテナ。

    Returns:
        自己記述コンテナのバイト列。

    Raises:
        SerializationError: 未対応の型が含まれる場合。
    """
    tree, arrays = encode(obj)

    descriptors: list[dict[str, Any]] = []
    payloads: list[bytes] = []
    offset = 0
    for array in arrays:
        data = array.tobytes()
        descriptors.append(
            {
                "dtype": array.dtype.str,  # バイトオーダーを含む（例: "<f4"）
                "shape": list(array.shape),
                "offset": offset,
                "nbytes": len(data),
            }
        )
        payloads.append(data)
        padding = _padding_for(len(data))
        if padding:
            payloads.append(b"\x00" * padding)
        offset += len(data) + padding

    header = {
        "schema_version": SCHEMA_VERSION,
        "root": tree,
        "buffers": descriptors,
    }
    header_bytes = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    header_padding = _padding_for(len(MAGIC) + _HEADER_LEN_SIZE + len(header_bytes))

    return b"".join(
        [
            MAGIC,
            struct.pack("<Q", len(header_bytes) + header_padding),
            header_bytes,
            b" " * header_padding,  # JSON として無害な空白でパディング
            *payloads,
        ]
    )


def unpack(data: bytes | bytearray | memoryview, copy: bool = False) -> Any:
    """:func:`pack` が生成したバイト列からオブジェクトを復元する。

    Args:
        data: :func:`pack` の出力。
        copy: ``True`` なら配列をコピーして書き込み可能にする。
            ``False``（既定）ではバッファ上のビューを返すため**読み取り専用**になる。
            大きな点群・画像を扱うため既定はゼロコピー。書き込みが必要な場合、
            または元のバイト列を解放したい場合に ``True`` を指定する。

    Raises:
        SerializationError: 形式が不正、またはスキーマバージョンが非互換の場合。
    """
    view = memoryview(data)
    if len(view) < len(MAGIC) + _HEADER_LEN_SIZE:
        raise SerializationError("payload is too short to be a jidohub container")
    if bytes(view[: len(MAGIC)]) != MAGIC:
        raise SerializationError("payload does not start with the jidohub magic bytes")

    header_len = struct.unpack("<Q", view[len(MAGIC) : len(MAGIC) + _HEADER_LEN_SIZE])[0]
    header_start = len(MAGIC) + _HEADER_LEN_SIZE
    header_end = header_start + header_len
    if header_end > len(view):
        raise SerializationError("declared header length exceeds payload size")

    try:
        header = json.loads(bytes(view[header_start:header_end]).decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise SerializationError(f"container header is not valid JSON: {error}") from error

    declared_version = header.get("schema_version")
    if not isinstance(declared_version, str):
        raise SerializationError("container header is missing 'schema_version'")
    try:
        assert_compatible(declared_version)
    except ValueError as error:
        raise SerializationError(str(error)) from error

    body = view[header_end:]
    arrays = [_read_array(body, descriptor, copy) for descriptor in header.get("buffers", [])]
    return decode(header.get("root"), arrays)


def _read_array(body: memoryview, descriptor: dict[str, Any], copy: bool) -> np.ndarray:
    try:
        offset = int(descriptor["offset"])
        nbytes = int(descriptor["nbytes"])
        dtype = np.dtype(descriptor["dtype"])
        shape = tuple(int(dim) for dim in descriptor["shape"])
    except (KeyError, TypeError, ValueError) as error:
        raise SerializationError(f"invalid buffer descriptor: {descriptor!r}") from error

    if offset < 0 or offset + nbytes > len(body):
        raise SerializationError("buffer descriptor points outside the payload")
    expected = int(np.prod(shape)) * dtype.itemsize if shape else dtype.itemsize
    if expected != nbytes:
        raise SerializationError(
            f"buffer size mismatch: descriptor says {nbytes} bytes "
            f"but shape {shape} with dtype {dtype} needs {expected}"
        )

    array = np.frombuffer(body[offset : offset + nbytes], dtype=dtype).reshape(shape)
    return array.copy() if copy else array


def _padding_for(length: int) -> int:
    """8 バイト境界に整列するためのパディング長を返す。

    numpy が配列を整列アクセスできるようにするため。
    """
    remainder = length % _ALIGNMENT
    return 0 if remainder == 0 else _ALIGNMENT - remainder
