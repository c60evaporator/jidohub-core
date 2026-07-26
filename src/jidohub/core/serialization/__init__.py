"""スキーマオブジェクトの直列化。

プロセス境界（docker runner との RPC）とネットワーク境界（server の API）で
:class:`~jidohub.core.schemas.Sample` と各 Output をやり取りするために使う。

用途に応じて 2 つの粒度を提供する。

- :func:`pack` / :func:`unpack`
    バイト列との相互変換。通常はこちらを使う。
- :func:`encode` / :func:`decode`
    「JSON 可能ツリー + numpy 配列のリスト」との相互変換。
    multipart や事前署名 URL で配列本体を別送したい場合に使う。

Example:
    >>> payload = pack(sample)
    >>> restored = unpack(payload)
"""

from __future__ import annotations

from jidohub.core.serialization.codec import (
    TYPE_REGISTRY,
    SerializationError,
    decode,
    encode,
)
from jidohub.core.serialization.envelope import MAGIC, pack, unpack

__all__ = [
    "pack",
    "unpack",
    "encode",
    "decode",
    "TYPE_REGISTRY",
    "SerializationError",
    "MAGIC",
]
