"""標準スキーマのバージョン。

``SCHEMA_VERSION`` はプラットフォーム全体の契約バージョンであり、
``agent_config.json`` の ``input_schema`` / ``output_schema`` の解決や、
シリアライズされたデータの互換判定に使う。

0.x の期間は**破壊的変更を許容する**。以下の 2 つを実際にこの型へ
押し込んで歪みがないことを確認するまで 1.0 にしない。

1. CenterPoint（``object_detection_3d``）
2. UniAD（``sensing_to_planning``、中間出力を含む）

このモジュールがスキーマ契約バージョンと互換判定の**唯一の正**。
serialization / config など他モジュールは判定ロジックを再実装せず、ここへ委譲すること。
"""

from __future__ import annotations

import re

__all__ = ["SCHEMA_VERSION", "assert_compatible", "is_compatible"]

_VERSION_PATTERN = re.compile(r"^(\d+)\.(\d+)$")

SCHEMA_VERSION = "0.1"
"""現在のスキーマバージョン（major.minor）。

- **major** の変更 = 後方互換のない変更。読み込み側はエラーにする。
- **minor** の変更 = フィールド追加など後方互換のある変更。読み込み側は許容する。
"""


def assert_compatible(declared: str) -> None:
    """宣言されたバージョンが core と互換かを検証する。

    ``agent_config.json`` の検証とシリアライズ形式の検証の双方から呼ばれる。
    互換判定のルールを 1 箇所に集約するため、各所で個別に実装しないこと。

    互換ルール
        - major が異なる場合は非互換。
        - **0.x の期間は minor 差も非互換**として扱う（破壊的変更を許容する期間のため）。
        - 1.0 以降は、宣言側の minor が core 以下であれば互換。

    Args:
        declared: 検証対象のバージョン文字列（``"0.1"`` 形式）。

    Raises:
        ValueError: 形式が不正、または非互換の場合。
    """
    matched = _VERSION_PATTERN.match(declared)
    if not matched:
        raise ValueError(f"schema version must be 'major.minor', got {declared!r}")

    declared_major, declared_minor = int(matched.group(1)), int(matched.group(2))
    current_major, current_minor = (int(part) for part in SCHEMA_VERSION.split("."))

    if declared_major != current_major:
        raise ValueError(
            f"schema version {declared!r} is incompatible with core {SCHEMA_VERSION!r} "
            "(major version mismatch)"
        )
    if current_major == 0 and declared_minor != current_minor:
        raise ValueError(
            f"schema version {declared!r} is incompatible with core {SCHEMA_VERSION!r} "
            "(0.x is unstable; minor versions are not compatible)"
        )
    if declared_minor > current_minor:
        raise ValueError(
            f"schema version {declared!r} is newer than core {SCHEMA_VERSION!r}; "
            "please upgrade jidohub-core"
        )


def is_compatible(declared: str) -> bool:
    """:func:`assert_compatible` の真偽値版。"""
    try:
        assert_compatible(declared)
    except ValueError:
        return False
    return True
