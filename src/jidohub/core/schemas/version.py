"""標準スキーマのバージョン。

``SCHEMA_VERSION`` はプラットフォーム全体の契約バージョンであり、
``agent_config.json`` の ``input_schema`` / ``output_schema`` の解決や、
シリアライズされたデータの互換判定に使う。

0.x の期間は**破壊的変更を許容する**。以下の 2 つを実際にこの型へ
押し込んで歪みがないことを確認するまで 1.0 にしない。

1. CenterPoint（``sensing-to-detection``）
2. UniAD（``sensing-to-planning``、中間出力を含む）
"""

from __future__ import annotations

__all__ = ["SCHEMA_VERSION"]

SCHEMA_VERSION = "0.1"
"""現在のスキーマバージョン（major.minor）。

- **major** の変更 = 後方互換のない変更。読み込み側はエラーにする。
- **minor** の変更 = フィールド追加など後方互換のある変更。読み込み側は許容する。
"""
