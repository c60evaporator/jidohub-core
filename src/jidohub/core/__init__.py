"""jidohub-core: jidohub プラットフォームの共通基盤。

標準スキーマ・Hub クライアント・config パーサ／バリデータを提供する。
他の jidohub パッケージ（agents / datasets / interfaces）はすべて core に依存し、
**core は他の jidohub パッケージに依存しない**（星形依存）。

Note:
    ``src/jidohub/`` は PEP 420 の namespace package であり、
    ``src/jidohub/__init__.py`` は**存在してはならない**。
"""

from __future__ import annotations

from jidohub.core.schemas import SCHEMA_VERSION
from jidohub.core.tasks import TaskType

__all__ = ["SCHEMA_VERSION", "TaskType"]
