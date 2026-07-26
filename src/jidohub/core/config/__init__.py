"""``agent_config.json`` 等の設定ファイルのパースと検証。

config 面は Pydantic を使う（データ面の dataclass と使い分ける。CLAUDE.md 4 章）。
JSON Schema は :meth:`AgentConfig.model_json_schema` で自動生成でき、
jidohub-web のアップロード検証にそのまま再利用できる。
"""

from __future__ import annotations

from jidohub.core.config.agent import (
    AgentConfig,
    ConfigValidationError,
    FrameworkSpec,
    Implementation,
    RuntimeSpec,
    SensorRequirement,
    WeightsSpec,
)
from jidohub.core.config.loader import (
    AGENT_CONFIG_FILENAME,
    load_agent_config,
    parse_agent_config,
    validate_strict,
)

__all__ = [
    "AGENT_CONFIG_FILENAME",
    "AgentConfig",
    "SensorRequirement",
    "Implementation",
    "WeightsSpec",
    "RuntimeSpec",
    "FrameworkSpec",
    "ConfigValidationError",
    "load_agent_config",
    "parse_agent_config",
    "validate_strict",
]
