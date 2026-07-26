"""``agent_config.json`` の読み込みと検証。

検証には 2 段階ある。

1. **スキーマ検証**（:func:`load_agent_config`）
   型・必須項目・相互制約の検証。ロード時に常に実行される。
2. **strict 検証**（:func:`validate_strict`）
   プラットフォームのポリシー適合の検証。jidohub-web へのアップロード時や
   Verified バッジ付与の判定に使う。ロード時には実行されない
   （既存 Agent がポリシー変更で読めなくなるのを避けるため）。
"""

from __future__ import annotations

import json
from pathlib import Path

from jidohub.core.config.agent import AgentConfig, ConfigValidationError

__all__ = [
    "AGENT_CONFIG_FILENAME",
    "load_agent_config",
    "parse_agent_config",
    "validate_strict",
]

AGENT_CONFIG_FILENAME = "agent_config.json"
"""Agent リポジトリのルートに置かれる設定ファイル名。"""


def parse_agent_config(data: dict) -> AgentConfig:
    """dict から :class:`AgentConfig` を構築する。

    Raises:
        ConfigValidationError: 検証に失敗した場合。
    """
    try:
        return AgentConfig.model_validate(data)
    except Exception as error:  # pydantic.ValidationError を含む
        raise ConfigValidationError(f"invalid {AGENT_CONFIG_FILENAME}: {error}") from error


def load_agent_config(repo_path: str | Path) -> AgentConfig:
    """Agent リポジトリのルートから ``agent_config.json`` を読み込む。

    Args:
        repo_path: リポジトリのルートディレクトリ、または設定ファイル自体のパス。

    Raises:
        FileNotFoundError: 設定ファイルが存在しない場合。
        ConfigValidationError: 検証に失敗した場合。
    """
    path = Path(repo_path)
    if path.is_dir():
        path = path / AGENT_CONFIG_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"{AGENT_CONFIG_FILENAME} not found at {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfigValidationError(f"{path} is not valid JSON: {error}") from error

    return parse_agent_config(data)


def validate_strict(config: AgentConfig, repo_path: str | Path | None = None) -> list[str]:
    """プラットフォームのポリシー適合を検証し、違反理由のリストを返す。

    空リストが返れば適合。**例外を送出せずリストを返す**のは、
    アップロード UI で違反理由をまとめて提示するため。

    検証内容
        - 重みが safetensors 形式であること（pickle ベースの重みを排除する）
        - 重みのチェックサムが宣言されていること
        - ライセンスが宣言されていること
        - ``repo_path`` が与えられた場合、参照ファイルが実在すること

    Args:
        config: 検証対象の設定。
        repo_path: リポジトリのルート。与えるとファイルの実在検証も行う。

    Returns:
        違反理由の文字列リスト。適合していれば空リスト。
    """
    violations: list[str] = []

    for weight in config.weights:
        if weight.format != "safetensors":
            violations.append(
                f"weights[{weight.path!r}] uses {weight.format!r}; "
                "safetensors is required (pickle-based weights can execute arbitrary code)"
            )
        if weight.sha256 is None:
            violations.append(f"weights[{weight.path!r}] is missing sha256 checksum")

    if config.license is None:
        violations.append("license is required")

    if repo_path is not None:
        violations.extend(_check_referenced_files(config, Path(repo_path)))

    return violations


def _check_referenced_files(config: AgentConfig, repo_path: Path) -> list[str]:
    """config が参照するファイルがリポジトリ内に実在するかを検証する。"""
    violations: list[str] = []

    referenced: list[tuple[str, str]] = [
        (weight.path, f"weights[{weight.path!r}]") for weight in config.weights
    ]
    if config.runtime.dockerfile:
        referenced.append((config.runtime.dockerfile, "runtime.dockerfile"))
    for key, target in config.implementation.auto_map.items():
        referenced.append((target.split(":", 1)[0], f"implementation.auto_map[{key!r}]"))

    for relative, label in referenced:
        if not (repo_path / relative).is_file():
            violations.append(f"{label} points to a missing file: {relative}")

    return violations
